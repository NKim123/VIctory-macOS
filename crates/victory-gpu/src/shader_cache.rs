//! Translated shaders kept on disk (decision 0003): under the configured
//! cache folder, keyed by the container's content and by what the
//! translation depends on ([`build`]), so a changed translator never serves
//! an old translation, while rebuilding the rest of VIctory keeps them.
//! Each translator's entries are in a folder of their own, named by the
//! start of that key: older translators' folders can be deleted whole.
//! Entries go through `victory-pipeline`'s content-addressed cache, which
//! writes them atomically; an entry that does not read back is translated
//! again. A container that does not translate is kept too, with the reason,
//! so it is not tried again on every run.
//!
//! `victory translate --cache` fills the same cache ahead of a run, through
//! [`host_shader`].
//!
//! The entry is our own encoding of [`HostShader`]; this file's text is part
//! of the key, so changing the encoding invalidates every entry.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use victory_pipeline::{Cache, Converter};
use victory_shader::emit::{HostShader, Options, StaticSampler, UserData};
use victory_shader::{HwStage, RootType};

static CACHE: OnceLock<ShaderCache> = OnceLock::new();
static HITS: AtomicU64 = AtomicU64::new(0);
static MISSES: AtomicU64 = AtomicU64::new(0);

/// An offline or runtime cache using the same translation and fingerprint.
pub struct ShaderCache {
    dir: PathBuf,
    cache: Cache,
    options: Options,
    build: [u8; 32],
}

impl ShaderCache {
    /// Uses the current host, or targets D3DMetal's 32-lane GPU for
    /// offline translation. Both use the same environment overrides.
    pub fn open(dir: &Path, wine_defaults: bool) -> anyhow::Result<Self> {
        let options = if wine_defaults {
            options_for(true, true)
        } else {
            runtime_options()
        };
        let base = build(options.host_wave32);
        let dir = dir.join(&blake3::Hash::from(*base).to_hex()[..16]);
        let cache = Cache::open(&dir)?;
        tracing::info!(dir = %dir.display(), "shader cache");
        Ok(Self {
            dir,
            cache,
            options,
            build: *translation_build(options),
        })
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn host_shader(&self, container: &[u8]) -> (Result<HostShader, String>, bool) {
        cached(&self.cache, &self.build, container, |c| {
            translate(c, self.options)
        })
    }
}

/// Keeps translated shaders in a folder of `dir` for this process.
pub fn use_shader_cache(dir: PathBuf) -> Option<PathBuf> {
    if CACHE.get().is_none() {
        match ShaderCache::open(&dir, false) {
            Ok(cache) => {
                let _ = CACHE.set(cache);
            }
            Err(e) => {
                tracing::warn!(error = %e, "no shader cache");
                return None;
            }
        }
    }
    CACHE.get().map(|cache| cache.dir.clone())
}

/// Shaders found in the cache and translated since the start.
pub fn counts() -> (u64, u64) {
    (HITS.load(Ordering::Relaxed), MISSES.load(Ordering::Relaxed))
}

/// What a translation depends on besides its container: the translator
/// (`victory_shader::SOURCE_HASH`: its sources, manifest, the lock file and
/// the compiler), the entry's encoding (this file's text) and the
/// translation settings (the loop limit, which a diagnostics build can
/// change at run time, and whether the host's waves are 32 lanes at most,
/// [`host_wave32`]).
fn build(wave32: bool) -> &'static [u8; 32] {
    static BUILD: [OnceLock<[u8; 32]>; 2] = [const { OnceLock::new() }; 2];
    BUILD[usize::from(wave32)].get_or_init(|| {
        let limit = victory_shader::emit::loop_limit();
        if limit != victory_shader::emit::DEFAULT_LOOP_LIMIT {
            tracing::warn!(
                limit,
                "VICTORY_SHADER_LOOP_LIMIT: translated loops stop sooner"
            );
        }
        key(
            victory_shader::SOURCE_HASH,
            ENCODING,
            limit,
            wave32,
            victory_shader::emit::raw_root_cbvs(),
            victory_shader::emit::wide_scalar_loads(),
            victory_shader::emit::shader_opt(),
        )
    })
}

/// Whether translations are for a host of 32-lane waves: the host GPU's
/// (made here if it is not yet: the cache belongs to the GPU it was filled
/// on, as the pipeline library does); none without one.
#[cfg(windows)]
fn host_wave32() -> bool {
    crate::host::host().is_some_and(crate::host::Host::wave32)
}

#[cfg(not(windows))]
fn host_wave32() -> bool {
    false
}

/// This file, whose `encode` and `decode` define the entries.
const ENCODING: &[u8] = include_bytes!("shader_cache.rs");

fn key(
    translator: &str,
    encoding: &[u8],
    limit: u32,
    wave32: bool,
    port_cbv: bool,
    wide_scalar_loads: bool,
    shader_opt: u32,
) -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    h.update(translator.as_bytes());
    h.update(blake3::hash(encoding).as_bytes());
    h.update(&limit.to_le_bytes());
    h.update(&[u8::from(wave32), u8::from(port_cbv)]);
    if port_cbv {
        // The flag changes both root binding and table load lowering.
        h.update(b"bounded root CBVs and grouped table CBV loads");
    }
    h.update(&shader_opt.to_le_bytes());
    if wide_scalar_loads {
        h.update(b"wide scalar loads");
    }
    if victory_shader::emit::raw_structured_views() {
        h.update(b"raw structured views");
    }
    *h.finalize().as_bytes()
}

/// Respect the same emission overrides for native warm-up and Wine.
fn options_for(wine: bool, wave32: bool) -> Options {
    let flag = |name, default| match std::env::var(name).as_deref() {
        Ok("0") => false,
        Ok("1") => true,
        _ => default,
    };
    let atomic_locks = flag("VICTORY_ATOMIC_LOCKS", wine);
    let rounds = std::env::var("VICTORY_ATOMIC_LOCK_ROUNDS").ok();
    let atomic_lock_rounds = rounds_on(atomic_locks, rounds.as_deref());
    let atomic_umax = atomic_lock_rounds && flag("VICTORY_ATOMIC_UMAX", false);
    Options {
        view_marks: flag("VICTORY_VIEW_MARKS", true),
        atomic_locks,
        atomic_lock_rounds,
        atomic_umax,
        atomic_early_exit: atomic_umax && flag("VICTORY_ATOMIC_EARLY_EXIT", true),
        bias_as_gradients: flag("VICTORY_BIAS_AS_GRADIENTS", wine),
        host_wave32: wave32,
        narrow_waves: flag("VICTORY_NARROW_WAVES", wine),
    }
}

/// The rounds, given whether the locks are on and
/// `VICTORY_ATOMIC_LOCK_ROUNDS`: on with the locks unless it is `0`.
fn rounds_on(locks: bool, var: Option<&str>) -> bool {
    locks && var != Some("0")
}

/// Whether this runs under Wine (on a Mac: D3DMetal's D3D12).
#[cfg(windows)]
pub(crate) fn under_wine() -> bool {
    use windows_sys::Win32::System::LibraryLoader::{GetModuleHandleA, GetProcAddress};
    // SAFETY: ntdll is always loaded; both names end with a NUL.
    unsafe {
        GetProcAddress(
            GetModuleHandleA(c"ntdll.dll".as_ptr().cast()),
            c"wine_get_version".as_ptr().cast(),
        )
    }
    .is_some()
}

#[cfg(not(windows))]
pub(crate) fn under_wine() -> bool {
    false
}

/// [`build`] for translations with host workarounds (atomic locks, their
/// rounds and umax reduction, bias as gradients, narrow waves, and view
/// instrumentation): their own
/// cache entries, one per combination.
pub(crate) fn translation_build(options: Options) -> &'static [u8; 32] {
    let locks = options.atomic_locks;
    let bias = options.bias_as_gradients;
    let narrow = options.narrow_waves;
    let no_marks = !options.view_marks;
    let rounds = locks && options.atomic_lock_rounds;
    let umax = rounds && options.atomic_umax;
    let early_exit = umax && options.atomic_early_exit;
    if !locks && !bias && !narrow && !no_marks {
        return build(options.host_wave32);
    }
    static BUILDS: [OnceLock<[u8; 32]>; 256] = [const { OnceLock::new() }; 256];
    let n = usize::from(locks)
        | usize::from(bias) << 1
        | usize::from(narrow) << 2
        | usize::from(rounds) << 3
        | usize::from(umax) << 4
        | usize::from(early_exit) << 5
        | usize::from(options.host_wave32) << 6
        | usize::from(no_marks) << 7;
    BUILDS[n].get_or_init(|| {
        let mut h = blake3::Hasher::new();
        h.update(build(options.host_wave32));
        if locks {
            h.update(b"atomic locks");
        }
        if rounds {
            h.update(b"atomic lock rounds");
        }
        if umax {
            h.update(b"atomic umax");
        }
        if early_exit {
            h.update(b"atomic early exit");
        }
        if bias {
            h.update(b"bias as gradients");
        }
        if narrow {
            h.update(b"narrow waves");
        }
        if no_marks {
            h.update(b"no view marks");
        }
        *h.finalize().as_bytes()
    })
}

pub(crate) fn runtime_options() -> Options {
    static OPTIONS: OnceLock<Options> = OnceLock::new();
    *OPTIONS.get_or_init(|| options_for(under_wine(), host_wave32()))
}

fn translate(container: &[u8], options: Options) -> Result<HostShader, String> {
    victory_shader::emit::translate_with(container, options).map_err(|e| e.to_string())
}

/// The translation of a console shader container, and whether it was cached.
pub fn host_shader(container: &[u8]) -> (Result<HostShader, String>, bool) {
    match CACHE.get() {
        Some(cache) => cache.host_shader(container),
        None => (translate(container, runtime_options()), false),
    }
}

/// The cache entry's converter: the build and the container in, the
/// encoded translation (or refusal) out. `ran` tells a miss from a hit;
/// `made` keeps a miss's translation, so the entry just written is not read
/// back (the first read of a new file costs about as much as a small
/// translation where a virus scanner looks at it: 0.5-3 ms here).
struct Translate<F> {
    translate: F,
    ran: AtomicBool,
    made: Mutex<Option<Result<HostShader, String>>>,
}

impl<F: Fn(&[u8]) -> Result<HostShader, String> + Send + Sync> Converter for Translate<F> {
    fn id(&self) -> &'static str {
        "shader.dxil"
    }
    fn version(&self) -> u32 {
        // The build fingerprint in the input does the versioning.
        1
    }
    fn convert(&self, input: &[u8]) -> anyhow::Result<Vec<u8>> {
        self.ran.store(true, Ordering::Relaxed);
        let made = (self.translate)(&input[32..]);
        let bytes = encode(&made);
        *self.made.lock().unwrap_or_else(|e| e.into_inner()) = Some(made);
        Ok(bytes)
    }
}

fn cached(
    cache: &Cache,
    build: &[u8; 32],
    container: &[u8],
    translate: impl Fn(&[u8]) -> Result<HostShader, String> + Send + Sync,
) -> (Result<HostShader, String>, bool) {
    let input = [&build[..], container].concat();
    let converter = Translate {
        translate,
        ran: AtomicBool::new(false),
        made: Mutex::new(None),
    };
    let made = || {
        converter
            .made
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take()
    };
    for _ in 0..2 {
        let path = match cache.get_or_convert(&converter, &input) {
            Ok(path) => path,
            Err(e) => {
                tracing::warn!(error = format!("{e:#}"), "shader cache entry not written");
                break;
            }
        };
        if let Some(shader) = made() {
            MISSES.fetch_add(1, Ordering::Relaxed);
            tracing::debug!(path = %path.display(), "shader translated and cached");
            return (shader, false);
        }
        if let Some(shader) = crate::read_file(&path).ok().and_then(|b| decode(&b)) {
            let hit = !converter.ran.load(Ordering::Relaxed);
            let (counter, what) = if hit {
                (&HITS, "from the cache")
            } else {
                (&MISSES, "translated and cached")
            };
            counter.fetch_add(1, Ordering::Relaxed);
            tracing::debug!(path = %path.display(), "shader {what}");
            return (shader, hit);
        }
        tracing::debug!(path = %path.display(), "unreadable shader cache entry; translating again");
        let _ = std::fs::remove_file(&path);
    }
    MISSES.fetch_add(1, Ordering::Relaxed);
    // A translation whose entry could not be written is not made again.
    (
        made().unwrap_or_else(|| (converter.translate)(container)),
        false,
    )
}

const MAGIC: &[u8; 4] = b"VSH2";
/// A container that does not translate: the reason follows.
const REFUSED: &[u8; 4] = b"VSHE";
const STAGES: [HwStage; 8] = [
    HwStage::Ls,
    HwStage::Hs,
    HwStage::Es,
    HwStage::Gs,
    HwStage::Vs,
    HwStage::Ps,
    HwStage::Ms,
    HwStage::Cs,
];

fn root_code(r: RootType) -> u32 {
    match r {
        RootType::Constants => 0,
        RootType::Descriptor => 1,
        RootType::Table => 2,
        RootType::ImmediateSampler => 3,
        RootType::ResourceHeap => 4,
        RootType::SamplerHeap => 5,
        RootType::Other(x) => 6 + x,
    }
}

fn root_type(code: u32) -> RootType {
    match code {
        0 => RootType::Constants,
        1 => RootType::Descriptor,
        2 => RootType::Table,
        3 => RootType::ImmediateSampler,
        4 => RootType::ResourceHeap,
        5 => RootType::SamplerHeap,
        x => RootType::Other(x - 6),
    }
}

/// Little-endian words and length-prefixed lists; a refusal is its
/// reason's text.
fn encode(translation: &Result<HostShader, String>) -> Vec<u8> {
    let s = match translation {
        Ok(s) => s,
        Err(reason) => return [REFUSED.as_slice(), reason.as_bytes()].concat(),
    };
    // Every field by name: a field added to `HostShader` must be added here
    // (and to `decode`).
    let HostShader {
        stage,
        dxil,
        user_data,
        samplers,
        static_samplers,
        root_cbvs,
        root_srvs,
        root_uavs,
        root_srv_bytes,
        memory,
        memory_writes,
        view_addresses,
        heap,
        heap_writes,
        view_marks,
    } = s;
    let mut out = MAGIC.to_vec();
    let word = |out: &mut Vec<u8>, v: u32| out.extend(v.to_le_bytes());
    let stage = STAGES.iter().position(|x| x == stage).unwrap_or(0) as u32;
    word(&mut out, stage);
    word(&mut out, dxil.len() as u32);
    out.extend(dxil);
    word(&mut out, user_data.len() as u32);
    for u in user_data {
        for v in [u.index, u32::from(u.slot), root_code(u.root), u.dword] {
            word(&mut out, v);
        }
    }
    word(&mut out, samplers.len() as u32);
    for sampler in samplers {
        for &v in sampler {
            word(&mut out, v);
        }
    }
    word(&mut out, static_samplers.len() as u32);
    for t in static_samplers {
        let [u, v, w] = t.address;
        let words = [
            t.filter,
            u,
            v,
            w,
            t.mip_lod_bias.to_bits(),
            t.max_anisotropy,
            t.comparison,
            t.border,
            t.min_lod.to_bits(),
            t.max_lod.to_bits(),
        ];
        for v in words {
            word(&mut out, v);
        }
    }
    for slots in [root_cbvs, root_srvs, root_uavs] {
        word(&mut out, slots.len() as u32);
        out.extend(slots);
    }
    word(&mut out, root_srv_bytes.len() as u32);
    for &(slot, end) in root_srv_bytes {
        word(&mut out, u32::from(slot));
        word(&mut out, end);
    }
    word(
        &mut out,
        u32::from(*memory)
            | u32::from(*memory_writes) << 1
            | u32::from(*view_addresses) << 2
            | u32::from(*heap) << 3
            | u32::from(*heap_writes) << 4
            | u32::from(*view_marks) << 5,
    );
    out
}

fn decode(bytes: &[u8]) -> Option<Result<HostShader, String>> {
    struct Reader<'a>(&'a [u8]);
    impl<'a> Reader<'a> {
        fn take(&mut self, n: usize) -> Option<&'a [u8]> {
            let (head, rest) = self.0.split_at_checked(n)?;
            self.0 = rest;
            Some(head)
        }
        fn word(&mut self) -> Option<u32> {
            Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
        }
        fn list(&mut self) -> Option<Vec<u8>> {
            let n = self.word()? as usize;
            Some(self.take(n)?.to_vec())
        }
    }
    if let Some(reason) = bytes.strip_prefix(REFUSED) {
        return Some(Err(String::from_utf8(reason.to_vec()).ok()?));
    }
    let mut r = Reader(bytes);
    if r.take(4)? != MAGIC {
        return None;
    }
    let stage = *STAGES.get(r.word()? as usize)?;
    let dxil = r.list()?;
    let n = r.word()? as usize;
    let mut user_data = Vec::with_capacity(n.min(256));
    for _ in 0..n {
        user_data.push(UserData {
            index: r.word()?,
            slot: u8::try_from(r.word()?).ok()?,
            root: root_type(r.word()?),
            dword: r.word()?,
        });
    }
    let n = r.word()? as usize;
    let mut samplers = Vec::with_capacity(n.min(256));
    for _ in 0..n {
        samplers.push([r.word()?, r.word()?, r.word()?, r.word()?]);
    }
    let n = r.word()? as usize;
    let mut static_samplers = Vec::with_capacity(n.min(256));
    for _ in 0..n {
        static_samplers.push(StaticSampler {
            filter: r.word()?,
            address: [r.word()?, r.word()?, r.word()?],
            mip_lod_bias: f32::from_bits(r.word()?),
            max_anisotropy: r.word()?,
            comparison: r.word()?,
            border: r.word()?,
            min_lod: f32::from_bits(r.word()?),
            max_lod: f32::from_bits(r.word()?),
        });
    }
    let root_cbvs = r.list()?;
    let root_srvs = r.list()?;
    let root_uavs = r.list()?;
    let n = r.word()? as usize;
    let mut root_srv_bytes = Vec::with_capacity(n.min(256));
    for _ in 0..n {
        root_srv_bytes.push((u8::try_from(r.word()?).ok()?, r.word()?));
    }
    let memory = r.word()?;
    r.0.is_empty().then_some(Ok(HostShader {
        stage,
        dxil,
        user_data,
        samplers,
        static_samplers,
        root_cbvs,
        root_srvs,
        root_uavs,
        root_srv_bytes,
        memory: memory & 1 != 0,
        memory_writes: memory & 2 != 0,
        view_addresses: memory & 4 != 0,
        heap: memory & 8 != 0,
        heap_writes: memory & 16 != 0,
        view_marks: memory & 32 != 0,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The rounds are on wherever the locks are, `VICTORY_ATOMIC_LOCK_ROUNDS=0`
    /// turns them off, and without the locks there are none (#240).
    #[test]
    fn lock_rounds_are_on_with_the_locks_unless_turned_off() {
        assert!(rounds_on(true, None), "on by default with the locks");
        assert!(rounds_on(true, Some("1")), "the old opt-in still works");
        assert!(!rounds_on(true, Some("0")), "0 goes back to the turns");
        assert!(!rounds_on(false, None), "no rounds without the locks");
        assert!(!rounds_on(false, Some("1")), "nor when asked for");
    }

    fn shader() -> HostShader {
        HostShader {
            stage: HwStage::Ps,
            dxil: b"DXBC and the rest".to_vec(),
            user_data: vec![UserData {
                index: 8,
                slot: 1,
                root: RootType::Table,
                dword: 0,
            }],
            samplers: vec![[0x400, 0xc0_0000, 0x18f0_0000, 0]],
            static_samplers: vec![StaticSampler::from_words([
                0x400,
                0xc0_0000,
                0x18f0_0000,
                0,
            ])],
            root_cbvs: vec![2],
            root_srvs: vec![0, 3],
            root_uavs: vec![4],
            root_srv_bytes: vec![(1, 48), (5, 16384)],
            memory: true,
            memory_writes: true,
            view_addresses: true,
            heap: true,
            heap_writes: true,
            view_marks: true,
        }
    }

    #[test]
    fn translations_come_back_from_the_cache() {
        let dir = tempfile::tempdir().unwrap();
        let cache = Cache::open(dir.path()).unwrap();
        let build = [7u8; 32];
        let container = b"a console container".as_slice();
        let runs = AtomicU64::new(0);
        let translate = |_: &[u8]| {
            runs.fetch_add(1, Ordering::Relaxed);
            Ok(shader())
        };
        assert_eq!(
            cached(&cache, &build, container, translate),
            (Ok(shader()), false)
        );
        assert_eq!(
            cached(&cache, &build, container, translate),
            (Ok(shader()), true)
        );
        assert_eq!(runs.load(Ordering::Relaxed), 1, "the second came from disk");

        // Another build translates again.
        assert_eq!(
            cached(&cache, &[8; 32], container, translate).0,
            Ok(shader())
        );
        assert_eq!(runs.load(Ordering::Relaxed), 2);

        // A damaged entry is a miss, translated and stored again.
        for e in walk(dir.path()) {
            std::fs::write(e, b"VSH1 broken").unwrap();
        }
        assert_eq!(
            cached(&cache, &build, container, translate),
            (Ok(shader()), false)
        );
        assert_eq!(runs.load(Ordering::Relaxed), 3);
        assert_eq!(
            cached(&cache, &build, container, translate),
            (Ok(shader()), true)
        );
        assert_eq!(runs.load(Ordering::Relaxed), 3);

        // Refusals are stored with their reason, and not tried again.
        let fails = AtomicU64::new(0);
        let fail = |_: &[u8]| {
            fails.fetch_add(1, Ordering::Relaxed);
            Err::<HostShader, _>("unsupported".to_owned())
        };
        let refused = (Err("unsupported".to_owned()), false);
        assert_eq!(cached(&cache, &build, b"other", fail), refused);
        assert_eq!(cached(&cache, &build, b"other", fail), (refused.0, true));
        assert_eq!(fails.load(Ordering::Relaxed), 1);
    }

    /// A translation whose entry cannot be written (the cache folder is
    /// a file now) is handed back as it was made, not made again.
    #[test]
    fn a_translation_is_made_once_when_its_entry_cannot_be_written() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("cache");
        let cache = Cache::open(&root).unwrap();
        let _ = std::fs::remove_dir_all(&root);
        std::fs::write(&root, b"not a folder").unwrap();
        let runs = AtomicU64::new(0);
        let translate = |_: &[u8]| {
            runs.fetch_add(1, Ordering::Relaxed);
            Ok(shader())
        };
        assert_eq!(
            cached(&cache, &[7; 32], b"a container", translate),
            (Ok(shader()), false)
        );
        assert_eq!(runs.load(Ordering::Relaxed), 1);
    }

    /// The key's parts: a change to any of them (the translator's sources,
    /// this file's encoding, the loop limit, the host's wave width) makes
    /// another key, so no entry written before it is read.
    #[test]
    fn the_key_covers_translator_encoding_and_settings() {
        let version = "0123456789abcdef";
        let encoding = b"encoding";
        let limit = 1 << 20;
        let k = key(version, encoding, limit, false, false, false, 0);
        assert_eq!(k, key(version, encoding, limit, false, false, false, 0));
        assert_ne!(
            k,
            key("0123456789abcdee", encoding, limit, false, false, false, 0)
        );
        assert_ne!(k, key(version, b"encoding.", limit, false, false, false, 0));
        assert_ne!(k, key(version, encoding, limit / 2, false, false, false, 0));
        assert_ne!(k, key(version, encoding, limit, true, false, false, 0));
        assert_ne!(k, key(version, encoding, limit, false, true, false, 0));
        assert_ne!(k, key(version, encoding, limit, false, false, true, 0));
        for passes in [1, 2, 4, 8, 16, 32, 64] {
            assert_ne!(
                k,
                key(version, encoding, limit, false, false, false, passes)
            );
        }
        assert_eq!(
            *build(host_wave32()),
            key(
                victory_shader::SOURCE_HASH,
                ENCODING,
                victory_shader::emit::loop_limit(),
                host_wave32(),
                victory_shader::emit::raw_root_cbvs(),
                victory_shader::emit::wide_scalar_loads(),
                victory_shader::emit::shader_opt(),
            )
        );
        // The encoding part is this file as built.
        assert!(
            std::str::from_utf8(ENCODING)
                .unwrap()
                .contains("fn encode(translation: &Result<HostShader, String>)")
        );
    }

    #[test]
    fn view_marks_have_separate_translation_keys() {
        for wine in [false, true] {
            let options = options_for(wine, wine);
            let key = |view_marks| {
                translation_build(Options {
                    view_marks,
                    ..options
                })
            };
            assert_ne!(key(true), key(false));
        }
    }

    fn walk(dir: &std::path::Path) -> Vec<PathBuf> {
        let mut files = Vec::new();
        for e in std::fs::read_dir(dir).unwrap().flatten() {
            if e.path().is_dir() {
                files.extend(walk(&e.path()));
            } else {
                files.push(e.path());
            }
        }
        files
    }

    #[test]
    fn entries_round_trip_and_reject_damage() {
        let bytes = encode(&Ok(shader()));
        assert_eq!(decode(&bytes), Some(Ok(shader())));
        assert_eq!(decode(&bytes[..bytes.len() - 1]), None);
        assert_eq!(decode(&[bytes.as_slice(), &[0]].concat()), None);
        assert_eq!(decode(b"VSH0"), None);
        let refusal = Err("GFX10 code at 0x10: x is not supported".to_owned());
        assert_eq!(decode(&encode(&refusal)), Some(refusal));
        assert_eq!(decode(&[b"VSHE".as_slice(), &[0xff]].concat()), None);
    }
}
