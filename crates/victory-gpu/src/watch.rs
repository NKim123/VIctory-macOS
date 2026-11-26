//! Which title memory changed since a host copy took it (decision 0007,
//! "Copies of title memory" and "Change tracking"). Each host copy of title
//! memory (`crate::shadow`) and each texture's staging (`crate::texture`)
//! tracks its memory ([`Tracked`]): the 64 KiB pages it has not taken since
//! they last changed. Taking pages makes their blocks read-only (guards);
//! the first write to a guarded block faults into our vectored handler,
//! which gives write access back and marks the block changed in every copy
//! holding it. Writers that do not fault say so: our own and the kernel's
//! writes into title memory ([`writing`], before they write), and the
//! runtime mapping or unmapping memory ([`title_memory_remapped`]). So
//! bringing a copy up to date costs what changed since, not how much memory
//! it covers: no system call for memory nobody wrote.
//!
//! Only the runtime's GPU memory is tracked: its write-watched pools, its
//! physical-page views and their placeholders. Other memory a copy takes
//! (the C heap, where the kernel writes system call results into buffers a
//! guard would make fail) is copied whole every time.
//!
//! Blocks that cannot be made read-only (placeholders, nothing mapped)
//! read as zeros and count as taken: only a mapping changes them, which the
//! runtime reports.
//!
//! Physical pages may be mapped at several addresses (the title fills
//! pages through a pool view and maps them into its heaps), so a write at
//! one is a change at all of them: the runtime says where else memory is
//! mapped whenever it maps it ([`title_memory_aliased`]); taking a page
//! guards every address showing it, and a write at any of them marks the
//! trackers at all. (Asking the runtime at each take instead waited for
//! its lock, which the title's page mappings hold: 4 s per world frame.)
//! DirectStorage writing pages through views of the runtime's own reports
//! the addresses showing them as written.
//!
//! shortcut: a system call writing into a guarded block fails; the known
//! ones (file reads, DirectStorage) open their blocks first. Memory freed
//! and allocated again while a tracker holds it is not seen.
//!
//! Under Rosetta (Wine on macOS) memory is tracked only when asked for, with
//! the stores Rosetta cannot resume rewritten at load time: see
//! [`fault_tracking`].

use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, Mutex, RwLock, RwLockReadGuard, RwLockWriteGuard};

/// Whether title writes may fault into a tracker: the guards here and the
/// write watch of the runtime's pools (`victory_hle`'s `XMemVirtualAlloc`,
/// which Wine emulates with page protection). Every store that hits a
/// tracked page must be resumed after the fault. Rosetta cannot resume
/// some: it prints `unexpectedly need to EmulateForward on a synchronous
/// exception` and the process hangs (VIctory-macOS issue #1). On macOS 27
/// those are AVX masked stores that start in writable memory and end in a
/// protected page, and `vcvtps2ph` stores anywhere in one; the loader
/// rewrites both as plain stores whenever tracking runs under Rosetta
/// ([`stores_need_patching`]). Earlier Rosetta also stops on a plain 32-byte
/// `vmovups` across pages (macOS 26.6, #1), and #144 hung on an M1 with
/// both switches on, so under Rosetta tracking stays off unless
/// `VICTORY_FAULT_TRACKING=1` asks for it. Without tracking, copies take
/// their memory whole every time, as they do for memory that is not the
/// runtime's (tens of GB per world frame, #50). `0` turns tracking off
/// anywhere. Tests keep it on: their stores do not cross pages.
pub fn fault_tracking() -> bool {
    static ON: LazyLock<bool> = LazyLock::new(|| {
        decide(
            std::env::var("VICTORY_FAULT_TRACKING").ok().as_deref(),
            cfg!(test) || !rosetta(),
        )
    });
    *ON
}

/// Whether tracking runs under Rosetta, so the stores Rosetta cannot resume
/// after a fault must be rewritten at load time (`victory_loader::masked`).
pub fn stores_need_patching() -> bool {
    fault_tracking() && rosetta()
}

/// `fault_tracking` from the variable's value and the default.
fn decide(var: Option<&str>, default: bool) -> bool {
    match var {
        Some("1") => true,
        Some("0") => false,
        _ => default,
    }
}

/// Whether x86-64 code here runs under Rosetta: Wine on macOS, whose
/// x86-64 Wine only runs that way on the Apple silicon Macs D3DMetal needs.
pub fn rosetta() -> bool {
    use std::ffi::{CStr, c_char};
    use windows_sys::Win32::System::LibraryLoader::{GetModuleHandleA, GetProcAddress};
    type HostVersion = unsafe extern "C" fn(*mut *const c_char, *mut *const c_char);
    // SAFETY: ntdll is always loaded and both names end with a NUL. Wine's
    // `wine_get_host_version` (cdecl, two out pointers) sets them to static
    // strings: the host's `uname` system name and release.
    unsafe {
        let Some(f) = GetProcAddress(
            GetModuleHandleA(c"ntdll.dll".as_ptr().cast()),
            c"wine_get_host_version".as_ptr().cast(),
        ) else {
            return false;
        };
        let f: HostVersion = std::mem::transmute(f);
        let (mut sysname, mut release) = (std::ptr::null(), std::ptr::null());
        f(&mut sysname, &mut release);
        !sysname.is_null() && CStr::from_ptr(sysname) == c"Darwin"
    }
}

/// The unit tracked and guarded: 64 KiB, the console's page.
pub(crate) const PAGE: usize = 64 << 10;

const PART: usize = 4 << 10;
const WHOLE: u16 = u16::MAX;

fn page_parts() -> bool {
    static ON: LazyLock<bool> =
        LazyLock::new(|| std::env::var_os("VICTORY_PAGE_PARTS").is_none_or(|v| v != "0"));
    *ON
}

/// Enable shared guard operations. The exclusive path remains the default.
pub(crate) fn port_guard() -> bool {
    static ON: LazyLock<bool> =
        LazyLock::new(|| std::env::var_os("VICTORY_PORT_GUARD").is_some_and(|v| v == "1"));
    *ON
}

pub(crate) fn port_guard_plan() -> bool {
    static ON: LazyLock<bool> =
        LazyLock::new(|| std::env::var_os("VICTORY_PORT_GUARD_PLAN").is_none_or(|v| v != "0"));
    port_guard() && *ON
}

/// Enable sharded dirty journals and early descriptor-heap refresh asks.
pub(crate) fn port_journal() -> bool {
    static ON: LazyLock<bool> =
        LazyLock::new(|| std::env::var_os("VICTORY_PORT_JOURNAL").is_some_and(|v| v == "1"));
    *ON
}

/// Known dirty 4 KiB parts and whether the open guard has let through writes
/// still to be collected from the write watch. Unknown writes set every bit.
#[derive(Clone, Copy)]
struct DirtyPage {
    parts: u16,
    pending: bool,
}

impl DirtyPage {
    const WHOLE: Self = Self {
        parts: WHOLE,
        pending: false,
    };
    const WATCH: Self = Self {
        parts: 0,
        pending: true,
    };

    fn merge(&mut self, other: Self) {
        self.parts |= other.parts;
        self.pending = self.parts != WHOLE && (self.pending || other.pending);
    }

    fn resolve(&mut self, parts: u16) {
        if self.pending {
            self.parts |= parts;
            self.pending = false;
        }
    }

    fn mask(self) -> u16 {
        if self.pending { WHOLE } else { self.parts }
    }
}

#[cfg(target_arch = "x86_64")]
mod write_sites {
    use std::ops::Range;
    use std::sync::OnceLock;
    use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
    use windows_sys::Win32::System::Diagnostics::Debug::{CONTEXT, EXCEPTION_POINTERS};

    use super::PAGE;

    const SLOTS: usize = 4096;
    static GUARDED: [AtomicUsize; SLOTS] = [const { AtomicUsize::new(0) }; SLOTS];
    static OBSERVER: OnceLock<fn(&CONTEXT, usize)> = OnceLock::new();

    /// Guarded, write-watched page bases, indexed by the low bits of the
    /// page number. Collisions drop an entry and can only miss a fast path.
    pub fn write_site_guards() -> (&'static [AtomicUsize], u8) {
        (&GUARDED, PAGE.trailing_zeros() as u8)
    }

    /// Observe faults on guards of ordinary write-watched RAM, before the
    /// guard opens. The callback runs under the guard lock.
    pub fn observe_write_sites(observer: fn(&CONTEXT, usize)) {
        let _ = OBSERVER.set(observer);
    }

    pub(super) fn publish(pages: &[usize], ends: &[usize]) {
        if OBSERVER.get().is_none() {
            return;
        }
        for (&page, &end) in pages.iter().zip(ends) {
            // RuntimeRegions returns MAX only for a write-watched pool.
            if end == usize::MAX {
                GUARDED[page % SLOTS].store(page * PAGE, Relaxed);
            }
        }
    }

    pub(super) fn forget(pages: Range<usize>) {
        if OBSERVER.get().is_none() {
            return;
        }
        for page in pages {
            let _ = GUARDED[page % SLOTS].compare_exchange(page * PAGE, 0, Relaxed, Relaxed);
        }
    }

    /// The caller holds the guard lock and received `info` from the OS.
    pub(super) unsafe fn observe(info: *mut EXCEPTION_POINTERS, at: usize) {
        let page = at / PAGE;
        if page != 0
            && let Some(observer) = OBSERVER.get()
            && GUARDED[page % SLOTS].load(Relaxed) == page * PAGE
        {
            // SAFETY: the system provides the faulting thread's context.
            observer(unsafe { &*(*info).ContextRecord }, at);
        }
    }
}

#[cfg(target_arch = "x86_64")]
pub use write_sites::{observe_write_sites, write_site_guards};

/// The pages a tracker has not taken since they changed, what it saves of
/// pages it armed, its bit in [`DIRTY`] (none: always taken as dirty), and
/// whether it is a host buffer copy's (the only trackers [`retake`] marks).
struct Changed(DirtyJournal, Mutex<Saved>, Option<u32>, bool);

/// A bit per tracker, set while it holds changed pages: set and cleared
/// under the tracker's lock, read without one ([`Tracked::clean`]). Dense,
/// so that a submission asking about every copy its heaps reach (~14,000)
/// reads a few KB instead of a cache line per copy.
static DIRTY: [AtomicU64; DIRTY_WORDS] = [const { AtomicU64::new(0) }; DIRTY_WORDS];
const DIRTY_WORDS: usize = 1 << 14;

// Each bit names a journal shard that still holds dirty pages. Updating it
// under that shard's lock makes concurrent publishes and clears independent.
static DIRTY_SHARDS: [AtomicU64; DIRTY_WORDS * 64] =
    [const { AtomicU64::new(0) }; DIRTY_WORDS * 64];

const JOURNAL_SHARDS: usize = 64;
const SHARD_PAGES: usize = 16;
type DirtyMap = BTreeMap<usize, DirtyPage>;

enum DirtyJournal {
    Single(Mutex<DirtyMap>),
    Sharded(Box<[Mutex<DirtyMap>; JOURNAL_SHARDS]>),
}

impl DirtyJournal {
    fn new(pages: &[usize], bit: Option<u32>, sharded: bool) -> Self {
        // Keep the inline map for small copies. A 64-lock directory there
        // would cost more memory than their dirty entries.
        let distributed = sharded && pages.len() >= JOURNAL_SHARDS;
        let journal = if distributed {
            Self::Sharded(Box::new(std::array::from_fn(|_| Mutex::default())))
        } else {
            Self::Single(Mutex::default())
        };
        for &page in pages {
            journal.update(page, bit, |map| map.insert(page, DirtyPage::WHOLE));
        }
        journal
    }

    fn shard_of(page: usize) -> usize {
        page / SHARD_PAGES % JOURNAL_SHARDS
    }

    fn shard(&self, page: usize) -> (usize, &Mutex<DirtyMap>) {
        match self {
            Self::Single(map) => (0, map),
            Self::Sharded(shards) => {
                let i = Self::shard_of(page);
                (i, &shards[i])
            }
        }
    }

    fn get(&self, page: usize) -> Option<DirtyPage> {
        self.shard(page)
            .1
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&page)
            .copied()
    }

    fn update<T>(
        &self,
        page: usize,
        bit: Option<u32>,
        update: impl FnOnce(&mut DirtyMap) -> T,
    ) -> T {
        let (index, map) = self.shard(page);
        let mut map = map.lock().unwrap_or_else(|e| e.into_inner());
        let was_empty = map.is_empty();
        let result = update(&mut map);
        if let Some(bit) = bit
            && was_empty != map.is_empty()
        {
            if port_journal() {
                let entry = &DIRTY_SHARDS[bit as usize];
                if map.is_empty() {
                    entry.fetch_and(!(1 << index), Ordering::Release);
                } else {
                    entry.fetch_or(1 << index, Ordering::Release);
                }
            } else {
                set_dirty(Some(bit), !map.is_empty());
            }
        }
        result
    }

    fn range(&self, first: usize, last: usize) -> Vec<(usize, u16)> {
        let mut out = Vec::new();
        if first >= last {
            return out;
        }
        let mut collect = |map: &Mutex<DirtyMap>| {
            let map = map.lock().unwrap_or_else(|e| e.into_inner());
            out.extend(map.range(first..last).map(|(&page, _)| (page, WHOLE)));
        };
        match self {
            Self::Single(map) => collect(map),
            Self::Sharded(shards) => {
                let count = (last.div_ceil(SHARD_PAGES) - first / SHARD_PAGES).min(JOURNAL_SHARDS);
                let start = Self::shard_of(first);
                for i in 0..count {
                    collect(&shards[(start + i) % JOURNAL_SHARDS]);
                }
                out.sort_unstable_by_key(|&(page, _)| page);
            }
        }
        out
    }
}

/// Bits of [`DIRTY`] given back, and the first never handed out.
static FREE_BITS: Mutex<(Vec<u32>, u32)> = Mutex::new((Vec::new(), 0));

fn dirty_bit() -> Option<u32> {
    let mut free = FREE_BITS.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(bit) = free.0.pop() {
        return Some(bit);
    }
    let bit = free.1;
    (bit < (DIRTY_WORDS * 64) as u32).then(|| {
        free.1 += 1;
        bit
    })
}

fn set_dirty(bit: Option<u32>, dirty: bool) {
    if let Some(bit) = bit {
        let (word, mask) = (&DIRTY[bit as usize / 64], 1u64 << (bit % 64));
        if dirty {
            word.fetch_or(mask, Ordering::Release);
        } else {
            word.fetch_and(!mask, Ordering::Release);
        }
    }
}

/// Whether the tracker with bit `bit` holds no changed page (see
/// [`Tracked::clean`]).
pub(crate) fn clean_bit(bit: Option<u32>) -> bool {
    bit.is_some_and(|bit| {
        if port_journal() {
            DIRTY_SHARDS[bit as usize].load(Ordering::Acquire) == 0
        } else {
            DIRTY[bit as usize / 64].load(Ordering::Acquire) & 1 << (bit % 64) == 0
        }
    })
}

impl Drop for Changed {
    fn drop(&mut self) {
        if let Some(bit) = self.2 {
            if port_journal() {
                DIRTY_SHARDS[bit as usize].store(0, Ordering::Release);
            } else {
                set_dirty(Some(bit), false);
            }
            FREE_BITS.lock().unwrap_or_else(|e| e.into_inner()).0.push(bit);
        }
    }
}

/// Pages (page numbers) whose bytes a tracker keeps when they are first
/// written after it took them, and those bytes: while shaders may be
/// writing its copy of them, what it held before (`crate::shadow`'s
/// copy-back of marked pages), with what the copy took of the page since
/// laid over them ([`Tracked::refreshed`]).
#[derive(Default)]
struct Saved {
    /// The armed pages as runs `[first, last)`, sorted, apart and not
    /// touching (runs armed that meet merge): a root view declares its
    /// copy from its address to the end, the world's 174 MB heap many
    /// times a submission, and a set entry per page cost 30 microseconds a
    /// declaration; descriptor tables arm many sub-ranges end to end, and
    /// the write handler checks a page under the guards' lock.
    armed: Vec<(usize, usize)>,
    /// Pages of those runs that are not armed any more.
    unarmed: BTreeSet<usize>,
    bytes: std::collections::HashMap<usize, Box<[u8]>>,
}

impl Saved {
    /// Whether page `page` is armed.
    fn armed(&self, page: usize) -> bool {
        // The first run ending after `page` is the only one that can hold it.
        let i = self.armed.partition_point(|&(_, last)| last <= page);
        self.armed.get(i).is_some_and(|&(first, _)| first <= page) && !self.unarmed.contains(&page)
    }

    /// Arms pages `[first, last)`.
    fn arm(&mut self, first: usize, last: usize) {
        if first >= last {
            return;
        }
        if !self.unarmed.is_empty() {
            let pages: Vec<usize> = self.unarmed.range(first..last).copied().collect();
            for page in pages {
                self.unarmed.remove(&page);
            }
        }
        // Runs `[lo, hi)` overlap or touch the new one: it replaces them.
        let lo = self.armed.partition_point(|&(_, b)| b < first);
        let hi = self.armed.partition_point(|&(a, _)| a <= last);
        if lo + 1 == hi && self.armed[lo].0 <= first && last <= self.armed[lo].1 {
            return;
        }
        let run = if lo < hi {
            (self.armed[lo].0.min(first), self.armed[hi - 1].1.max(last))
        } else {
            (first, last)
        };
        self.armed.splice(lo..hi, [run]);
    }
}

impl Changed {
    fn clear_page(&self, page: usize) {
        self.0.update(page, self.2, |map| map.remove(&page));
        self.taken(page);
    }

    /// Page `page` was taken: what it saved of it is stale.
    fn taken(&self, page: usize) {
        self.1.lock().unwrap_or_else(|e| e.into_inner()).bytes.remove(&page);
    }

    /// Page `page` is about to change for the first time since it was
    /// taken: its bytes now, if armed for it.
    fn save(&self, page: usize) {
        let mut saved = self.1.lock().unwrap_or_else(|e| e.into_inner());
        if !saved.armed(page) || saved.bytes.contains_key(&page) {
            return;
        }
        let mut bytes = vec![0u8; PAGE].into_boxed_slice();
        if crate::host::read_into((page * PAGE) as *const u8, &mut bytes) {
            saved.bytes.insert(page, bytes);
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::SavedPages, 1);
        }
    }
}

/// Byte runs `[start, end)` in order.
pub(crate) type Runs = Vec<(usize, usize)>;

#[cfg(feature = "diagnostics")]
thread_local! {
    static PROTECT_OWNER: std::cell::Cell<Option<crate::copystat::C>> = const { std::cell::Cell::new(None) };
}

#[cfg(feature = "diagnostics")]
pub(crate) struct ProtectOwner(Option<crate::copystat::C>);

#[cfg(feature = "diagnostics")]
impl ProtectOwner {
    pub(crate) fn set(owner: crate::copystat::C) -> Self {
        Self(PROTECT_OWNER.with(|cell| cell.replace(Some(owner))))
    }
}

#[cfg(feature = "diagnostics")]
impl Drop for ProtectOwner {
    fn drop(&mut self) {
        PROTECT_OWNER.with(|cell| cell.set(self.0.take()));
    }
}

/// Title memory a host copy or texture staging takes: bytes `[start, end)`.
pub(crate) struct Tracked {
    start: usize,
    end: usize,
    changed: Arc<Changed>,
    /// Parts of the range on pages that are not the runtime's memory (in
    /// order): not tracked, copied whole at each use ([`Tracked::untracked`]).
    untracked: Vec<(usize, usize)>,
}

/// The blocks we made read-only. The value records whether the block uses
/// kernel write watch. Only the publishing thread holds this lock, and
/// only for a lookup or update, never across a memory system call.
#[derive(Default)]
struct Protected(RwLock<BTreeMap<usize, bool>>);

impl Protected {
    fn contains(&self, page: &usize) -> bool {
        self.0
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .contains_key(page)
    }

    fn watched(&self, page: usize) -> bool {
        self.0.read().unwrap_or_else(|e| e.into_inner()).get(&page) == Some(&true)
    }

    fn remove(&self, page: usize) -> bool {
        self.0
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .remove(&page)
            .is_some()
    }

    fn any(&self, first: usize, last: usize) -> bool {
        self.0
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .range(first..last)
            .next()
            .is_some()
    }

    fn range(&self, first: usize, last: usize) -> Vec<usize> {
        self.0
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .range(first..last)
            .map(|(&p, _)| p)
            .collect()
    }

    fn insert(&self, pages: &[usize], ends: &[usize]) {
        let mut protected = self.0.write().unwrap_or_else(|e| e.into_inner());
        protected.extend(
            pages
                .iter()
                .copied()
                .zip(ends.iter().map(|&end| page_parts() && end == usize::MAX)),
        );
    }
}

struct Guards {
    protected: Protected,
    /// The trackers holding each page.
    holders: BTreeMap<usize, Vec<Arc<Changed>>>,
    /// Pages `[first, last)` of writes the guards do not see that are under
    /// way ([`Writing`]): not guarded until they end, or the write fails.
    busy: Vec<(usize, usize)>,
    /// Pages showing the same physical pages elsewhere, both ways, as the
    /// runtime told when it mapped them: a write at one marks the others,
    /// with no question to the runtime (whose lock its page mappings hold
    /// for long).
    alias_of: BTreeMap<usize, Vec<usize>>,
    /// Tracked pages written this frame and the frame before (diagnostics).
    #[cfg(feature = "diagnostics")]
    written: Mutex<BTreeSet<usize>>,
    #[cfg(feature = "diagnostics")]
    written_before: BTreeSet<usize>,
}

static GUARDS: RwLock<Guards> = RwLock::new(Guards {
    protected: Protected(RwLock::new(BTreeMap::new())),
    holders: BTreeMap::new(),
    busy: Vec::new(),
    alias_of: BTreeMap::new(),
    #[cfg(feature = "diagnostics")]
    written: Mutex::new(BTreeSet::new()),
    #[cfg(feature = "diagnostics")]
    written_before: BTreeSet::new(),
});

thread_local! {
    /// Whether this thread holds the guard lock. A guarded block it faults
    /// on then is one the lock's own work touched (#50: an allocation made
    /// under the lock landed in a guarded block of the host heap, which
    /// title XMem blocks share), and taking the lock again would hang it.
    static HOLDING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Blocks opened by faults taken under the guard lock, for its next holder
/// to mark changed (0: free slot). Fixed: the handler must not allocate.
static OPENED_UNDER_LOCK: [std::sync::atomic::AtomicUsize; 64] =
    [const { std::sync::atomic::AtomicUsize::new(0) }; 64];

/// Whether any block waits in them: the next holder of the lock scans
/// [`OPENED_UNDER_LOCK`] only when one was opened, not on every take. Only
/// touched while the lock is held, so a clear scan leaves no block behind.
static OPENED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// A fault on the block `page` by the thread holding the guard lock: write
/// access back now, marked changed by the lock's next holder.
// shortcut: assumes a read-only block this thread writes while
// holding the lock is one of ours; with all 64 slots taken the block stays
// open without being marked (counted); a bigger table if that shows up.
fn open_under_lock(page: usize) {
    use std::sync::atomic::Ordering::{AcqRel, Relaxed, Release};
    use windows_sys::Win32::System::Memory::{PAGE_READWRITE, VirtualProtect};
    let mut old = 0;
    #[cfg(target_arch = "x86_64")]
    write_sites::forget(page..page + 1);
    // SAFETY: a block of this process; fails harmlessly if it is gone.
    unsafe { VirtualProtect((page * PAGE) as *const _, PAGE, PAGE_READWRITE, &mut old) };
    if OPENED_UNDER_LOCK.iter().any(|s| s.compare_exchange(0, page, AcqRel, Relaxed).is_ok()) {
        OPENED.store(true, Release);
    } else {
        static LOST: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        LOST.fetch_add(1, Relaxed);
    }
}

/// The guard lock, held while this lives. With diagnostics, timed for the
/// line that took it ([`held_times`]).
struct Held {
    guard: Option<RwLockWriteGuard<'static, Guards>>,
    #[cfg(feature = "diagnostics")]
    since: std::time::Instant,
    #[cfg(feature = "diagnostics")]
    line: u32,
}

#[track_caller]
fn guards() -> Held {
    use std::sync::atomic::Ordering::AcqRel;
    let guard = GUARDS.write().unwrap_or_else(|e| e.into_inner());
    HOLDING.with(|h| h.set(true));
    let held = Held {
        guard: Some(guard),
        #[cfg(feature = "diagnostics")]
        since: std::time::Instant::now(),
        #[cfg(feature = "diagnostics")]
        line: std::panic::Location::caller().line(),
    };
    if OPENED.swap(false, AcqRel) {
        for slot in &OPENED_UNDER_LOCK {
            let page = slot.swap(0, AcqRel);
            if page != 0 {
                changed(&held, page, page + 1, false);
            }
        }
    }
    held
}

/// Shared holders leave mappings, external writers and tracker lifetimes
/// fixed. Only page claims serialize protection changes and their dirty
/// marks. A structural change takes the lock exclusively and needs no
/// claims. Drain the exclusive path's emergency fault slots first.
fn guards_shared() -> RwLockReadGuard<'static, Guards> {
    loop {
        let guard = GUARDS.read().unwrap_or_else(|e| e.into_inner());
        if !OPENED.load(Ordering::Acquire) {
            return guard;
        }
        drop(guard);
        drop(guards());
    }
}

/// Fixed-size striped claims, indexed by 64 KiB block. Collisions only
/// serialize otherwise independent blocks. Always lock in stripe order
/// and include the other views before clearing a tracker's dirty entry.
const CLAIM_COUNT: usize = 4096;
static CLAIMS: [Mutex<()>; CLAIM_COUNT] = [const { Mutex::new(()) }; CLAIM_COUNT];
const CLAIM_BATCH: usize = 64;

fn claim_blocks(g: &Guards, pages: &[usize]) -> Vec<std::sync::MutexGuard<'static, ()>> {
    let mut stripes = [0u64; CLAIM_COUNT / 64];
    for page in pages {
        for block in std::iter::once(page).chain(g.alias_of.get(page).into_iter().flatten()) {
            let index = block % CLAIM_COUNT;
            stripes[index / 64] |= 1 << (index % 64);
        }
    }
    let mut claims =
        Vec::with_capacity(stripes.iter().map(|bits| bits.count_ones() as usize).sum());
    for (word, mut bits) in stripes.into_iter().enumerate() {
        while bits != 0 {
            let index = word * 64 + bits.trailing_zeros() as usize;
            bits &= bits - 1;
            claims.push(CLAIMS[index].lock().unwrap_or_else(|e| e.into_inner()));
        }
    }
    claims
}

impl std::ops::Deref for Held {
    type Target = Guards;
    fn deref(&self) -> &Guards {
        self.guard.as_ref().expect("held until dropped")
    }
}

impl std::ops::DerefMut for Held {
    fn deref_mut(&mut self) -> &mut Guards {
        self.guard.as_mut().expect("held until dropped")
    }
}

/// Per line taking the guard lock: the line, microseconds held in all, the
/// longest hold (no allocation: the fault handler takes the lock).
#[cfg(feature = "diagnostics")]
static HELD: [[std::sync::atomic::AtomicU64; 3]; 16] =
    [const { [const { std::sync::atomic::AtomicU64::new(0) }; 3] }; 16];

impl Drop for Held {
    fn drop(&mut self) {
        drop(self.guard.take());
        HOLDING.with(|h| h.set(false));
        #[cfg(feature = "diagnostics")]
        {
            use std::sync::atomic::Ordering::Relaxed;
            let us = self.since.elapsed().as_micros() as u64;
            let line = u64::from(self.line);
            if let Some(slot) = HELD
                .iter()
                .find(|s| s[0].load(Relaxed) == line || s[0].compare_exchange(0, line, Relaxed, Relaxed).is_ok())
            {
                slot[1].fetch_add(us, Relaxed);
                slot[2].fetch_max(us, Relaxed);
            }
        }
    }
}

/// Since the last call, per line of this file taking the guard lock: (line,
/// milliseconds held, longest hold in milliseconds), most held first.
#[cfg(feature = "diagnostics")]
pub(crate) fn held_times() -> Vec<(u64, f64, f64)> {
    use std::sync::atomic::Ordering::Relaxed;
    let mut out: Vec<(u64, f64, f64)> = HELD
        .iter()
        .filter(|s| s[0].load(Relaxed) != 0)
        .map(|s| (s[0].load(Relaxed), s[1].swap(0, Relaxed) as f64 / 1e3, s[2].swap(0, Relaxed) as f64 / 1e3))
        .collect();
    out.sort_by(|a, b| b.1.total_cmp(&a.1));
    out
}

/// (page, other page) for the pages (page numbers) showing the same memory
/// as `pages` elsewhere.
fn aliases(g: &Guards, pages: &[usize]) -> Vec<(usize, usize)> {
    pages
        .iter()
        .flat_map(|&p| g.alias_of.get(&p).into_iter().flatten().map(move |&a| (p, a)))
        .collect()
}

/// Guards the other mappings of taken pages, returning those that still
/// have an unguarded mapping (cannot be taken yet).
fn guard_aliases(g: &Guards, pairs: &[(usize, usize)]) -> BTreeSet<usize> {
    if pairs.is_empty() {
        return BTreeSet::new();
    }
    let others: Vec<usize> = pairs.iter().map(|&(_, a)| a).collect();
    let guarded: BTreeSet<usize> = protect(g, &others).into_iter().collect();
    pairs
        .iter()
        .filter(|(_, a)| !guarded.contains(a))
        .map(|&(p, _)| p)
        .collect()
}

/// Tracks `[start, start + len)` if it is the runtime's GPU memory (every
/// page changed at first); `None` for other memory, and with `views` false
/// for anything but its write-watched pools (physical-page views are
/// written through other views showing the same pages: the title fills
/// streamed textures and video planes through a pool view and maps the
/// pages into its tiled heap, where a guard sees none of it).
///
/// Only the pages of the range that are such memory are tracked; the rest is
/// never guarded and is copied whole at each use, as memory with no tracker.
pub(crate) fn track(start: usize, len: usize, views: bool) -> Option<Tracked> {
    track_as(start, len, views, false)
}

/// [`track`] for a host buffer copy: what shaders write lands in it, so a
/// [`retake`] of its memory marks it.
pub(crate) fn track_copy(start: usize, len: usize, views: bool) -> Option<Tracked> {
    track_as(start, len, views, true)
}

fn track_as(start: usize, len: usize, views: bool, copy: bool) -> Option<Tracked> {
    if len == 0 || !fault_tracking() {
        return None;
    }
    let end = start + len;
    let (first, last) = (start / PAGE, end.div_ceil(PAGE));
    let known_views = port_guard() && {
        let g = guards_shared();
        g.alias_of.range(first..last).count() == last - first
    };
    let pages = match (known_views, views) {
        (true, true) => (first..last).collect(),
        (true, false) => return None,
        _ => runtime_pages(start, end, views),
    };
    if pages.is_empty() {
        return None;
    }
    install();
    // The rest of the range, as byte runs.
    let mut untracked: Vec<(usize, usize)> = Vec::new();
    let mut at = start;
    for &page in pages.iter().chain([&usize::MAX]) {
        let (a, b) = ((page.saturating_mul(PAGE)).max(start).min(end), (page.saturating_add(1).saturating_mul(PAGE)).min(end));
        if at < a {
            untracked.push((at, a));
        }
        at = at.max(b);
    }
    // A tracker with untracked parts is never clean: no bit.
    let bit = if untracked.is_empty() { dirty_bit() } else { None };
    let dirty = DirtyJournal::new(&pages, bit, port_journal());
    let changed = Arc::new(Changed(dirty, Mutex::default(), bit, copy));
    let mut g = guards();
    for &page in &pages {
        g.holders.entry(page).or_default().push(changed.clone());
    }
    drop(g);
    Some(Tracked {
        start,
        end,
        changed,
        untracked,
    })
}

/// Whether the page holding `at` is in a write-watched pool (one lookup).
fn write_watched(at: usize) -> bool {
    use windows_sys::Win32::System::Memory::GetWriteWatch;
    let page = at & !(4096 - 1);
    let (mut found, mut count, mut granularity) = (0usize, 1usize, 0u32);
    // SAFETY: room for one answer; no reset, so nothing changes.
    unsafe { GetWriteWatch(0, page as *const _, 4096, (&raw mut found).cast(), &mut count, &mut granularity) == 0 }
}

const WRITE_WATCH_RESET: u32 = 1;

/// Take one block's write watch after making the block read-only again.
/// The guard lock serializes resets and publication to the block's holders.
/// Old kernel bits from before the first take may add parts to the answer.
fn watched_parts(page: usize, flags: u32) -> u16 {
    #[cfg(test)]
    if flags == WRITE_WATCH_RESET {
        tests::PART_WATCH_QUERIES.with(|n| n.set(n.get() + 1));
    }
    use windows_sys::Win32::System::Memory::GetWriteWatch;
    let start = page * PAGE;
    let mut addresses = [std::ptr::null_mut(); PAGE / PART];
    let (mut count, mut granularity) = (addresses.len(), 0);
    // SAFETY: the kernel receives space for every 4 KiB address in the block.
    let status = unsafe {
        GetWriteWatch(
            flags,
            start as *const _,
            PAGE,
            addresses.as_mut_ptr(),
            &mut count,
            &mut granularity,
        )
    };
    if status != 0 || granularity as usize != PART || count == 0 || count > addresses.len() {
        return WHOLE;
    }
    let mut parts = 0;
    for &address in &addresses[..count] {
        let offset = (address as usize).wrapping_sub(start);
        if offset >= PAGE || !offset.is_multiple_of(PART) {
            return WHOLE;
        }
        parts |= 1 << (offset / PART);
    }
    parts
}

/// The pages of `[start, end)` that are the runtime's GPU memory: a
/// write-watched pool, or (with `views`) anything but plain committed memory
/// and images (views, placeholders, nothing mapped yet). Decided for every
/// region the range covers, and a page touching any other memory is left
/// out: a range can run past its allocation (a 4 MiB window of memory
/// outside buffers, `crate::shadow`), and when the first page alone decided,
/// a window starting in free memory had the process heap's pages after it
/// guarded; the heap's own code then faulted while holding its lock, and the
/// guard handler waited for the guard lock, whose holder waited for the heap
/// lock (the crate's tests hung).
///
/// shortcut: memory free or reserved when the range is tracked counts as the
/// runtime's; another allocator committing it before the first take would
/// have it guarded.
fn runtime_pages(start: usize, end: usize, views: bool) -> Vec<usize> {
    use windows_sys::Win32::System::Memory::{
        GetWriteWatch, MEM_COMMIT, MEM_IMAGE, MEM_PRIVATE, MEMORY_BASIC_INFORMATION, VirtualQuery,
    };
    let watched = |at: usize, len: usize| {
        let (mut found, mut count, mut granularity) = (0usize, 1usize, 0u32);
        // SAFETY: room for one answer; no reset, so nothing changes.
        unsafe { GetWriteWatch(0, at as *const _, len, (&raw mut found).cast(), &mut count, &mut granularity) == 0 }
    };
    let (first, last) = (start / PAGE, end.div_ceil(PAGE));
    // One pool holds it all (the common case; asking `VirtualQuery` costs
    // milliseconds in the runtime's big pools).
    if watched(start & !(4096 - 1), end - (start & !(4096 - 1))) {
        return (first..last).collect();
    }
    let (mut runtime, mut other) = (std::collections::BTreeSet::new(), std::collections::BTreeSet::new());
    let mut at = start & !(4096 - 1);
    while at < end {
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        if unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) } == 0 {
            break;
        }
        let next = (info.BaseAddress as usize + info.RegionSize).max(at + 1).min(end);
        let ours = watched(at, next - at)
            || views && info.Type != MEM_IMAGE && !(info.State == MEM_COMMIT && info.Type == MEM_PRIVATE);
        (if ours { &mut runtime } else { &mut other }).extend(at / PAGE..next.div_ceil(PAGE));
        at = next;
    }
    runtime.difference(&other).copied().collect()
}

impl Drop for Tracked {
    /// Pages no tracker holds any more get write access back, so that the
    /// guard set never names memory freed and allocated again; unless another
    /// mapping of the page is held: the guard here is that tracker's too (it
    /// was set when that tracker took its page), and with it opened a write
    /// here went unseen there.
    fn drop(&mut self) {
        let mut g = guards();
        let mut free = Vec::new();
        for page in self.start / PAGE..self.end.div_ceil(PAGE) {
            if let Some(holders) = g.holders.get_mut(&page) {
                holders.retain(|c| !Arc::ptr_eq(c, &self.changed));
                if holders.is_empty() {
                    g.holders.remove(&page);
                    free.push(page);
                }
            }
        }
        for page in free {
            let watched_elsewhere = g.alias_of.get(&page).is_some_and(|o| o.iter().any(|p| g.holders.contains_key(p)));
            if !watched_elsewhere {
                open(&g, page);
            }
        }
    }
}

/// Gives write access back to page `page` if we guarded it.
fn open(g: &Guards, page: usize) -> bool {
    use windows_sys::Win32::System::Memory::{PAGE_READWRITE, VirtualProtect};
    if !g.protected.remove(page) {
        return false;
    }
    #[cfg(target_arch = "x86_64")]
    write_sites::forget(page..page + 1);
    let mut old = 0;
    // SAFETY: a block we made read-only (failing harmlessly if it is gone).
    unsafe { VirtualProtect((page * PAGE) as *const _, PAGE, PAGE_READWRITE, &mut old) };
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::OpenPages, 1);
    true
}

/// [`open`] of the guarded pages `[first, last)`, all in `protected`.
fn open_run(g: &Guards, first: usize, last: usize) {
    use windows_sys::Win32::System::Memory::{PAGE_READWRITE, VirtualProtect};
    if last - first == 1 {
        open(g, first);
        return;
    }
    let mut old = 0;
    #[cfg(target_arch = "x86_64")]
    write_sites::forget(first..last);
    // SAFETY: blocks we made read-only.
    if unsafe { VirtualProtect((first * PAGE) as *const _, (last - first) * PAGE, PAGE_READWRITE, &mut old) } == 0 {
        for page in first..last {
            open(g, page);
        }
        return;
    }
    for page in first..last {
        g.protected.remove(page);
    }
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::OpenPages, (last - first) as u64);
}

impl Tracked {
    /// Takes the changed pages of `[from, to)`: guards them (writes from
    /// now on mark them again) and returns the byte ranges to copy, cut to
    /// `[from, to)`, in order. The write watch can narrow those ranges to
    /// 4 KiB parts. A page whose tracked bytes lie only partly in
    /// `[from, to)` stays changed.
    pub(crate) fn take(&self, from: usize, to: usize) -> Vec<(usize, usize)> {
        let (copy, zero) = self.take_split(from, to);
        let mut runs: Vec<(usize, usize)> = copy.into_iter().chain(zero).collect();
        runs.sort_unstable();
        let mut merged: Vec<(usize, usize)> = Vec::with_capacity(runs.len());
        for (a, b) in runs {
            match merged.last_mut() {
                Some((_, end)) if *end == a => *end = b,
                _ => merged.push((a, b)),
            }
        }
        merged
    }

    /// [`Tracked::take`], the runs on pages with nothing mapped apart: they
    /// read as zeros, with no copy to try (each such page cost a dozen
    /// failing kernel reads, 7-30 ms a frame in run perf4).
    pub(crate) fn take_split(&self, from: usize, to: usize) -> (Runs, Runs) {
        let (from, to) = (from.max(self.start), to.min(self.end));
        if from >= to {
            return (Vec::new(), Vec::new());
        }
        let mut pages: Vec<(usize, u16)> = {
            #[cfg(feature = "diagnostics")]
            let waited = Timed(std::time::Instant::now(), crate::copystat::C::GuardWaitUs);
            let pages = self.changed.0.range(from / PAGE, to.div_ceil(PAGE));
            #[cfg(feature = "diagnostics")]
            drop(waited);
            pages
        };
        if pages.is_empty() {
            return (Vec::new(), Vec::new());
        }
        let whole: Vec<usize> = pages
            .iter()
            .map(|&(page, _)| page)
            .filter(|&p| (p * PAGE).max(self.start) >= from && ((p + 1) * PAGE).min(self.end) <= to)
            .collect();
        let mut unmapped = Vec::new();
        if port_guard() {
            for group in whole.chunks(CLAIM_BATCH) {
                #[cfg(feature = "diagnostics")]
                let waited = Timed(std::time::Instant::now(), crate::copystat::C::GuardWaitUs);
                let g = guards_shared();
                let _claims = claim_blocks(&g, group);
                #[cfg(feature = "diagnostics")]
                drop(waited);
                self.take_claimed(&g, group, &mut pages, &mut unmapped);
            }
        } else {
            #[cfg(feature = "diagnostics")]
            let waited = Timed(std::time::Instant::now(), crate::copystat::C::GuardWaitUs);
            let g = guards();
            #[cfg(feature = "diagnostics")]
            drop(waited);
            self.take_claimed(&g, &whole, &mut pages, &mut unmapped);
        }
        // Partial pages were copied but not taken. Read their remaining
        // dirty masks after the tracked pages in this range are rearmed.
        for (page, mask) in &mut pages {
            if whole.binary_search(page).is_err() {
                *mask = self.changed.0.get(*page).map_or(WHOLE, DirtyPage::mask);
            }
        }
        let (mut copy, mut zero): (Runs, Runs) = (Vec::new(), Vec::new());
        let mut unmapped = unmapped.into_iter().peekable();
        for (page, mask) in pages {
            if unmapped.next_if_eq(&page).is_some() {
                append_parts(&mut zero, page, WHOLE, from, to);
            } else {
                append_parts(&mut copy, page, mask, from, to);
            }
        }
        (copy, zero)
    }

    /// Rearm pages and consume their dirty parts with either the exclusive
    /// guard lock or the pages' claims. A failed rearm leaves dirt behind.
    fn take_claimed(
        &self,
        g: &Guards,
        group: &[usize],
        pages: &mut [(usize, u16)],
        unmapped: &mut Vec<usize>,
    ) {
        // Many exact copies share a block. A previous take, or a GPU
        // notification through a separate mapping, can leave every view
        // guarded with the dirty parts already resolved.
        if let [page] = *group
            && g.protected.contains(&page)
            && aliases_guarded(g, page)
            && let Some(dirty) = self.changed.0.get(page)
            && !dirty.pending
        {
            let i = pages
                .binary_search_by_key(&page, |&(p, _)| p)
                .expect("requested page");
            pages[i].1 = dirty.parts;
            self.changed.clear_page(page);
            return;
        }
        #[cfg(feature = "diagnostics")]
        let asked = Timed(std::time::Instant::now(), crate::copystat::C::AliasUs);
        let others = aliases(g, group);
        let unguarded = guard_aliases(g, &others);
        #[cfg(feature = "diagnostics")]
        drop(asked);
        let whole: Vec<_> = group
            .iter()
            .copied()
            .filter(|p| !unguarded.contains(p))
            .collect();
        let taken = protect_noting(g, &whole, unmapped);
        for &page in &taken {
            let pending = self.changed.0.get(page).is_some_and(|dirty| dirty.pending);
            if !pending {
                continue;
            }
            let parts = if g.protected.watched(page) && !g.alias_of.contains_key(&page) {
                watched_parts(page, WRITE_WATCH_RESET)
            } else {
                WHOLE
            };
            // A reset consumes a shared answer. Every holder must see it.
            for holder in &g.holders[&page] {
                holder.0.update(page, holder.2, |map| {
                    if let Some(dirty) = map.get_mut(&page) {
                        dirty.resolve(parts);
                    }
                });
            }
        }
        for &page in group {
            let i = pages
                .binary_search_by_key(&page, |&(p, _)| p)
                .expect("requested page");
            pages[i].1 = self.changed.0.get(page).map_or(WHOLE, DirtyPage::mask);
        }
        for page in taken {
            self.changed.clear_page(page);
        }
    }

    /// Whether page `page` (a page number) changed since it was taken.
    pub(crate) fn changed(&self, page: usize) -> bool {
        self.changed.0.get(page).is_some()
    }

    /// The caller made pages (page numbers) of its copy equal to title
    /// memory itself (what shaders wrote went back): taken and guarded.
    pub(crate) fn taken(&self, pages: &[usize]) {
        if port_guard() {
            for group in pages.chunks(CLAIM_BATCH) {
                let g = guards_shared();
                let _claims = claim_blocks(&g, group);
                self.taken_claimed(&g, group);
            }
        } else {
            let g = guards();
            self.taken_claimed(&g, pages);
        }
    }

    fn taken_claimed(&self, g: &Guards, pages: &[usize]) {
        let others = aliases(g, pages);
        let unguarded = guard_aliases(g, &others);
        let pages: Vec<usize> = pages
            .iter()
            .copied()
            .filter(|p| !unguarded.contains(p))
            .collect();
        for page in protect(g, &pages) {
            self.changed.clear_page(page);
        }
    }

    /// Whether no page changed since it was taken: then [`Tracked::take`]
    /// finds nothing, and asking costs no lock. A write racing this answer
    /// is found by the next asking, as one racing a take.
    pub(crate) fn clean(&self) -> bool {
        clean_bit(self.changed.2)
    }

    /// The tracker's bit for [`clean_bit`], for callers that ask about
    /// many trackers at once without touching each (none: never clean).
    pub(crate) fn bit(&self) -> Option<u32> {
        self.changed.2
    }

    /// Whether the entire part of this title page held by the copy is
    /// tracked. The remaining runs are sorted by byte address.
    pub(crate) fn covers_page(&self, page: usize) -> bool {
        let end = page.saturating_add(PAGE);
        let next = self.untracked.partition_point(|&(_, b)| b <= page);
        self.start < end
            && page < self.end
            && self.untracked.get(next).is_none_or(|&(a, _)| a >= end)
    }

    /// The parts of `[from, to)` that are not tracked (see
    /// [`Tracked::untracked`] field): to be copied whole at each use.
    pub(crate) fn untracked(&self, from: usize, to: usize) -> Vec<(usize, usize)> {
        self.untracked
            .iter()
            .map(|&(a, b)| (a.max(from), b.min(to)))
            .filter(|&(a, b)| a < b)
            .collect()
    }

    /// Shaders may write the copy's bytes `[from, to)` until [`Tracked::disarm`]:
    /// pages there written next after they were taken have their bytes kept
    /// first ([`Tracked::saved`]).
    pub(crate) fn arm(&self, from: usize, to: usize) {
        let (from, to) = (from.max(self.start), to.min(self.end));
        if from < to {
            self.changed.1.lock().unwrap_or_else(|e| e.into_inner()).arm(from / PAGE, to.div_ceil(PAGE));
        }
    }

    /// Page `page` (a page number) needs no saving any more: its copy-back
    /// is under way.
    pub(crate) fn unarm(&self, page: usize) {
        let mut saved = self.changed.1.lock().unwrap_or_else(|e| e.into_inner());
        if saved.armed(page) {
            saved.unarmed.insert(page);
        }
    }

    /// No shader writes the copy any more: nothing is armed or kept.
    pub(crate) fn disarm(&self) {
        let mut saved = self.changed.1.lock().unwrap_or_else(|e| e.into_inner());
        saved.armed.clear();
        saved.unarmed.clear();
        saved.bytes.clear();
    }

    /// The bytes page `page` (a page number) held when it was first written
    /// since it was taken, if it was armed then (and so changed), with what
    /// the copy took of it since: the copy's bytes before the submission.
    pub(crate) fn saved(&self, page: usize) -> Option<Box<[u8]>> {
        self.changed.1.lock().unwrap_or_else(|e| e.into_inner()).bytes.remove(&page)
    }

    /// The copy took `bytes` at title address `from`: any saved bytes of pages
    /// it covers take them too, so marked copy-back neither takes a partial
    /// refresh for the GPU's nor drops a shader write back to the pre-fault byte.
    pub(crate) fn refreshed(&self, from: usize, bytes: &[u8]) {
        if bytes.is_empty() {
            return;
        }
        let mut saved = self.changed.1.lock().unwrap_or_else(|e| e.into_inner());
        if saved.bytes.is_empty() {
            return;
        }
        let to = from + bytes.len();
        for page in from / PAGE..to.div_ceil(PAGE) {
            if let Some(before) = saved.bytes.get_mut(&page) {
                let (a, b) = (from.max(page * PAGE), to.min((page + 1) * PAGE));
                before[a - page * PAGE..b - page * PAGE]
                    .copy_from_slice(&bytes[a - from..b - from]);
            }
        }
    }
}

/// Append dirty parts, joining consecutive ones across block boundaries.
fn append_parts(runs: &mut Runs, page: usize, mut mask: u16, from: usize, to: usize) {
    while mask != 0 {
        let first = mask.trailing_zeros();
        let count = (mask >> first).trailing_ones();
        let end = first + count;
        mask &= (u32::MAX << end) as u16;
        let a = (page * PAGE + first as usize * PART).max(from);
        let b = (page * PAGE + end as usize * PART).min(to);
        if a < b {
            match runs.last_mut() {
                Some((_, last)) if *last == a => *last = b,
                _ => runs.push((a, b)),
            }
        }
    }
}

/// Makes `pages` read-only where not already (in runs), and returns those
/// now guarded or with nothing mapped (taken either way); a page that is
/// mapped but cannot be guarded is left out (it stays changed).
fn protect(g: &Guards, pages: &[usize]) -> Vec<usize> {
    protect_noting(g, pages, &mut Vec::new())
}

/// [`protect`], also listing in `unmapped` the pages taken because nothing
/// is mapped there.
fn protect_noting(g: &Guards, pages: &[usize], unmapped_pages: &mut Vec<usize>) -> Vec<usize> {
    #[cfg(feature = "diagnostics")]
    let counted = |calls: u64, pages: u64| {
        crate::copystat::add(crate::copystat::C::ProtectCalls, calls);
        crate::copystat::add(crate::copystat::C::ProtectPages, pages);
        PROTECT_OWNER.with(|cell| {
            if let Some(owner) = cell.get() {
                crate::copystat::add(owner, calls);
            }
        });
    };
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_READONLY, VirtualProtect, VirtualQuery,
    };
    #[cfg(feature = "diagnostics")]
    let _timed = Timed(std::time::Instant::now(), crate::copystat::C::ProtectUs);
    let set = |from: usize, to: usize| {
        #[cfg(feature = "diagnostics")]
        counted(1, (to - from) as u64);
        #[cfg(test)]
        tests::PROTECTS.with(|n| n.set(n.get() + 1));
        let mut old = 0;
        // SAFETY: title memory; only its protection changes.
        unsafe { VirtualProtect((from * PAGE) as *const _, (to - from) * PAGE, PAGE_READONLY, &mut old) != 0 }
    };
    let unmapped = |page: usize| {
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        let n = unsafe { VirtualQuery((page * PAGE) as *const _, &mut info, size_of_val(&info)) };
        n == 0 || info.State != MEM_COMMIT
    };
    let mut taken = Vec::with_capacity(pages.len());
    let busy = |page: usize| g.busy.iter().any(|&(a, b)| (a..b).contains(&page));
    let pages: Vec<usize> = pages.iter().copied().filter(|&p| !busy(p)).collect();
    // Only the runtime's GPU memory is guarded. A tracked range can start
    // in free or reserved memory and run on into the host's own pages: in
    // #50 that guarded the process heap's header, and a thread holding the
    // heap lock then faulted while another held `Guards` and allocated,
    // a deadlock. Pages left out stay changed, so they are copied whole.
    // Pages guarded already passed this check when they were guarded.
    // With each page, where its region is known to end: `VirtualProtect`
    // takes one view at a time under Wine, so a run that crossed two views
    // (the runtime's page mappings make one per page) failed and was then
    // guarded page by page, one call more than needed.
    let mut allowed = RuntimeRegions::default();
    let (pages, ends): (Vec<usize>, Vec<usize>) = pages
        .into_iter()
        .filter_map(|p| {
            if g.protected.contains(&p) {
                return Some((p, usize::MAX));
            }
            // Only xmem's registered views bypass the OS query. Each
            // view is one block, including where Wine refuses a run.
            if port_guard() && g.alias_of.contains_key(&p) {
                return Some((p, p + 1));
            }
            allowed.region(p).map(|end| (p, end))
        })
        .unzip();
    let mut i = 0;
    while i < pages.len() {
        if g.protected.contains(&pages[i]) {
            taken.push(pages[i]);
            i += 1;
            continue;
        }
        // A run of consecutive pages not guarded yet, in one region.
        let first = i;
        while i + 1 < pages.len()
            && pages[i + 1] == pages[i] + 1
            && pages[i + 1] < ends[first]
            && !g.protected.contains(&pages[i + 1])
        {
            i += 1;
        }
        let run = &pages[first..=i];
        let run_ends = &ends[first..=i];
        i += 1;
        if set(run[0], run[run.len() - 1] + 1) {
            #[cfg(target_arch = "x86_64")]
            // An allocation under the lock can fault and open a page again.
            write_sites::publish(run, run_ends);
            g.protected.insert(run, run_ends);
            taken.extend(run);
            continue;
        }
        for (&page, &end) in run.iter().zip(run_ends) {
            if (!port_guard() || run.len() > 1) && set(page, page + 1) {
                #[cfg(target_arch = "x86_64")]
                write_sites::publish(&[page], std::slice::from_ref(&end));
                g.protected.insert(&[page], &[end]);
                taken.push(page);
            } else if unmapped(page) {
                taken.push(page);
                unmapped_pages.push(page);
            }
        }
    }
    taken
}

/// Which pages are the runtime's GPU memory ([`runtime_memory`]), asked
/// once per memory region: a region shares its state, type and write watch.
#[derive(Default)]
struct RuntimeRegions {
    /// The last region asked about: its pages and whether it is runtime
    /// memory.
    last: Option<(usize, usize, bool)>,
}

impl RuntimeRegions {
    /// If `page` is the runtime's GPU memory, the page its region is known
    /// to end before (`usize::MAX` for a pool, whose end is not asked).
    fn region(&mut self, page: usize) -> Option<usize> {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_IMAGE, MEM_PRIVATE, MEMORY_BASIC_INFORMATION, VirtualQuery,
        };
        if let Some((a, b, yes)) = self.last
            && (a..b).contains(&page)
        {
            return yes.then_some(b);
        }
        // A pool's write watch answers with one lookup. `VirtualQuery` walks
        // Wine's page table to the region's end, under the guard lock that
        // every fault waits for.
        if write_watched(page * PAGE) {
            return Some(usize::MAX);
        }
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        let n = unsafe { VirtualQuery((page * PAGE) as *const _, &mut info, size_of_val(&info)) };
        if n == 0 {
            return None;
        }
        // Not watched: what `runtime_memory` asks of views, from this query.
        let yes = !(info.State == MEM_COMMIT && info.Type == MEM_PRIVATE)
            && !(port_guard() && info.Type == MEM_IMAGE);
        let (base, size) = (info.BaseAddress as usize, info.RegionSize);
        // In our pages, rounded inward so the region never claims a page it
        // only shares.
        let (a, b) = (base.div_ceil(PAGE).min(page), ((base + size) / PAGE).max(page + 1));
        self.last = Some((a, b, yes));
        yes.then_some(b)
    }
}

/// Pages `[first, last)` changed, and so the pages showing the same memory
/// elsewhere: gives write access back where we guarded them and marks them
/// in every tracker holding them. True if one of `[first, last)` was
/// guarded.
fn changed(g: &Guards, first: usize, last: usize, save: bool) -> bool {
    changed_with(g, first, last, save, DirtyPage::WHOLE)
}

fn changed_with(g: &Guards, first: usize, last: usize, save: bool, dirty: DirtyPage) -> bool {
    let guarded = g.protected.any(first, last);
    let others: Vec<usize> = g.alias_of.range(first..last).flat_map(|(_, o)| o.iter().copied()).collect();
    #[cfg(feature = "diagnostics")]
    {
        let held = others.iter().filter(|p| g.holders.contains_key(p)).count();
        crate::copystat::add(crate::copystat::C::AliasMarks, held as u64);
    }
    // Every page's bytes are saved before any is opened: once a page is
    // open, a write by another thread lands without a fault, and a save
    // after it would keep that write as the page's old bytes.
    mark_changed_with(g, first, last, save, dirty);
    // The pages showing the same memory are marked, not opened: nothing
    // writes through them now, and opening them only made the next take
    // guard them again, one Wine call each (two thirds of all protection
    // changes in #50's world).
    for page in others {
        mark_changed(g, page, page + 1, save);
    }
    open_guarded(g, first, last);
    guarded
}

/// Gives write access back to the pages of `[first, last)` we guarded.
fn open_guarded(g: &Guards, first: usize, last: usize) {
    if first.saturating_add(1) == last {
        open(g, first);
        return;
    }
    let guarded = g.protected.range(first, last);
    // In runs: one protection change for consecutive pages (a run across
    // two views fails, and goes page by page).
    let mut i = 0;
    while i < guarded.len() {
        let mut j = i + 1;
        while j < guarded.len() && guarded[j] == guarded[j - 1] + 1 {
            j += 1;
        }
        open_run(g, guarded[i], guarded[j - 1] + 1);
        i = j;
    }
}

/// Marks pages `[first, last)` changed in every tracker holding them,
/// leaving their protection as it is. With `save`, before the write lands:
/// trackers armed for a page they had taken keep its bytes.
fn mark_changed(g: &Guards, first: usize, last: usize, save: bool) {
    mark_changed_with(g, first, last, save, DirtyPage::WHOLE);
}

fn mark_changed_with(g: &Guards, first: usize, last: usize, save: bool, dirty: DirtyPage) {
    for (&page, holders) in g.holders.range(first..last) {
        for c in holders {
            let first_write = c.0.update(page, c.2, |map| match map.entry(page) {
                std::collections::btree_map::Entry::Occupied(mut entry) => {
                    entry.get_mut().merge(dirty);
                    false
                }
                std::collections::btree_map::Entry::Vacant(entry) => {
                    entry.insert(dirty);
                    true
                }
            });
            if first_write && save {
                c.save(page);
            }
        }
        #[cfg(feature = "diagnostics")]
        g.written
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(page);
    }
}

/// At each presented frame (diagnostics): counts the tracked pages written
/// in it, and those written the frame before too.
#[cfg(feature = "diagnostics")]
pub(crate) fn frame_done() {
    let mut g = guards();
    let written = std::mem::take(g.written.get_mut().unwrap_or_else(|e| e.into_inner()));
    let again = written.intersection(&g.written_before).count();
    crate::copystat::add(crate::copystat::C::WrittenPages, written.len() as u64);
    crate::copystat::add(crate::copystat::C::WrittenAgain, again as u64);
    g.written_before = written;
}

/// A writable alias (including a kernel or section writer in progress)
/// prevents taking a clean copy of any of the views.
fn aliases_guarded(g: &Guards, page: usize) -> bool {
    let busy = |p| g.busy.iter().any(|&(a, b)| (a..b).contains(&p));
    !busy(page)
        && g.alias_of
            .get(&page)
            .is_none_or(|others| others.iter().all(|p| g.protected.contains(p) && !busy(*p)))
}

/// A direct write opened the destination. Guard it again and take any
/// CPU bytes written while it was open before clearing the origin's dirt.
fn rearm_direct(g: &Guards, page: usize, at: usize, host: &mut [u8]) -> bool {
    protect(g, &[page]);
    g.protected.contains(&page)
        && aliases_guarded(g, page)
        && crate::host::read_into(at as *const u8, host)
}

pub(crate) struct LandPage<'a> {
    pub(crate) at: usize,
    pub(crate) host: &'a mut [u8],
    pub(crate) old: Option<&'a [u8]>,
    pub(crate) tracked: Option<&'a Tracked>,
    pub(crate) written: Option<usize>,
}

pub(crate) fn land(
    at: usize,
    host: &mut [u8],
    old: Option<&[u8]>,
    tracked: Option<&Tracked>,
    scratch: &mut [u8],
) -> Option<usize> {
    let mut page = LandPage {
        at,
        host,
        old,
        tracked,
        written: None,
    };
    land_batch(std::slice::from_mut(&mut page), scratch);
    page.written
}

pub(crate) const LAND_BATCH: usize = CLAIM_BATCH;

/// Land consecutive pages in order under one pin per stable mapping span.
/// Explicit baselines can reject unchanged pages before looking up the map.
pub(crate) fn land_batch(pages: &mut [LandPage<'_>], scratch: &mut [u8]) {
    use crate::land::LandingPage;
    assert!(pages.len() <= LAND_BATCH);
    for p in &mut *pages {
        debug_assert!(!p.host.is_empty() && p.at % PAGE + p.host.len() <= PAGE);
        debug_assert!(p.old.is_some() || p.tracked.is_some());
        if p.old == Some(p.host) {
            p.written = Some(0);
        }
    }
    let mut runs = Vec::new();
    let mut pages = pages;
    while let Some(first) = pages.first() {
        if first.written.is_some() {
            pages = &mut pages[1..];
            continue;
        }
        let mut count = 1;
        let mut end = first.at + first.host.len();
        for p in &pages[1..] {
            if p.at != end || p.written.is_some() {
                break;
            }
            end += p.host.len();
            count += 1;
        }
        let (mut run, rest) = pages.split_at_mut(count);
        while let Some(first) = run.first() {
            let at = first.at;
            let mut consumed = 0;
            crate::land::prefix(at, end - at, |destination, len| {
                consumed = run.partition_point(|p| p.at < at + len);
                let group = &mut run[..consumed];
                let mut apply = |g: &Guards, group: &mut [LandPage<'_>], runs: &mut Runs| {
                    for p in group {
                        let destination = match destination {
                            LandingPage::Beside(addr) => LandingPage::Beside(addr + p.at - at),
                            LandingPage::Direct => LandingPage::Direct,
                            LandingPage::Replaced => LandingPage::Replaced,
                        };
                        land_claimed(g, p, destination, scratch, runs);
                    }
                };
                if port_guard() {
                    let g = guards_shared();
                    let mut blocks = [0; LAND_BATCH];
                    for (block, p) in blocks.iter_mut().zip(group.iter()) {
                        *block = p.at / PAGE;
                    }
                    let _claims = claim_blocks(&g, &blocks[..consumed]);
                    apply(&g, group, &mut runs);
                } else {
                    let g = guards();
                    apply(&g, group, &mut runs);
                }
            });
            assert_ne!(consumed, 0);
            run = &mut run[consumed..];
        }
        pages = rest;
    }
}

/// Collect exact stores and merge the current title bytes into the shadow.
/// A whole equal chunk needs no per-byte work. Runs may touch across chunks,
/// but a gap can hold a concurrent CPU write through an already open alias.
fn merge_landing(host: &mut [u8], title: &[u8], old: Option<&[u8]>, runs: &mut Runs) {
    const CHUNK: usize = 32;
    runs.clear();
    if host == title {
        return;
    }
    let (host, tail) = host.as_chunks_mut::<CHUNK>();
    let (title, title_tail) = title.as_chunks::<CHUNK>();
    let before = old.map(|slice| slice.as_chunks::<CHUNK>());
    for (i, (gpu, title)) in host.iter_mut().zip(title).enumerate() {
        if gpu == title {
            continue;
        }
        let old = before.map(|(chunks, _)| &chunks[i]);
        if old == Some(gpu) {
            *gpu = *title;
            continue;
        }
        merge_landing_chunk(i * CHUNK, gpu, title, old.map(|b| b.as_slice()), runs);
    }
    merge_landing_chunk(
        host.len() * CHUNK,
        tail,
        title_tail,
        before.map(|(_, tail)| tail),
        runs,
    );
}

fn merge_landing_chunk(
    base: usize,
    host: &mut [u8],
    title: &[u8],
    old: Option<&[u8]>,
    runs: &mut Runs,
) {
    for (i, (gpu, &current)) in host.iter_mut().zip(title).enumerate() {
        if *gpu == current || old.is_some_and(|before| *gpu == before[i]) {
            *gpu = current;
            continue;
        }
        let at = base + i;
        match runs.last_mut() {
            Some((_, end)) if *end == at => *end += 1,
            _ => runs.push((at, at + 1)),
        }
    }
}

/// The physical mapping is pinned, and faults cannot change any alias's
/// protection or dirty state until the guard lock or claims are released.
fn land_claimed(
    g: &Guards,
    entry: &mut LandPage<'_>,
    destination: crate::land::LandingPage,
    scratch: &mut [u8],
    runs: &mut Runs,
) {
    let LandPage {
        at,
        host,
        old,
        tracked,
        written,
    } = entry;
    let at = *at;
    let page = at / PAGE;
    let saved = if old.is_none()
        && let Some(t) = tracked
    {
        t.unarm(page);
        let saved = t.saved(page);
        if saved.is_none() && t.changed(page) {
            return;
        }
        saved
    } else {
        None
    };
    let beside = match destination {
        crate::land::LandingPage::Beside(addr) => Some(addr),
        crate::land::LandingPage::Direct => None,
        crate::land::LandingPage::Replaced => return,
    };
    let old = old.or_else(|| {
        saved
            .as_deref()
            .map(|s| &s[at % PAGE..at % PAGE + host.len()])
    });
    let guarded = g.protected.contains(&page) && aliases_guarded(g, page);
    let title = &mut scratch[..host.len()];
    if let Some(addr) = beside.filter(|_| guarded) {
        // SAFETY: xmem holds the mapping stable and all title aliases
        // are guarded. Section writers are excluded by xmem as well.
        unsafe {
            std::ptr::copy_nonoverlapping(addr as *const u8, title.as_mut_ptr(), title.len())
        };
    } else if !crate::host::read_into(at as *const u8, title) {
        return;
    }
    merge_landing(host, title, old, runs);
    let writes = !runs.is_empty();
    if writes {
        if beside.is_some() {
            // Save other armed shadows before any GPU byte is visible.
            let others = g.alias_of.get(&page).into_iter().flatten().copied();
            for p in std::iter::once(page).chain(others) {
                mark_changed(g, p, p + 1, true);
            }
        } else {
            changed(g, page, page + 1, true);
        }
        for &(first, end) in runs.iter() {
            let src = host.as_ptr().wrapping_add(first);
            if let Some(addr) = beside {
                // SAFETY: the writable view is stable through the lookup
                // callback and the store is confined to the changed run.
                unsafe {
                    std::ptr::copy_nonoverlapping(src, (addr + first) as *mut u8, end - first)
                };
            } else {
                crate::host::write_mapped((at + first) as *mut u8, src, end - first);
            }
        }
    }

    if guarded
        && crate::land::echo()
        && let Some(t) = tracked
        && at == (page * PAGE).max(t.start)
        && at + host.len() == ((page + 1) * PAGE).min(t.end)
        && (beside.is_some() || !writes || rearm_direct(g, page, at, host))
    {
        t.changed.clear_page(page);
    }
    *written = Some(if writes { host.len() } else { 0 });
}

/// A write into title memory the guards do not see, under way: its pages
/// stay open (a tracker taking them meanwhile would guard them, and the
/// kernel's write would fail) until it is dropped, which marks them changed
/// once more.
#[must_use = "the write must happen while this lives"]
pub struct Writing {
    first: usize,
    last: usize,
    beside: bool,
}

impl Drop for Writing {
    fn drop(&mut self) {
        let mut g = guards();
        if let Some(i) = g.busy.iter().position(|&r| r == (self.first, self.last)) {
            g.busy.swap_remove(i);
        }
        if self.beside {
            mark_changed(&g, self.first, self.last, false);
        } else {
            changed(&g, self.first, self.last, false);
        }
    }
}

/// Before our own write into title memory `[at, at + len)` through the
/// kernel (which does not fault into our handler) or another write the
/// guards do not see: opens its guarded blocks, marks them changed and
/// keeps them open while the result lives. The runtime calls it (as
/// `kernel_will_write`) around the title's kernel reads into its own
/// memory and DirectStorage's writes to addresses.
pub fn writing(at: usize, len: usize) -> Writing {
    // The length is the title's (`fread`'s size times count, saturated).
    let (first, last) = (at / PAGE, at.saturating_add(len).div_ceil(PAGE));
    let mut g = guards();
    g.busy.push((first, last));
    changed(&g, first, last, true);
    Writing {
        first,
        last,
        beside: false,
    }
}

/// The runtime writes through a writable view of a section. It reports
/// every title alias, including views added while the write is in progress.
pub fn writing_beside(at: usize, len: usize) -> Writing {
    let (first, last) = (at / PAGE, at.saturating_add(len).div_ceil(PAGE));
    let mut g = guards();
    g.busy.push((first, last));
    mark_changed(&g, first, last, true);
    Writing {
        first,
        last,
        beside: true,
    }
}

/// The runtime is about to split a view. Keep the baseline for armed
/// copies of the surviving pages, which return at the same addresses.
pub fn title_memory_moving(at: usize, len: usize) {
    let g = guards();
    mark_changed(&g, at / PAGE, at.saturating_add(len).div_ceil(PAGE), true);
}

/// The runtime mapped or unmapped title memory `[at, at + len)`: what it
/// shows is new, and read-write.
pub fn title_memory_remapped(at: usize, len: usize) {
    crate::texback::remapped(at, len);
    title_memory_restored(at, len);
}

/// Blocks texture write-back during a physical view split. The same
/// physical pages at the same title addresses retain their read-back tickets.
#[must_use = "keep the guard until all surviving pages have been restored"]
pub struct TitleMemorySplit(crate::texback::MappingChange);

impl TitleMemorySplit {
    /// Discards tickets where the old mapping was removed or could not be restored.
    pub fn discard(&mut self, at: usize, len: usize) {
        self.0.discard(at, len);
    }
}

pub fn title_memory_splitting(
    at: usize,
    len: usize,
    removed_at: usize,
    removed_len: usize,
) -> TitleMemorySplit {
    let mut split = TitleMemorySplit(crate::texback::mapping_change());
    split.discard(removed_at, removed_len);
    title_memory_restored(at, len);
    split
}

/// Resets watch state while a split rebuilds the same mapping at this address.
pub fn title_memory_restored(at: usize, len: usize) {
    let (first, last) = (at / PAGE, at.saturating_add(len).div_ceil(PAGE));
    let mut g = guards();
    // What it showed is gone already: nothing to keep.
    changed(&g, first, last, false);
    // What it showed is not what it shows now: its links go, both ways.
    let stale: Vec<usize> = g.alias_of.range(first..last).map(|(&p, _)| p).collect();
    for page in stale {
        for other in g.alias_of.remove(&page).unwrap_or_default() {
            if let Some(links) = g.alias_of.get_mut(&other) {
                links.retain(|&p| p != page);
                if links.is_empty() {
                    g.alias_of.remove(&other);
                }
            }
        }
    }
}

/// Reports GPU shader writes into aliased title memory `[at, at + len)` so
/// any other trackers over those pages (such as textures' staging) know they
/// changed and re-upload on next use. Fast-paths out when no tracker or alias
/// covers the range.
pub(crate) fn notify_written(at: usize, len: usize) {
    if len == 0 {
        return;
    }
    let (first, last) = (at / PAGE, at.saturating_add(len).div_ceil(PAGE));
    let g = guards();
    // The GPU wrote through its own mapping. The title mappings can stay
    // read-only and still catch the next CPU write.
    mark_changed(&g, first, last, false);
    #[cfg(feature = "diagnostics")]
    let mut held = 0;
    for &page in g
        .alias_of
        .range(first..last)
        .flat_map(|(_, aliases)| aliases)
    {
        #[cfg(feature = "diagnostics")]
        {
            held += u64::from(g.holders.contains_key(&page));
        }
        mark_changed(&g, page, page + 1, false);
    }
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::AliasMarks, held);
}

/// What shaders wrote into host copies of `[at, at + len)` is dropped:
/// every host buffer copy holding the memory takes it from title memory
/// again at its next refresh (the pages count as changed in each, aliases
/// too). Title memory itself did not change, so its guards stay (opened page
/// by page, they cost ~15 % of the submitting thread at each frame end, run
/// perf15), and other trackers (textures' staging, heaps) are left alone:
/// shaders' writes never reach them (marked, textures over withheld ranges
/// were uploaded again for nothing: 19 % of the uploads, run evict3).
pub(crate) fn retake(at: usize, len: usize) {
    let (first, last) = (at / PAGE, at.saturating_add(len).div_ceil(PAGE));
    let g = guards();
    let others: Vec<usize> = g.alias_of.range(first..last).flat_map(|(_, o)| o.iter().copied()).collect();
    mark_copies(&g, first, last);
    for page in others {
        mark_copies(&g, page, page + 1);
    }
}

/// Pages `[first, last)` count as changed in every host buffer copy's
/// tracker holding them.
fn mark_copies(g: &Guards, first: usize, last: usize) {
    for (&page, holders) in g.holders.range(first..last) {
        for c in holders.iter().filter(|c| c.3) {
            c.0.update(page, c.2, |map| map.insert(page, DirtyPage::WHOLE));
        }
    }
}

/// The runtime mapped the page at `at` where `others` show it already: the
/// pages are linked both ways, and if we guard it there, it is guarded here
/// too (a write through the new mapping must not go unseen by the trackers
/// there).
pub fn title_memory_aliased(at: usize, others: &[usize]) {
    let mut g = guards();
    let page = at / PAGE;
    for &other in others {
        for (x, y) in [(other / PAGE, page), (page, other / PAGE)] {
            let links = g.alias_of.entry(x).or_default();
            if !links.contains(&y) {
                links.push(y);
            }
        }
    }
    // If the new mapping cannot be guarded (a write the guards do not see is
    // under way through it), the page is changed; its bytes now may hold that
    // write already, so nothing is saved of them.
    if others.iter().any(|&o| g.protected.contains(&(o / PAGE))) && protect(&g, &[page]).is_empty()
    {
        changed(&g, page, page + 1, false);
    }
}

/// Whether a page of `[at, at + len)` is shown elsewhere too, as the
/// runtime told when it mapped it (tracked or not): a write here changes
/// the bytes there. Asked at submission, by a batch of refreshes that
/// keeps what a copy into title memory cannot have changed; it takes the
/// guard lock for one lookup and allocates nothing under it.
pub(crate) fn aliased(at: usize, len: usize) -> bool {
    let (first, last) = (at / PAGE, at.saturating_add(len).div_ceil(PAGE));
    guards().alias_of.range(first..last).next().is_some()
}

/// Registers the handler that takes writes to guarded blocks, once, ahead
/// of the crash reporter's.
fn install() {
    use windows_sys::Win32::Foundation::{EXCEPTION_ACCESS_VIOLATION, NTSTATUS};
    use windows_sys::Win32::System::Diagnostics::Debug::{
        AddVectoredExceptionHandler, EXCEPTION_CONTINUE_EXECUTION, EXCEPTION_CONTINUE_SEARCH,
        EXCEPTION_POINTERS,
    };
    unsafe extern "system" fn on_write(info: *mut EXCEPTION_POINTERS) -> i32 {
        // SAFETY: the system passes the exception being dispatched.
        let r = unsafe { &*(*info).ExceptionRecord };
        const WRITE: usize = 1;
        if r.ExceptionCode != EXCEPTION_ACCESS_VIOLATION as NTSTATUS
            || r.NumberParameters < 2
            || r.ExceptionInformation[0] != WRITE
        {
            return EXCEPTION_CONTINUE_SEARCH;
        }
        let at = r.ExceptionInformation[1];
        #[cfg(feature = "diagnostics")]
        let _timed = Timed(std::time::Instant::now(), crate::copystat::C::HandlerUs);
        // Held across both checks: another thread may have opened the block
        // since this one faulted, and a copy may have guarded it again since
        // (then it is ours again, not read-only for some other reason).
        if HOLDING.with(|h| h.get()) {
            if read_only(at) {
                open_under_lock(at / PAGE);
                return EXCEPTION_CONTINUE_EXECUTION;
            }
            return EXCEPTION_CONTINUE_SEARCH;
        }
        let page = at / PAGE;
        let opened = if port_guard() {
            let g = guards_shared();
            let _claims = claim_blocks(&g, &[page]);
            // SAFETY: the system is dispatching this exception.
            unsafe { open_faulted(&g, info, at) }
        } else {
            let g = guards();
            // SAFETY: the system is dispatching this exception.
            unsafe { open_faulted(&g, info, at) }
        };
        if opened {
            EXCEPTION_CONTINUE_EXECUTION
        } else {
            EXCEPTION_CONTINUE_SEARCH
        }
    }

    unsafe fn open_faulted(g: &Guards, info: *mut EXCEPTION_POINTERS, at: usize) -> bool {
        #[cfg(target_arch = "x86_64")]
        // SAFETY: the system is dispatching this exception. The page's
        // protection cannot change under the exclusive lock or claim.
        unsafe {
            write_sites::observe(info, at)
        };
        #[cfg(not(target_arch = "x86_64"))]
        let _ = info;
        let page = at / PAGE;
        let dirty = if g.protected.watched(page) && !g.alias_of.contains_key(&page) {
            DirtyPage::WATCH
        } else {
            DirtyPage::WHOLE
        };
        if changed_with(g, page, page + 1, true, dirty) {
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::GuardFaults, 1);
            return true;
        }
        writable(at)
    }

    static ONCE: std::sync::Once = std::sync::Once::new();
    // SAFETY: the handler only touches our guard state and protections.
    ONCE.call_once(|| unsafe {
        AddVectoredExceptionHandler(1, Some(on_write));
    });
}

/// Whether `at` is committed memory that is read-only, as our guards make
/// it.
fn read_only(at: usize) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_READONLY, VirtualQuery,
    };
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
    n != 0 && info.State == MEM_COMMIT && info.Protect == PAGE_READONLY
}

/// Whether `at` is committed read-write memory now.
fn writable(at: usize) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_READWRITE, VirtualQuery,
    };
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
    n != 0 && info.State == MEM_COMMIT && info.Protect == PAGE_READWRITE
}

/// Adds the time since it was made to a counter when dropped (diagnostics).
#[cfg(feature = "diagnostics")]
pub(crate) struct Timed(pub(crate) std::time::Instant, pub(crate) crate::copystat::C);

#[cfg(feature = "diagnostics")]
impl Drop for Timed {
    fn drop(&mut self) {
        let c = std::mem::replace(&mut self.1, crate::copystat::C::Submissions);
        crate::copystat::add(c, self.0.elapsed().as_micros() as u64);
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    #[test]
    fn landing_merge_matches_exact_byte_stores() {
        let mut runs = Vec::new();
        for len in [1, 31, 32, 33, 64, 65, PAGE] {
            let old = vec![7; len];
            let mut title = old.clone();
            let mut gpu = old.clone();
            for i in (0..len).step_by(5) {
                title[i] = 9;
            }
            for i in (0..len).filter(|i| i % 29 < 6) {
                gpu[i] = if i % 2 == 0 { 9 } else { 11 };
            }
            for old in [None, Some(old.as_slice())] {
                let mut merged = gpu.clone();
                merge_landing(&mut merged, &title, old, &mut runs);
                let mut written = title.clone();
                for &(start, end) in &runs {
                    assert!((start..end).all(|i| title[i] != gpu[i]));
                    written[start..end].copy_from_slice(&gpu[start..end]);
                }
                assert!(runs.windows(2).all(|pair| pair[0].1 < pair[1].0));
                for i in 0..len {
                    let expected = if old.is_some_and(|old| old[i] == gpu[i]) {
                        title[i]
                    } else {
                        gpu[i]
                    };
                    assert_eq!((merged[i], written[i]), (expected, expected));
                }
            }
        }
    }

    #[test]
    fn landing_claims_hold_alias_faults_and_allow_unrelated_faults() {
        if !fault_tracking() || !port_guard() {
            return;
        }
        use std::sync::mpsc::{RecvTimeoutError, channel};
        use std::time::{Duration, Instant};
        use windows_sys::Win32::System::Memory::{MEMORY_MAPPED_VIEW_ADDRESS, UnmapViewOfFile};

        let (a, b) = twin_views(1);
        let pool_base = pool(3);
        let alias_stripes = [a / PAGE % CLAIM_COUNT, b / PAGE % CLAIM_COUNT];
        let other = (0..3)
            .map(|i| pool_base + i * PAGE)
            .find(|p| !alias_stripes.contains(&(p / PAGE % CLAIM_COUNT)))
            .unwrap();
        let t = track_copy(a, PAGE, true).unwrap();
        let u = track_copy(b, PAGE, true).unwrap();
        let unrelated = track(other, PAGE, false).unwrap();
        assert_eq!(t.take(a, a + PAGE), [(a, a + PAGE)]);
        assert_eq!(u.take(b, b + PAGE), [(b, b + PAGE)]);
        assert_eq!(unrelated.take(other, other + PAGE), [(other, other + PAGE)]);
        t.arm(a, a + PAGE);
        let write = |at, byte| {
            // SAFETY: the test calls this only for its own live title views.
            unsafe { (at as *mut u8).write_volatile(byte) };
        };
        write(a + 17, 4);
        write(a + 31, 5);
        assert!(!writable(b));

        let limit = Duration::from_secs(5);
        std::thread::scope(|scope| {
            let saved = t.changed.1.lock().unwrap();
            let landing = scope.spawn(|| {
                let mut gpu = vec![0; PAGE];
                gpu[31..33].copy_from_slice(&[7, 8]);
                let bytes = land(a, &mut gpu, None, Some(&t), &mut vec![0; PAGE]);
                (bytes, gpu)
            });
            // Landing is paused at the saved baseline, after claiming the
            // aliases. A fault in a different stripe can still complete.
            let stripe = &CLAIMS[alias_stripes[1]];
            let deadline = Instant::now() + limit;
            let claimed = loop {
                if stripe.try_lock().is_err() {
                    break true;
                }
                if Instant::now() >= deadline {
                    break false;
                }
                std::thread::sleep(Duration::from_millis(1));
            };
            let (attempted, attempt) = channel();
            let (done, finished) = channel();
            scope.spawn(move || {
                attempted.send(()).unwrap();
                write(b + 34, 6);
                done.send(()).unwrap();
            });
            let (opened, open) = channel();
            scope.spawn(move || {
                write(other, 1);
                opened.send(()).unwrap();
            });
            let attempted = attempt.recv_timeout(limit);
            let independent = open.recv_timeout(limit);
            let blocked = finished.recv_timeout(Duration::from_millis(20));
            drop(saved);
            let (bytes, gpu) = landing.join().unwrap();
            let finished = finished.recv_timeout(limit);
            assert!(claimed);
            assert_eq!(attempted, Ok(()));
            assert_eq!(independent, Ok(()));
            assert_eq!(blocked, Err(RecvTimeoutError::Timeout));
            assert_eq!(finished, Ok(()));
            assert_eq!(bytes, Some(PAGE));
            assert_eq!((gpu[17], &gpu[31..35]), (4, &[7, 8, 0, 0][..]));
        });
        for view in [a, b] {
            // SAFETY: both mapped views are live and the writers have joined.
            let bytes = unsafe { std::slice::from_raw_parts(view as *const u8, PAGE) };
            assert_eq!((bytes[17], &bytes[31..35]), (4, &[7, 8, 0, 6][..]));
        }
        assert!(!t.clean() && !u.clean() && !unrelated.clean());
        drop((t, u, unrelated));
        for view in [a, b] {
            title_memory_remapped(view, PAGE);
            let address = MEMORY_MAPPED_VIEW_ADDRESS {
                Value: view as *mut _,
            };
            // SAFETY: the fixture owns the view and all writers are done.
            assert_ne!(unsafe { UnmapViewOfFile(address) }, 0);
        }
        free(pool_base);
    }

    #[test]
    fn landing_equal_baseline_needs_no_guard() {
        let guard = guards();
        let (send, recv) = std::sync::mpsc::channel();
        let worker = std::thread::spawn(move || {
            let mut gpu = [17; 512];
            let written = land(PAGE, &mut gpu, Some(&[17; 512]), None, &mut []);
            send.send((written, PROTECTS.with(std::cell::Cell::get)))
                .unwrap();
        });
        let result = recv.recv_timeout(std::time::Duration::from_secs(3));
        drop(guard);
        worker.join().unwrap();
        assert_eq!(result, Ok((Some(0), 0)));
    }

    #[test]
    fn a_fault_under_the_guard_lock_does_not_hang() {
        use windows_sys::Win32::System::Memory::{PAGE_READONLY, VirtualProtect};
        install();
        // SAFETY: a fresh block of our own.
        let block = unsafe { VirtualAlloc(std::ptr::null(), PAGE, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) }.cast::<u8>();
        assert!(!block.is_null());
        let mut old = 0;
        // SAFETY: as above.
        assert_ne!(unsafe { VirtualProtect(block.cast(), PAGE, PAGE_READONLY, &mut old) }, 0);
        {
            let _g = guards();
            // SAFETY: inside the block; the handler gives write access back.
            unsafe { block.add(100).write_volatile(7) };
        }
        // SAFETY: as above.
        assert_eq!(unsafe { block.add(100).read_volatile() }, 7);
        assert!(writable(block as usize));
        drop(guards());
        assert!(OPENED_UNDER_LOCK.iter().all(|s| s.load(std::sync::atomic::Ordering::Relaxed) != block as usize / PAGE));
        // SAFETY: allocated above.
        assert_ne!(unsafe { VirtualFree(block.cast(), 0, MEM_RELEASE) }, 0);
    }

    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
    };
    pub(crate) const MEM_WRITE_WATCH: u32 = 0x20_0000;

    #[test]
    fn fault_tracking_follows_the_variable() {
        assert!(decide(Some("1"), false));
        assert!(!decide(Some("0"), true));
        assert!(decide(None, true));
        assert!(!decide(None, false));
        assert!(!decide(Some("yes"), false));
        // Tests keep tracking on, under Rosetta too.
        assert!(fault_tracking() || std::env::var("VICTORY_FAULT_TRACKING").as_deref() == Ok("0"));
    }

    #[test]
    fn tracking_is_opt_in_under_rosetta() {
        // The variable decides when it is set; otherwise tracking is on
        // except under Rosetta (`fault_tracking`'s default is `!rosetta()`).
        assert!(decide(Some("1"), false));
        assert!(!decide(Some("0"), true));
        assert!(decide(None, true));
        assert!(!decide(None, false));
        assert!(!decide(Some("2"), false));
        // Stores are patched exactly when tracking runs under Rosetta.
        assert_eq!(stores_need_patching(), fault_tracking() && rosetta());
    }

    /// A write-watched allocation of `pages` pages (as the runtime's pools),
    /// on a 64 KiB boundary (the allocation granularity).
    pub(crate) fn pool(pages: usize) -> usize {
        // SAFETY: a fresh allocation, freed by the test.
        let p = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                pages * PAGE,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        } as usize;
        assert!(p != 0 && p.is_multiple_of(PAGE));
        p
    }

    /// Frees a test's allocation at `p`, unless the guards still guard or
    /// hold a page of it: then it stays allocated, as the other tests'
    /// pools do.
    ///
    /// A copy that a batch, a page table or a widened view still holds
    /// keeps its tracker after the test drops its own handle. Freed under
    /// it, the pages stayed in `protected`, and a test running at the same
    /// time that was given the same address got them unguarded:
    /// [`protect`] skips a page it has guarded already, so the new
    /// allocation stayed writable and its writes went unseen (#31: a guard
    /// test failed in 7 of 150 full runs on an M5 Pro, whichever drew the
    /// address; 2 of 300 with this). The title's own memory does not go
    /// this way: the runtime tells the guards when it unmaps it
    /// ([`title_memory_remapped`]).
    pub(crate) fn free(p: usize) {
        if still_watched(p) {
            return;
        }
        // SAFETY: an allocation of ours.
        unsafe { VirtualFree(p as *mut _, 0, MEM_RELEASE) };
    }

    /// Wine can report all constituent 4 KiB parts of a larger host page.
    /// Inspect the kernel answer without consuming it.
    pub(crate) fn expected_write_watch(start: usize) -> Runs {
        let page = start / PAGE;
        let parts = if page_parts() {
            watched_parts(page, 0)
        } else {
            WHOLE
        };
        let mut runs = Vec::new();
        append_parts(&mut runs, page, parts, start, start + PAGE);
        runs
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn write_site_lookup_is_limited_to_closed_write_watched_guards() {
        observe_write_sites(|_, _| {});
        let base = pool(2);
        {
            let page = base / PAGE;
            let (slots, _) = write_site_guards();
            let first = &slots[page % slots.len()];
            let second = &slots[(page + 1) % slots.len()];
            let g = guards();
            assert_eq!(protect(&g, &[page, page + 1]), [page, page + 1]);
            assert_eq!(first.load(Ordering::Relaxed), base);
            assert_eq!(second.load(Ordering::Relaxed), base + PAGE);
            open_run(&g, page, page + 2);
            assert_eq!(first.load(Ordering::Relaxed), 0);
            assert_eq!(second.load(Ordering::Relaxed), 0);
        }
        free(base);
    }

    /// Whether the guards guard or hold a page of the allocation at `p`.
    fn still_watched(p: usize) -> bool {
        use windows_sys::Win32::System::Memory::{MEMORY_BASIC_INFORMATION, VirtualQuery};
        // The allocation's length: its regions, while they start from `p`
        // (guarded and open pages are regions of their own).
        let mut len = 0;
        loop {
            // SAFETY: plain data filled in by the query.
            let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
            // SAFETY: a valid out pointer of the right size.
            let n = unsafe { VirtualQuery((p + len) as *const _, &mut info, size_of_val(&info)) };
            if n == 0 || info.AllocationBase as usize != p || info.RegionSize == 0 {
                break;
            }
            len += info.RegionSize;
        }
        let (first, last) = (p / PAGE, (p + len).div_ceil(PAGE));
        let g = guards();
        g.protected.any(first, last) || g.holders.range(first..last).next().is_some()
    }

    /// Everything is changed at first; after taking, only what is written
    /// (by faults, or reported) is, and every tracker of a page sees it.
    /// A tracked range that starts in reserved memory and runs into plain
    /// committed memory (the host's, as the process heap was in #50) never
    /// makes that memory read-only; it is left out of the tracker instead
    /// (copied whole at each use, so the tracker is never clean).
    #[test]
    fn plain_committed_memory_is_never_guarded() {
        if !fault_tracking() {
            return;
        }
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, MEMORY_BASIC_INFORMATION, PAGE_READWRITE, VirtualAlloc,
            VirtualFree, VirtualQuery,
        };
        // SAFETY: plain reservation and commit of our own memory.
        let base = unsafe { VirtualAlloc(std::ptr::null(), 4 * PAGE, MEM_RESERVE, PAGE_READWRITE) } as usize;
        assert_ne!(base, 0);
        // The last two blocks committed as plain private memory.
        // SAFETY: inside our reservation.
        assert!(!unsafe { VirtualAlloc((base + 2 * PAGE) as *const _, 2 * PAGE, MEM_COMMIT, PAGE_READWRITE) }.is_null());
        let t = track(base, 4 * PAGE, true).expect("reserved memory counts as runtime memory");
        let _ = t.take(base, base + 4 * PAGE);
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        unsafe { VirtualQuery((base + 2 * PAGE) as *const _, &mut info, size_of_val(&info)) };
        assert_eq!(info.Protect, PAGE_READWRITE, "still writable");
        assert!(!t.changed(base / PAGE + 2) && !t.changed(base / PAGE + 3), "not tracked");
        assert_eq!(t.untracked(base, base + 4 * PAGE), [(base + 2 * PAGE, base + 4 * PAGE)], "copied whole");
        assert!(!t.clean(), "never clean");
        // SAFETY: writable, so no fault.
        unsafe { ((base + 2 * PAGE + 8) as *mut u8).write_volatile(1) };
        drop(t);
        // SAFETY: our reservation.
        unsafe { VirtualFree(base as *mut _, 0, MEM_RELEASE) };
    }

    #[test]
    fn writes_mark_every_tracker_of_their_page() {
        if !fault_tracking() {
            return;
        }
        let p = pool(4);
        let a = track(p, 4 * PAGE, true).expect("pool memory is tracked");
        let b = track(p + PAGE, 2 * PAGE, true).expect("tracked");
        assert!(!a.clean() && !b.clean(), "changed at first");
        assert_eq!(a.take(p, p + 4 * PAGE), [(p, p + 4 * PAGE)]);
        assert!(a.clean() && !b.clean());
        assert_eq!(b.take(p, p + 4 * PAGE), [(p + PAGE, p + 3 * PAGE)]);
        assert!(a.take(p, p + 4 * PAGE).is_empty() && b.take(p, p + 4 * PAGE).is_empty());
        assert!(a.clean() && b.clean());
        // SAFETY: committed pages of ours, now read-only: the handler opens
        // the block; the second write needs no fault.
        unsafe {
            ((p + PAGE + 8) as *mut u8).write_volatile(1);
            ((p + PAGE + 9) as *mut u8).write_volatile(2);
        }
        assert!(!a.clean() && !b.clean(), "a write is seen by every tracker");
        let expected = expected_write_watch(p + PAGE);
        assert_eq!(a.take(p, p + 4 * PAGE), expected);
        assert!(a.clean());
        // `a` guarded it again; `b` still has it changed.
        assert!(b.changed(p / PAGE + 1));
        assert_eq!(b.take(p, p + 4 * PAGE), expected);
        // SAFETY: as above: a fault again, for both.
        unsafe { ((p + PAGE + 10) as *mut u8).write_volatile(3) };
        assert!(a.changed(p / PAGE + 1) && b.changed(p / PAGE + 1));
        let expected = expected_write_watch(p + PAGE);
        assert_eq!(a.take(p, p + 4 * PAGE), expected);
        drop(writing(p + 3 * PAGE + 5, 1));
        assert_eq!(a.take(p, p + 4 * PAGE), [(p + 3 * PAGE, p + 4 * PAGE)]);
        title_memory_remapped(p, PAGE);
        assert!(a.changed(p / PAGE) && !a.changed(p / PAGE + 1));
        drop((a, b));
        free(p);
    }

    #[test]
    fn first_dirty_refresh_can_include_initialization_writes() {
        if !fault_tracking() {
            return;
        }
        let p = pool(1);
        // SAFETY: the whole fresh pool, before tracking or guard setup.
        unsafe { (p as *mut u8).write_bytes(1, PAGE) };
        let t = track(p, PAGE, true).expect("write-watched pool");
        assert_eq!(t.take(p, p + PAGE), [(p, p + PAGE)]);
        let refresh = |value| {
            // SAFETY: this byte belongs to the fixture. The handler opens
            // the guard before the store completes.
            unsafe { ((p + 8) as *mut u8).write_volatile(value) };
            let expected = expected_write_watch(p);
            let ranges = t.take(p, p + PAGE);
            assert_eq!(ranges, expected);
            assert!(ranges.iter().any(|&(a, b)| (a..b).contains(&(p + 8))));
            ranges.iter().map(|(a, b)| b - a).sum::<usize>()
        };
        let first = refresh(2);
        let second = refresh(3);
        eprintln!("initialized pool: first dirty refresh {first} bytes, second {second} bytes");
        drop(t);
        free(p);
    }

    #[test]
    fn racing_takes_keep_each_watch_answer_for_slower_trackers() {
        if !fault_tracking() {
            return;
        }
        let p = pool(1);
        let [a, b, slow] =
            std::array::from_fn(|_| track(p, PAGE, true).expect("pool memory is tracked"));
        for tracker in [&a, &b, &slow] {
            assert_eq!(tracker.take(p, p + PAGE), [(p, p + PAGE)]);
        }
        let write = |offset| {
            // SAFETY: a byte of our live allocation; the handler opens the guard.
            unsafe { ((p + offset) as *mut u8).write_volatile(1) };
        };
        let contains =
            |runs: &Runs, offset| runs.iter().any(|&(a, b)| (a..b).contains(&(p + offset)));
        write(8);
        write(6 * PART + 8); // A second store after the 64 KiB guard is open.
        assert_eq!(a.take(p, p + PART), [(p, p + PART)]);
        assert!(!a.clean(), "the other parts are still due");
        let expected = expected_write_watch(p);
        assert!(contains(&expected, 8) && contains(&expected, 6 * PART + 8));
        let ready = std::sync::Barrier::new(2);
        std::thread::scope(|scope| {
            let taking = scope.spawn(|| {
                ready.wait();
                a.take(p, p + PAGE)
            });
            ready.wait();
            assert_eq!(b.take(p, p + PAGE), expected);
            assert_eq!(taking.join().unwrap(), expected);
        });
        write(15 * PART + 8);
        let latest = expected_write_watch(p);
        assert!(contains(&latest, 15 * PART + 8));
        assert_eq!(a.take(p, p + PAGE), latest);
        let runs = slow.take(p, p + PAGE);
        for offset in [8, 6 * PART + 8, 15 * PART + 8] {
            assert!(
                contains(&runs, offset),
                "missing offset {offset:#x}: {runs:x?}"
            );
        }
        assert!(slow.clean());
        drop((a, b, slow));
        free(p);
    }

    #[test]
    fn disjoint_faults_and_takes_keep_a_shared_trackers_dirty_pages() {
        if !fault_tracking() {
            return;
        }
        let base = pool(4 * SHARD_PAGES);
        let end = base + 4 * SHARD_PAGES * PAGE;
        let tracker = track(base, end - base, true).unwrap();
        assert_eq!(tracker.take(base, end), [(base, end)]);
        let start = std::sync::Barrier::new(4);
        std::thread::scope(|scope| {
            for (i, page) in (base..end).step_by(SHARD_PAGES * PAGE).enumerate() {
                let tracker = &tracker;
                let start = &start;
                scope.spawn(move || {
                    start.wait();
                    for byte in 1..=32u8 {
                        // SAFETY: each worker writes only its own page.
                        unsafe { ((page + i) as *mut u8).write_volatile(byte) };
                        assert!(tracker.changed(page / PAGE));
                        let runs = tracker.take(page, page + PAGE);
                        assert!(runs.iter().any(|&(a, b)| (a..b).contains(&(page + i))));
                        assert!(!tracker.changed(page / PAGE));
                    }
                });
            }
        });
        assert!(tracker.clean());
        drop(tracker);
        free(base);
    }

    #[test]
    fn parts_are_clipped_and_adjacent_parts_share_a_run() {
        let mut runs = Vec::new();
        append_parts(&mut runs, 1, 0x8001, PAGE + 7, 3 * PAGE);
        append_parts(&mut runs, 2, 0b101, PAGE + 7, 2 * PAGE + 2 * PART + 1);
        assert_eq!(
            runs,
            [
                (PAGE + 7, PAGE + PART),
                (2 * PAGE - PART, 2 * PAGE + PART),
                (2 * PAGE + 2 * PART, 2 * PAGE + 2 * PART + 1),
            ]
        );
    }

    /// A range running from free memory over another allocation into a pool
    /// (a 4 MiB window of memory outside buffers) tracks the pool's pages
    /// and leaves the other allocation alone: taking guards the pool only.
    #[test]
    fn a_range_over_foreign_memory_guards_only_the_runtimes() {
        if !fault_tracking() {
            return;
        }
        use windows_sys::Win32::System::Memory::{MEMORY_BASIC_INFORMATION, PAGE_READONLY, VirtualQuery};
        // This test tracks a mostly free address range. In the parallel suite,
        // other tests can map their views in its holes even after both of our
        // allocations succeed. Keep those views out of this address space.
        const CHILD: &str = "VICTORY_TEST_FOREIGN_WINDOW_CHILD";
        if std::env::var_os(CHILD).is_none() {
            let name = "watch::tests::a_range_over_foreign_memory_guards_only_the_runtimes";
            let result = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", name, "--nocapture"])
                .env(CHILD, "1")
                .output()
                .expect("run foreign-window test in a child");
            let stdout = String::from_utf8_lossy(&result.stdout);
            assert!(
                result.status.success() && stdout.contains("1 passed"),
                "foreign-window child {}: {stdout}{}",
                result.status,
                String::from_utf8_lossy(&result.stderr)
            );
            return;
        }
        const WINDOW: usize = 4 << 20;
        // In a free 4 MiB-aligned window: plain committed memory (as the
        // process heap's) 1 MiB in, a pool 2 MiB in.
        let (base, foreign, pool) = (0..8)
            .find_map(|_| {
                // SAFETY: a fresh reservation, released at once; then
                // allocations in the window it left free, freed below.
                unsafe {
                    let p = VirtualAlloc(std::ptr::null(), 2 * WINDOW, MEM_RESERVE, PAGE_READWRITE) as usize;
                    VirtualFree(p as *mut _, 0, MEM_RELEASE);
                    let base = p.next_multiple_of(WINDOW);
                    let at = |offset: usize, kind: u32| {
                        VirtualAlloc((base + offset) as *const _, PAGE, MEM_RESERVE | MEM_COMMIT | kind, PAGE_READWRITE) as usize
                    };
                    let (foreign, pool) = (at(1 << 20, 0), at(2 << 20, MEM_WRITE_WATCH));
                    if p != 0 && foreign == base + (1 << 20) && pool == base + (2 << 20) {
                        return Some((base, foreign, pool));
                    }
                    [foreign, pool].into_iter().filter(|&q| q != 0).for_each(free);
                    None
                }
            })
            .expect("a free window");
        let t = track(base, WINDOW, true).expect("the pool is tracked");
        let taken = t.take(base, base + WINDOW);
        assert!(taken.iter().any(|&(a, b)| a <= pool && pool + PAGE <= b), "{taken:x?}");
        assert!(!taken.iter().any(|&(a, b)| a < foreign + PAGE && foreign < b), "{taken:x?}");
        let protection = |p: usize| {
            // SAFETY: plain data filled in by the query.
            let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
            // SAFETY: a valid out pointer of the right size.
            unsafe { VirtualQuery(p as *const _, &mut info, size_of_val(&info)) };
            info.Protect
        };
        assert_eq!(protection(pool), PAGE_READONLY, "the pool is guarded");
        assert_eq!(protection(foreign), PAGE_READWRITE, "the other allocation is not");
        drop(t);
        free(foreign);
        free(pool);
    }

    /// A tracker's dirty bit goes back when it goes, and the next tracker
    /// starts dirty on it (everything changed at first).
    #[test]
    fn a_new_tracker_starts_dirty_on_a_bit_given_back() {
        if !fault_tracking() {
            return;
        }
        let p = pool(1);
        let a = track(p, PAGE, true).expect("tracked");
        assert_eq!(a.take(p, p + PAGE).len(), 1);
        assert!(a.clean() && clean_bit(a.bit()));
        drop(a);
        let b = track(p, PAGE, true).expect("tracked");
        assert!(!b.clean() && !clean_bit(b.bit()));
        assert!(!clean_bit(None), "no bit: never clean");
        drop(b);
        free(p);
    }

    /// A page with nothing mapped is taken as zeros: `take_split` lists it
    /// apart from the pages to copy.
    #[test]
    fn unmapped_pages_are_taken_apart() {
        if !fault_tracking() {
            return;
        }
        // SAFETY: a fresh write-watched reservation of two pages, the first
        // committed; freed below.
        let p = unsafe {
            let p = VirtualAlloc(std::ptr::null(), 2 * PAGE, MEM_RESERVE | MEM_WRITE_WATCH, PAGE_READWRITE) as usize;
            assert_ne!(p, 0);
            assert_ne!(VirtualAlloc(p as *const _, PAGE, MEM_COMMIT, PAGE_READWRITE) as usize, 0);
            p
        };
        let t = track(p, 2 * PAGE, true).expect("tracked");
        assert_eq!(t.take_split(p, p + 2 * PAGE), (vec![(p, p + PAGE)], vec![(p + PAGE, p + 2 * PAGE)]));
        assert!(t.take(p, p + 2 * PAGE).is_empty(), "both taken");
        drop(t);
        free(p);
    }

    /// A page only partly in the range asked is copied in part and stays
    /// changed; edge pages count whole when the tracked part is covered.
    #[test]
    fn partial_pages_stay_changed() {
        if !fault_tracking() {
            return;
        }
        let p = pool(3);
        let t = track(p + 100, 2 * PAGE, true).expect("tracked");
        assert_eq!(t.take(p + 100, p + PAGE + 10), [(p + 100, p + PAGE + 10)]);
        assert!(!t.changed(p / PAGE) && t.changed(p / PAGE + 1));
        assert!(!t.clean(), "a page taken in part stays changed");
        assert_eq!(t.take(p, p + 3 * PAGE), [(p + PAGE, p + 2 * PAGE + 100)]);
        assert!(t.take(p, p + 3 * PAGE).is_empty());
        assert!(t.clean());
        drop(t);
        free(p);
    }

    /// Pages of the ranges armed keep their bytes from before the first
    /// write after they were taken, whichever overlapping ranges armed
    /// them; a page unarmed keeps none until it is armed again, and none
    /// does once the tracker is disarmed.
    #[test]
    fn armed_ranges_keep_their_pages_bytes_from_before_the_first_write() {
        if !fault_tracking() {
            return;
        }
        let p = pool(4);
        let t = track(p, 4 * PAGE, true).expect("pool memory is tracked");
        let all = || t.take(p, p + 4 * PAGE);
        // SAFETY: (every write below) a byte of our pool; a guarded page
        // faults into the handler, which saves it first where it is armed.
        let write = |page: usize, value: u8| unsafe { ((p + page * PAGE + 8) as *mut u8).write_volatile(value) };
        let saved = |page: usize| t.saved(p / PAGE + page).map(|bytes| bytes[8]);
        all();
        write(0, 1);
        assert_eq!(saved(0), None, "nothing armed yet");
        all();
        // Bytes 100 of page 0 to 1 of page 2: pages 0 to 2, page 3 not.
        t.arm(p + 100, p + 2 * PAGE + 1);
        t.arm(p + PAGE, p + 2 * PAGE);
        t.arm(p + PAGE + 5, p + 2 * PAGE + 1);
        t.unarm(p / PAGE + 1);
        for page in 0..4 {
            write(page, 2);
        }
        assert_eq!([saved(0), saved(1), saved(2), saved(3)], [Some(1), None, Some(0), None]);
        assert_eq!(saved(0), None, "handed out once");
        all();
        // Armed again by a range holding it.
        t.arm(p + PAGE, p + 2 * PAGE);
        write(1, 3);
        write(1, 4);
        assert_eq!(saved(1), Some(2), "from before the first write");
        t.disarm();
        all();
        write(2, 5);
        assert_eq!(saved(2), None, "disarmed");
        drop(t);
        free(p);
    }

    /// Runs armed that touch or overlap merge into one, so a page's check
    /// stays a search however many sub-ranges are armed; pages unarmed in
    /// them stay unarmed until a range holding them is armed again.
    #[test]
    fn adjacent_and_overlapping_armed_runs_coalesce() {
        let mut s = Saved::default();
        // Adjacent: [2k, 2k + 2) end to end, in order and out of it.
        for k in 0..200 {
            s.arm(2 * k, 2 * k + 2);
        }
        assert_eq!(s.armed, [(0, 400)]);
        let mut s = Saved::default();
        for k in (0..200).rev() {
            s.arm(2 * k, 2 * k + 2);
        }
        assert_eq!(s.armed, [(0, 400)]);
        // Overlapping, nested and disjoint.
        let mut s = Saved::default();
        s.arm(10, 20);
        s.arm(15, 25);
        assert_eq!(s.armed, [(10, 25)]);
        s.arm(12, 14);
        assert_eq!(s.armed, [(10, 25)], "nested");
        s.arm(40, 50);
        s.arm(0, 5);
        assert_eq!(s.armed, [(0, 5), (10, 25), (40, 50)], "disjoint, sorted");
        s.arm(5, 10);
        assert_eq!(s.armed, [(0, 25), (40, 50)], "touching both sides");
        s.arm(30, 35);
        s.arm(20, 45);
        assert_eq!(s.armed, [(0, 50)], "spanning several");
        s.arm(60, 70);
        s.arm(55, 55);
        assert_eq!(s.armed, [(0, 50), (60, 70)], "an empty run arms nothing");
        assert!(s.armed(0) && s.armed(49) && !s.armed(50) && !s.armed(59));
        assert!(s.armed(60) && s.armed(69) && !s.armed(70) && !s.armed(1000));
        // Unarmed pages stay out of a merged run until armed again.
        s.unarmed.insert(45);
        s.unarmed.insert(65);
        s.arm(50, 60);
        assert_eq!(s.armed, [(0, 70)]);
        assert!(!s.armed(45) && !s.armed(65) && s.armed(55));
        s.arm(44, 46);
        assert!(s.armed(45) && !s.armed(65));
        s.arm(0, 70);
        assert!(s.armed(65) && s.unarmed.is_empty());
    }

    /// Plain committed memory (the C heap) is not tracked; reserved memory
    /// is, reads as taken, and changes when the runtime maps it.
    #[test]
    fn heap_memory_is_not_tracked_and_placeholders_wait_for_mappings() {
        if !fault_tracking() {
            return;
        }
        let plain = vec![0u8; 2 * PAGE];
        assert!(track(plain.as_ptr() as usize, plain.len(), true).is_none());
        // SAFETY: a reservation of ours, freed below.
        let r = unsafe { VirtualAlloc(std::ptr::null(), 2 * PAGE, MEM_RESERVE, PAGE_READWRITE) } as usize;
        assert_ne!(r, 0);
        let t = track(r, 2 * PAGE, true).expect("reserved memory is tracked");
        assert_eq!(t.take(r, r + 2 * PAGE).len(), 1, "copied once (as zeros)");
        assert!(t.take(r, r + 2 * PAGE).is_empty());
        title_memory_remapped(r, PAGE);
        assert_eq!(t.take(r, r + 2 * PAGE), [(r, r + PAGE)]);
        drop(t);
        // SAFETY: our reservation.
        unsafe { VirtualFree(r as *mut _, 0, MEM_RELEASE) };
    }

    /// A kernel read into a guarded view block fails; opened first, it
    /// lands and the block counts as changed.
    #[test]
    fn kernel_reads_into_guarded_blocks_after_opening() {
        if !fault_tracking() {
            return;
        }
        use std::io::{Read, Seek, Write};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile, UnmapViewOfFile,
        };
        // SAFETY: a pagefile section of two blocks, mapped read-write.
        let view = unsafe {
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (2 * PAGE) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, 2 * PAGE).Value as usize
        };
        assert_ne!(view, 0);
        assert!(track(view, 2 * PAGE, false).is_none(), "views only where asked");
        let t = track(view, 2 * PAGE, true).expect("views are tracked");
        assert_eq!(t.take(view, view + 2 * PAGE), [(view, view + 2 * PAGE)]); // now read-only
        let mut file = tempfile::tempfile().unwrap();
        file.write_all(b"from the disk").unwrap();
        // `ReadFile` inside: the kernel writes the view.
        let read = |file: &mut std::fs::File| {
            file.rewind().unwrap();
            // SAFETY: 13 bytes of our view, nothing else refers to them.
            let into = unsafe { std::slice::from_raw_parts_mut((view + 100) as *mut u8, 13) };
            file.read(into).is_ok_and(|n| n == 13)
        };
        assert!(!read(&mut file), "the kernel does not fault into our handler");
        let write = writing(view + 100, 13);
        // A tracker taking the page meanwhile leaves it open.
        assert_eq!(t.take(view, view + 2 * PAGE), [(view, view + PAGE)]);
        assert!(read(&mut file));
        drop(write);
        // SAFETY: the bytes just read.
        assert_eq!(unsafe { std::slice::from_raw_parts((view + 100) as *const u8, 13) }, b"from the disk");
        assert_eq!(t.take(view, view + 2 * PAGE), [(view, view + PAGE)]);
        drop(t);
        // SAFETY: our view.
        unsafe { UnmapViewOfFile(windows_sys::Win32::System::Memory::MEMORY_MAPPED_VIEW_ADDRESS { Value: view as *mut _ }) };
    }

    /// `pages` pages of a fresh pagefile section mapped twice (as the
    /// title's pool view and tiled heap show the same physical pages),
    /// reported as the runtime does when it maps them.
    pub(crate) fn twin_views(pages: usize) -> (usize, usize) {
        use windows_sys::Win32::System::Memory::{CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile};
        let len = pages * PAGE;
        // SAFETY: a pagefile section mapped read-write twice; the test
        // keeps both views.
        let (a, b) = unsafe {
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                len as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            (
                MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, len).Value as usize,
                MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, len).Value as usize,
            )
        };
        assert!(a != 0 && b != 0);
        for k in (0..len).step_by(PAGE) {
            title_memory_aliased(b + k, &[a + k]);
        }
        (a, b)
    }

    thread_local! {
        /// `VirtualProtect` calls this thread made in `protect`.
        pub(crate) static PROTECTS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
        pub(super) static PART_WATCH_QUERIES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    }

    /// `single` views of one page each side by side, as the runtime's page
    /// mappings make them, then one view of `single` pages: the address of
    /// the first, and of the long one.
    fn views_side_by_side(single: usize) -> (usize, usize) {
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, MEM_PRESERVE_PLACEHOLDER, MEM_REPLACE_PLACEHOLDER,
            MEM_RESERVE_PLACEHOLDER, MapViewOfFile3, PAGE_NOACCESS, VirtualAlloc2,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        let len = 2 * single * PAGE;
        // SAFETY: a pagefile section and a placeholder of our own, split
        // and filled with views of the section; the test keeps them.
        unsafe {
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                len as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            let base = VirtualAlloc2(
                GetCurrentProcess(),
                std::ptr::null(),
                len,
                MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                PAGE_NOACCESS,
                std::ptr::null_mut(),
                0,
            ) as usize;
            assert_ne!(base, 0);
            let map = |at: usize, offset: usize, n: usize| {
                VirtualFree(at as *mut _, n * PAGE, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER);
                let view = MapViewOfFile3(
                    section,
                    GetCurrentProcess(),
                    at as *const _,
                    offset as u64,
                    n * PAGE,
                    MEM_REPLACE_PLACEHOLDER,
                    PAGE_READWRITE,
                    std::ptr::null_mut(),
                    0,
                );
                assert_eq!(view.Value as usize, at);
            };
            for k in 0..single {
                map(base + k * PAGE, k * PAGE, 1);
            }
            let long = base + single * PAGE;
            map(long, single * PAGE, single);
            (base, long)
        }
    }

    #[test]
    fn section_views_refresh_whole_without_collecting_write_watch() {
        if !fault_tracking() {
            return;
        }
        use windows_sys::Win32::System::Memory::{MEMORY_MAPPED_VIEW_ADDRESS, UnmapViewOfFile};
        let (a, b) = twin_views(1);
        title_memory_remapped(b, PAGE);
        let t = track(a, PAGE, true).expect("section view is tracked");
        assert_eq!(t.take(a, a + PAGE), [(a, a + PAGE)]);
        let queries = || PART_WATCH_QUERIES.with(std::cell::Cell::get);
        let initial = queries();
        // SAFETY: a byte of our first view; the guard handler opens it.
        unsafe { ((a + 8) as *mut u8).write_volatile(1) };
        assert_eq!(t.take(a, a + PAGE), [(a, a + PAGE)]);
        assert_eq!(queries(), initial, "the unaliased view has no write watch");
        title_memory_aliased(b, &[a]);
        // SAFETY: a byte of our second, now guarded view.
        unsafe { ((b + 16) as *mut u8).write_volatile(2) };
        assert_eq!(t.take(a, a + PAGE), [(a, a + PAGE)]);
        assert_eq!(queries(), initial, "aliases use whole-page refreshes");
        eprintln!(
            "section refresh: {PAGE} bytes with {} write-watch collections",
            queries() - initial
        );
        drop(t);
        for view in [a, b] {
            title_memory_remapped(view, PAGE);
            // SAFETY: mapped by twin_views and no remaining tracker owns it.
            let unmapped = unsafe {
                UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
                    Value: view as *mut _,
                })
            };
            assert_ne!(unmapped, 0);
        }
    }

    /// Guarding pages of separate views takes one call per view, not a
    /// call over all of them that fails first and then one per page.
    #[test]
    fn takes_guard_one_view_per_call() {
        if !fault_tracking() {
            return;
        }
        let calls = || PROTECTS.with(std::cell::Cell::get);
        // Four one-page views and a four-page view, taken together.
        let (base, _) = views_side_by_side(4);
        let t = track(base, 8 * PAGE, true).expect("views are tracked");
        let before = calls();
        assert_eq!(t.take(base, base + 8 * PAGE), [(base, base + 8 * PAGE)]);
        assert_eq!(calls() - before, 5);
        assert!((0..8).all(|k| read_only(base + k * PAGE)));
        drop(t);
        // Each kind alone, written after: the guards catch it.
        let (base, long) = views_side_by_side(4);
        for (at, want) in [(base, 4), (long, 1)] {
            let t = track(at, 4 * PAGE, true).expect("views are tracked");
            let before = calls();
            assert_eq!(t.take(at, at + 4 * PAGE), [(at, at + 4 * PAGE)]);
            assert_eq!(calls() - before, want, "calls for the views at {at:#x}");
            for k in 0..4 {
                assert!(read_only(at + k * PAGE), "page {k} at {at:#x} guarded");
            }
            // Guarded, they mark the tracker when written.
            // SAFETY: a page of the views; the handler opens it.
            unsafe { ((at + 2 * PAGE + 8) as *mut u8).write_volatile(1) };
            assert!(t.changed(at / PAGE + 2));
            assert!(!t.changed(at / PAGE + 1));
        }
    }

    /// A write through another mapping of the same pages marks the tracker
    /// (the guard there faults, the runtime says where else it is mapped);
    /// so does a write the runtime reports at the other mapping.
    #[test]
    fn writes_through_another_mapping_mark_the_tracker() {
        if !fault_tracking() {
            return;
        }
        let (a, b) = twin_views(2);
        let t = track(a, 2 * PAGE, true).expect("views are tracked");
        assert_eq!(t.take(a, a + 2 * PAGE), [(a, a + 2 * PAGE)]);
        // SAFETY: the second view of the pages, guarded by the take.
        unsafe { ((b + PAGE + 4) as *mut u32).write_volatile(7) };
        assert!(read_only(a + PAGE), "marked, still guarded: no call to guard it again");
        assert_eq!(t.take(a, a + 2 * PAGE), [(a + PAGE, a + 2 * PAGE)]);
        // SAFETY: the tracked view, which shows the write.
        assert_eq!(unsafe { ((a + PAGE + 4) as *const u32).read_volatile() }, 7);
        drop(writing(b, 8));
        assert_eq!(t.take(a, a + 2 * PAGE), [(a, a + PAGE)]);
        assert!(t.take(a, a + 2 * PAGE).is_empty());
        // A third mapping made later starts guarded, so its writes count.
        let (c, _) = twin_views(1);
        title_memory_aliased(c, &[a]);
        // SAFETY: a view of ours (another section: only the tracking
        // takes it for `a`'s pages here).
        unsafe { (c as *mut u32).write_volatile(1) };
        assert_eq!(t.take(a, a + 2 * PAGE), [(a, a + PAGE)]);
    }

    /// Pages are reported aliased while the runtime says they are shown
    /// twice, and not after either mapping changes.
    #[test]
    fn aliased_pages_are_found_by_range() {
        let (a, b) = twin_views(3);
        assert!(
            aliased(a, 1) && aliased(b + 2 * PAGE + 5, 10),
            "either view, any byte"
        );
        assert!(aliased(a + PAGE - 1, 2), "a range that reaches a page");
        assert!(!aliased(a, 0), "no bytes");
        let plain = pool(1);
        assert!(!aliased(plain, PAGE));
        free(plain);
        // One page shown once only again: its link goes, the others stay.
        title_memory_remapped(b + PAGE, PAGE);
        assert!(!aliased(a + PAGE, PAGE) && !aliased(b + PAGE, PAGE));
        assert!(aliased(a, PAGE) && aliased(b + 2 * PAGE, PAGE));
        assert!(aliased(a, usize::MAX), "no overflow");
    }

    #[test]
    fn gpu_notifications_mark_both_mappings_without_rearming_them() {
        if !fault_tracking() {
            return;
        }
        let (a, b) = twin_views(1);
        let copy = track_copy(a, PAGE, true).unwrap();
        let texture = track(b, PAGE, true).unwrap();
        assert_eq!(copy.take(a, a + PAGE), [(a, a + PAGE)]);
        assert_eq!(texture.take(b, b + PAGE), [(b, b + PAGE)]);
        let protects = PROTECTS.with(std::cell::Cell::get);

        notify_written(a + 4, 4);
        assert!(read_only(a) && read_only(b));
        assert_eq!(copy.take(a, a + PAGE), [(a, a + PAGE)]);
        assert_eq!(texture.take(b, b + PAGE), [(b, b + PAGE)]);
        assert_eq!(PROTECTS.with(std::cell::Cell::get), protects);

        // SAFETY: this title store must still fault and dirty the trackers.
        unsafe { ((b + 8) as *mut u32).write_volatile(3) };
        assert_eq!(copy.take(a, a + PAGE), [(a, a + PAGE)]);
        assert_eq!(texture.take(b, b + PAGE), [(b, b + PAGE)]);
    }

    /// A tracker of one mapping going away leaves the guard its page shares
    /// with a tracker of another mapping, so a write there still counts (it
    /// was opened, and the pages it went unseen in stayed clean).
    #[test]
    fn a_tracker_dropped_on_one_mapping_leaves_the_other_guarded() {
        if !fault_tracking() {
            return;
        }
        let (a, b) = twin_views(1);
        let t = track(a, PAGE, true).expect("views are tracked");
        let u = track(b, PAGE, true).expect("views are tracked");
        assert_eq!(t.take(a, a + PAGE), [(a, a + PAGE)]);
        assert_eq!(u.take(b, b + PAGE), [(b, b + PAGE)]);
        drop(u);
        // SAFETY: the second view, guarded for `t`'s page too.
        unsafe { ((b + 4) as *mut u32).write_volatile(7) };
        assert_eq!(t.take(a, a + PAGE), [(a, a + PAGE)]);
    }

    /// A retake has every copy of the memory, and of its other mappings,
    /// take it again, and leaves it guarded: a later write still counts.
    #[test]
    fn a_retake_marks_the_copies_and_keeps_the_guard() {
        if !fault_tracking() {
            return;
        }
        use windows_sys::Win32::System::Memory::{MEMORY_BASIC_INFORMATION, PAGE_READONLY, VirtualQuery};
        let protection = |at: usize| {
            // SAFETY: plain data filled in by the query.
            let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
            // SAFETY: a valid out pointer of the right size.
            unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
            info.Protect
        };
        let (a, b) = twin_views(2);
        let t = track_copy(a, 2 * PAGE, true).expect("views are tracked");
        let u = track_copy(b, 2 * PAGE, true).expect("views are tracked");
        assert_eq!(t.take(a, a + 2 * PAGE), [(a, a + 2 * PAGE)]);
        assert_eq!(u.take(b, b + 2 * PAGE), [(b, b + 2 * PAGE)]);
        assert!(t.clean() && u.clean());
        retake(a + PAGE + 16, 32);
        // Marked in both copies, the second mapping's too, still guarded.
        assert!(!t.clean() && !u.clean());
        assert!(t.changed((a + PAGE) / PAGE) && u.changed((b + PAGE) / PAGE));
        assert!(!t.changed(a / PAGE));
        assert_eq!(protection(a + PAGE), PAGE_READONLY);
        assert_eq!(protection(b + PAGE), PAGE_READONLY);
        // Taken again (the copies re-copy that page), then guarded and clean.
        assert_eq!(t.take(a, a + 2 * PAGE), [(a + PAGE, a + 2 * PAGE)]);
        assert_eq!(u.take(b, b + 2 * PAGE), [(b + PAGE, b + 2 * PAGE)]);
        assert!(t.clean() && u.clean());
        assert_eq!(protection(a + PAGE), PAGE_READONLY);
        // SAFETY: the second view, guarded all along.
        unsafe { ((b + PAGE + 8) as *mut u32).write_volatile(3) };
        assert_eq!(t.take(a, a + 2 * PAGE), [(a + PAGE, a + 2 * PAGE)]);
        assert_eq!(u.take(b, b + 2 * PAGE), [(b + PAGE, b + 2 * PAGE)]);
        // A retake while marked and guarded, then a drop: nothing breaks.
        retake(a, 2 * PAGE);
        drop(u);
        assert_eq!(t.take(a, a + 2 * PAGE), [(a, a + 2 * PAGE)]);
        assert!(t.clean());
    }

    /// A retake marks host buffer copies holding the memory, not other
    /// trackers of it (a texture's staging): shaders' writes reach only the
    /// copies. A title write still marks both.
    #[test]
    fn a_retake_marks_buffer_copies_alone() {
        if !fault_tracking() {
            return;
        }
        let p = pool(2);
        let copy = track_copy(p, 2 * PAGE, true).expect("pool memory is tracked");
        let texture = track(p, 2 * PAGE, true).expect("pool memory is tracked");
        assert_eq!(copy.take(p, p + 2 * PAGE), [(p, p + 2 * PAGE)]);
        assert_eq!(texture.take(p, p + 2 * PAGE), [(p, p + 2 * PAGE)]);
        retake(p + PAGE, 16);
        assert!(copy.changed(p / PAGE + 1), "the copy takes it again");
        assert!(!texture.changed(p / PAGE + 1) && texture.clean(), "the texture's tracker is left alone");
        assert_eq!(copy.take(p, p + 2 * PAGE), [(p + PAGE, p + 2 * PAGE)]);
        // SAFETY: a page of our pool, guarded: the handler opens it.
        unsafe { ((p + PAGE + 8) as *mut u8).write_volatile(1) };
        assert!(copy.changed(p / PAGE + 1) && texture.changed(p / PAGE + 1), "a title write marks both");
    }

    /// A write under way on another mapping of a page keeps the page open
    /// there, so taking the page (or marking it taken) through its tracked
    /// mapping must leave it changed and keep what was saved of it; aliasing
    /// a busy mapping to a guarded one must mark the guarded page changed.
    #[test]
    fn a_busy_alias_leaves_the_page_changed_and_keeps_saved_bytes() {
        if !fault_tracking() {
            return;
        }
        let (a, b) = twin_views(1);
        let t = track(a, PAGE, true).expect("views are tracked");
        assert_eq!(t.take(a, a + PAGE), [(a, a + PAGE)]);
        t.arm(a, a + PAGE);
        let write = writing(b, 16);
        assert!(t.changed(a / PAGE));
        assert_eq!(t.take(a, a + PAGE), [(a, a + PAGE)]);
        assert!(t.changed(a / PAGE), "take must leave the page changed");
        t.taken(&[a / PAGE]);
        assert!(t.changed(a / PAGE), "taken must leave the page changed");
        // SAFETY: the second view of the page, kept open by `write`.
        unsafe { (b as *mut u32).write_volatile(0x1234) };
        assert_eq!(t.take(a, a + PAGE), [(a, a + PAGE)]);
        drop(write);
        let saved = t.saved(a / PAGE).expect("saved bytes survive");
        assert_eq!(&saved[..4], &[0, 0, 0, 0]);
        assert_eq!(t.take(a, a + PAGE), [(a, a + PAGE)]);
        assert!(t.take(a, a + PAGE).is_empty(), "taken once the write ended");
        let (c, _) = twin_views(1);
        let busy_c = writing(c, 16);
        title_memory_aliased(c, &[a]);
        assert!(
            t.changed(a / PAGE),
            "aliasing a busy page marks the tracked one"
        );
        drop(busy_c);
    }

    /// A write or mapping that runs past the end of the address space (a
    /// title's `fread` with a huge count) covers up to the end; the sum used
    /// to overflow. Up there no test has guarded blocks.
    #[test]
    fn ranges_past_the_address_space_end_do_not_overflow() {
        let top = usize::MAX - 16;
        drop(writing(top, PAGE));
        title_memory_remapped(top, usize::MAX);
    }
}
