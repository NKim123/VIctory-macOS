//! Opt-in title pipeline records for offline replay. Shader containers are
//! content-addressed and published before any record refers to them. The
//! records and containers contain title data and live in the user's cache.

use std::collections::HashSet;
use std::ffi::CString;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::PCSTR;

use super::host::GraphicsState;
use super::{Prebuild, SPILL_DIR, Shader, spill_path};

const STAGES: [&str; 8] = [
    "vertex",
    "pixel",
    "domain",
    "hull",
    "geometry",
    "compute",
    "amplification",
    "mesh",
];

#[derive(Default)]
struct Recorded {
    pipelines: HashSet<String>,
    shaders: HashSet<blake3::Hash>,
}

static BOOK: OnceLock<(PathBuf, Mutex<Recorded>)> = OnceLock::new();

pub(super) fn enable(dir: PathBuf) {
    let _ = BOOK.set((dir, Mutex::new(Recorded::default())));
}

pub(super) fn enabled() -> bool {
    BOOK.get().is_some()
}

pub(super) fn valid_key(key: &str) -> bool {
    key.len() == 32
        && key
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

pub(super) fn record(key: &str, shaders: &[Arc<Shader>], graphics: Option<&GraphicsState>) {
    let Some((dir, recorded)) = BOOK.get() else {
        return;
    };
    // Match the kinds the host knows how to build.
    let supported = matches!(shaders, [cs] if cs.stage == "compute")
        || graphics.is_some()
            && shaders.iter().any(|s| matches!(s.stage, "vertex" | "mesh"))
            && shaders
                .iter()
                .all(|s| matches!(s.stage, "vertex" | "mesh" | "pixel"));
    if !supported {
        return;
    }

    let mut recorded = recorded.lock().unwrap_or_else(|e| e.into_inner());
    if !recorded.pipelines.contains(key) {
        match write_record(
            dir,
            &mut recorded.shaders,
            key,
            shaders,
            graphics,
            SPILL_DIR.get().is_some(),
        ) {
            Ok(()) => {
                recorded.pipelines.insert(key.to_owned());
            }
            Err(e) => tracing::warn!(error = %e, key, "pipeline record not saved"),
        }
    }
}

fn shader_path(dir: &Path, hash: blake3::Hash) -> PathBuf {
    dir.join(format!("{hash}.shader"))
}

fn record_path(dir: &Path, key: &str) -> PathBuf {
    dir.join(format!("{key}.pso"))
}

fn invalid(what: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, what)
}

fn write_record(
    dir: &Path,
    known: &mut HashSet<blake3::Hash>,
    key: &str,
    shaders: &[Arc<Shader>],
    graphics: Option<&GraphicsState>,
    use_spill: bool,
) -> io::Result<()> {
    let mut record = vec![
        shaders
            .len()
            .try_into()
            .map_err(|_| invalid("too many stages"))?,
    ];
    for shader in shaders {
        let stage = STAGES
            .iter()
            .position(|&s| s == shader.stage)
            .ok_or_else(|| invalid("unknown stage"))?;
        record.push(stage as u8);
        let hash = blake3::Hash::from(shader.full_hash);
        record.extend(hash.as_bytes());
        if !known.contains(&hash) {
            let bytes = shader.container.lock().unwrap_or_else(|e| e.into_inner());
            // The spill file already has the bytes. Legacy records that
            // have a private copy still load from the same record format.
            if !use_spill || shader.spilled.get().is_none() {
                let path = shader_path(dir, hash);
                let valid = match std::fs::File::open(&path) {
                    Ok(file) => blake3::Hasher::new().update_reader(file)?.finalize() == hash,
                    Err(e) if e.kind() == io::ErrorKind::NotFound => false,
                    Err(e) => return Err(e),
                };
                if !valid {
                    if bytes.is_empty() {
                        return Err(invalid("shader container is no longer available"));
                    }
                    super::library::write_atomically(&path, &[&bytes])?;
                }
            }
            known.insert(hash);
        }
    }
    record.push(u8::from(graphics.is_some()));
    if let Some(g) = graphics {
        g.put(&mut record)
            .ok_or_else(|| invalid("too many input elements or name too long"))?;
    }
    let checksum = blake3::hash(&record);
    super::library::write_atomically(&record_path(dir, key), &[checksum.as_bytes(), &record])
}

pub(super) fn load(dir: &Path, key: &str) -> io::Result<Prebuild> {
    if !valid_key(key) {
        return Err(invalid("invalid pipeline key"));
    }
    let record = crate::read_file(&record_path(dir, key))?;
    let (checksum, content) = record
        .split_first_chunk::<32>()
        .ok_or_else(|| invalid("short pipeline record"))?;
    if blake3::hash(content).as_bytes() != checksum {
        return Err(invalid("pipeline record checksum mismatch"));
    }
    let spill = dir.join("../../shader-containers");
    let result = parse(content, |hash| {
        let checked = |path: PathBuf| {
            let bytes = crate::read_file(&path)?;
            if blake3::hash(&bytes) != hash {
                return Err(invalid("recorded shader checksum mismatch"));
            }
            Ok(bytes)
        };
        checked(shader_path(dir, hash)).or_else(|error| {
            checked(spill_path(&spill, hash.as_bytes())).map_err(|fallback| {
                if fallback.kind() == io::ErrorKind::NotFound {
                    error
                } else {
                    fallback
                }
            })
        })
    })?;
    if result.key() != key {
        return Err(invalid("recorded pipeline key mismatch"));
    }
    Ok(result)
}

fn parse(
    mut rest: &[u8],
    mut read: impl FnMut(blake3::Hash) -> io::Result<Vec<u8>>,
) -> io::Result<Prebuild> {
    let bad = || invalid("malformed pipeline record");
    let (&count, after) = rest.split_first().ok_or_else(bad)?;
    if !(1..=STAGES.len()).contains(&usize::from(count)) {
        return Err(bad());
    }
    rest = after;
    let mut shaders = Vec::new();
    for _ in 0..count {
        let (head, after) = rest.split_first_chunk::<33>().ok_or_else(bad)?;
        rest = after;
        let stage = STAGES.get(usize::from(head[0])).ok_or_else(bad)?;
        let hash = blake3::Hash::from(*head[1..].first_chunk::<32>().unwrap());
        shaders.push(Arc::new(Shader::new(stage, &read(hash)?)));
    }
    let graphics = match rest {
        [0] => None,
        [1, state @ ..] => Some(Box::new(GraphicsState::get(state).ok_or_else(bad)?)),
        _ => return Err(bad()),
    };
    Ok(Prebuild { shaders, graphics })
}

/// Calls `$f` with each number of a graphics state but its input elements',
/// as a place of 32 bits, in one order.
macro_rules! numbers {
    ($g:expr, $f:ident) => {{
        $f!($g.blend.AlphaToCoverageEnable.0);
        $f!($g.blend.IndependentBlendEnable.0);
        for t in &mut $g.blend.RenderTarget {
            $f!(t.BlendEnable.0);
            $f!(t.LogicOpEnable.0);
            $f!(t.SrcBlend.0);
            $f!(t.DestBlend.0);
            $f!(t.BlendOp.0);
            $f!(t.SrcBlendAlpha.0);
            $f!(t.DestBlendAlpha.0);
            $f!(t.BlendOpAlpha.0);
            $f!(t.LogicOp.0);
            $f!(t.RenderTargetWriteMask);
        }
        $f!($g.sample_mask);
        $f!($g.rasterizer.FillMode.0);
        $f!($g.rasterizer.CullMode.0);
        $f!($g.rasterizer.FrontCounterClockwise.0);
        $f!($g.rasterizer.DepthBias);
        $f!($g.rasterizer.DepthClipEnable.0);
        $f!($g.rasterizer.MultisampleEnable.0);
        $f!($g.rasterizer.AntialiasedLineEnable.0);
        $f!($g.rasterizer.ForcedSampleCount);
        $f!($g.rasterizer.ConservativeRaster.0);
        $f!($g.depth.DepthEnable.0);
        $f!($g.depth.DepthWriteMask.0);
        $f!($g.depth.DepthFunc.0);
        $f!($g.depth.StencilEnable.0);
        $f!($g.depth.StencilReadMask);
        $f!($g.depth.StencilWriteMask);
        for face in [&mut $g.depth.FrontFace, &mut $g.depth.BackFace] {
            $f!(face.StencilFailOp.0);
            $f!(face.StencilDepthFailOp.0);
            $f!(face.StencilPassOp.0);
            $f!(face.StencilFunc.0);
        }
        $f!($g.strip_cut.0);
        $f!($g.topology.0);
        $f!($g.targets.NumRenderTargets);
        for format in &mut $g.targets.RTFormats {
            $f!(format.0);
        }
        $f!($g.depth_format.0);
        $f!($g.samples.Count);
        $f!($g.samples.Quality);
    }};
}

impl GraphicsState {
    /// The state as bytes.
    fn put(&self, out: &mut Vec<u8>) -> Option<()> {
        for f in [
            self.rasterizer.DepthBiasClamp,
            self.rasterizer.SlopeScaledDepthBias,
        ] {
            out.extend(f.to_le_bytes());
        }
        let mut plain = GraphicsState {
            blend: self.blend,
            sample_mask: self.sample_mask,
            rasterizer: self.rasterizer,
            depth: self.depth,
            strip_cut: self.strip_cut,
            topology: self.topology,
            targets: self.targets,
            depth_format: self.depth_format,
            samples: self.samples,
            ..Default::default()
        };
        macro_rules! write {
            ($x:expr) => {
                out.extend(($x as u32).to_le_bytes())
            };
        }
        numbers!(plain, write);
        out.push(self.inputs.len().try_into().ok()?);
        for (e, name) in self.inputs.iter().zip(&self.names) {
            let name = name.as_bytes();
            out.push(name.len().try_into().ok()?);
            out.extend(name);
            for v in [
                e.SemanticIndex,
                e.Format.0 as u32,
                e.InputSlot,
                e.AlignedByteOffset,
                e.InputSlotClass.0 as u32,
                e.InstanceDataStepRate,
            ] {
                out.extend(v.to_le_bytes());
            }
        }
        // Older records omit this byte and have depth bounds disabled.
        if self.depth_bounds {
            out.push(1);
        }
        Some(())
    }

    fn get(bytes: &[u8]) -> Option<GraphicsState> {
        let mut g = GraphicsState::default();
        let mut rest = bytes;
        let mut next = || {
            let (head, after) = rest.split_first_chunk::<4>()?;
            rest = after;
            Some(u32::from_le_bytes(*head))
        };
        g.rasterizer.DepthBiasClamp = f32::from_bits(next()?);
        g.rasterizer.SlopeScaledDepthBias = f32::from_bits(next()?);
        macro_rules! read {
            ($x:expr) => {
                $x = next()? as _
            };
        }
        numbers!(g, read);
        let (&count, mut rest) = rest.split_first()?;
        let mut elements = Vec::new();
        for _ in 0..count {
            let (&len, after) = rest.split_first()?;
            let (name, after) = after.split_at_checked(usize::from(len))?;
            let (numbers, after) = after.split_first_chunk::<24>()?;
            rest = after;
            let mut v = [0u32; 6];
            for (v, b) in v.iter_mut().zip(numbers.as_chunks::<4>().0) {
                *v = u32::from_le_bytes(*b);
            }
            elements.push((CString::new(name).ok()?, v));
        }
        g.depth_bounds = match rest {
            [] => false,
            [1] => true,
            _ => return None,
        };
        g.names = elements.iter().map(|e| e.0.clone()).collect();
        g.inputs = g
            .names
            .iter()
            .zip(&elements)
            .map(|(name, (_, v))| D3D12_INPUT_ELEMENT_DESC {
                SemanticName: PCSTR(name.as_ptr().cast()),
                SemanticIndex: v[0],
                Format: DXGI_FORMAT(v[1] as i32),
                InputSlot: v[2],
                AlignedByteOffset: v[3],
                InputSlotClass: D3D12_INPUT_CLASSIFICATION(v[4] as i32),
                InstanceDataStepRate: v[5],
            })
            .collect();
        Some(g)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_state_comes_back_as_it_was() {
        let mut g = GraphicsState {
            sample_mask: 7,
            ..Default::default()
        };
        g.rasterizer.DepthBias = -3;
        g.rasterizer.SlopeScaledDepthBias = 1.25;
        g.blend.RenderTarget[3].RenderTargetWriteMask = 5;
        g.depth.StencilReadMask = 9;
        g.targets.NumRenderTargets = 2;
        g.targets.RTFormats[1] = DXGI_FORMAT_R8_UNORM;
        g.depth_format = DXGI_FORMAT_D32_FLOAT;
        g.names = vec![
            CString::new("POSITION").unwrap(),
            CString::new("T").unwrap(),
        ];
        g.inputs = g
            .names
            .iter()
            .enumerate()
            .map(|(n, name)| D3D12_INPUT_ELEMENT_DESC {
                SemanticName: PCSTR(name.as_ptr().cast()),
                SemanticIndex: n as u32,
                Format: DXGI_FORMAT_R32G32_FLOAT,
                InputSlot: 2,
                AlignedByteOffset: D3D12_APPEND_ALIGNED_ELEMENT,
                InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA,
                InstanceDataStepRate: 4,
            })
            .collect();
        let mut bytes = Vec::new();
        g.put(&mut bytes).unwrap();
        let back = GraphicsState::get(&bytes).expect("whole");
        assert_eq!(back.describe(), g.describe());
        assert!(GraphicsState::get(&bytes[..bytes.len() - 1]).is_none());
        assert!(GraphicsState::get(&bytes[..9]).is_none());

        g.depth_bounds = true;
        bytes.clear();
        g.put(&mut bytes).unwrap();
        let back = GraphicsState::get(&bytes).expect("depth bounds");
        assert_eq!(back.describe(), g.describe());
    }

    #[test]
    fn recorded_compute_round_trips_and_rejects_bad_cache_data() {
        let dir = tempfile::tempdir().unwrap();
        let p = Prebuild::compute(b"synthetic shader");
        let key = p.key();
        let mut shaders = HashSet::new();
        write_record(dir.path(), &mut shaders, &key, &p.shaders, None, false).unwrap();
        assert_eq!(load(dir.path(), &key).unwrap().key(), key);
        assert_eq!(shaders.len(), 1);
        assert!(load(dir.path(), "../invalid").is_err());

        let path = record_path(dir.path(), &key);
        std::fs::write(&path, [1, 5]).unwrap();
        assert!(load(dir.path(), &key).is_err());
        write_record(dir.path(), &mut shaders, &key, &p.shaders, None, false).unwrap();
        let hash = blake3::Hash::from(p.shaders[0].full_hash);
        std::fs::write(shader_path(dir.path(), hash), b"corrupt").unwrap();
        assert!(load(dir.path(), &key).is_err());
    }

    #[test]
    fn a_new_session_replaces_a_truncated_shader() {
        let dir = tempfile::tempdir().unwrap();
        let p = Prebuild::compute(b"synthetic shader");
        let key = p.key();
        let hash = blake3::Hash::from(p.shaders[0].full_hash);
        let path = shader_path(dir.path(), hash);
        std::fs::write(&path, b"synthetic").unwrap();

        let mut known = HashSet::new();
        write_record(dir.path(), &mut known, &key, &p.shaders, None, false).unwrap();
        assert_eq!(load(dir.path(), &key).unwrap().key(), key);

        std::fs::write(&path, b"synthetic").unwrap();
        write_record(dir.path(), &mut known, &key, &p.shaders, None, false).unwrap();
        assert!(
            load(dir.path(), &key).is_err(),
            "already checked in this session"
        );

        known.clear();
        write_record(dir.path(), &mut known, &key, &p.shaders, None, false).unwrap();
        assert_eq!(load(dir.path(), &key).unwrap().key(), key);
    }

    #[test]
    fn a_spilled_shader_can_still_be_recorded() {
        let cache = tempfile::tempdir().unwrap();
        let spill = cache.path().join("shader-containers");
        let records = cache.path().join("pipelines/known-v1");
        std::fs::create_dir_all(&records).unwrap();
        let p = Prebuild::compute(b"synthetic spilled shader");
        let shader = &p.shaders[0];
        super::super::spill_to(&spill, shader, &shader.full_hash);
        assert!(shader.container.lock().unwrap().is_empty());
        let key = p.key();
        write_record(&records, &mut HashSet::new(), &key, &p.shaders, None, true).unwrap();
        assert!(!shader_path(&records, blake3::Hash::from(shader.full_hash)).exists());
        assert_eq!(load(&records, &key).unwrap().key(), key);
    }
}
