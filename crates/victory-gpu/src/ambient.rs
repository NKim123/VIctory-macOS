//! An opt-in estimate of the indirect light, placed into the screen-probe
//! resolve before the title lights the scene. It uses the previous radiance
//! cache, scene depth and a slow mean to estimate light outside the view.
//! See VIctory-7900XTX 82ca134 and 7da523a, based on beloooo's resolve hook.

use std::sync::{Arc, Mutex, MutexGuard, OnceLock, Weak};

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::{Error, Result, s};

use crate::compute::{self, Param};
use crate::descriptor::host::TextureView;
use crate::host::{HostList, host, transition};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Role {
    Resolve,
    Radiance,
    Air,
}

impl Role {
    pub(crate) fn detect(container: &[u8]) -> Option<Self> {
        Self::in_container(container, settings())
    }

    fn in_container(container: &[u8], settings: Settings) -> Option<Self> {
        if settings.mode == Mode::Off {
            return None;
        }
        let has = |name: &[u8]| container.windows(name.len()).any(|w| w == name);
        if has(b"CS_ScreenProbes_TemporalResolve") {
            Some(Self::Resolve)
        } else if matches!(settings.mode, Mode::Estimate(_)) {
            if has(b"CS_CacheScreenSpaceRadiance") {
                Some(Self::Radiance)
            } else if settings.air > 0.0 && has(b"CS_AccumulateLocalLightInscatteringVolume") {
                Some(Self::Air)
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Root parameter and offset in dwords of the title's output.
    pub(crate) fn image(self) -> (u8, u32) {
        match self {
            Self::Resolve => (1, 176),
            Self::Radiance => (1, 16),
            Self::Air => (1, 32),
        }
    }

    /// Root parameter and offset in dwords of the resolve's scene depth.
    pub(crate) const DEPTH: (u8, u32) = (0, 464);

    /// The stand-in works on full, mip-zero 2D images and a 3D volume.
    pub(crate) fn extent(self, view: TextureView, desc: D3D12_RESOURCE_DESC) -> Option<[u32; 3]> {
        let TextureView::Uav(view) = view else {
            return None;
        };
        let volume = self == Self::Air;
        let dimension = if volume {
            D3D12_RESOURCE_DIMENSION_TEXTURE3D
        } else {
            D3D12_RESOURCE_DIMENSION_TEXTURE2D
        };
        let format = if self == Self::Resolve {
            DXGI_FORMAT_R32_UINT
        } else {
            DXGI_FORMAT_R11G11B10_FLOAT
        };
        if effective_format(view.map(|d| d.Format), desc.Format) != format
            || desc.Dimension != dimension
            || desc.SampleDesc.Count != 1
            || (!volume && desc.DepthOrArraySize != 1)
        {
            return None;
        }
        if let Some(view) = view {
            if volume {
                if view.ViewDimension != D3D12_UAV_DIMENSION_TEXTURE3D {
                    return None;
                }
                // SAFETY: the dimension selects the union member.
                let shape = unsafe { view.Anonymous.Texture3D };
                if shape.MipSlice != 0
                    || shape.FirstWSlice != 0
                    || (shape.WSize != u32::MAX && shape.WSize != u32::from(desc.DepthOrArraySize))
                {
                    return None;
                }
            } else {
                if view.ViewDimension != D3D12_UAV_DIMENSION_TEXTURE2D {
                    return None;
                }
                // SAFETY: the dimension selects the union member.
                let shape = unsafe { view.Anonymous.Texture2D };
                if shape.MipSlice != 0 || shape.PlaneSlice != 0 {
                    return None;
                }
            }
        }
        Some([
            u32::try_from(desc.Width).ok()?,
            desc.Height,
            u32::from(desc.DepthOrArraySize),
        ])
    }
}

fn effective_format(view: Option<DXGI_FORMAT>, resource: DXGI_FORMAT) -> DXGI_FORMAT {
    view.filter(|&f| f != DXGI_FORMAT_UNKNOWN)
        .unwrap_or(resource)
}

pub(crate) fn depth_matches(view: TextureView, desc: D3D12_RESOURCE_DESC, size: [u32; 3]) -> bool {
    let TextureView::Srv(view) = view else {
        return false;
    };
    if desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D
        || desc.DepthOrArraySize != 1
        || desc.SampleDesc.Count != 1
        || [desc.Width, u64::from(desc.Height)] != [u64::from(size[0]), u64::from(size[1])]
    {
        return false;
    }
    if let Some(view) = view {
        if view.ViewDimension != D3D12_SRV_DIMENSION_TEXTURE2D
            || view.Shader4ComponentMapping != D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING
        {
            return false;
        }
        // SAFETY: the dimension selects the union member.
        let shape = unsafe { view.Anonymous.Texture2D };
        if shape.MostDetailedMip != 0
            || shape.MipLevels == 0
            || shape.PlaneSlice != 0
            || shape.ResourceMinLODClamp != 0.0
        {
            return false;
        }
    }
    matches!(
        effective_format(view.map(|d| d.Format), desc.Format),
        DXGI_FORMAT_R32_FLOAT
            | DXGI_FORMAT_R16_UNORM
            | DXGI_FORMAT_R24_UNORM_X8_TYPELESS
            | DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS
    )
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Mode {
    Off,
    Estimate(f32),
    Constant(f32),
}

impl Mode {
    fn parse(value: &str) -> Self {
        let value = value.trim();
        match value.strip_prefix("constant:") {
            Some(level) => positive(level).map_or(Self::Off, Self::Constant),
            None => positive(value).map_or(Self::Off, Self::Estimate),
        }
    }
}

fn positive(value: &str) -> Option<f32> {
    value
        .trim()
        .parse()
        .ok()
        .filter(|v: &f32| v.is_finite() && *v > 0.0)
}

#[derive(Clone, Copy)]
struct Settings {
    mode: Mode,
    shade: [f32; 2],
    air: f32,
}

fn settings() -> Settings {
    static SETTINGS: OnceLock<Settings> = OnceLock::new();
    *SETTINGS.get_or_init(|| {
        let mode = Mode::parse(&std::env::var("VICTORY_AMBIENT").unwrap_or_default());
        let mut shade = [1.0, 0.6];
        if mode == Mode::Off {
            return Settings {
                mode,
                shade,
                air: 0.0,
            };
        }
        let value = std::env::var("VICTORY_AMBIENT_SHADE").unwrap_or_default();
        for (slot, word) in shade.iter_mut().zip(value.split(',')) {
            if let Ok(value) = word.trim().parse::<f32>()
                && value.is_finite()
                && value >= 0.0
            {
                *slot = value;
            }
        }
        Settings {
            mode,
            shade,
            air: positive(&std::env::var("VICTORY_AMBIENT_AIR").unwrap_or_default()).unwrap_or(0.0),
        }
    })
}

pub(crate) fn enabled() -> bool {
    #[cfg(test)]
    if TEST_RECORDER.with(|r| r.borrow().is_some()) {
        return true;
    }
    settings().mode != Mode::Off
}

const SOURCE: &str = include_str!("ambient.hlsl");
const CELLS_BYTES: u64 = 64 * 36 * 16;
// The last float4 indicates whether the gather had radiance to work with.
const POINTS_X: u32 = 80;
const POINTS_Y: u32 = 45;
const POINTS_BYTES: u64 = (POINTS_X as u64 * POINTS_Y as u64 * 3 + 1) * 16;
const MEAN_BYTES: u64 = 16;

const COMMON: D3D12_RESOURCE_STATES = D3D12_RESOURCE_STATE_COMMON;
const READ: D3D12_RESOURCE_STATES = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
const WRITE: D3D12_RESOURCE_STATES = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;

struct Kernel {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
}

impl Kernel {
    fn new(part: &str, name: windows::core::PCSTR, inputs: &[Param<'_>]) -> Result<Self> {
        let mut params = vec![Param::Constants {
            register: 0,
            count: 15,
        }];
        params.extend_from_slice(inputs);
        let (root, pso) = compute::pipeline(
            &format!("#define {part}\n#define PX {POINTS_X}\n#define PY {POINTS_Y}\n{SOURCE}"),
            name,
            0,
            &params,
        )?;
        Ok(Self { root, pso })
    }

    /// Scratch buffers enter and leave in COMMON so independently recorded
    /// lists can run in submission order on the host's single queue.
    fn run(
        &self,
        list: &HostList,
        words: &[u32; 15],
        groups: [u32; 3],
        views: &[D3D12_GPU_DESCRIPTOR_HANDLE],
        scratch: (&[&ID3D12Resource], Option<&ID3D12Resource>),
        gate: Option<&Gate>,
    ) {
        let (reads, output) = scratch;
        let mut barriers = Vec::with_capacity(reads.len() + usize::from(output.is_some()));
        barriers.extend(reads.iter().map(|r| transition(r, COMMON, READ)));
        barriers.extend(output.map(|r| transition(r, COMMON, WRITE)));
        let l = &list.list;
        // SAFETY: an open list, live descriptors in its bound heap, and
        // scratch buffers transitioned for each read or write.
        unsafe {
            if !barriers.is_empty() {
                l.ResourceBarrier(&barriers);
            }
            l.SetComputeRootSignature(&self.root);
            l.SetPipelineState(&self.pso);
            l.SetComputeRoot32BitConstants(0, 15, words.as_ptr().cast(), 0);
            let mut slot = 1;
            for &view in views {
                l.SetComputeRootDescriptorTable(slot, view);
                slot += 1;
            }
            for r in reads {
                l.SetComputeRootShaderResourceView(slot, r.GetGPUVirtualAddress());
                slot += 1;
            }
            if let Some(output) = output {
                l.SetComputeRootUnorderedAccessView(slot, output.GetGPUVirtualAddress());
            }
            if let Some(gate) = gate {
                gate.dispatch(l, groups);
            } else {
                l.Dispatch(groups[0], groups[1], groups[2]);
            }
            barriers.clear();
            barriers.extend(reads.iter().map(|r| transition(r, READ, COMMON)));
            barriers.extend(output.map(|r| transition(r, WRITE, COMMON)));
            if !barriers.is_empty() {
                l.ResourceBarrier(&barriers);
            }
        }
    }
}

struct Estimate {
    keep: Kernel,
    mean: Kernel,
    publish: Kernel,
    place: Kernel,
    gather: Kernel,
    raise: Kernel,
    cells: ID3D12Resource,
    positions: ID3D12Resource,
    points: ID3D12Resource,
    history: ID3D12Resource,
    next_mean: ID3D12Resource,
}

const DEPTH_TABLE: Param<'_> = Param::Table(&[compute::srv(0, 1).volatile()]);
const IMAGE_TABLE: Param<'_> = Param::Table(&[compute::uav(0, 1).volatile()]);
const OUTPUT: Param<'_> = Param::Uav { register: 1 };

const fn scratch_read(register: u32) -> Param<'static> {
    Param::Srv { register, space: 0 }
}

impl Estimate {
    fn new() -> Result<Self> {
        let device = &host().ok_or_else(Error::empty)?.device;
        let mut support = D3D12_FEATURE_DATA_FORMAT_SUPPORT {
            Format: DXGI_FORMAT_R11G11B10_FLOAT,
            ..Default::default()
        };
        // SAFETY: a plain query into a struct of the feature's size.
        unsafe {
            device.CheckFeatureSupport(
                D3D12_FEATURE_FORMAT_SUPPORT,
                std::ptr::from_mut(&mut support).cast(),
                std::mem::size_of_val(&support) as u32,
            )?;
        }
        if !support
            .Support2
            .contains(D3D12_FORMAT_SUPPORT2_UAV_TYPED_LOAD)
        {
            return Err(Error::new(
                windows::Win32::Foundation::E_NOTIMPL,
                "R11G11B10_FLOAT typed UAV loads are unavailable",
            ));
        }
        Ok(Self {
            keep: Kernel::new("KEEP", s!("ambient keep"), &[IMAGE_TABLE, OUTPUT])?,
            mean: Kernel::new(
                "MEAN",
                s!("ambient mean"),
                &[scratch_read(1), scratch_read(2), OUTPUT],
            )?,
            publish: Kernel::new("PUBLISH", s!("ambient publish"), &[scratch_read(2), OUTPUT])?,
            place: Kernel::new("PLACE", s!("ambient place"), &[DEPTH_TABLE, OUTPUT])?,
            gather: Kernel::new(
                "GATHER",
                s!("ambient gather"),
                &[
                    DEPTH_TABLE,
                    scratch_read(1),
                    scratch_read(2),
                    scratch_read(3),
                    OUTPUT,
                ],
            )?,
            raise: Kernel::new(
                "RAISE",
                s!("ambient resolve"),
                &[DEPTH_TABLE, IMAGE_TABLE, scratch_read(2), scratch_read(4)],
            )?,
            cells: buffer(CELLS_BYTES, true)?,
            positions: buffer(CELLS_BYTES, true)?,
            points: buffer(POINTS_BYTES, true)?,
            history: buffer(MEAN_BYTES, true)?,
            next_mean: buffer(MEAN_BYTES, true)?,
        })
    }
}

fn buffer(size: u64, writable: bool) -> Result<ID3D12Resource> {
    let device = &host().ok_or_else(Error::empty)?.device;
    let mut buffer = None;
    // SAFETY: plain creation. The default heap starts zeroed, so history
    // and points are invalid until their corresponding dispatch runs.
    unsafe {
        device.CreateCommittedResource(
            &D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_DEFAULT,
                ..Default::default()
            },
            D3D12_HEAP_FLAG_NONE,
            &D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: size,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                Flags: if writable {
                    D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
                } else {
                    D3D12_RESOURCE_FLAG_NONE
                },
                ..Default::default()
            },
            COMMON,
            None,
            &mut buffer,
        )?;
    }
    buffer.ok_or_else(Error::empty)
}

/// A version of one host descriptor slot. Rewrites wait for an executing
/// pair, and invalidate the version before writing a new descriptor.
#[derive(Clone)]
pub(crate) struct Watch(Arc<Mutex<Version>>);

pub(crate) struct Version {
    valid: bool,
    frozen: Weak<Frozen>,
}

/// The native fence that protects a recorded descriptor version. A
/// descriptor writer can wait for it even while holding its slot lock.
pub(crate) struct Frozen {
    fence: ID3D12Fence,
    value: u64,
}

impl Watch {
    pub(crate) fn new() -> Self {
        Self(Arc::new(Mutex::new(Version {
            valid: true,
            frozen: Weak::new(),
        })))
    }

    pub(crate) fn invalidate(&self) {
        let mut version = self.0.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(frozen) = version.frozen.upgrade() {
            crate::host::poll_fence(&frozen.fence, frozen.value);
        }
        version.valid = false;
    }

    fn key(&self) -> usize {
        Arc::as_ptr(&self.0) as usize
    }
}

pub(crate) struct Binding {
    pub(crate) resource: ID3D12Resource,
    pub(crate) view: TextureView,
    pub(crate) handle: D3D12_GPU_DESCRIPTOR_HANDLE,
    pub(crate) watch: Watch,
}

pub(crate) struct CheckedViews {
    pub(crate) image: Binding,
    pub(crate) depth: Option<Binding>,
    pub(crate) size: [u32; 3],
}

/// One count word and space for the indirect arguments of a complete pass.
const DISPATCH_BYTES: u64 = 64;
pub(crate) const MAX_PASSES: usize = 64;
pub(crate) const UPLOAD_BYTES: u64 = DISPATCH_BYTES * MAX_PASSES as u64;

#[derive(Clone)]
pub(crate) struct Candidate {
    image: Watch,
    depth: Option<Watch>,
    count: usize,
}

impl Candidate {
    pub(crate) fn new(views: &CheckedViews, count: usize) -> Self {
        Self {
            image: views.image.watch.clone(),
            depth: views.depth.as_ref().map(|depth| depth.watch.clone()),
            count,
        }
    }

    fn watches(&self) -> impl Iterator<Item = &Watch> {
        std::iter::once(&self.image).chain(self.depth.iter())
    }
}

/// Under the submission lock, after all heap sync and deferred CPU copies.
/// Hold each version at most once, even if many passes share the same slot.
pub(crate) fn arm(candidates: &[Candidate]) -> Vec<MutexGuard<'_, Version>> {
    let mut watches: Vec<_> = candidates.iter().flat_map(Candidate::watches).collect();
    watches.sort_unstable_by_key(|w| w.key());
    watches.dedup_by_key(|w| w.key());
    let locks: Vec<_> = watches
        .iter()
        .map(|w| w.0.lock().unwrap_or_else(|e| e.into_inner()))
        .collect();
    for pass in candidates {
        let ready = pass.watches().all(|w| {
            let index = watches.binary_search_by_key(&w.key(), |w| w.key()).unwrap();
            locks[index].valid
        });
        // SAFETY: a mapped word owned by the closed list, with no other
        // instance executing (submissions hold the host lock).
        unsafe { (pass.count as *mut u32).write_volatile(u32::from(ready)) };
    }
    locks
}

/// Arm the gate words while these exact descriptor versions are locked,
/// then hand the hold to the native submission fence.
pub(crate) fn freeze(
    candidates: &[Candidate],
    fence: &ID3D12Fence,
    value: u64,
) -> Option<Arc<Frozen>> {
    if candidates.is_empty() {
        return None;
    }
    let frozen = Arc::new(Frozen {
        fence: fence.clone(),
        value,
    });
    for mut version in arm(candidates) {
        if version.valid {
            version.frozen = Arc::downgrade(&frozen);
        }
    }
    Some(frozen)
}

pub(crate) struct Gate {
    buffer: ID3D12Resource,
    signature: ID3D12CommandSignature,
    base: u64,
    data: usize,
    next: std::cell::Cell<u64>,
}

impl Gate {
    pub(crate) fn new(buffer: &ID3D12Resource, data: usize, pass: usize) -> Option<Self> {
        static SIGNATURE: OnceLock<Option<ID3D12CommandSignature>> = OnceLock::new();
        let signature = SIGNATURE
            .get_or_init(|| {
                let host = host()?;
                let arg = D3D12_INDIRECT_ARGUMENT_DESC {
                    Type: D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH,
                    ..Default::default()
                };
                let desc = D3D12_COMMAND_SIGNATURE_DESC {
                    ByteStride: 12,
                    NumArgumentDescs: 1,
                    pArgumentDescs: &arg,
                    NodeMask: 0,
                };
                let mut signature = None;
                // SAFETY: one dispatch argument, no root arguments.
                unsafe {
                    host.device.CreateCommandSignature(
                        &desc,
                        None::<&ID3D12RootSignature>,
                        &mut signature,
                    )
                }
                .ok()?;
                signature
            })
            .as_ref()?
            .clone();
        let base = pass as u64 * DISPATCH_BYTES;
        let data = data + base as usize;
        // SAFETY: one count and its arguments per pass in the mapped upload.
        unsafe { (data as *mut u32).write(0) };
        Some(Self {
            buffer: buffer.clone(),
            signature,
            base,
            data,
            next: 4.into(),
        })
    }

    pub(crate) fn count(&self) -> usize {
        self.data
    }

    fn dispatch(&self, list: &ID3D12GraphicsCommandList, groups: [u32; 3]) {
        let at = self.next.get();
        assert!(at + 12 <= DISPATCH_BYTES);
        self.next.set(at + 12);
        // SAFETY: distinct arguments in the mapped upload, written before
        // submission. The same count gates every dispatch of this pass.
        unsafe {
            ((self.data + at as usize) as *mut [u32; 3]).write(groups);
            list.ExecuteIndirect(
                &self.signature,
                1,
                &self.buffer,
                self.base + at,
                Some(&self.buffer),
                self.base,
            );
        }
    }
}

pub(crate) struct Views {
    pub(crate) resource: ID3D12Resource,
    pub(crate) image: D3D12_GPU_DESCRIPTOR_HANDLE,
    pub(crate) depth: Option<D3D12_GPU_DESCRIPTOR_HANDLE>,
    pub(crate) size: [u32; 3],
}

impl CheckedViews {
    pub(crate) fn handles(&self) -> Views {
        Views {
            resource: self.image.resource.clone(),
            image: self.image.handle,
            depth: self.depth.as_ref().map(|depth| depth.handle),
            size: self.size,
        }
    }
}

fn uav_barrier(list: &HostList, resource: &ID3D12Resource) {
    let barrier = D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                // SAFETY: a borrowed pointer the barrier never releases.
                pResource: unsafe { std::mem::transmute_copy(resource) },
            }),
        },
    };
    // SAFETY: an open list; the barrier orders unordered accesses to the
    // title's image without changing its state.
    unsafe { list.list.ResourceBarrier(&[barrier]) };
}

/// Record after the title's dispatch with the same view heap bound.
/// Returns whether the host pipeline and compute root have been replaced.
pub(crate) fn after(list: &HostList, role: Role, views: &Views, gate: &Gate) -> bool {
    #[cfg(test)]
    if let Some(ran) = TEST_RECORDER.with(|r| {
        r.borrow()
            .as_ref()
            .map(|r| r.after(list, role, views, Some(gate)))
    }) {
        return ran;
    }
    static RECORDER: OnceLock<Recorder> = OnceLock::new();
    RECORDER
        .get_or_init(|| Recorder::new(settings()))
        .after(list, role, views, Some(gate))
}

#[cfg(test)]
thread_local! {
    static TEST_RECORDER: std::cell::RefCell<Option<Recorder>> = const { std::cell::RefCell::new(None) };
}

#[cfg(test)]
pub(crate) fn test_constant() {
    TEST_RECORDER.with(|r| {
        *r.borrow_mut() = Some(Recorder::new(Settings {
            mode: Mode::Constant(0.25),
            shade: [0.0, 0.6],
            air: 0.0,
        }))
    });
}

/// Light history follows elapsed time, with an immediate response to a
/// sufficiently large change in luminance.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Settle {
    ms: f32,
    jump: f32,
    fresh: f32,
}

impl Settle {
    fn parse(spec: &str) -> Option<Self> {
        let mut parts = spec.split(',').map(|p| {
            p.trim()
                .parse::<f32>()
                .ok()
                .filter(|v| v.is_finite() && *v >= 0.0)
        });
        Some(Self {
            ms: parts.next().flatten().filter(|&ms| ms > 0.0)?,
            jump: parts.next().flatten().unwrap_or(3.0),
            fresh: parts.next().flatten().unwrap_or(0.0),
        })
    }

    fn follow(self, dt_ms: f32) -> f32 {
        -(-dt_ms / self.ms).exp_m1()
    }
}

fn settle() -> Option<Settle> {
    static SETTLE: OnceLock<Option<Settle>> = OnceLock::new();
    *SETTLE.get_or_init(|| {
        Settle::parse(&std::env::var("VICTORY_AMBIENT_SETTLE").unwrap_or_else(|_| "250,3".into()))
    })
}

fn radiance_ms(last: Option<std::time::Instant>, now: std::time::Instant) -> f32 {
    last.map(|prev| now.saturating_duration_since(prev).as_secs_f32() * 1000.0)
        .filter(|&ms| ms <= 1000.0)
        .map_or(1000.0 / 30.0, |ms| ms.max(1.0))
}

struct Recorder {
    settings: Settings,
    constant: OnceLock<Option<Kernel>>,
    estimate: OnceLock<Option<Estimate>>,
    air: OnceLock<Option<Kernel>>,
    last_radiance: std::sync::Mutex<Option<std::time::Instant>>,
}

impl Recorder {
    fn new(settings: Settings) -> Self {
        Self {
            settings,
            constant: OnceLock::new(),
            estimate: OnceLock::new(),
            air: OnceLock::new(),
            last_radiance: std::sync::Mutex::new(None),
        }
    }

    fn after(&self, list: &HostList, role: Role, views: &Views, gate: Option<&Gate>) -> bool {
        let tuning = settle();
        let mut words = constants(self.settings, views.size);
        words[13] = tuning.map_or(0.0, |s| s.fresh).to_bits();
        let [width, height, slices] = views.size;
        let groups = [width.div_ceil(8), height.div_ceil(8), 1];
        match (self.settings.mode, role, views.depth) {
            (Mode::Constant(_), Role::Resolve, Some(depth)) => {
                let Some(kernel) = self
                    .constant
                    .get_or_init(|| {
                        optional(Kernel::new(
                            "CONSTANT",
                            s!("ambient constant"),
                            &[DEPTH_TABLE, IMAGE_TABLE],
                        ))
                    })
                    .as_ref()
                else {
                    return false;
                };
                uav_barrier(list, &views.resource);
                kernel.run(
                    list,
                    &words,
                    groups,
                    &[depth, views.image],
                    (&[], None),
                    gate,
                );
            }
            (Mode::Estimate(_), _, depth) => {
                let Some(pass) = self
                    .estimate
                    .get_or_init(|| optional(Estimate::new()))
                    .as_ref()
                else {
                    return false;
                };
                match (role, depth) {
                    (Role::Radiance, _) => {
                        if let Some(tuning) = tuning {
                            let mut last =
                                self.last_radiance.lock().unwrap_or_else(|e| e.into_inner());
                            let now = std::time::Instant::now();
                            let dt = radiance_ms(last.replace(now), now);
                            words[8] = tuning.follow(dt).to_bits();
                            words[12] = tuning.jump.to_bits();
                            words[14] = (dt / 1000.0).to_bits();
                        }
                        uav_barrier(list, &views.resource);
                        pass.keep.run(
                            list,
                            &words,
                            [8, 9, 1],
                            &[views.image],
                            (&[], Some(&pass.cells)),
                            gate,
                        );
                        pass.mean.run(
                            list,
                            &words,
                            [1, 1, 1],
                            &[],
                            (&[&pass.cells, &pass.history], Some(&pass.next_mean)),
                            gate,
                        );
                        pass.publish.run(
                            list,
                            &words,
                            [1, 1, 1],
                            &[],
                            (&[&pass.next_mean], Some(&pass.history)),
                            gate,
                        );
                    }
                    (Role::Resolve, Some(depth)) => {
                        uav_barrier(list, &views.resource);
                        pass.place.run(
                            list,
                            &words,
                            [8, 9, 1],
                            &[depth],
                            (&[], Some(&pass.positions)),
                            gate,
                        );
                        pass.gather.run(
                            list,
                            &words,
                            [POINTS_X / 8, POINTS_Y, 1],
                            &[depth],
                            (
                                &[&pass.cells, &pass.history, &pass.positions],
                                Some(&pass.points),
                            ),
                            gate,
                        );
                        pass.raise.run(
                            list,
                            &words,
                            groups,
                            &[depth, views.image],
                            (&[&pass.history, &pass.points], None),
                            gate,
                        );
                    }
                    (Role::Air, _) if self.settings.air > 0.0 => {
                        let Some(air) = self
                            .air
                            .get_or_init(|| {
                                optional(Kernel::new(
                                    "AIR",
                                    s!("ambient air"),
                                    &[IMAGE_TABLE, scratch_read(2), scratch_read(4)],
                                ))
                            })
                            .as_ref()
                        else {
                            return false;
                        };
                        uav_barrier(list, &views.resource);
                        air.run(
                            list,
                            &words,
                            [groups[0], groups[1], slices.div_ceil(2)],
                            &[views.image],
                            (&[&pass.history, &pass.points], None),
                            gate,
                        );
                    }
                    _ => return false,
                }
            }
            _ => return false,
        }
        uav_barrier(list, &views.resource);
        list.keep(&views.resource);
        true
    }
}

fn optional<T>(result: Result<T>) -> Option<T> {
    result
        .inspect_err(|e| tracing::warn!(error = %e, "could not build an indirect-light pass"))
        .ok()
}

fn constants(settings: Settings, [width, height, slices]: [u32; 3]) -> [u32; 15] {
    let (level, constant) = match settings.mode {
        Mode::Estimate(level) => (level, 0.0),
        Mode::Constant(level) => (0.0, level),
        Mode::Off => (0.0, 0.0),
    };
    let [strength, radius] = settings.shade;
    [
        level.to_bits(),
        constant.to_bits(),
        strength.to_bits(),
        radius.to_bits(),
        0.1_f32.to_bits(),
        0.45_f32.to_bits(),
        width,
        height,
        0.04_f32.to_bits(),
        4.0_f32.to_bits(),
        settings.air.to_bits(),
        slices,
        0,
        0,
        0,
    ]
}

#[cfg(test)]
mod tests;
