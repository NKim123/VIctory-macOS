//! Host descriptor heaps behind the title's shader-visible heaps (decision
//! 0007, step 3). Translated shaders index the heaps directly at the
//! title's descriptor positions, so each host heap keeps the title heap's
//! order: slot `i` is made from what the title wrote at slot `i`, when it
//! writes it (view creation and descriptor copies) and again when a draw
//! reads it through a table and finds the slot changed.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

pub(crate) use super::paged::Paged;
use super::{
    ChangedRecord, VIEW_TAG, View, ViewPin, changed_records, pin_changed_views, record_at, view_at,
    view_gen_current, views_at_batch, views_forgot,
};
use crate::backend::{HostGpu, HostMemory};
use crate::host::host;
use crate::resource::HostTexture;

mod templates;
use templates::Templates;

/// Opt-in. Use null SRVs in shared title slots until the first upload runs.
pub(crate) fn show_after() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("VICTORY_SHOW_AFTER").is_some_and(|v| v == "1"))
}

fn record_key([_, view, generation]: [u32; 3]) -> u64 {
    u64::from(generation) << 32 | u64::from(view)
}

pub(super) fn header_matches(address: u64, record: [u32; 3]) -> bool {
    #[cfg(target_arch = "x86_64")]
    if super::scan::on() {
        return super::scan::matches(address as *const u8, record);
    }
    read_header(address) == Some(record)
}

fn read_header(address: u64) -> Option<[u32; 3]> {
    let mut bytes = [[0u8; 4]; 3];
    crate::host::read_into(address as *const u8, bytes.as_flattened_mut())
        .then(|| bytes.map(u32::from_le_bytes))
}

fn capture_header(address: u64) -> Option<[u32; 3]> {
    #[cfg(target_arch = "x86_64")]
    if super::scan::on()
        && let Some(record) = super::scan::read(address as *const u8)
    {
        return Some(record);
    }
    read_header(address)
}

#[cfg(test)]
thread_local! {
    pub(crate) static RELEASE_BEFORE_PENDING_PUBLICATION: std::cell::Cell<Option<usize>> =
        const { std::cell::Cell::new(None) };
}

struct PendingTexture {
    record: [u32; 3],
    copy: Weak<HostTexture>,
    placed: bool,
    resource: usize,
    memory: usize,
    layout: crate::resource::ResourceDesc,
    demanded: bool,
    desc: Option<D3D12_SHADER_RESOURCE_VIEW_DESC>,
}

impl PendingTexture {
    fn current(&self, copy: &HostTexture) -> bool {
        !copy.evicted.load(Ordering::Acquire) && (!self.placed || copy.current_for_placement())
    }
}

struct Mirrored {
    /// The record (view index and generation) each slot was last made from.
    keys: Vec<u64>,
    /// Unpublished texture SRVs, guarded by the same lock as the records.
    pending: BTreeMap<u32, PendingTexture>,
    templates: [OnceLock<Option<Templates>>; TEMPLATE_SHARDS],
    textures: [OnceLock<Option<TextureTemplates>>; TEMPLATE_SHARDS],
}

impl Mirrored {
    fn null_template(&self) -> Option<D3D12_CPU_DESCRIPTOR_HANDLE> {
        self.templates
            .iter()
            .filter_map(OnceLock::get)
            .find_map(Option::as_ref)
            .map(Templates::null)
            .or_else(|| {
                self.textures
                    .iter()
                    .filter_map(OnceLock::get)
                    .find_map(Option::as_ref)
                    .map(TextureTemplates::null)
            })
    }
}

/// Header and index captured in one scan. The x86-64 scanner writes this layout.
#[repr(C)]
#[derive(Debug, Eq, PartialEq)]
pub(super) struct Candidate {
    pub(super) index: u32,
    pub(super) record: [u32; 3],
}

struct SlotScan {
    candidates: Vec<Candidate>,
    forgot: u64,
}

enum Publication {
    Clear(u32, Kept),
    View(u32, u64, bool, Binding, Kept),
}

enum Binding {
    Buffer(BufferView, bool),
    Texture(Option<usize>),
    Left {
        kind: (usize, u64, u64),
        deferred: bool,
    },
    Sampler,
}

const TEMPLATE_SHARDS: usize = 4;
const STAGING_WORKERS: usize = TEMPLATE_SHARDS;
#[cfg(test)]
thread_local! {
    static STAGED_BATCHES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

const STAGING_SLOTS: usize = 4096;
const MIN_STAGED_RECORDS: usize = 64;

/// Scratch descriptors live until every destination has copied them. A
/// worker owns its CPU-only heap and template cache for the entire round.
struct ViewStaging {
    _heap: ID3D12DescriptorHeap,
    cpu: usize,
    step: usize,
    templates: Templates,
    textures: OnceLock<Option<TextureTemplates>>,
}

impl ViewStaging {
    fn new() -> Option<Self> {
        let device = &host()?.device;
        let kind = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        // SAFETY: a CPU-only resource heap, independent of other workers.
        let (heap, cpu, step) = unsafe {
            let heap: ID3D12DescriptorHeap = device
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: kind,
                    NumDescriptors: STAGING_SLOTS as u32,
                    ..Default::default()
                })
                .ok()?;
            let cpu = heap.GetCPUDescriptorHandleForHeapStart().ptr;
            let step = device.GetDescriptorHandleIncrementSize(kind) as usize;
            (heap, cpu, step)
        };
        Some(Self {
            _heap: heap,
            cpu,
            step,
            templates: Templates::new(false)?,
            textures: OnceLock::new(),
        })
    }

    fn stage<'a>(&mut self, slot: usize, view: &'a ViewPin) -> Option<StagedView<'a>> {
        let at = D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: self.cpu + slot * self.step,
        };
        if is_placed_texture(view) {
            // Heap sync leaves these null until a table or a shader
            // reads the slot. Stage the exact null view once per key.
            null_for(view, at);
            let kind = crate::bindless::left_kind(view);
            return Some(StagedView {
                view,
                at,
                held: None,
                mark: None,
                source: StagedSource::DeferredTexture(kind),
            });
        }
        let epoch = crate::shadow::lookup_epoch();
        let (held, mark, source) =
            if matches!(**view, View::Constant(..)) || buffer_range(view).is_some() {
                let (held, backing, mark) = if let Some(hit) = self.templates.get(view.key, epoch) {
                    templates::copy(false, at, hit.at);
                    (hit.held, hit.backing?, hit.mark)
                } else {
                    let backing = prepare(view)?;
                    let (held, mark) =
                        self.templates
                            .make(at, view.key, epoch, view, Some(&backing));
                    (held?, backing, mark)
                };
                (
                    held,
                    mark,
                    StagedSource::Buffer(AtomicU64::new(epoch), backing),
                )
            } else {
                let (resource, srv, ambient) = match **view {
                    View::Shader(resource, _) => (resource, true, true),
                    View::Unordered(resource, counter, _) => (resource, false, counter == 0),
                    _ => return None,
                };
                // A cold descriptor can view a texture that already exists.
                // Leave host texture allocation, depth twins and pending uploads
                // on the ordinary path, which owns their state transitions.
                let resource = resource as crate::object::This;
                let copy = crate::resource::made_copy(resource)?;
                if copy.evicted.load(Ordering::Acquire) {
                    return None;
                }
                if show_after()
                    && srv
                    && !copy.initialized.load(Ordering::Acquire)
                    && crate::texture::eligible_for_gating(resource, &copy)
                {
                    return None;
                }
                let cache = self.textures.get_or_init(TextureTemplates::new);
                let cached = cache
                    .as_ref()
                    .and_then(|cache| cache.entry(view.key))
                    .filter(|entry| std::ptr::eq(entry.copy.as_ptr(), Arc::as_ptr(&copy)));
                let desc = if let Some(entry) = cached {
                    templates::copy(false, at, entry.at);
                    entry.view
                } else {
                    let desc = match &**view {
                        View::Shader(_, desc) => {
                            TextureView::Srv(texture_srv_desc(resource, &copy, desc.as_ref())?)
                        }
                        View::Unordered(_, _, desc) => {
                            // SAFETY: the copy owns the resource.
                            if unsafe { copy.texture.GetDesc() }.Flags.0
                                & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL.0
                                != 0
                            {
                                return None;
                            }
                            TextureView::Uav(color_uav(resource, &copy, desc.as_ref())?)
                        }
                        _ => unreachable!("resource texture"),
                    };
                    let cache = self
                        .textures
                        .get_mut()
                        .and_then(Option::as_mut)
                        .map(|cache| (cache, view.key));
                    write_texture_view(&copy, desc, at, cache);
                    desc
                };
                let ambient = ambient.then(|| desc.for_ambient(&copy.texture)).flatten();
                let held = Some((copy.texture.clone(), ambient));
                (held, None, StagedSource::Texture(copy, srv))
            };
        Some(StagedView {
            view,
            at,
            held,
            mark,
            source,
        })
    }
}

enum StagedSource {
    Buffer(AtomicU64, BufferView),
    Texture(Arc<HostTexture>, bool),
    DeferredTexture((usize, u64, u64)),
}

struct StagedView<'a> {
    view: &'a ViewPin,
    at: D3D12_CPU_DESCRIPTOR_HANDLE,
    held: Kept,
    mark: Option<(u32, u32)>,
    source: StagedSource,
}

impl StagedView<'_> {
    fn recheck_buffer(&self, epoch: u64) -> bool {
        let StagedSource::Buffer(created, (shadow, address, len)) = &self.source else {
            return false;
        };
        // Preparing other descriptors can add unrelated shadows and
        // invalidate the epoch before this batch reaches publication.
        // Reuse the staged descriptor if the same range still resolves
        // to the exact backing it names.
        let current = created.load(Ordering::Relaxed);
        if current != epoch
            && crate::shadow::lookup_epoch() == epoch
            && prepare(self.view).is_some_and(|(current, from, bytes)| {
                Arc::ptr_eq(&current, shadow) && from == *address && bytes == *len
            })
            && crate::shadow::lookup_epoch() == epoch
        {
            created.store(epoch, Ordering::Relaxed);
            return true;
        }
        false
    }

    fn hit(&self, epoch: u64) -> Option<TemplateHit> {
        let mut texture = None;
        let binding = match &self.source {
            StagedSource::Buffer(created, backing) if created.load(Ordering::Relaxed) == epoch => {
                Binding::Buffer(backing.clone(), is_widened(&self.held))
            }
            StagedSource::Texture(copy, srv) => {
                let (View::Shader(resource, _) | View::Unordered(resource, ..)) = **self.view
                else {
                    unreachable!("staged resource texture");
                };
                if !crate::resource::made_copy(resource as crate::object::This).is_some_and(
                    |current| {
                        Arc::ptr_eq(&current, copy) && !current.evicted.load(Ordering::Acquire)
                    },
                ) {
                    return None;
                }
                texture = Some((copy.clone(), *srv));
                Binding::Texture((*srv && copy.title_written()).then_some(resource))
            }
            StagedSource::DeferredTexture(kind) => Binding::Left {
                kind: *kind,
                deferred: true,
            },
            _ => return None,
        };
        Some(TemplateHit {
            at: self.at,
            mark: self.mark,
            held: self.held.clone(),
            binding,
            texture,
        })
    }
}

struct StagedViews<'a> {
    views: HashMap<u64, StagedView<'a>>,
    null: D3D12_CPU_DESCRIPTOR_HANDLE,
}

impl StagedViews<'_> {
    /// The epoch can move between staging and publication even after the
    /// initial worker recheck. Retry only the unresolved, staged mappings.
    /// Publication still validates the current epoch and the title header.
    fn recheck_pending(&self, records: &[ChangedRecord]) -> bool {
        let epoch = crate::shadow::lookup_epoch();
        let mut seen = HashSet::new();
        let mut jobs: [_; STAGING_WORKERS] = std::array::from_fn(|_| Vec::new());
        for record in records.iter().filter(|record| !record.clear) {
            let key = record_key(record.record);
            if let Some(view) = self.views.get(&key)
                && let StagedSource::Buffer(created, _) = &view.source
                && created.load(Ordering::Relaxed) != epoch
                && seen.insert(key)
            {
                jobs[template_shard(key)].push(view);
            }
        }
        let jobs: Vec<_> = jobs.iter().filter(|job| !job.is_empty()).collect();
        let refreshed = AtomicBool::new(false);
        crate::texture::in_parallel(jobs.len(), |i| {
            let mut any = false;
            for view in jobs[i] {
                any |= view.recheck_buffer(epoch);
            }
            if any {
                refreshed.store(true, Ordering::Relaxed);
            }
        });
        refreshed.load(Ordering::Relaxed)
    }
}

#[derive(Default)]
struct PublishBatch {
    copies: DescriptorCopies,
    published: Vec<Publication>,
}

fn is_widened(held: &Kept) -> bool {
    matches!(held, Some((_, Some(HeldSource::Widened(..)))))
}

fn template_shard(key: u64) -> usize {
    ((key as u32) ^ ((key >> 32) as u32)) as usize % TEMPLATE_SHARDS
}

/// One process-wide set of scratch heaps, kept locked through publication
/// so another heap sync cannot overwrite descriptors still being copied.
type StagingWorkers = [Mutex<ViewStaging>; STAGING_WORKERS];

static VIEW_STAGING: Mutex<OnceLock<Option<StagingWorkers>>> = Mutex::new(OnceLock::new());

fn view_staging() -> Option<StagingWorkers> {
    (0..STAGING_WORKERS)
        .map(|_| ViewStaging::new().map(Mutex::new))
        .collect::<Option<Vec<_>>>()?
        .try_into()
        .ok()
}

fn stage_views<'a>(
    workers: &StagingWorkers,
    records: &[ChangedRecord],
    pins: &'a HashMap<u64, ViewPin>,
) -> StagedViews<'a> {
    let mut seen = HashSet::new();
    let mut jobs: [_; STAGING_WORKERS] = std::array::from_fn(|_| Vec::new());
    for record in records.iter().filter(|record| !record.clear) {
        let key = record_key(record.record);
        let worker = template_shard(key);
        let job = &mut jobs[worker];
        if job.len() < STAGING_SLOTS
            && seen.insert(key)
            && let Some(view) = pins.get(&key)
        {
            job.push(view);
        }
    }
    let before = crate::shadow::lookup_epoch();
    let results: [_; STAGING_WORKERS] = std::array::from_fn(|_| Mutex::new(Vec::new()));
    let active: Vec<_> = (0..STAGING_WORKERS)
        .filter(|&i| !jobs[i].is_empty())
        .collect();
    crate::texture::in_parallel(active.len(), |worker| {
        let i = active[worker];
        let mut staging = workers[i].lock().unwrap_or_else(|e| e.into_inner());
        *results[i].lock().unwrap_or_else(|e| e.into_inner()) = jobs[i]
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(slot, view)| staging.stage(slot, view))
            .collect();
    });
    let epoch = crate::shadow::lookup_epoch();
    if before != epoch {
        crate::texture::in_parallel(active.len(), |worker| {
            let i = active[worker];
            for entry in &*results[i].lock().unwrap_or_else(|e| e.into_inner()) {
                entry.recheck_buffer(epoch);
            }
        });
    }
    let views = results
        .into_iter()
        .flat_map(|result| result.into_inner().unwrap())
        .map(|entry| (entry.view.key, entry))
        .collect();
    let null = workers[0]
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .templates
        .null();
    StagedViews { views, null }
}

#[derive(Default)]
struct DescriptorCopies {
    dst: Vec<D3D12_CPU_DESCRIPTOR_HANDLE>,
    lengths: Vec<u32>,
    next_dst: Option<usize>,
    src: Vec<D3D12_CPU_DESCRIPTOR_HANDLE>,
}

impl DescriptorCopies {
    fn push(
        &mut self,
        dst: D3D12_CPU_DESCRIPTOR_HANDLE,
        src: D3D12_CPU_DESCRIPTOR_HANDLE,
        step: usize,
    ) {
        if self.next_dst == Some(dst.ptr) {
            *self.lengths.last_mut().unwrap() += 1;
        } else {
            self.dst.push(dst);
            self.lengths.push(1);
        }
        self.next_dst = Some(dst.ptr + step);
        self.src.push(src);
    }

    fn publish(&mut self, sampler: bool) {
        let n = self.src.len();
        if n == 1 {
            templates::copy(sampler, self.dst[0], self.src[0]);
        } else if n > 1 {
            let kind = if sampler {
                D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER
            } else {
                D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV
            };
            // SAFETY: the destination ranges are disjoint and enumerate
            // the same number of descriptors as the single-slot sources.
            // All CPU-only templates live under the mirror lock.
            unsafe {
                host().unwrap().device.CopyDescriptors(
                    self.dst.len() as u32,
                    self.dst.as_ptr(),
                    Some(self.lengths.as_ptr()),
                    n as u32,
                    self.src.as_ptr(),
                    None,
                    kind,
                );
            }
        }
        self.dst.clear();
        self.lengths.clear();
        self.next_dst = None;
        self.src.clear();
    }
}

/// CPU-only descriptors for resource textures. The record key includes the
/// view generation. A hit also checks the current host copy because eviction
/// can replace the texture without changing the title record.
struct TextureTemplates {
    _heap: ID3D12DescriptorHeap,
    cpu: usize,
    step: usize,
    entries: Box<[Option<TextureTemplate>]>,
}

struct TextureTemplate {
    key: u64,
    at: D3D12_CPU_DESCRIPTOR_HANDLE,
    copy: Weak<HostTexture>,
    view: TextureView,
}

impl TextureTemplates {
    const COUNT: u32 = 8192 / TEMPLATE_SHARDS as u32;

    fn new() -> Option<Self> {
        let device = &host()?.device;
        let kind = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        // SAFETY: a CPU-only heap of a supported type and size.
        let (heap, cpu, step) = unsafe {
            let heap: ID3D12DescriptorHeap = device
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: kind,
                    NumDescriptors: Self::COUNT + 1,
                    ..Default::default()
                })
                .ok()?;
            let cpu = heap.GetCPUDescriptorHandleForHeapStart().ptr;
            let step = device.GetDescriptorHandleIncrementSize(kind) as usize;
            (heap, cpu, step)
        };
        let templates = Self {
            _heap: heap,
            cpu,
            step,
            entries: std::iter::repeat_with(|| None)
                .take(Self::COUNT as usize)
                .collect(),
        };
        null_view(templates.null());
        Some(templates)
    }

    fn slot(&self, bucket: usize) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: self.cpu + bucket * self.step,
        }
    }

    fn null(&self) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        self.slot(Self::COUNT as usize)
    }

    fn bucket(key: u64) -> usize {
        ((key ^ key.rotate_left(32)).wrapping_mul(0x9e37_79b9_7f4a_7c15)
            >> (64 - Self::COUNT.trailing_zeros())) as usize
    }

    fn entry(&self, key: u64) -> Option<&TextureTemplate> {
        let entry = self.entries[Self::bucket(key)].as_ref()?;
        (entry.key == key).then_some(entry)
    }

    fn publish(
        &mut self,
        key: u64,
        copy: &Arc<HostTexture>,
        view: TextureView,
        dst: D3D12_CPU_DESCRIPTOR_HANDLE,
    ) {
        let bucket = Self::bucket(key);
        let at = self.slot(bucket);
        view.write(copy, at);
        templates::copy(false, dst, at);
        self.entries[bucket] = Some(TextureTemplate {
            key,
            at,
            copy: Arc::downgrade(copy),
            view,
        });
    }
}

struct TemplateHit {
    at: D3D12_CPU_DESCRIPTOR_HANDLE,
    held: Kept,
    mark: Option<(u32, u32)>,
    binding: Binding,
    texture: Option<(Arc<HostTexture>, bool)>,
}

impl TextureTemplate {
    fn hit(&self, view: &ViewPin) -> Option<TemplateHit> {
        let (resource, srv, ambient) = match **view {
            View::Shader(resource, _) => (resource, true, true),
            View::Unordered(resource, counter, _) => (resource, false, counter == 0),
            _ => return None,
        };
        let copy = crate::resource::made_copy(resource as crate::object::This)?;
        if !std::ptr::eq(self.copy.as_ptr(), Arc::as_ptr(&copy))
            || copy.evicted.load(Ordering::Acquire)
        {
            return None;
        }
        // A gated SRV enters this cache only after it can be published.
        let source = ambient
            .then(|| self.view.for_ambient(&copy.texture))
            .flatten();
        Some(TemplateHit {
            at: self.at,
            held: Some((copy.texture.clone(), source)),
            mark: None,
            binding: Binding::Texture((srv && copy.title_written()).then_some(resource)),
            texture: Some((copy, srv)),
        })
    }
}

const SYNC_BLOCK_SLOTS: u32 = 128;

/// Whether due texture slots in [`UploadQueue`] alternate retries and fresh
/// slots 1:1 (`VICTORY_FAIR_DUE=0` restores retries-first ordering).
fn fair_due() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_FAIR_DUE").is_none_or(|v| v != "0"))
}

/// Maximum backoff interval in milliseconds for re-checking unmapped slots
/// (`VICTORY_RELOOK_MS`, default 20,000 ms).
pub(crate) fn relook_ms() -> u64 {
    static MS: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *MS.get_or_init(|| {
        std::env::var("VICTORY_RELOOK_MS")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(20_000)
    })
}

/// Backoff cap in presented frames for a frame interval `frame_ms` and relook
/// ceiling `relook_ms`.
pub(crate) fn relook_frames(frame_ms: u64, relook_ms: u64) -> u64 {
    (relook_ms / frame_ms.max(1)).clamp(16, 1024)
}

/// Classification of an unmade slot reported by GPU read-marks when the
/// background heap worker is enabled.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum UnmadeRead {
    None,
    Background,
    Sync,
}

#[derive(Clone, Copy)]
struct QueuedSlot {
    write: u64,
    due_frame: u64,
    looks: u32,
    urgent: bool,
    range: Option<(usize, usize)>,
}

/// Queue of shader-visible texture slots waiting for background creation and
/// detiling (`crate::heap_worker`).
///
/// Serves GPU read-marked (`urgent`) slots first, then alternates previously
/// retried slots (`again`, newest write first) 1:1 with never-looked-at slots
/// (`fresh`, alternating newest/oldest write) so streaming backlogs never
/// starve newly visible textures.
#[derive(Default)]
pub(crate) struct UploadQueue {
    next_write: u64,
    slots: HashMap<u32, QueuedSlot>,
}

impl UploadQueue {
    pub(crate) fn written(&mut self, index: u32, frame: u64) {
        self.next_write += 1;
        self.slots.insert(
            index,
            QueuedSlot {
                write: self.next_write,
                due_frame: frame,
                looks: 0,
                urgent: false,
                range: None,
            },
        );
    }

    pub(crate) fn prioritize(&mut self, indices: &[u32], frame: u64) {
        for &index in indices {
            match self.slots.get_mut(&index) {
                Some(entry) => {
                    entry.urgent = true;
                    entry.due_frame = entry.due_frame.min(frame);
                }
                None => {
                    self.next_write += 1;
                    self.slots.insert(
                        index,
                        QueuedSlot {
                            write: self.next_write,
                            due_frame: frame,
                            looks: 0,
                            urgent: true,
                            range: None,
                        },
                    );
                }
            }
        }
    }

    pub(crate) fn requeue(&mut self, indices: &[u32], frame: u64) {
        for &index in indices {
            if !self.slots.contains_key(&index) {
                self.next_write += 1;
                self.slots.insert(
                    index,
                    QueuedSlot {
                        write: self.next_write,
                        due_frame: frame,
                        looks: 0,
                        urgent: true,
                        range: None,
                    },
                );
            }
        }
    }

    #[allow(dead_code)] // hook for urgent bake-source pre-upload
    pub(crate) fn queued(&self, indices: &[u32]) -> Vec<(u32, u64)> {
        indices
            .iter()
            .filter_map(|&index| self.slots.get(&index).map(|w| (index, w.write)))
            .collect()
    }

    pub(crate) fn remove(&mut self, index: u32) {
        self.slots.remove(&index);
    }

    pub(crate) fn done(&mut self, index: u32, write: u64) {
        if self.slots.get(&index).is_some_and(|w| w.write == write) {
            self.slots.remove(&index);
        }
    }

    pub(crate) fn later(&mut self, index: u32, write: u64, frame: u64, max_wait: u64) {
        if let Some(entry) = self.slots.get_mut(&index)
            && entry.write == write
        {
            entry.looks = entry.looks.saturating_add(1);
            entry.urgent = false;
            let backoff = (1u64 << entry.looks.saturating_sub(1).min(10)).min(max_wait.max(1));
            entry.due_frame = frame.saturating_add(backoff);
        }
    }

    fn note_range(&mut self, index: u32, write: u64, range: (usize, usize)) {
        if let Some(entry) = self.slots.get_mut(&index)
            && entry.write == write
        {
            entry.range = Some(range);
        }
    }

    fn wake_remapped(
        &mut self,
        ranges: &std::collections::BTreeMap<usize, usize>,
        frame: u64,
    ) -> bool {
        let mut woke = false;
        for entry in self.slots.values_mut() {
            if let Some((from, to)) = entry.range
                && ranges
                    .range(..to)
                    .next_back()
                    .is_some_and(|(_, end)| *end > from)
            {
                entry.due_frame = frame;
                entry.looks = 0;
                woke = true;
            }
        }
        woke
    }

    pub(crate) fn due(&self, frame: u64, max: usize) -> Vec<(u32, u64)> {
        self.due_with(frame, max, fair_due())
    }

    pub(crate) fn due_with(&self, frame: u64, max: usize, fair: bool) -> Vec<(u32, u64)> {
        if max == 0 || self.slots.is_empty() {
            return Vec::new();
        }
        let mut urgent: Vec<(u32, u64)> = Vec::new();
        let mut again: Vec<(u32, u64)> = Vec::new();
        let mut fresh: Vec<(u32, u64)> = Vec::new();
        for (&index, w) in &self.slots {
            if w.due_frame > frame {
                continue;
            }
            if w.urgent {
                urgent.push((index, w.write));
            } else if w.looks > 0 {
                again.push((index, w.write));
            } else {
                fresh.push((index, w.write));
            }
        }
        let newest_first = |a: &(u32, u64), b: &(u32, u64)| b.1.cmp(&a.1);
        urgent.sort_unstable_by(newest_first);
        again.sort_unstable_by(newest_first);
        fresh.sort_unstable_by(newest_first);

        let mut order = Vec::with_capacity(max.min(urgent.len() + again.len() + fresh.len()));
        order.extend(urgent.into_iter().take(max));
        if order.len() >= max {
            return order;
        }
        let remaining = max - order.len();
        let mut fresh_order: Vec<(u32, u64)> = Vec::with_capacity(fresh.len().min(remaining));
        let (mut new, mut old) = (0, fresh.len());
        while new < old && fresh_order.len() < remaining {
            fresh_order.push(fresh[new]);
            new += 1;
            if new < old && fresh_order.len() < remaining {
                old -= 1;
                fresh_order.push(fresh[old]);
            }
        }
        if !fair {
            order.extend(again.into_iter().chain(fresh_order).take(remaining));
            return order;
        }
        let mut again_iter = again.into_iter();
        let mut fresh_iter = fresh_order.into_iter();
        while order.len() < max {
            let (a, b) = (again_iter.next(), fresh_iter.next());
            if a.is_none() && b.is_none() {
                break;
            }
            order.extend(a);
            if order.len() < max {
                order.extend(b);
            }
        }
        order
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }
}


/// Pre-populated null descriptor templates in [`StagingHeap`] (slot 0 for
/// sampler heaps; slots `0..17` covering every null SRV/UAV kind for view
/// heaps). Remaining slots hold per-batch templates for repeated views.
const STAGING_NULL_SLOTS: u32 = 17;
const STAGING_TOTAL_SLOTS: u32 = 256;

fn fast_heap_sync_value(value: Option<&std::ffi::OsStr>) -> bool {
    value != Some(std::ffi::OsStr::new("0"))
}

fn fast_heap_sync() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| fast_heap_sync_value(std::env::var_os("VICTORY_FAST_HEAP_SYNC").as_deref()))
}

/// CPU-only (`D3D12_DESCRIPTOR_HEAP_FLAG_NONE`) staging heap holding null
/// templates (`0..STAGING_NULL_SLOTS`) and scratch template slots for
/// `CopyDescriptorsSimple` during batched heap syncs.
struct StagingHeap {
    _heap: ID3D12DescriptorHeap,
    start: usize,
    step: u32,
    kind: D3D12_DESCRIPTOR_HEAP_TYPE,
    count: u32,
}

impl StagingHeap {
    fn new(sampler: bool, step: u32) -> Option<Self> {
        let host = host()?;
        let kind = if sampler {
            D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER
        } else {
            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV
        };
        // SAFETY: CPU-only staging descriptor heap; valid source for
        // `CopyDescriptorsSimple` into shader-visible heaps.
        let heap: ID3D12DescriptorHeap = unsafe {
            host.device
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: kind,
                    NumDescriptors: STAGING_TOTAL_SLOTS,
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
                    NodeMask: 0,
                })
        }
        .ok()?;
        // SAFETY: live heap.
        let start = unsafe { heap.GetCPUDescriptorHandleForHeapStart() }.ptr;
        let staging = Self {
            _heap: heap,
            start,
            step,
            kind,
            count: STAGING_TOTAL_SLOTS,
        };
        if sampler {
            null_sampler(staging.cpu(0));
        } else {
            null_view(staging.cpu(0));
            null_buffer_view(staging.cpu(1));
            for dim in 2..=10u32 {
                null_texture(dim, DXGI_FORMAT_R8G8B8A8_UNORM, staging.cpu(dim));
            }
            let buf_uav = D3D12_UNORDERED_ACCESS_VIEW_DESC {
                Format: DXGI_FORMAT_R32_UINT,
                ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
                ..Default::default()
            };
            let uav_1d = null_uav_desc(Some(D3D12_UAV_DIMENSION_TEXTURE1D.0 as u32));
            let uav_1d_arr = null_uav_desc(Some(D3D12_UAV_DIMENSION_TEXTURE1DARRAY.0 as u32));
            let uav_2d_arr = null_uav_desc(Some(D3D12_UAV_DIMENSION_TEXTURE2DARRAY.0 as u32));
            let uav_3d = null_uav_desc(Some(D3D12_UAV_DIMENSION_TEXTURE3D.0 as u32));
            let uav_2d = null_uav_desc(None);
            // SAFETY: null resources with explicit descriptions in our staging heap.
            unsafe {
                host.device.CreateUnorderedAccessView(
                    None::<&ID3D12Resource>,
                    None::<&ID3D12Resource>,
                    Some(&buf_uav),
                    staging.cpu(11),
                );
                host.device.CreateUnorderedAccessView(
                    None::<&ID3D12Resource>,
                    None::<&ID3D12Resource>,
                    Some(&uav_1d),
                    staging.cpu(12),
                );
                host.device.CreateUnorderedAccessView(
                    None::<&ID3D12Resource>,
                    None::<&ID3D12Resource>,
                    Some(&uav_1d_arr),
                    staging.cpu(13),
                );
                host.device.CreateUnorderedAccessView(
                    None::<&ID3D12Resource>,
                    None::<&ID3D12Resource>,
                    Some(&uav_2d_arr),
                    staging.cpu(14),
                );
                host.device.CreateUnorderedAccessView(
                    None::<&ID3D12Resource>,
                    None::<&ID3D12Resource>,
                    Some(&uav_3d),
                    staging.cpu(15),
                );
                host.device.CreateUnorderedAccessView(
                    None::<&ID3D12Resource>,
                    None::<&ID3D12Resource>,
                    Some(&uav_2d),
                    staging.cpu(16),
                );
            }
        }
        Some(staging)
    }

    fn cpu(&self, index: u32) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: self.start + index as usize * self.step as usize,
        }
    }
}

/// Slot index in [`StagingHeap`] matching [`null_for`] / [`null_view`] /
/// [`null_sampler`].
fn null_slot_index(sampler: bool, view: Option<&View>) -> u32 {
    if sampler {
        return 0;
    }
    let Some(view) = view else {
        return 0;
    };
    let is_buffer = |r: &usize| crate::resource::buffer_bytes(*r as crate::object::This).is_some();
    let (buffer, dimension) = match view {
        View::Unordered(r, _, desc) => (
            is_buffer(r) || desc.is_some_and(|d| d[1] == 1),
            desc.map(|d| d[1]),
        ),
        View::Placed(true, resource, desc) => (resource[0] == 1, Some(desc[1])),
        View::Shader(r, desc) if is_buffer(r) || desc.is_some_and(|d| d[1] == 1) => return 1,
        View::Placed(false, resource, _) if resource[0] == 1 => return 1,
        View::Shader(_, Some(desc)) => {
            return if (2..=10).contains(&desc[1]) { desc[1] } else { 0 };
        }
        View::Placed(false, _, desc) => {
            return if (2..=10).contains(&desc[1]) { desc[1] } else { 0 };
        }
        View::Constant(..) => return 1,
        _ => return 0,
    };
    if buffer {
        return 11;
    }
    match dimension.map(|d| D3D12_UAV_DIMENSION(d as i32)) {
        Some(D3D12_UAV_DIMENSION_TEXTURE1D) => 12,
        Some(D3D12_UAV_DIMENSION_TEXTURE1DARRAY) => 13,
        Some(D3D12_UAV_DIMENSION_TEXTURE2DARRAY) => 14,
        Some(D3D12_UAV_DIMENSION_TEXTURE3D) => 15,
        _ => 16,
    }
}
fn sync_fast_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_SYNC_FAST").as_deref() != Some(std::ffi::OsStr::new("0")))
}

/// Monotonic identity for each [`HostHeap`] so stale dynamic-slot releases
/// targeting a released heap's recycled address never free a newer heap's slots.
static NEXT_ID: AtomicU64 = AtomicU64::new(1);

pub(crate) struct HostHeap {
    id: u64,
    pub(crate) heap: ID3D12DescriptorHeap,
    cpu: usize,
    step: u32,
    /// Slots mirroring the title's; view heaps have `dynamic` more.
    count: u32,
    /// Dynamic slots past `count` (none in sampler heaps): [`DYNAMIC`]
    /// where the host takes them, else [`MIN_DYNAMIC`].
    dynamic: u32,
    sampler: bool,
    /// The record (view index and generation) each slot was last made from.
    made: Mutex<Mirrored>,
    /// Published after host metadata, including intentionally null views.
    synced: Box<[AtomicU64]>,
    buffer_generations: Mutex<HashMap<u32, u64>>,
    /// One bit per slot: whether `made[index] != 0`, so empty slots skip
    /// `made` and `VIEWS` locks during `sync_slots`.
    made_nonzero: Box<[AtomicU64]>,
    /// Per `SYNC_BLOCK_SLOTS` block, the `views_forgot()` epoch last verified.
    block_forgot: Box<[AtomicU64]>,
    /// Per title slot, its record once the slot is made from it and holds
    /// what a draw reads (a texture's view not left null), else 0: what
    /// [`HostHeap::mirror`] finds without the `made` lock, which every
    /// title thread binding or writing views would otherwise wait on. Set
    /// last, under that lock, once the slot is made; cleared first.
    ready: Box<[AtomicU64]>,
    /// CPU-only staging heap for null and repeated-view template copies during
    /// batched heap syncs, created on first use.
    staging: std::sync::OnceLock<Option<StagingHeap>>,
    /// The host resource each slot views, held while the slot does: shaders
    /// index heaps directly, so a slot the title has not rewritten since it
    /// released the resource may still be read, and a freed host resource
    /// read there is a GPU page fault (the console reads stale memory).
    /// With it the view's own copy of title memory, if it has one. Paged:
    /// the title views through a fraction of the slots, and a dense table
    /// of them all was 37.7 MB, written when the heap was made.
    kept: Mutex<Paged<Held>>,
    /// Write-back observers of shared slots. A descriptor rewrite retires
    /// its observer before changing the GPU-visible descriptor.
    texture_bindings:
        Mutex<std::collections::HashMap<u32, std::sync::Weak<crate::texback::Binding>>>,
    has_texture_bindings: AtomicBool,
    /// Whether any recorded ambient pass has watched a slot of this heap.
    ambient_watched: AtomicBool,
    /// Unordered views of textures left null by [`HostHeap::evict`]
    /// (`VICTORY_EVICT_HEAP_UAV`), made again by
    /// [`HostHeap::remake_unordered`] once shaders write through heaps.
    nulled_unordered: Mutex<Vec<u32>>,
    /// Where the search for free dynamic slots starts, and which are taken
    /// (by lists not yet reset), a bit for each.
    next: Mutex<(u32, Vec<u64>)>,
    /// Slot pairs kept for host copies' views (`crate::memory`), past the
    /// dynamic slots and below the null buffer views: the next never
    /// taken, those given back, and where they end.
    pairs: Mutex<(u32, Vec<u32>)>,
    pairs_end: u32,
    /// Null raw SRV for explicitly bound root descriptors. Its UAV is one slot below.
    null_root_srv: Option<u32>,
    /// View heaps: the view-stride table translated shaders read
    /// (`victory_shader::emit::VIEW_STRIDES`), a dword per title slot: the
    /// byte stride of the buffer view there, 0 for none. Mapped at the
    /// address kept beside it.
    strides: Option<(ID3D12Resource, usize)>,
    /// View heaps: the view-mark table translated shaders read
    /// (`victory_shader::emit`, "Write marks"), two dwords per slot as the
    /// stride table has one: the mark bit and phase of the unordered buffer
    /// view there, `VIEW_MARK_NONE` for none. Mapped at the address kept
    /// beside it.
    marks: Option<(ID3D12Resource, usize)>,
    /// Per title slot and per dynamic slot (tables made afresh lie there,
    /// and shaders index the view addresses at the slots their tables are
    /// in), the console descriptor's dwords 0-3 as far as the host knows
    /// them (shaders reading a descriptor as data, `crate::memory`;
    /// [`write_address`]): the buffer, its mapping and host address, made on
    /// first use.
    addresses: std::sync::OnceLock<Option<(ID3D12Resource, usize, u64)>>,
    /// What the title slots view that title memory feeds (`crate::bindless`).
    pub(crate) viewed: std::sync::Arc<crate::bindless::Viewed>,
    /// Live widened SRV buffers, by host address, with the number of slots
    /// showing each. Heap-indexing shaders can read any of them.
    widened: Mutex<HashMap<usize, (usize, std::sync::Weak<crate::shadow::Widened>)>>,
    /// Slots queued for background texture upload (`crate::heap_worker`).
    uploads: Mutex<UploadQueue>,
}

/// Templates for a list's fresh tables. CopyDescriptorsSimple requires a
/// CPU-only source heap, and each destination still keeps its backing alive.
pub(crate) struct ListViews {
    _heap: ID3D12DescriptorHeap,
    cpu: usize,
    step: usize,
    made: std::collections::HashMap<[u32; 3], ListedView>,
}

struct ListedView {
    at: D3D12_CPU_DESCRIPTOR_HANDLE,
    held: Held,
    stride: u32,
    mark: Option<(u32, u32)>,
}

impl ListViews {
    const CAPACITY: u32 = 512;

    pub(crate) fn new() -> Option<Self> {
        let host = host()?;
        // SAFETY: a CPU-only heap of a supported descriptor type.
        let (heap, cpu, step) = unsafe {
            let kind = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
            let heap: ID3D12DescriptorHeap = host
                .device
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: kind,
                    NumDescriptors: Self::CAPACITY,
                    ..Default::default()
                })
                .ok()?;
            let cpu = heap.GetCPUDescriptorHandleForHeapStart().ptr;
            let step = host.device.GetDescriptorHandleIncrementSize(kind) as usize;
            (heap, cpu, step)
        };
        Some(Self {
            _heap: heap,
            cpu,
            step,
            made: Default::default(),
        })
    }

    /// Only ordinary buffer resources are eligible. An address-only view
    /// can acquire a different memory backing during a long recording.
    pub(crate) fn can_cache(view: &View) -> bool {
        match view {
            View::Shader(resource, Some(words)) | View::Unordered(resource, _, Some(words)) => {
                words[1] == 1
                    && crate::resource::buffer_bytes(*resource as crate::object::This).is_some()
            }
            _ => false,
        }
    }

    pub(crate) fn make(
        &mut self,
        heap: &HostHeap,
        index: u32,
        record: [u32; 3],
        view: &View,
        backing: &std::sync::Arc<crate::shadow::Shadow>,
    ) -> bool {
        let Some(host) = host() else { return false };
        let count = self.made.len();
        let template = self.made.get(&record);
        let matches = template
            .and_then(|template| held_backing(&template.held))
            .is_some_and(|cached| std::sync::Arc::ptr_eq(cached, backing));
        if !matches {
            let at = match template {
                Some(template) => template.at,
                None if count < Self::CAPACITY as usize => D3D12_CPU_DESCRIPTOR_HANDLE {
                    ptr: self.cpu + count * self.step,
                },
                None => return false,
            };
            let (kept, mark) = make_marked_with_backing(view, at, Some(backing));
            let Some(held) = kept.flatten() else {
                return false;
            };
            self.made.insert(
                record,
                ListedView {
                    at,
                    held,
                    stride: view_stride(view) | view_kind(view),
                    mark,
                },
            );
        }
        let template = &self.made[&record];
        // SAFETY: initialized CPU-only source, live destination owned by
        // this list. The slot and its mark/stride are private to the list.
        unsafe {
            host.device.CopyDescriptorsSimple(
                1,
                heap.cpu(index),
                template.at,
                D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
            );
        }
        heap.keep(index, Some(template.held.clone()));
        heap.write_stride(index, template.stride);
        heap.set_mark(index, template.mark);
        heap.note_address(index, Some(view));
        true
    }
}

/// Host heaps' GPU ranges (start, bytes, bytes per slot, sampler), for
/// device-removed reports: a shader indexing a heap past its end faults.
static GPU_RANGES: Mutex<Vec<(u64, u64, u64, bool)>> = Mutex::new(Vec::new());

/// For device-removed reports: where host GPU address `va` lies against
/// the host descriptor heaps.
pub(crate) fn near_heaps(va: u64) -> String {
    let ranges = GPU_RANGES.lock().unwrap_or_else(|e| e.into_inner());
    let mut text = String::new();
    for &(start, bytes, step, sampler) in ranges.iter() {
        let kind = if sampler { "sampler" } else { "view" };
        let place = if va < start {
            format!("{:#x} below", start - va)
        } else if va < start + bytes {
            format!("inside, slot {}", (va - start) / step)
        } else {
            format!(
                "{:#x} past its end (slot {})",
                va - start - bytes,
                (va - start) / step
            )
        };
        text.push_str(&format!("; {kind} heap {start:#x}+{bytes:#x}: {place}"));
    }
    text
}

/// The host's limit for shader-visible view heaps on resource binding
/// tiers 1 and 2; tier 3 hardware may take more.
const PORTABLE_VIEWS: u32 = 1_000_000;

/// Slots past the title's in a view heap, for tables a draw needs made
/// afresh (constant buffers copied per draw). A list holds the slots it
/// took until it is reset (its submissions have run) or released.
/// 2^16 ran out seconds into the world (every world run on an RX 9070 XT,
/// each a device removal shortly after; #59).
const DYNAMIC: u32 = 1 << 18;

/// The dynamic slots of a heap whose host would not take [`DYNAMIC`] more
/// (a capped heap, NVIDIA): the title's slots keep their room first.
const MIN_DYNAMIC: u32 = 1 << 16;

impl HostHeap {
    pub(crate) fn new(sampler: bool, count: u32) -> Option<Self> {
        let host = host()?;
        let kind = if sampler {
            D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER
        } else {
            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV
        };
        let make = |count: u32| -> windows::core::Result<ID3D12DescriptorHeap> {
            // SAFETY: plain heap creation.
            unsafe {
                host.device
                    .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                        Type: kind,
                        NumDescriptors: count,
                        Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                        NodeMask: 0,
                    })
            }
        };
        // Every slot a shader can index (title slots, also past a capped
        // heap, and dynamic ones): shaders read the table as a root view,
        // which has no bounds, at indices clamped below `HEAP_VIEWS`.
        let strides = (!sampler)
            .then(|| host.upload_buffer(u64::from(count.max(victory_shader::emit::HEAP_VIEWS)) * 4))
            .flatten()
            .map(|(buffer, data)| (buffer, data as usize));
        let marks = (!sampler)
            .then(|| {
                let bytes = u64::from(count.max(victory_shader::emit::HEAP_VIEWS)) * 8;
                let (buffer, data) = host.upload_buffer(bytes)?;
                crate::host::name(&buffer, || "view marks".into());
                // Every slot unmarked (all ones) until a view is made there.
                // SAFETY: our mapped buffer of `bytes` bytes.
                unsafe { std::ptr::write_bytes(data, 0xFF, bytes as usize) };
                Some((buffer, data as usize))
            })
            .flatten();
        // Other threads' submissions held off from here until the heap is
        // filled (`Host::quiet`; a title heap's `host` holds them already).
        let _quiet = (!sampler).then(|| host.quiet()).flatten();
        // The larger dynamic range only where the host takes it on top of
        // every title slot; else the smaller one, and the cap after that.
        // Translated shaders clamp their indices below `HEAP_VIEWS` or
        // `HEAP_SAMPLERS`: slots up to there exist and stay null.
        let (views, samplers) = (
            victory_shader::emit::HEAP_VIEWS,
            victory_shader::emit::HEAP_SAMPLERS,
        );
        let (heap, count, extra) = if sampler {
            match make(count.max(samplers)) {
                Ok(heap) => (heap, count, 0),
                Err(e) => {
                    tracing::warn!(error = %e, count, sampler, "no host descriptor heap");
                    return None;
                }
            }
        } else if let Ok(heap) = make((count + DYNAMIC).max(views)) {
            (heap, count, DYNAMIC)
        } else {
            // shortcut: a host that caps its heaps below `HEAP_VIEWS` (NVIDIA)
            // faults on an index past them, as before the clamp.
            match make(count + MIN_DYNAMIC) {
                Ok(heap) => (heap, count, MIN_DYNAMIC),
                Err(e) if count + MIN_DYNAMIC > PORTABLE_VIEWS => {
                    // shortcut: descriptors past the host's limit are left
                    // out; a title that reaches them needs remapping.
                    tracing::warn!(error = %e, count, "host view heap capped at {PORTABLE_VIEWS}");
                    (
                        make(PORTABLE_VIEWS).ok()?,
                        PORTABLE_VIEWS - MIN_DYNAMIC,
                        MIN_DYNAMIC,
                    )
                }
                Err(e) => {
                    tracing::warn!(error = %e, count, sampler, "no host descriptor heap");
                    return None;
                }
            }
        };
        // SAFETY: a live heap.
        let (cpu, step) = unsafe {
            (
                heap.GetCPUDescriptorHandleForHeapStart().ptr,
                host.device.GetDescriptorHandleIncrementSize(kind),
            )
        };
        tracing::debug!(count, sampler, "host descriptor heap");
        // SAFETY: a live shader-visible heap.
        let (gpu, total) = unsafe {
            (
                heap.GetGPUDescriptorHandleForHeapStart().ptr,
                heap.GetDesc().NumDescriptors,
            )
        };
        GPU_RANGES.lock().unwrap_or_else(|e| e.into_inner()).push((
            gpu,
            u64::from(total) * u64::from(step),
            u64::from(step),
            sampler,
        ));
        let mut h = HostHeap {
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
            heap,
            cpu,
            step,
            count,
            dynamic: extra,
            sampler,
            made: Mutex::new(Mirrored {
                keys: vec![0; count as usize],
                pending: BTreeMap::new(),
                templates: std::array::from_fn(|_| OnceLock::new()),
                textures: std::array::from_fn(|_| OnceLock::new()),
            }),
            synced: (0..count).map(|_| AtomicU64::new(0)).collect(),
            buffer_generations: Mutex::default(),
            made_nonzero: (0..(count as usize).div_ceil(64))
                .map(|_| AtomicU64::new(0))
                .collect(),
            block_forgot: (0..(count as usize).div_ceil(SYNC_BLOCK_SLOTS as usize))
                .map(|_| AtomicU64::new(0))
                .collect(),
            ready: (0..count).map(|_| AtomicU64::new(0)).collect(),
            staging: std::sync::OnceLock::new(),
            kept: Mutex::new(Paged::new((count + extra) as usize)),
            texture_bindings: Mutex::default(),
            has_texture_bindings: AtomicBool::new(false),
            ambient_watched: AtomicBool::new(false),
            nulled_unordered: Mutex::new(Vec::new()),
            next: Mutex::new((0, vec![0; (extra as usize).div_ceil(64)])),
            pairs: Mutex::new((count + extra, Vec::new())),
            null_root_srv: None,
            pairs_end: match victory_shader::emit::NULL_BUFFER_VIEW {
                null if !sampler && null > count + extra && null < total => null - 1,
                _ => total,
            },
            strides,
            marks,
            addresses: std::sync::OnceLock::new(),
            viewed: Arc::new(crate::bindless::Viewed::new(if sampler {
                0
            } else {
                count
            })),
            widened: Default::default(),
            uploads: Default::default(),
        };
        if !sampler {
            crate::bindless::register(&h.viewed);
        }
        // A slot never written is undefined: shaders reading one (a slot
        // the title wrote in a way we do not know, or a table entry no
        // used-resource entry names) read a wild address. Every slot starts
        // null instead.
        let started = std::time::Instant::now();
        if let Err(e) = h.fill_null(kind, total) {
            tracing::warn!(error = %e, "host descriptor heap not made null");
        }
        tracing::debug!(
            ms = started.elapsed().as_millis(),
            "host descriptor heap made null"
        );
        // What shaders read instead of a slot without a buffer view where
        // they take one at a computed index (`victory_shader::emit`, "Buffer
        // views at computed heap indices"): a null raw SRV, and UAV below it.
        if !sampler {
            let fixed = victory_shader::emit::NULL_BUFFER_VIEW;
            let null = if fixed > count + extra && fixed < total {
                Some(fixed)
            } else {
                tracing::warn!(
                    count,
                    total,
                    "no room for the null buffer views: computed indices of slots without a buffer view read what is there"
                );
                // Root descriptors receive a heap index at bind time and can
                // use a pair inside a capped heap. Keep it for the heap's life.
                h.pair().map(|first| first + 1)
            };
            if let Some(srv) = null {
                h.raw_view_at(srv, None, 0, 0, false);
                h.raw_view_at(srv - 1, None, 0, 0, true);
                h.null_root_srv = Some(srv);
            }
        }
        Some(h)
    }

    pub(crate) fn null_root_view(&self, uav: bool) -> Option<u32> {
        self.null_root_srv.map(|srv| srv - u32::from(uav))
    }

    /// Makes the first `total` slots null: one null slot in a CPU-side heap,
    /// doubled by copies to a block, the block copied over the heap.
    fn fill_null(&self, kind: D3D12_DESCRIPTOR_HEAP_TYPE, total: u32) -> windows::core::Result<()> {
        const BLOCK: u32 = 1 << 16;
        let Some(host) = host() else { return Ok(()) };
        // SAFETY: plain heap creation and copies inside live heaps.
        unsafe {
            let block: ID3D12DescriptorHeap =
                host.device
                    .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                        Type: kind,
                        NumDescriptors: BLOCK,
                        ..Default::default()
                    })?;
            let start = block.GetCPUDescriptorHandleForHeapStart();
            let at = |i: u32| D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: start.ptr + (i * self.step) as usize,
            };
            if self.sampler {
                null_sampler(start);
            } else {
                null_view(start);
            }
            let mut filled = 1;
            while filled < BLOCK {
                let n = filled.min(BLOCK - filled);
                host.device
                    .CopyDescriptorsSimple(n, at(filled), start, kind);
                filled += n;
            }
            let mut done = 0;
            while done < total {
                let n = BLOCK.min(total - done);
                host.device
                    .CopyDescriptorsSimple(n, self.cpu(done), start, kind);
                done += n;
            }
        }
        Ok(())
    }

    /// A plain point sampler at slot `index`, for sampler slots without a
    /// view.
    fn null_sampler(&self, index: u32) {
        null_sampler(self.cpu(index));
    }
}

/// A plain point sampler at `at`.
fn null_sampler(at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let desc = D3D12_SAMPLER_DESC {
        Filter: D3D12_FILTER_MIN_MAG_MIP_POINT,
        AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        MaxAnisotropy: 1,
        ComparisonFunc: D3D12_COMPARISON_FUNC_NEVER,
        ..Default::default()
    };
    // SAFETY: a slot of a sampler heap of ours.
    unsafe { host.device.CreateSampler(&desc, at) };
}

/// A sampler heap of our own, for the lists that set a root signature whose
/// sampler heap is indexed directly (`SAMPLER_HEAP_DIRECTLY_INDEXED`) while
/// the title binds no sampler heap: D3D12 requires one to be bound first
/// (the debug layer's id 1419), and breaking that rule removes the device on
/// NVIDIA. Slot 0 holds a point sampler, so an indexed read of it returns
/// something defined rather than whatever the memory held.
pub(crate) fn fallback_samplers() -> Option<&'static HostHeap> {
    static HEAP: std::sync::OnceLock<Option<HostHeap>> = std::sync::OnceLock::new();
    HEAP.get_or_init(|| {
        let heap = HostHeap::new(true, 1)?;
        null_sampler(heap.cpu(0));
        Some(heap)
    })
    .as_ref()
}

/// The first of `n` clear bits in a row within `[from, to)` of `taken`
/// (a set of slots, a bit each), if there are any. A word's bits are
/// looked at together.
fn free_run(taken: &[u64], from: usize, to: usize, n: usize) -> Option<usize> {
    let (mut at, mut run) = (from, 0);
    while at < to {
        let room = (64 - at % 64).min(to - at);
        let window = u64::MAX >> (64 - room);
        // Bit p: slot `at + p` is clear.
        let free = !(taken[at / 64] >> (at % 64)) & window;
        // The clear slots from `at` on, which go on a run begun before `at`.
        let lead = ((!free).trailing_zeros() as usize).min(room);
        if run + lead >= n {
            return Some(at - run);
        }
        if n <= 64 {
            let fits = starts(free, n);
            if fits != 0 {
                return Some(at + fits.trailing_zeros() as usize);
            }
        }
        if lead == room {
            run += room;
        } else {
            run = (free << (64 - room)).leading_ones() as usize;
        }
        at += room;
    }
    None
}

/// Bit p set when bits p to p + n - 1 of `free` are all set (n <= 64):
/// runs of 1, 2, 4... bits are joined until they make n.
fn starts(free: u64, n: usize) -> u64 {
    let (mut acc, mut len, mut cur, mut pow, mut rem) = (u64::MAX, 0usize, free, 1usize, n);
    while rem > 0 {
        if rem & 1 == 1 {
            acc = if len == 0 { cur } else { acc & (cur >> len) };
            len += pow;
        }
        rem >>= 1;
        if rem > 0 {
            cur &= cur >> pow;
            pow *= 2;
        }
    }
    acc
}

/// Sets (`on`) or clears the bits of `taken` from `from` to `to`.
fn set_taken(taken: &mut [u64], from: usize, to: usize, on: bool) {
    let mut at = from;
    while at < to {
        let (bit, room) = (at % 64, (64 - at % 64).min(to - at));
        let mask = (u64::MAX >> (64 - room)) << bit;
        if on {
            taken[at / 64] |= mask;
        } else {
            taken[at / 64] &= !mask;
        }
        at += room;
    }
}

impl HostHeap {
    /// `n` consecutive free dynamic slots (view heaps only), taken until
    /// [`HostHeap::release`]: the first's index.
    pub(crate) fn dynamic(&self, n: u32) -> Option<u32> {
        if self.sampler || n == 0 || n > self.dynamic {
            return None;
        }
        let mut guard = self.next.lock().unwrap_or_else(|e| e.into_inner());
        let (next, taken) = &mut *guard;
        let n = n as usize;
        // Search from `next` on, then from the start (extending past cursor by
        // n - 1 so contiguous runs crossing the cursor are not missed).
        let cursor = *next as usize;
        let end = self.dynamic as usize;
        let second_end = cursor.saturating_add(n.saturating_sub(1)).min(end);
        let Some(first) =
            free_run(taken, cursor, end, n).or_else(|| free_run(taken, 0, second_end, n))
        else {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| {
                tracing::warn!(n, "no free dynamic descriptor slots: a table is not made")
            });
            return None;
        };
        set_taken(taken, first, first + n, true);
        *next = ((first + n) % self.dynamic as usize) as u32;
        Some(self.count + first as u32)
    }

    /// Two consecutive slots (view heaps only) kept until
    /// [`HostHeap::free_pair`]: past the dynamic slots where the heap has
    /// room (a host copy's views, `crate::memory`, for as long as it
    /// lives), else dynamic ones.
    pub(crate) fn pair(&self) -> Option<u32> {
        if self.sampler {
            return None;
        }
        {
            let mut pairs = self.pairs.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(index) = pairs.1.pop() {
                return Some(index);
            }
            if pairs.0 + 2 <= self.pairs_end {
                pairs.0 += 2;
                return Some(pairs.0 - 2);
            }
        }
        self.dynamic(2)
    }

    /// Gives back a pair [`HostHeap::pair`] returned.
    pub(crate) fn free_pair(&self, index: u32) {
        if index >= self.count + self.dynamic {
            self.pairs
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .1
                .push(index);
        } else {
            self.release(self.id, index, 2);
        }
    }

    /// The title slots this heap has host slots for; fewer than the title's
    /// when the host caps its size.
    pub(crate) fn count(&self) -> u32 {
        self.count
    }

    /// This heap's unique identity ([`NEXT_ID`]): dynamic slots are returned
    /// only when the caller's identity matches this heap.
    pub(crate) fn id(&self) -> u64 {
        self.id
    }

    /// Gives back `n` dynamic slots from `first` (an index [`dynamic`]
    /// returned), only if `id` matches this heap's [`Self::id`].
    ///
    /// [`dynamic`]: HostHeap::dynamic
    pub(crate) fn release(&self, id: u64, first: u32, n: u32) {
        if id != self.id {
            return;
        }
        let Some(at) = first.checked_sub(self.count) else {
            return;
        };
        let mut guard = self.next.lock().unwrap_or_else(|e| e.into_inner());
        let end = (at + n).min(self.dynamic) as usize;
        if (at as usize) < end {
            // A computed title index may still reach the vacant descriptor
            // below the shader's clamp. Keep its entire backing, including
            // any mark owner, until the next recorder replaces it. Avoid
            // descriptor/metadata writes while another list can use the heap.
            let unreachable = first.max(victory_shader::emit::HEAP_VIEWS) as usize;
            let limit = self.count as usize + end;
            let retired: Vec<_> = if unreachable < limit {
                let mut kept = self.kept.lock().unwrap_or_else(|e| e.into_inner());
                (unreachable..limit)
                    .filter_map(|index| kept.slot(index, false)?.take())
                    .collect()
            } else {
                Vec::new()
            };
            set_taken(&mut guard.1, at as usize, end, false);
            drop(guard);
            // Enqueue complete backing ownership outside both locks.
            for held in retired {
                if let Some(HeldSource::Widened(view, _)) = &held.1 {
                    self.note_widened(Some(view), false);
                }
                crate::host::retire(held);
            }
        }
    }

    /// Makes slot `index` (a dynamic one) from `view`; a null view if the
    /// host cannot.
    pub(crate) fn make_at(&self, index: u32, view: &View) {
        self.make_at_with_backing(index, view, None);
    }

    /// Build a fresh-table view from the same backing its draw will refresh.
    pub(crate) fn make_at_with_backing(
        &self,
        index: u32,
        view: &View,
        backing: Option<&std::sync::Arc<crate::shadow::Shadow>>,
    ) {
        let view_is_sampler = is_sampler(view);
        if view_is_sampler != self.sampler {
            warn_heap_mismatch(index, self.sampler, view_is_sampler);
            self.make_null(index);
            return;
        }
        self.invalidate_ambient(index);
        let (kept, mark) = make_marked_with_backing(view, self.cpu(index), backing);
        if kept.is_none() {
            null_for(view, self.cpu(index));
        }
        self.keep(index, kept.flatten());
        self.set_stride(index, Some(view));
        self.set_mark(index, mark);
        // A table made afresh: shaders reading a view's address there (e.g.
        // the BVH builder's base) index the view addresses at this slot.
        self.note_address(index, Some(view));
    }

    /// The backing of a buffer in a reused fresh table. The slot owns the
    /// exact shadow its descriptor names, including a widened view's source.
    pub(crate) fn table_backing(
        &self,
        index: u32,
    ) -> Option<std::sync::Arc<crate::shadow::Shadow>> {
        let kept = self.kept.lock().unwrap_or_else(|e| e.into_inner());
        held_backing(kept.get(index as usize)?).cloned()
    }

    /// Makes slot `index` (a dynamic one) a raw view, shader resource or
    /// unordered access (`uav`), of `bytes` of `buffer` from byte `first`
    /// (both multiples of 4), which it holds; a null one for no buffer, or
    /// for unordered access to a buffer shaders cannot write.
    pub(crate) fn raw_view_at(
        &self,
        index: u32,
        buffer: Option<&ID3D12Resource>,
        first: u64,
        bytes: u64,
        uav: bool,
    ) {
        self.raw_view(index, buffer, first, raw_elements(bytes, false), uav);
    }

    /// A root view indexed by an offset from the start of the shadow.
    pub(crate) fn raw_shadow_at(&self, index: u32, shadow: &crate::shadow::Shadow, uav: bool) {
        let elements = raw_elements(shadow.range().1, shadow.is_tiled());
        self.raw_view(index, Some(&shadow.buffer), 0, elements, uav);
    }

    fn raw_view(
        &self,
        index: u32,
        buffer: Option<&ID3D12Resource>,
        first: u64,
        elements: u32,
        uav: bool,
    ) {
        self.invalidate_ambient(index);
        let Some(host) = host() else { return };
        let writable = |b: &ID3D12Resource| {
            // SAFETY: a live buffer.
            unsafe { b.GetDesc() }.Flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 != 0
        };
        let buffer = buffer.filter(|&b| !uav || writable(b));
        let (first, elements) = match buffer {
            Some(_) => (first / 4, elements.max(1)),
            // A null view still names its size.
            None => (0, 1),
        };
        // SAFETY: a slot of our heap; the buffer is live (or none: a null view).
        unsafe {
            if uav {
                let desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
                    Format: DXGI_FORMAT_R32_TYPELESS,
                    ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
                    Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                        Buffer: D3D12_BUFFER_UAV {
                            FirstElement: first,
                            NumElements: elements,
                            Flags: D3D12_BUFFER_UAV_FLAG_RAW,
                            ..Default::default()
                        },
                    },
                };
                host.device
                    .CreateUnorderedAccessView(buffer, None, Some(&desc), self.cpu(index));
            } else {
                let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
                    Format: DXGI_FORMAT_R32_TYPELESS,
                    ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
                    Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                    Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                        Buffer: D3D12_BUFFER_SRV {
                            FirstElement: first,
                            NumElements: elements,
                            StructureByteStride: 0,
                            Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                        },
                    },
                };
                host.device
                    .CreateShaderResourceView(buffer, Some(&desc), self.cpu(index));
            }
        }
        self.keep(index, buffer.map(|b| (b.clone(), None)));
        self.write_stride(index, 4);
        self.set_mark(index, None);
        // Not a title view: no entry an earlier table left there stays.
        self.note_address(index, None);
    }

    /// Makes slot `index` read nothing: a null view, or a plain sampler.
    pub(crate) fn make_null(&self, index: u32) {
        self.clear_descriptor(index, None);
        if index < self.count {
            self.viewed.set(index, None, false);
        }
    }

    fn clear_descriptor(&self, index: u32, template: Option<D3D12_CPU_DESCRIPTOR_HANDLE>) {
        self.clear_slot_metadata(index);
        if let Some(src) = template {
            templates::copy(self.sampler, self.cpu(index), src);
        } else if self.sampler {
            self.null_sampler(index);
        } else {
            null_view(self.cpu(index));
        }
        self.keep(index, None);
    }

    fn clear_slot_metadata(&self, index: u32) {
        self.invalidate_ambient(index);
        self.invalidate_texture_binding(index);
        self.set_stride(index, None);
        self.set_mark(index, None);
        self.note_address(index, None);
    }

    /// Enters the byte stride of `view` (0 for none or not a buffer view)
    /// for slot `index` in the view-stride table, with the integer-format
    /// bits of a typed buffer view ([`view_kind`]).
    fn set_stride(&self, index: u32, view: Option<&View>) {
        self.write_stride(index, view.map_or(0, |v| view_stride(v) | view_kind(v)));
    }

    fn write_stride(&self, index: u32, stride: u32) {
        let Some((_, data)) = &self.strides else {
            return;
        };
        if index >= self.count.max(victory_shader::emit::HEAP_VIEWS) {
            return;
        }
        // SAFETY: a dword of the mapped table, which has one per slot below
        // the larger of the title's count and `HEAP_VIEWS`.
        unsafe {
            (*data as *mut u32)
                .add(index as usize)
                .write_volatile(stride)
        };
    }

    /// The view-stride table, for root SRV `VIEW_STRIDES`.
    pub(crate) fn strides(&self) -> Option<&ID3D12Resource> {
        self.strides.as_ref().map(|(buffer, _)| buffer)
    }

    /// Enters slot `index`'s view-mark entry: the mark bit and phase of the
    /// unordered buffer view there ([`crate::shadow::Shadow::mark_entry`]),
    /// none for anything else.
    pub(crate) fn set_mark(&self, index: u32, entry: Option<(u32, u32)>) {
        let Some((_, data)) = &self.marks else { return };
        if index >= self.count.max(victory_shader::emit::HEAP_VIEWS) {
            return;
        }
        let entry = entry.unwrap_or((victory_shader::emit::VIEW_MARK_NONE, 0));
        // SAFETY: two dwords of the mapped table, which has two per slot
        // below the larger of the title's count and `HEAP_VIEWS`.
        unsafe {
            let at = (*data as *mut u32).add(2 * index as usize);
            at.write_volatile(entry.0);
            at.add(1).write_volatile(entry.1);
        }
    }

    /// The view-mark table, for the root SRV of shaders marking their writes.
    pub(crate) fn marks(&self) -> Option<&ID3D12Resource> {
        self.marks.as_ref().map(|(buffer, _)| buffer)
    }

    pub(crate) fn keeps_texture(&self, index: u32, texture: &ID3D12Resource) -> bool {
        self.kept
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(index as usize)
            .is_some_and(|(resource, _)| resource == texture)
    }

    /// Watches the exact shared slot and host texture used by a recorded
    /// UAV. The mirror lock serializes the snapshot with descriptor writes.
    pub(crate) fn watch_texture_binding(
        &self,
        index: u32,
        view: &ViewPin,
        texture: &ID3D12Resource,
    ) -> Option<std::sync::Arc<crate::texback::Binding>> {
        let _made = self.made.lock().unwrap_or_else(|e| e.into_inner());
        if self.ready.get(index as usize)?.load(Ordering::Acquire) != view.key {
            return None;
        }
        if !self.keeps_texture(index, texture) {
            return None;
        }
        let mut bindings = self
            .texture_bindings
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if let Some(binding) = bindings.get(&index).and_then(std::sync::Weak::upgrade) {
            return Some(binding);
        }
        bindings.retain(|_, binding| binding.strong_count() != 0);
        let binding = std::sync::Arc::new(crate::texback::Binding::default());
        bindings.insert(index, std::sync::Arc::downgrade(&binding));
        self.has_texture_bindings.store(true, Ordering::Release);
        Some(binding)
    }

    fn invalidate_texture_binding(&self, index: u32) {
        if index >= self.count
            || !self.has_texture_bindings.load(Ordering::Acquire)
            || !crate::texback::tracking_bindings()
        {
            return;
        }
        let mut bindings = self
            .texture_bindings
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let binding = bindings.remove(&index);
        if bindings.is_empty() {
            self.has_texture_bindings.store(false, Ordering::Release);
        }
        if let Some(binding) = binding.and_then(|binding| binding.upgrade()) {
            binding.invalidate();
        }
    }

    /// Slot `index` now views `resource` (or nothing of the host's).
    fn keep(&self, index: u32, resource: Kept) {
        let mut kept = self.kept.lock().unwrap_or_else(|e| e.into_inner());
        let old = self.keep_slot(&mut kept, index, resource);
        drop(kept);
        if let Some(old) = old {
            crate::host::retire(old);
        }
    }

    fn keep_slot(&self, kept: &mut Paged<Held>, index: u32, resource: Kept) -> Kept {
        // A slot that held nothing and gets nothing is left alone (no page).
        let slot = kept.slot(index as usize, resource.is_some())?;
        if let (Some((old, old_source)), Some((new, new_source))) =
            (slot.as_ref(), resource.as_ref())
            && windows::core::Interface::as_raw(old) == windows::core::Interface::as_raw(new)
            && same_lease(old_source.as_ref(), new_source.as_ref())
        {
            return None;
        }
        let widened = |held: &Kept| match held {
            Some((_, Some(HeldSource::Widened(view, _)))) => Some(view.clone()),
            _ => None,
        };
        let old = std::mem::replace(slot, resource);
        self.note_widened(widened(&old).as_ref(), false);
        self.note_widened(widened(slot).as_ref(), true);
        #[cfg(feature = "diagnostics")]
        if let Some((old, _)) = &old
            && !self.sampler
            && slot.as_ref().map(|s| windows::core::Interface::as_raw(&s.0))
                != Some(windows::core::Interface::as_raw(old))
            // SAFETY: a live resource.
            && unsafe { old.GetDesc() }.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER
        {
            use windows::core::Interface;
            let left = crate::resource::life::holders(old);
            crate::resource::life::note(
                old.as_raw() as usize,
                crate::resource::life::SLOT,
                u64::from(index),
                left,
            );
        }
        // The GPU may still use the old descriptor. The caller retires its
        // backing and mark owner after dropping the kept lock.
        old
    }

    /// Batched [`HostHeap::keep`]: applies all slot updates under one hold of
    /// `self.kept` and pushes displaced host resources into `retired` for the
    /// caller to retire outside locks.
    fn keep_batch(&self, updates: Vec<(u32, Kept)>, retired: &mut Vec<Held>) {
        if updates.is_empty() {
            return;
        }
        let mut kept = self.kept.lock().unwrap_or_else(|e| e.into_inner());
        retired.extend(
            updates
                .into_iter()
                .filter_map(|(index, held)| self.keep_slot(&mut kept, index, held)),
        );
    }

    fn note_widened(&self, view: Option<&std::sync::Arc<crate::shadow::Widened>>, add: bool) {
        let Some(view) = view else { return };
        let key = std::sync::Arc::as_ptr(view) as usize;
        let mut widened = self.widened.lock().unwrap_or_else(|e| e.into_inner());
        if add {
            let entry = widened
                .entry(key)
                .or_insert_with(|| (0, std::sync::Arc::downgrade(view)));
            entry.0 += 1;
        } else if let Some((count, _)) = widened.get_mut(&key) {
            *count = count.saturating_sub(1);
            if *count == 0 {
                widened.remove(&key);
            }
        }
    }

    /// Widened SRV buffers a heap-indexing shader may read.
    pub(crate) fn widened_buffers(&self) -> Vec<ID3D12Resource> {
        self.widened
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .values()
            .filter_map(|(_, view)| view.upgrade().map(|view| view.buffer.clone()))
            .collect()
    }

    /// The host address of the slots' view addresses (see the field), made
    /// on first use from the slots made so far.
    pub(crate) fn view_addresses(&self) -> Option<u64> {
        let made = self.addresses.get_or_init(|| {
            // The view is a root view, which has no bounds: an entry for
            // every slot a table can lie in, the dynamic ones too (a table
            // made afresh on a host whose heap is smaller than the title's
            // read past the title slots' entries: whatever followed the
            // buffer, or a GPU fault).
            let (buffer, data) = host()?.upload_buffer(u64::from(self.address_slots()) * 16)?;
            crate::host::name(&buffer, || "view addresses".into());
            let data = data as usize;
            let made = self.made.lock().unwrap_or_else(|e| e.into_inner());
            for (i, &key) in made.keys.iter().enumerate() {
                let view = (key != 0)
                    .then(|| view_at(key as u32, (key >> 32) as u32))
                    .flatten();
                write_address(data, i, view.as_deref());
            }
            // Dynamic slots are entered as they are made ([`Self::make_at`],
            // [`Self::make_null`], raw views).
            for i in self.count..self.address_slots() {
                write_address(data, i as usize, None);
            }
            // SAFETY: a live buffer.
            let gpu = unsafe { buffer.GetGPUVirtualAddress() };
            Some((buffer, data, gpu))
        });
        made.as_ref().map(|m| m.2)
    }

    /// The slots the view addresses have an entry for: the title's and the
    /// dynamic ones.
    fn address_slots(&self) -> u32 {
        self.count + self.dynamic
    }

    /// Slot `index` now holds `view` (or no view): its view address.
    fn note_address(&self, index: u32, view: Option<&View>) {
        if index < self.address_slots()
            && let Some(Some((_, data, _))) = self.addresses.get()
        {
            write_address(*data, index as usize, view);
        }
    }

    /// The view address entered for slot `index`; none for a slot past the
    /// view addresses' buffer (tests).
    #[cfg(test)]
    fn address_at(&self, index: u32) -> Option<u64> {
        let (buffer, data, _) = self.addresses.get()?.as_ref()?;
        // SAFETY: a live buffer.
        if 16 * u64::from(index) + 16 > unsafe { buffer.GetDesc() }.Width {
            return None;
        }
        // SAFETY: an entry of the mapped view addresses (checked).
        let [lo, hi] =
            unsafe { ((data + 16 * index as usize) as *const [u32; 2]).read_unaligned() };
        Some(u64::from(hi) << 32 | u64::from(lo))
    }

    /// Makes slots `index` and `index + 1` (dynamic ones) a raw shader
    /// resource view and a raw unordered access view of the shadow's buffer,
    /// which they hold; null views for no shadow.
    pub(crate) fn raw_pair_at(&self, index: u32, shadow: Option<&crate::shadow::Shadow>) {
        self.invalidate_ambient(index);
        self.invalidate_ambient(index + 1);
        let Some(host) = host() else { return };
        let (buffer, elements) = shadow.map_or((None, 0), |shadow| {
            let bytes = shadow.range().1.next_multiple_of(4);
            (Some(&shadow.buffer), raw_elements(bytes, shadow.is_tiled()))
        });
        let srv = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R32_TYPELESS,
            ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_SRV {
                    FirstElement: 0,
                    NumElements: elements,
                    StructureByteStride: 0,
                    Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                },
            },
        };
        let uav = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R32_TYPELESS,
            ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_UAV {
                    NumElements: elements,
                    Flags: D3D12_BUFFER_UAV_FLAG_RAW,
                    ..Default::default()
                },
            },
        };
        // SAFETY: slots of our heap; the buffer is live (or none: null views).
        unsafe {
            host.device
                .CreateShaderResourceView(buffer, Some(&srv), self.cpu(index));
            host.device
                .CreateUnorderedAccessView(buffer, None, Some(&uav), self.cpu(index + 1));
        }
        self.keep(index, buffer.map(|b| (b.clone(), None)));
        self.keep(index + 1, buffer.map(|b| (b.clone(), None)));
        self.write_stride(index, 4);
        self.write_stride(index + 1, 4);
        // Writes through these are title memory's, marked in its page table.
        self.set_mark(index, None);
        self.set_mark(index + 1, None);
        // Not a title view: no entry an earlier table left there stays.
        self.note_address(index, None);
        self.note_address(index + 1, None);
    }

    /// The shader-visible handle of slot `index`.
    pub(crate) fn gpu(&self, index: u32) -> D3D12_GPU_DESCRIPTOR_HANDLE {
        // SAFETY: a live shader-visible heap.
        let start = unsafe { self.heap.GetGPUDescriptorHandleForHeapStart() };
        D3D12_GPU_DESCRIPTOR_HANDLE {
            ptr: start.ptr + u64::from(index) * u64::from(self.step),
        }
    }

    /// Watch the texture descriptor actually emitted into this slot.
    pub(crate) fn ambient_view(&self, index: u32) -> Option<crate::ambient_wave::Binding> {
        // Title slots are mutated under made. Dynamic slots are owned by
        // this recorder until it finishes recording the list.
        let _made =
            (index < self.count).then(|| self.made.lock().unwrap_or_else(|e| e.into_inner()));
        let mut kept = self.kept.lock().unwrap_or_else(|e| e.into_inner());
        let (resource, Some(HeldSource::Ambient(view))) =
            kept.slot(index as usize, false)?.as_mut()?
        else {
            return None;
        };
        let watch = view
            .watch
            .get_or_insert_with(crate::ambient_wave::Watch::new)
            .clone();
        self.ambient_watched.store(true, Ordering::Release);
        Some(crate::ambient_wave::Binding {
            resource: resource.clone(),
            view: view.desc,
            handle: self.gpu(index),
            watch,
        })
    }

    fn invalidate_ambient(&self, index: u32) {
        if !self.ambient_watched.load(Ordering::Acquire) {
            return;
        }
        let watch = {
            let kept = self.kept.lock().unwrap_or_else(|e| e.into_inner());
            kept.get(index as usize)
                .and_then(|(_, source)| match source {
                    Some(HeldSource::Ambient(view)) => view.watch.clone(),
                    _ => None,
                })
        };
        if let Some(watch) = watch {
            // Wait for any submitted producer/helper pair consuming this version.
            watch.invalidate();
        }
    }

    fn cpu(&self, index: u32) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: self.cpu + index as usize * self.step as usize,
        }
    }

    fn set_made_nonzero(&self, index: u32, nonzero: bool) {
        let word = &self.made_nonzero[(index / 64) as usize];
        let mask = 1u64 << (index % 64);
        if nonzero {
            word.fetch_or(mask, Ordering::Release);
        } else {
            word.fetch_and(!mask, Ordering::Release);
        }
    }

    fn made_is_nonzero(&self, index: u32) -> bool {
        let word = self.made_nonzero[(index / 64) as usize].load(Ordering::Acquire);
        (word >> (index % 64)) & 1 != 0
    }

    /// Capture disjoint dirty spans, using workers for large reads.
    /// Large miss batches can use workers regardless of the capture path.
    /// Only the submission thread publishes into the shared heap.
    pub(super) fn sync_ranges(
        &self,
        ranges: &[(u32, u32)],
        start: u64,
        step: u64,
        audit: Option<&dyn Fn(u32)>,
    ) {
        const SLOTS_PER_SPAN: u32 = 2048;
        let spans: Vec<_> = ranges
            .iter()
            .flat_map(|&(first, last)| {
                let mut at = first;
                std::iter::from_fn(move || {
                    if at == last {
                        return None;
                    }
                    let lo = at;
                    at = ((at / SLOTS_PER_SPAN + 1) * SLOTS_PER_SPAN).min(last);
                    Some((lo, at))
                })
            })
            .collect();
        let slots: u64 = spans.iter().map(|&(lo, hi)| u64::from(hi - lo)).sum();
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::SyncSlots, slots);
        static PARALLEL: OnceLock<bool> = OnceLock::new();
        let mut bytes = Vec::new();
        let parallel = spans.len() > 1
            && slots >= 4096
            && *PARALLEL
                .get_or_init(|| std::env::var_os("VICTORY_SYNC_FINE").is_none_or(|v| v != "0"));
        let scans = if parallel {
            let spans_per_job = spans.len().div_ceil(8);
            let scans: Vec<OnceLock<Vec<Option<SlotScan>>>> = spans
                .chunks(spans_per_job)
                .map(|_| OnceLock::new())
                .collect();
            crate::texture::in_parallel(scans.len(), |i| {
                let first = i * spans_per_job;
                let job = &spans[first..(first + spans_per_job).min(spans.len())];
                let mut bytes = Vec::new();
                let captured = job
                    .iter()
                    .map(|&(lo, hi)| self.capture_slots(lo, hi, start, step, &mut bytes))
                    .collect();
                let _ = scans[i].set(captured);
            });
            scans
                .into_iter()
                .flat_map(|job| job.into_inner().unwrap())
                .collect()
        } else {
            spans
                .iter()
                .map(|&(lo, hi)| self.capture_slots(lo, hi, start, step, &mut bytes))
                .collect::<Vec<_>>()
        };
        let mut pending = Vec::new();
        for ((lo, hi), scan) in spans.into_iter().zip(scans) {
            if let Some(scan) = scan {
                pending.push((lo, hi, scan));
            } else {
                self.publish_scans(start, step, std::mem::take(&mut pending), audit);
                self.sync_slots_serial(lo, hi, start, step, &mut bytes, audit);
            }
        }
        self.publish_scans(start, step, pending, audit);
    }

    #[cfg(test)]
    fn sync_slots(&self, first: u32, last: u32, start: u64, step: u64, bytes: &mut Vec<u8>) {
        let last = last.min(self.count);
        if first < last {
            self.sync_slots_serial(first, last, start, step, bytes, None);
        }
    }

    #[cfg(test)]
    fn sync_slots_bytes_with(
        &self,
        slots: std::ops::Range<u32>,
        start: u64,
        step: u64,
        buf_start: u64,
        buf: &[u8],
        fast: bool,
    ) {
        let at = |i| start + u64::from(i) * step;
        let stale: Vec<_> = slots
            .filter_map(|i| {
                let offset = (at(i) - buf_start) as usize;
                let bytes = buf.get(offset..offset + 12)?;
                let record: [u32; 3] = std::array::from_fn(|k| {
                    u32::from_le_bytes(bytes[4 * k..4 * k + 4].try_into().unwrap())
                });
                let key = if record[0] == VIEW_TAG && view_gen_current(record[1], record[2]) {
                    record_key(record)
                } else {
                    0
                };
                (self.synced[i as usize].load(Ordering::Acquire) != key).then_some((i, record))
            })
            .collect();
        if fast {
            self.mirror_synced_batch(&stale, at);
        } else {
            for (i, record) in stale {
                let previous = self.synced[i as usize].load(Ordering::Acquire);
                let _ = self.mirror_record(i, false, Some((previous, at(i))), || record);
            }
        }
    }

    fn sync_slots_serial(
        &self,
        first: u32,
        last: u32,
        start: u64,
        step: u64,
        bytes: &mut Vec<u8>,
        audit: Option<&dyn Fn(u32)>,
    ) {
        if let Some(scan) = self.capture_slots(first, last, start, step, bytes) {
            self.publish_scans(start, step, vec![(first, last, scan)], audit);
        } else {
            // A failed read can cover multiple Windows pages. Retry
            // smaller ranges so a hole does not hide readable neighbors.
            const WINDOWS_PAGE: u64 = 4096;
            let address = start + u64::from(first) * step;
            let end = start + u64::from(last - 1) * step + 12;
            if last - first > 1 && address / WINDOWS_PAGE != (end - 1) / WINDOWS_PAGE {
                let middle = first + (last - first) / 2;
                self.sync_slots_serial(first, middle, start, step, bytes, audit);
                self.sync_slots_serial(middle, last, start, step, bytes, audit);
            }
        }
    }

    fn capture_slots(
        &self,
        first: u32,
        last: u32,
        start: u64,
        step: u64,
        bytes: &mut Vec<u8>,
    ) -> Option<SlotScan> {
        #[cfg(feature = "diagnostics")]
        let _timed =
            crate::watch::Timed(std::time::Instant::now(), crate::copystat::C::SyncCaptureUs);
        let at = |i: u32| start + u64::from(i) * step;
        let forgot = views_forgot();
        let blocks_verified = sync_fast_enabled()
            && (first / SYNC_BLOCK_SLOTS..=(last - 1) / SYNC_BLOCK_SLOTS)
                .all(|b| self.block_forgot[b as usize].load(Ordering::Acquire) == forgot);

        #[cfg(target_arch = "x86_64")]
        if step == 32
            && blocks_verified
            && super::scan::on()
            && let Some(candidates) = super::scan::changed(
                at(first) as *const u8,
                &self.synced,
                &self.made_nonzero,
                first,
                last,
            )
        {
            return Some(SlotScan { candidates, forgot });
        }
        // Snapshot the record headers through the kernel. The title can
        // unmap placed heap pages, and a slot can straddle a tracked page.
        let address = at(first);
        let end = at(last - 1) + 12;
        bytes.resize((end - address) as usize, 0);
        if !crate::host::read_into(address as *const u8, bytes) {
            return None;
        }

        let candidates: Vec<_> = (first..last)
            .filter_map(|index| {
                let offset = (at(index) - address) as usize;
                let bytes = &bytes[offset..offset + 12];
                let record = std::array::from_fn(|i| {
                    u32::from_le_bytes(bytes[4 * i..4 * i + 4].try_into().unwrap())
                });
                let needed = if record[0] != VIEW_TAG {
                    self.made_is_nonzero(index)
                } else {
                    let key = record_key(record);
                    let synced = self.synced[index as usize].load(Ordering::Acquire);
                    !fast_heap_sync()
                        || key != synced
                        || !blocks_verified && !view_gen_current(record[1], record[2])
                };
                needed.then_some(Candidate { index, record })
            })
            .collect();
        Some(SlotScan { candidates, forgot })
    }

    fn publish_scans(
        &self,
        start: u64,
        step: u64,
        mut scans: Vec<(u32, u32, SlotScan)>,
        audit: Option<&dyn Fn(u32)>,
    ) {
        if scans.is_empty() {
            return;
        }
        #[cfg(feature = "diagnostics")]
        let _timed =
            crate::watch::Timed(std::time::Instant::now(), crate::copystat::C::SyncPublishUs);
        if scans.iter().any(|(_, _, scan)| !scan.candidates.is_empty()) {
            let records = {
                let made = self.made.lock().unwrap_or_else(|e| e.into_inner());
                changed_records(
                    &made.keys,
                    scans.iter_mut().flat_map(|(_, _, scan)| {
                        std::mem::take(&mut scan.candidates)
                            .into_iter()
                            .map(|c| (c.index, c.record))
                    }),
                )
            };
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::SyncChanged, records.len() as u64);
            if let Some(record_check) = audit {
                for record in &records {
                    record_check(record.index);
                }
            }
            self.publish_records(&records, start, step);
        }

        for (first, last, scan) in scans {
            for b in first / SYNC_BLOCK_SLOTS..=(last - 1) / SYNC_BLOCK_SLOTS {
                let b_start = b * SYNC_BLOCK_SLOTS;
                let b_end = ((b + 1) * SYNC_BLOCK_SLOTS).min(self.count);
                if first <= b_start && last >= b_end {
                    self.block_forgot[b as usize].store(scan.forgot, Ordering::Release);
                }
            }
        }
    }

    fn publish_records(&self, records: &[ChangedRecord], start: u64, step: u64) {
        use crate::queue::{CpuPhase, CpuTimer};
        let mut misses = Vec::new();
        let cache = templates::enabled();
        if !cache && fast_heap_sync() && !show_after() {
            let stale: Vec<_> = records.iter().map(|r| (r.index, r.record)).collect();
            self.mirror_synced_batch(&stale, |index| start + u64::from(index) * step);
            return;
        }
        if cache {
            let pinning = CpuTimer::new(CpuPhase::DescriptorPins);
            let pins = pin_changed_views(records);
            drop(pinning);
            let mut batch = PublishBatch::default();
            self.publish_templates(records, start, step, (&pins, None), &mut batch, &mut misses);
            if !self.sampler && misses.len() >= MIN_STAGED_RECORDS {
                let staging = VIEW_STAGING.lock().unwrap_or_else(|e| e.into_inner());
                if let Some(workers) = staging.get_or_init(view_staging) {
                    #[cfg(test)]
                    STAGED_BATCHES.with(|batches| batches.set(batches.get() + 1));
                    // A worker temporarily owns this heap's template shard.
                    // Later frames can hit the cache without another worker
                    // round, while scratch sources stay live through the copy.
                    let staging_timer = CpuTimer::new(CpuPhase::DescriptorStage);
                    self.swap_templates(workers);
                    let staged = stage_views(workers, &misses, &pins);
                    self.swap_templates(workers);
                    drop(staging_timer);
                    let mut remaining = Vec::new();
                    self.publish_templates(
                        &misses,
                        start,
                        step,
                        (&pins, Some(&staged)),
                        &mut batch,
                        &mut remaining,
                    );
                    if staged.recheck_pending(&remaining) {
                        misses.clear();
                        self.publish_templates(
                            &remaining,
                            start,
                            step,
                            (&pins, Some(&staged)),
                            &mut batch,
                            &mut misses,
                        );
                    } else {
                        misses = remaining;
                    }
                }
            }
        }
        let records = if cache { &misses } else { records };
        if records.is_empty() {
            return;
        }
        let _fallback = CpuTimer::new(CpuPhase::DescriptorFallback);
        for &ChangedRecord {
            index,
            record,
            previous,
            ..
        } in records
        {
            let address = start + u64::from(index) * step;
            let _ = self.mirror_record(index, false, Some((previous, address)), || record);
        }
    }

    fn swap_templates(&self, workers: &StagingWorkers) {
        let mut made = self.made.lock().unwrap_or_else(|e| e.into_inner());
        for (i, worker) in workers.iter().enumerate() {
            let mut worker = worker.lock().unwrap_or_else(|e| e.into_inner());
            let templates = &mut made.templates[i];
            templates.get_or_init(|| Templates::new(false));
            if let Some(templates) = templates.get_mut().and_then(Option::as_mut) {
                std::mem::swap(templates, &mut worker.templates);
            }
            std::mem::swap(&mut made.textures[i], &mut worker.textures);
        }
    }

    /// Publish ready descriptors without ending the batch at each miss.
    /// Each scan contains disjoint slots. The delayed slots still check
    /// both the captured header and the previous key before publication.
    fn publish_templates(
        &self,
        records: &[ChangedRecord],
        start: u64,
        step: u64,
        (pins, staged): (&HashMap<u64, ViewPin>, Option<&StagedViews<'_>>),
        batch: &mut PublishBatch,
        misses: &mut Vec<ChangedRecord>,
    ) {
        let _timer = crate::queue::CpuTimer::new(crate::queue::CpuPhase::DescriptorTemplates);
        // Give recording and live descriptor writes a chance to take the
        // mirror lock between bounded publication batches.
        const RECORDS_PER_BATCH: usize = 2048;
        let PublishBatch { copies, published } = batch;
        for records in records.chunks(RECORDS_PER_BATCH) {
            self.publish_template_batch(
                records,
                start,
                step,
                (pins, staged),
                (copies, published),
                misses,
            );
        }
    }

    fn publish_template_batch(
        &self,
        records: &[ChangedRecord],
        start: u64,
        step: u64,
        (pins, staged): (&HashMap<u64, ViewPin>, Option<&StagedViews<'_>>),
        (copies, published): (&mut DescriptorCopies, &mut Vec<Publication>),
        misses: &mut Vec<ChangedRecord>,
    ) {
        let mut made = self.made.lock().unwrap_or_else(|e| e.into_inner());
        let shadow_generation = crate::shadow::generation();
        let null = staged
            .map(|staged| staged.null)
            .or_else(|| made.null_template());
        for &change in records {
            let ChangedRecord {
                index,
                record,
                previous,
                clear,
            } = change;
            let address = start + u64::from(index) * step;
            if made.keys[index as usize] != previous {
                continue;
            }
            if clear {
                if previous == 0 {
                    continue;
                }
                let Some(null) = null else {
                    misses.push(change);
                    continue;
                };
                if self.snapshot_matches(index, address, record) {
                    self.forget_record(&mut made, index);
                    self.clear_slot_metadata(index);
                    copies.push(self.cpu(index), null, self.step as usize);
                    published.push(Publication::Clear(index, None));
                }
                continue;
            }
            let key = record_key(record);
            let epoch = crate::shadow::lookup_epoch();
            if let Some(staged) = staged.and_then(|staged| staged.views.get(&key))
                && let Some(hit) = staged.hit(epoch)
            {
                if self.snapshot_matches(index, address, record) {
                    self.queue_publication(&mut made, index, staged.view, hit, copies, published);
                }
                continue;
            }
            let buffer = made.templates[self.template_shard(key)]
                .get()
                .and_then(Option::as_ref)
                .and_then(|templates| templates.get(key, epoch));
            let texture = if buffer.is_none() {
                made.textures[template_shard(key)]
                    .get()
                    .and_then(Option::as_ref)
                    .and_then(|templates| templates.entry(key))
            } else {
                None
            };
            if buffer.is_none() && texture.is_none() && (self.sampler || staged.is_some()) {
                misses.push(change);
                continue;
            }
            let Some(view) = pins.get(&key) else {
                misses.push(change);
                continue;
            };
            let hit = if let Some(hit) = buffer {
                let binding = if let Some(backing) = hit.backing {
                    let widened = is_widened(&hit.held);
                    Binding::Buffer(backing, widened)
                } else {
                    Binding::Sampler
                };
                Some(TemplateHit {
                    at: hit.at,
                    held: hit.held,
                    mark: hit.mark,
                    binding,
                    texture: None,
                })
            } else if let Some(texture) = texture {
                texture.hit(view)
            } else {
                let deferred = is_placed_texture(view);
                if deferred
                    || matches!(
                        **view,
                        View::Shader(0, _) | View::Unordered(0, _, _) | View::Constant(_, 0..=3)
                    )
                {
                    null_template_for(view).map(|at| TemplateHit {
                        at,
                        held: None,
                        mark: None,
                        binding: Binding::Left {
                            kind: crate::bindless::left_kind(view),
                            deferred,
                        },
                        texture: None,
                    })
                } else {
                    None
                }
            };
            if let Some(hit) = hit {
                if self.snapshot_matches(index, address, record) {
                    self.queue_publication(&mut made, index, view, hit, copies, published);
                }
            } else {
                misses.push(change);
            }
        }
        if !copies.src.is_empty() {
            copies.publish(self.sampler);
            let mut kept = self.kept.lock().unwrap_or_else(|e| e.into_inner());
            for publication in published.iter_mut() {
                match publication {
                    Publication::Clear(index, held) | Publication::View(index, _, _, _, held) => {
                        *held = self.keep_slot(&mut kept, *index, held.take());
                    }
                }
            }
        }
        let mut bindings = self.viewed.updates();
        let mut hits = 0;
        let mut generations = self.buffer_generations.lock().unwrap_or_else(|e| e.into_inner());
        for publication in published.iter_mut() {
            match publication {
                Publication::Clear(index, old) => {
                    if let Some(old) = old.take() {
                        crate::host::retire(old);
                    }
                    generations.remove(index);
                    bindings.clear(*index);
                }
                Publication::View(index, key, uav, binding, old) => {
                    if let Some(old) = old.take() {
                        crate::host::retire(old);
                    }
                    let mut ready = *key;
                    if matches!(binding, Binding::Buffer(..)) {
                        generations.insert(*index, shadow_generation);
                    } else {
                        generations.remove(index);
                    }
                    match std::mem::replace(binding, Binding::Sampler) {
                        Binding::Buffer(backing, widened) => {
                            bindings.buffer(*index, backing, *uav, widened);
                        }
                        Binding::Texture(resource) => bindings.texture(*index, resource),
                        Binding::Left { kind, deferred } => {
                            bindings.left(*index, kind);
                            if deferred {
                                ready = 0;
                            }
                        }
                        Binding::Sampler => {}
                    }
                    self.ready[*index as usize].store(ready, Ordering::Release);
                    self.synced[*index as usize].store(*key, Ordering::Release);
                    if ready == 0 && !self.sampler {
                        self.enqueue_texture(*index);
                    }
                    hits += 1;
                }
            }
        }
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::SyncTemplateHits, hits);
        #[cfg(not(feature = "diagnostics"))]
        let _ = hits;
        drop(generations);
        drop(bindings);
        drop(made);
        published.clear();
    }

    fn queue_publication(
        &self,
        made: &mut Mirrored,
        index: u32,
        view: &ViewPin,
        hit: TemplateHit,
        copies: &mut DescriptorCopies,
        published: &mut Vec<Publication>,
    ) {
        if let Some((copy, srv)) = hit.texture {
            copy.viewed.store(true, Ordering::Relaxed);
            if !srv {
                if crate::resource::evict_heap_uav() {
                    copy.note_heap_writable();
                } else {
                    copy.note_state(0x8);
                }
            }
        }
        self.begin_record(made, index, view.key, view);
        if matches!(hit.binding, Binding::Left { deferred: true, .. }) {
            crate::shadow::forget_read(index);
        }
        made.pending.remove(&index);
        self.set_mark(index, hit.mark);
        copies.push(self.cpu(index), hit.at, self.step as usize);
        let uav = matches!(**view, View::Unordered(..) | View::Placed(true, ..));
        published.push(Publication::View(
            index,
            view.key,
            uav,
            hit.binding,
            hit.held,
        ));
    }

    fn template_shard(&self, key: u64) -> usize {
        if self.sampler { 0 } else { template_shard(key) }
    }

    fn begin_record(&self, made: &mut Mirrored, index: u32, key: u64, view: &View) {
        self.invalidate_ambient(index);
        self.synced[index as usize].store(0, Ordering::Release);
        self.ready[index as usize].store(0, Ordering::Relaxed);
        self.invalidate_texture_binding(index);
        self.set_stride(index, Some(view));
        self.note_address(index, Some(view));
        if made.keys[index as usize] == 0 {
            self.set_made_nonzero(index, true);
        }
        made.keys[index as usize] = key;
    }

    fn clear_record(
        &self,
        made: &mut Mirrored,
        index: u32,
        template: Option<D3D12_CPU_DESCRIPTOR_HANDLE>,
    ) {
        self.forget_record(made, index);
        self.clear_descriptor(index, template);
        self.buffer_generations.lock().unwrap_or_else(|e| e.into_inner()).remove(&index);
    }

    fn forget_record(&self, made: &mut Mirrored, index: u32) {
        self.synced[index as usize].store(0, Ordering::Release);
        self.remove_upload(index);
        self.ready[index as usize].store(0, Ordering::Relaxed);
        made.keys[index as usize] = 0;
        made.pending.remove(&index);
        self.set_made_nonzero(index, false);
    }

    fn snapshot_matches(&self, index: u32, address: u64, record: [u32; 3]) -> bool {
        let ready = &self.ready[index as usize];
        let saved = ready.swap(0, Ordering::AcqRel);
        let matches = header_matches(address, record);
        if !matches {
            ready.store(saved, Ordering::Release);
        }
        matches
    }

    /// Mirrors a page's stale slots (`for_draw == false`) under one hold each
    /// of `VIEWS`, `self.made`, `self.kept`, `self.viewed`, and `shadow::READS`,
    /// deduplicating `prepare` and descriptor creation via [`StagingHeap`] +
    /// `CopyDescriptorsSimple`.
    fn mirror_synced_batch(&self, stale: &[(u32, [u32; 3])], at: impl Fn(u32) -> u64) {
        struct CachedTemplate {
            src: D3D12_CPU_DESCRIPTOR_HANDLE,
            kept: Option<Kept>,
            mark: Option<(u32, u32)>,
        }

        let mut unique_keys: Vec<u64> = Vec::new();
        let mut key_counts: Vec<u32> = Vec::new();
        let mut key_to_idx: HashMap<u64, usize> = HashMap::new();
        let mut slot_key_idx: Vec<Option<usize>> = Vec::with_capacity(stale.len());

        for &(_, [tag, view, generation]) in stale {
            if tag == VIEW_TAG && generation != 0 {
                let key = u64::from(generation) << 32 | u64::from(view);
                let idx = match key_to_idx.get(&key) {
                    Some(&idx) => {
                        key_counts[idx] += 1;
                        idx
                    }
                    None => {
                        let idx = unique_keys.len();
                        unique_keys.push(key);
                        key_counts.push(1);
                        key_to_idx.insert(key, idx);
                        idx
                    }
                };
                slot_key_idx.push(Some(idx));
            } else {
                slot_key_idx.push(None);
            }
        }

        let shadow_generation = crate::shadow::generation();
        let pins = views_at_batch(&unique_keys);
        let mut is_placed: Vec<bool> = Vec::with_capacity(unique_keys.len());
        let mut is_tex: Vec<bool> = Vec::with_capacity(unique_keys.len());
        let mut strides: Vec<u32> = Vec::with_capacity(unique_keys.len());
        for pin in &pins {
            if let Some(view) = pin.as_deref() {
                let placed = is_placed_texture(view);
                if !placed {
                    prepare(view);
                }
                is_placed.push(placed);
                is_tex.push(is_texture(view));
                strides.push(view_stride(view) | view_kind(view));
            } else {
                is_placed.push(false);
                is_tex.push(false);
                strides.push(0);
            }
        }

        // Initialize the staging heap before acquiring `self.made`.
        let staging = self
            .staging
            .get_or_init(|| StagingHeap::new(self.sampler, self.step))
            .as_ref();
        let host = host();

        let mut again: Vec<u32> = Vec::new();
        let mut retired: Vec<Held> = Vec::new();

        {
            let mut made = self.made.lock().unwrap_or_else(|e| e.into_inner());
            let mut buffer_generations = self
                .buffer_generations
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            let mut templates: Vec<Option<CachedTemplate>> =
                (0..unique_keys.len()).map(|_| None).collect();
            let mut next_template = if self.sampler { 1 } else { STAGING_NULL_SLOTS };
            let mut kept_updates: Vec<(u32, Kept)> = Vec::with_capacity(stale.len());
            let mut copy_dsts: Vec<D3D12_CPU_DESCRIPTOR_HANDLE> = Vec::with_capacity(stale.len());
            let mut copy_srcs: Vec<D3D12_CPU_DESCRIPTOR_HANDLE> = Vec::with_capacity(stale.len());
            let mut viewed_updates: Vec<(u32, Option<&View>, bool)> = if self.sampler {
                Vec::new()
            } else {
                Vec::with_capacity(stale.len())
            };
            let mut forgot_reads: Vec<u32> = Vec::new();
            let mut finalize: Vec<(u32, u64, u64)> = Vec::with_capacity(stale.len());
            let addr_data = self
                .addresses
                .get()
                .and_then(Option::as_ref)
                .map(|&(_, data, _)| data);
            let addr_slots = self.address_slots();

            for (entry_idx, &(index, snapshot)) in stale.iter().enumerate() {
                let addr = at(index);
                // Re-read live title memory under `self.made`: if a title
                // thread overwrote the slot since the page snapshot (including
                // during `prepare`), defer to `mirror` after dropping locks.
                let Some(live) = capture_header(addr) else {
                    continue;
                };
                if live != snapshot {
                    again.push(index);
                    continue;
                }
                let resolved = slot_key_idx[entry_idx].and_then(|kidx| {
                    let [_, view_i, generation] = snapshot;
                    if !view_gen_current(view_i, generation) {
                        return None;
                    }
                    pins[kidx]
                        .as_deref()
                        .filter(|view| is_sampler(view) == self.sampler)
                        .map(|view| (kidx, unique_keys[kidx], view))
                });
                let Some((kidx, key, view)) = resolved else {
                    if made.keys[index as usize] != 0 {
                        self.forget_record(&mut made, index);
                        self.clear_slot_metadata(index);
                        if let (Some(_), Some(st)) = (host, staging) {
                            copy_dsts.push(self.cpu(index));
                            copy_srcs.push(st.cpu(0));
                        } else if self.sampler {
                            self.null_sampler(index);
                        } else {
                            null_view(self.cpu(index));
                        }
                        self.write_stride(index, 0);
                        self.set_mark(index, None);
                        if let Some(data) = addr_data
                            && index < addr_slots
                        {
                            write_address(data, index as usize, None);
                        }
                        buffer_generations.remove(&index);
                        kept_updates.push((index, None));
                        if !self.sampler {
                            viewed_updates.push((index, None, false));
                        }
                        finalize.push((index, 0, 0));
                    }
                    continue;
                };
                if made.keys[index as usize] == key {
                    continue;
                }
                self.begin_record(&mut made, index, key, view);
                self.write_stride(index, strides[kidx]);
                if let Some(data) = addr_data
                    && index < addr_slots
                {
                    write_address(data, index as usize, Some(view));
                }
                let worker = self.worker_texture(view);
                let later = is_placed[kidx] || worker;
                if later {
                    forgot_reads.push(index);
                    if worker {
                        self.enqueue_texture(index);
                    }
                }
                made.keys[index as usize] = key;

                let (kept, mark) = if later {
                    if !self.sampler {
                        if let (Some(_), Some(st)) = (host, staging) {
                            let null_slot = null_slot_index(false, Some(view));
                            copy_dsts.push(self.cpu(index));
                            copy_srcs.push(st.cpu(null_slot));
                        } else {
                            null_for(view, self.cpu(index));
                        }
                    }
                    (None, None)
                } else if key_counts[kidx] >= 2
                    && let (Some(_), Some(st)) = (host, staging)
                {
                    if let Some(tpl) = &templates[kidx] {
                        copy_dsts.push(self.cpu(index));
                        copy_srcs.push(tpl.src);
                        (tpl.kept.clone(), tpl.mark)
                    } else if next_template < st.count {
                        let t = next_template;
                        next_template += 1;
                        let (k, m) = make_marked_with_backing(view, st.cpu(t), None);
                        let src = if k.is_none() && !self.sampler {
                            st.cpu(null_slot_index(false, Some(view)))
                        } else {
                            st.cpu(t)
                        };
                        copy_dsts.push(self.cpu(index));
                        copy_srcs.push(src);
                        templates[kidx] = Some(CachedTemplate {
                            src,
                            kept: k.clone(),
                            mark: m,
                        });
                        (k, m)
                    } else {
                        let (k, m) = make_marked_with_backing(view, self.cpu(index), None);
                        if k.is_none() && !self.sampler {
                            let null_slot = null_slot_index(false, Some(view));
                            copy_dsts.push(self.cpu(index));
                            copy_srcs.push(st.cpu(null_slot));
                        }
                        (k, m)
                    }
                } else {
                    let (k, m) = make_marked_with_backing(view, self.cpu(index), None);
                    if k.is_none() && !self.sampler {
                        if let (Some(_), Some(st)) = (host, staging) {
                            let null_slot = null_slot_index(false, Some(view));
                            copy_dsts.push(self.cpu(index));
                            copy_srcs.push(st.cpu(null_slot));
                        } else {
                            null_for(view, self.cpu(index));
                        }
                    }
                    (k, m)
                };

                self.set_mark(index, mark);
                if !self.sampler {
                    viewed_updates.push((index, Some(view), kept.is_some()));
                }
                let kept = kept.flatten();
                let ready = if (is_tex[kidx] || later) && kept.is_none() {
                    0
                } else {
                    key
                };
                if buffer_range(view).is_some() {
                    buffer_generations.insert(index, shadow_generation);
                } else {
                    buffer_generations.remove(&index);
                }
                kept_updates.push((index, kept));
                finalize.push((index, key, ready));
            }

            if !copy_dsts.is_empty()
                && let (Some(h), Some(st)) = (host, staging)
            {
                let count = copy_dsts.len() as u32;
                // SAFETY: `copy_dsts` and `copy_srcs` have length `count`; each
                // source handle lives in our CPU-only staging heap (already
                // populated above) and each dest handle in our shader-visible
                // heap; `None` range sizes specify 1 descriptor per range.
                unsafe {
                    h.device.CopyDescriptors(
                        count,
                        copy_dsts.as_ptr(),
                        None,
                        count,
                        copy_srcs.as_ptr(),
                        None,
                        st.kind,
                    );
                }
            }

            self.keep_batch(kept_updates, &mut retired);
            if !self.sampler {
                self.viewed.set_batch(&viewed_updates);
            }
            crate::shadow::forget_reads(&forgot_reads);
            // Coalesce `made_nonzero` bit updates per 64-slot word (stale indices
            // arrive in ascending order): set nonzero bits before publishing
            // `synced`/`ready`, and clear nonzero bits after clearing them.
            let mut set_word = usize::MAX;
            let mut set_mask = 0u64;
            for &(index, new_synced, _) in &finalize {
                if new_synced == 0 {
                    continue;
                }
                let word = (index / 64) as usize;
                if word != set_word {
                    if set_word != usize::MAX && set_mask != 0 {
                        self.made_nonzero[set_word].fetch_or(set_mask, Ordering::Release);
                    }
                    set_word = word;
                    set_mask = 0;
                }
                set_mask |= 1u64 << (index & 63);
            }
            if set_word != usize::MAX && set_mask != 0 {
                self.made_nonzero[set_word].fetch_or(set_mask, Ordering::Release);
            }
            for &(index, new_synced, new_ready) in &finalize {
                self.synced[index as usize].store(new_synced, Ordering::Release);
                self.ready[index as usize].store(new_ready, Ordering::Release);
            }
            let mut clr_word = usize::MAX;
            let mut clr_mask = 0u64;
            for &(index, new_synced, _) in &finalize {
                if new_synced != 0 {
                    continue;
                }
                let word = (index / 64) as usize;
                if word != clr_word {
                    if clr_word != usize::MAX && clr_mask != 0 {
                        self.made_nonzero[clr_word].fetch_and(!clr_mask, Ordering::Release);
                    }
                    clr_word = word;
                    clr_mask = 0;
                }
                clr_mask |= 1u64 << (index & 63);
            }
            if clr_word != usize::MAX && clr_mask != 0 {
                self.made_nonzero[clr_word].fetch_and(!clr_mask, Ordering::Release);
            }
        }

        crate::host::retire_all(retired);
        drop(pins);
        for index in again {
            self.mirror(index, at(index), false);
        }
    }

    /// Whether the view made in slot `index` must be made again for a draw
    /// that reads it: a placed texture's view left null is made when a draw
    /// reads the slot through a table (heap syncs revisit it at every
    /// submission), a texture's unordered view left null when the texture was
    /// given up (`VICTORY_EVICT_HEAP_UAV`) is made again like a shader view (a
    /// table copying the null slot loses writes), and a later unordered or
    /// renderable view that upgraded the memory's `HostTexture` leaves the
    /// made view on the old one.
    fn needs_remake(&self, index: u32, view: &View, for_draw: bool) -> bool {
        if buffer_range(view).is_some() {
            return !self.buffer_generation_current(index);
        }
        if !for_draw || !(is_texture(view) || is_unordered_texture(view)) {
            return false;
        }
        let held = self
            .kept
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(index as usize)
            .map(|(res, _)| windows::core::Interface::as_raw(res));
        let Some(held) = held else {
            return true;
        };
        is_placed_texture(view)
            && placed_texture(view).is_some_and(|p| {
                let (View::Shader(r, _) | View::Unordered(r, ..)) = p else {
                    return false;
                };
                !crate::resource::host_copy(r as crate::object::This).is_some_and(|copy| {
                    windows::core::Interface::as_raw(&copy.texture) == held
                        || copy.twin.get().and_then(Option::as_ref).is_some_and(|t| {
                            t.planes
                                .iter()
                                .any(|p| windows::core::Interface::as_raw(&p.texture) == held)
                        })
                })
            })
    }

    fn buffer_generation_current(&self, index: u32) -> bool {
        self.buffer_generations
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&index)
            .is_some_and(|&made| made == crate::shadow::generation())
    }

    /// Makes host slot `index` from the record the title has at `handle`
    /// if it changed since, and returns the view recorded there, pinned for
    /// as long as the caller uses it. Buffer views are made only for draws
    /// (`for_draw`), which also copy their data: a host copy of a buffer is
    /// made on first use.
    /// Whether host slot `index` already holds the mirrored view currently
    /// named by `handle`, including the owner generation of buffer views.
    pub(crate) fn slot_ready(&self, index: u32, handle: u64) -> bool {
        let Some(slot) = self.ready.get(index as usize) else {
            return false;
        };
        let [tag, i, generation] = record_at(handle);
        let key = u64::from(generation) << 32 | u64::from(i);
        tag == VIEW_TAG
            && key != 0
            && slot.load(Ordering::Acquire) == key
            && view_gen_current(i, generation)
            && self
                .buffer_generations
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .get(&index)
                .is_none_or(|&made| made == crate::shadow::generation())
    }

    pub(crate) fn mirror(&self, index: u32, handle: u64, for_draw: bool) -> Option<ViewPin> {
        if index >= self.count {
            // Diagnostic (NVIDIA heap cap): the highest title slot past the
            // host heap seen so far.
            static HIGHEST: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if HIGHEST.fetch_max(index, std::sync::atomic::Ordering::Relaxed) < index {
                tracing::warn!(
                    index,
                    count = self.count,
                    for_draw,
                    "title slot past the host heap"
                );
            }
            return None;
        }
        let [tag, i, generation] = record_at(handle);
        // Generations start at 1, so a made slot's key is never 0.
        let key = record_key([tag, i, generation]);
        // Made from this record already and readable as it is (what the
        // checks below find under the lock): nothing to make.
        let ready = &self.ready[index as usize];
        // A same-key live write still participates in snapshot publication:
        // its release either precedes the snapshot's acquire, or the cleared
        // ready key makes it take the lock and finish after publication.
        let ready_for_record = || {
            if for_draw {
                ready.load(Ordering::Acquire) == key
            } else {
                ready
                    .compare_exchange(key, key, Ordering::AcqRel, Ordering::Acquire)
                    .is_ok()
            }
        };
        if tag == VIEW_TAG
            && ready_for_record()
            && let Some(view) = view_at(i, generation)
            && is_sampler(&view) == self.sampler
            && !self.needs_remake(index, &view, for_draw)
        {
            return Some(view);
        }
        loop {
            if let Ok(view) = self.mirror_record(index, for_draw, None, || record_at(handle)) {
                return view;
            }
        }
    }

    /// Materializes a captured record or the latest record at a live slot.
    /// Snapshot publication closes the live-write fast path and validates
    /// the capture under the slot lock, including if the title has unmapped
    /// the slot since capture.
    fn mirror_record(
        &self,
        index: u32,
        for_draw: bool,
        snapshot: Option<(u64, u64)>,
        read: impl Fn() -> [u32; 3],
    ) -> Result<Option<ViewPin>, ()> {
        let shadow_generation = crate::shadow::generation();
        let mut made = self.made.lock().unwrap_or_else(|e| e.into_inner());
        if snapshot.is_some_and(|(key, _)| made.keys[index as usize] != key) {
            return Ok(None);
        }
        let record = read();
        let [tag, i, generation] = record;
        let snapshot_changed = |made: u64| {
            snapshot.is_some_and(|(previous, address)| {
                if made != previous {
                    return true;
                }
                !self.snapshot_matches(index, address, record)
            })
        };
        let key = record_key(record);
        let view = (tag == VIEW_TAG).then(|| view_at(i, generation)).flatten();
        let Some(view) = view else {
            // Not a view of ours (never written, or of a resource the
            // title released): nothing to read.
            if made.keys[index as usize] != 0 && !snapshot_changed(made.keys[index as usize]) {
                let template = made.null_template();
                self.clear_record(&mut made, index, template);
                self.viewed.set(index, None, false);
            }
            return Ok(None);
        };
        let view_is_sampler = is_sampler(&view);
        if view_is_sampler != self.sampler {
            warn_heap_mismatch(index, self.sampler, view_is_sampler);
            if made.keys[index as usize] != 0 && !snapshot_changed(made.keys[index as usize]) {
                let template = made.null_template();
                self.clear_record(&mut made, index, template);
                self.viewed.set(index, None, false);
            }
            return Ok(None);
        }
        // A placed texture's view left null below is made when a draw reads
        // the slot through a table; until then the slot is not made again
        // (heap syncs revisit it at every submission). Also re-mirror if a
        // later unordered or renderable view upgraded the memory's HostTexture.
        let needs_remake = |made: &Mirrored| {
            made.pending
                .get(&index)
                .is_none_or(|p| for_draw && p.copy.upgrade().is_none_or(|copy| !p.current(&copy)))
                && self.needs_remake(index, &view, for_draw)
        };
        if made.keys[index as usize] == key && !needs_remake(&made) {
            return Ok(Some(view));
        }
        // Shaders index the heap itself (`crate::bindless`): buffer views
        // are made when written.
        // shortcut: views of textures the title placed itself are made
        // when a draw reads them through a table: a host texture (and a
        // staging copy) for each of the ~20,000 the title has views of
        // ran the host out of memory (run b2).
        let worker = !for_draw && self.worker_texture(&view);
        let later = !for_draw && (worker || is_placed_texture(&view));
        let cache = snapshot.is_some() && !later && templates::enabled();
        let epoch = if cache {
            crate::shadow::lookup_epoch()
        } else {
            0
        };
        // An earlier serial fallback can have filled this entry after
        // staging. Reuse it for later slots in the same batch when the
        // current mapping still names that exact backing.
        let cached = cache
            .then(|| made.templates[self.template_shard(key)].get())
            .flatten()
            .and_then(Option::as_ref)
            .and_then(|templates| templates.get(key, epoch));
        let backing = if let Some(hit) = &cached {
            hit.backing.clone()
        } else if !later {
            // What the view needs (the texture it views, the copy of the
            // memory) is made without the slot lock, which title threads
            // binding and writing views wait on (run evict5).
            drop(made);
            let backing = prepare(&view);
            made = self.made.lock().unwrap_or_else(|e| e.into_inner());
            // For live writes, retry if the title replaced the record while
            // the slot lock was released. For snapshots `read` is constant.
            if read() != [tag, i, generation] {
                drop(made);
                return Err(());
            }
            if made.keys[index as usize] == key && !needs_remake(&made) {
                return Ok(Some(view));
            }
            backing
        } else {
            None
        };
        if snapshot_changed(made.keys[index as usize]) {
            drop(made);
            return Ok(None);
        }
        {
            self.begin_record(&mut made, index, key, &view);
            if later {
                if worker {
                    self.enqueue_texture(index);
                }
                // Made once a shader reads it (`crate::shadow::take_reads`).
                crate::shadow::forget_read(index);
            }
            if !self.sampler && show_after() {
                made.pending.remove(&index);
                if !later
                    && backing.is_none()
                    && self.mirror_texture(index, &view, record, &mut made, cache)
                {
                    self.buffer_generations.lock().unwrap_or_else(|e| e.into_inner()).remove(&index);
                    self.synced[index as usize].store(key, Ordering::Release);
                    return Ok(Some(view));
                }
            }
            let (kept, mark) = if later {
                (None, None)
            } else {
                let dst = self.cpu(index);
                if let Some(hit) = cached {
                    templates::copy(self.sampler, dst, hit.at);
                    (Some(hit.held), hit.mark)
                } else if cache && (backing.is_some() || self.sampler) {
                    let templates = &mut made.templates[self.template_shard(key)];
                    templates.get_or_init(|| Templates::new(self.sampler));
                    if let Some(templates) = templates.get_mut().and_then(Option::as_mut) {
                        templates.make(dst, key, epoch, &view, backing.as_ref())
                    } else {
                        make_marked_with_backing(&view, dst, backing.as_ref().map(|b| &b.0))
                    }
                } else if cache && (is_texture(&view) || is_unordered_texture(&view)) {
                    let cache = texture_cache(&mut made, key);
                    (make_with_texture_cache(&view, dst, None, cache), None)
                } else {
                    make_marked_with_backing(&view, dst, backing.as_ref().map(|b| &b.0))
                }
            };
            self.set_mark(index, mark);
            if kept.is_none() && !self.sampler {
                null_for(&view, self.cpu(index));
            }
            let was_made = kept.is_some();
            let kept = kept.flatten();
            if !self.sampler {
                let widened = matches!(&kept, Some((_, Some(HeldSource::Widened(..)))));
                self.viewed
                    .set_with_backing(index, Some(&view), was_made, backing, widened);
            }
            // A texture's view left null is made again when a draw reads it.
            let ready = if (later || is_texture(&view)) && kept.is_none() {
                0
            } else {
                key
            };
            self.keep(index, kept);
            let mut generations = self.buffer_generations.lock().unwrap_or_else(|e| e.into_inner());
            if buffer_range(&view).is_some() {
                generations.insert(index, shadow_generation);
            } else {
                generations.remove(&index);
            }
            self.ready[index as usize].store(ready, Ordering::Release);
            self.synced[index as usize].store(key, Ordering::Release);
        }
        Ok(Some(view))
    }

    /// Mirror an SRV of a title-fed texture using the exact copy checked
    /// for readiness. GPU-written textures and other view kinds take the
    /// ordinary path. Layouts without staging use the ordinary descriptor.
    fn mirror_texture(
        &self,
        index: u32,
        view: &View,
        record: [u32; 3],
        made: &mut Mirrored,
        cache: bool,
    ) -> bool {
        let placed = is_placed_texture(view);
        let resolved = placed_texture(view).unwrap_or_else(|| view.clone());
        let View::Shader(resource, desc) = &resolved else {
            return false;
        };
        let resource = *resource as crate::object::This;
        let Some(copy) = crate::resource::made_copy(resource) else {
            return false;
        };
        if !crate::texture::eligible_for_gating(resource, &copy) {
            return false;
        }
        let Some(desc) = texture_srv_desc(resource, &copy, desc.as_ref()) else {
            return false;
        };
        self.set_mark(index, None);
        if copy.evicted.load(Ordering::Acquire) {
            null_texture_srv(&copy, desc.as_ref(), self.cpu(index));
            self.keep(index, None);
            self.viewed.set(index, Some(view), false);
            crate::shadow::forget_read(index);
        } else if !copy.initialized.load(Ordering::Acquire)
            && let Some((memory, layout)) = crate::resource::texture_memory(resource)
        {
            null_texture_srv(&copy, desc.as_ref(), self.cpu(index));
            self.keep(index, None);
            self.viewed.set(index, Some(view), false);
            crate::shadow::forget_read(index);
            made.pending.insert(
                index,
                PendingTexture {
                    record,
                    copy: Arc::downgrade(&copy),
                    placed,
                    resource: resource as usize,
                    memory: memory as usize,
                    layout,
                    demanded: false,
                    desc,
                },
            );
        } else {
            let cache = (cache && !placed)
                .then(|| texture_cache(made, record_key(record)))
                .flatten();
            let view = TextureView::Srv(desc);
            write_texture_view(&copy, view, self.cpu(index), cache);
            let source = view.for_ambient(&copy.texture);
            self.keep(index, Some((copy.texture.clone(), source)));
            self.viewed.set(index, Some(&resolved), true);
            self.ready[index as usize].store(record_key(record), Ordering::Release);
        }
        true
    }

    /// Visits current pending slots under the mirror lock. A raw title
    /// descriptor copy can supersede a pending record without calling mirror.
    /// Recheck the bytes and generation even when heap sync saw no writes.
    pub(super) fn service_pending(
        &self,
        start: u64,
        step: u64,
        tables: Option<&std::collections::BTreeSet<u32>>,
        heap_read: bool,
        pins: &mut Vec<ViewPin>,
        mut upload: Option<&mut dyn FnMut(super::PendingUpload)>,
    ) {
        // Normal descriptor writes and tracked raw writes have already
        // mirrored this heap. A demanded slot can still be lazy (placed or
        // evicted) or contain a newly copied record in untracked memory.
        if upload.is_some()
            && let Some(tables) = tables
        {
            static REUSE: OnceLock<bool> = OnceLock::new();
            let reuse = *REUSE.get_or_init(|| crate::switch::on("VICTORY_REUSE_TABLE_PINS"));
            // A collision only loses reuse. Each entry names a pin still
            // held in pins, and a live descriptor generation is nonzero.
            let mut pinned = [0u64; 512];
            let mut ready_visits = 0;
            let mut reused = 0;
            for &index in tables.range(..self.count) {
                let handle = start + u64::from(index) * step;
                // Preserve the acquire before reading the title header.
                let _ = self.ready[index as usize].load(Ordering::Acquire);
                let Some(record @ [tag, view, generation]) = capture_header(handle)
                else {
                    continue;
                };
                if tag == VIEW_TAG
                    && self.ready[index as usize].load(Ordering::Acquire) == record_key(record)
                {
                    let key = record_key(record);
                    ready_visits += 1;
                    // The first occurrence stays pinned through submission.
                    // Placed views still need a per-slot replacement check.
                    let cached = (view ^ generation.rotate_left(7)) as usize % pinned.len();
                    if reuse && key != 0 && pinned[cached] == key {
                        reused += 1;
                        continue;
                    }
                    if let Some(pin) = view_at(view, generation) {
                        let placed = is_placed_texture(&pin);
                        let remake = placed && self.needs_remake(index, &pin, true);
                        if reuse && !placed {
                            pinned[cached] = key;
                        }
                        pins.push(pin);
                        if !remake {
                            continue;
                        }
                    }
                }
                let previous =
                    self.made.lock().unwrap_or_else(|e| e.into_inner()).keys[index as usize];
                // The title can unmap a placed heap after the kernel read.
                // Keep the same checked snapshot path used by heap sync.
                if let Ok(Some(pin)) =
                    self.mirror_record(index, true, Some((previous, handle)), || record)
                {
                    pins.push(pin);
                }
            }
            crate::queue::note_table_pins(ready_visits, reused);
        }
        let mut made = self.made.lock().unwrap_or_else(|e| e.into_inner());
        let Mirrored { keys, pending, .. } = &mut *made;
        pending.retain(|&index, texture| {
            let Some(copy) = texture.copy.upgrade() else {
                keys[index as usize] = 0;
                self.make_null(index);
                self.note_address(index, None);
                return false;
            };
            let handle = start + u64::from(index) * step;
            let ready = copy.initialized.load(Ordering::Acquire);
            let demanded = (heap_read && texture.demanded)
                || tables.is_some_and(|slots| slots.contains(&index));
            if !ready && (!demanded || upload.is_none()) && !copy.evicted.load(Ordering::Acquire) {
                return true;
            }
            let Some(pin) = view_at(texture.record[1], texture.record[2]) else {
                keys[index as usize] = 0;
                self.make_null(index);
                self.note_address(index, None);
                return false;
            };
            pins.push(pin);
            #[cfg(test)]
            if ready
                && RELEASE_BEFORE_PENDING_PUBLICATION.with(|release| {
                    release.get() == Some(texture.resource) && release.replace(None).is_some()
                })
            {
                assert!(crate::object::release_ref(texture.resource as crate::object::This) > 0);
            }
            let current = texture.current(&copy);
            let same_record = super::with_current_record(handle, texture.record, || {
                if !current || !ready {
                    return;
                }
                self.invalidate_ambient(index);
                write_texture_srv(&copy, texture.desc.as_ref(), self.cpu(index));
                let source = TextureView::Srv(texture.desc).for_ambient(&copy.texture);
                self.keep(index, Some((copy.texture.clone(), source)));
                self.viewed
                    .set(index, Some(&View::Shader(texture.resource, None)), true);
                self.ready[index as usize].store(record_key(texture.record), Ordering::Release);
            });
            if !same_record {
                keys[index as usize] = 0;
                self.make_null(index);
                self.note_address(index, None);
                return false;
            }
            if !current {
                // Keep the title record so a subsequent shader read can
                // make its placed view again. The host slot is already null.
                self.viewed
                    .set(index, pins.last().map(|view| &**view), false);
                crate::shadow::forget_read(index);
                return false;
            }
            if !ready
                && let Some(upload) = upload.as_mut()
                && let Some(bytes) = copy.gated_upload_bytes()
            {
                copy.stamp();
                upload(super::PendingUpload {
                    copy,
                    memory: texture.memory,
                    layout: texture.layout,
                    bytes,
                    initial: true,
                });
            }
            !ready
        });
        // Ready shared views still need their ordinary refreshes. Match the
        // current record and the exact resource held by the GPU slot before
        // reading its title backing, just as for a pending first fill.
        if let (Some(tables), Some(upload)) = (tables, upload) {
            for &index in tables {
                let Some(&key) = keys.get(index as usize) else {
                    continue;
                };
                if key == 0 || self.ready[index as usize].load(Ordering::Acquire) != key {
                    continue;
                }
                let Some(resource) = self.viewed.texture_resource(index) else {
                    continue;
                };
                let (view, generation) = (key as u32, (key >> 32) as u32);
                let Some(pin) = view_at(view, generation) else {
                    continue;
                };
                pins.push(pin);
                let Some(copy) = crate::resource::made_copy(resource as crate::object::This) else {
                    continue;
                };
                let Some(bytes) = copy.gated_upload_bytes() else {
                    continue;
                };
                if !copy.initialized.load(Ordering::Acquire)
                    || copy.evicted.load(Ordering::Acquire)
                    || copy.host_written.load(Ordering::Relaxed)
                    || !self
                        .kept
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .get(index as usize)
                        .is_some_and(|(held, _)| *held == copy.texture)
                {
                    continue;
                }
                let Some((memory, layout)) =
                    crate::resource::texture_memory(resource as crate::object::This)
                else {
                    continue;
                };
                let handle = start + u64::from(index) * step;
                // The pin above keeps this generation's resources live.
                // No shared descriptor is changed on this path.
                if header_matches(handle, [VIEW_TAG, view, generation]) {
                    copy.stamp();
                    upload(super::PendingUpload {
                        copy,
                        memory: memory as usize,
                        layout,
                        bytes,
                        initial: false,
                    });
                }
            }
        }
    }

    fn worker_texture(&self, view: &View) -> bool {
        !self.sampler
            && crate::heap_worker::on()
            && matches!(view, View::Shader(..) | View::Placed(false, ..))
            && is_texture(view)
    }

    fn enqueue_texture(&self, index: u32) {
        if !self.sampler && crate::heap_worker::on() {
            let mut uploads = self.uploads.lock().unwrap_or_else(|e| e.into_inner());
            uploads.written(index, crate::queue::presented());
        }
    }

    fn remove_upload(&self, index: u32) {
        if !self.sampler && crate::heap_worker::on() {
            self.uploads
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .remove(index);
        }
    }

    /// Publishes host slot `index` from `handle` if the title slot still holds
    /// record `key`. Called inside `Host::execute` once the texture's upload is
    /// recorded in the submission's upload list (`crate::heap_worker`).
    pub(crate) fn publish(&self, index: u32, handle: u64, key: u64) -> bool {
        if handle == 0 || index >= self.count {
            return false;
        }
        // SAFETY: the handle is a descriptor slot in title memory.
        let [tag, i, generation] = unsafe { (handle as *const [u32; 3]).read_unaligned() };
        if tag != VIEW_TAG || (u64::from(generation) << 32 | u64::from(i)) != key {
            return false;
        }
        let previous = self.synced[index as usize].load(Ordering::Acquire);
        self.mirror_record(index, true, Some((previous, handle)), || {
            [tag, i, generation]
        })
        .is_ok_and(|view| view.is_some())
            && self.slot_ready(index, handle)
    }

    /// Whether slot `index` currently shows a non-null host view made from the
    /// exact record at `handle`.
    pub(crate) fn texture_slot_made_from(&self, index: u32, handle: u64) -> bool {
        if handle == 0 || index >= self.count {
            return false;
        }
        // SAFETY: descriptor slot in title memory.
        let [tag, i, generation] = unsafe { (handle as *const [u32; 3]).read_unaligned() };
        if tag != VIEW_TAG {
            return false;
        }
        let key = u64::from(generation) << 32 | u64::from(i);
        self.ready[index as usize].load(Ordering::Acquire) == key
    }

    pub(crate) fn note_texture_slot_range(&self, index: u32, write: u64, range: (usize, usize)) {
        self.uploads
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .note_range(index, write, range);
    }

    pub(crate) fn wake_remapped_textures(
        &self,
        ranges: &std::collections::BTreeMap<usize, usize>,
        frame: u64,
    ) -> bool {
        self.uploads
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .wake_remapped(ranges, frame)
    }

    pub(crate) fn texture_slots_due(&self, frame: u64, max: usize) -> Vec<(u32, u64)> {
        self.uploads
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .due(frame, max)
    }

    pub(crate) fn texture_slot_done(&self, index: u32, write: u64) {
        self.uploads
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .done(index, write);
    }

    pub(crate) fn texture_slot_later(&self, index: u32, write: u64, frame: u64, max_wait: u64) {
        self.uploads
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .later(index, write, frame, max_wait);
    }

    pub(crate) fn prioritize_texture_slots(&self, indices: &[u32], frame: u64) {
        self.uploads
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .prioritize(indices, frame);
    }

    pub(crate) fn requeue_texture_slots(&self, indices: &[u32], frame: u64) {
        self.uploads
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .requeue(indices, frame);
    }

    #[allow(dead_code)] // hook for urgent bake-source pre-upload
    pub(crate) fn texture_slots_queued(&self, indices: &[u32]) -> Vec<(u32, u64)> {
        self.uploads
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .queued(indices)
    }

    pub(crate) fn has_queued_textures(&self) -> bool {
        !self
            .uploads
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .is_empty()
    }

    /// The view host slot `index` was made from, in a few words, with the
    /// title and host addresses of a buffer view (diagnostics). "null" for
    /// a slot never made or left null.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn describe_slot(&self, index: u32) -> Option<String> {
        let key = *self
            .made
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .keys
            .get(index as usize)?;
        // A view the host could not make leaves a null view, its record
        // noted all the same; it keeps no resource.
        let kept = self
            .kept
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(index as usize)
            .is_some();
        if key == 0 || (!kept && !self.sampler) {
            return Some("null".into());
        }
        let view = view_at(key as u32, (key >> 32) as u32)?;
        let mut text: String = format!("{view:?}").chars().take(120).collect();
        if let Some((address, len)) = buffer_range(&view) {
            let host = crate::shadow::find(address, len.max(1)).map(|s| s.gpu(address));
            text.push_str(&format!(" title {address:#x}+{len:#x} host {host:#x?}"));
        }
        Some(text)
    }

    /// Title slots viewing one of the host textures `textures` (raw pointers),
    /// which are given up: each gets a null view of its kind and holds the
    /// texture no more (retired until the running submission has run), its
    /// record stays noted so heap syncs leave it, and the next draw reading
    /// it through a table or shader reading it through the heap (its read
    /// mark, cleared) makes it again. Slots past `count` stay made: recorded
    /// lists and host copies already name them, and there is no title record
    /// from which to make them again.
    /// The slots are found by a walk of the heap, once per pass that gives
    /// textures up: a map of each texture's slots, kept at every slot the
    /// title rewrote, cost ~25 us a slot (a texture viewed in thousands of
    /// slots had its list searched each time; 62,000 slots a frame, run
    /// evict3).
    pub(crate) fn evict(&self, textures: &std::collections::HashSet<usize>) {
        // Keep a title slot from being rebound between testing its held texture
        // and clearing the descriptor. Mirroring takes made before kept as well.
        let mut made = self.made.lock().unwrap_or_else(|e| e.into_inner());
        let slots: Vec<u32> = {
            let kept = self.kept.lock().unwrap_or_else(|e| e.into_inner());
            kept.iter()
                .filter(|&(i, k)| {
                    i < self.count as usize
                        && matches!(k.1, None | Some(HeldSource::Ambient(_)))
                        && textures.contains(&(windows::core::Interface::as_raw(&k.0) as usize))
                })
                .map(|(i, _)| i as u32)
                .collect()
        };
        made.pending.retain(|&index, pending| {
            use windows::core::Interface;
            if pending
                .copy
                .upgrade()
                .is_some_and(|copy| textures.contains(&(copy.texture.as_raw() as usize)))
            {
                self.viewed.set(index, None, false);
                crate::shadow::forget_read(index);
                return false;
            }
            true
        });
        for index in slots {
            self.ready[index as usize].store(0, Ordering::Relaxed);
            let key = made.keys.get(index as usize).copied().unwrap_or(0);
            let view = (key != 0)
                .then(|| view_at(key as u32, (key >> 32) as u32))
                .flatten();
            self.invalidate_ambient(index);
            self.invalidate_texture_binding(index);
            match &view {
                Some(v) => null_for(v, self.cpu(index)),
                None => null_view(self.cpu(index)),
            }
            self.keep(index, None);
            if index < self.count {
                self.viewed.set(index, view.as_deref(), false);
                crate::shadow::forget_read(index);
                if view.as_deref().is_some_and(is_unordered_texture) {
                    self.nulled_unordered
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .push(index);
                }
            }
        }
    }

    /// Makes again the unordered texture views [`HostHeap::evict`] left
    /// null, the title's heap starting at `start` with `step` bytes a slot:
    /// shaders indexing the heap may write through them now.
    pub(crate) fn remake_unordered(&self, start: u64, step: u64) -> usize {
        let slots = std::mem::take(
            &mut *self
                .nulled_unordered
                .lock()
                .unwrap_or_else(|e| e.into_inner()),
        );
        for &index in &slots {
            {
                let mut made = self.made.lock().unwrap_or_else(|e| e.into_inner());
                self.forget_record(&mut made, index);
            }
            self.mirror(index, start + u64::from(index) * step, true);
        }
        slots.len()
    }

    /// The host textures (raw pointers) slots `slots` view.
    pub(crate) fn textures_at(&self, slots: &[u32]) -> Vec<usize> {
        let kept = self.kept.lock().unwrap_or_else(|e| e.into_inner());
        slots
            .iter()
            .filter_map(|&i| {
                kept.get(i as usize)
                    .filter(|k| matches!(k.1, None | Some(HeldSource::Ambient(_))))
            })
            .map(|k| windows::core::Interface::as_raw(&k.0) as usize)
            .collect()
    }

    /// The view of slot `index` if it is a placed texture's left without a
    /// host view (made when read).
    pub(crate) fn unmade_placed(&self, index: u32) -> Option<View> {
        let made = self.made.lock().unwrap_or_else(|e| e.into_inner());
        let key = *made.keys.get(index as usize)?;
        if made.pending.contains_key(&index) {
            return None;
        }
        let view = (key != 0)
            .then(|| view_at(key as u32, (key >> 32) as u32))
            .flatten()?;
        let kept = self
            .kept
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(index as usize)
            .is_some();
        (is_placed_texture(&view) && !kept).then(|| (*view).clone())
    }

    /// How `make_read` should handle an unmade slot when `heap_worker::on()`.
    pub(crate) fn unmade_read_kind(&self, index: u32) -> UnmadeRead {
        let Some(key) = self
            .made
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .keys
            .get(index as usize)
            .copied()
        else {
            return UnmadeRead::None;
        };
        if key == 0 {
            return UnmadeRead::None;
        }
        let kept = self
            .kept
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(index as usize)
            .is_some();
        if kept {
            return UnmadeRead::None;
        }
        let Some(view) = view_at(key as u32, (key >> 32) as u32) else {
            return UnmadeRead::None;
        };
        if is_texture_view(&view) {
            UnmadeRead::Background
        } else if is_placed_texture(&view) {
            UnmadeRead::Sync
        } else {
            UnmadeRead::None
        }
    }

    /// A shader read slot `index` through the heap: the texture it views
    /// was used now.
    pub(crate) fn stamp_read(&self, index: u32) {
        let key = {
            let mut made = self.made.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(texture) = made.pending.get_mut(&index) {
                texture.demanded = true;
            }
            made.keys.get(index as usize).copied().unwrap_or(0)
        };
        let Some(view) = (key != 0)
            .then(|| view_at(key as u32, (key >> 32) as u32))
            .flatten()
        else {
            return;
        };
        let placed = placed_texture(&view);
        let current = placed.as_ref().unwrap_or(&view);
        if let View::Shader(resource, _) = current
            && let Some(t) = crate::resource::made_copy(*resource as crate::object::This)
        {
            t.stamp();
        }
    }

    /// Whether slot `index` was made from a record (tests).
    #[cfg(test)]
    pub(crate) fn made(&self, index: u32) -> bool {
        self.made.lock().unwrap_or_else(|e| e.into_inner()).keys[index as usize] != 0
    }

    /// Whether slot `index` holds a host view of a resource.
    pub(crate) fn kept(&self, index: u32) -> bool {
        self.kept
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(index as usize)
            .is_some()
    }
}

/// Logs `what` the first time it happens.
macro_rules! once {
    ($($arg:tt)*) => {{
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!($($arg)*));
    }};
}

/// A view description from its words, with the format as the host has it.
fn srv_desc(words: &[u32; 10]) -> D3D12_SHADER_RESOURCE_VIEW_DESC {
    let mut words = *words;
    words[0] = crate::resource::pc_format(words[0]);
    // SAFETY: 40 bytes in the host's layout.
    unsafe {
        std::ptr::from_ref(&words)
            .cast::<D3D12_SHADER_RESOURCE_VIEW_DESC>()
            .read_unaligned()
    }
}

/// The unsigned integer format with the layout of a 16-bit float or
/// signed integer format (one, two or four channels), if `format` is one.
pub(crate) fn half_format(format: u32) -> Option<u32> {
    match format {
        // R16G16B16A16_FLOAT and _SINT: R16G16B16A16_UINT.
        10 | 14 => Some(12),
        // R16G16_FLOAT and _SINT: R16G16_UINT.
        34 | 38 => Some(36),
        // R16_FLOAT and _SINT: R16_UINT.
        54 | 59 => Some(57),
        _ => None,
    }
}

/// `view` as a shader is shown it where it moves a buffer's 16-bit
/// channels as unsigned integers
/// (`victory_shader::emit::HostShader::half_views`): a buffer view in a
/// 16-bit float or signed format, in the unsigned format of the same
/// layout, through which every 16-bit pattern is read and stored as it
/// is. `None` for every other view: the title's own serves (an unsigned
/// one already carries the bits; a texture can only be viewed in the
/// formats it was made for).
pub(crate) fn half_view(view: &View) -> Option<View> {
    // Buffer views: dimension 1 in the title's description of either kind.
    const BUFFER: u32 = 1;
    match view {
        View::Shader(resource, Some(desc)) if desc[1] == BUFFER => {
            let mut desc = *desc;
            desc[0] = half_format(desc[0])?;
            Some(View::Shader(*resource, Some(desc)))
        }
        View::Unordered(resource, counter, Some(desc)) if desc[1] == BUFFER => {
            let mut desc = *desc;
            desc[0] = half_format(desc[0])?;
            Some(View::Unordered(*resource, *counter, Some(desc)))
        }
        View::Placed(unordered, resource, desc) if resource[0] == BUFFER && desc[1] == BUFFER => {
            let mut desc = *desc;
            desc[0] = half_format(desc[0])?;
            Some(View::Placed(*unordered, *resource, desc))
        }
        _ => None,
    }
}

/// The format of `view`, a buffer view through which a shader moves
/// 16-bit channels as unsigned integers
/// (`victory_shader::emit::HostShader::half_views`), if those integers
/// are not what the console gives: a format of wider channels, a
/// normalised one or a signed 8-bit one, whose values the console converts
/// to and from halves. `None` for the formats the integers serve exactly
/// (16 bits a channel of any kind, through [`half_view`] where needed, and
/// unsigned channels of fewer bits) and for any other view.
pub(crate) fn half_inexact(view: &View) -> Option<u32> {
    // Buffer views: dimension 1 in the title's description of either kind.
    const BUFFER: u32 = 1;
    // R16G16B16A16, R16G16 and R16 (float, unsigned, signed), then
    // R10G10B10A2, R8G8B8A8, R8G8 and R8 unsigned.
    const EXACT: [u32; 13] = [10, 12, 14, 34, 36, 38, 54, 57, 59, 25, 30, 50, 62];
    let format = match view {
        View::Shader(_, Some(desc)) | View::Unordered(_, _, Some(desc)) if desc[1] == BUFFER => {
            desc[0]
        }
        View::Placed(_, resource, desc) if resource[0] == BUFFER && desc[1] == BUFFER => desc[0],
        _ => return None,
    };
    (!EXACT.contains(&format)).then_some(format)
}

fn uav_desc(words: &[u32; 10]) -> D3D12_UNORDERED_ACCESS_VIEW_DESC {
    let mut words = *words;
    words[0] = crate::resource::pc_format(words[0]);
    // SAFETY: 40 bytes in the host's layout.
    unsafe {
        std::ptr::from_ref(&words)
            .cast::<D3D12_UNORDERED_ACCESS_VIEW_DESC>()
            .read_unaligned()
    }
}

/// Format, structure stride and raw flag from a console buffer descriptor.
fn buffer_layout(view: &View) -> Option<(DXGI_FORMAT, u32, bool)> {
    let (unordered, words) = match view {
        View::Shader(_, Some(desc)) => (false, desc),
        View::Unordered(_, _, Some(desc)) => (true, desc),
        View::Placed(unordered, resource, desc) if resource[0] == 1 => {
            (*unordered, desc[..10].try_into().ok()?)
        }
        _ => return None,
    };
    if words[1] != 1 {
        return None;
    }
    if unordered {
        let desc = uav_desc(words);
        // SAFETY: dimension 1 selects the buffer union member.
        let buffer = unsafe { desc.Anonymous.Buffer };
        let raw = buffer.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0;
        Some((desc.Format, buffer.StructureByteStride, raw))
    } else {
        let desc = srv_desc(words);
        // SAFETY: dimension 1 selects the buffer union member.
        let buffer = unsafe { desc.Anonymous.Buffer };
        let raw = buffer.Flags.0 & D3D12_BUFFER_SRV_FLAG_RAW.0 != 0;
        Some((desc.Format, buffer.StructureByteStride, raw))
    }
}

/// The console byte stride in the shader's view-stride table. A zero
/// tells heap-indexed shaders to read the null buffer view.
pub(crate) fn view_stride(view: &View) -> u32 {
    if matches!(view, View::Constant(..)) {
        return 4;
    }
    let Some((format, stride, raw)) = buffer_layout(view) else {
        return 0;
    };
    buffer_element(stride, raw, format.0).unwrap_or(0).max(1) as u32
}

/// The view-stride table's integer format bits for typed textures and buffers.
fn view_kind(view: &View) -> u32 {
    use victory_shader::emit::{VIEW_INTEGER, VIEW_SIGNED};
    let resource_format = |resource| {
        crate::resource::title_format(resource as crate::object::This)
            .map_or(0, crate::resource::pc_format)
    };
    let kind = |format| match integer_format(format) {
        Some(true) => VIEW_INTEGER | VIEW_SIGNED,
        Some(false) => VIEW_INTEGER,
        None => 0,
    };
    let (unordered, words, default_format): (bool, &[u32; 10], u32) = match view {
        View::Shader(resource, None) | View::Unordered(resource, _, None) => {
            return kind(resource_format(*resource));
        }
        View::Shader(resource, Some(desc)) => (
            false,
            desc,
            if desc[0] == 0 && desc[1] >= 2 {
                resource_format(*resource)
            } else {
                0
            },
        ),
        View::Unordered(resource, _, Some(desc)) => (
            true,
            desc,
            if desc[0] == 0 && desc[1] >= 2 {
                resource_format(*resource)
            } else {
                0
            },
        ),
        View::Placed(unordered, resource, desc) => {
            // DXGI format in the resource description, after mip levels.
            (
                *unordered,
                desc[..10].try_into().unwrap(),
                crate::resource::pc_format(resource[8]),
            )
        }
        _ => return 0,
    };
    // Buffer union members only describe dimension 1; textures are typed.
    let (format, typed) = if words[1] != 1 {
        (
            if words[0] == 0 {
                default_format
            } else {
                words[0]
            },
            words[1] >= 2,
        )
    } else if unordered {
        let desc = uav_desc(words);
        // SAFETY: a buffer view's union member.
        let b = unsafe { desc.Anonymous.Buffer };
        let raw = b.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0;
        (
            desc.Format.0 as u32,
            typed_buffer(b.StructureByteStride, raw),
        )
    } else {
        let desc = srv_desc(words);
        // SAFETY: a buffer view's union member.
        let b = unsafe { desc.Anonymous.Buffer };
        let raw = b.Flags.0 & D3D12_BUFFER_SRV_FLAG_RAW.0 != 0;
        (
            desc.Format.0 as u32,
            typed_buffer(b.StructureByteStride, raw),
        )
    };
    if typed { kind(format) } else { 0 }
}

/// The four-channel host format for a three-channel 32-bit typed SRV.
fn widened_format(format: i32) -> Option<DXGI_FORMAT> {
    match format {
        6 => Some(DXGI_FORMAT_R32G32B32A32_FLOAT),
        7 => Some(DXGI_FORMAT_R32G32B32A32_UINT),
        8 => Some(DXGI_FORMAT_R32G32B32A32_SINT),
        _ => None,
    }
}

/// The fourth component's bits for a widened three-channel SRV. Integer
/// formats must not receive the bit pattern of floating-point 1.0.
fn widened_w(format: i32) -> Option<u32> {
    match format {
        6 => Some(1.0f32.to_bits()),
        7 | 8 => Some(1),
        _ => None,
    }
}

/// Whether `view` is a typed three-channel 32-bit SRV backed by a widened
/// host copy under Wine.
pub(crate) fn three_channel_srv(view: &View) -> bool {
    matches!(view, View::Shader(..) | View::Placed(false, ..))
        && matches!(buffer_layout(view), Some((DXGI_FORMAT(6..=8), 0, false)))
}

/// Whether a DXGI format is an integer one, and if so whether signed:
/// `Some(false)` for `UINT`, `Some(true)` for `SINT`, `None` otherwise.
fn integer_format(format: u32) -> Option<bool> {
    match format {
        3 | 7 | 12 | 17 | 25 | 30 | 36 | 42 | 50 | 57 | 62 => Some(false),
        4 | 8 | 14 | 18 | 32 | 38 | 43 | 52 | 59 | 64 => Some(true),
        _ => None,
    }
}

/// Bytes per element of a buffer view: its structure stride if it has one,
/// 4 for a raw view, else its format's element size (`None` if the format
/// has none).
fn buffer_element(stride: u32, raw: bool, format: i32) -> Option<u64> {
    if stride > 0 {
        Some(u64::from(stride))
    } else if raw {
        Some(4)
    } else {
        crate::resource::element(format as u32).map(|(bytes, _)| bytes)
    }
}

/// Whether a buffer view is typed: no structure stride, not raw.
fn typed_buffer(stride: u32, raw: bool) -> bool {
    stride == 0 && !raw
}

/// Bytes per element of a shader resource buffer view, 0 for a format with
/// none.
fn element_size(desc: &D3D12_SHADER_RESOURCE_VIEW_DESC) -> u64 {
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { desc.Anonymous.Buffer };
    let raw = buffer.Flags.0 & D3D12_BUFFER_SRV_FLAG_RAW.0 != 0;
    buffer_element(buffer.StructureByteStride, raw, desc.Format.0).unwrap_or(0)
}

/// Writes slot `index`'s words into the view addresses at `data` (the
/// caller keeps `index` below [`HostHeap::address_slots`]), 16 bytes a
/// slot: a buffer view's title address (dword 0 low, dword 1 bits 0-15
/// high), a texture view's size fields ([`texture_words`]), zeros for
/// anything else.
fn write_address(data: usize, index: usize, view: Option<&View>) {
    let desc = match view {
        Some(View::Shader(_, Some(desc))) => Some(desc.as_slice()),
        Some(View::Placed(false, _, desc)) => Some(&desc[..10]),
        _ => None,
    };
    let rtas = desc
        .filter(|d| d[1] == D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE.0 as u32)
        .map(|d| u64::from(d[4]) | u64::from(d[5]) << 32);
    let address = rtas
        .or_else(|| view.and_then(buffer_range).map(|(a, _)| a))
        .unwrap_or(0);
    let words = match view.filter(|_| rtas.is_none()).and_then(texture_words) {
        Some(words) => words,
        None => [address as u32, (address >> 32) as u32 & 0xFFFF, 0, 0],
    };
    // SAFETY: the mapped view addresses hold 16 bytes per title and dynamic
    // slot (below `address_slots`).
    unsafe { ((data + 16 * index) as *mut [u32; 4]).write_unaligned(words) };
}

/// A texture view's console image descriptor dwords 0-3 as far as its size
/// goes: width - 1 and height - 1 of the resource's level 0 (dword 1 bits
/// 30-31, dword 2 bits 0-13 and 14-29), dword 3 an identity swizzle, the
/// view's first and last level (bits 12-15, 16-19) and the image type (bits
/// 28-31). `None` for buffers and samplers.
// shortcut: the address, format and tiling fields read 0 (no translated
// shader reads them: the shader refuses such reads where it can tell),
// and the swizzle is the identity whatever the view's component mapping.
fn texture_words(view: &View) -> Option<[u32; 4]> {
    let (resource, desc, unordered) = match view {
        View::Shader(r, desc) => (
            crate::resource::texture_desc(*r as crate::object::This)?,
            *desc,
            false,
        ),
        View::Unordered(r, _, desc) => (
            crate::resource::texture_desc(*r as crate::object::This)?,
            *desc,
            true,
        ),
        View::Placed(unordered, resource, desc) if resource[0] != 1 => {
            (*resource, desc[..10].try_into().ok(), *unordered)
        }
        _ => return None,
    };
    Some(image_words(&resource, desc.as_ref(), unordered))
}

/// [`texture_words`] of a texture of resource description `resource` (as
/// words) through a view of description `desc` (none: the resource's own).
fn image_words(resource: &[u32; 14], desc: Option<&[u32; 10]>, unordered: bool) -> [u32; 4] {
    let width = u64::from(resource[4]) | u64::from(resource[5]) << 32;
    let height = resource[6];
    let (depth, mips) = (resource[7] & 0xFFFF, resource[7] >> 16);
    let (samples, dimension) = (resource[9], resource[0]);
    // Mip count 0: the whole chain.
    let deepest = if dimension == 4 { depth } else { 1 };
    let chain = 64 - (width.max(u64::from(height)).max(u64::from(deepest)) | 1).leading_zeros();
    let mips = if mips == 0 { chain } else { mips };
    // Image types: 8 1D, 9 2D, 10 3D, 11 cube, 12 1D array, 13 2D array,
    // 14 2D MSAA, 15 2D MSAA array.
    let (ty, first, count) = match desc {
        // Shader and unordered view dimensions number alike up to 3D (8).
        Some(d) => {
            let ty = [0, 0, 8, 12, 9, 13, 14, 15, 10, 11, 11]
                .get(d[1] as usize)
                .copied()
                .unwrap_or(0);
            match (unordered, d[1]) {
                (_, 6 | 7) => (ty, 0, 1),
                (true, _) => (ty, d[4], 1),
                (false, _) => (
                    ty,
                    d[4],
                    if d[5] == u32::MAX {
                        mips.saturating_sub(d[4])
                    } else {
                        d[5]
                    },
                ),
            }
        }
        None => {
            let array = depth > 1;
            let ty = match dimension {
                2 if array => 12,
                2 => 8,
                3 if samples > 1 => 14 + u32::from(array),
                3 => 9 + 4 * u32::from(array),
                _ => 10,
            };
            (ty, 0, if unordered { 1 } else { mips })
        }
    };
    let (w, h) = (width.saturating_sub(1) as u32, height.saturating_sub(1));
    let last = first + count.max(1) - 1;
    [
        0,
        (w & 3) << 30,
        (w >> 2 & 0x3FFF) | (h & 0xFFFF) << 14,
        0xFAC | (first & 0xF) << 12 | (last & 0xF) << 16 | ty << 28,
    ]
}

/// The title memory a buffer shader or unordered access view reads: its
/// address and length.
pub(crate) fn buffer_range(view: &View) -> Option<(u64, u64)> {
    let (base, words): (u64, &[u32; 10]) = match view {
        View::Shader(r, Some(desc)) => (
            crate::resource::buffer_bytes(*r as crate::object::This)?.0 as u64,
            desc,
        ),
        View::Placed(false, resource, desc) if resource[0] == 1 => (
            u64::from(desc[10]) | u64::from(desc[11]) << 32,
            desc[..10].try_into().ok()?,
        ),
        View::Unordered(r, _, Some(desc)) => {
            let base = crate::resource::buffer_bytes(*r as crate::object::This)?.0 as u64;
            return unordered_range(base, desc);
        }
        View::Placed(true, resource, desc) if resource[0] == 1 => {
            let base = u64::from(desc[10]) | u64::from(desc[11]) << 32;
            return unordered_range(base, desc[..10].try_into().ok()?);
        }
        _ => return None,
    };
    let desc = srv_desc(words);
    if desc.ViewDimension != D3D12_SRV_DIMENSION_BUFFER {
        return None;
    }
    let element = element_size(&desc);
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { desc.Anonymous.Buffer };
    checked_buffer_range(base, buffer.FirstElement, buffer.NumElements, element)
}

/// D3DMetal accepts raw views of the entire physical reservation. Both
/// root descriptors and computed addresses use offsets from its start. Other
/// buffers use the 512 MiB view limit.
fn raw_elements(bytes: u64, tiled: bool) -> u32 {
    let limit = if tiled {
        crate::tiled::MAX_TILED_BUFFER as u64
    } else {
        512 << 20
    };
    (bytes.min(limit) / 4) as u32
}

fn checked_buffer_range(base: u64, first: u64, count: u32, element: u64) -> Option<(u64, u64)> {
    let start = base.checked_add(first.checked_mul(element)?)?;
    let len = u64::from(count).checked_mul(element)?;
    start.checked_add(len)?;
    Some((start, len))
}

/// The enclosing buffer and the view's title range, independent of any
/// temporary copy required by the view's element alignment.
type BufferView = (std::sync::Arc<crate::shadow::Shadow>, u64, u64);

pub(crate) fn view_owner(view: &View) -> Option<BufferView> {
    let (address, len) = buffer_range(view)?;
    Some((crate::shadow::find(address, len)?, address, len))
}

/// The host uses a raw descriptor for a structured view of whole dwords.
fn structured_as_raw(format: DXGI_FORMAT, stride: u32) -> bool {
    format == DXGI_FORMAT_UNKNOWN
        && stride >= 4
        && stride.is_multiple_of(4)
        && victory_shader::emit::raw_structured_views()
}

/// D3DMetal accepts dword offsets for raw descriptors (tested under Wine).
/// Keep D3D12's base alignment on native drivers.
fn raw_structured_alignment() -> u64 {
    if crate::under_wine() {
        4
    } else {
        u64::from(D3D12_RAW_UAV_SRV_BYTE_ALIGNMENT)
    }
}

fn direct_widen_source(shadow: &crate::shadow::Shadow, address: u64) -> bool {
    // The compute path for UAV writes uses a raw root SRV at this address.
    shadow.is_aliased() && shadow.offset(address).is_multiple_of(16)
}

/// The exact copy the host view reads, including an alignment copy when needed.
pub(crate) fn align_view_copy(
    view: &View,
    (owner, address, len): BufferView,
) -> Option<BufferView> {
    let (format, stride, raw) = buffer_layout(view)?;
    if crate::under_wine()
        && typed_buffer(stride, raw)
        && matches!(view, View::Shader(..) | View::Placed(false, ..))
        && widened_w(format.0).is_some()
        && direct_widen_source(&owner, address)
    {
        return Some((owner, address, len));
    }
    let alignment = if structured_as_raw(format, stride) {
        raw_structured_alignment()
    } else {
        buffer_element(stride, raw, format.0).unwrap_or(1)
    };
    let (shadow, _) = crate::shadow::view_in(owner, address, len, alignment)?;
    Some((shadow, address, len))
}

pub(crate) fn view_copy(view: &View) -> Option<BufferView> {
    align_view_copy(view, view_owner(view)?)
}

/// [`buffer_range`] of an unordered access view of memory at `base`.
fn unordered_range(base: u64, words: &[u32; 10]) -> Option<(u64, u64)> {
    let desc = uav_desc(words);
    if desc.ViewDimension != D3D12_UAV_DIMENSION_BUFFER {
        return None;
    }
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { desc.Anonymous.Buffer };
    let raw = buffer.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0;
    let element = buffer_element(buffer.StructureByteStride, raw, desc.Format.0)?;
    checked_buffer_range(base, buffer.FirstElement, buffer.NumElements, element)
}

/// The host resource a slot made by [`make`] views, and the view's own copy
/// of title memory ([`crate::shadow::for_view`]): the slot holds it, so that
/// draws reading the view find and refresh that copy.
pub(crate) type Held = (ID3D12Resource, Option<HeldSource>);

/// The fields keep the normal and widened copies alive for as long as the
/// slot shows them.
#[allow(dead_code)]
#[derive(Clone)]
pub(crate) enum HeldSource {
    /// The effective host descriptor for an eligible ambient pass.
    Ambient(Box<AmbientTexture>),
    Shadow(std::sync::Arc<crate::shadow::Shadow>),
    /// A widened three-channel view and the normal shadow it reads.
    Widened(
        std::sync::Arc<crate::shadow::Widened>,
        std::sync::Arc<crate::shadow::Shadow>,
    ),
}

fn same_lease(a: Option<&HeldSource>, b: Option<&HeldSource>) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(HeldSource::Shadow(a)), Some(HeldSource::Shadow(b))) => Arc::ptr_eq(a, b),
        (Some(HeldSource::Widened(a, aa)), Some(HeldSource::Widened(b, bb))) => {
            Arc::ptr_eq(a, b) && Arc::ptr_eq(aa, bb)
        }
        _ => false,
    }
}

fn held_backing(held: &Held) -> Option<&std::sync::Arc<crate::shadow::Shadow>> {
    match &held.1 {
        Some(HeldSource::Shadow(shadow) | HeldSource::Widened(_, shadow)) => Some(shadow),
        None | Some(HeldSource::Ambient(_)) => None,
    }
}

#[derive(Clone)]
pub(crate) struct AmbientTexture {
    desc: TextureView,
    watch: Option<crate::ambient_wave::Watch>,
}

#[derive(Clone, Copy)]
pub(crate) enum TextureView {
    Srv(Option<D3D12_SHADER_RESOURCE_VIEW_DESC>),
    Uav(Option<D3D12_UNORDERED_ACCESS_VIEW_DESC>),
}

impl TextureView {
    fn write(self, copy: &HostTexture, at: D3D12_CPU_DESCRIPTOR_HANDLE) {
        match self {
            Self::Srv(desc) => write_texture_srv(copy, desc.as_ref(), at),
            Self::Uav(desc) => {
                // SAFETY: a live texture and a slot in a view heap.
                unsafe {
                    host().unwrap().device.CreateUnorderedAccessView(
                        &copy.texture,
                        None::<&ID3D12Resource>,
                        desc.as_ref().map(std::ptr::from_ref),
                        at,
                    );
                }
            }
        }
    }

    fn for_ambient(self, texture: &ID3D12Resource) -> Option<HeldSource> {
        if !crate::ambient_wave::enabled() {
            return None;
        }
        let default_format = || {
            // SAFETY: the descriptor was made from this live resource.
            unsafe { texture.GetDesc() }.Format
        };
        let format = |d: Option<DXGI_FORMAT>| {
            d.filter(|&value| value != DXGI_FORMAT_UNKNOWN)
                .unwrap_or_else(default_format)
        };
        let eligible = match self {
            Self::Srv(d) => {
                d.is_none_or(|d| d.ViewDimension == D3D12_SRV_DIMENSION_TEXTURE2D)
                    && matches!(
                        format(d.map(|d| d.Format)),
                        DXGI_FORMAT_R32_FLOAT
                            | DXGI_FORMAT_R16_UNORM
                            | DXGI_FORMAT_R24_UNORM_X8_TYPELESS
                            | DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS
                    )
            }
            Self::Uav(d) => {
                d.is_none_or(|d| {
                    matches!(
                        d.ViewDimension,
                        D3D12_UAV_DIMENSION_TEXTURE2D | D3D12_UAV_DIMENSION_TEXTURE3D
                    )
                }) && matches!(
                    format(d.map(|d| d.Format)),
                    DXGI_FORMAT_R32_UINT | DXGI_FORMAT_R11G11B10_FLOAT
                )
            }
        };
        eligible.then(|| {
            HeldSource::Ambient(Box::new(AmbientTexture {
                desc: self,
                watch: None,
            }))
        })
    }
}

/// What a slot made by [`make`] holds, if anything.
type Kept = Option<Held>;

/// Buffer views refused so far (diagnostics: the first are logged).
#[cfg(feature = "diagnostics")]
static REFUSED: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

fn descriptor_shadow(
    start: u64,
    len: u64,
    element: u64,
    backing: Option<&std::sync::Arc<crate::shadow::Shadow>>,
) -> Option<(std::sync::Arc<crate::shadow::Shadow>, bool)> {
    backing
        .map(|shadow| (shadow.clone(), true))
        .or_else(|| crate::shadow::for_view(start, len, element))
}

fn buffer_srv(
    address: u64,
    len: u64,
    mut desc: D3D12_SHADER_RESOURCE_VIEW_DESC,
    at: D3D12_CPU_DESCRIPTOR_HANDLE,
    backing: Option<&std::sync::Arc<crate::shadow::Shadow>>,
) -> Option<Kept> {
    let element = element_size(&desc);
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { &mut desc.Anonymous.Buffer };
    if element == 0 {
        once!(len, "buffer view of elements without a size");
        return None;
    }
    // The copy its own bytes are in, its first element counted from there.
    let (start, bytes) =
        checked_buffer_range(address, buffer.FirstElement, buffer.NumElements, element)?;
    let count = buffer.NumElements;
    let as_raw = structured_as_raw(desc.Format, buffer.StructureByteStride);
    let alignment = if as_raw {
        raw_structured_alignment()
    } else {
        element
    };
    let host = host()?;
    let widening = crate::under_wine()
        && typed_buffer(
            buffer.StructureByteStride,
            buffer.Flags == D3D12_BUFFER_SRV_FLAG_RAW,
        )
        && widened_w(desc.Format.0).is_some();
    // Widen straight from a suitably aligned alias. Only the widened
    // destination needs a typed descriptor.
    let owner = backing
        .cloned()
        .or_else(|| crate::shadow::find(start, bytes));
    if widening
        && let Some(source) = owner
            .as_ref()
            .filter(|source| direct_widen_source(source, start))
    {
        if bytes == 0 || bytes > source.left(start) {
            return None;
        }
        return widen_srv(&host.device, source, start, count, desc, at);
    }
    let Some((shadow, own)) = owner.and_then(|owner| {
        if backing.is_some() {
            Some((owner, true))
        } else {
            crate::shadow::view_in(owner, start, bytes, alignment)
        }
    }) else {
        #[cfg(feature = "diagnostics")]
        if REFUSED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 200 {
            tracing::info!(
                start = format_args!("{start:#x}"),
                len,
                element,
                count,
                "refused buffer view: no host copy"
            );
        }
        return None;
    };
    let count = if as_raw {
        desc.Format = DXGI_FORMAT_R32_TYPELESS;
        buffer.StructureByteStride = 0;
        buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
        buffer.NumElements = u32::try_from(bytes / 4).ok()?;
        buffer.NumElements
    } else {
        count
    };
    let element = if as_raw { 4 } else { element };
    buffer.FirstElement = shadow.offset(start) / element;
    let first = buffer.FirstElement;
    if !in_buffer(&shadow.buffer, first, count, element) {
        #[cfg(feature = "diagnostics")]
        if REFUSED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 200 {
            tracing::info!(
                start = format_args!("{start:#x}"),
                first,
                count,
                element,
                "refused buffer view: empty or past its copy"
            );
        }
        once!(
            first,
            count,
            element,
            "buffer view empty or past its host copy"
        );
        return None;
    }
    let stride = u64::from(buffer.StructureByteStride);
    if widening && let Some(widened) = widen_srv(&host.device, &shadow, start, count, desc, at) {
        return Some(widened);
    }
    if stride > 0 && raw_structured() {
        // Scalar and offset-only buffer reads emit `rawBufferLoad` on a raw
        // handle (`victory_shader::emit`), which D3D12 leaves undefined on a
        // structured SRV (NVIDIA treats the byte offset as an element index,
        // hanging `WindNew` loops). NVIDIA, WARP and D3DMetal all read both
        // raw handles and structured handles accurately from a raw view of the
        // same bytes; under Wine/D3DMetal any dword-aligned start (`+8B` or
        // `+680B`) works too, while native D3D12 requires 16-byte alignment.
        let (byte, dwords) = (first * stride, u64::from(count) * stride / 4);
        let align = if crate::shader_cache::under_wine() { 4 } else { 16 };
        match u32::try_from(dwords) {
            Ok(dwords) if byte.is_multiple_of(align) && stride.is_multiple_of(4) => {
                desc.Format = DXGI_FORMAT_R32_TYPELESS;
                desc.Anonymous.Buffer = D3D12_BUFFER_SRV {
                    FirstElement: byte / 4,
                    NumElements: dwords,
                    StructureByteStride: 0,
                    Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                };
            }
            _ => once!(byte, stride, "structured buffer view not made raw: unaligned"),
        }
    }
    // SAFETY: a live buffer and a slot of our heap.
    unsafe {
        host.device
            .CreateShaderResourceView(&shadow.buffer, Some(&desc), at)
    };
    Some(Some((
        shadow.buffer.clone(),
        own.then(|| HeldSource::Shadow(shadow)),
    )))
}

fn widen_srv(
    device: &ID3D12Device,
    source: &std::sync::Arc<crate::shadow::Shadow>,
    start: u64,
    count: u32,
    mut desc: D3D12_SHADER_RESOURCE_VIEW_DESC,
    at: D3D12_CPU_DESCRIPTOR_HANDLE,
) -> Option<Kept> {
    let w = widened_w(desc.Format.0)?;
    let widened = source.widened(start, u64::from(count) * 12, w)?;
    desc.Format = widened_format(desc.Format.0)?;
    // SAFETY: a buffer SRV and a live widened buffer in a slot of our heap.
    unsafe {
        desc.Anonymous.Buffer.FirstElement = 0;
        device.CreateShaderResourceView(&widened.buffer, Some(&desc), at);
    }
    Some(Some((
        widened.buffer.clone(),
        Some(HeldSource::Widened(widened, source.clone())),
    )))
}

/// Whether structured buffer views are made raw ([`buffer_srv`]): under Wine
/// (D3DMetal) and on NVIDIA hosts, both of which serve structured-handle and
/// raw-handle `rawBufferLoad`s from a raw view of the same bytes.
/// `VICTORY_RAW_STRUCTURED=0` or `=1` overrides.
fn raw_structured() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = match std::env::var("VICTORY_RAW_STRUCTURED").as_deref() {
            Ok("0" | "false" | "off") => false,
            Ok("1" | "true" | "on") => true,
            _ => crate::shader_cache::under_wine() || crate::host::host_vendor() == 0x10DE,
        };
        tracing::info!(on, "structured buffer views made raw");
        on
    })
}

/// Whether `count` elements of `element` bytes from element `first` lie in
/// `buffer`. The console takes a view that is empty or runs past its
/// memory; the host removes the device for one.
fn in_buffer(buffer: &ID3D12Resource, first: u64, count: u32, element: u64) -> bool {
    // SAFETY: a live buffer.
    within(unsafe { buffer.GetDesc() }.Width, first, count, element)
}

fn within(width: u64, first: u64, count: u32, element: u64) -> bool {
    count > 0
        && checked_buffer_range(0, first, count, element)
            .is_some_and(|(start, len)| start + len <= width)
}

/// An unordered access view of title buffer memory at `address` (`len`
/// bytes) on its host copy, as [`buffer_srv`]; false if the copy cannot be
/// written by shaders (upload memory).
/// shortcut: what shaders write stays in the host copy; title memory (and
/// so the CPU, and copies run on the CPU) does not see it.
fn buffer_uav(
    address: u64,
    len: u64,
    mut desc: D3D12_UNORDERED_ACCESS_VIEW_DESC,
    at: D3D12_CPU_DESCRIPTOR_HANDLE,
    backing: Option<&std::sync::Arc<crate::shadow::Shadow>>,
) -> Option<Kept> {
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { &mut desc.Anonymous.Buffer };
    let raw = buffer.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0;
    let element = buffer_element(buffer.StructureByteStride, raw, desc.Format.0).unwrap_or(0);
    if element == 0 {
        once!(len, "unordered buffer view of elements without a size");
        return None;
    }
    let format = desc.Format.0;
    let typed_three_channel =
        typed_buffer(buffer.StructureByteStride, raw) && matches!(format, 6..=8);
    // D3DMetal 3.0 reports typed-UAV support for these formats, then tries
    // to make an unsupported three-channel Metal format and aborts inside the
    // void CreateUnorderedAccessView call. D3D12 exposes no D3DMetal version,
    // so keep the Wine-wide guard until 4.0 is characterized. The caller
    // installs a null UAV and keeps no resource or mark.
    if typed_three_channel && crate::under_wine() {
        once!(
            format,
            "three-channel typed unordered buffer view under Wine"
        );
        return None;
    }
    let (start, bytes) =
        checked_buffer_range(address, buffer.FirstElement, buffer.NumElements, element)?;
    let as_raw = structured_as_raw(desc.Format, buffer.StructureByteStride);
    let alignment = if as_raw {
        raw_structured_alignment()
    } else {
        element
    };
    let (shadow, own) = descriptor_shadow(start, bytes, alignment, backing)?;
    // SAFETY: a live buffer.
    let flags = unsafe { shadow.buffer.GetDesc() }.Flags;
    if flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 == 0 {
        once!("unordered access to a buffer whose host copy shaders cannot write");
        return None;
    }
    if as_raw {
        desc.Format = DXGI_FORMAT_R32_TYPELESS;
        buffer.StructureByteStride = 0;
        buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
        buffer.NumElements = u32::try_from(bytes / 4).ok()?;
    }
    let element = if as_raw { 4 } else { element };
    buffer.FirstElement = shadow.offset(start) / element;
    buffer.CounterOffsetInBytes = 0;
    let (first, count) = (buffer.FirstElement, buffer.NumElements);
    if !in_buffer(&shadow.buffer, first, count, element) {
        once!(
            first,
            count,
            element,
            "unordered buffer view empty or past its host copy"
        );
        return None;
    }
    let host = host()?;
    if typed_three_channel {
        // The COM method has no HRESULT. On native hosts, reject a format
        // the device does not advertise and a device already removed.
        // SAFETY: plain device queries.
        if unsafe { host.device.GetDeviceRemovedReason() }.is_err() || !typed_store(format as u32) {
            once!(
                format,
                "three-channel typed unordered buffer view the host cannot make"
            );
            return None;
        }
    }
    // SAFETY: a live buffer and a slot of our heap.
    unsafe {
        host.device.CreateUnorderedAccessView(
            &shadow.buffer,
            None::<&ID3D12Resource>,
            Some(&desc),
            at,
        )
    };
    // CreateUnorderedAccessView is void and cannot report failure directly.
    // If it removed the native device, do not keep its descriptor or mark.
    // SAFETY: a plain device query.
    if typed_three_channel && unsafe { host.device.GetDeviceRemovedReason() }.is_err() {
        once!(
            format,
            "device removed making a three-channel typed unordered buffer view"
        );
        return None;
    }
    MARK.with(|m| m.set(shadow.mark_entry(start)));
    Some(Some((
        shadow.buffer.clone(),
        own.then(|| HeldSource::Shadow(shadow)),
    )))
}

thread_local! {
    /// The view-mark entry of the unordered buffer view [`make`] last made
    /// on this thread ([`make_marked`]).
    static MARK: std::cell::Cell<Option<(u32, u32)>> = const { std::cell::Cell::new(None) };
}

/// [`make`] with the view-mark entry for writes through a buffer UAV.
fn make_marked_with_backing(
    view: &View,
    at: D3D12_CPU_DESCRIPTOR_HANDLE,
    backing: Option<&std::sync::Arc<crate::shadow::Shadow>>,
) -> (Option<Kept>, Option<(u32, u32)>) {
    MARK.with(|m| m.set(None));
    let kept = make(view, at, backing);
    (kept, MARK.with(std::cell::Cell::take))
}

/// Whether the host can make a shader (or unordered access) view of
/// `texture` in `format` (none: the texture's own). The console allows any
/// view; the host refuses (and removes the device for) views without the
/// resource's permission, of typed depth, and in a format neither of the
/// resource's family nor among the formats it was made castable to
/// (`castable`, `resource::castable`). A view of a typed texture in another
/// format of its family takes the texture's format unless it is castable.
/// shortcut: such a view (sRGB of UNORM, say, on a texture the host made
/// without castable formats) reads without its conversion.
fn fits(copy: &crate::resource::HostTexture, mut format: Option<&mut DXGI_FORMAT>, unordered: bool) -> bool {
    use crate::resource::{DEPTH_FORMATS, family};
    // SAFETY: a live resource.
    let d = unsafe { copy.texture.GetDesc() };
    let castable = &copy.castable;
    let allowed = if unordered {
        d.Flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 != 0
    } else {
        d.Flags.0 & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE.0 == 0
    };
    let own = d.Format.0 as u32;
    if !allowed || DEPTH_FORMATS.iter().any(|&(typed, _)| typed == own) {
        once!(
            unordered,
            format = own,
            flags = d.Flags.0,
            "view the host does not allow"
        );
        return false;
    }
    // Every family has more than one format, its typeless one first.
    let is_typeless = |f: u32| family(f) == f && family(f + 1) == f;
    let typeless = is_typeless(own);
    if let Some(format) = format.as_deref_mut()
        && unordered
        && let Some(&(_, linear)) = crate::resource::SRGB
            .iter()
            .find(|&&(s, _)| s == format.0 as u32)
    {
        format.0 = linear as i32;
    }
    let view = format.as_ref().map_or(0, |f| f.0 as u32);
    // A typed unordered view in a format the host cannot store to
    // (B5G5R5A1_UNORM on NVIDIA) removes the device when made, in the
    // texture's own format too: the slot stays null, its writes dropped.
    // The format checked is the one the view is made in: a typed
    // texture's own unless the requested one is castable (below).
    let typed = if view == 0 || (!typeless && !castable.contains(&view)) {
        own
    } else {
        view
    };
    if unordered && !(view == 0 && typeless) && !typed_store(typed) {
        once!(
            view = typed,
            own,
            "an unordered view in a format the host cannot store"
        );
        return false;
    }
    if view == 0 {
        return !typeless && (!unordered || typed_store(own));
    }
    // A packed 16-bit target made R8G8B8A8_UNORM on this GPU
    // (`resource::target_format`): views name the format it has.
    if let Some(packed) = copy.promoted_from {
        if packed != view {
            return false;
        }
        if let Some(format) = format {
            *format = d.Format;
        }
        return true;
    }
    if DEPTH_FORMATS.iter().any(|&(typed, _)| typed == view) || is_typeless(view) {
        once!(unordered, view, own, "view the host does not allow");
        return false;
    }
    let cast = castable.contains(&view);
    if family(view) != family(own) && !cast {
        once!(view, own, "view format of another family");
        return false;
    }
    let effective = if typeless || cast { view } else { own };
    if unordered && !typed_store(effective) {
        once!(
            view = effective,
            own,
            "an unordered view in a format the host cannot store"
        );
        return false;
    }
    if let Some(format) = format
        && !typeless
        && !cast
    {
        *format = d.Format;
    }
    true
}

/// Whether a view without a description of `resource` (a title texture)
/// may take its host texture's defaults: not when the host texture has
/// another format than the title's (typeless, or shared with a title
/// texture of another format of the same memory).
/// shortcut: such views are null; spell the description out (dimension,
/// mips) if the title makes them.
fn default_view(resource: crate::object::This) -> bool {
    let other = crate::resource::view_format(resource);
    if other.is_some() {
        once!(
            format = other,
            "a view without a description of a texture whose host texture has another format"
        );
    }
    other.is_none()
}

fn texture_srv_desc(
    resource: crate::object::This,
    copy: &HostTexture,
    desc: Option<&[u32; 10]>,
) -> Option<Option<D3D12_SHADER_RESOURCE_VIEW_DESC>> {
    let mut desc = desc.map(srv_desc);
    if desc.is_none() && !default_view(resource) {
        return None;
    }
    if let Some(desc) = desc.as_mut()
        && desc.Format.0 == 0
        && let Some(format) = crate::resource::view_format(resource)
    {
        desc.Format = DXGI_FORMAT(format as i32);
    }
    if copy.base_level != 0
        && let Some(desc) = desc.as_mut()
    {
        // SAFETY: a live host copy, with the retained mip count.
        let own_mips = u32::from(unsafe { copy.texture.GetDesc() }.MipLevels);
        from_base(desc, copy.base_level, own_mips);
    }
    fits(
        copy,
        desc.as_mut().map(|d| &mut d.Format),
        false,
    )
    .then_some(desc)
}

fn from_base(desc: &mut D3D12_SHADER_RESOURCE_VIEW_DESC, base: u32, own_mips: u32) {
    if own_mips == 0 {
        return;
    }
    let clamp = |most: &mut u32, levels: &mut u32, lod: &mut f32| {
        let lost = base.saturating_sub(*most);
        *most = most.saturating_sub(base).min(own_mips - 1);
        if *levels != u32::MAX {
            *levels = levels.saturating_sub(lost).clamp(1, own_mips - *most);
        }
        *lod = (*lod - base as f32).max(0.0);
    };
    // SAFETY: each arm selects the union member named by the dimension.
    unsafe {
        match desc.ViewDimension {
            D3D12_SRV_DIMENSION_TEXTURE2D => {
                let t = &mut desc.Anonymous.Texture2D;
                clamp(
                    &mut t.MostDetailedMip,
                    &mut t.MipLevels,
                    &mut t.ResourceMinLODClamp,
                );
            }
            D3D12_SRV_DIMENSION_TEXTURE2DARRAY => {
                let t = &mut desc.Anonymous.Texture2DArray;
                clamp(
                    &mut t.MostDetailedMip,
                    &mut t.MipLevels,
                    &mut t.ResourceMinLODClamp,
                );
            }
            D3D12_SRV_DIMENSION_TEXTURECUBE => {
                let t = &mut desc.Anonymous.TextureCube;
                clamp(
                    &mut t.MostDetailedMip,
                    &mut t.MipLevels,
                    &mut t.ResourceMinLODClamp,
                );
            }
            D3D12_SRV_DIMENSION_TEXTURECUBEARRAY => {
                let t = &mut desc.Anonymous.TextureCubeArray;
                clamp(
                    &mut t.MostDetailedMip,
                    &mut t.MipLevels,
                    &mut t.ResourceMinLODClamp,
                );
            }
            _ => {}
        }
    }
}

fn write_texture_srv(
    copy: &HostTexture,
    desc: Option<&D3D12_SHADER_RESOURCE_VIEW_DESC>,
    at: D3D12_CPU_DESCRIPTOR_HANDLE,
) {
    let Some(host) = host() else { return };
    copy.viewed
        .store(true, std::sync::atomic::Ordering::Relaxed);
    // SAFETY: a live texture and a slot of our heap.
    unsafe {
        host.device
            .CreateShaderResourceView(&copy.texture, desc.map(std::ptr::from_ref), at)
    };
}

fn null_texture_srv(
    copy: &HostTexture,
    desc: Option<&D3D12_SHADER_RESOURCE_VIEW_DESC>,
    at: D3D12_CPU_DESCRIPTOR_HANDLE,
) {
    if let Some(desc) = desc.filter(|desc| desc.Format != DXGI_FORMAT_UNKNOWN) {
        null_texture(desc.ViewDimension.0 as u32, desc.Format, at);
        return;
    }
    // A null SRV has no resource from which to infer DXGI_FORMAT_UNKNOWN.
    // SAFETY: the host texture is live.
    let texture = unsafe { copy.texture.GetDesc() };
    let dimension = desc.map_or_else(
        || {
            let array = texture.DepthOrArraySize > 1;
            match texture.Dimension {
                D3D12_RESOURCE_DIMENSION_TEXTURE1D => 2 + u32::from(array),
                D3D12_RESOURCE_DIMENSION_TEXTURE3D => 8,
                _ if texture.SampleDesc.Count > 1 => 6 + u32::from(array),
                _ => 4 + u32::from(array),
            }
        },
        |desc| desc.ViewDimension.0 as u32,
    );
    null_texture(dimension, texture.Format, at);
}

/// Whether the host stores to typed unordered views of `format`.
fn typed_store(format: u32) -> bool {
    static KNOWN: [AtomicU8; 256] = [const { AtomicU8::new(0) }; 256];
    let cached = KNOWN.get(format as usize);
    if let Some(answer) = cached.map(|entry| entry.load(Ordering::Relaxed))
        && answer != 0
    {
        return answer == 2;
    }
    let mut support = D3D12_FEATURE_DATA_FORMAT_SUPPORT {
        Format: DXGI_FORMAT(format as i32),
        ..Default::default()
    };
    let ok = host().is_some_and(|h| {
        // SAFETY: a plain query into a struct of its size.
        unsafe {
            h.device.CheckFeatureSupport(
                D3D12_FEATURE_FORMAT_SUPPORT,
                (&raw mut support).cast(),
                size_of_val(&support) as u32,
            )
        }
        .is_ok()
            && support.Support1.0 & D3D12_FORMAT_SUPPORT1_TYPED_UNORDERED_ACCESS_VIEW.0 != 0
    });
    if let Some(cached) = cached {
        cached.store(1 + u8::from(ok), Ordering::Relaxed);
    }
    ok
}

type TextureCache<'a> = Option<(&'a mut TextureTemplates, u64)>;

fn texture_cache(made: &mut Mirrored, key: u64) -> TextureCache<'_> {
    let templates = &mut made.textures[template_shard(key)];
    templates.get_or_init(TextureTemplates::new);
    templates.get_mut()?.as_mut().map(|cache| (cache, key))
}

fn write_texture_view(
    copy: &Arc<HostTexture>,
    view: TextureView,
    at: D3D12_CPU_DESCRIPTOR_HANDLE,
    cache: TextureCache<'_>,
) {
    if let Some((cache, key)) = cache {
        cache.publish(key, copy, view, at);
    } else {
        view.write(copy, at);
    }
}

/// Makes the host descriptor for `view` at `at` (in any host view heap):
/// the host resource it views, if it can.
pub(crate) fn make_view(view: &View, at: D3D12_CPU_DESCRIPTOR_HANDLE) -> Option<ID3D12Resource> {
    make(view, at, None).flatten().map(|k| k.0)
}

/// The UAV description the host actually uses, after format fallback.
/// `Some(None)` means the host infers it from the resource.
pub(crate) fn color_uav(
    resource: crate::object::This,
    copy: &crate::resource::HostTexture,
    words: Option<&[u32; 10]>,
) -> Option<Option<D3D12_UNORDERED_ACCESS_VIEW_DESC>> {
    let mut desc = words.map(uav_desc);
    if desc.is_none() && !default_view(resource) {
        return None;
    }
    if let Some(desc) = desc.as_mut()
        && desc.Format.0 == 0
        && let Some(format) = crate::resource::view_format(resource)
    {
        desc.Format = DXGI_FORMAT(format as i32);
    }
    if !fits(
        copy,
        desc.as_mut().map(|d| &mut d.Format),
        true,
    ) {
        return None;
    }
    Some(desc)
}

/// Makes the host descriptor for `view` at `at`: the host resource it
/// views, or `None` if it cannot.
fn make(
    view: &View,
    at: D3D12_CPU_DESCRIPTOR_HANDLE,
    backing: Option<&std::sync::Arc<crate::shadow::Shadow>>,
) -> Option<Kept> {
    make_with_texture_cache(view, at, backing, None)
}

fn make_with_texture_cache(
    view: &View,
    at: D3D12_CPU_DESCRIPTOR_HANDLE,
    backing: Option<&Arc<crate::shadow::Shadow>>,
    texture_cache: TextureCache<'_>,
) -> Option<Kept> {
    let host = host()?;
    let device = &host.device;
    match view {
        &View::Constant(address, size) => {
            // Constant buffers in the title's slots, through the heap itself
            // or a table, are read as raw buffers by translated shaders
            // (`victory_shader::emit`, "Constant buffers in the heap"): the
            // console needs 16-byte alignment, a host constant buffer view
            // 256, which most of the title's do not have on the host.
            let shadow = backing
                .cloned()
                .or_else(|| crate::shadow::find(address, u64::from(size)))?;
            let bytes = u64::from(size).min(shadow.left(address));
            let offset = shadow.offset(address);
            let direct = victory_shader::emit::raw_root_cbvs()
                && (shadow.is_aliased() || shadow.is_tiled())
                && offset.is_multiple_of(u64::from(D3D12_RAW_UAV_SRV_BYTE_ALIGNMENT))
                && bytes >= 4;
            let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: DXGI_FORMAT_R32_TYPELESS,
                ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
                Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    Buffer: D3D12_BUFFER_SRV {
                        FirstElement: if direct { offset / 4 } else { 0 },
                        NumElements: (bytes / 4) as u32,
                        Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                        ..Default::default()
                    },
                },
            };
            if direct {
                // The entire view was resolved in a single alias, including
                // the mapped tile coverage for a reserved buffer.
                // SAFETY: a live buffer and a slot of our heap.
                unsafe { device.CreateShaderResourceView(&shadow.buffer, Some(&desc), at) };
                return Some(Some((
                    shadow.buffer.clone(),
                    Some(HeldSource::Shadow(shadow)),
                )));
            }
            buffer_srv(address, bytes, desc, at, backing)
        }
        View::Shader(resource, desc) => {
            let resource = *resource as crate::object::This;
            if let (Some((data, len)), Some(desc)) = (crate::resource::buffer_bytes(resource), desc)
            {
                buffer_srv(data as u64, len, srv_desc(desc), at, backing)
            } else if let Some(copy) = crate::resource::host_copy(resource) {
                copy.viewed
                    .store(true, std::sync::atomic::Ordering::Relaxed);
                let desc = texture_srv_desc(resource, &copy, desc.as_ref())?;
                let view = TextureView::Srv(desc);
                write_texture_view(&copy, view, at, texture_cache);
                Some(Some((
                    copy.texture.clone(),
                    view.for_ambient(&copy.texture),
                )))
            } else {
                None
            }
        }
        View::Unordered(resource, counter, desc) => {
            let resource = *resource as crate::object::This;
            if let (Some((data, len)), Some(desc)) = (crate::resource::buffer_bytes(resource), desc)
            {
                return buffer_uav(data as u64, len, uav_desc(desc), at, backing);
            }
            let copy = crate::resource::host_copy(resource)?;
            copy.viewed
                .store(true, std::sync::atomic::Ordering::Relaxed);
            // Shaders indexing the heap write through this view without
            // moving the texture to a write state: kept from now on, unless
            // `VICTORY_EVICT_HEAP_UAV` lets it go until shaders write
            // through heaps (`crate::resource::evictable`).
            if crate::resource::evict_heap_uav() {
                copy.note_heap_writable();
            } else {
                copy.note_state(0x8);
            }
            let texture = &copy.texture;
            // A depth texture's unordered views write its twin.
            // SAFETY: a live resource.
            if unsafe { texture.GetDesc() }.Flags.0 & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL.0 != 0
            {
                return crate::twin::uav(resource, &copy, desc.as_ref(), at)
                    .map(|twin| Some((twin, None)));
            }
            let view = TextureView::Uav(color_uav(resource, &copy, desc.as_ref())?);
            write_texture_view(&copy, view, at, texture_cache);
            let source = (*counter == 0).then(|| view.for_ambient(texture)).flatten();
            Some(Some((texture.clone(), source)))
        }
        View::Sampler(words) => {
            // SAFETY: 13 words in the host's layout.
            let desc = unsafe {
                std::ptr::from_ref(words)
                    .cast::<D3D12_SAMPLER_DESC>()
                    .read_unaligned()
            };
            // SAFETY: a slot of our sampler heap.
            unsafe { device.CreateSampler(&desc, at) };
            Some(None)
        }
        View::ConsoleSampler(words) => {
            let desc = crate::sampler::console(words, crate::sampler::aniso_point_mip(device));
            // SAFETY: a slot of our sampler heap.
            unsafe { device.CreateSampler(&desc, at) };
            Some(None)
        }
        View::Placed(unordered, resource, desc) => {
            if let Some(view) = placed_texture(view) {
                return make_with_texture_cache(&view, at, None, texture_cache);
            }
            const BUFFER: u32 = 1;
            if resource[0] != BUFFER {
                return None;
            }
            // The resource description's width, after the alignment.
            let len = u64::from(resource[4]) | u64::from(resource[5]) << 32;
            let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
            let words: &[u32; 10] = desc[..10].try_into().unwrap_or(&[0; 10]);
            if *unordered {
                buffer_uav(address, len, uav_desc(words), at, backing)
            } else {
                buffer_srv(address, len, srv_desc(words), at, backing)
            }
        }
    }
}

/// Whether `view` is a sampler descriptor (must only be created on a sampler
/// heap, never on a CBV/SRV/UAV heap).
fn is_sampler(view: &View) -> bool {
    matches!(view, View::Sampler(_) | View::ConsoleSampler(_))
}

fn warn_heap_mismatch(index: u32, sampler_heap: bool, sampler_view: bool) {
    static WARNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if !WARNED.swap(true, Ordering::Relaxed) {
        tracing::warn!(
            index,
            sampler_heap,
            sampler_view,
            "descriptor record kind mismatches host heap type; clearing slot"
        );
    }
}

/// Whether `view` is an unordered view of a resource object's texture.
fn is_unordered_texture(view: &View) -> bool {
    matches!(view, View::Unordered(r, _, _) if crate::resource::texture_memory(*r as crate::object::This).is_some())
}

/// Whether `view` is a view of a texture the title placed in memory itself.
pub(crate) fn is_placed_texture(view: &View) -> bool {
    matches!(view, View::Placed(_, resource, _) if resource[0] != 1)
}

/// Makes what a host view of `view` needs, if it is not made yet: the
/// texture it views (made on first use) or the host copy of the buffer
/// memory it views. The slot's view is then quick to make.
fn prepare(view: &View) -> Option<BufferView> {
    #[cfg(test)]
    tests::while_preparing();
    match view {
        &View::Constant(address, size) => {
            let owner = crate::shadow::find(address, u64::from(size))?;
            let bytes = u64::from(size).min(owner.left(address));
            let (shadow, _) = crate::shadow::view_in(owner, address, bytes / 4 * 4, 4)?;
            Some((shadow, address, bytes))
        }
        View::Placed(..) if is_placed_texture(view) => {
            let _ = placed_texture(view);
            None
        }
        View::Shader(r, _) | View::Unordered(r, _, _)
            if crate::resource::texture_memory(*r as crate::object::This).is_some() =>
        {
            let _ = crate::resource::host_resource(*r as crate::object::This);
            None
        }
        _ => view_copy(view),
    }
}

/// Whether `view` is a shader view of a texture (one placed in memory, or a
/// resource object's): such a slot left null is made when read.
pub(crate) fn is_texture(view: &View) -> bool {
    is_placed_texture(view)
        || matches!(view, View::Shader(r, _) if crate::resource::texture_memory(*r as crate::object::This).is_some())
}

/// Whether `view` is a read-only (SRV) texture view suitable for background
/// creation and upload (`View::Placed(false, ..)` texture or `View::Shader` of
/// a title texture).
pub(crate) fn is_texture_view(view: &View) -> bool {
    matches!(view, View::Placed(false, resource, _) if resource[0] != 1)
        || matches!(view, View::Shader(r, _) if crate::resource::texture_memory(*r as crate::object::This).is_some())
}

/// A view of a texture the title placed in memory itself, as a view of the
/// resource made for that memory (on first use).
pub(crate) fn placed_texture(view: &View) -> Option<View> {
    const BUFFER: u32 = 1;
    let View::Placed(unordered, resource, desc) = view else {
        return None;
    };
    if resource[0] == BUFFER {
        return None;
    }
    let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
    // Only an explicitly typed 2D SRV can restrict which mips are backed.
    // Its MostDetailedMip already clamps shader reads to that suffix.
    let first_mip = if !unordered
        && desc[1] == D3D12_SRV_DIMENSION_TEXTURE2D.0 as u32
        && desc[5] != 0
        && crate::resource::pc_format(desc[0]) == crate::resource::pc_format(resource[8])
    {
        desc[4]
    } else {
        0
    };
    let texture = crate::resource::texture_at_mip(address, resource, first_mip)?;
    let view_desc = desc[..10].try_into().ok();
    Some(if *unordered {
        View::Unordered(texture as usize, 0, view_desc)
    } else {
        View::Shader(texture as usize, view_desc)
    })
}

/// A null view of the kind `view` is, for a slot whose view the host
/// cannot make: unordered access (to a buffer or a 2D texture) or a shader
/// view. Writes to it are dropped, reads give zeros.
fn null_for(view: &View, at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    null_shape(view).write(at);
}

/// Null descriptors depend only on the host view kind and dimension.
#[derive(Clone, Copy)]
enum NullShape {
    Srv(u8),
    TextureUav(u8),
    TypedBufferUav,
    RawBufferUav,
}

impl NullShape {
    fn slot(self) -> usize {
        match self {
            Self::Srv(dimension) => usize::from(dimension),
            Self::TextureUav(dimension) => 11 + usize::from(dimension),
            Self::TypedBufferUav => 20,
            Self::RawBufferUav => 21,
        }
    }

    fn write(self, at: D3D12_CPU_DESCRIPTOR_HANDLE) {
        let desc = match self {
            Self::Srv(1) => return null_buffer_view(at),
            Self::Srv(dimension) => {
                return null_texture(u32::from(dimension), DXGI_FORMAT_R8G8B8A8_UNORM, at);
            }
            Self::TextureUav(dimension) => null_uav_desc(Some(u32::from(dimension))),
            Self::TypedBufferUav => D3D12_UNORDERED_ACCESS_VIEW_DESC {
                Format: DXGI_FORMAT_R32_UINT,
                ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
                ..Default::default()
            },
            Self::RawBufferUav => D3D12_UNORDERED_ACCESS_VIEW_DESC {
                Format: DXGI_FORMAT_R32_TYPELESS,
                ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
                Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                    Buffer: D3D12_BUFFER_UAV {
                        NumElements: 1,
                        Flags: D3D12_BUFFER_UAV_FLAG_RAW,
                        ..Default::default()
                    },
                },
            },
        };
        let Some(host) = host() else { return };
        // SAFETY: a slot of our heap; a null resource is allowed with a desc.
        unsafe {
            host.device.CreateUnorderedAccessView(
                None::<&ID3D12Resource>,
                None::<&ID3D12Resource>,
                Some(&desc),
                at,
            )
        };
    }
}

fn null_shape(view: &View) -> NullShape {
    let is_buffer = |r: &usize| crate::resource::buffer_bytes(*r as crate::object::This).is_some();
    let srv = |dimension| {
        NullShape::Srv(if (2..=10).contains(&dimension) {
            dimension as u8
        } else {
            0
        })
    };
    let (buffer, dimension) = match view {
        View::Unordered(r, _, desc) => (
            is_buffer(r) || desc.is_some_and(|d| d[1] == 1),
            desc.map(|d| d[1]),
        ),
        View::Placed(true, resource, desc) => (resource[0] == 1, Some(desc[1])),
        View::Shader(r, desc) if is_buffer(r) || desc.is_some_and(|d| d[1] == 1) => {
            return NullShape::Srv(1);
        }
        View::Placed(false, resource, _) if resource[0] == 1 => return NullShape::Srv(1),
        View::Shader(_, Some(desc)) => return srv(desc[1]),
        View::Placed(false, _, desc) => return srv(desc[1]),
        View::Constant(..) => return NullShape::Srv(1),
        _ => return NullShape::Srv(0),
    };
    if buffer_layout(view).is_some_and(|(format, stride, _)| structured_as_raw(format, stride)) {
        NullShape::RawBufferUav
    } else if buffer {
        NullShape::TypedBufferUav
    } else {
        let dimension = match dimension {
            Some(n @ (2 | 3 | 5 | 8)) => n as u8,
            _ => 4,
        };
        NullShape::TextureUav(dimension)
    }
}

fn null_template_for(view: &View) -> Option<D3D12_CPU_DESCRIPTOR_HANDLE> {
    static NULLS: OnceLock<Option<(ID3D12DescriptorHeap, usize, usize)>> = OnceLock::new();
    let (_, base, step) = NULLS
        .get_or_init(|| {
            let device = &host()?.device;
            let kind = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
            // SAFETY: a CPU-only heap with storage for every null shape.
            let (heap, base, step) = unsafe {
                let heap: ID3D12DescriptorHeap = device
                    .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                        Type: kind,
                        NumDescriptors: 22,
                        ..Default::default()
                    })
                    .ok()?;
                let base = heap.GetCPUDescriptorHandleForHeapStart().ptr;
                let step = device.GetDescriptorHandleIncrementSize(kind) as usize;
                (heap, base, step)
            };
            let shapes = (0..=10)
                .map(NullShape::Srv)
                .chain([2, 3, 4, 5, 8].map(NullShape::TextureUav))
                .chain([NullShape::TypedBufferUav, NullShape::RawBufferUav]);
            for shape in shapes {
                shape.write(D3D12_CPU_DESCRIPTOR_HANDLE {
                    ptr: base + shape.slot() * step,
                });
            }
            Some((heap, base, step))
        })
        .as_ref()?;
    Some(D3D12_CPU_DESCRIPTOR_HANDLE {
        ptr: base + null_shape(view).slot() * step,
    })
}

/// A null texture UAV of the title's view dimension (1D, 1D or 2D arrays,
/// 3D), 2D if it names none: a 2D null view where the shader writes a 3D
/// texture is a mismatch GPU-based validation reports (the world's wind
/// field, CS_CopyWindFieldBeforeUpdateWithSmooth, 2026-11-24).
fn null_uav_desc(dimension: Option<u32>) -> D3D12_UNORDERED_ACCESS_VIEW_DESC {
    let mut desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
        Format: DXGI_FORMAT_R32_UINT,
        ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
        ..Default::default()
    };
    match dimension.map(|d| D3D12_UAV_DIMENSION(d as i32)) {
        Some(D3D12_UAV_DIMENSION_TEXTURE1D) => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE1D;
        }
        Some(D3D12_UAV_DIMENSION_TEXTURE1DARRAY) => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE1DARRAY;
            desc.Anonymous.Texture1DArray = D3D12_TEX1D_ARRAY_UAV {
                ArraySize: 1,
                ..Default::default()
            };
        }
        Some(D3D12_UAV_DIMENSION_TEXTURE2DARRAY) => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
            desc.Anonymous.Texture2DArray = D3D12_TEX2D_ARRAY_UAV {
                ArraySize: 1,
                ..Default::default()
            };
        }
        Some(D3D12_UAV_DIMENSION_TEXTURE3D) => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE3D;
            desc.Anonymous.Texture3D = D3D12_TEX3D_UAV {
                WSize: u32::MAX,
                ..Default::default()
            };
        }
        _ => {}
    }
    desc
}

/// A buffer view that reads zeros (raw, as translated shaders mostly read
/// buffers).
fn null_buffer_view(at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: DXGI_FORMAT_R32_TYPELESS,
        ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Buffer: D3D12_BUFFER_SRV {
                NumElements: 1,
                Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                ..Default::default()
            },
        },
    };
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device
            .CreateShaderResourceView(None::<&ID3D12Resource>, Some(&desc), at)
    };
}

fn null_texture(dimension: u32, format: DXGI_FORMAT, at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    const TEXTURES: std::ops::RangeInclusive<u32> = 2..=10;
    if !TEXTURES.contains(&dimension) {
        return null_view(at);
    }
    let Some(host) = host() else { return };
    // One mip, one slice or cube, plane 0: fields a null view still checks.
    let mut w = [0u32; 10];
    w[1] = dimension;
    match dimension {
        6 => {}
        7 => w[5] = 1,
        3 | 5 | 10 => (w[5], w[7]) = (1, 1),
        _ => w[5] = 1,
    }
    let mut desc = srv_desc(&w);
    desc.Format = format;
    desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device
            .CreateShaderResourceView(None::<&ID3D12Resource>, Some(&desc), at)
    };
}

/// A view that reads zeros, for slots whose view the host cannot make.
/// shortcut: slots with no view at all (kind unknown) get this null 2D
/// view; null buffer views there did not stop the world's low-address
/// page faults (2026-11-24, run w10), so they are not the cause.
fn null_view(at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: DXGI_FORMAT_R8G8B8A8_UNORM,
        ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Texture2D: D3D12_TEX2D_SRV {
                MipLevels: 1,
                ..Default::default()
            },
        },
    };
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device
            .CreateShaderResourceView(None::<&ID3D12Resource>, Some(&desc), at)
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires Rosetta, D3DMetal and VICTORY_PHYSICAL_TILES=1"]
    fn tiled_buffer_raw_views_reach_last_tile() {
        use crate::compute::{Param, srv, uav};
        use crate::watch::PAGE;
        use windows_sys::Win32::System::Memory::{
            MEM_RELEASE, MEM_RESERVE, PAGE_NOACCESS, VirtualAlloc, VirtualFree,
        };

        assert!(crate::alias::physical_tiles_enabled());
        let host = host().unwrap();
        let size = crate::tiled::MAX_TILED_BUFFER;
        let offset = size - PAGE;
        // SAFETY: the reserved title range has no CPU mappings.
        let base = unsafe { VirtualAlloc(std::ptr::null(), size, MEM_RESERVE, PAGE_NOACCESS) };
        assert!(!base.is_null());
        let title = crate::tiled::TiledAlias::new(base as usize, size).unwrap();
        let physical = Arc::new(crate::alias::PhysicalHeap::new(&host.device, PAGE).unwrap());
        let address = base as usize + offset;
        title.map(address, PAGE, Some((&physical, 0)));
        let words = physical.cpu_address() as *mut [u32; 3];
        // SAFETY: the heap's CPU mapping is live and no GPU work is in flight.
        unsafe { words.write([91, 0, 0]) };
        let shadow = crate::shadow::find(address as u64, 4).unwrap();
        assert!(shadow.is_tiled());
        let heap = HostHeap::new(false, 3).unwrap();
        heap.raw_shadow_at(0, &shadow, false);
        heap.raw_pair_at(1, Some(&shadow));
        let (output, data) = host.shared_buffer(PAGE as u64).unwrap();
        let (root, pipeline) = crate::compute::pipeline(
            r"
cbuffer Address : register(b0) { uint dw; };
StructuredBuffer<uint> root_view : register(t0);
StructuredBuffer<uint> computed_view : register(t1);
RWByteAddressBuffer computed_store : register(u0);
RWByteAddressBuffer dst : register(u1);
[numthreads(1, 1, 1)]
void main() {
    dst.Store2(0, uint2(root_view[dw], computed_view[dw]));
    computed_store.Store(dw * 4 + 8, 97);
}
",
            windows::core::s!("raw views of the last physical tile"),
            0,
            &[
                Param::Table(&[srv(0, 2)]),
                Param::Table(&[uav(0, 1)]),
                Param::Uav { register: 1 },
                Param::Constants { register: 0, count: 1 },
            ],
        )
        .unwrap();
        let list = host.new_list().unwrap();
        shadow.refresh(&list, address as u64, 12);
        // SAFETY: all descriptors refer to live resources on the open list.
        unsafe {
            list.list.SetDescriptorHeaps(&[Some(heap.heap.clone())]);
            list.list.SetComputeRootSignature(&root);
            list.list.SetPipelineState(&pipeline);
            list.list.SetComputeRootDescriptorTable(0, heap.gpu(0));
            list.list.SetComputeRootDescriptorTable(1, heap.gpu(2));
            list.list
                .SetComputeRootUnorderedAccessView(2, output.GetGPUVirtualAddress());
            list.list.SetComputeRoot32BitConstant(3, (offset / 4) as u32, 0);
            list.list.Dispatch(1, 1, 1);
        }
        list.close();
        host.execute(&[&list]);
        // SAFETY: the GPU is done and both mappings are live.
        let (reads, written) = unsafe { (data.cast::<[u32; 2]>().read(), words.read()[2]) };
        drop((list, heap, shadow, title));
        // SAFETY: the test owns the reservation.
        unsafe { VirtualFree(base, 0, MEM_RELEASE) };
        assert_eq!((reads, written), ([91; 2], 97));
    }

    #[test]
    fn capped_texture_array_views_clamp_mips_and_keep_slices() {
        let mut desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2DARRAY,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture2DArray: D3D12_TEX2D_ARRAY_SRV {
                    MostDetailedMip: 1,
                    MipLevels: 8,
                    FirstArraySlice: 2,
                    ArraySize: 3,
                    ResourceMinLODClamp: 4.0,
                    ..Default::default()
                },
            },
            ..Default::default()
        };
        from_base(&mut desc, 3, 4);
        // SAFETY: this is the same union member initialized above.
        let t = unsafe { desc.Anonymous.Texture2DArray };
        assert_eq!(
            (t.MostDetailedMip, t.MipLevels, t.ResourceMinLODClamp),
            (0, 4, 1.0)
        );
        assert_eq!((t.FirstArraySlice, t.ArraySize), (2, 3));

        desc.Anonymous.Texture2DArray = D3D12_TEX2D_ARRAY_SRV {
            MostDetailedMip: 99,
            MipLevels: u32::MAX,
            ..t
        };
        from_base(&mut desc, 3, 4);
        // SAFETY: the view dimension stayed the same.
        let t = unsafe { desc.Anonymous.Texture2DArray };
        assert_eq!((t.MostDetailedMip, t.MipLevels), (3, u32::MAX));
    }

    #[test]
    fn table_cbv_reads_a_bounded_raw_view_of_a_live_alias() {
        use crate::compute::{Param, srv, uav};
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
        };
        if !crate::watch::rosetta() {
            eprintln!("skipping the mach_vm_remap test outside Rosetta");
            return;
        }
        let host = host().expect("a D3DMetal device");
        let len = crate::watch::PAGE;
        // SAFETY: a private allocation with the alignment required for aliasing.
        let data = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                len,
                MEM_RESERVE | MEM_COMMIT,
                PAGE_READWRITE,
            )
        }
        .cast::<u8>();
        assert!(!data.is_null());
        assert!(crate::alias::alias_region(data as usize, len, false));
        let base = data as u64;
        let alias = crate::shadow::find(base, len as u64).unwrap();
        assert!(alias.is_aliased());
        // SAFETY: a live title mapping with space for the test values.
        unsafe {
            data.cast::<[u32; 8]>()
                .write([100, 200, 300, 400, 10, 20, 30, 40])
        };
        let heap = HostHeap::new(false, 2).unwrap();
        let view = View::Constant(base + 16, 12);
        heap.make_at(0, &view);
        if victory_shader::emit::raw_root_cbvs() {
            assert!(Arc::ptr_eq(&heap.table_backing(0).unwrap(), &alias));
        }
        let (result, ptr) = host.shared_buffer(65536).unwrap();
        heap.raw_view_at(1, Some(&result), 0, 16, true);
        let (root, pso) = crate::compute::pipeline(
            "ByteAddressBuffer src : register(t0);
             RWByteAddressBuffer dst : register(u0);
             [numthreads(1, 1, 1)] void main() {
                 dst.Store4(0, uint4(src.Load(0), src.Load(4), src.Load(8), src.Load(12)));
             }",
            windows::core::s!("bounded aliased table CBV"),
            0,
            &[Param::Table(&[srv(0, 1)]), Param::Table(&[uav(0, 1)])],
        )
        .unwrap();
        let list = host.new_list().unwrap();
        // SAFETY: the host views are live; the title writes after the view
        // is created and supplies no upload or refresh before the read.
        unsafe {
            data.add(16).cast::<u32>().write(77);
            list.list.SetDescriptorHeaps(&[Some(heap.heap.clone())]);
            list.list.SetComputeRootSignature(&root);
            list.list.SetComputeRootDescriptorTable(0, heap.gpu(0));
            list.list.SetComputeRootDescriptorTable(1, heap.gpu(1));
            list.list.SetPipelineState(&pso);
            list.list.Dispatch(1, 1, 1);
        }
        list.close();
        host.execute(&[&list]);
        // SAFETY: a mapped destination after the fence.
        assert_eq!(unsafe { ptr.cast::<[u32; 4]>().read() }, [77, 20, 30, 0]);
        drop((list, heap, alias));
        assert!(crate::alias::unalias_region(base as usize));
        // SAFETY: this test owns the allocation; the GPU read finished.
        assert_ne!(unsafe { VirtualFree(data.cast(), 0, MEM_RELEASE) }, 0);
    }

    /// Typed buffer views of integer formats carry their kind in the
    /// view-stride table; float, normalized, raw and structured ones none.
    #[test]
    fn buffer_views_reject_overflow_and_publish_rtas_addresses() {
        assert_eq!(checked_buffer_range(16, 2, 4, 4), Some((24, 16)));
        assert_eq!(checked_buffer_range(0, u64::MAX, 1, 4), None);
        assert_eq!(checked_buffer_range(u64::MAX - 7, 2, 1, 4), None);
        assert_eq!(checked_buffer_range(u64::MAX - 7, 1, 1, 4), None);
        let mut desc = [0u32; 10];
        desc[1] = 11;
        (desc[4], desc[5]) = (0xabcd_1234, 0xa321);
        let mut words = [0u32; 4];
        write_address(
            words.as_mut_ptr() as usize,
            0,
            Some(&View::Shader(0, Some(desc))),
        );
        assert_eq!(words, [0xabcd_1234, 0xa321, 0, 0]);
    }

    #[test]
    fn table_buffer_views_follow_new_shadow_owners() {
        let Some(heap) = HostHeap::new(false, 4) else {
            return;
        };
        let address = crate::watch::tests::pool(2);
        let page = crate::watch::PAGE;
        crate::shadow::register(address, address as u64, (2 * page) as u64);
        let original = crate::shadow::find(address as u64, page as u64).unwrap();
        let mut resource = [0u32; 14];
        (resource[0], resource[4]) = (1, page as u32);
        let mut slots = [[0u32; 8]; 2];
        for (index, slot) in slots.iter_mut().enumerate() {
            let uav = index == 1;
            let mut words = [0u32; 16];
            (words[0], words[1]) = (42, 1); // R32_UINT buffer
            words[if uav { 4 } else { 6 }] = 16;
            words[2] = if uav {
                0
            } else {
                D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING
            };
            (words[10], words[11]) = (address as u32, (address >> 32) as u32);
            let handle = slot.as_mut_ptr() as u64;
            if uav {
                crate::descriptor::create_placed_uav(std::ptr::null(), &resource, &words, handle);
            } else {
                crate::descriptor::create_placed_srv(std::ptr::null(), &resource, &words, handle);
            }
            heap.mirror(index as u32, handle, true).unwrap();
            let kept = heap.kept.lock().unwrap();
            assert_eq!(kept.get(index).map(|k| &k.0), Some(&original.buffer));
        }
        crate::shadow::unregister(address, address as u64);
        crate::shadow::register(address, address as u64, page as u64);
        let replacement = crate::shadow::find(address as u64, page as u64).unwrap();
        assert_ne!(original.buffer, replacement.buffer);
        for (index, slot) in slots.iter().enumerate() {
            heap.mirror(index as u32, slot.as_ptr() as u64, true)
                .unwrap();
            let kept = heap.kept.lock().unwrap();
            assert_eq!(kept.get(index).map(|k| &k.0), Some(&replacement.buffer));
        }
        crate::shadow::unregister(address, address as u64);
        drop(heap);
        crate::watch::tests::free(address);
    }

    #[test]
    fn remapped_texture_slots_leave_backoff_only_for_overlapping_mappings() {
        let mut queue = UploadQueue::default();
        for (index, range) in [(0, (100, 200)), (1, (200, 300))] {
            queue.written(index, 1);
            let write = queue.queued(&[index])[0].1;
            queue.note_range(index, write, range);
            for _ in 0..7 {
                queue.later(index, write, 1, 1024);
            }
        }
        assert!(queue.due(2, 2).is_empty());
        assert!(queue.wake_remapped(&[(190, 200)].into(), 2));
        let due = queue.due(2, 2);
        assert_eq!(
            due.iter().map(|&(index, _)| index).collect::<Vec<_>>(),
            vec![0]
        );
        queue.later(0, due[0].1, 2, 1024);
        assert_eq!(queue.due(3, 2), due, "the retry starts at one frame again");
    }

    #[test]
    fn integer_typed_buffer_views_are_marked_in_the_stride_table() {
        use victory_shader::emit::{VIEW_INTEGER, VIEW_SIGNED};
        // Format, dimension (buffer), first element, count, stride, counter, flags.
        let uav = |format, stride, flags| {
            View::Unordered(0, 0, Some([format, 1, 0, 0, 64, stride, 0, 0, flags, 0]))
        };
        assert_eq!(view_kind(&uav(12, 0, 0)), VIEW_INTEGER, "R16G16B16A16_UINT");
        assert_eq!(view_kind(&uav(7, 0, 0)), VIEW_INTEGER, "R32G32B32_UINT");
        assert_eq!(
            view_kind(&uav(8, 0, 0)),
            VIEW_INTEGER | VIEW_SIGNED,
            "R32G32B32_SINT"
        );
        assert_eq!(view_kind(&uav(57, 0, 0)), VIEW_INTEGER, "R16_UINT");
        assert_eq!(
            view_kind(&uav(59, 0, 0)),
            VIEW_INTEGER | VIEW_SIGNED,
            "R16_SINT"
        );
        assert_eq!(
            view_kind(&uav(64, 0, 0)),
            VIEW_INTEGER | VIEW_SIGNED,
            "R8_SINT"
        );
        assert_eq!(view_kind(&uav(10, 0, 0)), 0, "R16G16B16A16_FLOAT");
        assert_eq!(view_kind(&uav(58, 0, 0)), 0, "R16_SNORM");
        assert_eq!(view_kind(&uav(39, 0, 1)), 0, "raw");
        assert_eq!(view_kind(&uav(0, 16, 0)), 0, "structured");
        // Format, dimension, mapping, first element, count, stride, flags.
        let srv = View::Shader(0, Some([62, 1, 5768, 0, 0, 0, 64, 0, 0, 0]));
        assert_eq!(view_kind(&srv), VIEW_INTEGER, "R8_UINT");
        for (format, kind) in [
            (42, VIEW_INTEGER),
            (59, VIEW_INTEGER | VIEW_SIGNED),
            (62, VIEW_INTEGER),
            (41, 0),
        ] {
            let mut words = [0; 10];
            words[0] = format;
            words[1] = 4; // texture 2D
            assert_eq!(view_kind(&View::Shader(0, Some(words))), kind);
            assert_eq!(view_kind(&View::Unordered(0, 0, Some(words))), kind);
        }
        assert_eq!(view_stride(&uav(12, 0, 0)) | view_kind(&uav(12, 0, 0)), 8 | VIEW_INTEGER);
        assert_eq!(widened_format(6), Some(DXGI_FORMAT_R32G32B32A32_FLOAT));
        assert_eq!(widened_format(7), Some(DXGI_FORMAT_R32G32B32A32_UINT));
        assert_eq!(widened_format(8), Some(DXGI_FORMAT_R32G32B32A32_SINT));
        assert_eq!(widened_format(2), None);
        assert_eq!(widened_w(6), Some(0x3f80_0000));
        assert_eq!(widened_w(7), Some(1));
        assert_eq!(widened_w(8), Some(1));
        assert_eq!(widened_w(2), None);
    }

    #[test]
    fn off_element_structured_views_use_raw_descriptors_on_d3dmetal() {
        if !crate::under_wine() || !victory_shader::emit::raw_structured_views() {
            return;
        }
        let host = host().unwrap();
        let alias = |data: &[u32]| {
            let (buffer, mapped) = host.shared_buffer(65536).unwrap();
            let base = mapped as u64;
            // SAFETY: live mapping with room for all the initial dwords.
            let gpu = unsafe {
                std::ptr::copy_nonoverlapping(data.as_ptr(), mapped.cast::<u32>(), data.len());
                buffer.GetGPUVirtualAddress()
            };
            crate::shadow::Shadow::new_aliased(buffer, mapped, gpu, base, 65536)
        };
        let source = alias(&[10, 20, 30, 40, 50, 60]);
        let dest = alias(&[u32::MAX; 7]);
        let start = source.range().0 + 4;
        let end = dest.range().0 + 4;
        let view = |uav, addr: u64, mut words: [u32; 16]| {
            words[10] = addr as u32;
            words[11] = (addr >> 32) as u32;
            View::Placed(uav, [1, 0, 0, 0, 65536, 0, 1, 0, 0, 1, 0, 0, 0, 0], words)
        };
        let srv = view(
            false,
            start,
            [
                0,
                1,
                D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                0,
                0,
                0,
                1,
                12,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
        );
        let uav = view(true, end, [0, 1, 0, 0, 1, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(view_stride(&srv), 12);
        assert_eq!(view_stride(&uav), 20);
        let (srv_backing, _, _) = align_view_copy(&srv, (source.clone(), start, 12)).unwrap();
        let (uav_backing, _, _) = align_view_copy(&uav, (dest.clone(), end, 20)).unwrap();
        assert!(std::sync::Arc::ptr_eq(&srv_backing, &source));
        assert!(std::sync::Arc::ptr_eq(&uav_backing, &dest));
        // SAFETY: plain descriptor heap creation.
        let heap: ID3D12DescriptorHeap = unsafe {
            host.device
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                    NumDescriptors: 2,
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                    ..Default::default()
                })
        }
        .unwrap();
        // SAFETY: a live heap and the increment for this heap type.
        let (cpu, gpu, step) = unsafe {
            (
                heap.GetCPUDescriptorHandleForHeapStart(),
                heap.GetGPUDescriptorHandleForHeapStart(),
                host.device
                    .GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV),
            )
        };
        let (srv_buffer, _) = make(&srv, cpu, Some(&srv_backing)).flatten().unwrap();
        let uav_cpu = D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: cpu.ptr + step as usize,
        };
        let uav_gpu = D3D12_GPU_DESCRIPTOR_HANDLE {
            ptr: gpu.ptr + u64::from(step),
        };
        let (uav_buffer, _) = make(&uav, uav_cpu, Some(&uav_backing)).flatten().unwrap();
        assert_eq!(srv_buffer, source.buffer);
        assert_eq!(uav_buffer, dest.buffer);
        let (root, pso) = crate::compute::pipeline(
            "ByteAddressBuffer src : register(t0);
             RWByteAddressBuffer dst : register(u0);
             [numthreads(1, 1, 1)] void main() {
                 dst.Store4(0, uint4(src.Load(0), src.Load(4), src.Load(8), src.Load(12)));
             }",
            windows::core::s!("off-element raw views"),
            0,
            &[
                crate::compute::Param::Table(&[crate::compute::srv(0, 1)]),
                crate::compute::Param::Table(&[crate::compute::uav(0, 1)]),
            ],
        )
        .unwrap();
        let list = host.new_list().unwrap();
        // SAFETY: an open list and live resources.
        unsafe {
            list.list.SetDescriptorHeaps(&[Some(heap.clone())]);
            list.list.SetComputeRootSignature(&root);
            list.list.SetComputeRootDescriptorTable(0, gpu);
            list.list.SetComputeRootDescriptorTable(1, uav_gpu);
            list.list.SetPipelineState(&pso);
            list.list.Dispatch(1, 1, 1);
        }
        list.close();
        host.execute(&[&list]);
        assert_eq!(
            dest.read_host(dest.range().0, 7),
            [u32::MAX, 20, 30, 40, 0, u32::MAX, u32::MAX]
        );
    }

    #[test]
    fn off_element_rgb32_reads_the_alias_directly_when_widening() {
        if !crate::under_wine() {
            return;
        }
        let Some(host) = host() else { return };
        let (buffer, data) = host.shared_buffer(65536).expect("a mapped buffer");
        let base = data as u64;
        // SAFETY: a live buffer and three mapped dwords at offset 16.
        let gpu = unsafe {
            data.add(16).cast::<[u32; 3]>().write([11, 22, 33]);
            buffer.GetGPUVirtualAddress()
        };
        let alias = crate::shadow::Shadow::new_aliased(buffer, data, gpu, base, 65536);
        let address = base + 16;
        let view = View::Placed(
            false,
            [1, 0, 0, 0, 65536, 0, 1, 0, 0, 1, 0, 0, 0, 0],
            [
                7,
                1,
                D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                address as u32,
                (address >> 32) as u32,
                0,
                0,
                0,
                0,
            ],
        );
        let (backing, start, len) =
            align_view_copy(&view, (alias.clone(), address, 12)).expect("a buffer source");
        assert!(std::sync::Arc::ptr_eq(&backing, &alias));
        assert_eq!((start, len), (address, 12));
        // SAFETY: plain descriptor-heap creation and handle query.
        let (heap, at) = unsafe {
            let heap: ID3D12DescriptorHeap = host
                .device
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                    NumDescriptors: 1,
                    ..Default::default()
                })
                .expect("a descriptor heap");
            let at = heap.GetCPUDescriptorHandleForHeapStart();
            (heap, at)
        };
        let Some((_, Some(HeldSource::Widened(widened, source)))) =
            make(&view, at, Some(&backing)).flatten()
        else {
            panic!("the descriptor should read a widened buffer");
        };
        assert!(std::sync::Arc::ptr_eq(&source, &alias));
        assert_eq!(widened.read_host(4), [11, 22, 33, 1]);
        // SAFETY: the same live, mapped element before the next refresh.
        unsafe { data.add(16).cast::<[u32; 3]>().write([44, 55, 66]) };
        assert_eq!(source.update(address as usize, 12), 0);
        assert_eq!(widened.read_host(4), [44, 55, 66, 1]);
        drop(heap);
    }

    /// Three-channel 32-bit SRVs are backed by a widened copy on hosts that
    /// read R32G32B32 as R32G32. The copy keeps 12-byte source elements,
    /// supplies w in the source format and follows later title writes.
    #[test]
    fn three_channel_srvs_use_a_widened_refreshed_copy() {
        if !crate::under_wine() {
            return;
        }
        let Some(host) = host() else { return };
        let p = crate::watch::tests::pool(1);
        let len = 4 * 12;
        let first: [u32; 12] = [
            1000, 2000, 3000, 1003, 2003, 3003, 1006, 2006, 3006, 1009, 2009, 3009,
        ];
        // SAFETY: 48 bytes of our pool page.
        unsafe { std::ptr::copy_nonoverlapping(first.as_ptr().cast::<u8>(), p as *mut u8, len) };
        crate::shadow::register(p, p as u64, len as u64);
        let shadow = crate::shadow::find(p as u64, len as u64).expect("a host copy");
        shadow.update(p, len);

        // SAFETY: plain descriptor-heap creation.
        let heap: ID3D12DescriptorHeap = unsafe {
            host.device
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                    NumDescriptors: 4,
                    ..Default::default()
                })
        }
        .expect("a descriptor heap");
        // SAFETY: a live heap.
        let at = unsafe { heap.GetCPUDescriptorHandleForHeapStart() };
        let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R32G32B32_FLOAT,
            ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_SRV {
                    NumElements: 4,
                    ..Default::default()
                },
            },
        };
        let (resource, source) = buffer_srv(p as u64, len as u64, desc, at, None)
            .expect("a view")
            .expect("a kept view");
        let Some(HeldSource::Widened(widened, _)) = source else {
            panic!("a widened source");
        };
        // SAFETY: a live resource.
        let made = unsafe { resource.GetDesc() };
        assert_eq!(made.Width, 4 * 16);
        let float_one = 0x3f80_0000;
        let want = |w| {
            [
                1000, 2000, 3000, w, 1003, 2003, 3003, w, 1006, 2006, 3006, w, 1009, 2009, 3009, w,
            ]
        };
        assert_eq!(widened.read_host(16), want(float_one));

        // SAFETY: a live heap; each later slot is one descriptor stride on.
        let step = unsafe {
            host.device
                .GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV)
        };
        let at_slot = |slot: usize| D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: at.ptr + slot * step as usize,
        };
        let (second_resource, second_source) =
            buffer_srv(p as u64, len as u64, desc, at_slot(1), None)
                .expect("a second view")
                .expect("a second kept view");
        let Some(HeldSource::Widened(second, _)) = second_source else {
            panic!("a second widened source");
        };
        assert!(
            std::sync::Arc::ptr_eq(&widened, &second),
            "one widened copy per range and w"
        );
        drop((second_resource, second));

        let uint_desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R32G32B32_UINT,
            ..desc
        };
        let (uint_resource, uint_source) =
            buffer_srv(p as u64, len as u64, uint_desc, at_slot(2), None)
                .expect("a uint view")
                .expect("a kept uint view");
        let Some(HeldSource::Widened(uint, _)) = uint_source else {
            panic!("a widened uint source");
        };
        let sint_desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R32G32B32_SINT,
            ..desc
        };
        let (sint_resource, sint_source) =
            buffer_srv(p as u64, len as u64, sint_desc, at_slot(3), None)
                .expect("a sint view")
                .expect("a kept sint view");
        let Some(HeldSource::Widened(sint, _)) = sint_source else {
            panic!("a widened sint source");
        };
        assert!(
            std::sync::Arc::ptr_eq(&uint, &sint),
            "integer formats share their widened copy"
        );
        assert!(
            !std::sync::Arc::ptr_eq(&widened, &uint),
            "float and integer w bits need distinct copies"
        );
        assert_eq!(uint.read_host(16), want(1));
        assert_eq!(sint.read_host(16), want(1));

        let changed = [0x1111_1111, 0x2222_2222, 0x3333_3333];
        // SAFETY: one 12-byte element of our pool page.
        unsafe {
            std::ptr::copy_nonoverlapping(changed.as_ptr().cast::<u8>(), (p + 12) as *mut u8, 12)
        };
        assert!(shadow.update(p, len) > 0, "the title write is copied");
        let changed_want = |w| {
            [
                1000,
                2000,
                3000,
                w,
                0x1111_1111,
                0x2222_2222,
                0x3333_3333,
                w,
                1006,
                2006,
                3006,
                w,
                1009,
                2009,
                3009,
                w,
            ]
        };
        assert_eq!(widened.read_host(16), changed_want(float_one));
        assert_eq!(uint.read_host(16), changed_want(1));
        assert_eq!(sint.read_host(16), changed_want(1));

        drop((
            resource,
            widened,
            uint_resource,
            uint,
            sint_resource,
            sint,
            heap,
            shadow,
        ));
        crate::shadow::unregister(p, p as u64);
        crate::watch::tests::free(p);
    }

    /// A placed texture's view the title writes into a heap is left null
    /// and noted once: heap syncs (every submission) pass it by, a draw
    /// reading it through a table makes it.
    #[test]
    fn a_placed_texture_slot_left_null_is_made_once_a_draw_reads_it() {
        if host().is_none() {
            return; // no host GPU here
        }
        let null = std::ptr::null();
        let heap_desc: [u32; 8] = [0, 4, 1, 0, 0, 0, 0, 0];
        let mut heap = null;
        let iid = crate::object::guid(
            0x8efb471d,
            0x616c,
            0x4f49,
            [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51],
        );
        assert_eq!(
            crate::descriptor::create_heap(null, heap_desc.as_ptr().cast(), &iid, &mut heap),
            0
        );
        let title_heap = crate::descriptor::heap(heap);
        // A 4x1 RGBA8 texture, linear, in memory of its own, and a 2D view.
        let memory = vec![0u8; 1 << 16];
        let address = memory.as_ptr() as u64;
        let words: [u32; 14] = [3, 0, 0, 0, 4, 0, 1, 1 | 1 << 16, 28, 1, 0, 0x100, 0, 0];
        let mut view = [0u32; 16];
        view[..6].copy_from_slice(&[28, 4, 0x1688, 0, 0, 1]);
        (view[10], view[11]) = (address as u32, (address >> 32) as u32);
        crate::descriptor::create_placed_srv(null, &words, &view, title_heap.start);
        let host = title_heap.host().expect("a host view heap");
        for _ in 0..2 {
            assert!(host.mirror(0, title_heap.start, false).is_some());
            assert!(host.made(0), "the record is noted");
            assert!(!host.kept(0), "no host texture for it yet");
        }
        assert!(host.mirror(0, title_heap.start, true).is_some());
        assert!(host.kept(0), "made for the draw");
    }

    /// A slot viewing a texture given up (`crate::resource::evict`) holds it
    /// no more: heap syncs pass it by, a draw reading it through a table
    /// makes it again.
    #[test]
    fn a_slot_viewing_a_texture_given_up_is_made_again_when_read() {
        if host().is_none() {
            return; // no host GPU here
        }
        let null = std::ptr::null();
        let heap_desc: [u32; 8] = [0, 4, 1, 0, 0, 0, 0, 0];
        let mut heap = null;
        let iid = crate::object::guid(
            0x8efb471d,
            0x616c,
            0x4f49,
            [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51],
        );
        assert_eq!(
            crate::descriptor::create_heap(null, heap_desc.as_ptr().cast(), &iid, &mut heap),
            0
        );
        let title_heap = crate::descriptor::heap(heap);
        // A 4x1 RGBA8 texture, linear, in memory of its own, and a 2D view.
        let memory = vec![0u8; 1 << 16];
        let address = memory.as_ptr() as u64;
        let words: [u32; 14] = [3, 0, 0, 0, 4, 0, 1, 1 | 1 << 16, 28, 1, 0, 0x100, 0, 0];
        let mut view = [0u32; 16];
        view[..6].copy_from_slice(&[28, 4, 0x1688, 0, 0, 1]);
        (view[10], view[11]) = (address as u32, (address >> 32) as u32);
        crate::descriptor::create_placed_srv(null, &words, &view, title_heap.start);
        let host = title_heap.host().expect("a host view heap");
        assert!(host.mirror(0, title_heap.start, true).is_some());
        assert!(host.kept(0), "made for the draw");
        let texture = crate::resource::texture_at(address, &words).expect("its resource");
        assert!(crate::resource::give_up_now(texture));
        assert!(!host.kept(0), "the slot holds it no more");
        assert!(host.made(0), "its record stays noted");
        assert!(host.mirror(0, title_heap.start, false).is_some());
        assert!(!host.kept(0), "heap syncs pass it by");
        assert!(host.mirror(0, title_heap.start, true).is_some());
        assert!(host.kept(0), "made again for the draw");
        assert!(
            crate::resource::made_copy(texture).is_some(),
            "with a host texture again"
        );
    }

    /// Work to run while a view's texture is prepared, once (tests).
    pub(crate) static WHILE_PREPARING: std::sync::Mutex<Option<crate::descriptor::tests::Work>> =
        std::sync::Mutex::new(None);

    pub(crate) fn while_preparing() {
        let work = WHILE_PREPARING
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take();
        if let Some(work) = work {
            work();
        }
    }

    /// A resource's final release during preparation returns at once, and the
    /// resource stays alive for the make: the view's pin frees it when the
    /// caller drops the view.
    #[test]
    fn a_resource_released_while_its_view_is_prepared_outlives_the_make() {
        if host().is_none() {
            return; // no host GPU here
        }
        let resource = crate::resource::test_buffer(256);
        let mut record = [0u32; 12];
        let at = record.as_mut_ptr() as u64;
        // R8_UINT buffer SRV, all 256 elements.
        let desc = [62, 1, 0x1688, 0, 0, 256, 0, 0, 0, 0];
        crate::descriptor::create_srv(std::ptr::null(), resource, &desc, at);
        let Some(heap) = HostHeap::new(false, 1) else {
            crate::object::slot::<extern "system" fn(crate::object::This) -> u32>(resource, 0x10)(
                resource,
            );
            return;
        };

        let address = resource as usize;
        let release =
            crate::object::slot::<extern "system" fn(crate::object::This) -> u32>(resource, 0x10);
        *WHILE_PREPARING.lock().unwrap() = Some(Box::new(move || {
            // The title's last reference goes while the view is being made.
            // The call must not wait for the make: this very thread is the
            // one making the view, so a wait would never end.
            let released = std::thread::spawn(move || release(address as crate::object::This));
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
            while !released.is_finished() {
                assert!(
                    std::time::Instant::now() < deadline,
                    "the title's release waited for the view's make"
                );
                std::thread::yield_now();
            }
            assert_eq!(released.join().unwrap(), 1, "the pin still holds it");
        }));

        let view = heap.mirror(0, at, false).expect("the current view");
        assert!(WHILE_PREPARING.lock().unwrap().is_none(), "the prepare ran");
        assert!(heap.made(0), "the view was made from the live resource");
        assert!(
            crate::descriptor::view(at).is_some(),
            "still current while the caller holds the view"
        );
        // The last pin's drop frees the resource and makes its views stale.
        drop(view);
        assert!(
            crate::descriptor::view(at).is_none(),
            "stale once it is freed"
        );
    }

    /// A record the title writes into a slot while another thread prepares
    /// the slot's old record is the one the slot ends up viewing.
    #[test]
    fn a_slot_rewritten_while_its_view_is_prepared_views_the_new_record() {
        if host().is_none() {
            return; // no host GPU here
        }
        let null = std::ptr::null();
        let heap_desc: [u32; 8] = [0, 1024, 1, 0, 0, 0, 0, 0];
        let mut heap = null;
        let iid = crate::object::guid(
            0x8efb471d,
            0x616c,
            0x4f49,
            [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51],
        );
        assert_eq!(
            crate::descriptor::create_heap(null, heap_desc.as_ptr().cast(), &iid, &mut heap),
            0
        );
        let title_heap = crate::descriptor::heap(heap);
        let s = 960u32;
        let at = title_heap.start + u64::from(s) * 32;
        let memory = vec![0u8; 1 << 17];
        let words: [u32; 14] = [3, 0, 0, 0, 4, 0, 1, 1 | 1 << 16, 28, 1, 0, 0x100, 0, 0];
        let view_of = |address: u64| {
            let mut view = [0u32; 16];
            view[..6].copy_from_slice(&[28, 4, 0x1688, 0, 0, 1]);
            (view[10], view[11]) = (address as u32, (address >> 32) as u32);
            view
        };
        let (first, second) = (memory.as_ptr() as u64, memory.as_ptr() as u64 + (1 << 16));
        crate::descriptor::create_placed_srv(null, &words, &view_of(first), at);
        let host = title_heap.host().expect("a host view heap");
        let (done, waited) = std::sync::mpsc::channel();
        let second_view = view_of(second);
        *WHILE_PREPARING.lock().unwrap() = Some(Box::new(move || {
            // The title writes the other record into the slot meanwhile.
            std::thread::spawn(move || {
                crate::descriptor::create_placed_srv(std::ptr::null(), &words, &second_view, at);
                done.send(()).unwrap();
            });
            assert!(
                waited
                    .recv_timeout(std::time::Duration::from_secs(5))
                    .is_ok(),
                "the write waits"
            );
        }));
        // A draw reads the slot: its first record is prepared.
        assert!(host.mirror(s, at, true).is_some());
        let second_texture = crate::resource::texture_at(second, &words).expect("its resource");
        let raw = windows::core::Interface::as_raw(
            &crate::resource::made_copy(second_texture)
                .expect("made")
                .texture,
        ) as usize;
        assert_eq!(
            host.textures_at(&[s]),
            [raw],
            "the slot views the record written last"
        );
    }

    /// What a view needs is made without the slot lock: another thread
    /// binding a view of the same heap meanwhile is not held up.
    #[test]
    fn views_bound_while_another_slot_gets_its_texture_do_not_wait() {
        if host().is_none() {
            return; // no host GPU here
        }
        let null = std::ptr::null();
        let heap_desc: [u32; 8] = [0, 1024, 1, 0, 0, 0, 0, 0];
        let mut heap = null;
        let iid = crate::object::guid(
            0x8efb471d,
            0x616c,
            0x4f49,
            [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51],
        );
        assert_eq!(
            crate::descriptor::create_heap(null, heap_desc.as_ptr().cast(), &iid, &mut heap),
            0
        );
        let title_heap = crate::descriptor::heap(heap);
        let (first, second) = (950u32, 951u32);
        let memory = vec![0u8; 1 << 17];
        let words: [u32; 14] = [3, 0, 0, 0, 4, 0, 1, 1 | 1 << 16, 28, 1, 0, 0x100, 0, 0];
        let slot = |i: u32| title_heap.start + u64::from(i) * 32;
        for (i, at) in [(first, 0u64), (second, 1 << 16)] {
            let address = memory.as_ptr() as u64 + at;
            let mut view = [0u32; 16];
            view[..6].copy_from_slice(&[28, 4, 0x1688, 0, 0, 1]);
            (view[10], view[11]) = (address as u32, (address >> 32) as u32);
            crate::descriptor::create_placed_srv(null, &words, &view, slot(i));
        }
        let host = title_heap.host().expect("a host view heap");
        let heap_at = title_heap.start;
        let heap = heap as usize;
        let (done, waited) = std::sync::mpsc::channel();
        *WHILE_PREPARING.lock().unwrap() = Some(Box::new(move || {
            std::thread::spawn(move || {
                let h = crate::descriptor::heap(heap as crate::object::This);
                h.host()
                    .unwrap()
                    .mirror(second, heap_at + u64::from(second) * 32, true);
                done.send(()).unwrap();
            });
            assert!(
                waited
                    .recv_timeout(std::time::Duration::from_secs(5))
                    .is_ok(),
                "a view bound while another slot's texture is made waits"
            );
        }));
        assert!(host.mirror(first, slot(first), true).is_some());
        assert!(WHILE_PREPARING.lock().unwrap().is_none(), "the prepare ran");
        assert!(host.kept(first) && host.kept(second), "both made");
    }

    /// The textures of slots shaders read are made without the heaps' lock:
    /// a title thread writing a view meanwhile is not held up by the make.
    #[test]
    fn views_written_while_read_slots_get_their_textures_do_not_wait() {
        if host().is_none() {
            return; // no host GPU here
        }
        let null = std::ptr::null();
        let heap_desc: [u32; 8] = [0, 1024, 1, 0, 0, 0, 0, 0];
        let mut heap = null;
        let iid = crate::object::guid(
            0x8efb471d,
            0x616c,
            0x4f49,
            [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51],
        );
        assert_eq!(
            crate::descriptor::create_heap(null, heap_desc.as_ptr().cast(), &iid, &mut heap),
            0
        );
        let title_heap = crate::descriptor::heap(heap);
        let (read, written) = (900u32, 901u32);
        // A 4x1 RGBA8 texture, linear, in memory of its own, and a 2D view.
        let memory = vec![0u8; 1 << 16];
        let address = memory.as_ptr() as u64;
        let words: [u32; 14] = [3, 0, 0, 0, 4, 0, 1, 1 | 1 << 16, 28, 1, 0, 0x100, 0, 0];
        let mut view = [0u32; 16];
        view[..6].copy_from_slice(&[28, 4, 0x1688, 0, 0, 1]);
        (view[10], view[11]) = (address as u32, (address >> 32) as u32);
        let slot = |i: u32| title_heap.start + u64::from(i) * 32;
        crate::descriptor::create_placed_srv(null, &words, &view, slot(read));
        let host = title_heap.host().expect("a host view heap");
        assert!(!host.kept(read), "left null until read");
        let (done, waited) = std::sync::mpsc::channel();
        let at = slot(written);
        *crate::descriptor::tests::WHILE_MAKING.lock().unwrap() = Some((
            read,
            Box::new(move || {
                std::thread::spawn(move || {
                    crate::descriptor::create_placed_srv(std::ptr::null(), &words, &view, at);
                    done.send(()).unwrap();
                });
                assert!(
                    waited
                        .recv_timeout(std::time::Duration::from_secs(5))
                        .is_ok(),
                    "a view written while a read slot's texture is made waits"
                );
            }),
        ));
        crate::descriptor::make_read(&[read]);
        assert!(
            crate::descriptor::tests::WHILE_MAKING
                .lock()
                .unwrap()
                .is_none(),
            "the make ran"
        );
        assert!(host.kept(read), "made for the read");
    }

    /// A heap sync makes the slots whose records the title changed as bytes
    /// (a copy of another slot's record, a record cleared) and passes the
    /// others by.
    #[test]
    fn heap_syncs_make_the_slots_whose_records_changed() {
        if host().is_none() {
            return; // no host GPU here
        }
        let null = std::ptr::null();
        let heap_desc: [u32; 8] = [0, 4, 1, 0, 0, 0, 0, 0];
        let mut heap = null;
        let iid = crate::object::guid(
            0x8efb471d,
            0x616c,
            0x4f49,
            [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51],
        );
        assert_eq!(
            crate::descriptor::create_heap(null, heap_desc.as_ptr().cast(), &iid, &mut heap),
            0
        );
        let title_heap = crate::descriptor::heap(heap);
        let start = title_heap.start;
        // A view of no memory: a view of the test's own heap memory would get
        // a host copy whose 4 MiB window the guards may cover (the process
        // heap guarded read-only hung the suite).
        crate::descriptor::create_cbv(null, &[0, 0, 256, 0], start);
        let host = title_heap.host().expect("a host view heap");
        // SAFETY: slots 0 and 1 of the heap's memory, 32 bytes each.
        unsafe { std::ptr::copy_nonoverlapping(start as *const u8, (start + 32) as *mut u8, 32) };
        assert!(!host.made(1), "copied as bytes: not seen yet");
        let mut bytes = Vec::new();
        host.sync_slots(0, 4, start, 32, &mut bytes);
        assert!(host.made(0) && host.made(1) && !host.made(2));
        // SAFETY: as above.
        unsafe { std::ptr::write_bytes((start + 32) as *mut u8, 0, 32) };
        host.sync_slots(0, u32::MAX, start, 32, &mut bytes);
        assert!(
            host.made(0) && !host.made(1),
            "a cleared record leaves its slot"
        );
        host.sync_slots(0, 4, 1, 32, &mut bytes);
        assert!(host.made(0), "an inaccessible scan leaves the slot alone");
    }

    #[test]
    fn shared_templates_copy_the_slot_and_follow_buffer_replacement() {
        let Some(heap) = HostHeap::new(false, 3) else {
            return;
        };
        if !templates::enabled() {
            return;
        }
        let allocation = crate::watch::tests::pool(1);
        let base = allocation as u64;
        let address = base + 256;
        crate::shadow::register(1, base, crate::watch::PAGE as u64);
        let original = crate::shadow::find(address, 256).unwrap();
        let mut slots = [[0u64; 4]; 3];
        let start = slots.as_mut_ptr() as u64;
        crate::descriptor::create_cbv(
            std::ptr::null(),
            &[address as u32, (address >> 32) as u32, 256, 0],
            start,
        );
        slots[1] = slots[0];
        // Only the header supplies the view. Padding on a byte copy is opaque.
        slots[1][3] = u64::MAX;
        let record = record_at(start);
        let key = record_key(record);
        let mut bytes = Vec::new();
        heap.sync_slots(0, 2, start, 32, &mut bytes);
        assert!(Arc::ptr_eq(&heap.table_backing(0).unwrap(), &original));
        assert!(Arc::ptr_eq(&heap.table_backing(1).unwrap(), &original));
        {
            let made = heap.made.lock().unwrap();
            let template = made.templates[heap.template_shard(key)]
                .get()
                .unwrap()
                .as_ref()
                .unwrap();
            assert!(template.get(key, crate::shadow::lookup_epoch()).is_some());
        }

        // The title address and record stay the same when an inner buffer
        // supersedes the original range.
        crate::shadow::register(2, address, 4096);
        let replacement = crate::shadow::find(address, 256).unwrap();
        assert!(!Arc::ptr_eq(&original, &replacement));
        {
            let made = heap.made.lock().unwrap();
            let template = made.templates[heap.template_shard(key)]
                .get()
                .unwrap()
                .as_ref()
                .unwrap();
            assert!(template.get(key, crate::shadow::lookup_epoch()).is_none());
        }
        slots[2] = slots[0];
        heap.sync_slots(2, 3, start, 32, &mut bytes);
        assert!(Arc::ptr_eq(&heap.table_backing(2).unwrap(), &replacement));
        slots[1] = [0; 4];
        heap.sync_slots(1, 2, start, 32, &mut bytes);
        assert!(!heap.made(1) && heap.table_backing(1).is_none());
        std::hint::black_box(&slots);
        crate::shadow::unregister(2, address);
        crate::shadow::unregister(1, base);
        drop((heap, original, replacement));
        crate::watch::tests::free(allocation);
    }

    #[test]
    fn staged_views_recheck_the_live_header_and_buffer_mapping() {
        let Some(heap) = HostHeap::new(false, 3) else {
            return;
        };
        if !templates::enabled() {
            return;
        }
        let workers = view_staging().expect("CPU-only staging heaps");
        let allocation = crate::watch::tests::pool(1);
        let base = allocation as u64;
        let address = base + 256;
        crate::shadow::register(allocation, base, crate::watch::PAGE as u64);
        let original = crate::shadow::find(address, 256).unwrap();
        let mut slots = [[0u64; 4]; 3];
        let start = slots.as_mut_ptr() as u64;
        super::super::record(start, View::Constant(address, 256));
        slots[1] = slots[0];
        slots[2] = slots[0];
        let record = record_at(start);
        let records: [_; 3] = std::array::from_fn(|index| ChangedRecord {
            index: index as u32,
            record,
            previous: 0,
            clear: false,
        });
        let pins = pin_changed_views(&records);
        let staged = stage_views(&workers, &records, &pins);
        assert_eq!(staged.views.len(), 1);
        let mut batch = PublishBatch::default();
        slots[0] = [0; 4];
        std::hint::black_box(&slots);
        let mut misses = Vec::new();
        heap.publish_templates(
            &records[..2],
            start,
            32,
            (&pins, Some(&staged)),
            &mut batch,
            &mut misses,
        );
        assert!(misses.is_empty());
        assert!(
            !heap.made(0),
            "a raw rewrite invalidates the stage's header"
        );
        assert!(Arc::ptr_eq(&heap.table_backing(1).unwrap(), &original));

        crate::shadow::register(allocation + 1, address, 4096);
        let replacement = crate::shadow::find(address, 256).unwrap();
        assert!(!Arc::ptr_eq(&original, &replacement));
        assert!(!staged.recheck_pending(&records[2..]));
        heap.publish_templates(
            &records[2..],
            start,
            32,
            (&pins, Some(&staged)),
            &mut batch,
            &mut misses,
        );
        assert_eq!(misses.len(), 1);
        heap.publish_records(&records[2..], start, 32);
        assert!(Arc::ptr_eq(&heap.table_backing(2).unwrap(), &replacement));

        crate::shadow::unregister(allocation + 1, address);
        crate::shadow::unregister(allocation, base);
        drop((staged, heap, original, replacement));
        crate::watch::tests::free(allocation);
    }

    #[test]
    fn materializing_another_shadow_keeps_a_valid_staged_cbv() {
        let Some(workers) = view_staging() else {
            return;
        };
        let page = crate::watch::PAGE;
        let allocation = crate::watch::tests::pool(3);
        let addresses = [allocation, allocation + page];
        let mut slots = [[0u64; 4]; 2];
        for (slot, address) in slots.iter_mut().zip(addresses) {
            crate::shadow::register(address, address as u64, page as u64);
            super::super::record(slot.as_mut_ptr() as u64, View::Constant(address as u64, 16));
        }
        let records: [_; 2] = std::array::from_fn(|index| ChangedRecord {
            index: index as u32,
            record: record_at(slots[index].as_ptr() as u64),
            previous: 0,
            clear: false,
        });
        let before = crate::shadow::lookup_epoch();
        let pins = pin_changed_views(&records);
        let staged = stage_views(&workers, &records, &pins);
        let after = crate::shadow::lookup_epoch();
        assert_ne!(before, after, "staging materializes the two shadows");
        for record in &records {
            let staged = &staged.views[&record_key(record.record)];
            assert!(staged.hit(after).is_some());
        }
        let unrelated = allocation + 2 * page;
        crate::shadow::register(unrelated, unrelated as u64, page as u64);
        let newer = crate::shadow::lookup_epoch();
        assert!(staged.views.values().all(|view| view.hit(newer).is_none()));
        assert!(staged.recheck_pending(&records));
        assert!(staged.views.values().all(|view| view.hit(newer).is_some()));
        crate::shadow::unregister(unrelated, unrelated as u64);
        drop(staged);
        for address in addresses {
            crate::shadow::unregister(address, address as u64);
        }
        crate::watch::tests::free(allocation);
    }

    #[test]
    fn serial_scans_stage_a_large_miss_batch_during_a_stable_epoch() {
        if !templates::enabled() {
            return;
        }
        let count = MIN_STAGED_RECORDS as u32;
        let Some(heap) = HostHeap::new(false, count) else {
            return;
        };
        let allocation = crate::watch::tests::pool(1);
        crate::shadow::register(allocation, allocation as u64, crate::watch::PAGE as u64);
        let mut slots = vec![[0u64; 4]; count as usize];
        let start = slots.as_mut_ptr() as u64;
        let mut bytes = Vec::new();
        let before = STAGED_BATCHES.with(std::cell::Cell::get);
        for offset in [256, 512] {
            let address = allocation as u64 + offset;
            super::super::record(start, View::Constant(address, 256));
            let record = slots[0];
            slots.fill(record);
            std::hint::black_box(&slots);
            heap.sync_slots(0, count, start, 32, &mut bytes);
            assert!((0..count).all(|index| heap.kept(index)));
        }
        assert_eq!(STAGED_BATCHES.with(std::cell::Cell::get) - before, 2);
        crate::shadow::unregister(allocation, allocation as u64);
        drop(heap);
        crate::host::release_retired();
        crate::watch::tests::free(allocation);
    }

    #[test]
    fn batched_placed_texture_slots_remain_lazy() {
        if !templates::enabled() {
            return;
        }
        let count = MIN_STAGED_RECORDS as u32;
        let Some(heap) = HostHeap::new(false, count) else {
            return;
        };
        let memory = crate::watch::tests::pool(1);
        let mut slots = vec![[0u64; 4]; count as usize];
        let start = slots.as_mut_ptr() as u64;
        let mut words = [0; 16];
        words[..10].copy_from_slice(&[42, 4, 0x1688, 0, 0, 1, 0, 0, 0, 0]);
        words[10] = memory as u32;
        words[11] = ((memory as u64) >> 32) as u32;
        let view = View::Placed(
            false,
            placed_tex_desc(4, 4, DXGI_FORMAT_R32_UINT.0 as u32, 4),
            words,
        );
        super::super::record(start, view);
        let record = slots[0];
        slots.fill(record);
        std::hint::black_box(&slots);

        let before = STAGED_BATCHES.with(std::cell::Cell::get);
        heap.sync_slots(0, count, start, 32, &mut Vec::new());
        assert_eq!(STAGED_BATCHES.with(std::cell::Cell::get) - before, 0);
        for index in 0..count {
            assert!(heap.unmade_placed(index).is_some());
            assert_eq!(heap.ready[index as usize].load(Ordering::Acquire), 0);
        }
        drop(heap);
        crate::watch::tests::free(memory);
    }

    fn publication_texture(address: u64) -> crate::object::This {
        let desc = placed_tex_desc(4, 4, DXGI_FORMAT_R32_UINT.0 as u32, 4);
        let iid = crate::object::guid(
            0x696442be,
            0xa72e,
            0x4059,
            [0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad],
        );
        let null = std::ptr::null();
        let mut texture = null;
        assert_eq!(
            crate::resource::create_at_address(
                null,
                address,
                desc.as_ptr().cast(),
                0,
                std::ptr::null(),
                &iid,
                &mut texture,
            ),
            0
        );
        texture
    }

    #[test]
    fn scattered_texture_templates_match_and_expire_with_the_resource() {
        let Some(host) = host() else { return };
        if !templates::enabled() {
            return;
        }
        let heap = HostHeap::new(false, 6).unwrap();
        let memory = crate::watch::tests::pool(2);
        let first = publication_texture(memory as u64);
        let mut slots = [[0u64; 4]; 6];
        let start = slots.as_mut_ptr() as u64;
        let srv = |resource| View::Shader(resource, Some([42, 4, 0x1688, 0, 0, 1, 0, 0, 0, 0]));
        let uav = |resource| View::Unordered(resource, 0, Some([42, 4, 0, 0, 0, 0, 0, 0, 0, 0]));
        super::super::record(start, srv(first as usize));
        super::super::record(start + 32, uav(first as usize));
        let mut bytes = Vec::new();
        heap.sync_slots(0, 2, start, 32, &mut bytes);
        let srv_record = record_at(start);
        let srv_key = record_key(srv_record);
        let uav_key = record_key(record_at(start + 32));
        let old = crate::resource::made_copy(first).unwrap();
        {
            let made = heap.made.lock().unwrap();
            let entry = |key| {
                made.textures[template_shard(key)]
                    .get()
                    .unwrap()
                    .as_ref()
                    .unwrap()
                    .entry(key)
                    .unwrap()
            };
            let cached_srv = entry(srv_key);
            assert_eq!(entry(uav_key).key, uav_key);
            let pin = view_at(srv_record[1], srv_record[2]).unwrap();
            old.evicted.store(true, Ordering::Release);
            assert!(cached_srv.hit(&pin).is_none());
            old.evicted.store(false, Ordering::Release);
        }

        slots[3] = slots[0];
        slots[5] = slots[1];
        std::hint::black_box(&slots);
        heap.sync_slots(2, 4, start, 32, &mut bytes);
        assert!(heap.keeps_texture(3, &old.texture));
        let workers = view_staging().expect("CPU-only staging heaps");
        slots[4] = slots[0];
        std::hint::black_box(&slots);
        let records = [4, 5].map(|index| ChangedRecord {
            index,
            previous: 0,
            record: record_at(start + u64::from(index) * 32),
            clear: false,
        });
        let pins = pin_changed_views(&records);
        let staged = stage_views(&workers, &records, &pins);
        assert!(staged.views.contains_key(&srv_key));
        assert!(staged.views.contains_key(&uav_key));
        old.evicted.store(true, Ordering::Release);
        assert!(
            staged.views[&srv_key]
                .hit(crate::shadow::lookup_epoch())
                .is_none()
        );
        old.evicted.store(false, Ordering::Release);
        let mut misses = Vec::new();
        heap.publish_templates(
            &records,
            start,
            32,
            (&pins, Some(&staged)),
            &mut PublishBatch::default(),
            &mut misses,
        );
        assert!(misses.is_empty());
        assert!(heap.keeps_texture(4, &old.texture));
        assert!(heap.keeps_texture(5, &old.texture));
        drop(staged);
        drop(pins);

        let (writer, writer_pso) = crate::compute::pipeline(
            "RWTexture2D<uint> tex : register(u0);
             [numthreads(1, 1, 1)] void main() { tex[uint2(0, 0)] = 712; }",
            windows::core::s!("cached texture UAV"),
            0,
            &[crate::compute::Param::Table(&[crate::compute::uav(0, 1)])],
        )
        .unwrap();
        let (reader, reader_pso) = crate::compute::pipeline(
            "Texture2D<uint> a : register(t0);
             Texture2D<uint> b : register(t1);
             RWByteAddressBuffer result : register(u0);
             [numthreads(1, 1, 1)] void main() {
                 result.Store2(0, uint2(a.Load(int3(0, 0, 0)), b.Load(int3(0, 0, 0))));
             }",
            windows::core::s!("cached texture SRV"),
            0,
            &[
                crate::compute::Param::Table(&[crate::compute::srv(0, 1)]),
                crate::compute::Param::Table(&[crate::compute::srv(1, 1)]),
                crate::compute::Param::Uav { register: 0 },
            ],
        )
        .unwrap();
        let (result, data) = host.shared_buffer(256).unwrap();
        let list = host.new_list().unwrap();
        // SAFETY: all resources and signatures remain alive through the fence.
        unsafe {
            data.cast::<[u32; 2]>().write([0, 0]);
            list.list.SetDescriptorHeaps(&[Some(heap.heap.clone())]);
            list.list.SetComputeRootSignature(&writer);
            list.list.SetComputeRootDescriptorTable(0, heap.gpu(5));
            list.list.SetPipelineState(&writer_pso);
            list.list.Dispatch(1, 1, 1);
            list.list.ResourceBarrier(&[crate::host::transition(
                &old.texture,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            )]);
            list.list.SetComputeRootSignature(&reader);
            list.list.SetComputeRootDescriptorTable(0, heap.gpu(0));
            list.list.SetComputeRootDescriptorTable(1, heap.gpu(4));
            list.list
                .SetComputeRootUnorderedAccessView(2, result.GetGPUVirtualAddress());
            list.list.SetPipelineState(&reader_pso);
            list.list.Dispatch(1, 1, 1);
        }
        list.close();
        host.execute(&[&list]);
        // SAFETY: the mapped two dwords after the completed submission.
        assert_eq!(unsafe { data.cast::<[u32; 2]>().read() }, [712, 712]);

        assert_eq!(crate::object::release_ref(first), 0);
        assert!(view_at(srv_record[1], srv_record[2]).is_none());
        let next = publication_texture((memory + crate::watch::PAGE) as u64);
        super::super::record(start + 64, srv(next as usize));
        assert_ne!(record_key(record_at(start + 64)), srv_key);
        heap.sync_slots(0, 6, start, 32, &mut bytes);
        let replacement = crate::resource::made_copy(next).unwrap();
        assert!(heap.keeps_texture(2, &replacement.texture));
        assert!([0, 1, 3, 4, 5].into_iter().all(|slot| !heap.kept(slot)));
        assert_eq!(crate::object::release_ref(next), 0);
        drop((heap, list, old, replacement));
        crate::host::release_retired();
        crate::watch::tests::free(memory);
    }

    #[test]
    #[ignore = "manual publication timing on a D3D12 host"]
    fn synthetic_shared_heap_publication() {
        measure_shared_heap_publication(false, false);
        measure_shared_heap_publication(true, false);
        measure_shared_heap_publication(true, true);
    }

    fn measure_shared_heap_publication(aliased: bool, world_mix: bool) {
        const SLOTS: usize = 44_043;
        // The world mix uses the recorded order of magnitude for distinct
        // records (about 3,200), 38% clears, many buffer owners and 6% textures.
        let (keys, owners, pages) = if world_mix {
            (4096, 64, 1)
        } else {
            (1024, 1, 8)
        };
        let heap = HostHeap::new(false, SLOTS as u32).expect("D3D12 heap");
        let len = (pages * crate::watch::PAGE) as u64;
        let mut allocations = Vec::new();
        for _ in 0..owners {
            let allocation = crate::watch::tests::pool(pages);
            let base = allocation as u64;
            if aliased {
                let (buffer, mapped) = host().unwrap().shared_buffer(len).unwrap();
                // SAFETY: the live D3D12 buffer supplies the alias's GPU address.
                let gpu = unsafe { buffer.GetGPUVirtualAddress() };
                let shadow = crate::shadow::Shadow::new_aliased(buffer, mapped, gpu, base, len);
                crate::shadow::register_aliased(shadow);
            } else {
                crate::shadow::register(allocation, base, len);
            }
            let backing = crate::shadow::find(base, len).unwrap();
            assert_eq!(backing.is_aliased(), aliased);
            allocations.push((allocation, backing));
        }
        let texture_memory = world_mix.then(|| crate::watch::tests::pool(keys / 16));
        let textures: Vec<_> = texture_memory
            .into_iter()
            .flat_map(|base| (0..keys / 16).map(move |i| base + i * crate::watch::PAGE))
            .map(|address| publication_texture(address as u64))
            .collect();
        let dictionary = |pass: usize| {
            let mut records = vec![[0u64; 4]; keys];
            for (i, record) in records.iter_mut().enumerate() {
                let at = record.as_mut_ptr() as u64;
                let owner = if world_mix {
                    (37 * (i / 4) + i) % owners
                } else {
                    0
                };
                let base = allocations[owner].0 as u64;
                let addr = base + (64 * (i / owners + pass * keys / owners)) as u64;
                let kind = if world_mix {
                    match i % 32 {
                        0..=8 => 0,
                        9..=25 => 1 + i % 2,
                        26..=29 => 3,
                        30 => 4,
                        _ => 5,
                    }
                } else {
                    i % 4
                };
                let mut view = [0u32; 16];
                view[10] = addr as u32;
                view[11] = (addr >> 32) as u32;
                let buffer = [1, 0, 0, 0, 512, 0, 1, 1 | (1 << 16), 0, 1, 0, 0, 4, 0];
                match kind {
                    0 => super::super::record(at, View::Constant(addr, 256)),
                    1 => {
                        view[..8].copy_from_slice(&[0, 1, 0x1688, 0, 0, 0, 16, 12]);
                        super::super::record(at, View::Placed(false, buffer, view));
                    }
                    2 => {
                        view[..8].copy_from_slice(&[0, 1, 0x1688, 0, 0, 0, 16, 16]);
                        super::super::record(at, View::Placed(false, buffer, view));
                    }
                    3 => {
                        view[..6].copy_from_slice(&[0, 1, 0, 0, 16, 16]);
                        super::super::record(at, View::Placed(true, buffer, view));
                    }
                    4 => {
                        let texture = textures[pass * keys / 32 + i / 32];
                        super::super::record(
                            at,
                            View::Shader(
                                texture as usize,
                                Some([42, 4, 0x1688, 0, 0, 1, 0, 0, 0, 0]),
                            ),
                        );
                    }
                    _ => {
                        let texture = textures[pass * keys / 32 + i / 32];
                        super::super::record(
                            at,
                            View::Unordered(
                                texture as usize,
                                0,
                                Some([42, 4, 0, 0, 0, 0, 0, 0, 0, 0]),
                            ),
                        );
                    }
                }
            }
            records
        };
        let dictionaries = [dictionary(0), dictionary(1)];
        let mut slots = vec![[0u64; 4]; SLOTS];
        let start = slots.as_mut_ptr() as u64;
        let valid = |i, round| {
            if world_mix {
                (i + 3 * round) % 8 >= 3
            } else {
                (i + round) % 7 != 0
            }
        };
        let mut measurements = Vec::new();
        let mut cold = [0.0; 2];
        for round in 0..11 {
            for (i, slot) in slots.iter_mut().enumerate() {
                *slot = if valid(i, round) {
                    let key = if world_mix {
                        ((37 * i) ^ (i >> 3)) % keys
                    } else {
                        (37 * i) % keys
                    };
                    dictionaries[round % 2][key]
                } else {
                    [0; 4]
                };
            }
            std::hint::black_box(&slots);
            let now = std::time::Instant::now();
            heap.sync_ranges(&[(0, SLOTS as u32)], start, 32, None);
            let elapsed = now.elapsed().as_secs_f64() * 1000.0;
            if round < 2 {
                cold[round] = elapsed;
            } else {
                measurements.push(elapsed);
            }
            for i in (0..SLOTS).step_by(97) {
                let kept = valid(i, round);
                assert_eq!(heap.kept(i as u32), kept, "round {round} slot {i}");
            }
            crate::host::release_retired();
        }
        measurements.sort_by(f64::total_cmp);
        eprintln!(
            "44,043-slot scan and publication, aliased={aliased}, world_mix={world_mix}, cold ms: {cold:?}, warm ms: {measurements:?}"
        );
        drop(heap);
        crate::host::release_retired();
        for texture in textures {
            assert_eq!(crate::object::release_ref(texture), 0);
        }
        if let Some(memory) = texture_memory {
            crate::watch::tests::free(memory);
        }
        for (allocation, backing) in allocations {
            drop(backing);
            let base = allocation as u64;
            if aliased {
                assert!(crate::shadow::unregister_aliased(base));
            } else {
                crate::shadow::unregister(allocation, base);
            }
            crate::watch::tests::free(allocation);
        }
    }

    #[test]
    fn heap_sync_stops_before_dynamic_slots() {
        let Some(heap) = HostHeap::new(false, 1) else {
            return;
        };
        let buffer = host().unwrap().shared_buffer(65536).unwrap().0;
        let slot = heap.dynamic(1).unwrap();
        assert_eq!(slot, heap.count());
        heap.raw_view_at(slot, Some(&buffer), 0, 256, false);

        // Simulate a title heap one slot larger than the host's mirrored
        // range. Both title records are valid, but the second must leave the
        // list's descriptor alone.
        let mut records = [[0u32; 8]; 2];
        let start = records.as_mut_ptr() as u64;
        for index in 0..2 {
            crate::descriptor::create_cbv(std::ptr::null(), &[0, 0, 256, 0], start + 32 * index);
        }
        let mut bytes = Vec::new();
        heap.sync_slots(0, 2, start, 32, &mut bytes);
        assert_eq!(bytes.len(), 12, "only the mirrored header was read");
        assert!(heap.made(0) && heap.kept(slot));
        heap.release(heap.id(), slot, 1);
    }

    #[test]
    fn an_unreadable_neighbor_does_not_hide_readable_slot_headers() {
        use windows_sys::Win32::System::Memory::{PAGE_NOACCESS, VirtualProtect};

        let Some(heap) = HostHeap::new(true, 4) else {
            return;
        };
        let memory = crate::watch::tests::pool(2);
        let page = crate::watch::PAGE;
        let start = (memory + page - 40) as u64;
        let mut sampler = [0u32; 4];
        let null = std::ptr::null();
        let words = [21, 3, 3, 3, 0, 1, 0, 0, 0, 0, 0, 0x7f7f_ffff, 0];
        crate::descriptor::create_sampler(null, &words, sampler.as_mut_ptr() as u64);
        // The second header begins eight bytes from the end of the page.
        // SAFETY: both headers lie within the two committed pages.
        unsafe {
            (start as *mut [u32; 4]).write_unaligned(sampler);
            ((start + 32) as *mut [u32; 4]).write_unaligned(sampler);
        }
        let mut old = 0;
        // SAFETY: the second page is ours and covers the end of slot 1.
        let protected =
            unsafe { VirtualProtect((memory + page) as *mut _, page, PAGE_NOACCESS, &mut old) };
        assert_ne!(protected, 0);
        heap.sync_slots(0, 3, start, 32, &mut Vec::new());
        assert!(heap.made(0));
        assert!(!heap.made(1) && !heap.made(2));
        crate::watch::tests::free(memory);
    }

    #[test]
    fn snapshot_cannot_replace_a_restored_live_record() {
        let Some(heap) = HostHeap::new(true, 2) else {
            return;
        };
        let null = std::ptr::null();
        let mut slot = [0u32; 4];
        let handle = slot.as_mut_ptr() as u64;
        let mut sampler = [21, 3, 3, 3, 0, 1, 0, 0, 0, 0, 0, 0x7f7f_ffff, 0];
        crate::descriptor::create_sampler(null, &sampler, handle);
        let original = record_at(handle);
        heap.mirror(0, handle, false).unwrap();
        let previous = heap.made.lock().unwrap().keys[0];

        sampler[0] = 0;
        crate::descriptor::create_sampler(null, &sampler, handle);
        let captured = record_at(handle);
        assert_ne!(original, captured);
        sampler[0] = 21;
        crate::descriptor::create_sampler(null, &sampler, handle);
        heap.mirror(0, handle, false).unwrap();
        assert_eq!(heap.made.lock().unwrap().keys[0], previous);

        // Capture of the changed sampler had already happened. The live
        // write restored the original descriptor, so its host key never changed.
        let _ = heap.mirror_record(0, false, Some((previous, handle)), || captured);
        assert_eq!(heap.made.lock().unwrap().keys[0], previous);
    }

    #[test]
    #[ignore = "synthetic descriptor scan benchmark"]
    fn bench_sync_heap_snapshots() {
        use std::time::Instant;
        let page = crate::watch::PAGE;
        let pages = 342;
        let slots = (pages * page / 32) as u32;
        let memory = crate::watch::tests::pool(pages);
        let heap = HostHeap::new(false, slots).expect("a host view heap");
        assert_eq!(heap.count(), slots);
        let mut record = [0u32; 8];
        crate::descriptor::create_cbv(
            std::ptr::null(),
            &[0, 0, 256, 0],
            record.as_mut_ptr() as u64,
        );
        for i in (0..slots as usize).step_by(8) {
            // SAFETY: each aligned 32-byte record lies inside our allocation.
            unsafe { ((memory + i * 32) as *mut [u32; 8]).write(record) };
        }
        let mut bytes = Vec::new();
        for round in 0..12 {
            for pages_at_once in [1, 4] {
                let started = Instant::now();
                let records = (pages_at_once * page / 32) as u32;
                for first in (0..slots).step_by(records as usize) {
                    heap.sync_slots(
                        first,
                        (first + records).min(slots),
                        memory as u64,
                        32,
                        &mut bytes,
                    );
                }
                eprintln!(
                    "heap sync round={round} pages/chunk={pages_at_once} elapsed={:?}",
                    started.elapsed()
                );
            }
        }
        assert!(heap.made(slots - 8));
        crate::watch::tests::free(memory);
    }

    #[test]
    #[ignore = "synthetic heap read benchmark"]
    fn bench_read_heap_snapshots() {
        use std::time::Instant;
        let page = crate::watch::PAGE;
        let pages = 342;
        let start = crate::watch::tests::pool(pages);
        // SAFETY: these pages are ours. Commit their backing before timing.
        unsafe { std::ptr::write_bytes(start as *mut u8, 1, pages * page) };
        fn sweep(start: usize, page: usize, chunk: usize, begin: usize, end: usize) {
            let mut bytes = vec![0; page * chunk];
            for i in (begin..end).step_by(chunk) {
                let len = end.abs_diff(i).min(chunk) * page;
                assert!(crate::host::read_into(
                    (start + i * page) as *const u8,
                    &mut bytes[..len]
                ));
                std::hint::black_box(&bytes[..len]);
            }
        }
        for _ in 0..11 {
            for (chunk, parallel) in [(1, false), (4, false), (1, true), (4, true)] {
                let now = Instant::now();
                if parallel {
                    std::thread::scope(|scope| {
                        for i in 0..3 {
                            scope.spawn(move || {
                                sweep(start, page, chunk, i * pages / 3, (i + 1) * pages / 3)
                            });
                        }
                    });
                } else {
                    sweep(start, page, chunk, 0, pages);
                }
                eprintln!(
                    "heap read pages/chunk={chunk} parallel={parallel} elapsed={:?}",
                    now.elapsed()
                );
            }
        }
        crate::watch::tests::free(start);
    }

    /// The world's 16-byte view of 0 elements removed the device.
    #[test]
    fn empty_and_overlong_buffer_views_are_refused() {
        assert!(within(64, 0, 4, 16));
        assert!(within(64, 3, 1, 16));
        assert!(!within(64, 0, 0, 16));
        assert!(!within(64, 3, 2, 16));
        assert!(!within(64, u64::MAX, 1, 16));
    }

    /// A null texture UAV keeps the title's dimension (3D for the world's
    /// wind field), and is 2D when the title names none.
    #[test]
    fn null_uavs_keep_the_titles_dimension() {
        let d = null_uav_desc(Some(8));
        assert_eq!(d.ViewDimension, D3D12_UAV_DIMENSION_TEXTURE3D);
        // SAFETY: the member the dimension names.
        assert_eq!(unsafe { d.Anonymous.Texture3D.WSize }, u32::MAX);
        let d = null_uav_desc(Some(5));
        assert_eq!(d.ViewDimension, D3D12_UAV_DIMENSION_TEXTURE2DARRAY);
        // SAFETY: the member the dimension names.
        assert_eq!(unsafe { d.Anonymous.Texture2DArray.ArraySize }, 1);
        let dimension = |d| null_uav_desc(d).ViewDimension;
        assert_eq!(dimension(Some(2)), D3D12_UAV_DIMENSION_TEXTURE1D);
        assert_eq!(dimension(Some(4)), D3D12_UAV_DIMENSION_TEXTURE2D);
        assert_eq!(dimension(None), D3D12_UAV_DIMENSION_TEXTURE2D);
        // Multisampled UAVs need a cap; they stay 2D.
        assert_eq!(dimension(Some(6)), D3D12_UAV_DIMENSION_TEXTURE2D);
    }

    /// A null 3D UAV made on the real device leaves it alive.
    #[test]
    fn null_3d_uav_is_accepted_by_the_device() {
        let Some(host) = host() else {
            return; // no host GPU
        };
        let Some(heap) = HostHeap::new(false, 4) else {
            return;
        };
        let view = View::Unordered(0, 0, Some([28, 8, 0, 0, 0, 0, 0, 0, 0, 0]));
        null_for(&view, heap.cpu(0));
        // SAFETY: a plain device query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
    }

    /// A table made afresh lies in dynamic slots, and a shader reading a
    /// view's address (e.g. the BVH builder's base) indexes the view
    /// addresses there: they have an entry for each dynamic slot, entered
    /// whenever the slot is made (a view, null, or a raw view of ours).
    /// Giving a slot back leaves its entry: every table made afresh makes
    /// all its slots. (Hosts whose view heaps are smaller than the title's,
    /// NVIDIA, make such tables; the shader read past the buffer.)
    ///
    /// Any heap size the host takes will do: a title heap of 4 slots leaves
    /// dynamic slots past it either way ([`DYNAMIC`] or [`MIN_DYNAMIC`]).
    #[test]
    fn view_addresses_cover_tables_made_afresh() {
        let Some(heap) = HostHeap::new(false, 4) else {
            return; // no host GPU
        };
        let end = heap.count() + heap.dynamic;
        assert!(heap.view_addresses().is_some());
        let first = heap.dynamic(2).unwrap();
        assert!(first >= heap.count() && first + 1 < end);
        // A buffer the title placed at 0xd_6013_e000: 16 elements of 4 bytes.
        let address = 0xd_6013_e000u64;
        let mut resource = [0u32; 14];
        (resource[0], resource[4]) = (1, 64);
        let mut desc = [0u32; 16];
        (desc[1], desc[2], desc[6], desc[7]) = (1, D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING, 16, 4);
        (desc[10], desc[11]) = (address as u32, (address >> 32) as u32);
        let view = View::Placed(false, resource, desc);
        assert_eq!(buffer_range(&view), Some((address, 64)));
        assert_eq!(heap.address_at(first), Some(0), "not made yet");
        heap.make_at(first, &view);
        assert_eq!(heap.address_at(first), Some(address));
        assert_eq!(heap.address_at(first + 1), Some(0), "not made yet");
        // The last dynamic slot has its entry too.
        heap.make_at(end - 1, &view);
        assert_eq!(heap.address_at(end - 1), Some(address));
        // A slot given back keeps its entry until it is made again; made
        // null, it has its entry cleared.
        heap.release(heap.id(), first, 2);
        assert_eq!(
            heap.address_at(first),
            Some(address),
            "given back, not made"
        );
        heap.make_null(first);
        assert_eq!(heap.address_at(first), Some(0));
        assert_eq!(heap.address_at(end - 1), Some(address), "still made");
        // Raw views of ours (root views, host copies' view pairs) in dynamic
        // slots clear what an earlier table entered there.
        heap.make_at(first, &view);
        heap.raw_view_at(first, None, 0, 0, false);
        assert_eq!(heap.address_at(first), Some(0), "a root view");
        heap.make_at(first, &view);
        heap.make_at(first + 1, &view);
        heap.raw_pair_at(first, None);
        assert_eq!(heap.address_at(first), Some(0), "a view pair");
        assert_eq!(heap.address_at(first + 1), Some(0), "a view pair");
        // Slots past the dynamic ones (host copies' view pairs) have no
        // entry, and nothing is written past the last: a mark there (the
        // buffer's allocation reaches past it) stays.
        assert_eq!(heap.address_at(end), None);
        let host = host().unwrap();
        let (buffer, data, _) = heap.addresses.get().unwrap().as_ref().unwrap();
        // SAFETY: plain queries of a live buffer.
        let (desc, size) = unsafe {
            let desc = buffer.GetDesc();
            let info = host.device.GetResourceAllocationInfo(0, &[desc]);
            (desc, info.SizeInBytes)
        };
        assert_eq!(desc.Width, 16 * u64::from(end));
        assert!(size >= desc.Width + 8, "room for the mark");
        let past = (data + 16 * end as usize) as *mut u64;
        let mark = 0x5a5a_5a5a_5a5a_5a5a;
        // SAFETY: mapped memory of the buffer's allocation (checked).
        unsafe { past.write_unaligned(mark) };
        heap.note_address(end, Some(&view));
        // SAFETY: as above.
        assert_eq!(unsafe { past.read_unaligned() }, mark);
        // SAFETY: a plain device query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
    }

    /// Dynamic slots stay taken until given back, and runs do not wrap.
    #[test]
    fn dynamic_slots_are_held_until_released() {
        let Some(heap) = HostHeap::new(false, 4) else {
            return; // no host GPU
        };
        // Every index a translated shader can reach exists (clamped below
        // `HEAP_VIEWS`, `HEAP_SAMPLERS`).
        // SAFETY: plain queries of live heaps.
        unsafe {
            assert!(heap.heap.GetDesc().NumDescriptors >= victory_shader::emit::HEAP_VIEWS);
            let samplers = HostHeap::new(true, 128).unwrap();
            assert!(samplers.heap.GetDesc().NumDescriptors >= victory_shader::emit::HEAP_SAMPLERS);
        }
        let a = heap.dynamic(heap.dynamic - 100).unwrap();
        let b = heap.dynamic(60).unwrap();
        assert_eq!((a, b), (4, 4 + heap.dynamic - 100));
        // 40 left at the end: 50 do not fit until something is given back.
        assert_eq!(heap.dynamic(50), None);
        // A slot that stops viewing a resource leaves it alive until the GPU
        // work in flight has run (a read of a freed host texture otherwise).
        // SAFETY: a plain committed buffer; one reference taken and given back.
        unsafe {
            let host = host().unwrap();
            let mut buffer: Option<ID3D12Resource> = None;
            host.device
                .CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES {
                        Type: D3D12_HEAP_TYPE_DEFAULT,
                        ..Default::default()
                    },
                    D3D12_HEAP_FLAG_NONE,
                    &D3D12_RESOURCE_DESC {
                        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                        Width: 256,
                        Height: 1,
                        DepthOrArraySize: 1,
                        MipLevels: 1,
                        SampleDesc: DXGI_SAMPLE_DESC {
                            Count: 1,
                            Quality: 0,
                        },
                        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                        ..Default::default()
                    },
                    D3D12_RESOURCE_STATE_COMMON,
                    None,
                    &mut buffer,
                )
                .unwrap();
            let buffer = buffer.unwrap();
            let references = |b: &ID3D12Resource| {
                let raw = windows::core::Interface::as_raw(b);
                let unknown = &*(raw as *const *const windows::core::IUnknown_Vtbl);
                ((**unknown).AddRef)(raw);
                ((**unknown).Release)(raw)
            };
            let list = host.new_list().unwrap();
            list.close();
            let wait_for_retirement = || {
                crate::host::with_bury_lag(0, || host.execute(&[&list]));
                let since = std::time::Instant::now();
                while references(&buffer) > 1 && since.elapsed() < std::time::Duration::from_secs(5)
                {
                    std::thread::yield_now();
                }
                assert_eq!(references(&buffer), 1, "ours");
            };

            // A low vacancy remains reachable by computed heap indices.
            heap.raw_view_at(a, Some(&buffer), 0, 256, false);
            assert_eq!(references(&buffer), 2, "ours and the slot's");
            heap.release(heap.id(), a, 50);
            assert!(heap.kept(a));
            assert_eq!(heap.dynamic(50), Some(a));
            heap.make_null(a);
            assert!(!heap.kept(a));
            wait_for_retirement();

            // This path requires a host capable of allocating a full heap
            // plus spill slots. A capped device exercises the case above.
            let bound = victory_shader::emit::HEAP_VIEWS;
            if let Some(high) = HostHeap::new(false, bound)
                && high.count() >= bound
            {
                let slot = high.dynamic(1).unwrap();
                assert!(slot >= bound);
                eprintln!(
                    "checking retirement above the computed resource heap limit at slot {slot}"
                );
                high.raw_view_at(slot, Some(&buffer), 0, 256, false);
                assert_eq!(references(&buffer), 2, "ours and the slot's");
                high.release(high.id(), slot, 1);
                assert!(!high.kept(slot));
                wait_for_retirement();
            }
        }
        assert_eq!(heap.dynamic(40), Some(4 + heap.dynamic - 40));
    }

    #[test]
    fn free_runs_are_found_across_words() {
        // Slots 0-69, 100-109 and 201-255 taken: free runs 70-99, 110-200
        // and 256-300 (the last one past a word's end, `to` at 301).
        let mut taken = vec![0u64; 5];
        set_taken(&mut taken, 0, 70, true);
        set_taken(&mut taken, 100, 110, true);
        set_taken(&mut taken, 201, 256, true);
        assert_eq!(free_run(&taken, 0, 301, 30), Some(70));
        assert_eq!(free_run(&taken, 0, 301, 31), Some(110));
        assert_eq!(free_run(&taken, 0, 301, 91), Some(110));
        assert_eq!(free_run(&taken, 0, 301, 92), None);
        assert_eq!(free_run(&taken, 0, 200, 91), None, "a run ends before `to`");
        assert_eq!(free_run(&taken, 0, 201, 91), Some(110));
        assert_eq!(free_run(&taken, 120, 301, 81), Some(120));
        assert_eq!(free_run(&taken, 120, 301, 82), None);
        // Every other slot taken: no two clear in a row.
        let alternate = [0x5555_5555_5555_5555u64; 2];
        assert_eq!(free_run(&alternate, 0, 128, 2), None);
        assert_eq!(free_run(&alternate, 0, 128, 1), Some(1));
        assert_eq!(free_run(&alternate, 3, 128, 1), Some(3));
        // A word's last (or first) slot taken ends the run there.
        assert_eq!(free_run(&[1 << 63, 0], 0, 128, 63), Some(0));
        assert_eq!(free_run(&[1 << 63, 0], 0, 128, 64), Some(64));
        assert_eq!(free_run(&[0, 1], 0, 128, 64), Some(0));
        assert_eq!(free_run(&[0, 1], 0, 128, 65), None);
        // Runs longer than a word, from a clear word on.
        assert_eq!(free_run(&[0u64; 2], 10, 128, 118), Some(10));
        assert_eq!(free_run(&[0u64; 2], 10, 128, 119), None);
        assert_eq!(free_run(&[0u64; 2], 10, 128, 100), Some(10));
        // 128-191 is a whole free word between taken ones.
        set_taken(&mut taken, 128, 192, true);
        assert_eq!(free_run(&taken, 0, 301, 45), Some(256));
        assert_eq!(free_run(&taken, 0, 301, 46), None);
        assert_eq!(free_run(&taken, 100, 301, 20), Some(256));
        set_taken(&mut taken, 128, 192, false);
        assert_eq!(free_run(&taken, 128, 301, 64), Some(128));
        set_taken(&mut taken, 128, 192, true);
        assert_eq!(free_run(&taken, 128, 301, 1), Some(192));
    }

    #[test]
    fn dynamic_search_can_find_a_run_across_the_cursor() {
        let Some(heap) = HostHeap::new(false, 4) else {
            eprintln!("no host descriptor heap; skipped");
            return;
        };
        let first = heap.dynamic(1).unwrap();
        let left = heap.dynamic(2).unwrap();
        let right = heap.dynamic(2).unwrap();
        let rest = heap.dynamic(heap.dynamic - 5).unwrap();
        assert_eq!((first, left, right, rest), (4, 5, 7, 9));

        // Reallocating the left half puts the cursor back in the middle.
        heap.release(heap.id(), left, 2);
        assert_eq!(heap.dynamic(2), Some(left));
        heap.release(heap.id(), left, 2);
        heap.release(heap.id(), right, 2);
        assert_eq!(heap.dynamic(4), Some(left));
        assert_eq!(heap.dynamic(1), None);

        // A free slot at each physical end does not form a two-slot range.
        heap.release(heap.id(), first, 1);
        let last = first + heap.dynamic - 1;
        heap.release(heap.id(), last, 1);
        assert_eq!(heap.dynamic(2), None);
    }

    /// A range of dynamic slots is returned only to the heap that allocated it:
    /// a stale release carrying another heap's identity must be a no-op.
    #[test]
    fn dynamic_slots_are_given_back_only_to_the_heap_that_took_them() {
        let Some(a) = HostHeap::new(false, 4) else {
            return; // no host GPU
        };
        let Some(b) = HostHeap::new(false, 4) else {
            return;
        };
        assert_ne!(a.id(), b.id());
        let first = a.dynamic(2).expect("two dynamic slots");
        let rest = a.dynamic(a.dynamic - 2).expect("remaining dynamic slots");
        assert_eq!(rest, first + 2);
        assert_eq!(a.dynamic(1), None, "heap is full");

        // Releasing with another heap's identity must not free A's slots.
        a.release(b.id(), first, 2);
        assert_eq!(a.dynamic(1), None, "foreign heap id must not free A's slots");

        // Releasing with A's own identity gives the range back.
        a.release(a.id(), first, 2);
        assert_eq!(a.dynamic(2), Some(first), "own heap id returns the range");
    }

    /// A texture view's size words are the console image descriptor's
    /// fields: level 0's size of the resource, the view's levels, its type.
    #[test]
    fn a_texture_view_gives_its_image_descriptor_size_fields() {
        // A 1000x600 2D texture of 10 levels, then an array of 6 with the
        // whole chain (mip count 0, 1024 wide: 11 levels).
        let mut texture = [0u32; 14];
        (texture[0], texture[4], texture[6], texture[7], texture[9]) =
            (3, 1000, 600, 1 | 10 << 16, 1);
        let mut array = texture;
        (array[4], array[7]) = (1024, 6);
        let mut view = [0u32; 10];
        // Levels 2-4 of the texture: width - 1 = 999 (0b11_1110_0111).
        (view[1], view[4], view[5]) = (4, 2, 3);
        let words = image_words(&texture, Some(&view), false);
        assert_eq!(
            words,
            [
                0,
                3 << 30,
                249 | 599 << 14,
                0xFAC | 2 << 12 | 4 << 16 | 9 << 28
            ]
        );
        // Its whole chain from level 1.
        (view[4], view[5]) = (1, u32::MAX);
        assert_eq!(
            image_words(&texture, Some(&view), false)[3],
            0xFAC | 1 << 12 | 9 << 16 | 9 << 28
        );
        // An unordered view of level 3: that level alone.
        let uav = [0, 4, 0, 0, 3, 0, 0, 0, 0, 0];
        assert_eq!(
            image_words(&texture, Some(&uav), true)[3],
            0xFAC | 3 << 12 | 3 << 16 | 9 << 28
        );
        // The array's own view: a 2D array, levels 0-10.
        let words = image_words(&array, None, false);
        assert_eq!(
            words[1..],
            [3 << 30, 255 | 599 << 14, 0xFAC | 10 << 16 | 13 << 28]
        );
    }

    fn placed_memory() -> u64 {
        let layout = std::alloc::Layout::from_size_align(1 << 16, 1 << 16).unwrap();
        // SAFETY: non-zero size and power-of-two alignment; leaked for the test.
        unsafe { std::alloc::alloc_zeroed(layout) as u64 }
    }

    fn placed_tex_desc(width: u64, height: u32, format: u32, flags: u32) -> [u32; 14] {
        [
            3,
            0,
            0,
            0,
            width as u32,
            (width >> 32) as u32,
            height,
            1 | (1 << 16),
            format,
            1,
            0,
            0,
            flags,
            0,
        ]
    }

    fn placed_tex_uav_desc(va: u64, view_format: u32) -> [u32; 16] {
        let mut desc = [0u32; 16];
        desc[0] = view_format;
        desc[1] = 4; // D3D12_UAV_DIMENSION_TEXTURE2D
        desc[10] = va as u32;
        desc[11] = (va >> 32) as u32;
        desc
    }

    fn placed_tex_srv_desc(va: u64, view_format: u32) -> [u32; 16] {
        let mut desc = [0u32; 16];
        desc[0] = view_format;
        desc[1] = 4; // D3D12_SRV_DIMENSION_TEXTURE2D
        desc[2] = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        desc[5] = 1; // MipLevels
        desc[10] = va as u32;
        desc[11] = (va >> 32) as u32;
        desc
    }

    #[test]
    fn placed_uav_textures_create_real_host_uavs() {
        let Some(host) = host() else {
            return;
        };
        let Some(heap) = HostHeap::new(false, 32) else {
            return;
        };
        let _ = host.log_messages();
        let bgrx_uav = typed_store(88);
        let cases = [
            (28u32, 28u32, 4u32, true),     // R8G8B8A8_UNORM
            (27u32, 28u32, 4u32, true),     // R8G8B8A8_TYPELESS -> R8G8B8A8_UNORM
            (29u32, 29u32, 4u32, true),     // R8G8B8A8_UNORM_SRGB -> linear R8G8B8A8_UNORM
            (29u32, 0u32, 4u32, true),      // R8G8B8A8_UNORM_SRGB + UNKNOWN -> linear 28
            (91u32, 91u32, 4u32, true),     // B8G8R8A8_UNORM_SRGB -> linear B8G8R8A8_UNORM
            (91u32, 0u32, 4u32, true),      // B8G8R8A8_UNORM_SRGB + UNKNOWN -> linear 87
            (28u32, 28u32, 0u32, false),    // missing ALLOW_UNORDERED_ACCESS -> null fallback
            (93u32, 93u32, 4u32, bgrx_uav), // B8G8R8X8_UNORM_SRGB -> linear 88 if supported
            (93u32, 0u32, 4u32, bgrx_uav),  // B8G8R8X8_UNORM_SRGB + UNKNOWN -> linear 88
            (92u32, 88u32, 4u32, bgrx_uav), // B8G8R8X8_TYPELESS -> B8G8R8X8_UNORM
            (92u32, 93u32, 4u32, bgrx_uav), // B8G8R8X8_TYPELESS -> B8G8R8X8_UNORM_SRGB
            (86u32, 86u32, 4u32, true),     // B5G5R5A1_UNORM, widened if unsupported
            (115u32, 115u32, 4u32, true),   // B4G4R4A4_UNORM, widened if unsupported
            (40u32, 40u32, 4u32, false),    // D32_FLOAT + UAV flag (no depth flag) -> null fallback
            (55u32, 55u32, 4u32, false),    // D16_UNORM + UAV flag (no depth flag) -> null fallback
            (39u32, 40u32, 4u32, false),    // R32_TYPELESS + D32_FLOAT view -> null fallback
            (53u32, 55u32, 4u32, false),    // R16_TYPELESS + D16_UNORM view -> null fallback
            (27u32, 27u32, 4u32, false),    // R8G8B8A8_TYPELESS + typeless view -> null fallback
        ];
        let mut slots = [[0u32; 8]; 24];
        for (idx, (res_fmt, view_fmt, flags, expect_real)) in cases.into_iter().enumerate() {
            let va = placed_memory();
            let resource = placed_tex_desc(4, 4, res_fmt, flags);
            let desc = placed_tex_uav_desc(va, view_fmt);
            let slot_addr = slots[idx].as_mut_ptr() as u64;
            crate::descriptor::create_placed_uav(std::ptr::null(), &resource, &desc, slot_addr);
            let view = crate::descriptor::view(slot_addr).unwrap();
            assert!(matches!(
                placed_texture(&view),
                Some(View::Unordered(_, 0, Some(_)))
            ));
            assert!(heap.mirror(idx as u32, slot_addr, false).is_some());
            assert!(
                heap.kept.lock().unwrap().get(idx).is_none(),
                "case {idx} mirror(false) should defer host creation"
            );
            let mirrored = heap.mirror(idx as u32, slot_addr, true).unwrap();
            let dyn_slot = heap.count + idx as u32;
            heap.make_at(dyn_slot, &mirrored);
            let kept = heap.kept.lock().unwrap();
            assert_eq!(kept.get(idx).is_some(), expect_real, "case {idx} mirror");
            assert_eq!(
                kept.get(dyn_slot as usize).is_some(),
                expect_real,
                "case {idx} make_at"
            );
        }
        for (offset, (res_fmt, view_fmt, expect_real)) in [
            (39u32, 40u32, false), // R32_TYPELESS + D32_FLOAT SRV -> null fallback
            (53u32, 55u32, false), // R16_TYPELESS + D16_UNORM SRV -> null fallback
            (29u32, 0u32, true),   // R8G8B8A8_UNORM_SRGB + UAV + UNKNOWN SRV -> 29
            (91u32, 0u32, true),   // B8G8R8A8_UNORM_SRGB + UAV + UNKNOWN SRV -> 91
        ]
        .into_iter()
        .enumerate()
        {
            let idx = cases.len() + offset;
            let va = placed_memory();
            let resource = placed_tex_desc(4, 4, res_fmt, 4);
            let desc = placed_tex_srv_desc(va, view_fmt);
            let slot_addr = slots[idx].as_mut_ptr() as u64;
            crate::descriptor::create_placed_srv(std::ptr::null(), &resource, &desc, slot_addr);
            heap.mirror(idx as u32, slot_addr, true).unwrap();
            assert_eq!(
                heap.kept.lock().unwrap().get(idx).is_some(),
                expect_real,
                "srv case {offset}"
            );
        }
        // SAFETY: plain device query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
        assert_eq!(host.log_messages(), None);
    }

    /// Wine's D3DMetal aborts while creating these typed UAVs. The production
    /// path must leave a null descriptor, no kept resource and no write mark.
    fn three_channel_uav_falls_back_under_wine(format: DXGI_FORMAT) {
        let Some(host) = host() else { return };
        let Some(heap) = HostHeap::new(false, 4) else {
            return;
        };
        let p = crate::watch::tests::pool(1);
        crate::shadow::register(p, p as u64, 64);
        let mut resource = [0u32; 14];
        resource[0] = 1; // buffer
        resource[4] = 64;
        let mut words = [0u32; 16];
        words[0] = format.0 as u32;
        words[1] = 1; // buffer
        words[4] = 1; // one 12-byte element
        (words[10], words[11]) = (p as u32, (p >> 32) as u32);
        let slot = heap.dynamic(1).expect("a dynamic slot");
        heap.make_at(slot, &View::Placed(true, resource, words));
        assert!(!heap.kept(slot), "a refused UAV keeps no resource");
        let marks = heap.marks.as_ref().expect("the write-mark table").1;
        // SAFETY: two dwords of the mapped table for this live slot.
        let mark =
            unsafe { std::slice::from_raw_parts((marks as *const u32).add(slot as usize * 2), 2) };
        assert_eq!(
            mark,
            [victory_shader::emit::VIEW_MARK_NONE, 0],
            "a refused UAV has no write mark"
        );
        assert!(
            // SAFETY: a plain device query.
            unsafe { host.device.GetDeviceRemovedReason() }.is_ok(),
            "the null fallback leaves the device alive"
        );
        crate::shadow::unregister(p, p as u64);
        crate::watch::tests::free(p);
    }

    /// Synthetic hardware characterization: create a typed R32G32B32 UAV,
    /// store one three-component element, and copy the raw bytes back. Wine
    /// exercises the safe fallback instead, because raw creation aborts.
    fn characterize_r32g32b32(format: DXGI_FORMAT, scalar: &str, value: &str, expected: [u32; 3]) {
        eprintln!("typed UAV characterization: {format:?}");
        if crate::under_wine() {
            three_channel_uav_falls_back_under_wine(format);
            eprintln!("typed UAV characterization: Wine fallback passed");
            return;
        }
        let Some(host) = host() else {
            eprintln!("typed UAV characterization: no host device");
            return;
        };
        let device = &host.device;
        let mut support = D3D12_FEATURE_DATA_FORMAT_SUPPORT {
            Format: format,
            ..Default::default()
        };
        // SAFETY: a plain query into a struct of its size.
        let support_result = unsafe {
            device.CheckFeatureSupport(
                D3D12_FEATURE_FORMAT_SUPPORT,
                (&raw mut support).cast(),
                size_of_val(&support) as u32,
            )
        };
        eprintln!(
            "typed UAV characterization: format support {support_result:?}, Support1={:#x}",
            support.Support1.0
        );
        if support_result.is_err()
            || support.Support1.0 & D3D12_FORMAT_SUPPORT1_TYPED_UNORDERED_ACCESS_VIEW.0 == 0
        {
            eprintln!(
                "typed UAV characterization: host does not advertise typed UAV support; skipped"
            );
            return;
        }
        let mut made: Option<ID3D12Resource> = None;
        // SAFETY: a plain committed buffer description.
        unsafe {
            device
                .CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES {
                        Type: D3D12_HEAP_TYPE_DEFAULT,
                        ..Default::default()
                    },
                    D3D12_HEAP_FLAG_NONE,
                    &D3D12_RESOURCE_DESC {
                        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                        Width: 256,
                        Height: 1,
                        DepthOrArraySize: 1,
                        MipLevels: 1,
                        Format: DXGI_FORMAT_UNKNOWN,
                        SampleDesc: DXGI_SAMPLE_DESC {
                            Count: 1,
                            Quality: 0,
                        },
                        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                        Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                        ..Default::default()
                    },
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    None,
                    &mut made,
                )
                .unwrap();
        }
        let buffer = made.unwrap();
        // SAFETY: a plain shader-visible descriptor heap.
        let heap: ID3D12DescriptorHeap = unsafe {
            device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: 1,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                ..Default::default()
            })
        }
        .unwrap();
        let desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: format,
            ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_UAV {
                    NumElements: 1,
                    ..Default::default()
                },
            },
        };
        // SAFETY: a live buffer and one slot of our heap.
        unsafe {
            let cpu = heap.GetCPUDescriptorHandleForHeapStart();
            device.CreateUnorderedAccessView(&buffer, None::<&ID3D12Resource>, Some(&desc), cpu);
        }
        // SAFETY: a plain device query.
        let removed = unsafe { device.GetDeviceRemovedReason() };
        eprintln!("typed UAV characterization: after creation {removed:?}");
        removed.expect("creating the typed UAV leaves the device alive");
        let source = format!(
            "// {format:?}\nRWBuffer<{scalar}3> output : register(u0);\n[numthreads(1, 1, 1)]\nvoid main() {{ output[0] = {scalar}3({value}); }}\n"
        );
        let (root, pso) = crate::compute::pipeline(
            &source,
            windows::core::s!("typed uav characterization"),
            0,
            &[crate::compute::Param::Table(&[crate::compute::uav(0, 1)])],
        )
        .expect("the typed-store shader compiles");
        let list = host.new_list().unwrap();
        let (readback, data) = list.readback(host, 256).unwrap();
        // SAFETY: an open list, live objects, and a descriptor in our heap.
        unsafe {
            let gpu = heap.GetGPUDescriptorHandleForHeapStart();
            list.list.SetDescriptorHeaps(&[Some(heap.clone())]);
            list.list.SetComputeRootSignature(&root);
            list.list.SetComputeRootDescriptorTable(0, gpu);
            list.list.SetPipelineState(&pso);
            list.list.Dispatch(1, 1, 1);
            list.list.ResourceBarrier(&[crate::host::transition(
                &buffer,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            list.list.CopyResource(&readback, &buffer);
        }
        list.close();
        host.execute(&[&list]);
        // SAFETY: execute waited for the copy; the mapped readback is 256 bytes.
        let bytes = unsafe { std::slice::from_raw_parts(data.cast::<u8>(), 36) };
        eprintln!("typed UAV characterization: raw readback {bytes:?}");
        let want: Vec<u8> = expected
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect();
        assert_eq!(bytes, want, "typed UAV store/raw readback");
    }

    #[test]
    fn characterize_r32g32b32_float_typed_uav() {
        characterize_r32g32b32(
            DXGI_FORMAT_R32G32B32_FLOAT,
            "float",
            "1.25, -2.5, 3.75",
            [1.25f32.to_bits(), (-2.5f32).to_bits(), 3.75f32.to_bits()],
        );
    }

    #[test]
    fn characterize_r32g32b32_uint_typed_uav() {
        characterize_r32g32b32(
            DXGI_FORMAT_R32G32B32_UINT,
            "uint",
            "0x11223344u, 0x55667788u, 0x99aabbccu",
            [0x1122_3344, 0x5566_7788, 0x99aa_bbcc],
        );
    }

    #[test]
    fn characterize_r32g32b32_sint_typed_uav() {
        characterize_r32g32b32(
            DXGI_FORMAT_R32G32B32_SINT,
            "int",
            "-1, -2, -3",
            [(-1i32) as u32, (-2i32) as u32, (-3i32) as u32],
        );
    }

    #[test]
    fn sync_slots_skips_empty_and_clears_after_forget() {
        use crate::object::{HRESULT, S_OK};
        let Some(host) = HostHeap::new(false, 256) else {
            return;
        };
        let desc: [u32; 14] = [1, 0, 0, 0, 64, 0, 1, 1 | 1 << 16, 0, 1, 0, 0, 0, 0];
        let properties = [1u32, 0, 0, 0, 0];
        let mut resource = std::ptr::null();
        let iid = crate::object::guid(
            0x696442be,
            0xa72e,
            0x4059,
            [0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad],
        );
        let made: HRESULT = crate::resource::create_committed(
            std::ptr::null(),
            properties.as_ptr().cast(),
            0,
            desc.as_ptr().cast(),
            0,
            std::ptr::null(),
            &iid,
            &mut resource,
        );
        assert_eq!(made, S_OK);

        let mut title_heap = vec![0u8; 256 * 32];
        let mut bytes = Vec::new();
        let start = title_heap.as_ptr() as u64;
        crate::descriptor::create_srv(std::ptr::null(), resource, std::ptr::null(), start);
        host.sync_slots(0, 256, start, 32, &mut bytes);
        assert!(host.made(0));
        assert!(host.made_is_nonzero(0));
        assert!(!host.made(1));
        assert!(!host.made_is_nonzero(1));

        // Second sync over the verified block keeps slot 0 live and skips empty slots.
        host.sync_slots(0, 256, start, 32, &mut bytes);
        assert!(host.made(0));

        // Forgetting the resource bumps VIEWS_FORGOT so the verified block re-checks
        // and clears slot 0 even though the title memory bytes didn't change.
        crate::descriptor::forget(resource as usize);
        host.sync_slots(0, 256, start, 32, &mut bytes);
        assert!(!host.made(0));
        assert!(!host.made_is_nonzero(0));

        crate::object::release(resource);
        let _ = title_heap.as_mut_ptr();
    }

    /// PR #1246 fair queue: previously retried slots and never-looked-at fresh
    /// slots alternate 1:1 (a retried slot first), so a large retry backlog
    /// cannot starve fresh slots; `fair = false` restores retries-first order.
    #[test]
    fn upload_queue_interleaves_retries_and_fresh_slots() {
        let mut q = UploadQueue::default();
        for i in 0..10u32 {
            q.written(100 + i, 1);
        }
        for i in 0..10u32 {
            let w = q
                .due_with(u64::MAX, 100, true)
                .into_iter()
                .find(|d| d.0 == 100 + i)
                .unwrap()
                .1;
            q.later(100 + i, w, 1, 1);
        }
        for i in 1..4u32 {
            q.written(i, 2);
        }
        let fair: Vec<u32> = q.due_with(10, 6, true).into_iter().map(|d| d.0).collect();
        assert_eq!(fair.len(), 6);
        assert!(fair[0] >= 100, "first slot is a retry: {fair:?}");
        assert!(fair[1] < 100, "second slot is fresh: {fair:?}");
        assert!(fair[2] >= 100, "third slot is a retry: {fair:?}");
        assert!(fair[3] < 100, "fourth slot is fresh: {fair:?}");
        assert_eq!(
            fair.iter().filter(|&&i| i < 100).count(),
            3,
            "all fresh slots reached in pass of 6: {fair:?}"
        );
        let legacy: Vec<u32> = q.due_with(10, 6, false).into_iter().map(|d| d.0).collect();
        assert!(
            legacy.iter().all(|&i| i >= 100),
            "legacy mode puts retries ahead of fresh: {legacy:?}"
        );
    }

    /// Tier-0 read-marked slots preempt retries and fresh slots, `requeue`
    /// restores unqueued slots without disturbing queued ones, and relook
    /// backoff clamps to `(VICTORY_RELOOK_MS / frame_ms).clamp(16, 1024)`.
    #[test]
    fn upload_queue_prioritizes_tier0_reads_and_respects_relook_cap() {
        assert_eq!(relook_frames(1000, 20_000), 20);
        assert_eq!(relook_frames(5000, 20_000), 16, "clamps low at 16");
        assert_eq!(relook_frames(1, 20_000), 1024, "clamps high at 1024");
        assert_eq!(relook_frames(250, 20_000), 80);

        let mut q = UploadQueue::default();
        q.written(10, 1);
        q.written(11, 1);
        let w10 = q.due_with(1, 10, true).into_iter().find(|d| d.0 == 10).unwrap().1;
        q.later(10, w10, 1, 16);
        // After 1 look, due_frame = 2; after repeated looks, backoff caps at 16.
        for f in [2u64, 4, 8, 16, 32, 64] {
            let w = q.queued(&[10])[0].1;
            q.later(10, w, f, 16);
        }
        assert!(
            !q.due_with(79, 10, true).iter().any(|d| d.0 == 10),
            "capped at +16 frames (due at 80)"
        );
        assert!(q.due_with(80, 10, true).iter().any(|d| d.0 == 10));

        // Tier-0 prioritize puts slot 50 ahead of both retries and fresh slots.
        q.prioritize(&[50], 1);
        let first = q.due_with(80, 3, true);
        assert_eq!(first[0].0, 50, "Tier-0 read-marked slot comes first");

        // `requeue` adds unqueued slot 60 as urgent without resetting slot 11.
        let before11 = q.queued(&[11]);
        q.requeue(&[11, 60], 80);
        assert_eq!(q.queued(&[11]), before11);
        let due = q.due_with(80, 4, true);
        assert_eq!(due[0].0, 60);
        assert_eq!(due[1].0, 50);
    }

    /// PR #1007 publication invariant: a slot published with its matching key
    /// becomes live without re-entering the upload queue, and a stale key is
    /// rejected after the title overwrites the slot.
    #[test]
    fn published_slot_stays_out_of_queue_and_rejects_stale_key() {
        let Some(heap) = HostHeap::new(false, 8) else {
            return; // no host GPU
        };
        let _worker = crate::heap_worker::EnableForTest::new();
        let mem = crate::watch::tests::pool(1);
        let placed = |at: u64| {
            let mut desc = [0u32; 16];
            // RGBA8 4x1 linear, 1 mip
            (desc[0], desc[1], desc[2], desc[5]) = (28, 4, 0x1688, 1);
            (desc[10], desc[11]) = (at as u32, (at >> 32) as u32);
            View::Placed(
                false,
                [3, 0, 0, 0, 4, 0, 1, 1 | 1 << 16, 28, 1, 0, 0x100, 0, 0],
                desc,
            )
        };
        let mut title = [0u32; 8];
        let handle = title.as_mut_ptr() as u64;
        crate::descriptor::record(handle, placed(mem as u64));
        let (_, key1) = crate::descriptor::view_and_key(handle).unwrap();
        assert!(heap.mirror(1, handle, false).is_some());
        assert!(!heap.kept(1), "placed texture starts null until published");
        let due = heap.texture_slots_due(u64::MAX, 8);
        assert_eq!(due.len(), 1);
        assert_eq!(due[0].0, 1);
        heap.texture_slot_done(1, due[0].1);
        assert!(heap.publish(1, handle, key1));
        assert!(heap.kept(1), "slot now shows the published texture");
        assert!(
            heap.texture_slots_due(u64::MAX, 8).is_empty(),
            "publish must not re-enqueue the slot"
        );
        // Overwrite the slot with another record: stale `key1` must be rejected.
        crate::descriptor::record(handle, placed((mem + 256) as u64));
        let (_, key2) = crate::descriptor::view_and_key(handle).unwrap();
        assert_ne!(key1, key2);
        assert!(!heap.publish(1, handle, key1));
        crate::watch::tests::free(mem);
    }

    #[test]
    fn fast_heap_sync_env_toggle() {
        use std::ffi::OsStr;
        assert!(fast_heap_sync_value(None));
        assert!(fast_heap_sync_value(Some(OsStr::new("1"))));
        assert!(fast_heap_sync_value(Some(OsStr::new(""))));
        assert!(!fast_heap_sync_value(Some(OsStr::new("0"))));
    }

    #[test]
    fn fast_heap_sync_handles_placed_null_duplicates_forget_and_concurrent_rewrite() {
        let Some(host) = HostHeap::new(false, 512) else {
            return;
        };
        let make_shadowed_buffer = || {
            let b = crate::resource::test_buffer(256);
            let (ptr, len) = crate::resource::buffer_bytes(b).expect("buffer bytes");
            crate::shadow::register(b as usize, ptr as u64, len);
            b
        };
        let buf_a = make_shadowed_buffer();
        let buf_b = make_shadowed_buffer();

        let mut title_heap = vec![0u8; 512 * 32];
        let start = title_heap.as_ptr() as u64;
        let slot_addr = |i: u32| start + u64::from(i) * 32;
        // R8_UINT buffer SRV, all 256 elements (valid host buffer SRV; word 6 = NumElements).
        let srv_desc = [62u32, 1, 0x1688, 0, 0, 0, 256, 0, 0, 0];

        // 10 duplicate copies of buffer SRV A (exercises template copy via StagingHeap).
        for i in 0..10u32 {
            crate::descriptor::create_srv(std::ptr::null(), buf_a, &srv_desc, slot_addr(i));
        }

        // 10 duplicate copies of a placed 2D texture (left null on sync, ready == 0, synced == key).
        let tex_mem = vec![0u8; 1 << 16];
        let tex_addr = tex_mem.as_ptr() as u64;
        let tex_words: [u32; 14] = [3, 0, 0, 0, 4, 0, 1, 1 | 1 << 16, 28, 1, 0, 0x100, 0, 0];
        let mut tex_view = [0u32; 16];
        tex_view[..6].copy_from_slice(&[28, 4, 0x1688, 0, 0, 1]);
        (tex_view[10], tex_view[11]) = (tex_addr as u32, (tex_addr >> 32) as u32);
        for i in 20..30u32 {
            crate::descriptor::create_placed_srv(
                std::ptr::null(),
                &tex_words,
                &tex_view,
                slot_addr(i),
            );
        }

        // First fast sync populates buffer slots via staging template and placed textures via null template.
        host.sync_slots_bytes_with(0..512, start, 32, start, &title_heap, true);
        for i in 0..10u32 {
            assert!(host.made(i), "buffer slot {i} made");
            assert!(host.kept(i), "buffer slot {i} kept");
            assert_ne!(host.synced[i as usize].load(Ordering::Acquire), 0);
            assert_eq!(
                host.ready[i as usize].load(Ordering::Acquire),
                host.synced[i as usize].load(Ordering::Acquire)
            );
        }
        for i in 20..30u32 {
            assert!(host.made(i), "placed texture slot {i} noted");
            assert!(!host.kept(i), "placed texture slot {i} left null until drawn");
            assert_ne!(host.synced[i as usize].load(Ordering::Acquire), 0);
            assert_eq!(host.ready[i as usize].load(Ordering::Acquire), 0);
        }

        // Second fast sync skips all slots locklessly and preserves placed texture ready == 0.
        host.sync_slots_bytes_with(0..512, start, 32, start, &title_heap, true);
        for i in 20..30u32 {
            assert!(!host.kept(i));
            assert_eq!(host.ready[i as usize].load(Ordering::Acquire), 0);
        }
        // Drawing slot 20 still remakes the placed texture on demand.
        assert!(host.mirror(20, slot_addr(20), true).is_some());
        assert!(host.kept(20), "placed texture made on draw");

        // Forgetting buffer A immediately invalidates all 10 buffer slots on next sync
        // even though title heap bytes did not change.
        crate::descriptor::forget(buf_a as usize);
        host.sync_slots_bytes_with(0..512, start, 32, start, &title_heap, true);
        for i in 0..10u32 {
            assert!(!host.made(i), "forgotten slot {i} cleared");
            assert!(!host.made_is_nonzero(i), "forgotten slot {i} nonzero bit cleared");
            assert!(!host.kept(i), "forgotten slot {i} no longer kept");
            assert_eq!(host.synced[i as usize].load(Ordering::Acquire), 0);
            assert_eq!(host.ready[i as usize].load(Ordering::Acquire), 0);
        }

        // Concurrent title rewrite during `prepare` leaves the slot holding the latest view.
        let buf_c = make_shadowed_buffer();
        let race_slot = 64u32;
        let race_addr = slot_addr(race_slot);
        crate::descriptor::create_srv(std::ptr::null(), buf_b, &srv_desc, race_addr);
        let buf_c_addr = buf_c as usize;
        let mut expected_c = [0u32; 4];
        crate::descriptor::create_srv(
            std::ptr::null(),
            buf_c,
            &srv_desc,
            expected_c.as_mut_ptr() as u64,
        );
        let expected_c_key = u64::from(expected_c[2]) << 32 | u64::from(expected_c[1]);
        *WHILE_PREPARING.lock().unwrap_or_else(|e| e.into_inner()) = Some(Box::new(move || {
            crate::descriptor::create_srv(
                std::ptr::null(),
                buf_c_addr as crate::object::This,
                &srv_desc,
                race_addr,
            );
        }));
        host.sync_slots_bytes_with(0..128, start, 32, start, &title_heap[..128 * 32], true);
        assert!(WHILE_PREPARING.lock().unwrap_or_else(|e| e.into_inner()).is_none());
        assert_eq!(
            host.synced[race_slot as usize].load(Ordering::Acquire),
            expected_c_key,
            "concurrent rewrite during prepare won"
        );
        assert!(host.kept(race_slot));
        // SAFETY: plain device health check after batched CopyDescriptors.
        assert!(unsafe { crate::host::host().unwrap().device.GetDeviceRemovedReason() }.is_ok());

        crate::object::release(buf_a);
        crate::object::release(buf_b);
        crate::object::release(buf_c);
        let _ = title_heap.as_mut_ptr();
        let _ = tex_mem.as_ptr();
    }

    #[test]
    fn bench_sync_slots_1_5m_fast_vs_legacy() {
        const SLOTS: u32 = 65_536; // 32 x 64 KiB pages (2,048 32-byte slots/page)
        const PASSES: usize = 23; // 23 * 65,536 = 1,507,328 slots scanned
        const PAGE_SLOTS: u32 = 2_048;

        let Some(host_legacy) = HostHeap::new(false, SLOTS) else {
            return;
        };
        let Some(host_fast) = HostHeap::new(false, SLOTS) else {
            return;
        };

        // Valid R8_UINT buffer SRV covering all 256 bytes (word 6 = NumElements).
        let srv_desc = [62u32, 1, 0x1688, 0, 0, 0, 256, 0, 0, 0];

        // 150 distinct buffer resources (10x duplication across active/stale slots),
        // registered with shadow so every buffer SRV creates a real D3D12 view + bindless hull.
        let make_shadowed_buffer = || {
            let b = crate::resource::test_buffer(256);
            let (ptr, len) = crate::resource::buffer_bytes(b).expect("buffer bytes");
            crate::shadow::register(b as usize, ptr as u64, len);
            b
        };
        let buffers: Vec<_> = (0..150).map(|_| make_shadowed_buffer()).collect();
        // Extra resource forgotten after benchmark to verify Views/HostHeap invalidation.
        let ephemeral = make_shadowed_buffer();

        let tex_mem = vec![0u8; 1 << 16];
        let tex_addr = tex_mem.as_ptr() as u64;
        let tex_words: [u32; 14] = [3, 0, 0, 0, 4, 0, 1, 1 | 1 << 16, 28, 1, 0, 0x100, 0, 0];
        let mut placed_records: Vec<[u32; 3]> = Vec::with_capacity(40);
        for k in 0..40u64 {
            let mut view = [0u32; 16];
            view[..6].copy_from_slice(&[28, 4, 0x1688, 0, 0, 1]);
            let addr = tex_addr + k * 256;
            (view[10], view[11]) = (addr as u32, (addr >> 32) as u32);
            let mut rec = [0u32; 3];
            crate::descriptor::create_placed_srv(
                std::ptr::null(),
                &tex_words,
                &view,
                rec.as_mut_ptr() as u64,
            );
            placed_records.push(rec);
        }

        let mut buf_records: Vec<[u32; 3]> = Vec::with_capacity(buffers.len());
        for &b in &buffers {
            let mut rec = [0u32; 3];
            crate::descriptor::create_srv(
                std::ptr::null(),
                b,
                &srv_desc,
                rec.as_mut_ptr() as u64,
            );
            buf_records.push(rec);
        }
        let mut eph_record = [0u32; 3];
        crate::descriptor::create_srv(
            std::ptr::null(),
            ephemeral,
            &srv_desc,
            eph_record.as_mut_ptr() as u64,
        );

        let write_rec = |mem: &mut [u8], slot: usize, rec: [u32; 3]| {
            let off = slot * 32;
            mem[off..off + 4].copy_from_slice(&rec[0].to_le_bytes());
            mem[off + 4..off + 8].copy_from_slice(&rec[1].to_le_bytes());
            mem[off + 8..off + 12].copy_from_slice(&rec[2].to_le_bytes());
        };

        let build_heap = || {
            let mut mem = vec![0u8; SLOTS as usize * 32];
            // 10,000 steady-state buffer views (real host SRVs), 20,000 steady-state
            // placed-texture-left-null slots (ready == 0, synced == key), 16 ephemeral
            // slots, remainder empty.
            for i in 0..10_000usize {
                write_rec(&mut mem, i, buf_records[i % 75]);
            }
            for i in 0..20_000usize {
                write_rec(&mut mem, 12_288 + i, placed_records[i % 20]);
            }
            for i in 0..16usize {
                write_rec(&mut mem, 34_816 + i, eph_record);
            }
            mem
        };

        let mut mem_legacy = build_heap();
        let mut mem_fast = build_heap();

        let sync_all_pages = |host: &HostHeap, mem: &[u8], fast: bool| {
            let start = mem.as_ptr() as u64;
            for p in 0..(SLOTS / PAGE_SLOTS) {
                let first = p * PAGE_SLOTS;
                let last = first + PAGE_SLOTS;
                let byte_start = first as usize * 32;
                let byte_end = last as usize * 32;
                host.sync_slots_bytes_with(
                    first..last,
                    start,
                    32,
                    start + byte_start as u64,
                    &mem[byte_start..byte_end],
                    fast,
                );
            }
        };

        // Initial warm sync so steady-state slots (including 20,000 placed-left-null) are populated.
        sync_all_pages(&host_legacy, &mem_legacy, false);
        sync_all_pages(&host_fast, &mem_fast, true);
        assert!(host_fast.kept(0), "benchmark buffer SRVs are real host views");
        assert!(!host_fast.kept(12_288), "placed textures remain null on sync");

        // Mutate ~1,700 slots/pass (740 real buffer views of 75 keys + 960 placed-null/cleared)
        // across 23 passes = ~39,100 stale slots + 1,507,328 total scanned slots.
        let mutate_pass = |mem: &mut [u8], pass: usize| {
            let toggle = pass & 1;
            for j in 0..740usize {
                write_rec(mem, 40_960 + j, buf_records[toggle * 75 + (j % 75)]);
            }
            for j in 0..960usize {
                let rec = if toggle == 0 {
                    placed_records[20 + (j % 20)]
                } else {
                    [0, 0, 0]
                };
                write_rec(mem, 45_056 + j, rec);
            }
        };

        // Two extra resources whose views are forgotten mid-run so each pass
        // exercises a real `VIEWS_FORGOT` epoch bump without affecting live slots.
        let bump_legacy = crate::resource::test_buffer(256);
        let bump_fast = crate::resource::test_buffer(256);
        for &r in &[bump_legacy, bump_fast] {
            let mut scratch = [0u32; 4];
            crate::descriptor::create_srv(
                std::ptr::null(),
                r,
                &srv_desc,
                scratch.as_mut_ptr() as u64,
            );
        }

        // Run legacy across 23 passes (1.5M scanned + ~39k stale) with a mid-run forget() epoch bump.
        let t_legacy = std::time::Instant::now();
        for pass in 0..PASSES {
            mutate_pass(&mut mem_legacy, pass);
            if pass == PASSES / 2 {
                crate::descriptor::forget(bump_legacy as usize);
            }
            sync_all_pages(&host_legacy, &mem_legacy, false);
        }
        let legacy_ms = t_legacy.elapsed().as_secs_f64() * 1000.0;

        // Run fast across identical 23 passes (1.5M scanned + ~39k stale) with identical forget() epoch bump.
        let t_fast = std::time::Instant::now();
        for pass in 0..PASSES {
            mutate_pass(&mut mem_fast, pass);
            if pass == PASSES / 2 {
                crate::descriptor::forget(bump_fast as usize);
            }
            sync_all_pages(&host_fast, &mem_fast, true);
        }
        let fast_ms = t_fast.elapsed().as_secs_f64() * 1000.0;

        // Verify identical final made/kept state across all 65,536 slots.
        for i in 0..SLOTS {
            assert_eq!(host_fast.made(i), host_legacy.made(i), "made mismatch at slot {i}");
            assert_eq!(host_fast.kept(i), host_legacy.kept(i), "kept mismatch at slot {i}");
        }

        // Now actually forget ephemeral and verify both clear slots 34,816..34,832.
        crate::descriptor::forget(ephemeral as usize);
        sync_all_pages(&host_legacy, &mem_legacy, false);
        sync_all_pages(&host_fast, &mem_fast, true);
        for i in 34_816..34_832u32 {
            assert!(!host_legacy.made(i));
            assert!(!host_fast.made(i));
            assert!(!host_fast.kept(i));
        }

        let speedup = legacy_ms / fast_ms.max(0.001);
        eprintln!(
            "bench_sync_slots_1_5m_fast_vs_legacy: slots={} stale=~39100 legacy={:.2}ms fast={:.2}ms speedup={:.2}x",
            SLOTS as usize * PASSES,
            legacy_ms,
            fast_ms,
            speedup
        );

        crate::object::release(bump_legacy);
        crate::object::release(bump_fast);
        crate::object::release(ephemeral);
        for b in buffers {
            crate::object::release(b);
        }
        let _ = tex_mem.as_ptr();
    }

    /// Mismatched descriptor kinds (a sampler record in a CBV/SRV/UAV heap or
    /// a CBV/SRV record in a sampler heap) must be cleared/skipped instead of
    /// calling `CreateSampler` / `CreateShaderResourceView` on the wrong heap
    /// handle type.
    #[test]
    fn mismatched_heap_kind_views_are_cleared_not_created() {
        let Some(host) = host() else {
            return; // no host GPU here
        };
        let Some(view_heap) = HostHeap::new(false, 4) else {
            return;
        };
        let Some(sampler_heap) = HostHeap::new(true, 4) else {
            return;
        };
        let null = std::ptr::null();
        let sampler_desc: [u32; 13] = [21, 3, 3, 3, 0, 1, 0, 0, 0, 0, 0, 0x7f7f_ffff, 0];
        let mut console_sampler = [0u32; 27];
        console_sampler[6..9].copy_from_slice(&[3, 3, 3]);

        // View heap: valid view in slot 0, then overwritten with sampler records.
        let mut view_slots = [[0u32; 8]; 4];
        let v0 = view_slots[0].as_mut_ptr() as u64;
        let v1 = view_slots[1].as_mut_ptr() as u64;
        crate::descriptor::create_cbv(null, &[0, 0, 256, 0], v0);
        assert!(view_heap.mirror(0, v0, false).is_some());
        assert!(view_heap.made(0));
        assert!(view_heap.made_is_nonzero(0));

        crate::descriptor::create_sampler(null, &sampler_desc, v0);
        assert!(view_heap.mirror(0, v0, false).is_none());
        assert!(!view_heap.made(0));
        assert!(!view_heap.made_is_nonzero(0));

        crate::descriptor::create_sampler_x(null, &console_sampler, v1);
        assert!(view_heap.mirror(1, v1, true).is_none());
        assert!(!view_heap.made(1));

        let dyn_slot = view_heap.dynamic(1).expect("a dynamic slot");
        view_heap.make_at(dyn_slot, &View::Sampler(sampler_desc));
        assert!(!view_heap.kept(dyn_slot));
        view_heap.release(view_heap.id(), dyn_slot, 1);

        // Sampler heap: valid sampler in slot 0, then overwritten with CBV/SRV records.
        let mut sampler_slots = [[0u32; 4]; 4];
        let s0 = sampler_slots[0].as_mut_ptr() as u64;
        let s1 = sampler_slots[1].as_mut_ptr() as u64;
        crate::descriptor::create_sampler(null, &sampler_desc, s0);
        assert!(sampler_heap.mirror(0, s0, false).is_some());
        assert!(sampler_heap.made(0));
        assert!(sampler_heap.made_is_nonzero(0));

        crate::descriptor::create_cbv(null, &[0, 0, 256, 0], s0);
        assert!(sampler_heap.mirror(0, s0, false).is_none());
        assert!(!sampler_heap.made(0));
        assert!(!sampler_heap.made_is_nonzero(0));

        crate::descriptor::create_srv(null, null, std::ptr::null(), s1);
        assert!(sampler_heap.mirror(1, s1, true).is_none());
        assert!(!sampler_heap.made(1));

        // SAFETY: plain device query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
    }

    /// The unsigned twin of a 16-bit float or signed format has its layout,
    /// and only buffer views in such a format are shown as another view.
    #[test]
    fn only_16_bit_float_and_signed_buffer_views_get_an_unsigned_twin() {
        for (format, unsigned) in [(10, 12), (14, 12), (34, 36), (38, 36), (54, 57), (59, 57)] {
            assert_eq!(half_format(format), Some(unsigned));
            assert_eq!(
                crate::resource::element(format),
                crate::resource::element(unsigned),
                "format {format}"
            );
        }
        // The unsigned formats themselves, normalised 16-bit formats, and
        // wider and narrower ones.
        for other in [0, 12, 36, 57, 11, 13, 35, 37, 56, 58, 42, 17, 30, 61] {
            assert_eq!(half_format(other), None, "format {other}");
        }
        const BUFFER: u32 = 1;
        let words = |format: u32, dimension: u32| [format, dimension, 5768, 0, 0, 0, 64, 0, 0, 0];
        // R16G16B16A16_FLOAT: a buffer of halves a program moves as they are.
        assert_eq!(
            half_view(&View::Unordered(7, 0, Some(words(10, BUFFER)))),
            Some(View::Unordered(7, 0, Some(words(12, BUFFER))))
        );
        assert_eq!(
            half_view(&View::Shader(7, Some(words(54, BUFFER)))),
            Some(View::Shader(7, Some(words(57, BUFFER))))
        );
        let mut placed = [0u32; 16];
        placed[..10].copy_from_slice(&words(38, BUFFER));
        placed[10] = 0x1234_0000;
        let mut twin = placed;
        twin[0] = 36;
        let memory = [BUFFER, 0, 0, 0, 512, 0, 1, 0x1_0001, 0, 1, 0, 1, 0, 0];
        assert_eq!(
            half_view(&View::Placed(true, memory, placed)),
            Some(View::Placed(true, memory, twin))
        );
        // Unsigned 16-bit buffers, texture views (dimension 4: 2D), views
        // without a description, and constant buffers keep their own view.
        assert_eq!(
            half_view(&View::Unordered(7, 0, Some(words(12, BUFFER)))),
            None
        );
        assert_eq!(half_view(&View::Shader(7, Some(words(57, BUFFER)))), None);
        assert_eq!(half_view(&View::Unordered(7, 0, Some(words(10, 4)))), None);
        assert_eq!(half_view(&View::Shader(7, None)), None);
        assert_eq!(half_view(&View::Constant(0x1000, 256)), None);
    }

    /// Buffer views whose values the console would convert to and from halves
    /// are reported by `half_inexact`; 16-bit formats and narrow unsigned
    /// integer formats are not.
    #[test]
    fn views_whose_values_the_console_would_convert_are_named() {
        const BUFFER: u32 = 1;
        let words = |format: u32, dimension: u32| [format, dimension, 5768, 0, 0, 0, 64, 0, 0, 0];
        for exact in [10, 12, 14, 34, 36, 38, 54, 57, 59, 25, 30, 50, 62] {
            let view = View::Unordered(7, 0, Some(words(exact, BUFFER)));
            assert_eq!(half_inexact(&view), None, "format {exact}");
        }
        for inexact in [41, 42, 11, 28, 64, 0] {
            let view = View::Shader(7, Some(words(inexact, BUFFER)));
            assert_eq!(half_inexact(&view), Some(inexact));
        }
        let mut placed = [0u32; 16];
        placed[..10].copy_from_slice(&words(41, BUFFER));
        let memory = [BUFFER, 0, 0, 0, 512, 0, 1, 0x1_0001, 0, 1, 0, 1, 0, 0];
        assert_eq!(half_inexact(&View::Placed(true, memory, placed)), Some(41));
        // Not a buffer view.
        assert_eq!(half_inexact(&View::Shader(7, Some(words(41, 4)))), None);
        assert_eq!(half_inexact(&View::Constant(0x1000, 256)), None);
    }

    /// Moving all 65,536 16-bit patterns through an `R16G16B16A16_UINT`
    /// buffer view (the unsigned twin `half_view` gives a 16-bit float view)
    /// preserves every pattern bit-exactly when accessed as unsigned integers.
    #[test]
    fn halves_cross_an_unsigned_buffer_view_unchanged() {
        const MOVES: &str = r#"
RWBuffer<uint4> from : register(u0);
RWBuffer<uint4> to : register(u1);
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint4 h = from[id.x] & 0xffff;
    uint low = h.x | h.y << 16;
    uint high = h.z | h.w << 16;
    to[id.x] = uint4(low & 0xffff, low >> 16, high & 0xffff, high >> 16);
}
"#;
        const ELEMENTS: usize = 1 << 14;
        const BYTES: u64 = ELEMENTS as u64 * 8;
        const UNSIGNED: u32 = 12; // R16G16B16A16_UINT
        let Some(host) = host() else {
            return; // no host GPU
        };
        assert_eq!(half_format(10), Some(UNSIGNED));
        let uav_range = [crate::compute::uav(0, 2)];
        let (root, pso) = crate::compute::pipeline(
            MOVES,
            windows::core::s!("half_views_test"),
            0,
            &[crate::compute::Param::Table(&uav_range)],
        )
        .expect("half-views compute pipeline");
        let heap: ID3D12DescriptorHeap = {
            let desc = D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: 2,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                NodeMask: 0,
            };
            // SAFETY: plain heap creation.
            unsafe { host.device.CreateDescriptorHeap(&desc) }.expect("a view heap")
        };
        let (from, filled) = host.shared_buffer(BYTES).expect("a buffer");
        let (to, result) = host.shared_buffer(BYTES).expect("a buffer");
        // SAFETY: both mappings hold `BYTES` bytes; the views are of live
        // buffers in slots of the heap made above; an open list.
        let unsigned = unsafe {
            let filled = std::slice::from_raw_parts_mut(filled.cast::<u16>(), ELEMENTS * 4);
            for (pattern, half) in filled.iter_mut().enumerate() {
                *half = pattern as u16;
            }
            std::ptr::write_bytes(result, 0xee, BYTES as usize);
            let view = D3D12_UNORDERED_ACCESS_VIEW_DESC {
                Format: DXGI_FORMAT(UNSIGNED as i32),
                ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
                Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                    Buffer: D3D12_BUFFER_UAV {
                        NumElements: ELEMENTS as u32,
                        ..Default::default()
                    },
                },
            };
            let start = heap.GetCPUDescriptorHandleForHeapStart();
            let step = host
                .device
                .GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV)
                as usize;
            host.device
                .CreateUnorderedAccessView(&from, None::<&ID3D12Resource>, Some(&view), start);
            let second = D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: start.ptr + step,
            };
            host.device
                .CreateUnorderedAccessView(&to, None::<&ID3D12Resource>, Some(&view), second);
            let list = host.new_list().expect("a host list");
            list.list.SetDescriptorHeaps(&[Some(heap.clone())]);
            list.list.SetComputeRootSignature(&root);
            list.list.SetPipelineState(&pso);
            list.list
                .SetComputeRootDescriptorTable(0, heap.GetGPUDescriptorHandleForHeapStart());
            list.list.Dispatch(ELEMENTS as u32 / 64, 1, 1);
            list.close();
            host.execute(&[&list]);
            assert!(
                host.device.GetDeviceRemovedReason().is_ok(),
                "the device, after views in format {UNSIGNED}"
            );
            let result = std::slice::from_raw_parts(result.cast::<u16>(), ELEMENTS * 4);
            (0..=u16::MAX)
                .filter(|&p| result[usize::from(p)] == p)
                .count()
        };
        assert_eq!(
            unsigned, 65_536,
            "patterns unchanged, moved as unsigned integers"
        );
    }

    /// End-to-end test of [`buffer_srv`] converting a structured buffer SRV
    /// (`StructureByteStride = 680`, starting at `FirstElement = 1` = `+680`
    /// bytes, `8 mod 16`) into a raw SRV and reading it through a translated
    /// DXIL compute shader via MUBUF raw byte offset (`0xC4`), SMEM scalar
    /// byte offset (`0xC4`), and MUBUF structured element+field (`[2] + 12`).
    #[test]
    fn structured_buffer_srv_made_raw_reads_byte_and_element_offsets() {
        use crate::pipeline::{Shader, Translated, host::HostPipeline};
        use victory_shader::native::{ComputeIo, StageIo};
        use victory_shader::{
            Container, HwStage, NativeProgram, ResourceType, RootParam, RootType, UsedResource,
            emit, io, ir, resource, structure,
        };

        let Some(host) = host() else {
            return;
        };
        let mut sm = D3D12_FEATURE_DATA_SHADER_MODEL {
            HighestShaderModel: D3D_SHADER_MODEL_6_6,
        };
        // SAFETY: valid feature support query.
        let ok = unsafe {
            host.device.CheckFeatureSupport(
                D3D12_FEATURE_SHADER_MODEL,
                (&raw mut sm).cast(),
                size_of_val(&sm) as u32,
            )
        };
        if ok.is_err() || sm.HighestShaderModel.0 < D3D_SHADER_MODEL_6_6.0 {
            return;
        }

        let code_words: &[u32] = &[
            0xBE84_0300, // s_mov_b32 s4, s0
            0xBE85_0380, // s_mov_b32 s5, 0
            0xF408_0202,
            0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], 0x0 (SRV slot 0)
            0xBE86_0301, // s_mov_b32 s6, s1
            0xBE87_0380, // s_mov_b32 s7, 0
            0xF408_0303,
            0xFA00_0000, // s_load_dwordx4 s[12:15], s[6:7], 0x0 (UAV slot 1)
            0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
            0xF420_0404,
            0xFA00_00C4, // s_buffer_load_dword s16, s[8:11], 0xC4
            0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
            0x7E08_0210, // v_mov_b32 v4, s16
            0xE030_00C4,
            0x8002_0100, // buffer_load_dword v1, off, s[8:11], 0 offset:196
            0x7E04_0282, // v_mov_b32 v2, 2
            0xE030_200C,
            0x8002_0302, // buffer_load_dword v3, v2, s[8:11], 0 idxen offset:12
            0xBF8C_3F70, // s_waitcnt vmcnt(0)
            0xE070_0000,
            0x8003_0100, // buffer_store_dword v1, off, s[12:15], 0 offset:0
            0xE070_0004,
            0x8003_0400, // buffer_store_dword v4, off, s[12:15], 0 offset:4
            0xE070_0008,
            0x8003_0300, // buffer_store_dword v3, off, s[12:15], 0 offset:8
            0xBF81_0000, // s_endpgm
        ];
        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64: false,
            vgprs: 8,
            layout: (0..2)
                .map(|slot| RootParam {
                    root: RootType::Table,
                    table_offset_dwords: 0,
                    sgprs: 1,
                    base_register: 0,
                    user_data: slot,
                    slot,
                })
                .collect(),
            used: vec![
                UsedResource {
                    kind: ResourceType::Srv,
                    root: RootType::Table,
                    table_offset_dwords: 0,
                    lower_bound: 0,
                    stride: 680,
                    dwords: 4,
                    user_data: 0,
                    slot: 0,
                },
                UsedResource {
                    kind: ResourceType::Uav,
                    root: RootType::Table,
                    table_offset_dwords: 0,
                    lower_bound: 0,
                    stride: 0,
                    dwords: 4,
                    user_data: 1,
                    slot: 1,
                },
            ],
            io: StageIo::Cs(ComputeIo {
                rsrc2: 2 << 1,
                threads: [1, 1, 1],
            }),
            code: code_words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let mut container = vec![0; 32];
        container[..4].copy_from_slice(b"DXBC");
        container[20..24].copy_from_slice(&1u32.to_le_bytes());
        container[24..28].copy_from_slice(&32u32.to_le_bytes());
        let c = Container::parse(&container).unwrap();
        let p = ir::build(&native.code, native.wave64, native.vgprs).unwrap();
        let st = structure::structure(&p).unwrap();
        let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        let io = io::raise(&p, &native).unwrap();
        let hs = emit::emit(&c, &native, &p, &st, &io, &res).unwrap();
        let ud_spec = hs.user_data.clone();
        let shader = Shader::for_test(
            "compute",
            Translated {
                host: hs,
                used: native.used,
                used_written: Vec::new(),
                write_unknown: true,
            },
        );
        let Some(hp) = HostPipeline::new(&[shader], None, |_, _| {}) else {
            panic!("synthetic structured-SRV pipeline");
        };

        // Allocate 8 KiB of title memory (2048 dwords) filled with distinct words.
        let bytes_len = 8192usize;
        let mut backing: Vec<u32> = (0..(bytes_len / 4) as u32)
            .map(|i| 0x2000_0000 + i)
            .collect();
        let mem = backing.as_mut_ptr();
        crate::shadow::register(mem as usize, mem as u64, bytes_len as u64);

        // SAFETY: shader-visible heap with 2 slots.
        let d3d_heap: ID3D12DescriptorHeap = unsafe {
            host.device
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                    NumDescriptors: 2,
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                    ..Default::default()
                })
                .unwrap()
        };
        // SAFETY: live descriptor heap.
        let cpu = unsafe { d3d_heap.GetCPUDescriptorHandleForHeapStart() };
        // SAFETY: plain device query.
        let step = unsafe {
            host.device
                .GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV)
        } as usize;

        // Structured SRV: stride = 680, FirstElement = 1 (+680B = +170 dwords), NumElements = 4.
        let srv_desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_UNKNOWN,
            ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_SRV {
                    FirstElement: 1,
                    NumElements: 4,
                    StructureByteStride: 680,
                    Flags: D3D12_BUFFER_SRV_FLAG_NONE,
                },
            },
        };
        let _kept = buffer_srv(mem as u64, bytes_len as u64, srv_desc, cpu, None)
            .expect("buffer_srv succeeds")
            .expect("holds shadow buffer");

        let list = host.new_list().unwrap();
        let shadow = crate::shadow::find(mem as u64, bytes_len as u64).expect("registered shadow");
        shadow.refresh(&list, mem as u64, bytes_len as u64);

        let mut uav_buf: Option<ID3D12Resource> = None;
        // SAFETY: valid default UAV buffer description.
        unsafe {
            host.device
                .CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES {
                        Type: D3D12_HEAP_TYPE_DEFAULT,
                        ..Default::default()
                    },
                    D3D12_HEAP_FLAG_NONE,
                    &D3D12_RESOURCE_DESC {
                        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                        Width: 256,
                        Height: 1,
                        DepthOrArraySize: 1,
                        MipLevels: 1,
                        SampleDesc: DXGI_SAMPLE_DESC {
                            Count: 1,
                            Quality: 0,
                        },
                        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                        Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                        ..Default::default()
                    },
                    D3D12_RESOURCE_STATE_COMMON,
                    None,
                    &mut uav_buf,
                )
                .unwrap();
        }
        let uav_buf = uav_buf.unwrap();
        let uav_desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R32_TYPELESS,
            ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_UAV {
                    NumElements: 8,
                    Flags: D3D12_BUFFER_UAV_FLAG_RAW,
                    ..Default::default()
                },
            },
        };
        // SAFETY: valid UAV slot 1.
        unsafe {
            host.device.CreateUnorderedAccessView(
                &uav_buf,
                None,
                Some(&uav_desc),
                D3D12_CPU_DESCRIPTOR_HANDLE {
                    ptr: cpu.ptr + step,
                },
            );
        }

        let mut ud = vec![0u32; hp.stages[0].user_data.map_or(1, |(_, n)| n as usize)];
        for u in &ud_spec {
            ud[u.index as usize] = 32 * u32::from(u.slot);
        }
        let (readback, rb_ptr) = list.readback(host, 256).unwrap();
        // SAFETY: open list with heap, root signature, constants, and PSO bound.
        unsafe {
            list.list.SetDescriptorHeaps(&[Some(d3d_heap.clone())]);
            list.list.SetComputeRootSignature(&hp.root);
            list.list
                .SetComputeRoot32BitConstants(0, ud.len() as u32, ud.as_ptr().cast(), 0);
            list.list.SetPipelineState(&hp.pso);
            list.list.Dispatch(1, 1, 1);
            list.list.ResourceBarrier(&[crate::host::transition(
                &uav_buf,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            list.list.CopyResource(&readback, &uav_buf);
        }
        list.close();
        host.execute(&[&list]);

        // SAFETY: GPU execution finished; read 3 dwords from mapped readback.
        let got = unsafe { *rb_ptr.cast::<[u32; 3]>() };
        // SAFETY: plain device query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
        crate::shadow::unregister(mem as usize, mem as u64);
        drop(backing);

        let expected = [
            0x2000_0000 + 170 + 49,  // MUBUF raw @ 0xC4
            0x2000_0000 + 170 + 49,  // SMEM scalar raw @ 0xC4
            0x2000_0000 + 170 + 343, // MUBUF structured [2] + 12
        ];
        assert_eq!(got, expected);
    }
}
