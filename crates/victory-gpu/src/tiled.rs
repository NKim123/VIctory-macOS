//! Stable GPU addresses for XMem placeholder buffers. The Windows view
//! and the reserved GPU buffer both refer to the physical allocation.

use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, Weak};

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;

use crate::alias::{PhysicalHeap, physical_tiles_enabled};
use crate::host::host;
use crate::ranges::Ranges;
use crate::shadow::{self, Shadow};
use crate::watch::PAGE;

// Computed shader addresses use a 32-bit byte offset into the raw view.
// Keep large CPU reservations in chunks below that limit.
pub const MAX_TILED_BUFFER: usize = 2 << 30;
const MAX_BYTES: usize = 64 << 30;
const MAX_BUFFERS: usize = 128;

static BUFFERS: Mutex<Vec<Weak<TiledBuffer>>> = Mutex::new(Vec::new());

/// Orders XMem's CPU view changes and GPU tile edits against preparation,
/// execution and copy-back. A caller must acquire its XMem lock inside this
/// boundary. The disabled path takes no GPU lock.
pub fn with_physical_mapping<T>(edit: impl FnOnce() -> T) -> T {
    let _quiet = if physical_tiles_enabled() {
        host().and_then(|h| h.quiet())
    } else {
        None
    };
    edit()
}

/// Owns an alias registration for the lifetime of a title reservation.
pub struct TiledAlias {
    buffer: Arc<TiledBuffer>,
    registered: bool,
}

pub(crate) struct TiledBuffer {
    resource: ID3D12Resource,
    base: usize,
    len: usize,
    mappings: Mutex<Mappings>,
    // Once a VA can be held by a recorded binding, it cannot fall back.
    bound: AtomicBool,
    partial_release: AtomicBool,
    released: Mutex<Ranges>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct Tile {
    heap: NonZeroUsize,
    offset: u32,
}

impl Tile {
    fn follows(&self, prev: &Self) -> bool {
        self.heap == prev.heap && self.offset == prev.offset + 1
    }
}

struct Mappings {
    tiles: Vec<Option<Tile>>,
    dirty: Ranges,
    live: HashMap<NonZeroUsize, (usize, Arc<PhysicalHeap>)>,
    // The queue can still refer to previously mapped heaps until the
    // fence following UpdateTileMappings. Keep one reference per heap.
    applied: Vec<Arc<PhysicalHeap>>,
}

impl TiledAlias {
    /// Publishes a reserved UAV buffer inside the physical mapping boundary.
    pub fn new(base: usize, len: usize) -> Option<Self> {
        let end = base.checked_add(len)? as u64;
        let _binding = shadow::physical_binding()?;
        if base == 0
            || len == 0
            || len > MAX_TILED_BUFFER
            || !(base | len).is_multiple_of(PAGE)
            || shadow::has_aliased_overlap(base as u64, end)
            || shadow::has_materialized_overlap(base as u64, end)
        {
            return None;
        }
        let mut buffers = BUFFERS.lock().unwrap_or_else(|e| e.into_inner());
        let mut used = 0;
        buffers.retain(|weak| {
            if let Some(buffer) = weak.upgrade() {
                used += buffer.len;
                true
            } else {
                false
            }
        });
        if buffers.len() >= MAX_BUFFERS || used + len > MAX_BYTES {
            return None;
        }
        let host = host()?;
        let mut resource = None;
        // SAFETY: a row-major buffer of a whole number of buffer tiles.
        let result = unsafe {
            host.device.CreateReservedResource(
                &D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                    Width: len as u64,
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
                &mut resource,
            )
        };
        if let Err(error) = result {
            tracing::warn!(base, len, %error, "reserved buffer unavailable; using shadows");
            return None;
        }
        let resource: ID3D12Resource = resource?;
        // SAFETY: a live buffer resource.
        let gpu = unsafe { resource.GetGPUVirtualAddress() };
        if gpu == 0 {
            return None;
        }
        crate::host::name(&resource, || {
            format!("tiled title buffer {base:#x}+{len:#x}")
        });
        let mut dirty = Ranges::default();
        dirty.insert(0, len / PAGE, 0);
        let tiled = Arc::new(TiledBuffer {
            resource: resource.clone(),
            base,
            len,
            bound: AtomicBool::new(false),
            partial_release: AtomicBool::new(false),
            released: Mutex::default(),
            mappings: Mutex::new(Mappings {
                tiles: vec![None; len / PAGE],
                dirty,
                live: HashMap::new(),
                applied: Vec::new(),
            }),
        });
        buffers.push(Arc::downgrade(&tiled));
        shadow::register_aliased(Shadow::new_tiled(resource, gpu, base, len, tiled.clone()));
        tracing::debug!(base, len, "reserved title buffer aliased zero-copy");
        Some(Self {
            buffer: tiled,
            registered: true,
        })
    }

    /// Removes an alias before any descriptor or list has retained it.
    pub fn demote_if_unused(&mut self) -> bool {
        if shadow::unregister_unused_aliased(self.buffer.base as u64) {
            self.registered = false;
            return true;
        }
        false
    }

    /// Whether this address still belongs to the original reservation.
    pub fn owns(&self, at: usize) -> bool {
        self.buffer.covers(at, at + 1)
    }

    /// A split placeholder was individually released. Other pages keep
    /// their live bindings, and reuse of the hole uses ordinary shadows.
    pub fn release(&self, from: usize, to: usize) {
        self.map(from, to - from, None);
        self.buffer
            .released
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(from, to, 0);
        self.buffer.partial_release.store(true, Ordering::Release);
        shadow::alias_coverage_changed();
    }

    /// Replaces the final mapping of an aligned range, or makes it NULL.
    /// All edits from one CPU mapping operation precede the next submission.
    pub fn map(&self, base: usize, len: usize, source: Option<(&Arc<PhysicalHeap>, usize)>) {
        if let Some((heap, offset)) = source
            && self.buffer.partial_release.load(Ordering::Acquire)
        {
            let released = self
                .buffer
                .released
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            for (from, to) in released.missing(base, base + len) {
                self.map_live(from, to - from, Some((heap, offset + from - base)));
            }
        } else {
            self.map_live(base, len, source);
        }
    }

    fn map_live(&self, base: usize, len: usize, source: Option<(&Arc<PhysicalHeap>, usize)>) {
        let first = (base - self.buffer.base) / PAGE;
        let count = len / PAGE;
        debug_assert!((base | len).is_multiple_of(PAGE));
        let mut state = self
            .buffer
            .mappings
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let source = source.map(|(heap, offset)| {
            assert!(
                offset.is_multiple_of(PAGE)
                    && offset.checked_add(len).is_some_and(|end| end <= heap.len)
            );
            (
                NonZeroUsize::new(Arc::as_ptr(heap) as usize).unwrap(),
                heap,
                offset / PAGE,
            )
        });
        let Mappings {
            tiles, live, dirty, ..
        } = &mut *state;
        let mut changed = false;
        for (i, tile) in tiles[first..first + count].iter_mut().enumerate() {
            let next = source.map(|(heap, _, offset)| Tile {
                heap,
                offset: (offset + i) as u32,
            });
            if *tile == next {
                continue;
            }
            if let Some(prev) = *tile {
                let (count, _) = live.get_mut(&prev.heap).unwrap();
                *count -= 1;
                if *count == 0 {
                    live.remove(&prev.heap);
                }
            }
            if let Some((key, heap, _)) = source {
                live.entry(key).or_insert_with(|| (0, heap.clone())).0 += 1;
            }
            *tile = next;
            changed = true;
        }
        if changed {
            dirty.insert(first, first + count, 0);
        }
    }
}

impl Drop for TiledAlias {
    fn drop(&mut self) {
        if self.registered {
            shadow::unregister_aliased(self.buffer.base as u64);
        }
        // Lists and descriptors can still hold the original resource. Its
        // tiles must no longer give access to the released title mapping.
        self.map(self.buffer.base, self.buffer.len, None);
    }
}

struct Batch<'a> {
    heap: Option<&'a PhysicalHeap>,
    starts: Vec<D3D12_TILED_RESOURCE_COORDINATE>,
    sizes: Vec<D3D12_TILE_REGION_SIZE>,
    offsets: Vec<u32>,
    counts: Vec<u32>,
}

fn tile_runs<'a>(
    tiles: &'a [Option<Tile>],
    dirty: &Ranges,
) -> impl Iterator<Item = (usize, usize, Option<&'a Tile>)> {
    dirty.iter().flat_map(move |(mut first, end)| {
        std::iter::from_fn(move || {
            if first == end {
                return None;
            }
            let start = first;
            first += 1;
            while first < end
                && match (&tiles[first], &tiles[first - 1]) {
                    (Some(next), Some(prev)) => next.follows(prev),
                    (None, None) => true,
                    _ => false,
                }
            {
                first += 1;
            }
            Some((start, first - start, tiles[start].as_ref()))
        })
    })
}

impl TiledBuffer {
    pub(crate) fn covers(&self, from: usize, to: usize) -> bool {
        !self.partial_release.load(Ordering::Acquire)
            || self
                .released
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .missing(from, to)
                .next()
                == Some((from, to))
    }

    pub(crate) fn bind(&self) {
        self.bound.store(true, Ordering::Relaxed);
    }

    pub(crate) fn bound(&self) -> bool {
        self.bound.load(Ordering::Relaxed)
    }

    /// CPU-side inspection for views that have to convert their format.
    /// Reads through the heap's own view, including zeroes for holes.
    pub(crate) fn read(&self, mut offset: usize, mut dst: &mut [u8]) {
        let state = self.mappings.lock().unwrap_or_else(|e| e.into_inner());
        while !dst.is_empty() {
            let within = offset % PAGE;
            let len = (PAGE - within).min(dst.len());
            let (out, rest) = dst.split_at_mut(len);
            if let Some(tile) = &state.tiles[offset / PAGE] {
                let (_, heap) = &state.live[&tile.heap];
                let at = heap.cpu_address() + tile.offset as usize * PAGE + within;
                // SAFETY: the tile owns the backing and this range is within it.
                unsafe { std::ptr::copy_nonoverlapping(at as *const u8, out.as_mut_ptr(), len) };
            } else {
                out.fill(0);
            }
            offset += len;
            dst = rest;
        }
    }

    fn flush(&self, queue: &ID3D12CommandQueue, retired: &mut Vec<Arc<PhysicalHeap>>) -> bool {
        let mut state = self.mappings.lock().unwrap_or_else(|e| e.into_inner());
        let dirty = std::mem::take(&mut state.dirty);
        if dirty == Ranges::default() {
            return false;
        }
        let mut batches = HashMap::new();
        for (first, count, tile) in tile_runs(&state.tiles, &dirty) {
            let key = tile.map(|t| t.heap);
            let batch = batches.entry(key).or_insert_with(|| Batch {
                heap: key.map(|key| state.live[&key].1.as_ref()),
                starts: Vec::new(),
                sizes: Vec::new(),
                offsets: Vec::new(),
                counts: Vec::new(),
            });
            let count = count as u32;
            batch.starts.push(D3D12_TILED_RESOURCE_COORDINATE {
                X: first as u32,
                ..Default::default()
            });
            batch.sizes.push(D3D12_TILE_REGION_SIZE {
                NumTiles: count,
                ..Default::default()
            });
            batch.offsets.push(tile.map_or(0, |t| t.offset));
            batch.counts.push(count);
        }
        for batch in batches.values() {
            let n = batch.starts.len();
            let heap = batch.heap.map(|h| &h.heap);
            let flags = heap.is_none().then(|| vec![D3D12_TILE_RANGE_FLAG_NULL; n]);
            // SAFETY: each source and destination range is within a live
            // 64 KiB-aligned allocation. All arrays have n entries.
            unsafe {
                queue.UpdateTileMappings(
                    &self.resource,
                    n as u32,
                    Some(batch.starts.as_ptr()),
                    Some(batch.sizes.as_ptr()),
                    heap,
                    n as u32,
                    flags.as_ref().map(|f| f.as_ptr()),
                    heap.map(|_| batch.offsets.as_ptr()),
                    Some(batch.counts.as_ptr()),
                    D3D12_TILE_MAPPING_FLAG_NONE,
                );
            }
        }
        let live = state.live.values().map(|(_, heap)| heap.clone()).collect();
        retired.extend(std::mem::replace(&mut state.applied, live));
        true
    }
}

/// Resources and old heap mappings held through the submission fence.
#[must_use]
pub(crate) struct Flushed {
    buffers: Vec<Arc<TiledBuffer>>,
    retired: Vec<Arc<PhysicalHeap>>,
}

pub(crate) fn flush(queue: &ID3D12CommandQueue) -> Flushed {
    let mut flushed = Flushed {
        buffers: Vec::new(),
        retired: Vec::new(),
    };
    if physical_tiles_enabled() {
        let mut buffers = BUFFERS.lock().unwrap_or_else(|e| e.into_inner());
        buffers.retain(|weak| {
            let Some(buffer) = weak.upgrade() else {
                return false;
            };
            if buffer.flush(queue, &mut flushed.retired) {
                flushed.buffers.push(buffer);
            }
            true
        });
    }
    flushed
}

/// A replacement queue replays the desired mappings before running work.
pub(crate) fn replay() {
    if physical_tiles_enabled() {
        for buffer in BUFFERS
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .filter_map(Weak::upgrade)
        {
            buffer
                .mappings
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .dirty
                .insert(0, buffer.len / PAGE, 0);
        }
    }
}

/// Exercises native CBV, SRV, and UAV bindings through the real submission
/// queue for the XMem hardware integration test.
#[cfg(feature = "diagnostics")]
#[doc(hidden)]
pub fn probe_tiled_alias(base: usize, replace_queue: bool, write: bool) -> (u64, [u32; 2]) {
    use crate::compute::{self, Param};
    let host = host().expect("D3D12 device");
    let shadow = shadow::find(base as u64, PAGE as u64).expect("buffer for the title address");
    assert!(shadow.is_tiled());
    assert!(!shadow::has_materialized_overlap(
        base as u64,
        (base + PAGE) as u64
    ));
    assert!(!shadow.tracks_page(base));
    assert_eq!(shadow.update(base, PAGE), 0);
    let tracker = write
        .then(|| crate::watch::track(base, PAGE, true))
        .flatten();
    if write && crate::watch::fault_tracking() {
        assert!(
            tracker.is_some(),
            "texture-style tracker covers the XMem page"
        );
    }
    if let Some(track) = &tracker {
        track.take(base, base + PAGE);
        assert!(!track.changed(base / PAGE));
    }
    let address = shadow.gpu(base as u64);
    if replace_queue {
        host.replace_queue_for_probe();
    }

    let (result, bytes) = host
        .shared_buffer(PAGE as u64)
        .expect("a mapped result buffer");
    let (root, pipeline) = compute::pipeline(
        r"
cbuffer Input : register(b0) { uint cb; };
cbuffer Control : register(b1) { uint writeWord; };
StructuredBuffer<uint> input : register(t0);
RWByteAddressBuffer output : register(u0);
RWByteAddressBuffer title : register(u1);
[numthreads(1, 1, 1)]
void main() {
    output.Store(0, cb);
    output.Store(4, input[0]);
    if (writeWord != 0) {
        title.Store(4, cb ^ 0x12345678u);
    }
}
",
        windows::core::s!("XMem tiled alias probe"),
        0,
        &[
            Param::Cbv { register: 0 },
            Param::Constants {
                register: 1,
                count: 1,
            },
            Param::Srv {
                register: 0,
                space: 0,
            },
            Param::Uav { register: 0 },
            Param::Uav { register: 1 },
        ],
    )
    .expect("a tiled buffer test shader");
    let list = host.new_list().expect("a host list");
    list.defer_refresh(&shadow, base, PAGE);
    if write {
        shadow.gpu_writes(&list, base as u64 + 4, 4, false);
    }
    // SAFETY: the aligned root addresses refer to live buffers. The
    // dispatch only reads the first word and may write the second.
    unsafe {
        bytes.cast::<u64>().write(u64::MAX);
        list.list.SetPipelineState(&pipeline);
        list.list.SetComputeRootSignature(&root);
        list.list.SetComputeRootConstantBufferView(0, address);
        list.list
            .SetComputeRoot32BitConstant(1, u32::from(write), 0);
        list.list.SetComputeRootShaderResourceView(2, address);
        list.list
            .SetComputeRootUnorderedAccessView(3, result.GetGPUVirtualAddress());
        list.list.SetComputeRootUnorderedAccessView(4, address);
        list.list.Dispatch(1, 1, 1);
    }
    list.close();
    host.execute(&[&list]);
    if let Some(track) = &tracker {
        assert!(
            track.changed(base / PAGE),
            "texture-style tracker saw the GPU write"
        );
    }
    // SAFETY: the synchronous submission completed the writes.
    let words = unsafe { bytes.cast::<[u32; 2]>().read() };
    (address, words)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tile_runs_merge_adjacent_offsets_and_keep_holes_and_heap_boundaries() {
        let tile = |heap, offset| {
            Some(Tile {
                heap: NonZeroUsize::new(heap).unwrap(),
                offset,
            })
        };
        let tiles = [
            None,
            tile(1, 3),
            tile(1, 4),
            tile(1, 8),
            tile(2, 9),
            None,
            None,
            tile(2, 10),
            tile(2, 11),
            None,
        ];
        let mut dirty = Ranges::default();
        dirty.insert(1, 7, 0);
        dirty.insert(8, 10, 0);
        let runs = tile_runs(&tiles, &dirty)
            .map(|(at, n, tile)| (at, n, tile.map(|t| (t.heap.get(), t.offset))))
            .collect::<Vec<_>>();
        assert_eq!(
            runs,
            [
                (1, 2, Some((1, 3))),
                (3, 1, Some((1, 8))),
                (4, 1, Some((2, 9))),
                (5, 2, None),
                (8, 1, Some((2, 11))),
                (9, 1, None),
            ]
        );
    }
}
