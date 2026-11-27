//! Physical backing for suffixes of reserved 2D mip chains.
//!
//! Adapted from DexNom/VIctory's tile pool and packed-mip grouping (GPL-3.0).
//! Each backing exposes one mip suffix for its lifetime. A finer SRV can
//! get a wider backing without changing descriptors already recorded.

use std::collections::BTreeMap;
use std::mem::ManuallyDrop;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

const TILE: u64 = 64 << 10;
const HEAP: u64 = 8 << 20;
const BUDGET: u64 = 256 << 20;

pub(crate) fn enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(
        || match std::env::var("VICTORY_LEVELS_ON_DEMAND").as_deref() {
            Ok("1") => true,
            Ok("0") => false,
            _ => crate::watch::rosetta(),
        },
    )
}

fn tier_two(device: &ID3D12Device) -> bool {
    static TIER_TWO: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *TIER_TWO.get_or_init(|| {
        let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS::default();
        // SAFETY: the option structure matches the requested feature.
        let result = unsafe {
            device.CheckFeatureSupport(
                D3D12_FEATURE_D3D12_OPTIONS,
                (&raw mut options).cast(),
                size_of_val(&options) as u32,
            )
        };
        result.is_ok() && options.TiledResourcesTier.0 >= D3D12_TILED_RESOURCES_TIER_2.0
    })
}

struct Heap {
    resource: ID3D12Heap,
    size: u64,
    free: BTreeMap<u64, u64>,
}

#[derive(Default)]
struct Pool {
    heaps: Vec<Option<Heap>>,
    pending: Vec<(ID3D12Resource, Run)>,
    bytes: u64,
}

struct Run {
    heap: usize,
    offset: u64,
    bytes: u64,
}

static POOL: Mutex<Pool> = Mutex::new(Pool {
    heaps: Vec::new(),
    pending: Vec::new(),
    bytes: 0,
});

impl Pool {
    fn allocate(&mut self, device: &ID3D12Device, bytes: u64) -> Option<Run> {
        let mut i = 0;
        while i < self.pending.len() {
            if crate::placement::last_holder(&self.pending[i].0) {
                let (resource, run) = self.pending.swap_remove(i);
                // Destroy its mapping before reusing the physical tiles.
                drop(resource);
                let heap = self.heaps[run.heap].as_mut().unwrap();
                crate::placement::give(&mut heap.free, run.offset, run.bytes);
                if heap.free.get(&0) == Some(&heap.size) {
                    self.bytes -= heap.size;
                    self.heaps[run.heap] = None;
                }
            } else {
                i += 1;
            }
        }
        for (slot, heap) in self.heaps.iter_mut().enumerate() {
            if let Some(heap) = heap
                && let Some(offset) = crate::placement::take(&mut heap.free, bytes, TILE)
            {
                return Some(Run {
                    heap: slot,
                    offset,
                    bytes,
                });
            }
        }
        let available = BUDGET - self.bytes;
        if bytes > available {
            return None;
        }
        let size = HEAP.max(bytes).min(available);
        let desc = D3D12_HEAP_DESC {
            SizeInBytes: size,
            Properties: D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_DEFAULT,
                ..Default::default()
            },
            Flags: D3D12_HEAP_FLAG_ALLOW_ONLY_NON_RT_DS_TEXTURES,
            ..Default::default()
        };
        let mut resource = None;
        // SAFETY: a whole number of tiles and a typed out pointer.
        unsafe { device.CreateHeap(&desc, &mut resource) }.ok()?;
        let heap = Heap {
            resource: resource?,
            size,
            free: (bytes < size)
                .then_some((bytes, size - bytes))
                .into_iter()
                .collect(),
        };
        let slot = self
            .heaps
            .iter()
            .position(Option::is_none)
            .unwrap_or(self.heaps.len());
        if slot == self.heaps.len() {
            self.heaps.push(Some(heap));
        } else {
            self.heaps[slot] = Some(heap);
        }
        self.bytes += size;
        Some(Run {
            heap: slot,
            offset: 0,
            bytes,
        })
    }
}

pub(crate) struct Tiled {
    pub(crate) first_mip: u32,
    run: Run,
    coords: Vec<D3D12_TILED_RESOURCE_COORDINATE>,
    sizes: Vec<D3D12_TILE_REGION_SIZE>,
    mapped: AtomicBool,
}

impl Tiled {
    pub(crate) fn create(
        device: &ID3D12Device,
        mut desc: D3D12_RESOURCE_DESC,
        first_mip: u32,
    ) -> Option<(ID3D12Resource, Self)> {
        if first_mip == 0
            || first_mip >= u32::from(desc.MipLevels)
            || desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D
            || desc.DepthOrArraySize != 1
            || desc.SampleDesc.Count != 1
            || desc.Flags != D3D12_RESOURCE_FLAG_NONE
            || !matches!(
                desc.Format,
                DXGI_FORMAT_BC1_UNORM | DXGI_FORMAT_BC7_UNORM | DXGI_FORMAT_R8G8B8A8_UNORM
            )
            || !enabled()
            || !tier_two(device)
        {
            return None;
        }
        desc.Layout = D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE;
        let mut texture = None;
        // SAFETY: standard single-layer reserved texture and typed output.
        unsafe {
            device.CreateReservedResource(&desc, D3D12_RESOURCE_STATE_COMMON, None, &mut texture)
        }
        .ok()?;
        let texture = texture?;
        let (mut total, mut packed) = (0, D3D12_PACKED_MIP_INFO::default());
        let mips = u32::from(desc.MipLevels);
        let mut count = mips;
        let mut tilings = vec![D3D12_SUBRESOURCE_TILING::default(); mips as usize];
        // SAFETY: tilings has room for count entries.
        unsafe {
            device.GetResourceTiling(
                &texture,
                Some(&mut total),
                Some(&mut packed),
                None,
                Some(&mut count),
                0,
                tilings.as_mut_ptr(),
            );
        }
        let standard = u32::from(packed.NumStandardMips);
        let tail = u32::from(packed.NumPackedMips);
        if count < standard || standard + tail != mips {
            return None;
        }
        let mut coords = Vec::new();
        let mut sizes = Vec::new();
        let mut tiles = 0u32;
        let mut group = |mip, num_tiles| {
            if num_tiles == 0 {
                return None;
            }
            tiles = tiles.checked_add(num_tiles)?;
            coords.push(D3D12_TILED_RESOURCE_COORDINATE {
                Subresource: mip,
                ..Default::default()
            });
            sizes.push(D3D12_TILE_REGION_SIZE {
                NumTiles: num_tiles,
                ..Default::default()
            });
            Some(())
        };
        for mip in first_mip..standard {
            let d = tilings[mip as usize];
            group(
                mip,
                d.WidthInTiles
                    .checked_mul(u32::from(d.HeightInTiles))?
                    .checked_mul(u32::from(d.DepthInTiles))?,
            )?;
        }
        if tail != 0 {
            group(standard, packed.NumTilesForPackedMips)?;
        }
        if tiles == 0 || tiles >= total {
            return None;
        }
        let run = POOL
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .allocate(device, u64::from(tiles) * TILE)?;
        Some((
            texture,
            Self {
                first_mip,
                run,
                coords,
                sizes,
                mapped: AtomicBool::new(false),
            },
        ))
    }

    pub(crate) fn bytes(&self) -> u64 {
        self.run.bytes
    }

    /// Map all groups of the exposed suffix on the queue that uploads it.
    pub(crate) fn prepare_upload(
        &self,
        list: &ID3D12GraphicsCommandList,
        texture: &ID3D12Resource,
    ) {
        self.map(texture);
        // Recording can happen ahead of submission. Activate in each list
        // that writes these tiles so a discarded list cannot consume it.
        activate(list, texture);
    }

    fn map(&self, texture: &ID3D12Resource) {
        if self.mapped.load(Ordering::Acquire) {
            return;
        }
        let host = crate::host::host().expect("the device that made the texture");
        // Serialize UpdateTileMappings with submissions on the direct queue
        // before taking POOL (matches Host::execute's lock order).
        let _queue = host.queue_lock();
        let pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
        if self.mapped.load(Ordering::Acquire) {
            return;
        }
        let heap = &pool.heaps[self.run.heap].as_ref().unwrap().resource;
        let offset = (self.run.offset / TILE) as u32;
        let count = (self.run.bytes / TILE) as u32;
        // SAFETY: each region came from GetResourceTiling; their tile counts
        // sum to the one contiguous physical range reserved in this heap.
        unsafe {
            host.queue().UpdateTileMappings(
                texture,
                self.coords.len() as u32,
                Some(self.coords.as_ptr()),
                Some(self.sizes.as_ptr()),
                heap,
                1,
                Some(&D3D12_TILE_RANGE_FLAG_NONE),
                Some(&offset),
                Some(&count),
                D3D12_TILE_MAPPING_FLAG_NONE,
            );
        }
        self.mapped.store(true, Ordering::Release);
    }

    pub(crate) fn retire(self, texture: ID3D12Resource) {
        // The resource is also held by descriptors, command lists and the
        // host's retirement queue. Recycle only after those releases.
        POOL.lock()
            .unwrap_or_else(|e| e.into_inner())
            .pending
            .push((texture, self.run));
    }
}

pub(crate) fn activate(list: &ID3D12GraphicsCommandList, texture: &ID3D12Resource) {
    let barrier = D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_ALIASING,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Aliasing: ManuallyDrop::new(D3D12_RESOURCE_ALIASING_BARRIER {
                pResourceBefore: ManuallyDrop::new(None),
                // SAFETY: a borrowed pointer the barrier never releases.
                pResourceAfter: unsafe { std::mem::transmute_copy(texture) },
            }),
        },
        ..Default::default()
    };
    // SAFETY: a live mapped texture and a recording command list.
    unsafe { list.ResourceBarrier(&[barrier]) };
}
