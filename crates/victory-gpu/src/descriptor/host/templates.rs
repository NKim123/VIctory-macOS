//! CPU-only templates for shared title slots. A descriptor record's three
//! header words fully identify its view. The remaining slot bytes are never
//! interpreted. The buffer epoch also covers address-only records and aliases.

use super::*;

pub(super) fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_DESCRIPTOR_CACHE").is_none_or(|v| v != "0"))
}

fn kind(sampler: bool) -> D3D12_DESCRIPTOR_HEAP_TYPE {
    if sampler {
        D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER
    } else {
        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV
    }
}

pub(super) fn copy(
    sampler: bool,
    dst: D3D12_CPU_DESCRIPTOR_HANDLE,
    src: D3D12_CPU_DESCRIPTOR_HANDLE,
) {
    // SAFETY: the CPU-only source and destination belong to live heaps of
    // the same kind. The title heap's mirror lock serializes both slots.
    unsafe {
        host()
            .unwrap()
            .device
            .CopyDescriptorsSimple(1, dst, src, kind(sampler))
    };
}

pub(super) struct Templates {
    _heap: ID3D12DescriptorHeap,
    cpu: usize,
    step: usize,
    sampler: bool,
    entries: Box<[Option<Entry>]>,
}

struct Entry {
    key: u64,
    epoch: u64,
    mark: Option<(u32, u32)>,
    buffer: Option<Buffer>,
}

/// Weak leases allow idle template entries to relinquish arena slices and
/// GPU resources. A copy pins its exact backing before reading the CPU heap.
struct Buffer {
    shadow: Weak<crate::shadow::Shadow>,
    widened: Option<Weak<crate::shadow::Widened>>,
    address: u64,
    len: u64,
}

pub(super) struct Hit {
    pub(super) at: D3D12_CPU_DESCRIPTOR_HANDLE,
    pub(super) held: Kept,
    pub(super) mark: Option<(u32, u32)>,
    pub(super) backing: Option<BufferView>,
}

pub(super) const RESOURCE_BUCKETS: usize = 16_384;

impl Templates {
    pub(super) fn new(sampler: bool) -> Option<Self> {
        let device = &host()?.device;
        let kind = kind(sampler);
        let count = if sampler { 256 } else { RESOURCE_BUCKETS as u32 };
        // SAFETY: a CPU-only heap of a supported descriptor type.
        let (heap, cpu, step) = unsafe {
            let heap: ID3D12DescriptorHeap = device
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: kind,
                    NumDescriptors: count + 1,
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
            sampler,
            entries: std::iter::repeat_with(|| None)
                .take(count as usize)
                .collect(),
        };
        if sampler {
            null_sampler(templates.null());
        } else {
            null_view(templates.null());
        }
        Some(templates)
    }

    pub(super) fn bucket(&self, key: u64) -> usize {
        ((key ^ key.rotate_left(32)).wrapping_mul(0x9e37_79b9_7f4a_7c15)
            >> (64 - self.entries.len().trailing_zeros())) as usize
    }

    pub(super) fn slot(&self, bucket: usize) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: self.cpu + bucket * self.step,
        }
    }

    pub(super) fn null(&self) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        self.slot(self.entries.len())
    }

    pub(super) fn get(&self, key: u64, epoch: u64) -> Option<Hit> {
        let bucket = self.bucket(key);
        let entry = self.entries[bucket].as_ref()?;
        if entry.key != key || (!self.sampler && entry.epoch != epoch) {
            return None;
        }
        let (held, backing) = if let Some(buffer) = &entry.buffer {
            let shadow = buffer.shadow.upgrade()?;
            let (resource, source) = match &buffer.widened {
                Some(widened) => {
                    let widened = widened.upgrade()?;
                    (
                        widened.buffer.clone(),
                        HeldSource::Widened(widened, shadow.clone()),
                    )
                }
                None => (shadow.buffer.clone(), HeldSource::Shadow(shadow.clone())),
            };
            (
                Some((resource, Some(source))),
                Some((shadow, buffer.address, buffer.len)),
            )
        } else {
            (None, None)
        };
        Some(Hit {
            at: self.slot(bucket),
            held,
            mark: entry.mark,
            backing,
        })
    }

    pub(super) fn make(
        &mut self,
        dst: Option<D3D12_CPU_DESCRIPTOR_HANDLE>,
        key: u64,
        epoch: u64,
        view: &View,
        backing: Option<&BufferView>,
    ) -> (Option<Kept>, Option<(u32, u32)>) {
        let bucket = self.bucket(key);
        let at = self.slot(bucket);
        self.entries[bucket] = None;
        let (kept, mark) = make_marked_with_backing(view, at, backing.map(|b| &b.0));
        if let Some(dst) = dst.filter(|_| kept.is_some()) {
            copy(self.sampler, dst, at);
        }
        let buffer = match (kept.as_ref(), backing) {
            (Some(None), None) if self.sampler => None,
            (Some(Some((_, source))), Some((shadow, address, len)))
                if epoch == crate::shadow::lookup_epoch() =>
            {
                let widened = match source {
                    Some(HeldSource::Widened(widened, _)) => Some(Arc::downgrade(widened)),
                    Some(HeldSource::Shadow(_)) => None,
                    _ => return (kept, mark),
                };
                Some(Buffer {
                    shadow: Arc::downgrade(shadow),
                    widened,
                    address: *address,
                    len: *len,
                })
            }
            _ => return (kept, mark),
        };
        self.entries[bucket] = Some(Entry {
            key,
            epoch,
            mark,
            buffer,
        });
        (kept, mark)
    }
}
