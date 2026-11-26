//! Write shader-produced textures back to the console's unified title memory.
//! The allowlist covers the ocean height and displacement maps the title CPU
//! samples. Record the UAV subresources at list close and land their texels
//! after the host fence, before the title's completion writes.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};

use victory_formats::texture::Layout;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT;

use crate::host::{HostList, host};
use crate::object::This;
use crate::resource::{self, HostTexture};

type Kind = (u64, u32, u32);
const OCEAN: &str = "256x256f41,256x256f16,64x64f16,128x128f36";

fn parse_kinds(spec: &str) -> Vec<Kind> {
    let spec = if spec == "1" { OCEAN } else { spec };
    spec.split(',')
        .filter_map(|kind| {
            let (size, format) = kind.trim().split_once('f')?;
            let (width, height) = size.split_once('x')?;
            Some((
                width.parse().ok()?,
                height.parse().ok()?,
                format.parse().ok()?,
            ))
        })
        .collect()
}

fn kinds() -> &'static [Kind] {
    static KINDS: OnceLock<Vec<Kind>> = OnceLock::new();
    KINDS.get_or_init(|| {
        let kinds = parse_kinds(&std::env::var("VICTORY_TEXTURE_BACK").unwrap_or_default());
        if !kinds.is_empty() {
            tracing::info!(?kinds, "shader texture write-back enabled");
        }
        kinds
    })
}

#[cfg(test)]
thread_local! {
    pub(crate) static TEST_KINDS: std::cell::Cell<&'static [Kind]> = const { std::cell::Cell::new(&[]) };
}

fn allowlist() -> &'static [Kind] {
    let allow = kinds();
    #[cfg(test)]
    let allow = TEST_KINDS.with(|test| {
        if test.get().is_empty() {
            allow
        } else {
            test.get()
        }
    });
    allow
}

pub(crate) fn enabled() -> bool {
    !allowlist().is_empty()
}

/// Shared UAV slots can change between recording and execution, including
/// a change back to the original descriptor. A retired binding stays invalid.
#[derive(Default)]
pub(crate) struct Binding(AtomicBool);

impl Binding {
    pub(crate) fn invalidate(&self) {
        self.0.store(true, Ordering::Release);
    }

    fn invalid(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}

pub(crate) fn tracking_bindings() -> bool {
    !kinds().is_empty() || cfg!(test)
}

pub(crate) type SharedSlot<'a> = (
    &'a crate::descriptor::host::HostHeap,
    u32,
    &'a crate::descriptor::ViewPin,
);

/// A recorded write can only land while the same title allocation lives at
/// its address. Keep the host copy independently, through GPU execution.
struct Ticket {
    owner: usize,
    at: usize,
    end: usize,
    valid: AtomicBool,
}

static TICKETS: Mutex<Vec<Weak<Ticket>>> = Mutex::new(Vec::new());

fn tickets() -> std::sync::MutexGuard<'static, Vec<Weak<Ticket>>> {
    TICKETS.lock().unwrap_or_else(|e| e.into_inner())
}

fn invalidate(entries: &mut Vec<Weak<Ticket>>, matches: impl Fn(&Ticket) -> bool) {
    entries.retain(|weak| {
        let Some(ticket) = weak.upgrade() else {
            return false;
        };
        if matches(&ticket) {
            ticket.valid.store(false, Ordering::Relaxed);
            return false;
        }
        true
    });
}

/// Prevents landing or registering a read-back while a title view is
/// detached and its surviving section pages are being restored.
pub(crate) struct MappingChange(Option<std::sync::MutexGuard<'static, Vec<Weak<Ticket>>>>);

pub(crate) fn mapping_change() -> MappingChange {
    MappingChange((!kinds().is_empty() || cfg!(test)).then(tickets))
}

impl MappingChange {
    pub(crate) fn discard(&mut self, at: usize, len: usize) {
        let end = at.saturating_add(len);
        if at < end
            && let Some(entries) = &mut self.0
        {
            invalidate(entries, |ticket| at < ticket.end && ticket.at < end);
        }
    }
}

pub(crate) fn remapped(at: usize, len: usize) {
    mapping_change().discard(at, len);
}

pub(crate) fn released(owner: usize) {
    if !kinds().is_empty() || cfg!(test) {
        invalidate(&mut tickets(), |ticket| ticket.owner == owner);
    }
}

#[derive(Clone, Default)]
pub(crate) struct Wanted(BTreeMap<usize, Target>);

#[derive(Clone)]
struct Target {
    copy: Arc<HostTexture>,
    layout: Layout,
    ticket: Arc<Ticket>,
    subresources: BTreeMap<u32, Subresource>,
}

/// Only shared slots that could write this mip and array slice.
#[derive(Clone)]
struct Subresource {
    format: DXGI_FORMAT,
    bindings: Vec<Arc<Binding>>,
}

impl Subresource {
    fn watch(&mut self, binding: Arc<Binding>) {
        if !self
            .bindings
            .iter()
            .any(|other| Arc::ptr_eq(other, &binding))
        {
            self.bindings.push(binding);
        }
    }
}

fn bindings_live(bindings: &[Arc<Binding>]) -> bool {
    !bindings.iter().any(|binding| binding.invalid())
}

const MAX_TARGETS: usize = 64;
const MAX_TEXTURE_BYTES: u64 = 8 << 20;
const MAX_LIVE_BYTES: u64 = 32 << 20;
static LIVE_BYTES: AtomicU64 = AtomicU64::new(0);

/// Charges the memory until the last submission holding the list is done
/// and the list is released or reset.
struct Budget(u64);

impl Budget {
    fn reserve(bytes: u64) -> Option<Self> {
        let bytes = bytes.checked_next_multiple_of(64 << 10)?;
        LIVE_BYTES
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |used| {
                (used + bytes <= MAX_LIVE_BYTES).then_some(used + bytes)
            })
            .ok()?;
        Some(Self(bytes))
    }
}

impl Drop for Budget {
    fn drop(&mut self) {
        LIVE_BYTES.fetch_sub(self.0, Ordering::Relaxed);
    }
}

impl Wanted {
    pub(crate) fn mark(
        &mut self,
        resource: This,
        words: Option<&[u32; 10]>,
        slot: Option<SharedSlot<'_>>,
    ) {
        let Some((at, desc)) = resource::texture_memory(resource) else {
            return;
        };
        if !resource::texture_back_kind(&desc).is_some_and(|kind| allowlist().contains(&kind)) {
            return;
        }
        let target = match self.0.get_mut(&(resource as usize)) {
            Some(target) => target,
            None => {
                if self.0.len() >= MAX_TARGETS {
                    return;
                }
                let (Some(copy), Some(texture)) = (
                    resource::host_copy(resource),
                    resource::texture_layout(&desc),
                ) else {
                    return;
                };
                let [plane] = &texture.planes[..] else { return };
                if at.is_null() || !plane.layout.addressable() || texture.size > MAX_TEXTURE_BYTES {
                    return;
                }
                let at = at as usize;
                let Some(end) = at.checked_add(texture.size as usize) else {
                    return;
                };
                let ticket = Arc::new(Ticket {
                    owner: resource as usize,
                    at,
                    end,
                    valid: AtomicBool::new(true),
                });
                let mut live = tickets();
                live.retain(|weak| weak.strong_count() != 0);
                live.push(Arc::downgrade(&ticket));
                self.0.entry(resource as usize).or_insert(Target {
                    copy,
                    layout: plane.layout.clone(),
                    ticket,
                    subresources: BTreeMap::new(),
                })
            }
        };
        let binding = if let Some((heap, index, view)) = slot {
            let Some(binding) = heap.watch_texture_binding(index, view, &target.copy.texture)
            else {
                return;
            };
            Some(binding)
        } else {
            None
        };
        target.mark(resource, words, binding.as_ref());
    }

    /// Capture the texture in an immutable slot while its original title
    /// object is still available. Keep the host texture and allocation
    /// ticket, never a title reference that would defer its release.
    pub(crate) fn dynamic(
        heap: &crate::descriptor::host::HostHeap,
        index: u32,
        view: &crate::descriptor::View,
    ) -> Option<Self> {
        use crate::descriptor::View;
        let placed = match view {
            View::Placed(true, ..) => crate::descriptor::host::placed_texture(view),
            View::Unordered(..) => None,
            _ => return None,
        };
        let View::Unordered(resource, _, words) = placed.as_ref().unwrap_or(view) else {
            return None;
        };
        let mut wanted = Self::default();
        wanted.mark(*resource as This, words.as_ref(), None);
        let target = wanted.0.get(resource)?;
        (!target.subresources.is_empty() && heap.keeps_texture(index, &target.copy.texture))
            .then_some(wanted)
    }

    /// The dynamic descriptor still names this copy even if the original
    /// title slot can no longer supply its state during table preparation.
    pub(crate) fn prepare_dynamic(&self, list: &HostList) {
        for target in self.0.values() {
            if let Some(barrier) =
                list.move_texture(&target.copy, D3D12_RESOURCE_STATE_UNORDERED_ACCESS.0 as u32)
            {
                // SAFETY: an open list holding this live texture.
                unsafe { list.list.ResourceBarrier(&[barrier]) };
            }
        }
    }

    pub(crate) fn commit(&mut self, attempted: Self) {
        for (owner, target) in attempted.0 {
            if !target.ticket.valid.load(Ordering::Relaxed) {
                continue;
            }
            if let Some(current) = self.0.get_mut(&owner)
                && current.ticket.valid.load(Ordering::Relaxed)
                && Arc::ptr_eq(&current.copy, &target.copy)
            {
                for (index, subresource) in target.subresources {
                    match current.subresources.entry(index) {
                        std::collections::btree_map::Entry::Vacant(entry) => {
                            entry.insert(subresource);
                        }
                        std::collections::btree_map::Entry::Occupied(mut entry) => {
                            let current = entry.get_mut();
                            current.format = subresource.format;
                            for binding in subresource.bindings {
                                current.watch(binding);
                            }
                        }
                    }
                }
            } else if self.0.len() < MAX_TARGETS || self.0.contains_key(&owner) {
                self.0.insert(owner, target);
            }
        }
    }

    pub(crate) fn record(self, list: &HostList) -> Vec<Back> {
        self.0
            .into_values()
            .filter_map(|target| target.record(list))
            .collect()
    }
}

impl Target {
    fn mark(&mut self, resource: This, words: Option<&[u32; 10]>, binding: Option<&Arc<Binding>>) {
        let Some(view) = crate::descriptor::host::color_uav(resource, &self.copy, words) else {
            return;
        };
        // SAFETY: a live host texture.
        let desc = unsafe { self.copy.texture.GetDesc() };
        let (mip, first, count, format) = match view {
            // A default 2D UAV shows mip zero of the entire array.
            None => (0, 0, u32::from(desc.DepthOrArraySize), desc.Format),
            // SAFETY: ViewDimension selects the active union member.
            Some(view) => unsafe {
                match view.ViewDimension {
                    D3D12_UAV_DIMENSION_TEXTURE2D
                        if view.Anonymous.Texture2D.PlaneSlice == 0
                            && desc.DepthOrArraySize == 1 =>
                    {
                        (view.Anonymous.Texture2D.MipSlice, 0, 1, view.Format)
                    }
                    D3D12_UAV_DIMENSION_TEXTURE2DARRAY
                        if view.Anonymous.Texture2DArray.PlaneSlice == 0 =>
                    {
                        let array = view.Anonymous.Texture2DArray;
                        (
                            array.MipSlice,
                            array.FirstArraySlice,
                            array.ArraySize,
                            view.Format,
                        )
                    }
                    _ => return,
                }
            },
        };
        let mips = u32::from(desc.MipLevels);
        let slices = u32::from(desc.DepthOrArraySize);
        // UINT_MAX in a 2D array UAV covers the remaining array slices.
        let end = if count == u32::MAX {
            slices
        } else if let Some(end) = first.checked_add(count) {
            end
        } else {
            return;
        };
        if mip >= mips
            || first >= end
            || end > slices
            || mips as usize != self.layout.mips.len()
            || slices != self.layout.slices
        {
            return;
        }
        let own = desc.Format.0 as u32;
        // Copies use the host texture's own format. For typeless textures
        // use the actual UAV's typed sibling for the linear footprint.
        let format = if resource::family(own + 1) == own {
            format
        } else {
            desc.Format
        };
        let typed = format.0 as u32;
        if typed == 0
            || resource::family(typed + 1) == typed
            || resource::family(own) != resource::family(typed)
            || resource::element(typed) != Some((u64::from(self.layout.desc.element_bytes), 1))
        {
            return;
        }
        for slice in first..end {
            let subresource = self
                .subresources
                .entry(slice * mips + mip)
                .or_insert_with(|| Subresource {
                    format,
                    bindings: Vec::new(),
                });
            subresource.format = format;
            if let Some(binding) = binding {
                subresource.watch(binding.clone());
            }
        }
    }

    fn record(self, list: &HostList) -> Option<Back> {
        if self.subresources.is_empty() || !self.ticket.valid.load(Ordering::Relaxed) {
            return None;
        }
        let host = host()?;
        let texture = &self.copy.texture;
        // SAFETY: a live host texture.
        let mut desc = unsafe { texture.GetDesc() };
        let mips = u32::from(desc.MipLevels);
        let mut total = 0u64;
        let mut copies = Vec::with_capacity(self.subresources.len());
        for (index, Subresource { format, bindings }) in self.subresources {
            if !bindings_live(&bindings) {
                continue;
            }
            let mut footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
            let mut pitch_end = 0;
            let mut rows = 0;
            let mut row_bytes = 0;
            let offset = total
                .checked_next_multiple_of(u64::from(D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT))?;
            desc.Format = format;
            // SAFETY: one subresource, one output entry for each query.
            unsafe {
                host.device.GetCopyableFootprints(
                    &desc,
                    index,
                    1,
                    offset,
                    Some(&mut footprint),
                    Some(&mut rows),
                    Some(&mut row_bytes),
                    Some(&mut pitch_end),
                )
            };
            let mip = index % mips;
            let slice = index / mips;
            let level = &self.layout.mips[mip as usize];
            let f = footprint.Footprint;
            if pitch_end == u64::MAX
                || footprint.Offset != offset
                || f.Format != format
                || f.Depth != 1
                || f.Width != level.width
                || f.Height != level.height
                || rows != level.height
                || row_bytes != u64::from(level.width) * u64::from(self.layout.desc.element_bytes)
                || u64::from(f.RowPitch) < row_bytes
            {
                return None;
            }
            // Include the last row's padding for the pitched CPU slice.
            let len = u64::from(f.RowPitch) * u64::from(rows);
            total = offset.checked_add(len)?;
            if total > MAX_TEXTURE_BYTES {
                tracing::debug!(total, "texture write-back exceeds the per-texture budget");
                return None;
            }
            copies.push(Copy {
                index,
                mip,
                slice,
                footprint,
                len: len as usize,
                bindings,
            });
        }
        if copies.is_empty() {
            return None;
        }
        let Some(budget) = Budget::reserve(total) else {
            tracing::debug!(
                total,
                "texture write-back exceeds the live read-back budget"
            );
            return None;
        };
        let buffer = host.readback_buffer(total)?;
        if let Some(barrier) =
            list.move_texture(&self.copy, D3D12_RESOURCE_STATE_COPY_SOURCE.0 as u32)
        {
            // SAFETY: an open list. Its state tracking keeps the texture alive.
            unsafe { list.list.ResourceBarrier(&[barrier]) };
        }
        for copy in &copies {
            let to = D3D12_TEXTURE_COPY_LOCATION {
                // SAFETY: borrowed COM pointers; the list keeps both resources.
                pResource: unsafe { std::mem::transmute_copy(&buffer) },
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    PlacedFootprint: copy.footprint,
                },
            };
            let from = D3D12_TEXTURE_COPY_LOCATION {
                // SAFETY: as above.
                pResource: unsafe { std::mem::transmute_copy(texture) },
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    SubresourceIndex: copy.index,
                },
            };
            // SAFETY: matching footprints and copy states on an open list.
            unsafe { list.list.CopyTextureRegion(&to, 0, 0, 0, &from, None) };
        }
        Some(Back {
            buffer,
            copies,
            layout: self.layout,
            ticket: self.ticket,
            _budget: budget,
        })
    }
}

struct Copy {
    index: u32,
    mip: u32,
    slice: u32,
    footprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
    len: usize,
    bindings: Vec<Arc<Binding>>,
}

pub(crate) struct Back {
    buffer: ID3D12Resource,
    copies: Vec<Copy>,
    layout: Layout,
    ticket: Arc<Ticket>,
    _budget: Budget,
}

/// Marks only the elements visited by the layout walk, in any order.
struct WrittenTexels {
    words: Vec<u64>,
    element: usize,
}

impl WrittenTexels {
    fn new(bytes: usize, element: usize) -> Self {
        Self {
            words: vec![0; bytes.div_ceil(64 * element)],
            element,
        }
    }

    fn mark(&mut self, at: usize, len: usize) {
        debug_assert!(at.is_multiple_of(self.element) && len.is_multiple_of(self.element));
        let mut first = at / self.element;
        let end = first + len / self.element;
        while first < end {
            let offset = first % 64;
            let count = (end - first).min(64 - offset);
            self.words[first / 64] |= (u64::MAX >> (64 - count)) << offset;
            first += count;
        }
    }

    fn find(&self, at: usize, written: bool) -> Option<usize> {
        let first = at / 64;
        let mut mask = u64::MAX << (at % 64);
        for (i, &word) in self.words.iter().enumerate().skip(first) {
            let bits = (if written { word } else { !word }) & mask;
            if bits != 0 {
                return Some(i * 64 + bits.trailing_zeros() as usize);
            }
            mask = u64::MAX;
        }
        None
    }

    /// Maximal byte ranges of written texels, in address order.
    fn ranges(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        let mut at = 0;
        std::iter::from_fn(move || {
            let start = self.find(at, true)?;
            let end = self.find(start, false).unwrap_or(self.words.len() * 64);
            at = end;
            Some((start * self.element, end * self.element))
        })
    }
}

impl Back {
    pub(crate) fn land(&self) {
        if !self.ticket.valid.load(Ordering::Relaxed)
            || self
                .copies
                .iter()
                .all(|copy| !bindings_live(&copy.bindings))
        {
            return;
        }
        let mut data = std::ptr::null_mut();
        // SAFETY: the host fence completed the copy, and the buffer is live.
        if unsafe { self.buffer.Map(0, None, Some(&mut data)) }.is_err() {
            tracing::warn!("texture write-back buffer could not be mapped");
            return;
        }
        if !data.is_null() {
            self.land_mapped(data.cast());
        }
        // SAFETY: the map succeeded, and the CPU only read the buffer.
        unsafe {
            self.buffer
                .Unmap(0, Some(&D3D12_RANGE { Begin: 0, End: 0 }))
        };
    }

    fn land_mapped(&self, data: *const u8) {
        let mut tiled = vec![0; self.layout.size as usize];
        let mut texels = WrittenTexels::new(tiled.len(), self.layout.desc.element_bytes as usize);
        for copy in &self.copies {
            if !bindings_live(&copy.bindings) {
                continue;
            }
            // SAFETY: the recorded copy and allocation include these rows.
            let linear = unsafe {
                std::slice::from_raw_parts(data.add(copy.footprint.Offset as usize), copy.len)
            };
            let pitch = copy.footprint.Footprint.RowPitch as usize;
            let result = self.layout.tile_into(
                tiled.len(),
                copy.mip,
                copy.slice,
                linear,
                pitch,
                copy.len,
                |at, bytes| {
                    let end = at + bytes.len();
                    tiled[at..end].copy_from_slice(bytes);
                    texels.mark(at, bytes.len());
                },
            );
            if let Err(error) = result {
                tracing::warn!(
                    ?error,
                    mip = copy.mip,
                    slice = copy.slice,
                    "texture write-back could not be tiled"
                );
                return;
            }
        }
        // Allocation invalidation and landing share this lock. Release and unmap
        // invalidate before giving up the allocation.
        let _tickets = tickets();
        if !self.ticket.valid.load(Ordering::Relaxed) {
            return;
        }
        let at = self.ticket.at;
        let _writing = crate::watch::writing(at, tiled.len());
        for (start, end) in texels.ranges() {
            crate::host::write_mapped(
                (at + start) as *mut u8,
                tiled[start..end].as_ptr(),
                end - start,
            );
        }
    }
}
