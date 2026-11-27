//! `xmem.dll`: console memory, served from host memory. The GPU side works
//! from CPU addresses (unified memory), so memory types and bandwidth
//! classes change nothing here; what constrains addresses (alignment,
//! 2 MiB pages, address-bit limits) is honoured.
#![allow(non_snake_case)]
// Title code calls these through its import table, passing pointers the real
// API would trust.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::alloc::{Layout, alloc, dealloc};
use std::collections::{BTreeMap, HashMap};
use std::ffi::c_void;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, LazyLock, Mutex, MutexGuard, OnceLock};

use windows_sys::Win32::Foundation::{
    CloseHandle, ERROR_COMMITMENT_LIMIT, HANDLE, INVALID_HANDLE_VALUE,
};
use windows_sys::Win32::System::Memory::{
    CreateFileMappingW, MEM_ADDRESS_REQUIREMENTS, MEM_COMMIT, MEM_EXTENDED_PARAMETER,
    MEM_PRESERVE_PLACEHOLDER, MEM_RELEASE, MEM_REPLACE_PLACEHOLDER, MEM_RESERVE,
    MEM_RESERVE_PLACEHOLDER, MEMORY_BASIC_INFORMATION, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile3,
    MemExtendedParameterAddressRequirements, PAGE_NOACCESS, PAGE_READWRITE, SEC_COMMIT,
    SEC_RESERVE, UnmapViewOfFile2, VirtualAlloc, VirtualAlloc2, VirtualFree, VirtualQueryEx,
};
use windows_sys::Win32::System::ProcessStatus::{
    K32GetPerformanceInfo, K32GetProcessMemoryInfo, PERFORMANCE_INFORMATION,
    PROCESS_MEMORY_COUNTERS,
};
use windows_sys::Win32::System::Threading::GetCurrentProcess;
use windows_sys::core::BOOL;

use crate::{HleModule, hle_export};

const MIB: usize = 1 << 20;

/// Whether we run under Wine (its ntdll exports `wine_get_version`).
pub(crate) fn on_wine() -> bool {
    static WINE: LazyLock<bool> = LazyLock::new(|| {
        use windows_sys::Win32::System::LibraryLoader::{GetModuleHandleA, GetProcAddress};
        // SAFETY: plain lookups in ntdll, always loaded.
        unsafe {
            let ntdll = GetModuleHandleA(c"ntdll.dll".as_ptr().cast());
            !ntdll.is_null() && GetProcAddress(ntdll, c"wine_get_version".as_ptr().cast()).is_some()
        }
    });
    *WINE
}

/// Standard memory the title has handed to the system.
static TRANSFERRED: AtomicUsize = AtomicUsize::new(0);

// Console additions to the host's flags.
/// 2 MiB pages; the host reads this bit as large pages, which need a
/// privilege.
const PAGES_2M: u32 = 0x2000_0000;
/// With `PAGES_2M`: 64 KiB pages; the host reads it as AWE memory.
const PAGES_64K_BIT: u32 = 0x40_0000;
const PAGES_64K: u32 = PAGES_2M | PAGES_64K_BIT;
/// The host's protection constants and `PAGE_GUARD`. Above them the console
/// adds GPU access bits (0x800-0x40000). Write-combining and uncached are
/// dropped too: our GPU side reads this memory through the CPU.
const HOST_PROTECTION: u32 = 0x1ff;
/// XMem flags of memory both processors reach: all of ours.
const GRAPHICS: u64 = 3;
/// XMem flag bit of 32-bit graphics memory.
const ADDRESS_32BIT: u64 = 4;
/// The host's write watching (not in windows-sys' memory module).
const MEM_WRITE_WATCH: u32 = 0x20_0000;

/// The addresses XMem `flags` allow, first and last: the low 4 GiB for
/// 32-bit memory; else from 4 GiB (kept for 32-bit requests, which the
/// title's big pools would otherwise crowd out) up to 46 down to 40 bits
/// from the field at bit 12, or 40 bits without one: the console keeps
/// title memory there, and its ray-tracing structures must be (Grand Theft
/// Auto VI's BVH builder refuses one above; the title asks for its GPU heap
/// with no field). Decision 0001.
fn address_range(flags: u64) -> (usize, usize) {
    let bits = match (flags & ADDRESS_32BIT, flags >> 12 & 7) {
        (0, 0) => 40,
        (0, field) => 47 - field,
        _ => return (0, u32::MAX as usize),
    };
    (1 << 32, (1 << bits) - 1)
}

pub extern "system" fn XMemVirtualAlloc(
    base: *mut c_void,
    size: usize,
    kind: u32,
    flags: u64,
    protect: u32,
) -> *mut c_void {
    let (lowest, highest) = address_range(flags);
    let mut requirements = MEM_ADDRESS_REQUIREMENTS {
        LowestStartingAddress: lowest as *mut c_void,
        HighestEndingAddress: highest as *mut c_void,
        Alignment: if kind & (PAGES_2M | PAGES_64K_BIT) == PAGES_2M {
            2 * MIB
        } else {
            0
        },
    };
    let mut protect = match protect & HOST_PROTECTION {
        0 => PAGE_NOACCESS,
        p => p,
    };
    // A reservation of 64 KiB pages takes physical pages later
    // (`XMemMapPhysicalPages`); on the host only a placeholder can.
    // shortcut: committing part of such a reservation later would fail
    // (Grand Theft Auto VI's 23 calls never do); replace that part of the
    // placeholder then.
    let placeholder = (kind & PAGES_64K == PAGES_64K && kind & MEM_COMMIT == 0)
        || kind & MEM_RESERVE_PLACEHOLDER != 0;
    if placeholder {
        if kind & MEM_REPLACE_PLACEHOLDER != 0 {
            // Part of a placeholder the title split: it stays one.
            return base;
        }
        protect = PAGE_NOACCESS;
    }
    let mut kind = kind & !(PAGES_2M | PAGES_64K_BIT);
    if placeholder {
        kind |= MEM_RESERVE | MEM_RESERVE_PLACEHOLDER;
    }
    let mut parameter = MEM_EXTENDED_PARAMETER::default();
    parameter.Anonymous1._bitfield = MemExtendedParameterAddressRequirements as u64;
    parameter.Anonymous2.Pointer = (&raw mut requirements).cast();
    // Address requirements apply to reservations at an address of our choice.
    let constrained = base.is_null();
    if constrained {
        kind |= MEM_RESERVE;
        // The GPU side copies only the pages the title wrote since its last
        // copy (`victory_gpu`, decision 0007); placeholders cannot be
        // watched (their physical-page views are copied whole). Nor is
        // anything under Rosetta, where Wine's watch faults on first writes
        // Rosetta cannot resume (`victory_gpu::fault_tracking`).
        if kind & MEM_RESERVE_PLACEHOLDER == 0 && victory_gpu::fault_tracking() {
            kind |= MEM_WRITE_WATCH;
        }
    }
    // SAFETY: the title's own arguments, cleaned of console bits; the
    // parameter points at `requirements`, which outlives the call.
    let allocate = |parameter: &mut MEM_EXTENDED_PARAMETER| unsafe {
        VirtualAlloc2(
            GetCurrentProcess(),
            base,
            size,
            kind,
            protect,
            if constrained {
                parameter
            } else {
                std::ptr::null_mut()
            },
            constrained.into(),
        )
    };
    let mut at = allocate(&mut parameter);
    // A 32-bit request that found no room below 4 GiB: the guards there
    // make way for it.
    if at.is_null() && constrained && highest < FOUR_GIB {
        at = low_guards::retry(size, || allocate(&mut parameter));
    }
    if at.is_null() {
        let error = std::io::Error::last_os_error();
        if kind & MEM_COMMIT != 0 {
            commit_refused(size, "virtual memory", error);
        } else {
            tracing::warn!(
                size,
                kind = format_args!("{kind:#x}"),
                flags = format_args!("{flags:#x}"),
                %error,
                "XMemVirtualAlloc refused"
            );
        }
    }
    if !at.is_null() && placeholder && victory_gpu::physical_tiles_enabled() {
        physical().reserve(at as usize, size);
    }
    if !at.is_null()
        && victory_gpu::alias_buffers()
        && kind & MEM_COMMIT != 0
        && kind & MEM_RESERVE_PLACEHOLDER == 0
        && protect == PAGE_READWRITE
        && (at as usize).is_multiple_of(PAGE)
        && size >= PAGE
        && size.is_multiple_of(PAGE)
    {
        victory_gpu::alias_region(at as usize, size, false);
    }
    if base.is_null() {
        tracing::debug!(
            size,
            kind = format_args!("{kind:#x}"),
            flags = format_args!("{flags:#x}"),
            at = format_args!("{at:p}"),
            "XMemVirtualAlloc"
        );
    }
    at
}

/// `MEMORY_BASIC_INFORMATION` (the console's first 0x30 bytes are the
/// same), then the XMem flags and tag.
#[repr(C)]
pub struct BasicInformation {
    host: MEMORY_BASIC_INFORMATION,
    flags: u64,
    tag_code: u32,
    tag: [u8; 4],
}

// shortcut: every committed page reports graphics memory and no tag, since
// all of it is GPU-reachable here (Grand Theft Auto VI's BVH builder, the
// only caller, wants graphics memory); remember the flags of
// `XMemVirtualAlloc` regions if a title tells CPU-only memory apart.
pub extern "system" fn XMemVirtualQuery(
    process: HANDLE,
    address: *const c_void,
    out: *mut BasicInformation,
) -> BOOL {
    let size = size_of::<MEMORY_BASIC_INFORMATION>();
    // SAFETY: the title passes room for the whole structure; the host
    // fails on a null one.
    unsafe {
        if VirtualQueryEx(process, address, out.cast(), size) == 0 {
            return 0;
        }
        let info = &mut *out;
        info.flags = if info.host.State == MEM_COMMIT {
            GRAPHICS
        } else {
            0
        };
        info.tag_code = 0;
        info.tag = [0; 4];
    }
    1
}

// `XMemAlloc` attributes: alignment as log2 (5 bits at 24, 4 bytes and up
// from 2), then the memory type, then bit 33: a 32-bit address.
const ALIGNMENT_SHIFT: u64 = 24;
const ALLOC_32BIT: u64 = 1 << 33;
/// Our block header: the block's size and alignment, just below the
/// pointer handed out.
const HEADER: usize = 2 * size_of::<usize>();

fn layout(size: usize, attributes: u64) -> Option<Layout> {
    // An alignment of at least 16 leaves room for the header.
    let align = 1usize << (attributes >> ALIGNMENT_SHIFT & 0x1f).max(4);
    Layout::from_size_align(size.checked_add(align)?, align).ok()
}

type AllocRoutine = extern "system" fn(usize, u64) -> *mut c_void;
type FreeRoutine = extern "system" fn(*mut c_void, u64);

/// The allocation and free routines the title's exe exports as
/// `__XMemAllocationHooks__` (Grand Theft Auto VI does): `XMemAlloc` and
/// `XMemFree` call them, for the title's own calls and its middleware's;
/// the `…Default` exports do not.
fn hooks() -> [usize; 2] {
    static AT: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    let at = match AT.get() {
        Some(&at) => at,
        // Looked up once the title is mapped, which is before it runs.
        None => match crate::runtime::main_image() {
            Some(base) => *AT.get_or_init(|| {
                use windows_sys::Win32::System::LibraryLoader::GetProcAddress;
                // SAFETY: a mapped image and a NUL-terminated name.
                let at = unsafe {
                    GetProcAddress(base as _, c"__XMemAllocationHooks__".as_ptr().cast())
                }
                .map_or(0, |f| f as usize);
                tracing::info!(found = at != 0, "xmem allocation hooks");
                at
            }),
            None => 0,
        },
    };
    if at == 0 {
        return [0; 2];
    }
    // SAFETY: the export is two routine pointers.
    unsafe { (at as *const [usize; 2]).read() }
}

/// The title's general allocator, for middleware too: its hook if it has
/// one, else [`XMemAllocDefault`].
pub extern "system" fn XMemAlloc(size: usize, attributes: u64) -> *mut c_void {
    match hooks()[0] {
        0 => XMemAllocDefault(size, attributes),
        hook => {
            // SAFETY: the title's own routine of this type.
            let alloc: AllocRoutine = unsafe { std::mem::transmute(hook) };
            alloc(size, attributes)
        }
    }
}

/// See [`XMemAlloc`].
pub extern "system" fn XMemFree(address: *mut c_void, attributes: u64) {
    match hooks()[1] {
        0 => XMemFreeDefault(address, attributes),
        hook => {
            // SAFETY: the title's own routine of this type.
            let free: FreeRoutine = unsafe { std::mem::transmute(hook) };
            free(address, attributes)
        }
    }
}

/// The system allocator behind [`XMemAlloc`].
pub extern "system" fn XMemAllocDefault(size: usize, attributes: u64) -> *mut c_void {
    if attributes & ALLOC_32BIT != 0 {
        return XMemVirtualAlloc(
            std::ptr::null_mut(),
            size,
            MEM_RESERVE | MEM_COMMIT,
            GRAPHICS | ADDRESS_32BIT,
            PAGE_READWRITE,
        );
    }
    let Some(layout) = layout(size, attributes) else {
        return std::ptr::null_mut();
    };
    // shortcut: the host heap's addresses lie above 2^40, unlike the
    // console's; nothing seen builds ray-tracing structures in this memory
    // (Grand Theft Auto VI uses its GPU heap). Sub-allocate from a
    // reservation below 2^40 if one does.
    // SAFETY: the layout has a non-zero size.
    let block = unsafe { alloc(layout) };
    if block.is_null() {
        tracing::warn!(size, "XMemAllocDefault: the host heap is out of memory");
        return block.cast();
    }
    // SAFETY: the block holds the alignment (at least the header) and then
    // `size` bytes.
    unsafe {
        let user = block.add(layout.align());
        user.sub(HEADER)
            .cast::<[usize; 2]>()
            .write([layout.size(), layout.align()]);
        user.cast()
    }
}

/// Frees what [`XMemAllocDefault`] returned, given the same attributes.
pub extern "system" fn XMemFreeDefault(address: *mut c_void, attributes: u64) {
    if address.is_null() {
        return;
    }
    // SAFETY: the address came from `XMemAlloc` with these attributes: a
    // whole reservation, or a header-prefixed block.
    unsafe {
        if attributes & ALLOC_32BIT != 0 {
            victory_gpu::with_inflight_mapping(|| {
                victory_gpu::unalias_region(address as usize);
                VirtualFree(address, 0, MEM_RELEASE);
            });
            return;
        }
        let user = address.cast::<u8>();
        let [size, align] = user.sub(HEADER).cast::<[usize; 2]>().read();
        dealloc(
            user.sub(align),
            Layout::from_size_align_unchecked(size, align),
        );
    }
}

/// A retail console has no tools memory.
pub extern "system" fn XMemIsToolsMemAvailable() -> BOOL {
    0
}

// Physical pages: on the console, 64 KiB frames a title maps into its own
// reservations, at more than one address if it likes. Here each allocation
// is a pagefile-backed section, its pages are 64 KiB slices of it, and a
// mapping is a view of consecutive slices in place of a host placeholder.

const PAGE: usize = 64 << 10;

struct Section {
    handle: HANDLE,
    size: usize,
    /// Optional shared CPU/GPU backing. The file still supplies Wine's
    /// placeholder-view metadata.
    shared: Option<Arc<victory_gpu::PhysicalHeap>>,
    /// Our own view of the whole section, mapped the first time a
    /// DirectStorage request writes its pages and kept while the section
    /// lives (0: it could not be mapped). A view per request cost a map
    /// and an unmap, and under Wine each holds the process's one
    /// virtual-memory lock across a wineserver request: every page fault
    /// and protection change on other threads waits for it (#50).
    whole: OnceLock<usize>,
}

// SAFETY: a section handle may be used from any thread, and the view is
// plain memory.
unsafe impl Send for Section {}
// SAFETY: as above.
unsafe impl Sync for Section {}

impl Section {
    /// The address of our view of the whole section, mapped the first time;
    /// `None` if the host would not map it.
    fn whole(&self) -> Option<usize> {
        if let Some(shared) = &self.shared {
            return Some(shared.cpu_address());
        }
        use windows_sys::Win32::System::Memory::{FILE_MAP_WRITE, MapViewOfFile};
        let at = *self.whole.get_or_init(|| {
            // SAFETY: our section; the view is ours until the section drops.
            unsafe { MapViewOfFile(self.handle, FILE_MAP_WRITE, 0, 0, self.size) }.Value as usize
        });
        (at != 0).then_some(at)
    }
}

impl Drop for Section {
    fn drop(&mut self) {
        use windows_sys::Win32::System::Memory::UnmapViewOfFile;
        if let Some(&at) = self.whole.get()
            && at != 0
        {
            // SAFETY: our view of the section; nothing borrows it any more
            // (views of it hold the section).
            unsafe {
                UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
                    Value: at as *mut c_void,
                })
            };
        }
        // SAFETY: our handle; views keep the section alive on their own.
        unsafe { CloseHandle(self.handle) };
    }
}

/// Where a page lives, or what a view starts with: a section and a page
/// index in it.
#[derive(Clone)]
struct Slice {
    section: Arc<Section>,
    index: usize,
}

#[derive(Default)]
struct Physical {
    next_page: usize,
    reservations: BTreeMap<usize, TileReservation>,
    pages: HashMap<usize, Slice>,
    /// Mapped views by address: their length and what they show.
    views: BTreeMap<usize, (usize, Slice, u64)>,
    /// Where each slice (section, page index) is mapped: the addresses of
    /// its page in every view showing it (the GPU side's change tracking
    /// follows writes through one view to the others,
    /// `victory_gpu::title_memory_aliased`).
    shown: HashMap<(usize, usize), Vec<usize>>,
    next_write: usize,
    writes: HashMap<usize, SectionWrite>,
}

struct TileReservation {
    end: usize,
    chunk_size: usize,
    chunks: BTreeMap<usize, TileChunk>,
    remaining: Vec<(usize, usize)>,
}

enum TileChunk {
    Tiled(victory_gpu::TiledAlias),
    Ordinary,
}

impl TileReservation {
    fn release(&mut self, base: usize, at: usize, stop: usize) -> bool {
        let first = self.remaining.partition_point(|&(_, end)| end <= at);
        let last = self.remaining.partition_point(|&(start, _)| start < stop);
        if first == last {
            return false;
        }
        let start = self.remaining[first].0;
        let end = self.remaining[last - 1].1;
        let prefix = (start < at).then_some((start, at));
        let suffix = (stop < end).then_some((stop, end));
        self.remaining
            .splice(first..last, prefix.into_iter().chain(suffix));
        if self.remaining.is_empty() {
            return true;
        }
        let mut from = at;
        while from < stop {
            let index = (from - base) / self.chunk_size;
            let to = stop.min(base + (index + 1) * self.chunk_size);
            match self.chunks.entry(index).or_insert(TileChunk::Ordinary) {
                TileChunk::Tiled(alias) => alias.release(from, to),
                TileChunk::Ordinary => {}
            }
            from = to;
        }
        false
    }
}

impl Physical {
    fn reserve(&mut self, base: usize, size: usize) {
        const MAX_RESERVATIONS: usize = 256;
        const MAX_RESERVATION: usize = 64 << 30;
        let len = size;
        let Some(end) = base.checked_add(len) else {
            return;
        };
        if len == 0
            || len > MAX_RESERVATION
            || !(base | len).is_multiple_of(PAGE)
            || self.reservations.len() >= MAX_RESERVATIONS
        {
            return;
        }
        if self
            .reservations
            .range(..end)
            .next_back()
            .is_some_and(|(_, r)| r.end > base)
        {
            return;
        }
        let chunk_size = len.min(victory_gpu::MAX_TILED_BUFFER);
        self.reservations.insert(
            base,
            TileReservation {
                end,
                chunk_size,
                chunks: BTreeMap::new(),
                remaining: vec![(base, end)],
            },
        );
    }

    /// Keep a chunk on ordinary shadows when it must show file pages.
    /// A tiled VA already retained by a descriptor or list cannot migrate.
    fn admit_file(&mut self, at: usize) -> bool {
        let Some((&base, r)) = self.reservations.range_mut(..=at).next_back() else {
            return true;
        };
        if at >= r.end {
            return true;
        }
        let index = (at - base) / r.chunk_size;
        if let Some(TileChunk::Tiled(alias)) = r.chunks.get_mut(&index) {
            if !alias.owns(at) {
                return true;
            }
            if !alias.demote_if_unused() {
                return false;
            }
        }
        r.chunks.insert(index, TileChunk::Ordinary);
        true
    }

    fn update_tiles(&mut self, base: usize, len: usize, slice: Option<&Slice>) {
        let end = base + len;
        let first = self
            .reservations
            .range(..=base)
            .next_back()
            .map_or(base, |(&at, _)| at);
        for (&at, r) in self.reservations.range_mut(first..end) {
            let stop = end.min(r.end);
            let mut from = base.max(at);
            while from < stop {
                let index = (from - at) / r.chunk_size;
                let chunk = at + index * r.chunk_size;
                let chunk_end = (chunk + r.chunk_size).min(r.end);
                let to = stop.min(chunk_end);
                let entry = match slice {
                    Some(s) => Some(r.chunks.entry(index).or_insert_with(|| {
                        if s.section.shared.is_some()
                            && let Some(alias) =
                                victory_gpu::TiledAlias::new(chunk, chunk_end - chunk)
                        {
                            return TileChunk::Tiled(alias);
                        }
                        TileChunk::Ordinary
                    })),
                    None => r.chunks.get_mut(&index),
                };
                if let Some(TileChunk::Tiled(alias)) = entry {
                    let source = slice.and_then(|s| {
                        Some((s.section.shared.as_ref()?, s.index * PAGE + from - base))
                    });
                    alias.map(from, to - from, source);
                }
                from = to;
            }
        }
    }
}

/// Forwards title releases to Wine and retires the matching GPU alias.
pub(crate) fn release_virtual(address: *mut c_void, size: usize, kind: u32) -> BOOL {
    if kind != MEM_RELEASE || !victory_gpu::physical_tiles_enabled() {
        // SAFETY: forwards the title's arguments to the host API.
        return victory_gpu::with_inflight_mapping(|| unsafe { VirtualFree(address, size, kind) });
    }
    let release = || {
        let mut physical = physical();
        let at = address as usize;
        let reservation = physical
            .reservations
            .range(..=at)
            .next_back()
            .filter(|(_, r)| at < r.end)
            .map(|(&base, r)| (base, r.end));
        let mut info = MEMORY_BASIC_INFORMATION::default();
        let bytes = reservation
            .map(|(_, end)| {
                // SAFETY: our process and a correctly sized query structure.
                unsafe {
                    VirtualQueryEx(GetCurrentProcess(), address, &mut info, size_of_val(&info))
                };
                end.min(at + info.RegionSize) - at
            })
            .unwrap_or(0);
        // SAFETY: forwards the title's arguments to the host API.
        let released = unsafe { VirtualFree(address, size, kind) };
        if released != 0
            && bytes != 0
            && (at | bytes).is_multiple_of(PAGE)
            && let Some((base, _)) = reservation
            && physical
                .reservations
                .get_mut(&base)
                .unwrap()
                .release(base, at, at + bytes)
        {
            physical.reservations.remove(&base);
        }
        released
    };
    victory_gpu::with_physical_mapping(|_| true, release)
}

impl Slice {
    /// The key of page `k` of the slices from this one on in `shown`.
    fn key(&self, k: usize) -> (usize, usize) {
        (Arc::as_ptr(&self.section) as usize, self.index + k)
    }
}

/// Every address of a title view showing a page that title memory at
/// `[at, at + len)` shows, except the page's own address: pushed to `out`
/// as (page address, other address). Tests read `shown` back with it; the
/// GPU side's change tracking (`victory_gpu::watch`) marks writes there too.
#[cfg(test)]
fn aliases(at: usize, len: usize, out: &mut Vec<(usize, usize)>) {
    let physical = physical();
    let end = at.saturating_add(len);
    let first = physical
        .views
        .range(..=at)
        .next_back()
        .filter(|(base, (size, _, _))| *base + size > at)
        .map(|(base, _)| *base);
    let starts = first.into_iter().chain(physical.views.range(at.saturating_add(1)..end).map(|(b, _)| *b));
    for base in starts.collect::<Vec<_>>() {
        let (size, slice, _) = &physical.views[&base];
        let (from, to) = (at.max(base), end.min(base + size));
        for page in (from / PAGE..to.div_ceil(PAGE)).map(|p| p * PAGE) {
            let k = (page - base) / PAGE;
            if let Some(addresses) = physical.shown.get(&slice.key(k)) {
                out.extend(addresses.iter().filter(|&&a| a != page).map(|&a| (page, a)));
            }
        }
    }
}

/// The addresses of title views showing `pages` (page numbers): writes into
/// them through our own views must be reported there.
fn shown_at(pages: &[usize]) -> Vec<usize> {
    let physical = physical();
    pages
        .iter()
        .filter_map(|p| physical.pages.get(p))
        .filter_map(|slice| physical.shown.get(&slice.key(0)))
        .flatten()
        .copied()
        .collect()
}

/// Reports writes into physical pages through our own views at every title
/// view showing them (DirectStorage's page destinations).
fn report_written(shown: &[usize]) {
    for &page in shown {
        #[cfg(test)]
        REPORTED.with(|r| r.borrow_mut().push(page));
        drop(victory_gpu::kernel_will_write(page, PAGE));
    }
}

/// A CPU write through the section's own view. Its title aliases are kept
/// busy for write tracking, including any mapped during the decode.
struct SectionWrite {
    slice: Slice,
    pages: usize,
    guards: HashMap<usize, victory_gpu::KernelWrite>,
}

impl SectionWrite {
    fn covers(&self, section: &Arc<Section>, index: usize) -> bool {
        Arc::ptr_eq(&self.slice.section, section)
            && (self.slice.index..self.slice.index + self.pages).contains(&index)
    }

    fn overlaps(&self, slice: &Slice, count: usize) -> bool {
        Arc::ptr_eq(&self.slice.section, &slice.section)
            && self.slice.index < slice.index + count
            && slice.index < self.slice.index + self.pages
    }

    fn aliases<'a>(&'a self, physical: &'a Physical) -> impl Iterator<Item = usize> + 'a {
        (0..self.pages)
            .filter_map(|k| physical.shown.get(&self.slice.key(k)))
            .flatten()
            .copied()
    }
}

static PHYSICAL_WRITTEN: Condvar = Condvar::new();

struct PhysicalWrite(usize);

impl PhysicalWrite {
    fn begin(slice: &Slice, pages: usize) -> Self {
        let mut physical = physical();
        while physical.writes.values().any(|w| w.overlaps(slice, pages)) {
            physical = PHYSICAL_WRITTEN
                .wait(physical)
                .unwrap_or_else(|e| e.into_inner());
        }
        let mut write = SectionWrite {
            slice: slice.clone(),
            pages,
            guards: HashMap::new(),
        };
        write.guards = write
            .aliases(&physical)
            .map(|at| (at, victory_gpu::kernel_will_write_beside(at, PAGE)))
            .collect();
        let id = physical.next_write;
        physical.next_write += 1;
        physical.writes.insert(id, write);
        Self(id)
    }
}

impl Drop for PhysicalWrite {
    fn drop(&mut self) {
        let mut physical = physical();
        let write = physical.writes.remove(&self.0).expect("live section write");
        #[cfg(test)]
        REPORTED.with(|r| r.borrow_mut().extend(write.aliases(&physical)));
        drop(write);
        PHYSICAL_WRITTEN.notify_all();
    }
}

/// Runs a GPU merge with the current physical view pinned. Waiting for an
/// overlapping DirectStorage decode releases the map lock. A replacement
/// may occur during that wait, so check the mapping again afterwards.
pub(crate) fn landing_page(
    at: usize,
    len: usize,
    epoch: u64,
    beside: bool,
    apply: &mut dyn FnMut(victory_gpu::LandingPage, usize),
) {
    use victory_gpu::LandingPage;
    let mut physical = physical();
    let end = at + len;
    loop {
        let following = || physical.views.range(at + 1..end);
        let Some((&base, (size, slice, mapped))) = physical
            .views
            .range(..=at)
            .next_back()
            .filter(|(base, (size, _, _))| at < *base + size)
        else {
            let stop = following().next().map_or(end, |(&base, _)| base);
            apply(LandingPage::Direct, stop - at);
            return;
        };
        let mut stop = end.min(base + size);
        if *mapped > epoch {
            apply(LandingPage::Replaced, stop - at);
            return;
        }
        // XMem alias operations often produce consecutive single-page views.
        // They can share the pin when their slices are contiguous and all
        // still belong to this submission. Their title protections are separate.
        for (&next, (size, adjacent, mapped)) in following() {
            if next != stop
                || *mapped > epoch
                || !Arc::ptr_eq(&slice.section, &adjacent.section)
                || adjacent.index != slice.index + (next - base) / PAGE
            {
                break;
            }
            stop = end.min(next + size);
        }
        let within = at - base;
        let first = slice.index + within / PAGE;
        let last = slice.index + (stop - base).div_ceil(PAGE);
        let busy = physical
            .writes
            .values()
            .filter_map(|w| {
                let index = w.slice.index.max(first);
                (index < last && w.covers(&slice.section, index)).then_some(index)
            })
            .min();
        if busy == Some(first) {
            physical = PHYSICAL_WRITTEN
                .wait(physical)
                .unwrap_or_else(|e| e.into_inner());
            continue;
        }
        if let Some(index) = busy {
            stop = base + (index - slice.index) * PAGE;
        }
        let page = if beside {
            slice
                .section
                .whole()
                .map(|base| base + slice.index * PAGE + within)
        } else {
            None
        };
        apply(
            page.map_or(LandingPage::Direct, LandingPage::Beside),
            stop - at,
        );
        return;
    }
}

#[cfg(test)]
thread_local! {
    /// The title addresses this thread's writes into physical pages reported.
    static REPORTED: std::cell::RefCell<Vec<usize>> = const { std::cell::RefCell::new(Vec::new()) };
}

static PHYSICAL: LazyLock<Mutex<Physical>> = LazyLock::new(|| {
    Mutex::new(Physical {
        // Page numbers are opaque to titles; 0 stays unused.
        next_page: 0x10_0000,
        ..Default::default()
    })
});

fn physical() -> MutexGuard<'static, Physical> {
    PHYSICAL.lock().unwrap_or_else(|e| e.into_inner())
}

/// `count` is in and out (pages wanted, pages given); all or nothing here.
pub extern "system" fn XMemAllocatePhysicalPages(
    flags: u64,
    count: *mut usize,
    pages: *mut usize,
) -> BOOL {
    allocate_physical_pages(flags, count, pages, victory_gpu::physical_heap)
}

fn allocate_physical_pages(
    flags: u64,
    count: *mut usize,
    pages: *mut usize,
    heap: fn(usize) -> Option<victory_gpu::PhysicalHeap>,
) -> BOOL {
    if count.is_null() || pages.is_null() {
        return 0;
    }
    // SAFETY: checked non-null.
    let n = unsafe { *count };
    let Some(size) = n.checked_mul(PAGE).filter(|&s| s > 0) else {
        return 0;
    };
    // A pagefile-backed section of `size` bytes, its pages committed when
    // first mapped or written (`commit`): Grand Theft Auto VI asks for 6.1
    // GiB at once, which a busy host often cannot commit (error 1450).
    // SAFETY: plain section creation.
    let handle = unsafe {
        CreateFileMappingW(
            INVALID_HANDLE_VALUE,
            std::ptr::null(),
            // Under Wine a committed section is a sparse file, free until
            // touched, and committing a reserved one page by page costs a
            // wineserver request each that scans every range committed so
            // far: commit it whole there.
            PAGE_READWRITE | if on_wine() { SEC_COMMIT } else { SEC_RESERVE },
            (size as u64 >> 32) as u32,
            size as u32,
            std::ptr::null(),
        )
    };
    if handle.is_null() {
        let error = std::io::Error::last_os_error();
        tracing::warn!(n, %error, "XMemAllocatePhysicalPages failed");
        // SAFETY: checked non-null.
        unsafe { count.write(0) };
        return 0;
    }
    tracing::debug!(
        n,
        flags = format_args!("{flags:#x}"),
        "XMemAllocatePhysicalPages"
    );
    let section = Arc::new(Section {
        handle,
        size,
        shared: heap(size).map(Arc::new),
        whole: OnceLock::new(),
    });
    let mut physical = physical();
    let first = physical.next_page;
    physical.next_page += n * FRAMES;
    for index in 0..n {
        let slice = Slice {
            section: section.clone(),
            index,
        };
        let number = first + index * FRAMES;
        physical.pages.insert(number, slice);
        // SAFETY: the title passes room for `n` page numbers.
        unsafe { pages.add(index).write(number) };
    }
    1
}

/// Page numbers count 4 KiB frames, so a 64 KiB page's is a multiple of
/// 16: Grand Theft Auto VI keeps them as 21-bit `number >> 4` and rebuilds
/// them with `<< 4` (128 GiB of pages at most).
const FRAMES: usize = PAGE >> 12;

/// How many of `pages` are no allocated page (diagnostics: a DirectStorage
/// destination that could not be written).
pub(crate) fn unknown_pages(pages: &[usize]) -> usize {
    let physical = physical();
    pages.iter().filter(|p| !physical.pages.contains_key(p)).count()
}

/// Pages still mapped stay readable through their views until unmapped.
pub extern "system" fn XMemFreePhysicalPages(n: usize, pages: *const usize) -> BOOL {
    if pages.is_null() {
        return 0;
    }
    victory_gpu::with_inflight_mapping(|| {
        let mut physical = physical();
        for i in 0..n {
            // SAFETY: the title passes `n` page numbers.
            physical.pages.remove(&unsafe { *pages.add(i) });
        }
        1
    })
}

/// Consecutive slices of one section, committed and writable: through our
/// view of the whole section, or through a view of their own if the host
/// would not map all of it, unmapped when dropped.
struct View {
    at: usize,
    len: usize,
    /// The view this maps for itself, if any.
    own: Option<MEMORY_MAPPED_VIEW_ADDRESS>,
    /// Keeps the section, and with it the whole view, while this lives.
    _section: Arc<Section>,
}

impl View {
    /// `n` slices from `slice` on.
    fn map(slice: &Slice, n: usize) -> Option<View> {
        use windows_sys::Win32::System::Memory::{FILE_MAP_WRITE, MapViewOfFile};
        let (offset, len) = ((slice.index * PAGE) as u64, n * PAGE);
        let section = slice.section.clone();
        if let Some(whole) = section.whole() {
            let at = whole + offset as usize;
            return commit(at, len).then_some(View {
                at,
                len,
                own: None,
                _section: section,
            });
        }
        // SAFETY: slices of a section we own; the view is ours until dropped.
        let own = unsafe {
            MapViewOfFile(
                section.handle,
                FILE_MAP_WRITE,
                (offset >> 32) as u32,
                offset as u32,
                len,
            )
        };
        if own.Value.is_null() {
            return None;
        }
        let view = View {
            at: own.Value as usize,
            len,
            own: Some(own),
            _section: section,
        };
        commit(view.at, len).then_some(view)
    }

    fn bytes(&mut self) -> &mut [u8] {
        // SAFETY: the view holds `len` committed read-write bytes, ours
        // alone while `self` lives (other views of the slices only alias
        // them, as the console's do).
        unsafe { std::slice::from_raw_parts_mut(self.at as *mut u8, self.len) }
    }
}

impl Drop for View {
    fn drop(&mut self) {
        use windows_sys::Win32::System::Memory::UnmapViewOfFile;
        if let Some(own) = self.own {
            // SAFETY: the view this mapped.
            unsafe { UnmapViewOfFile(own) };
        }
    }
}

/// The pages behind bytes `[offset, offset + len)` of `pages` (64 KiB
/// each), as runs of consecutive slices of one section: each run's first
/// slice and its length in pages. One look at the page table, which the
/// title's own mapping calls share with every DirectStorage worker. `None`
/// if a page is unknown or `pages` ends early.
fn runs(pages: &[usize], offset: usize, len: usize) -> Option<Vec<(Slice, usize)>> {
    let end = offset.checked_add(len)?;
    let pages = pages.get(offset / PAGE..end.div_ceil(PAGE))?;
    let physical = physical();
    let mut runs: Vec<(Slice, usize)> = Vec::new();
    for page in pages {
        let slice = physical.pages.get(page)?;
        match runs.last_mut() {
            Some((first, n))
                if Arc::ptr_eq(&first.section, &slice.section)
                    && slice.index == first.index + *n =>
            {
                *n += 1;
            }
            _ => runs.push((slice.clone(), 1)),
        }
    }
    Some(runs)
}

/// Copies `data` into physical pages, from `offset` bytes into the first,
/// whether or not they are mapped anywhere (DirectStorage destinations).
/// Through each section's whole view ([`View`]), one run of consecutive
/// pages at a time. False if a page is unknown or the host fails.
pub(crate) fn write_physical(pages: &[usize], offset: usize, data: &[u8]) -> bool {
    let Some(runs) = runs(pages, offset, data.len()) else {
        return false;
    };
    let landing = victory_gpu::landing_enabled();
    let (mut within, mut done) = (offset % PAGE, 0);
    for (slice, n) in &runs {
        let Some(mut view) = View::map(slice, *n) else {
            return false;
        };
        let _writing = landing.then(|| PhysicalWrite::begin(slice, *n));
        let k = (n * PAGE - within).min(data.len() - done);
        view.bytes()[within..within + k].copy_from_slice(&data[done..done + k]);
        (within, done) = (0, done + k);
    }
    if !landing {
        report_written(&shown_at(
            pages
                .get(offset / PAGE..(offset + data.len()).div_ceil(PAGE))
                .unwrap_or(&[]),
        ));
    }
    true
}

/// Bytes `[offset, offset + len)` of physical pages, writable in place
/// until dropped: DirectStorage decodes into it instead of a buffer it
/// would then copy.
pub(crate) struct PageWindow {
    view: View,
    within: usize,
    len: usize,
    writing: Option<PhysicalWrite>,
    /// When landing is disabled, report the views showing these pages at
    /// the end of the decode, including views mapped during it.
    pages: Vec<usize>,
}

impl Drop for PageWindow {
    fn drop(&mut self) {
        if self.writing.is_none() {
            report_written(&shown_at(&self.pages));
        }
    }
}

impl PageWindow {
    pub(crate) fn bytes(&mut self) -> &mut [u8] {
        let (within, len) = (self.within, self.len);
        &mut self.view.bytes()[within..within + len]
    }
}

/// A window on bytes `[offset, offset + len)` of `pages` when they are
/// consecutive pages of one allocation (how titles usually hand them out);
/// `None` otherwise (pages of several allocations, out of order, or
/// unknown, or a host failure), which `write_physical` then handles or
/// reports.
pub(crate) fn page_window(pages: &[usize], offset: usize, len: usize) -> Option<PageWindow> {
    let runs = runs(pages, offset, len)?;
    let [(slice, n)] = runs.as_slice() else {
        return None;
    };
    let view = View::map(slice, *n)?;
    let landing = victory_gpu::landing_enabled();
    Some(PageWindow {
        view,
        within: offset % PAGE,
        len,
        writing: landing.then(|| PhysicalWrite::begin(slice, *n)),
        pages: if landing {
            Vec::new()
        } else {
            pages
                .get(offset / PAGE..(offset + len).div_ceil(PAGE))
                .unwrap_or(&[])
                .to_vec()
        },
    })
}

/// Maps `n` pages at `address`, in place of what is mapped there; null
/// `pages` unmaps. Returns `address`, or null if a page or the host fails.
pub extern "system" fn XMemMapPhysicalPages(
    address: *mut c_void,
    n: usize,
    pages: *const usize,
) -> *mut c_void {
    let base = address as usize;
    let Some(len) = n
        .checked_mul(PAGE)
        .filter(|&len| base.checked_add(len).is_some())
    else {
        return std::ptr::null_mut();
    };
    if n == 0 {
        return address;
    }
    if !base.is_multiple_of(PAGE) {
        return std::ptr::null_mut();
    }
    // SAFETY: the title passes a null pointer or `n` page numbers.
    let pages = (!pages.is_null()).then(|| unsafe { std::slice::from_raw_parts(pages, n) });
    victory_gpu::with_physical_mapping(
        |labels| {
            let physical = physical();
            match pages {
                None => overlaps_view(&physical, base, len),
                Some(pages) => {
                    if overlaps_view(&physical, base, len) {
                        return !same_physical_pages(&physical, base, pages);
                    }
                    // File pages can demote a whole tiled chunk, even
                    // when the destination itself is a placeholder.
                    pages.iter().any(|page| {
                        physical
                            .pages
                            .get(page)
                            .is_none_or(|s| s.section.shared.is_none())
                    }) || mapping_touches_labels(&physical, base, len, labels)
                }
            }
        },
        || {
            if map_physical(&mut physical(), base, len, pages) {
                address
            } else {
                std::ptr::null_mut()
            }
        },
    )
}

fn map_physical(physical: &mut Physical, base: usize, len: usize, pages: Option<&[usize]>) -> bool {
    let Some(pages) = pages else {
        return unmap(physical, base, len);
    };
    if same_physical_pages(physical, base, pages) {
        return true;
    }
    // Validate the complete request before demoting a tiled chunk or
    // removing any of the existing CPU/GPU mappings.
    if let Some(page) = pages.iter().find(|page| !physical.pages.contains_key(page)) {
        tracing::warn!(page, "XMemMapPhysicalPages: unknown page");
        return false;
    }
    if !physical.reservations.is_empty() {
        for (i, page) in pages.iter().enumerate() {
            if let Some(slice) = physical.pages.get(page)
                && slice.section.shared.is_none()
                && !physical.admit_file(base + i * PAGE)
            {
                tracing::warn!(
                    base,
                    "file-backed page cannot replace a published tiled buffer"
                );
                return false;
            }
        }
    }
    if !unmap(physical, base, len) {
        return false;
    }
    let mut i = 0;
    while i < pages.len() {
        let first = physical.pages[&pages[i]].clone();
        let run = 1 + pages[i + 1..]
            .iter()
            .zip(1..)
            .take_while(|&(page, k)| {
                physical.pages.get(page).is_some_and(|s| {
                    Arc::ptr_eq(&s.section, &first.section) && s.index == first.index + k
                })
            })
            .count();
        if !map(physical, base + i * PAGE, run * PAGE, first) {
            return false;
        }
        i += run;
    }
    true
}

fn same_physical_pages(physical: &Physical, base: usize, pages: &[usize]) -> bool {
    pages.iter().zip((base..).step_by(PAGE)).all(|(page, at)| {
        physical
            .pages
            .get(page)
            .is_some_and(|slice| shows(physical, at, slice))
    })
}

/// One page at each address; null `pages`, or a page number 0, unmaps
/// (Grand Theft Auto VI passes 0 for addresses to clear).
pub extern "system" fn XMemMapPhysicalPagesScatter(
    addresses: *const *mut c_void,
    n: usize,
    pages: *const usize,
) -> BOOL {
    if addresses.is_null() {
        return 0;
    }
    for i in 0..n {
        // SAFETY: the title passes `n` addresses and, unless null, `n` pages.
        let (address, page) = unsafe {
            (
                *addresses.add(i),
                if pages.is_null() || *pages.add(i) == 0 {
                    std::ptr::null()
                } else {
                    pages.add(i)
                },
            )
        };
        if XMemMapPhysicalPages(address, 1, page).is_null() {
            return 0;
        }
    }
    1
}

/// Maps the physical pages shown at `[from, from + len)` at `to` as well,
/// in place of what `to` shows; `from` 0 unmaps. The GPU's page-mapping
/// copies (`victory_gpu`, queue `CopyPageMappingsBatchX`): Grand Theft Auto
/// VI maps pages into a pool view, then has the GPU map them into its tiled
/// heap. Alias destinations remain one host view per page so that remapping
/// one of them does not temporarily unmap its neighbours.
pub fn map_alias(to: u64, from: u64, len: u64) -> bool {
    let (to, from, len) = (to as usize, from as usize, len as usize);
    if len == 0 {
        return true;
    }
    if !(to | from | len).is_multiple_of(PAGE) {
        return false;
    }
    if to.checked_add(len).is_none() || from.checked_add(len).is_none() {
        return false;
    }
    victory_gpu::with_physical_mapping(
        |labels| {
            let physical = physical();
            if from == 0 {
                return overlaps_view(&physical, to, len);
            }
            if (from..from + len).step_by(PAGE).any(|at| {
                slice_at(&physical, at).is_none_or(|slice| slice.section.shared.is_none())
            }) {
                return true;
            }
            if overlaps_view(&physical, to, len) {
                !same_alias_pages(&physical, to, from, len)
            } else {
                mapping_touches_labels(&physical, to, len, labels)
            }
        },
        || alias_pages(&mut physical(), to, from, len),
    )
}

/// An additive edit can make a pending write visible through a new alias.
/// If a label is already in any physical view, keep the publication fence
/// before changing the aliases, even when the destination is disjoint.
fn mapping_touches_labels(
    physical: &Physical,
    base: usize,
    len: usize,
    labels: &mut dyn Iterator<Item = (usize, usize)>,
) -> bool {
    for (at, bytes) in labels {
        if at
            .checked_add(bytes)
            .is_none_or(|end| (at < base + len && base < end) || overlaps_view(physical, at, bytes))
        {
            return true;
        }
    }
    false
}

fn overlaps_view(physical: &Physical, base: usize, len: usize) -> bool {
    physical
        .views
        .range(..base + len)
        .next_back()
        .is_some_and(|(&start, (size, ..))| start + size > base)
}

fn alias_pages(physical: &mut Physical, to: usize, from: usize, len: usize) -> bool {
    if from != 0 && !physical.reservations.is_empty() {
        for k in (0..len).step_by(PAGE) {
            if let Some(slice) = slice_at(physical, from + k)
                && slice.section.shared.is_none()
                && !physical.admit_file(to + k)
            {
                return false;
            }
        }
    }
    if len > PAGE
        && from != 0
        && to.abs_diff(from) >= len
        && (to | from | len).is_multiple_of(PAGE)
        && !has_multi_page_view(physical, to, len)
    {
        return alias_changed_pages(physical, to, from, len);
    }
    // A direct XMem mapping can span multiple pages in one host view.
    // Keep the original single unmap of the range in that case, as well as
    // for overlapping, unaligned, and unmap-only requests.
    if from != 0 && same_alias_pages(physical, to, from, len) {
        return true;
    }
    if !unmap(physical, to, len) {
        return false;
    }
    for k in (0..len).step_by(PAGE) {
        if from == 0 {
            break;
        }
        let Some(slice) = slice_at(physical, from + k) else {
            return false;
        };
        if !map(physical, to + k, PAGE, slice) {
            return false;
        }
    }
    true
}

fn same_alias_pages(physical: &Physical, to: usize, from: usize, len: usize) -> bool {
    (0..len)
        .step_by(PAGE)
        .all(|k| slice_at(physical, from + k).is_some_and(|slice| shows(physical, to + k, &slice)))
}

fn has_multi_page_view(physical: &Physical, base: usize, len: usize) -> bool {
    if len == 0 {
        return false;
    }
    let first = physical
        .views
        .range(..=base)
        .next_back()
        .filter(|(at, (size, _, _))| *at + size > base);
    first
        .into_iter()
        .chain(physical.views.range(base + 1..base + len))
        .any(|(_, (size, _, _))| *size > PAGE)
}

fn alias_changed_pages(physical: &mut Physical, to: usize, from: usize, len: usize) -> bool {
    let end = to + len;
    for k in (0..len).step_by(PAGE) {
        let at = to + k;
        let Some(slice) = slice_at(physical, from + k) else {
            unmap(physical, at, end - at);
            return false;
        };
        if let Some(current) = slice_at(physical, at) {
            if current.key(0) == slice.key(0) {
                continue;
            }
            if !unmap(physical, at, PAGE) {
                return false;
            }
        }
        if !map(physical, at, PAGE, slice) {
            unmap(physical, at + PAGE, end - at - PAGE);
            return false;
        }
    }
    true
}

/// The page a title view shows at `at`, if any.
fn slice_at(physical: &Physical, at: usize) -> Option<Slice> {
    physical
        .views
        .range(..=at)
        .next_back()
        .filter(|(base, (size, _, _))| at < *base + size)
        .map(|(base, (_, s, _))| Slice {
            index: s.index + (at - base) / PAGE,
            section: s.section.clone(),
        })
}

/// Whether a title view shows `slice` at `at` already. Mapping it there
/// again would change nothing, and would cost an unmap and a map, which
/// under Wine each hold the process's one virtual-memory lock across a
/// wineserver request: every page fault and protection change on other
/// threads waits for them (#50).
fn shows(physical: &Physical, at: usize, slice: &Slice) -> bool {
    slice_at(physical, at).is_some_and(|s| s.key(0) == slice.key(0))
}

/// Maps `slice` and the slices after it over `[base, base + len)`, which
/// must lie in one placeholder.
fn map(physical: &mut Physical, base: usize, len: usize, slice: Slice) -> bool {
    map_view(physical, base, len, slice, None)
}

fn map_view(
    physical: &mut Physical,
    base: usize,
    len: usize,
    slice: Slice,
    epoch: Option<u64>,
) -> bool {
    #[cfg(test)]
    tests::MAPS.with(|m| m.set(m.get() + 1));
    if slice.section.shared.is_none()
        && !physical.reservations.is_empty()
        && !(base..base + len)
            .step_by(PAGE)
            .all(|at| physical.admit_file(at))
    {
        return false;
    }
    // SAFETY: splits the placeholder around the range (failing harmlessly
    // when the range is the whole placeholder), then maps the section
    // there; both act only on placeholders.
    let view = unsafe {
        VirtualFree(
            base as *mut c_void,
            len,
            MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER,
        );
        MapViewOfFile3(
            slice.section.handle,
            GetCurrentProcess(),
            base as *const c_void,
            (slice.index * PAGE) as u64,
            len,
            MEM_REPLACE_PLACEHOLDER,
            PAGE_READWRITE,
            std::ptr::null_mut(),
            0,
        )
    };
    if view.Value as usize != base {
        tracing::warn!(
            address = format_args!("{base:#x}"),
            len,
            error = %std::io::Error::last_os_error(),
            "mapping physical pages failed (not a placeholder?)"
        );
        return false;
    }
    let mapped = commit(base, len);
    let mapped = mapped
        && slice.section.shared.as_ref().is_none_or(|shared| {
            // SAFETY: this live view is ours. The slice will hold the
            // backing heap until the view is unmapped.
            let aliased = unsafe { shared.alias_view(base, slice.index * PAGE, len) };
            if !aliased {
                tracing::warn!(
                    address = format_args!("{base:#x}"),
                    len,
                    "sharing the mapped XMem view with the host heap failed"
                );
            }
            aliased
        });
    if !mapped {
        // SAFETY: the view just mapped, back to a placeholder.
        unsafe { UnmapViewOfFile2(GetCurrentProcess(), view, MEM_PRESERVE_PLACEHOLDER) };
        return false;
    }
    // The host view is read-write. A split keeps tickets for the same
    // physical pages, even though the GPU's guard links must be rebuilt.
    if epoch.is_some() {
        victory_gpu::title_memory_restored(base, len);
    } else {
        victory_gpu::title_memory_remapped(base, len);
    }
    let epoch = epoch.unwrap_or_else(victory_gpu::next_mapping_epoch);
    remember_view(physical, base, len, slice, epoch);
    true
}

fn remember_view(physical: &mut Physical, base: usize, len: usize, slice: Slice, epoch: u64) {
    for k in 0..len / PAGE {
        let at = base + k * PAGE;
        for write in physical.writes.values_mut() {
            if write.covers(&slice.section, slice.index + k) {
                write
                    .guards
                    .insert(at, victory_gpu::kernel_will_write_beside(at, PAGE));
            }
        }
        let shown = physical.shown.entry(slice.key(k)).or_default();
        // Pages shown elsewhere already: guarded here too if they are there.
        victory_gpu::title_memory_aliased(at, shown);
        shown.push(at);
    }
    physical.update_tiles(base, len, Some(&slice));
    physical.views.insert(base, (len, slice, epoch));
}

/// Commits the section pages a view shows (once per page, whichever view).
fn commit(base: usize, len: usize) -> bool {
    if on_wine() {
        return true; // sections are committed whole there
    }
    // SAFETY: `[base, base + len)` is a view of a reserved section.
    if !unsafe { VirtualAlloc(base as *const c_void, len, MEM_COMMIT, PAGE_READWRITE) }.is_null() {
        return true;
    }
    commit_refused(len, "physical pages", std::io::Error::last_os_error());
    false
}

/// Reports memory the host would not give: at the Windows commit limit to
/// `crashes.log`, since the title usually faults soon after on the null it
/// gets; otherwise as a warning.
fn commit_refused(len: usize, what: &str, error: std::io::Error) {
    if error.raw_os_error() == Some(ERROR_COMMITMENT_LIMIT as i32) {
        victory_loader::log_crash(&commit_limit_message(
            len,
            what,
            &error,
            system_commit().as_ref(),
        ));
    } else {
        tracing::warn!(len, %error, "committing {what} failed");
    }
}

/// Reports to `crashes.log` that the host allocator refused `len_bytes` of
/// APU heap memory. There is no Windows error code to give: `std::alloc`
/// allocates with `HeapAlloc`, which does not set one when it fails.
pub(crate) fn report_allocator_refusal(len_bytes: usize) {
    victory_loader::log_crash(&allocator_refusal_message(
        len_bytes,
        system_commit().as_ref(),
    ));
}

/// The system commit, if Windows reports it.
fn system_commit() -> Option<PERFORMANCE_INFORMATION> {
    let mut info = PERFORMANCE_INFORMATION {
        cb: size_of::<PERFORMANCE_INFORMATION>() as u32,
        ..Default::default()
    };
    // SAFETY: the buffer and size describe a writable PERFORMANCE_INFORMATION.
    (unsafe { K32GetPerformanceInfo(&mut info, info.cb) } != 0).then_some(info)
}

fn commit_limit_message(
    len: usize,
    what: &str,
    error: &std::io::Error,
    info: Option<&PERFORMANCE_INFORMATION>,
) -> String {
    refusal_message(
        format_args!(
            "Windows commit limit reached while committing {len} bytes of {what}: {error}"
        ),
        info,
    )
}

fn allocator_refusal_message(len_bytes: usize, info: Option<&PERFORMANCE_INFORMATION>) -> String {
    refusal_message(
        format_args!("the host allocator refused {len_bytes} bytes of APU heap memory"),
        info,
    )
}

/// What the host refused, then the system commit and what the user can do.
fn refusal_message(
    refusal: std::fmt::Arguments<'_>,
    info: Option<&PERFORMANCE_INFORMATION>,
) -> String {
    let mut message = format!("VIctory: {refusal}.\n");
    if let Some(info) = info {
        use std::fmt::Write;

        let committed = info.CommitTotal * info.PageSize / MIB;
        let limit = info.CommitLimit * info.PageSize / MIB;
        let _ = writeln!(
            message,
            "System commit: {committed} MiB of {limit} MiB in use."
        );
    }
    message.push_str(
        "Close other applications or increase the Windows paging file size, then retry.\n",
    );
    message
}

/// Returns `[base, base + len)` to placeholders; views reaching past it
/// are mapped again outside it.
fn unmap(physical: &mut Physical, base: usize, len: usize) -> bool {
    // No bytes: no view to return. The search below would still find the
    // view holding `base` and cut it in two there, and its range of views
    // starting inside would run backwards (which `BTreeMap` panics on).
    if len == 0 {
        return true;
    }
    let end = base + len;
    let mut restored = true;
    // Views do not overlap: the one holding `base`, then those starting
    // inside (the GPU's page mappings make one view per page).
    let first = physical
        .views
        .range(..=base)
        .next_back()
        .filter(|(at, (size, _, _))| *at + size > base)
        .map(|(at, _)| *at);
    let hit: Vec<usize> = first
        .into_iter()
        .chain(physical.views.range(base + 1..end).map(|(at, _)| *at))
        .collect();
    for at in hit {
        let Some((size, slice, epoch)) = physical.views.remove(&at) else {
            continue;
        };
        for k in 0..size / PAGE {
            let page = at + k * PAGE;
            for write in physical.writes.values_mut() {
                write.guards.remove(&page);
            }
            let key = slice.key(k);
            if let Some(addresses) = physical.shown.get_mut(&key) {
                addresses.retain(|&a| a != at + k * PAGE);
                if addresses.is_empty() {
                    physical.shown.remove(&key);
                }
            }
        }
        if victory_gpu::landing_enabled() {
            if at < base {
                victory_gpu::title_memory_moving(at, base - at);
            }
            if at + size > end {
                victory_gpu::title_memory_moving(end, at + size - end);
            }
        }
        let end_of_view = at + size;
        let split = if at < base || end_of_view > end {
            let removed = at.max(base);
            Some(victory_gpu::title_memory_splitting(
                at,
                size,
                removed,
                end_of_view.min(end) - removed,
            ))
        } else {
            victory_gpu::title_memory_remapped(at, size);
            None
        };
        // SAFETY: a view we mapped; it becomes a placeholder again.
        let unmapped = unsafe {
            UnmapViewOfFile2(
                GetCurrentProcess(),
                MEMORY_MAPPED_VIEW_ADDRESS {
                    Value: at as *mut c_void,
                },
                MEM_PRESERVE_PLACEHOLDER,
            )
        };
        if unmapped == 0 {
            let error = std::io::Error::last_os_error();
            tracing::warn!(at, size, %error, "unmapping physical pages failed");
            remember_view(physical, at, size, slice, epoch);
            return false;
        }
        let removed = at.max(base);
        physical.update_tiles(removed, end_of_view.min(end) - removed, None);
        if let Some(mut split) = split {
            if at < base && !map_view(physical, at, base - at, slice.clone(), Some(epoch)) {
                split.discard(at, base - at);
                physical.update_tiles(at, base - at, None);
                restored = false;
            }
            if end_of_view > end {
                let index = slice.index + (end - at) / PAGE;
                let tail = end_of_view - end;
                if !map_view(physical, end, tail, Slice { index, ..slice }, Some(epoch)) {
                    split.discard(end, tail);
                    physical.update_tiles(end, tail, None);
                    restored = false;
                }
            }
        }
    }
    restored
}

/// The oldest layout we know; newer titles pass other flag values (Minecraft
/// 0x20000001) and read the same leading fields.
#[repr(C)]
#[derive(Debug, Default)]
pub struct WorkingSet {
    pub title_limit: usize,
    pub title_used: usize,
    pub tools_limit: usize,
    pub tools_used: usize,
    pub stack_used: u32,
    pub physical_pages: u32,
    pub gpu_optimal_limit: usize,
    pub gpu_optimal_used: usize,
    pub tools_gpu_optimal_limit: usize,
    pub tools_gpu_optimal_used: usize,
    pub kernel_pool_used: u32,
    pub kernel_pool_limit: u32,
}

/// The title's budget and our process's commit as what it uses. No tools
/// memory: a retail console.
// shortcut: all fields are filled whatever the flags ask for, and use is
// not split by bandwidth class (GPU-optimal use reads 0); track
// allocations per class if a title budgets from that field.
pub extern "system" fn XMemGetWorkingSetStatistics(_which: u32, out: *mut WorkingSet) -> BOOL {
    if out.is_null() {
        return 0;
    }
    let mut counters = PROCESS_MEMORY_COUNTERS {
        cb: size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
        ..Default::default()
    };
    // SAFETY: our own process and a sized structure.
    unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, counters.cb) };
    let (gpu_optimal, standard) = victory_gpu::console::current().title_memory();
    let stats = WorkingSet {
        title_limit: gpu_optimal + standard - TRANSFERRED.load(Ordering::Relaxed),
        title_used: counters.PagefileUsage,
        gpu_optimal_limit: gpu_optimal,
        ..Default::default()
    };
    // SAFETY: checked non-null; the title passes room for the structure.
    unsafe { out.write(stats) };
    1
}

/// Hands title memory (standard, in 2 MiB pages) to the system. The other
/// transfers involve tools memory, which a retail console has none of.
pub extern "system" fn XMemTransferMemory(kind: u32, size: usize) -> BOOL {
    const TITLE_TO_SYSTEM: u32 = 0;
    tracing::info!(kind, size, "XMemTransferMemory");
    if size == 0 {
        return 1;
    }
    let Some(size) = size.checked_next_multiple_of(2 * MIB) else {
        return 0;
    };
    let standard = victory_gpu::console::current().title_memory().1;
    let moved = kind == TITLE_TO_SYSTEM
        && TRANSFERRED
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |t| {
                t.checked_add(size).filter(|&t| t <= standard)
            })
            .is_ok();
    moved.into()
}

pub(crate) fn module() -> HleModule {
    HleModule {
        name: "xmem",
        dlls: &["xmem.dll"],
        exports: vec![
            hle_export!(XMemAlloc),
            hle_export!(XMemFree),
            hle_export!(XMemAllocDefault),
            hle_export!(XMemFreeDefault),
            hle_export!(XMemIsToolsMemAvailable),
            hle_export!(XMemAllocatePhysicalPages),
            hle_export!(XMemFreePhysicalPages),
            hle_export!(XMemMapPhysicalPages),
            hle_export!(XMemMapPhysicalPagesScatter),
            hle_export!(XMemVirtualAlloc),
            hle_export!(XMemVirtualQuery),
            hle_export!(XMemGetWorkingSetStatistics),
            hle_export!(XMemTransferMemory),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn query(address: *const c_void) -> BasicInformation {
        // SAFETY: plain data; zero is a valid value.
        let mut info: BasicInformation = unsafe { std::mem::zeroed() };
        // SAFETY: a pseudo handle needs no closing.
        let process = unsafe { GetCurrentProcess() };
        assert_eq!(XMemVirtualQuery(process, address, &mut info), 1);
        info
    }

    #[test]
    fn alloc_honours_alignment_and_frees() {
        for (field, align) in [(0, 16), (4, 16), (0xc, 4096), (0x10, 64 * 1024)] {
            let attributes = field << ALIGNMENT_SHIFT | 1 << 29; // a graphics type
            let p = XMemAlloc(100, attributes);
            assert!(!p.is_null());
            assert_eq!(p as usize % align, 0, "field {field:#x}");
            // SAFETY: 100 bytes were allocated.
            unsafe { p.cast::<u8>().write_bytes(0xab, 100) };
            XMemFree(p, attributes);
        }
        let p = XMemAlloc(4096, ALLOC_32BIT);
        assert!(!p.is_null() && (p as usize) < 1 << 32);
        XMemFree(p, ALLOC_32BIT);
    }

    #[test]
    fn virtual_alloc_meets_console_constraints() {
        const GPU_READ_WRITE: u32 = 0x2000;
        let p = XMemVirtualAlloc(
            std::ptr::null_mut(),
            4 * MIB,
            PAGES_2M | MEM_RESERVE | MEM_COMMIT,
            GRAPHICS | 7 << 12, // 40-bit addresses
            PAGE_READWRITE | GPU_READ_WRITE,
        );
        assert!(!p.is_null());
        assert_eq!(p as usize % (2 * MIB), 0);
        assert!((p as usize) < 1 << 40);
        let info = query(p);
        assert_eq!(info.host.State, MEM_COMMIT);
        assert_eq!(info.host.Protect, PAGE_READWRITE);
        assert_eq!(info.flags & GRAPHICS, GRAPHICS);
        // Write-watched, for the GPU side's copies.
        let (mut found, mut count, mut granularity) = ([0usize; 4], 4, 0);
        // SAFETY: room for `count` addresses.
        let watched = unsafe {
            windows_sys::Win32::System::Memory::GetWriteWatch(
                0,
                p,
                4 * MIB,
                found.as_mut_ptr().cast(),
                &mut count,
                &mut granularity,
            )
        };
        // Not under Rosetta (`victory_gpu::fault_tracking`).
        assert_eq!(watched == 0, victory_gpu::fault_tracking());
        // SAFETY: our own reservation.
        assert_ne!(unsafe { VirtualFree(p, 0, MEM_RELEASE) }, 0);
        assert_eq!(query(p).flags, 0);
        // No address field: above the 32-bit range and below 2^40, where
        // the console keeps it.
        let p = XMemVirtualAlloc(
            std::ptr::null_mut(),
            16 << 30,
            MEM_RESERVE,
            GRAPHICS,
            PAGE_NOACCESS,
        );
        assert!(!p.is_null() && (p as usize) >= 1 << 32, "{p:p}");
        assert!((p as usize) + (16 << 30) <= 1 << 40, "{p:p}");
        // SAFETY: our own reservation.
        assert_ne!(unsafe { VirtualFree(p, 0, MEM_RELEASE) }, 0);
    }

    #[test]
    fn transfers_shrink_the_title_budget() {
        let mut before = WorkingSet::default();
        assert_eq!(XMemGetWorkingSetStatistics(0x2000_0001, &mut before), 1);
        assert!(before.title_used > 0);
        assert_eq!(XMemTransferMemory(0, 128 * MIB - 1), 1);
        assert_eq!(XMemTransferMemory(2, MIB), 0, "no tools memory");
        let standard = victory_gpu::console::current().title_memory().1;
        assert_eq!(XMemTransferMemory(0, standard), 0, "over budget");
        let mut after = WorkingSet::default();
        XMemGetWorkingSetStatistics(0x2000_0001, &mut after);
        assert_eq!(before.title_limit - after.title_limit, 128 * MIB);
    }

    #[test]
    fn statistics_have_the_console_layout() {
        assert_eq!(size_of::<WorkingSet>(), 0x50);
        assert_eq!(size_of::<BasicInformation>(), 0x40);
        assert_eq!(std::mem::offset_of!(WorkingSet, gpu_optimal_limit), 0x28);
    }

    #[test]
    fn commit_limit_report_includes_request_and_system_commit() {
        let error = std::io::Error::from_raw_os_error(ERROR_COMMITMENT_LIMIT as i32);
        let info = PERFORMANCE_INFORMATION {
            CommitTotal: 512,
            CommitLimit: 768,
            PageSize: 4096,
            ..Default::default()
        };
        let report = commit_limit_message(4 * MIB, "physical pages", &error, Some(&info));
        assert!(report.contains("4194304 bytes of physical pages"));
        assert!(report.contains("2 MiB of 3 MiB in use"));
        let fallback = commit_limit_message(4 * MIB, "virtual memory", &error, None);
        assert!(fallback.contains("4194304 bytes of virtual memory"));
        for report in [&report, &fallback] {
            assert!(report.contains("os error 1455"));
            assert!(report.contains("Windows paging file size"));
        }
        assert!(!fallback.contains("System commit:"));
    }

    thread_local! {
        /// Views this thread mapped for the title (`map`).
        pub(super) static MAPS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    }

    #[test]
    fn additive_mapping_preserves_label_addresses_and_physical_aliases() {
        let pages = allocate(1);
        let base = reserve_pages(2);
        assert_eq!(XMemMapPhysicalPages(base, 1, pages.as_ptr()), base);
        let destination = base as usize + PAGE;
        let unrelated = 0u32;
        {
            let physical = physical();
            let touches = |address| {
                mapping_touches_labels(
                    &physical,
                    destination,
                    PAGE,
                    &mut [(address, 4)].into_iter(),
                )
            };
            assert!(!touches(&unrelated as *const u32 as usize));
            assert!(
                touches(destination + 4),
                "the edit changes the label's address"
            );
            assert!(
                touches(base as usize + 4),
                "the mapped source may gain another alias"
            );
        }
        assert_eq!(XMemMapPhysicalPages(base, 1, std::ptr::null()), base);
        release_placeholders(base, 2);
        assert_eq!(XMemFreePhysicalPages(1, pages.as_ptr()), 1);
    }

    #[test]
    fn mapping_what_shows_already_maps_nothing() {
        let pages = allocate(2);
        let reserve = || {
            XMemVirtualAlloc(
                std::ptr::null_mut(),
                2 * PAGE,
                MEM_RESERVE | PAGES_64K,
                0,
                PAGE_READWRITE,
            )
        };
        let (a, b) = (reserve(), reserve());
        assert!(!a.is_null() && !b.is_null());
        let maps = || MAPS.with(std::cell::Cell::get);
        assert_eq!(XMemMapPhysicalPages(a, 2, pages.as_ptr()), a);
        let before = maps();
        // The same pages at the same address: nothing to do.
        assert_eq!(XMemMapPhysicalPages(a, 2, pages.as_ptr()), a);
        let scattered = [a, a.wrapping_byte_add(PAGE)];
        assert_eq!(
            XMemMapPhysicalPagesScatter(scattered.as_ptr(), 2, pages.as_ptr()),
            1
        );
        assert_eq!(maps(), before);
        // The GPU's page-mapping copy, made twice: the second maps nothing.
        assert!(map_alias(b as u64, a as u64, 2 * PAGE as u64));
        assert_eq!(maps(), before + 2);
        assert!(map_alias(b as u64, a as u64, 2 * PAGE as u64));
        assert!(map_alias(
            b.wrapping_byte_add(PAGE) as u64,
            a.wrapping_byte_add(PAGE) as u64,
            PAGE as u64
        ));
        assert_eq!(maps(), before + 2);
        // Another page, or one page of the range changed, maps again.
        assert!(map_alias(
            b as u64,
            a.wrapping_byte_add(PAGE) as u64,
            PAGE as u64
        ));
        assert_eq!(maps(), before + 3);
        // SAFETY: both mapped read-write, showing the second page.
        unsafe {
            a.wrapping_byte_add(PAGE).cast::<u8>().write(6);
            assert_eq!(b.cast::<u8>().read(), 6);
        }
        // Pages swapped at `a`: both differ, so both map again.
        let swapped = [pages[1], pages[0]];
        assert_eq!(XMemMapPhysicalPages(a, 2, swapped.as_ptr()), a);
        assert_eq!(maps(), before + 5);
        // SAFETY: mapped read-write; `a` shows the second page now.
        assert_eq!(unsafe { a.cast::<u8>().read() }, 6);
        for at in [a, b] {
            assert_eq!(XMemMapPhysicalPages(at, 2, std::ptr::null()), at);
        }
        assert_eq!(XMemFreePhysicalPages(2, pages.as_ptr()), 1);
    }

    #[test]
    fn allocator_refusal_report_names_apu_heap_and_system_commit() {
        let info = PERFORMANCE_INFORMATION {
            CommitTotal: 512,
            CommitLimit: 768,
            PageSize: 4096,
            ..Default::default()
        };
        let report = allocator_refusal_message(4 * MIB, Some(&info));
        assert!(report.contains("host allocator refused 4194304 bytes of APU heap memory"));
        assert!(report.contains("System commit: 2 MiB of 3 MiB in use"));
        assert!(report.contains("Windows paging file size"));
        let fallback = allocator_refusal_message(4 * MIB, None);
        assert!(fallback.contains("4194304 bytes of APU heap memory"));
        assert!(!fallback.contains("System commit:"));
    }

    fn reserve_pages(n: usize) -> *mut c_void {
        let region = XMemVirtualAlloc(
            std::ptr::null_mut(),
            n * PAGE,
            MEM_RESERVE | PAGES_64K,
            0,
            PAGE_READWRITE,
        );
        assert!(!region.is_null());
        region
    }

    fn release_placeholders(base: *mut c_void, pages: usize) {
        let mut at = base;
        let end = base.wrapping_byte_add(pages * PAGE);
        while at < end {
            let bytes = query(at).host.RegionSize;
            assert!((1..=end as usize - at as usize).contains(&bytes));
            assert_ne!(release_virtual(at, 0, MEM_RELEASE), 0);
            at = at.wrapping_byte_add(bytes);
        }
    }

    #[test]
    fn reserve_tiles_uses_aligned_bounded_chunks() {
        let mut physical = Physical::default();
        let base = 1usize << 32;
        let chunk = victory_gpu::MAX_TILED_BUFFER;
        physical.reserve(base + 1, PAGE);
        physical.reserve(base, PAGE / 2);
        assert!(physical.reservations.is_empty());
        physical.reserve(base, chunk + PAGE);
        physical.reserve(base + PAGE, chunk);
        assert_eq!(physical.reservations.len(), 1);
        assert!(physical.admit_file(base + chunk));
        let reservation = &physical.reservations[&base];
        assert_eq!(reservation.chunk_size, chunk);
        assert_eq!(reservation.end, base + chunk + PAGE);
        assert!(matches!(
            reservation.chunks.get(&1),
            Some(TileChunk::Ordinary)
        ));
        assert!(!reservation.chunks.contains_key(&0));
    }

    #[test]
    fn unknown_physical_page_preserves_previous_mapping() {
        let pages = allocate(2);
        let base = reserve_pages(2);
        assert_eq!(XMemMapPhysicalPages(base, 2, pages.as_ptr()), base);

        let invalid = [pages[1], 0];
        assert!(XMemMapPhysicalPages(base, 2, invalid.as_ptr()).is_null());
        let physical = physical();
        let first = slice_at(&physical, base as usize).map(|slice| slice.key(0));
        let second = slice_at(&physical, base as usize + PAGE).map(|slice| slice.key(0));
        assert_eq!(first, Some(physical.pages[&pages[0]].key(0)));
        assert_eq!(second, Some(physical.pages[&pages[1]].key(0)));
        drop(physical);

        assert_eq!(XMemMapPhysicalPages(base, 2, std::ptr::null()), base);
        release_placeholders(base, 2);
        assert_eq!(XMemFreePhysicalPages(2, pages.as_ptr()), 1);
    }

    #[cfg(feature = "diagnostics")]
    fn allocate_file_page() -> usize {
        let (mut count, mut file) = (1, 0);
        assert_eq!(
            allocate_physical_pages(0, &mut count, &mut file, |_| None),
            1
        );
        file
    }

    #[cfg(feature = "diagnostics")]
    #[test]
    #[ignore = "requires Rosetta, D3DMetal and VICTORY_PHYSICAL_TILES=1"]
    fn file_mapping_uses_ordinary_shadows_before_binding() {
        assert!(victory_gpu::physical_tiles_enabled());
        let shared = allocate(2);
        let file = allocate_file_page();
        let base = reserve_pages(2);
        assert_eq!(XMemMapPhysicalPages(base, 2, shared.as_ptr()), base);
        let pages = [shared[0], file];
        assert_eq!(XMemMapPhysicalPages(base, 2, pages.as_ptr()), base);
        assert!(matches!(
            physical().reservations[&(base as usize)].chunks.get(&0),
            Some(TileChunk::Ordinary)
        ));
        // SAFETY: the second page is backed by the live file section.
        unsafe { base.wrapping_byte_add(PAGE).cast::<u8>().write(0x7a) };
        assert_eq!(page_bytes(file)[0], 0x7a);
        assert_eq!(XMemMapPhysicalPages(base, 2, std::ptr::null()), base);
        release_placeholders(base, 2);
        assert_eq!(XMemFreePhysicalPages(2, shared.as_ptr()), 1);
        assert_eq!(XMemFreePhysicalPages(1, &file), 1);
    }

    #[cfg(feature = "diagnostics")]
    #[test]
    #[ignore = "requires Rosetta, D3DMetal and VICTORY_PHYSICAL_TILES=1"]
    fn low_placeholder_is_a_coherent_tiled_buffer() {
        assert!(victory_gpu::physical_tiles_enabled());
        let (pages, other) = (allocate(3), allocate(1));
        let base = XMemVirtualAlloc(
            std::ptr::null_mut(),
            4 * PAGE,
            MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
            GRAPHICS | ADDRESS_32BIT,
            PAGE_NOACCESS,
        );
        assert!(!base.is_null() && (base as usize) < 1 << 32);
        let at = |k| base.wrapping_byte_add(k * PAGE);
        let map = |k, count, source| assert_eq!(XMemMapPhysicalPages(at(k), count, source), at(k));
        map(0, 2, pages.as_ptr());
        map(3, 1, other.as_ptr());
        let (first, second, third, last) = (10u32, 20u32, 30u32, 40u32);
        // SAFETY: the test owns the three mapped pages.
        unsafe {
            at(0).cast::<u32>().write(first);
            at(1).cast::<u32>().write(second);
            at(3).cast::<u32>().write(last);
        }
        assert!(write_physical(&pages[2..], 0, &third.to_le_bytes()));
        let probe =
            |k, replace, write| victory_gpu::probe_tiled_alias(at(k) as usize, replace, write);
        let (gpu, words) = probe(0, false, true);
        assert_eq!(words, [first; 2], "CPU writes reached both GPU views");
        // SAFETY: the GPU submission finished and the view is still mapped.
        let written = unsafe { at(0).cast::<u32>().add(1).read() };
        assert_eq!(
            written,
            first ^ 0x1234_5678,
            "UAV write reached title memory"
        );
        assert_eq!(probe(1, false, false), (gpu + PAGE as u64, [second; 2]));
        assert_eq!(probe(3, false, false), (gpu + 3 * PAGE as u64, [last; 2]));
        assert_eq!(probe(2, false, false), (gpu + 2 * PAGE as u64, [0; 2]));

        let file = allocate_file_page();
        assert!(XMemMapPhysicalPages(at(0), 1, &file).is_null());
        assert_eq!(probe(0, false, false), (gpu, [first; 2]));
        assert_eq!(XMemFreePhysicalPages(1, &file), 1);

        map(0, 1, pages[2..].as_ptr());
        assert_eq!(
            probe(0, true, false),
            (gpu, [third; 2]),
            "remap on the replacement queue"
        );
        assert_eq!(probe(1, false, false), (gpu + PAGE as u64, [second; 2]));
        assert_eq!(probe(3, false, false), (gpu + 3 * PAGE as u64, [last; 2]));
        assert_eq!(&page_bytes(pages[0])[..4], &first.to_le_bytes());

        assert_ne!(release_virtual(at(2), 0, MEM_RELEASE), 0);
        assert_eq!(probe(0, false, false), (gpu, [third; 2]));
        assert_eq!(probe(3, false, false), (gpu + 3 * PAGE as u64, [last; 2]));

        map(0, 4, std::ptr::null());
        assert_eq!(probe(0, false, false), (gpu, [0; 2]));
        assert_eq!(probe(3, false, false), (gpu + 3 * PAGE as u64, [0; 2]));
        release_placeholders(base, 2);
        release_placeholders(at(3), 1);
        assert!(!physical().reservations.contains_key(&(base as usize)));
        assert_eq!(XMemFreePhysicalPages(3, pages.as_ptr()), 1);
        assert_eq!(XMemFreePhysicalPages(1, other.as_ptr()), 1);
    }

    #[cfg(feature = "diagnostics")]
    #[test]
    #[ignore = "requires Rosetta, D3DMetal and VICTORY_PHYSICAL_TILES=1"]
    fn tiled_chunk_boundary_and_last_physical_reference() {
        assert!(victory_gpu::physical_tiles_enabled());
        let chunk_pages = victory_gpu::MAX_TILED_BUFFER / PAGE;
        let base = reserve_pages(chunk_pages + 1);
        let left = base.wrapping_byte_add((chunk_pages - 1) * PAGE);
        let right = left.wrapping_byte_add(PAGE);
        let pages = allocate(2);
        let backing = Arc::downgrade(physical().pages[&pages[0]].section.shared.as_ref().unwrap());
        assert_eq!(XMemMapPhysicalPages(left, 2, pages.as_ptr()), left);
        // SAFETY: both pages of the mapped run belong to the test.
        unsafe {
            left.cast::<u32>().write(71);
            right.cast::<u32>().write(92);
        }
        let probe =
            |at: *mut c_void, replace| victory_gpu::probe_tiled_alias(at as usize, replace, false);
        let (left_gpu, left_words) = probe(left, false);
        let (right_gpu, right_words) = probe(right, true);
        assert_eq!(left_words, [71; 2]);
        assert_eq!(right_words, [92; 2]);
        assert_eq!(XMemFreePhysicalPages(2, pages.as_ptr()), 1);
        assert_eq!(probe(right, false), (right_gpu, [92; 2]));
        assert_eq!(XMemMapPhysicalPages(left, 2, std::ptr::null()), left);
        assert!(
            backing.upgrade().is_some(),
            "GPU mappings still need the heap"
        );
        assert_eq!(probe(left, false), (left_gpu, [0; 2]));
        assert!(
            backing.upgrade().is_none(),
            "the fence retired both mappings"
        );
        assert_eq!(probe(right, false), (right_gpu, [0; 2]));
        release_placeholders(base, chunk_pages + 1);
    }

    #[test]
    fn alias_subrange_of_a_direct_view_keeps_both_neighbors() {
        const N: usize = 4;
        let pages = allocate(N);
        let (source, target) = (reserve_pages(N), reserve_pages(N));
        for base in [source, target] {
            assert_eq!(XMemMapPhysicalPages(base, N, pages.as_ptr()), base);
        }
        let at = |base: *mut c_void, k| base.wrapping_byte_add(k * PAGE);
        let keys = |base: *mut c_void| {
            let map = physical();
            std::array::from_fn::<_, N, _>(|k| slice_at(&map, at(base, k) as usize).unwrap().key(0))
        };
        let [first, second, _, last] = keys(source);
        let middle = at(target, 1);
        assert!(has_multi_page_view(&physical(), middle as usize, 2 * PAGE));
        assert!(map_alias(middle as u64, source as u64, (2 * PAGE) as u64));
        assert_eq!(keys(target), [first, first, second, last]);

        // The established overlapping-copy path reads successive source pages in order.
        assert!(map_alias(
            at(target, 2) as u64,
            middle as u64,
            (2 * PAGE) as u64
        ));
        assert_eq!(keys(target), [first; N]);
        for base in [source, target] {
            assert!(map_alias(base as u64, 0, (N * PAGE) as u64));
            release_placeholders(base, N);
        }
        assert_eq!(XMemFreePhysicalPages(N, pages.as_ptr()), 1);
    }

    #[test]
    fn alias_batch_remaps_only_changed_pages() {
        const N: usize = 8;
        let pages = allocate(N);
        let (source, target) = (reserve_pages(N), reserve_pages(N));
        let at = |base: *mut c_void, k| base.wrapping_byte_add(k * PAGE);
        let copy = |to, from, n: usize| map_alias(to as u64, from as u64, (n * PAGE) as u64);
        let maps = || MAPS.with(std::cell::Cell::get);
        assert_eq!(XMemMapPhysicalPages(source, N, pages.as_ptr()), source);
        let before = maps();
        assert!(copy(target, source, N));
        assert_eq!(maps() - before, N);
        // SAFETY: every page is mapped and writable.
        unsafe {
            for k in 0..N {
                at(source, k).cast::<u8>().write(k as u8);
            }
        }

        assert!(copy(at(target, 3), at(source, 4), 1));
        // SAFETY: the replacement still maps a writable section page.
        assert_eq!(unsafe { at(target, 3).cast::<u8>().read() }, 4);
        let before = maps();
        assert!(copy(target, source, N));
        assert_eq!(maps() - before, 1);
        {
            let physical = physical();
            for k in 0..N {
                let page = at(target, k);
                assert_eq!(physical.views[&(page as usize)].0, PAGE);
                // SAFETY: the recorded view still maps the source page.
                assert_eq!(unsafe { page.cast::<u8>().read() }, k as u8);
            }
        }

        // The changed page and an unchanged neighbour both retained their aliases.
        for k in [2, 3] {
            let page = at(target, k) as usize;
            let mut other = Vec::new();
            aliases(page, PAGE, &mut other);
            assert_eq!(other, [(page, at(source, k) as usize)]);
        }
        // A missing source still updates the valid prefix and clears the tail.
        assert!(copy(at(target, 3), source, 1));
        assert!(copy(at(source, 4), std::ptr::null_mut(), 1));
        let before = maps();
        assert!(!copy(target, source, N));
        assert_eq!(maps() - before, 1);
        // SAFETY: the successfully remapped prefix still shows source page 3.
        assert_eq!(unsafe { at(target, 3).cast::<u8>().read() }, 3);
        assert_eq!(query(at(target, 3)).host.State, MEM_COMMIT);
        assert_eq!(query(at(target, 4)).host.State, MEM_RESERVE);
        assert_eq!(query(at(target, N - 1)).host.State, MEM_RESERVE);
        for base in [source, target] {
            assert!(copy(base, std::ptr::null_mut(), N));
            release_placeholders(base, N);
        }
        assert_eq!(XMemFreePhysicalPages(N, pages.as_ptr()), 1);
    }

    #[test]
    fn refused_alias_mapping_preserves_the_prefix_and_clears_the_tail() {
        let pages = allocate(3);
        let (source, target) = (reserve_pages(3), reserve_pages(3));
        let middle = target.wrapping_byte_add(PAGE);
        let last = target.wrapping_byte_add(2 * PAGE);
        assert_eq!(XMemMapPhysicalPages(source, 3, pages.as_ptr()), source);
        assert!(map_alias(target as u64, source as u64, 3 * PAGE as u64));
        // Give the first target the wrong page so the batch must replace it.
        assert!(map_alias(
            target as u64,
            source as u64 + PAGE as u64,
            PAGE as u64
        ));
        assert!(map_alias(middle as u64, 0, PAGE as u64));
        // Map the middle page independently of xmem. Only its owner can
        // return this section view to a placeholder.
        let slice = slice_at(&physical(), source as usize).unwrap();
        // SAFETY: our isolated placeholder and one page of our live section.
        let middle_view = unsafe {
            MapViewOfFile3(
                slice.section.handle,
                GetCurrentProcess(),
                middle,
                (slice.index * PAGE) as u64,
                PAGE,
                MEM_REPLACE_PLACEHOLDER,
                PAGE_READWRITE,
                std::ptr::null_mut(),
                0,
            )
        };
        assert_eq!(middle_view.Value, middle);
        assert!(!map_alias(target as u64, source as u64, 3 * PAGE as u64));
        let mut other = Vec::new();
        aliases(target as usize, PAGE, &mut other);
        assert_eq!(other, [(target as usize, source as usize)]);
        assert_eq!(query(middle).host.AllocationBase, middle);
        assert_eq!(query(last).host.State, MEM_RESERVE);
        assert!(map_alias(source as u64, 0, 3 * PAGE as u64));
        assert!(map_alias(target as u64, 0, PAGE as u64));
        assert_ne!(
            // SAFETY: return the independently owned test view to its placeholder.
            unsafe { UnmapViewOfFile2(GetCurrentProcess(), middle_view, MEM_PRESERVE_PLACEHOLDER) },
            0
        );
        for region in [source, target, middle, last] {
            // SAFETY: these four separate reservations belong to this test.
            assert_ne!(unsafe { VirtualFree(region, 0, MEM_RELEASE) }, 0);
        }
        assert_eq!(XMemFreePhysicalPages(3, pages.as_ptr()), 1);
    }

    #[test]
    fn overflowing_alias_ranges_are_rejected_before_remapping() {
        let last_page = u64::MAX - PAGE as u64 + 1;
        assert!(!map_alias(last_page, 0, PAGE as u64));
        let pages = allocate(1);
        let target = reserve_pages(1);
        assert_eq!(XMemMapPhysicalPages(target, 1, pages.as_ptr()), target);
        assert!(!map_alias(target as u64, last_page, PAGE as u64));
        assert_eq!(query(target).host.State, MEM_COMMIT);
        assert!(map_alias(target as u64, 0, PAGE as u64));
        // SAFETY: our one placeholder is free of views.
        assert_ne!(unsafe { VirtualFree(target, 0, MEM_RELEASE) }, 0);
        assert_eq!(XMemFreePhysicalPages(1, pages.as_ptr()), 1);
    }

    #[test]
    fn physical_pages_map_alias_and_unmap() {
        let mut pages = [0usize; 3];
        let mut n = 3;
        assert_eq!(XMemAllocatePhysicalPages(0, &mut n, pages.as_mut_ptr()), 1);
        assert_eq!(n, 3);
        if std::env::var("VICTORY_PHYSICAL_TILES").as_deref() == Ok("1") {
            let physical = physical();
            assert!(
                physical.pages[&pages[0]].section.shared.is_some(),
                "the test must use shared physical pages"
            );
        }
        // Packed into 21 bits as `number >> 4`, as the title keeps them.
        for p in pages {
            assert_eq!((p >> 4 & 0x1f_ffff) << 4, p);
        }
        // Reservations of 64 KiB pages, as titles make them for mapping.
        let reserve = || {
            XMemVirtualAlloc(
                std::ptr::null_mut(),
                3 * PAGE,
                MEM_RESERVE | PAGES_64K,
                0,
                PAGE_READWRITE,
            )
        };
        let (a, b) = (reserve(), reserve());
        assert!(!a.is_null() && !b.is_null());
        let at = |p: *mut c_void, page: usize| p.wrapping_byte_add(page * PAGE);
        assert_eq!(XMemMapPhysicalPages(a, 3, pages.as_ptr()), a);

        // The same page at a second address shows the same bytes.
        assert_eq!(
            XMemMapPhysicalPages(at(b, 1), 1, pages[2..].as_ptr()),
            at(b, 1)
        );
        // SAFETY: both pages are mapped read-write.
        unsafe {
            at(a, 2).cast::<u8>().write(7);
            assert_eq!(at(b, 1).cast::<u8>().read(), 7);
            a.cast::<u8>().write(1);
        }
        // Each mapping knows the other (the GPU side's change tracking).
        let others = |p: *mut c_void| {
            let mut out = Vec::new();
            aliases(p as usize, PAGE, &mut out);
            out.into_iter().map(|(_, a)| a).collect::<Vec<_>>()
        };
        assert_eq!(others(at(a, 2)), [at(b, 1) as usize]);
        assert_eq!(others(at(b, 1)), [at(a, 2) as usize]);
        assert!(others(a).is_empty());
        // Writes into pages (DirectStorage's) show through their views.
        assert!(write_physical(&pages[2..], 1, &[9]));
        // SAFETY: mapped read-write.
        assert_eq!(unsafe { at(b, 1).cast::<u8>().add(1).read() }, 9);

        // Unmapping the middle page keeps its neighbours.
        assert_eq!(
            XMemMapPhysicalPages(at(a, 1), 1, std::ptr::null()),
            at(a, 1)
        );
        assert_eq!(query(at(a, 1)).host.State, MEM_RESERVE);
        assert_eq!(query(a).host.State, MEM_COMMIT);
        assert_eq!(query(at(a, 2)).host.State, MEM_COMMIT);
        // SAFETY: still mapped.
        assert_eq!(unsafe { at(a, 2).cast::<u8>().read() }, 7);

        // A scattered page lands in the hole.
        let addresses = [at(a, 1)];
        assert_eq!(
            XMemMapPhysicalPagesScatter(addresses.as_ptr(), 1, pages.as_ptr()),
            1
        );
        // SAFETY: page 0 is now mapped at both `a` and `a + PAGE`.
        assert_eq!(unsafe { at(a, 1).cast::<u8>().read() }, 1);
        // Page number 0 unmaps.
        assert_eq!(
            XMemMapPhysicalPagesScatter(addresses.as_ptr(), 1, [0usize].as_ptr()),
            1
        );
        assert_eq!(query(at(a, 1)).host.State, MEM_RESERVE);
        // The GPU's page-mapping copy: what `a + 2 pages` shows, at `b`.
        assert!(map_alias(b as u64, at(a, 2) as u64, PAGE as u64));
        // SAFETY: mapped read-write.
        assert_eq!(unsafe { b.cast::<u8>().read() }, 7);
        let mut both = others(at(a, 2));
        both.sort_unstable();
        assert_eq!(both, [b as usize, at(b, 1) as usize]);
        assert!(map_alias(b as u64, 0, PAGE as u64));
        assert_eq!(others(at(a, 2)), [at(b, 1) as usize], "unmapped views are forgotten");
        assert_eq!(query(b).host.State, MEM_RESERVE);
        assert!(!map_alias(b as u64, at(a, 1) as u64, PAGE as u64));
        assert_eq!(XMemFreePhysicalPages(3, pages.as_ptr()), 1);
        assert_eq!(XMemIsToolsMemAvailable(), 0);
    }

    #[test]
    fn mapping_no_pages_changes_nothing() {
        let pages = allocate(2);
        let a = XMemVirtualAlloc(
            std::ptr::null_mut(),
            2 * PAGE,
            MEM_RESERVE | PAGES_64K,
            0,
            PAGE_READWRITE,
        );
        assert!(!a.is_null());
        assert_eq!(XMemMapPhysicalPages(a, 2, pages.as_ptr()), a);
        // SAFETY: both pages are mapped read-write.
        unsafe { a.cast::<u8>().write(5) };
        let second = a.wrapping_byte_add(PAGE);

        // A count of 0 at the start of a view, inside it and past it: with
        // pages, without (an unmap), and as the GPU's page-mapping copy.
        for at in [a, second, a.wrapping_byte_add(2 * PAGE)] {
            assert_eq!(XMemMapPhysicalPages(at, 0, pages.as_ptr()), at);
            assert_eq!(XMemMapPhysicalPages(at, 0, std::ptr::null()), at);
            assert!(map_alias(at as u64, a as u64, 0));
            assert!(map_alias(at as u64, 0, 0));
        }

        // The view is still the one view it was.
        let view = physical().views.get(&(a as usize)).map(|v| v.0);
        assert_eq!(view, Some(2 * PAGE));
        assert_eq!(query(a).host.State, MEM_COMMIT);
        assert_eq!(query(second).host.State, MEM_COMMIT);
        // SAFETY: still mapped.
        assert_eq!(unsafe { a.cast::<u8>().read() }, 5);

        // A count of 1 still unmaps that page alone.
        assert_eq!(XMemMapPhysicalPages(second, 1, std::ptr::null()), second);
        assert_eq!(query(second).host.State, MEM_RESERVE);
        assert_eq!(query(a).host.State, MEM_COMMIT);
        assert_eq!(XMemMapPhysicalPages(a, 1, std::ptr::null()), a);
        assert_eq!(XMemFreePhysicalPages(2, pages.as_ptr()), 1);
    }

    /// `n` pages of one new allocation.
    fn allocate(n: usize) -> Vec<usize> {
        let (mut pages, mut given) = (vec![0; n], n);
        assert_eq!(
            XMemAllocatePhysicalPages(0, &mut given, pages.as_mut_ptr()),
            1
        );
        assert_eq!(given, n);
        pages
    }

    /// The bytes of one page, through our view of its section.
    fn page_bytes(page: usize) -> Vec<u8> {
        let slice = physical().pages.get(&page).cloned().expect("known page");
        View::map(&slice, 1).expect("mapped").bytes().to_vec()
    }

    #[test]
    fn runs_group_consecutive_pages_of_one_allocation() {
        let (a, b) = (allocate(3), allocate(1));
        let lengths = |pages: &[usize], offset, len| {
            runs(pages, offset, len).map(|r| r.iter().map(|&(_, n)| n).collect::<Vec<_>>())
        };
        assert_eq!(lengths(&a, 0, 3 * PAGE), Some(vec![3]));
        // Only the pages the bytes reach.
        assert_eq!(lengths(&a, PAGE + 5, 10), Some(vec![1]));
        assert_eq!(lengths(&a, PAGE - 1, 2), Some(vec![2]));
        assert_eq!(lengths(&a, 0, 0), Some(vec![]));
        // Out of order, or across allocations: one run each.
        assert_eq!(
            lengths(&[a[0], a[2], a[1]], 0, 3 * PAGE),
            Some(vec![1, 1, 1])
        );
        assert_eq!(lengths(&[a[1], a[2], b[0]], 0, 3 * PAGE), Some(vec![2, 1]));
        // A slice that follows on in another section starts a run.
        assert_eq!(lengths(&[b[0], a[1]], 0, 2 * PAGE), Some(vec![1, 1]));
        // Too few pages for the bytes, or an unknown one.
        assert_eq!(lengths(&a, 0, 3 * PAGE + 1), None);
        assert_eq!(lengths(&[a[0], 0x10], 0, 2 * PAGE), None);
        assert_eq!(lengths(&a, usize::MAX, 2), None);
        assert_eq!(XMemFreePhysicalPages(3, a.as_ptr()), 1);
        assert_eq!(XMemFreePhysicalPages(1, b.as_ptr()), 1);
    }

    #[test]
    fn writes_cross_runs_and_allocations() {
        let (a, b) = (allocate(2), allocate(1));
        let pages = [a[0], a[1], b[0]];
        let data: Vec<u8> = (0..2 * PAGE + 64).map(|i| (i * 7 % 251) as u8).collect();
        let offset = PAGE - 32;
        // Past the third page: the second run is too short.
        assert!(!write_physical(&pages, offset, &data));
        assert!(write_physical(&pages, offset, &data[..2 * PAGE]));
        let seen: Vec<u8> = pages.iter().flat_map(|&p| page_bytes(p)).collect();
        assert!(seen[..offset].iter().all(|&b| b == 0), "before the offset");
        assert_eq!(seen[offset..offset + 2 * PAGE], data[..2 * PAGE]);
        assert!(seen[offset + 2 * PAGE..].iter().all(|&b| b == 0), "after");
        assert!(
            !write_physical(&[a[0], 0x10], 0, &data[..PAGE + 1]),
            "unknown page"
        );
        assert_eq!(XMemFreePhysicalPages(2, a.as_ptr()), 1);
        assert_eq!(XMemFreePhysicalPages(1, b.as_ptr()), 1);
    }

    /// A page mapped at a second address while a window writes it is
    /// reported there too when the window is done (the title maps streamed
    /// pages into its GPU heap while their reads decode); an address that
    /// stopped showing it meanwhile is not.
    #[test]
    fn page_windows_report_where_the_pages_show_when_done() {
        let pages = allocate(1);
        let reserve = || XMemVirtualAlloc(std::ptr::null_mut(), PAGE, MEM_RESERVE | PAGES_64K, 0, PAGE_READWRITE);
        let (a, b) = (reserve(), reserve());
        assert!(!a.is_null() && !b.is_null());
        assert_eq!(XMemMapPhysicalPages(a, 1, pages.as_ptr()), a);
        REPORTED.with(|r| r.borrow_mut().clear());
        let mut window = page_window(&pages, 0, 16).expect("one run");
        assert_eq!(window.writing.is_some(), victory_gpu::landing_enabled());
        // Mapped at `b`, unmapped from `a`, while the bytes are written.
        assert_eq!(XMemMapPhysicalPages(b, 1, pages.as_ptr()), b);
        assert_eq!(XMemMapPhysicalPages(a, 1, std::ptr::null()), a);
        window.bytes().fill(3);
        drop(window);
        assert_eq!(REPORTED.with(|r| r.borrow().clone()), [b as usize]);
        // SAFETY: mapped read-write.
        assert_eq!(unsafe { b.cast::<u8>().read() }, 3);
        // `write_physical` the same way.
        REPORTED.with(|r| r.borrow_mut().clear());
        assert!(write_physical(&pages, 0, &[4]));
        assert_eq!(REPORTED.with(|r| r.borrow().clone()), [b as usize]);
        assert_eq!(XMemMapPhysicalPages(b, 1, std::ptr::null()), b);
        assert_eq!(XMemFreePhysicalPages(1, pages.as_ptr()), 1);
    }

    #[test]
    fn page_windows_cover_one_run_in_place() {
        let (a, b) = (allocate(3), allocate(1));
        let mut window = page_window(&a, PAGE + 3, PAGE).expect("one run");
        assert_eq!(window.bytes().len(), PAGE);
        window.bytes().fill(0x5a);
        drop(window);
        let (second, third) = (page_bytes(a[1]), page_bytes(a[2]));
        assert!(second[..3].iter().all(|&b| b == 0));
        assert!(second[3..].iter().all(|&b| b == 0x5a));
        assert!(third[..3].iter().all(|&b| b == 0x5a));
        assert!(third[3..].iter().all(|&b| b == 0));
        // Several runs, or an unknown page: no window.
        assert!(page_window(&[a[0], b[0]], 0, PAGE + 1).is_none());
        assert!(page_window(&[a[1], a[0]], 0, PAGE + 1).is_none());
        assert!(page_window(&[0x10], 0, 1).is_none());
        // One page of several is still one run.
        assert!(page_window(&[a[0], b[0]], 0, PAGE).is_some());
        assert_eq!(XMemFreePhysicalPages(3, a.as_ptr()), 1);
        assert_eq!(XMemFreePhysicalPages(1, b.as_ptr()), 1);
    }

    #[test]
    fn landing_view_follows_the_section_and_rejects_replacements() {
        use victory_gpu::LandingPage;
        use windows_sys::Win32::System::Memory::{PAGE_READONLY, VirtualProtect};
        let pages = allocate(4);
        let reserve = || {
            XMemVirtualAlloc(
                std::ptr::null_mut(),
                3 * PAGE,
                MEM_RESERVE | PAGES_64K,
                0,
                PAGE_READWRITE,
            )
        };
        let (a, b) = (reserve(), reserve());
        assert!(!a.is_null() && !b.is_null());
        let at = a as usize;
        assert_eq!(XMemMapPhysicalPages(a, 3, pages.as_ptr()), a);
        assert_eq!(XMemMapPhysicalPages(b, 1, pages.as_ptr()), b);
        let epoch = victory_gpu::next_mapping_epoch();
        let mut old = 0;
        for view in [a, b] {
            // SAFETY: writable OS section views allocated by the test.
            let protected = unsafe { VirtualProtect(view, PAGE, PAGE_READONLY, &mut old) };
            assert_ne!(protected, 0);
        }
        landing_page(at + 19, 1, epoch, true, &mut |page, _| {
            let LandingPage::Beside(address) = page else {
                panic!("missing writable section alias")
            };
            // SAFETY: xmem pins the alternate section view for this callback.
            unsafe { (address as *mut u8).write_volatile(0xb1) };
        });
        for view in [a, b] {
            assert_eq!(query(view).host.Protect, PAGE_READONLY);
            // SAFETY: the views remain readable.
            assert_eq!(unsafe { view.cast::<u8>().add(19).read_volatile() }, 0xb1);
        }
        landing_page(at, 1, epoch, false, &mut |p, _| {
            assert!(matches!(p, LandingPage::Direct))
        });
        landing_page(at + 19, 3 * PAGE - 19, epoch, true, &mut |p, len| {
            assert!(matches!(p, LandingPage::Beside(_)));
            assert_eq!(len, 3 * PAGE - 19);
        });
        let last = physical().pages[&pages[2]].clone();
        let writing_last = PhysicalWrite::begin(&last, 1);
        landing_page(at, 3 * PAGE, epoch, true, &mut |p, len| {
            assert!(matches!(p, LandingPage::Beside(_)));
            assert_eq!(len, 2 * PAGE, "stop before the busy section slice");
        });
        drop(writing_last);

        // The section view may also be the target of a DirectStorage
        // decode. A landing against that slice waits for the bytes.
        let slice = physical().pages[&pages[0]].clone();
        let mut view = View::map(&slice, 1).expect("a section view");
        let writing = PhysicalWrite::begin(&slice, 1);
        let (send, recv) = std::sync::mpsc::channel();
        let worker = std::thread::spawn(move || {
            landing_page(at + 19, 1, epoch, true, &mut |page, _| {
                let LandingPage::Beside(address) = page else {
                    panic!("missing writable section alias");
                };
                // SAFETY: the callback pins the view after the decoder.
                send.send(unsafe { (address as *const u8).read_volatile() })
                    .unwrap();
            });
        });
        assert_eq!(
            recv.recv_timeout(std::time::Duration::from_millis(20)),
            Err(std::sync::mpsc::RecvTimeoutError::Timeout)
        );
        view.bytes()[19] = 0xc2;
        drop(writing);
        assert_eq!(
            recv.recv_timeout(std::time::Duration::from_secs(5)),
            Ok(0xc2)
        );
        worker.join().unwrap();

        // Replacing the middle page splits the view. The neighbours still
        // show their original section slices and take the in-flight writes.
        let middle = a.wrapping_byte_add(PAGE);
        assert_eq!(XMemMapPhysicalPages(middle, 1, pages[3..].as_ptr()), middle);
        landing_page(at, 3 * PAGE, epoch, true, &mut |p, len| {
            assert!(matches!(p, LandingPage::Beside(_)));
            assert_eq!(len, PAGE, "stop before the replacement");
        });
        for beside in [true, false] {
            landing_page(at + PAGE, 1, epoch, beside, &mut |p, _| {
                assert!(matches!(p, LandingPage::Replaced))
            });
        }
        let after_replacement = victory_gpu::next_mapping_epoch();
        landing_page(at + PAGE, 1, after_replacement, false, &mut |p, _| {
            assert!(matches!(p, LandingPage::Direct))
        });
        landing_page(at + 2 * PAGE, 1, epoch, true, &mut |p, _| {
            assert!(matches!(p, LandingPage::Beside(_)))
        });
        assert_eq!(XMemMapPhysicalPages(middle, 1, pages[1..].as_ptr()), middle);
        let remapped = victory_gpu::next_mapping_epoch();
        landing_page(at, 3 * PAGE, remapped, true, &mut |p, len| {
            assert!(matches!(p, LandingPage::Beside(_)));
            assert_eq!(
                len,
                3 * PAGE,
                "contiguous slices in separate views share a pin"
            );
        });
        assert_eq!(XMemMapPhysicalPages(a, 3, std::ptr::null()), a);
        assert_eq!(XMemMapPhysicalPages(b, 1, std::ptr::null()), b);
        assert_eq!(XMemFreePhysicalPages(pages.len(), pages.as_ptr()), 1);
    }

    #[test]
    fn page_writes_share_one_view_of_their_section() {
        let a = allocate(3);
        let slice = |p| physical().pages.get(&p).cloned().expect("known page");
        let section = Arc::downgrade(&slice(a[0]).section);
        // Every request on the section goes through the same view, each
        // page at its place in it, with no view of its own to map and unmap.
        let (first, third) = (
            View::map(&slice(a[0]), 1).expect("mapped"),
            View::map(&slice(a[2]), 1).expect("mapped"),
        );
        assert_eq!(third.at - first.at, 2 * PAGE);
        assert!(first.own.is_none() && third.own.is_none());
        let at = first.at;
        drop((first, third));
        assert_eq!(View::map(&slice(a[1]), 2).expect("mapped").at, at + PAGE);
        assert!(write_physical(&a, 2 * PAGE + 3, &[4, 5]));
        assert_eq!(page_bytes(a[2])[3..5], [4, 5]);
        // The view lives as long as the section: until its pages are freed.
        assert_eq!(XMemFreePhysicalPages(3, a.as_ptr()), 1);
        assert!(section.upgrade().is_none());
    }
}

/// The space below 4 GiB, which the title's 32-bit memory (`XMemVirtualAlloc`
/// with the 32-bit address flag) has to fit in.
const FOUR_GIB: usize = 1 << 32;

/// Guards over the space below 4 GiB. Windows places allocations bottom up,
/// so the host's own memory (the driver's, heaps, the shader and pipeline
/// caches read at startup) filled it first: with a large pipeline cache only
/// 22 MB was left, in pieces, when Grand Theft Auto VI asked for a 185 MB
/// 32-bit placeholder at startup, the request was refused
/// (`ERROR_NOT_ENOUGH_MEMORY`) and the title stopped at an assertion. The
/// debug layer's memory did the same. So the free space below 4 GiB is
/// reserved at startup, past [`low_guards::SLACK`] left for the host, and a
/// 32-bit request that finds no room releases guards and tries again.
pub mod low_guards {
    use std::sync::Mutex;

    use windows_sys::Win32::System::Memory::{
        MEM_FREE, MEM_RELEASE, MEM_RESERVE, MEMORY_BASIC_INFORMATION, PAGE_NOACCESS, VirtualAlloc,
        VirtualFree, VirtualQuery,
    };

    use super::FOUR_GIB;

    /// Free space left to the host below 4 GiB (the lowest free space).
    pub(crate) const SLACK: usize = 256 << 20;
    /// The allocation granularity of Windows.
    const GRANULE: usize = 64 << 10;

    /// The guards, `(base, bytes)`.
    static GUARDS: Mutex<Vec<(usize, usize)>> = Mutex::new(Vec::new());

    fn guards() -> std::sync::MutexGuard<'static, Vec<(usize, usize)>> {
        GUARDS.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// The free runs in `[from, to)`, granule-aligned.
    fn free_runs(from: usize, to: usize) -> Vec<(usize, usize)> {
        let mut runs = Vec::new();
        let mut at = from.max(GRANULE);
        while at < to {
            // SAFETY: plain data filled in by the query.
            let mut m: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
            // SAFETY: a valid out pointer of the right size.
            if unsafe { VirtualQuery(at as *const _, &mut m, size_of_val(&m)) } == 0 {
                break;
            }
            let base = m.BaseAddress as usize;
            let end = (base + m.RegionSize).min(to);
            if m.State == MEM_FREE {
                let (start, end) = (
                    base.max(from).next_multiple_of(GRANULE),
                    end & !(GRANULE - 1),
                );
                if end > start {
                    runs.push((start, end - start));
                }
            }
            at = base + m.RegionSize;
        }
        runs
    }

    /// Reserves `[base, base + bytes)` as a guard; false if something took it.
    fn guard(base: usize, bytes: usize) -> bool {
        // SAFETY: a reservation of free address space; nothing refers to it.
        let at = unsafe { VirtualAlloc(base as *const _, bytes, MEM_RESERVE, PAGE_NOACCESS) };
        at as usize == base
    }

    /// Reserves the free space below 4 GiB but the lowest [`SLACK`]. Call
    /// once, before the host allocates much (the GPU device, the caches).
    pub fn install() {
        let mut guards = guards();
        let mut slack = SLACK;
        for (mut base, mut bytes) in free_runs(0, FOUR_GIB) {
            let left = slack.min(bytes);
            (base, bytes, slack) = (base + left, bytes - left, slack - left);
            if bytes > 0 && guard(base, bytes) {
                guards.push((base, bytes));
            }
        }
        let total: usize = guards.iter().map(|g| g.1).sum();
        tracing::info!(
            guards = guards.len(),
            mb = total >> 20,
            "space below 4 GiB kept for the title"
        );
    }

    /// Try the smallest fitting guards first, then try smaller guards in
    /// descending size order. A replacement guard cannot be retried here.
    pub(crate) fn retry(
        bytes: usize,
        mut allocate: impl FnMut() -> *mut std::ffi::c_void,
    ) -> *mut std::ffi::c_void {
        use windows_sys::Win32::Foundation::{GetLastError, SetLastError};

        // Preserve the allocation error across address queries and reservations.
        // SAFETY: reads this thread's last-error value.
        let mut error = unsafe { GetLastError() };
        let mut guards = guards();
        let mut candidates = guards.clone();
        candidates.sort_unstable_by_key(|&(_, size)| {
            let fits = size >= bytes;
            (!fits, if fits { size } else { usize::MAX - size })
        });
        for (base, size) in candidates {
            // SAFETY: our own reservation, with no allocations inside it.
            if unsafe { VirtualFree(base as *mut _, 0, MEM_RELEASE) } == 0 {
                continue;
            }
            guards.retain(|&(b, _)| b != base);
            let at = allocate();
            // SAFETY: reads this thread's last-error value.
            error = unsafe { GetLastError() };
            for (start, len) in free_runs(base, base + size) {
                if guard(start, len) {
                    guards.push((start, len));
                }
            }
            if !at.is_null() {
                return at;
            }
        }
        // SAFETY: sets this thread's last-error value.
        unsafe { SetLastError(error) };
        std::ptr::null_mut()
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        static TEST: Mutex<()> = Mutex::new(());

        #[test]
        fn a_refused_retry_leaves_the_guards_in_place() {
            let _one = TEST.lock().unwrap();
            // SAFETY: a fresh reservation, released at once.
            let hole =
                unsafe { VirtualAlloc(std::ptr::null(), 4 * GRANULE, MEM_RESERVE, PAGE_NOACCESS) };
            assert!(!hole.is_null());
            let first = hole as usize;
            // SAFETY: ours.
            assert_ne!(unsafe { VirtualFree(hole, 0, MEM_RELEASE) }, 0);
            assert!(guard(first, 4 * GRANULE));
            guards().push((first, 4 * GRANULE));
            let mut attempts = 0;
            let result = retry(4 * GRANULE, || {
                attempts += 1;
                std::ptr::null_mut()
            });
            assert!(result.is_null());
            assert_eq!(attempts, 1);
            assert!(guards().contains(&(first, 4 * GRANULE)));
            guards().retain(|&(base, _)| base != first);
            // SAFETY: the guard the retry put back, ours.
            assert_ne!(unsafe { VirtualFree(hole, 0, MEM_RELEASE) }, 0);
        }

        /// A released guard is guarded again around what took part of it.
        #[test]
        fn a_released_guard_is_guarded_again_around_what_took_it() {
            let _one = TEST.lock().unwrap();
            // A known hole of four granules: reserved, then freed.
            // SAFETY: a fresh reservation, released at once.
            let hole =
                unsafe { VirtualAlloc(std::ptr::null(), 4 * GRANULE, MEM_RESERVE, PAGE_NOACCESS) }
                    as usize;
            assert_ne!(hole, 0);
            // SAFETY: ours.
            unsafe { VirtualFree(hole as *mut _, 0, MEM_RELEASE) };
            assert!(guard(hole, 4 * GRANULE));
            guards().push((hole, 4 * GRANULE));
            let taken = retry(GRANULE, || {
                // SAFETY: a reservation of the first granule at a known address.
                unsafe { VirtualAlloc(hole as *const _, GRANULE, MEM_RESERVE, PAGE_NOACCESS) }
            }) as usize;
            assert_eq!(taken, hole);
            assert!(guards().contains(&(hole + GRANULE, 3 * GRANULE)));
            guards().retain(|&(b, _)| b != hole + GRANULE);
            // SAFETY: ours.
            unsafe {
                VirtualFree(taken as *mut _, 0, MEM_RELEASE);
                VirtualFree((hole + GRANULE) as *mut _, 0, MEM_RELEASE);
            }
        }
    }
}
