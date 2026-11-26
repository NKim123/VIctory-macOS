//! Records title descriptor writes from the byte-verified table stores and
//! imported copies. Based on VIctory-7900XTX's renderer port 2 (GPL-3.0).

use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU8, AtomicU64, Ordering};
use std::sync::{OnceLock, RwLock};

use Ordering::{AcqRel, Acquire, Relaxed, Release};

const STRIDE: usize = 32;
const CHUNK: usize = 128;
const MAX_SLOTS: usize = 2_000_000;

/// Opt-in launcher switch.
pub fn on() -> bool {
    std::env::var_os("VICTORY_PORT_DESCRIPTORS").is_some_and(|value| value == "1")
}

// Layout read by victory-loader's descriptor stubs. Only len can change
// after publishing it, and the arrays live until process exit.
#[repr(C)]
struct Marks {
    start: AtomicU64,
    len: AtomicU64,
    slots: AtomicPtr<AtomicU8>,
    chunks: AtomicPtr<AtomicU8>,
}

const _: () = assert!(size_of::<Marks>() == 32);

static MARKS: Marks = Marks {
    start: AtomicU64::new(0),
    len: AtomicU64::new(0),
    slots: AtomicPtr::new(std::ptr::null_mut()),
    chunks: AtomicPtr::new(std::ptr::null_mut()),
};

/// Stable address of the four cells used by the generated code.
pub fn marks_address() -> usize {
    &raw const MARKS as usize
}

static ENABLED: AtomicBool = AtomicBool::new(false);

/// Called at startup once all verified store hooks have been published.
pub fn enable() {
    ENABLED.store(true, Release);
}

struct Slots {
    slots: Box<[AtomicU8]>,
    chunks: Box<[AtomicU8]>,
}

impl Slots {
    fn new(count: usize) -> Self {
        Self {
            slots: (0..count).map(|_| AtomicU8::new(0)).collect(),
            chunks: (0..count.div_ceil(CHUNK))
                .map(|_| AtomicU8::new(0))
                .collect(),
        }
    }

    fn mark(&self, range: std::ops::Range<usize>) {
        if range.is_empty() {
            return;
        }
        let chunks = range.start / CHUNK..range.end.div_ceil(CHUNK);
        for index in range {
            self.slots[index].store(1, Release);
        }
        for index in chunks {
            self.chunks[index].store(1, Release);
        }
    }

    fn take(&self) -> Vec<(usize, usize)> {
        let mut runs: Vec<(usize, usize)> = Vec::new();
        for (chunk, dirty) in self.chunks.iter().enumerate() {
            if dirty.load(Relaxed) == 0 || dirty.swap(0, AcqRel) == 0 {
                continue;
            }
            // Clear the chunk before its slots. A racing writer marks the
            // slot first, then publishes the chunk again.
            let first = chunk * CHUNK;
            let end = (first + CHUNK).min(self.slots.len());
            for index in first..end {
                let mark = &self.slots[index];
                if mark.load(Relaxed) != 0 && mark.swap(0, AcqRel) != 0 {
                    match runs.last_mut() {
                        Some((_, end)) if *end == index => *end += 1,
                        _ => runs.push((index, index + 1)),
                    }
                }
            }
        }
        runs
    }
}

static SLOTS: OnceLock<Slots> = OnceLock::new();
static HEAP: RwLock<Option<usize>> = RwLock::new(None);
static LAST_SCAN: AtomicU64 = AtomicU64::new(u64::MAX);

pub(crate) fn adopt(id: usize, start: u64, count: u32) {
    let count = count as usize;
    if !ENABLED.load(Acquire)
        || count == 0
        || count > MAX_SLOTS
        || !start.is_multiple_of(STRIDE as u64)
    {
        return;
    }
    let mut heap = HEAP.write().unwrap_or_else(|e| e.into_inner());
    // The first eligible heap alone is adopted. Fixed storage and base
    // make an in-flight writer safe even if the heap is retired.
    if SLOTS.get().is_some() {
        tracing::info!(start, count, "placed view heap uses descriptor watch");
        return;
    }
    let slots = SLOTS.get_or_init(|| Slots::new(count));
    *heap = Some(id);
    MARKS.start.store(start, Relaxed);
    MARKS.slots.store(slots.slots.as_ptr().cast_mut(), Relaxed);
    MARKS
        .chunks
        .store(slots.chunks.as_ptr().cast_mut(), Relaxed);
    MARKS.len.store((count * STRIDE) as u64, Release);
    tracing::info!(start, count, "placed view heap uses descriptor marks");
}

pub(crate) fn retire(id: usize) {
    if MARKS.len.load(Acquire) == 0 {
        return;
    }
    let mut heap = HEAP.write().unwrap_or_else(|e| e.into_inner());
    if *heap == Some(id) {
        MARKS.len.store(0, Release);
        *heap = None;
    }
}

/// Taken with the descriptor registry held, which serializes retirement.
pub(crate) fn take(id: usize) -> Option<(Vec<(usize, usize)>, bool)> {
    if MARKS.len.load(Acquire) == 0 || *HEAP.read().unwrap_or_else(|e| e.into_inner()) != Some(id) {
        return None;
    }
    let frame = crate::frame::current();
    let scanned = LAST_SCAN.load(Relaxed);
    let audit = scanned == u64::MAX || frame.saturating_sub(scanned) >= 8;
    if audit {
        LAST_SCAN.store(frame, Relaxed);
    }
    Some((SLOTS.get().unwrap().take(), audit))
}

fn overlap(to: usize, len: usize, start: usize, bytes: usize) -> std::ops::Range<usize> {
    let lo = to.max(start).saturating_sub(start);
    let hi = to
        .saturating_add(len)
        .min(start + bytes)
        .saturating_sub(start);
    if lo >= hi {
        return 0..0;
    }
    lo / STRIDE..hi.div_ceil(STRIDE)
}

pub(crate) fn mark(handle: u64) {
    let bytes = MARKS.len.load(Acquire) as usize;
    if bytes != 0 {
        let start = MARKS.start.load(Relaxed) as usize;
        SLOTS
            .get()
            .unwrap()
            .mark(overlap(handle as usize, 12, start, bytes));
    }
}

type Copy = unsafe extern "C" fn(*mut u8, *const u8, usize) -> *mut u8;
static MEMCPY: OnceLock<usize> = OnceLock::new();
static MEMMOVE: OnceLock<usize> = OnceLock::new();

/// Wrap the resolved host copy function. A second, different binding of
/// the same name keeps its host implementation.
pub fn import(name: &str, address: usize) -> usize {
    if !ENABLED.load(Acquire) {
        return address;
    }
    let (real, wrapper) = match name {
        "memcpy" => (&MEMCPY, memcpy as Copy),
        "memmove" => (&MEMMOVE, memmove as Copy),
        _ => return address,
    };
    if *real.get_or_init(|| address) == address {
        wrapper as usize
    } else {
        address
    }
}

unsafe extern "C" fn memcpy(dst: *mut u8, src: *const u8, len: usize) -> *mut u8 {
    // SAFETY: same pointer contract as the resolved CRT import.
    unsafe { copied(dst, src, len, *MEMCPY.get().unwrap()) }
}

unsafe extern "C" fn memmove(dst: *mut u8, src: *const u8, len: usize) -> *mut u8 {
    // SAFETY: same pointer contract as the resolved CRT import.
    unsafe { copied(dst, src, len, *MEMMOVE.get().unwrap()) }
}

unsafe fn copied(dst: *mut u8, src: *const u8, len: usize, real: usize) -> *mut u8 {
    let bytes = MARKS.len.load(Acquire) as usize;
    let range = overlap(dst as usize, len, MARKS.start.load(Relaxed) as usize, bytes);
    if range.is_empty() {
        // SAFETY: address came from the import resolver and the caller owns the ranges.
        return unsafe { std::mem::transmute::<usize, Copy>(real)(dst, src, len) };
    }
    // SAFETY: the caller owns both ranges. These stores can resume across
    // guarded pages on Rosetta, including for overlapping memmove.
    unsafe { copy_guarded(dst, src, len) };
    SLOTS.get().unwrap().mark(range);
    dst
}

unsafe fn copy_guarded(dst: *mut u8, src: *const u8, len: usize) {
    if dst.cast_const() == src || len == 0 {
        return;
    }
    if (dst as usize).wrapping_sub(src as usize) < len {
        // No direction flag changes across a fault or exception callback.
        // SAFETY: each iteration accesses one byte of the caller's ranges.
        unsafe {
            std::arch::asm!(
                "2:",
                "dec {left}",
                "mov al, byte ptr [{src} + {left}]",
                "mov byte ptr [{dst} + {left}], al",
                "jnz 2b",
                left = inout(reg) len => _,
                src = in(reg) src,
                dst = in(reg) dst,
                out("al") _,
                options(nostack),
            );
        }
    } else {
        // SAFETY: forward traversal is safe for these ranges. The Win64
        // ABI leaves DF clear, and REP MOVSB is restartable on Rosetta.
        unsafe {
            std::arch::asm!(
                "rep movsb",
                inout("rcx") len => _,
                inout("rsi") src => _,
                inout("rdi") dst => _,
                options(nostack, preserves_flags),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn copy_marks_partial_slots_and_coalesces_across_chunks() {
        let slots = Slots::new(300);
        let start = 0x10000;
        assert!(overlap(start - 4, 4, start, 300 * STRIDE).is_empty());
        slots.mark(overlap(start - 4, 5, start, 300 * STRIDE));
        slots.mark(overlap(start + 128 * STRIDE - 1, 2, start, 300 * STRIDE));
        slots.mark(overlap(start + 300 * STRIDE - 1, 10, start, 300 * STRIDE));
        assert_eq!(slots.take(), [(0, 1), (127, 129), (299, 300)]);
        assert!(slots.take().is_empty());
    }

    #[test]
    fn guarded_copy_has_memmove_overlap_semantics() {
        for (from, to, len) in [(0, 4, 60), (4, 0, 60), (1, 1, 61), (0, 32, 10), (1, 0, 0)] {
            let mut got: [u8; 64] = std::array::from_fn(|i| i as u8);
            let mut expected = got;
            expected.copy_within(from..from + len, to);
            let data = got.as_mut_ptr();
            // SAFETY: both ranges lie in the test array.
            unsafe { copy_guarded(data.add(to), data.add(from), len) };
            assert_eq!(got, expected);
        }
    }
}
