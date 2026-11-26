//! Byte-verified descriptor table stores from VIctory-7900XTX (GPL-3.0).

use std::ops::Range;
use std::sync::OnceLock;

use windows_sys::Win32::System::Diagnostics::Debug::{
    AddVectoredExceptionHandler, EXCEPTION_CONTINUE_SEARCH, EXCEPTION_POINTERS,
    FlushInstructionCache, ReadProcessMemory,
};
use windows_sys::Win32::System::Memory::{
    MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_EXECUTE_READ, PAGE_READWRITE, VirtualAlloc,
    VirtualFree, VirtualProtect,
};
use windows_sys::Win32::System::Threading::GetCurrentProcess;

use crate::startup_patch::{self, FaultSite};

/// A store of the title's to hook: the window's offset in the image, the
/// store's offset in the window and its length, and the window's bytes.
pub struct Site {
    pub window: usize,
    pub store: usize,
    pub len: usize,
    pub bytes: &'static [u8],
}

/// The six record stores of the table fillers (game_scarlett_master.exe
/// 1.0.2.1, image offsets).
pub const SITES: [Site; 6] = [
    // +0x4d9efd0, loop 1: [rcx+rsi]
    Site {
        window: 0x4d9f04f,
        store: 10,
        len: 5,
        bytes: &[
            0x81, 0xe1, 0xe0, 0x7f, 0x00, 0x00, 0xc5, 0xfc, 0x10, 0x00, 0xc5, 0xfc, 0x11, 0x04,
            0x31, 0x0f, 0xb6, 0x47, 0x0b, 0x39, 0xc5,
        ],
    },
    // loop 2: [rcx+rsi]
    Site {
        window: 0x4d9f10d,
        store: 10,
        len: 5,
        bytes: &[
            0x81, 0xe1, 0xe0, 0x7f, 0x00, 0x00, 0xc5, 0xfc, 0x10, 0x00, 0xc5, 0xfc, 0x11, 0x04,
            0x31, 0x0f, 0xb6, 0x47, 0x0f, 0x41, 0x39, 0xc5,
        ],
    },
    // loop 3: [r9+rsi]
    Site {
        window: 0x4d9f19c,
        store: 12,
        len: 6,
        bytes: &[
            0x41, 0x81, 0xe1, 0xe0, 0x7f, 0x00, 0x00, 0xc4, 0xc1, 0x7c, 0x10, 0x02, 0xc4, 0xc1,
            0x7c, 0x11, 0x04, 0x31, 0x44, 0x0f, 0xb6, 0x4f, 0x0d, 0x44, 0x39, 0xc8,
        ],
    },
    // +0x4eec8e0, loop 1: [r9+r8]
    Site {
        window: 0x4eec95d,
        store: 11,
        len: 6,
        bytes: &[
            0x41, 0x81, 0xe1, 0xe0, 0x7f, 0x00, 0x00, 0xc5, 0xfc, 0x10, 0x00, 0xc4, 0x81, 0x7c,
            0x11, 0x04, 0x01, 0x0f, 0xb6, 0x42, 0x0b, 0x39, 0xc5,
        ],
    },
    // loop 2: [rdi+r8]
    Site {
        window: 0x4eeca08,
        store: 10,
        len: 6,
        bytes: &[
            0x81, 0xe7, 0xe0, 0x7f, 0x00, 0x00, 0xc5, 0xfc, 0x10, 0x06, 0xc4, 0xa1, 0x7c, 0x11,
            0x04, 0x07, 0x0f, 0xb6, 0x72, 0x0d, 0x39, 0xf0,
        ],
    },
    // loop 3: [r10+r8]
    Site {
        window: 0x4eeca82,
        store: 11,
        len: 6,
        bytes: &[
            0x41, 0x81, 0xe2, 0xe0, 0x7f, 0x00, 0x00, 0xc5, 0xfc, 0x10, 0x00, 0xc4, 0x81, 0x7c,
            0x11, 0x04, 0x02, 0x0f, 0xb6, 0x42, 0x0f, 0x41, 0x39, 0xc6,
        ],
    },
];

/// `lea rax, [base+index]` for a `vmovups [base+index], ymm0` (VEX, ModRM
/// 04h with a SIB byte, no displacement, the stack pointer not its base):
/// `None` for another form.
fn lea_of(store: &[u8]) -> Option<[u8; 4]> {
    let (x, b, op_at) = match *store.first()? {
        0xc5 => (false, false, 2),
        0xc4 => {
            let p = *store.get(1)?;
            if p & 0x1f != 1 {
                return None;
            }
            (p & 0x40 == 0, p & 0x20 == 0, 3)
        }
        _ => return None,
    };
    let (op, modrm, sib) = (
        *store.get(op_at)?,
        *store.get(op_at + 1)?,
        *store.get(op_at + 2)?,
    );
    if op != 0x11 || modrm != 0x04 || store.len() != op_at + 3 {
        return None;
    }
    // No base (mode 0, base 101), or the stack pointer as base, or rsp as index (none).
    if sib & 7 == 5 || (sib & 7 == 4 && !b) || (sib >> 3 & 7 == 4 && !x) {
        return None;
    }
    Some([0x48 | u8::from(x) << 1 | u8::from(b), 0x8d, 0x04, sib])
}

/// Split the store before saving registers: the write handler resumes
/// either 128-bit store with the title's original stack and registers.
fn stub(store: &[u8], at: usize, to: usize, marks: usize) -> Option<(Vec<u8>, usize)> {
    let lea = lea_of(store)?;
    let mut code = crate::wide::split_store(store, at as u64, to as u64)?;
    let stores_end = code.len();
    code.push(0x50); // push rax
    code.extend(lea); // lea rax, [base+index]
    code.extend([0x52, 0x51, 0x41, 0x53]); // push rdx, rcx, r11
    code.extend([0x48, 0xba]); // mov rdx, marks
    code.extend((marks as u64).to_le_bytes());
    code.extend([0x48, 0x2b, 0x02]); // sub rax, [rdx] (heap start)
    code.extend([0x49, 0x89, 0xc3]); // mov r11, rax
    code.extend([0x49, 0x83, 0xc3, 31]); // add r11, 31 (last byte)
    code.extend([0x49, 0x83, 0xe3, 0xe0]); // and r11, -32 (last slot start)
    code.extend([0x48, 0x83, 0xe0, 0xe0]); // and rax, -32 (first slot start)
    code.extend([0x4c, 0x39, 0xd8, 0x75, 7]); // cmp rax, r11; jne mark
    code.extend([0x49, 0xc7, 0xc3, 0xff, 0xff, 0xff, 0xff]); // mov r11, -1
    let mark = code.len();
    code.extend([0x48, 0x3b, 0x42, 0x08, 0x73, 24]); // cmp rax, [rdx+8]; jae next
    code.extend([0x48, 0xc1, 0xe8, 5]); // shr rax, 5 (slot)
    code.extend([0x48, 0x8b, 0x4a, 0x10]); // mov rcx, [rdx+16]
    code.extend([0xc6, 0x04, 0x01, 1]); // mov byte [rcx+rax], 1
    code.extend([0x48, 0xc1, 0xe8, 7]); // shr rax, 7 (chunk)
    code.extend([0x48, 0x8b, 0x4a, 0x18]); // mov rcx, [rdx+24]
    code.extend([0xc6, 0x04, 0x01, 1]); // mov byte [rcx+rax], 1
    // Aligned stores touch just one slot. An unaligned store can touch
    // a second slot, including when it starts just below the heap.
    code.extend([0x49, 0x83, 0xfb, 0xff, 0x74, 12]); // cmp r11, -1; je done
    code.extend([0x4c, 0x89, 0xd8]); // mov rax, r11
    code.extend([0x49, 0xc7, 0xc3, 0xff, 0xff, 0xff, 0xff]); // mov r11, -1
    code.extend([0xeb, (mark as isize - code.len() as isize - 2) as i8 as u8]);
    code.extend([0x41, 0x5b, 0x59, 0x5a, 0x58]); // pop r11, rcx, rdx, rax
    code.extend(crate::masked::jump(
        (to + code.len()) as u64,
        5,
        (at + store.len()) as u64,
    )?);
    Some((code, stores_end))
}

const PAGE: usize = 4096;
static FAULTS: OnceLock<Vec<FaultSite>> = OnceLock::new();

unsafe extern "system" fn remap_fault(info: *mut EXCEPTION_POINTERS) -> i32 {
    if let Some(sites) = FAULTS.get() {
        // SAFETY: Windows supplies the context. The guard handler runs first.
        unsafe { startup_patch::remap(info, sites) };
    }
    EXCEPTION_CONTINUE_SEARCH
}

/// Installs the known table stores and returns the windows owned by this
/// rewrite. The wide-store pass must leave those windows alone.
///
/// # Safety
/// The image stays mapped, no title code has run, and the cells at `marks`
/// remain live for the lifetime of the image.
pub unsafe fn apply(base: usize, size: usize, marks: usize) -> anyhow::Result<Vec<Range<u64>>> {
    // SAFETY: the same preconditions apply to the fixed sites.
    unsafe { install(base, size, marks, &SITES) }
}

unsafe fn install(
    base: usize,
    size: usize,
    marks: usize,
    sites: &[Site],
) -> anyhow::Result<Vec<Range<u64>>> {
    // Preflight the complete set before publishing any jump. The sync can
    // rely on the marks only if every site was installed.
    for site in sites {
        if site
            .window
            .checked_add(site.bytes.len())
            .is_none_or(|end| end > size)
        {
            return Ok(Vec::new());
        }
        let mut found = vec![0u8; site.bytes.len()];
        let mut done = 0;
        // SAFETY: the kernel checks the image range and writes to our buffer.
        if unsafe {
            ReadProcessMemory(
                GetCurrentProcess(),
                (base + site.window) as _,
                found.as_mut_ptr().cast(),
                found.len(),
                &mut done,
            )
        } == 0
            || found != site.bytes
        {
            tracing::warn!(
                rva = format_args!("{:#x}", site.window),
                "descriptor store bytes differ; page scanning retained"
            );
            return Ok(Vec::new());
        }
    }
    let mut pool = std::ptr::null_mut();
    let nearby = base & !0xffff;
    for hint in (1..=1024).map(|i| i << 20).flat_map(|distance| {
        [nearby.checked_sub(distance), nearby.checked_add(distance)]
            .into_iter()
            .flatten()
    }) {
        // SAFETY: private allocation within rel32 reach of the main image.
        pool = unsafe { VirtualAlloc(hint as _, PAGE, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE) };
        if !pool.is_null() {
            break;
        }
    }
    if pool.is_null() {
        return Ok(Vec::new());
    }
    let pool = Pool(pool);
    let mut edits = Vec::new();
    let mut faults = Vec::new();
    let mut used = 0;
    for site in sites {
        let at = base + site.window + site.store;
        let to = pool.0 as usize + used;
        let store = &site.bytes[site.store..site.store + site.len];
        let (Some((code, stores_end)), Some(jump)) = (
            stub(store, at, to, marks),
            crate::masked::jump(at as u64, site.len, to as u64),
        ) else {
            return Ok(Vec::new());
        };
        assert!(used + code.len() <= PAGE);
        // SAFETY: write into a private, writable allocation.
        unsafe { std::ptr::copy_nonoverlapping(code.as_ptr(), to as *mut u8, code.len()) };
        used += code.len().next_multiple_of(16);
        edits.push((
            at,
            jump,
            (base + site.window) as u64..(base + site.window + site.bytes.len()) as u64,
        ));
        faults.push(FaultSite {
            stub: to as u64..(to + stores_end) as u64,
            title: at as u64,
            saved: &[],
        });
    }
    let mut old = 0;
    // SAFETY: the pool is private until a jump can reach it.
    let executable = unsafe {
        VirtualProtect(pool.0, PAGE, PAGE_EXECUTE_READ, &mut old) != 0
            && FlushInstructionCache(GetCurrentProcess(), pool.0, used) != 0
    };
    if !executable {
        return Ok(Vec::new());
    }
    // Keep the fault metadata and RX code for the life of the title.
    if FAULTS.set(faults).is_err() {
        #[cfg(not(test))]
        return Ok(Vec::new());
    } else {
        // SAFETY: the handler uses immutable metadata published above.
        if unsafe { AddVectoredExceptionHandler(0, Some(remap_fault)) }.is_null() {
            return Ok(Vec::new());
        }
    }
    std::mem::forget(pool);
    let mut installed = Vec::new();
    for (at, jump, window) in edits {
        // SAFETY: the verified code is idle and the replacement fits a page.
        if unsafe { startup_patch::publish(at, &jump) }? {
            installed.push(window);
        }
    }
    Ok(installed)
}

struct Pool(*mut std::ffi::c_void);

impl Drop for Pool {
    fn drop(&mut self) {
        // SAFETY: still a private allocation with no installed references.
        unsafe { VirtualFree(self.0, 0, MEM_RELEASE) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_sites_lea_their_stores_address() {
        let leas: Vec<[u8; 4]> = SITES
            .iter()
            .map(|s| lea_of(&s.bytes[s.store..s.store + s.len]).unwrap())
            .collect();
        assert_eq!(leas[0], [0x48, 0x8d, 0x04, 0x31]); // lea rax, [rcx+rsi]
        assert_eq!(leas[1], [0x48, 0x8d, 0x04, 0x31]);
        assert_eq!(leas[2], [0x49, 0x8d, 0x04, 0x31]); // [r9+rsi]
        assert_eq!(leas[3], [0x4b, 0x8d, 0x04, 0x01]); // [r9+r8]
        assert_eq!(leas[4], [0x4a, 0x8d, 0x04, 0x07]); // [rdi+r8]
        assert_eq!(leas[5], [0x4b, 0x8d, 0x04, 0x02]); // [r10+r8]
        // Each window: the store follows the record's load and is followed
        // by a movzx and the cmp that sets the flags.
        for s in &SITES {
            let after = &s.bytes[s.store + s.len..];
            let movzx = if after[0] == 0x0f { 0 } else { 1 };
            assert_eq!(&after[movzx..movzx + 2], &[0x0f, 0xb6]);
            assert_eq!(after[after.len() - 2], 0x39, "cmp");
        }
        // A store with a displacement, of the stack, of another kind: not taken.
        assert!(lea_of(&[0xc5, 0xfc, 0x11, 0x44, 0x31, 0x08]).is_none());
        assert!(lea_of(&[0xc5, 0xfc, 0x11, 0x04, 0x24]).is_none());
        assert!(lea_of(&[0xc5, 0xfc, 0x10, 0x04, 0x31]).is_none());
    }

    /// The marks `victory-gpu` keeps, for a test.
    #[repr(C)]
    struct Marks {
        start: u64,
        len: u64,
        slots: *mut u8,
        chunks: *mut u8,
    }

    static GUARDED_PAGE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

    unsafe extern "system" fn open_test_page(info: *mut EXCEPTION_POINTERS) -> i32 {
        use std::sync::atomic::Ordering::Relaxed;
        use windows_sys::Win32::Foundation::EXCEPTION_ACCESS_VIOLATION;
        use windows_sys::Win32::System::Diagnostics::Debug::EXCEPTION_CONTINUE_EXECUTION;
        // SAFETY: supplied by Windows.
        let record = unsafe { &*(*info).ExceptionRecord };
        let page = GUARDED_PAGE.load(Relaxed);
        if page != 0
            && record.ExceptionCode == EXCEPTION_ACCESS_VIOLATION
            && record.NumberParameters >= 2
            && record.ExceptionInformation[0] == 1
            && (page..page + PAGE).contains(&record.ExceptionInformation[1])
        {
            let mut old = 0;
            // SAFETY: the test reserved this page and never releases it while armed.
            if unsafe { VirtualProtect(page as *const _, PAGE, PAGE_READWRITE, &mut old) } != 0 {
                GUARDED_PAGE.store(0, Relaxed);
                return EXCEPTION_CONTINUE_EXECUTION;
            }
        }
        EXCEPTION_CONTINUE_SEARCH
    }

    /// Each site's window in code of a test's, twice, one copy hooked, both
    /// run: the record is stored as before, the registers the window leaves
    /// are the same, and the slot and its chunk are marked when the table is
    /// in the heap, not otherwise.
    #[test]
    fn a_hooked_site_stores_and_marks_its_slot() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RESERVE, PAGE_EXECUTE_READWRITE, VirtualAlloc,
        };
        // mov to, from (64 bits)
        let mov = |to: u8, from: u8| {
            [
                0x48 | (from >> 3) << 2 | (to >> 3),
                0x89,
                0xc0 | (from & 7) << 3 | (to & 7),
            ]
        };
        for site in &SITES {
            let store = &site.bytes[site.store..site.store + site.len];
            let lea = lea_of(store).unwrap();
            let base = lea[3] & 7 | (lea[0] & 1) << 3;
            let index = lea[3] >> 3 & 7 | (lea[0] & 2) << 2;
            let load_at = site.store
                - if site.bytes[site.store - 5] == 0xc4 {
                    5
                } else {
                    4
                };
            let load = &site.bytes[load_at..site.store];
            let load_reg = if load[0] == 0xc4 {
                load[4] & 7 | 8
            } else {
                load[3] & 7
            };
            // fn(offset: rcx, table: rdx, record: r8, scratch: r9) -> rax:
            // the store's address as the window leaves its registers.
            let mut code: Vec<u8> = vec![0x56, 0x57, 0x53]; // push rsi ; push rdi ; push rbx
            code.extend(mov(11, 1)); // r11 = offset
            code.extend(mov(3, 2)); // rbx = table
            code.extend(mov(6, 8)); // rsi = record
            for scratch in [7, 10, 2] {
                code.extend(mov(scratch, 9)); // rdi, r10, rdx: what the movzx reads
            }
            // The window's `and` masks the offset's register; the table is in
            // the other. (The record first: rsi may be one of them.)
            let rex = usize::from(site.bytes[0] == 0x41);
            let offset = site.bytes[rex + 1] & 7 | if rex == 1 { 8 } else { 0 };
            assert!(offset == base || offset == index);
            let table = if offset == base { index } else { base };
            code.extend(mov(load_reg, 6));
            code.extend(mov(table, 3));
            code.extend(mov(offset, 11));
            let at = code.len();
            code.extend(site.bytes);
            code.extend(lea); // lea rax, [base+index]
            code.extend([0x5b, 0x5f, 0x5e]); // pop rbx ; pop rdi ; pop rsi
            code.extend([0xc5, 0xf8, 0x77, 0xc3]); // vzeroupper ; ret
            type F = unsafe extern "system" fn(u64, *mut u8, *const u8, *mut u8) -> u64;
            // SAFETY: fresh memory for code and data, one copy changed before it runs.
            unsafe {
                let place = VirtualAlloc(
                    std::ptr::null(),
                    0x10000,
                    MEM_COMMIT | MEM_RESERVE,
                    PAGE_EXECUTE_READWRITE,
                ) as usize;
                assert_ne!(place, 0);
                let (plain, hooked) = (place + 0x100 - at, place + 0x1100 - at);
                std::ptr::copy_nonoverlapping(code.as_ptr(), plain as *mut u8, code.len());
                std::ptr::copy_nonoverlapping(code.as_ptr(), hooked as *mut u8, code.len());
                let heap = VirtualAlloc(
                    std::ptr::null(),
                    0x10000,
                    MEM_COMMIT | MEM_RESERVE,
                    PAGE_EXECUTE_READWRITE,
                ) as *mut u8;
                let scratch = [0u8; 64];
                let (mut slots, mut chunks) =
                    (vec![0u8; 0x8000 / 32], vec![0u8; 0x8000 / 32 / 128]);
                let marks = Box::new(Marks {
                    start: heap as u64 + 0x2000,
                    len: 0x8000,
                    slots: slots.as_mut_ptr(),
                    chunks: chunks.as_mut_ptr(),
                });
                let mine = [Site {
                    window: 0x1100,
                    ..*site
                }];
                assert_eq!(
                    install(place, 0x10000, &raw const *marks as usize, &mine)
                        .unwrap()
                        .len(),
                    1
                );
                assert_eq!(*((place + 0x1100 + site.store) as *const u8), 0xe9);
                let plain = std::mem::transmute::<usize, F>(plain);
                let hooked = std::mem::transmute::<usize, F>(hooked);
                let record: [u8; 32] = std::array::from_fn(|k| k as u8 ^ 0x5a);
                // In the heap: slot 0x40 + 3 of its start.
                let table = heap.add(0x2000 + 0x40 * 32);
                let first = plain(
                    0x60 | 0x8000_0000,
                    table.add(0x4000),
                    record.as_ptr(),
                    scratch.as_ptr().cast_mut(),
                );
                let second = hooked(
                    0x60 | 0x8000_0000,
                    table,
                    record.as_ptr(),
                    scratch.as_ptr().cast_mut(),
                );
                assert_eq!(
                    first.wrapping_sub(0x4000),
                    second,
                    "the registers the window leaves"
                );
                assert_eq!(std::slice::from_raw_parts(table.add(0x60), 32), &record);
                assert_eq!(std::slice::from_raw_parts(table.add(0x4060), 32), &record);
                let slot = 0x40 + 3;
                assert_eq!(slots[slot], 1);
                assert_eq!(slots.iter().filter(|&&b| b != 0).count(), 1);
                assert_eq!(chunks[slot / 128], 1);
                // Before the heap: stored, not marked.
                slots.fill(0);
                chunks.fill(0);
                let below = heap.add(0x1000);
                hooked(0x20, below, record.as_ptr(), scratch.as_ptr().cast_mut());
                assert_eq!(std::slice::from_raw_parts(below.add(0x20), 32), &record);
                assert!(slots.iter().chain(&chunks).all(|&b| b == 0));
                // Original-width YMM store would cross into the guarded page.
                // The split store resumes and marks both unaligned slots.
                use std::sync::atomic::Ordering::Relaxed;
                use windows_sys::Win32::System::Diagnostics::Debug::RemoveVectoredExceptionHandler;
                use windows_sys::Win32::System::Memory::PAGE_READONLY;
                let page = heap.add(0x4000);
                let mut old = 0;
                let handler = AddVectoredExceptionHandler(1, Some(open_test_page));
                assert!(!handler.is_null());
                assert_ne!(
                    VirtualProtect(page.cast(), PAGE, PAGE_READONLY, &mut old),
                    0
                );
                GUARDED_PAGE.store(page as usize, Relaxed);
                hooked(
                    0,
                    page.sub(16),
                    record.as_ptr(),
                    scratch.as_ptr().cast_mut(),
                );
                let fault_handled = GUARDED_PAGE.swap(0, Relaxed) == 0;
                RemoveVectoredExceptionHandler(handler);
                assert!(fault_handled);
                assert_eq!(std::slice::from_raw_parts(page.sub(16), 32), &record);
                let end = (0x4000 - 0x2000) / 32;
                assert_eq!(&slots[end - 1..end + 1], &[1, 1]);
                assert_eq!(slots.iter().filter(|&&b| b != 0).count(), 2);
                VirtualFree(heap.cast(), 0, MEM_RELEASE);
                VirtualFree(place as *mut _, 0, MEM_RELEASE);
                drop(marks);
            }
        }
    }

    #[test]
    fn other_bytes_are_left_alone() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RESERVE, PAGE_EXECUTE_READWRITE, VirtualAlloc,
        };
        // SAFETY: fresh memory of our own.
        unsafe {
            let place = VirtualAlloc(
                std::ptr::null(),
                0x10000,
                MEM_COMMIT | MEM_RESERVE,
                PAGE_EXECUTE_READWRITE,
            ) as *mut u8;
            let mut bytes = SITES[0].bytes.to_vec();
            bytes[3] = 0x7e; // and ecx, 7EE0h
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), place.add(0x100), bytes.len());
            let mine = [Site {
                window: 0x100,
                ..SITES[0]
            }];
            assert_eq!(install(place as usize, 0x10000, 0, &mine).unwrap(), []);
            assert_eq!(
                std::slice::from_raw_parts(place.add(0x100), bytes.len()),
                &bytes[..]
            );
            let outside = [Site {
                window: 0xfff0,
                ..SITES[0]
            }];
            assert!(
                install(place as usize, 0x10000, 0, &outside)
                    .unwrap()
                    .is_empty()
            );
        }
    }
}
