//! Read descriptor headers in place, comparing each one as it is read.
//! A disappearing title mapping falls back to the kernel snapshot path.

use std::sync::OnceLock;
use std::sync::atomic::AtomicU64;

use windows_sys::Win32::Foundation::{
    EXCEPTION_ACCESS_VIOLATION, STATUS_GUARD_PAGE_VIOLATION, STATUS_IN_PAGE_ERROR,
};
use windows_sys::Win32::System::Diagnostics::Debug::{
    AddVectoredExceptionHandler, EXCEPTION_CONTINUE_EXECUTION, EXCEPTION_CONTINUE_SEARCH,
    EXCEPTION_POINTERS,
};

use super::{VIEW_TAG, host::Candidate};

// Win64 leaf: rcx points to title headers, rdx to the ready keys, r8 to
// the occupancy bitmap, r9 to the output. The stack args supply the
// first index and exclusive end. Only two instruction addresses can
// fail, and neither the stack pointer nor a nonvolatile register moves.
std::arch::global_asm!(
    ".globl victory_scan_headers, victory_scan_tag, victory_scan_key, victory_scan_failed",
    "victory_scan_headers:",
    "mov qword ptr [rsp + 8], r9",
    "mov r10, qword ptr [rsp + 40]",
    "mov r11, qword ptr [rsp + 48]",
    "2:",
    "victory_scan_tag:",
    "mov eax, dword ptr [rcx]",
    "movd xmm0, eax",
    "cmp eax, {tag}",
    "je 3f",
    "bt qword ptr [r8], r10",
    "jnc 5f",
    "3:",
    "victory_scan_key:",
    "mov rax, qword ptr [rcx + 4]",
    "movd dword ptr [rsp + 16], xmm0",
    "cmp dword ptr [rsp + 16], {tag}",
    "jne 4f",
    "cmp rax, qword ptr [rdx + r10 * 8]",
    "je 5f",
    "4:",
    "mov dword ptr [r9], r10d",
    "movd dword ptr [r9 + 4], xmm0",
    "mov qword ptr [r9 + 8], rax",
    "add r9, 16",
    "5:",
    "add rcx, 32",
    "inc r10",
    "cmp r10, r11",
    "jb 2b",
    "mov rax, r9",
    "sub rax, qword ptr [rsp + 8]",
    "shr rax, 4",
    "ret",
    "victory_scan_failed:",
    "mov rax, -1",
    "ret",
    ".globl victory_match_header, victory_match_tag, victory_match_key, victory_match_failed",
    "victory_match_header:",
    "xor eax, eax",
    "victory_match_tag:",
    "cmp edx, dword ptr [rcx]",
    "jne victory_match_failed",
    "victory_match_key:",
    "cmp r8, qword ptr [rcx + 4]",
    "sete al",
    "victory_match_failed:",
    "ret",
    tag = const VIEW_TAG,
);

unsafe extern "system" {
    fn victory_scan_headers(
        source: *const u8,
        ready: *const AtomicU64,
        occupied: *const AtomicU64,
        out: *mut Candidate,
        first: usize,
        end: usize,
    ) -> usize;
    fn victory_scan_tag();
    fn victory_scan_key();
    fn victory_scan_failed();
    fn victory_match_header(source: *const u8, tag: u32, key: u64) -> u32;
    fn victory_match_tag();
    fn victory_match_key();
    fn victory_match_failed();
}

unsafe extern "system" fn fault(info: *mut EXCEPTION_POINTERS) -> i32 {
    // SAFETY: the OS passes live exception structures.
    let (record, context) = unsafe { (&*(*info).ExceptionRecord, &mut *(*info).ContextRecord) };
    if matches!(
        record.ExceptionCode,
        EXCEPTION_ACCESS_VIOLATION | STATUS_GUARD_PAGE_VIOLATION | STATUS_IN_PAGE_ERROR
    ) && record.NumberParameters >= 2
        && record.ExceptionInformation[0] == 0
    {
        let at = context.Rip as *const ();
        if [victory_scan_tag as *const (), victory_scan_key as *const ()].contains(&at) {
            context.Rip = victory_scan_failed as *const () as u64;
            return EXCEPTION_CONTINUE_EXECUTION;
        }
        if [
            victory_match_tag as *const (),
            victory_match_key as *const (),
        ]
        .contains(&at)
        {
            context.Rip = victory_match_failed as *const () as u64;
            return EXCEPTION_CONTINUE_EXECUTION;
        }
    }
    EXCEPTION_CONTINUE_SEARCH
}

pub(super) fn on() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var_os("VICTORY_SYNC_IN_PLACE").is_none_or(|v| v != "0") && install()
    })
}

fn install() -> bool {
    static INSTALLED: OnceLock<bool> = OnceLock::new();
    // SAFETY: the handler only intercepts the guarded title reads above.
    *INSTALLED.get_or_init(|| !unsafe { AddVectoredExceptionHandler(1, Some(fault)) }.is_null())
}

/// Revalidate a captured header with a fault-contained read while its host
/// slot is locked. An unreadable mapping counts as a changed header.
pub(super) fn matches(source: *const u8, [tag, index, generation]: [u32; 3]) -> bool {
    debug_assert!(on());
    let key = u64::from(generation) << 32 | u64::from(index);
    // SAFETY: the installed handler covers both reads before the function
    // changes any stack or nonvolatile registers. A fault returns zero.
    unsafe { victory_match_header(source, tag, key) != 0 }
}

pub(super) fn changed(
    source: *const u8,
    ready: &[AtomicU64],
    occupied: &[AtomicU64],
    first: u32,
    last: u32,
) -> Option<Vec<Candidate>> {
    let (first, last) = (first as usize, last as usize);
    assert!(first < last && last <= ready.len() && last.div_ceil(64) <= occupied.len());
    let mut result = Vec::with_capacity(last - first);
    // SAFETY: the checked indices bound the atomic loads (acquire loads
    // on x86). At most one candidate can be written per slot. The source
    // is read at the instructions handled by the installed exception hook.
    let count = unsafe {
        victory_scan_headers(
            source,
            ready.as_ptr(),
            occupied.as_ptr(),
            result.as_mut_ptr(),
            first,
            last,
        )
    };
    if count == usize::MAX {
        return None;
    }
    // SAFETY: the assembler writes all fields of these candidates.
    unsafe { result.set_len(count) };
    Some(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn in_place_compares_headers_and_handles_unmapped_reads() {
        assert!(install());
        let mut records = [[0u32; 8]; 5];
        records[1][..3].copy_from_slice(&[VIEW_TAG, 1, 2]);
        records[2][..3].copy_from_slice(&[VIEW_TAG, 3, 4]);
        records[3][..3].copy_from_slice(&[0, 5, 6]);
        let ready = std::array::from_fn::<_, 5, _>(|index| {
            AtomicU64::new(if index == 1 { (2 << 32) + 1 } else { 0 })
        });
        let occupied = [AtomicU64::new(1 << 3)];
        assert_eq!(
            changed(records[1..].as_ptr().cast(), &ready, &occupied, 1, 5).unwrap(),
            [
                Candidate {
                    index: 2,
                    record: [VIEW_TAG, 3, 4]
                },
                Candidate {
                    index: 3,
                    record: [0, 5, 6]
                },
            ],
        );
        assert!(changed(std::ptr::null(), &ready, &occupied, 0, 5).is_none());
    }

    #[test]
    fn captured_header_validation_rejects_rewrites_and_unmapped_memory() {
        if !on() {
            return;
        }
        let record = [VIEW_TAG, 17, 3];
        let at = record.as_ptr().cast();
        assert!(matches(at, record));
        assert!(!matches(at, [0, 17, 3]));
        assert!(!matches(at, [VIEW_TAG, 17, 4]));
        assert!(!matches(std::ptr::null(), record));
        // The tag is readable, but the adjacent key crosses out of the pool.
        use windows_sys::Win32::System::Memory::{MEM_DECOMMIT, VirtualFree};
        let base = crate::watch::tests::pool(2);
        let end = (base + crate::watch::PAGE - 4) as *mut u32;
        // SAFETY: the reserved pool has two blocks; leave the second
        // inaccessible, with the record's tag in the first.
        unsafe {
            assert_ne!(
                VirtualFree(end.add(1).cast(), crate::watch::PAGE, MEM_DECOMMIT),
                0
            );
            end.write(VIEW_TAG);
        };
        assert!(!matches(end.cast(), record));
        crate::watch::tests::free(base);
    }
}
