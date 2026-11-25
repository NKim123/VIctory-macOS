//! Startup publishing and fault translation shared by title instruction rewrites.

use std::ops::Range;

use anyhow::{Result, bail};
use windows_sys::Win32::Foundation::{
    EXCEPTION_ACCESS_VIOLATION, STATUS_GUARD_PAGE_VIOLATION, STATUS_IN_PAGE_ERROR,
};
use windows_sys::Win32::System::Diagnostics::Debug::{
    EXCEPTION_POINTERS, FlushInstructionCache, ReadProcessMemory,
};
use windows_sys::Win32::System::Memory::{PAGE_EXECUTE_READWRITE, VirtualProtect};
use windows_sys::Win32::System::Threading::GetCurrentProcess;

/// Bytes written by the stub's pushes, in increasing stack address order.
pub(crate) struct FaultSite {
    pub(crate) stub: Range<u64>,
    pub(crate) title: u64,
    pub(crate) saved: &'static [Saved],
}

#[allow(dead_code)] // The startup rewrites may be built independently.
pub(crate) enum Saved {
    Rax,
    Rcx,
    Rdx,
    Flags,
    ReturnAddress,
}

/// Translate an unhandled data access in a stub before Windows looks up
/// the title's stack unwind data and exception handler. A retry of the
/// title PC re-enters the rewrite from its original register state.
///
/// # Safety
/// `info` is supplied by Windows to the vectored exception handler.
pub(crate) unsafe fn remap(info: *mut EXCEPTION_POINTERS, sites: &[FaultSite]) {
    // SAFETY: the OS supplies the structures for this exception.
    let (record, context) = unsafe { (&mut *(*info).ExceptionRecord, &mut *(*info).ContextRecord) };
    let min_parameters = match record.ExceptionCode {
        EXCEPTION_ACCESS_VIOLATION | STATUS_GUARD_PAGE_VIOLATION => 2,
        STATUS_IN_PAGE_ERROR => 3,
        _ => return,
    };
    if record.NumberParameters < min_parameters || !matches!(record.ExceptionInformation[0], 0 | 1)
    {
        return;
    }
    let Some(site) = sites.iter().find(|site| site.stub.contains(&context.Rip)) else {
        return;
    };
    let mut saved = [0u64; 5];
    let size = site.saved.len() * size_of::<u64>();
    if size != 0 {
        let mut done = 0;
        // SAFETY: the kernel checks the saved stack range before reading.
        if unsafe {
            ReadProcessMemory(
                GetCurrentProcess(),
                context.Rsp as *const _,
                saved.as_mut_ptr().cast(),
                size,
                &mut done,
            )
        } == 0
            || done != size
        {
            return;
        }
    }
    for (register, value) in site.saved.iter().zip(saved) {
        match register {
            Saved::Rax => context.Rax = value,
            Saved::Rcx => context.Rcx = value,
            Saved::Rdx => context.Rdx = value,
            Saved::Flags => context.EFlags = value as u32,
            Saved::ReturnAddress => {}
        }
    }
    context.Rsp += size as u64;
    context.Rip = site.title;
    record.ExceptionAddress = site.title as *mut _;
    // The speculative byte read stands in for a read-modify-write. Keep
    // the OS-supplied parameters of guard notifications and paging errors.
    if record.ExceptionCode == EXCEPTION_ACCESS_VIOLATION {
        record.ExceptionInformation[0] = 1;
    }
}

/// Publish one verified patch within a single page of an idle title image.
/// A refused first protection change leaves the bytes alone. After
/// publication, failure must be returned to the caller before title entry.
///
/// # Safety
/// `site` must name a mapped code range owned by the caller. No title
/// thread may run until this function succeeds.
pub(crate) unsafe fn publish(site: usize, bytes: &[u8]) -> Result<bool> {
    let mut old = 0;
    // SAFETY: the caller owns the idle page and old is a valid output pointer.
    if unsafe {
        VirtualProtect(
            site as *const _,
            bytes.len(),
            PAGE_EXECUTE_READWRITE,
            &mut old,
        )
    } == 0
    {
        return Ok(false);
    }
    // SAFETY: the single page is now writable and still idle.
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), site as *mut u8, bytes.len()) };
    let mut ignored = 0;
    // SAFETY: restores the protection of the page we just made writable.
    if unsafe { VirtualProtect(site as *const _, bytes.len(), old, &mut ignored) } == 0 {
        bail!(
            "restoring title code protection: {}",
            std::io::Error::last_os_error()
        );
    }
    // SAFETY: flushes the range just changed in this process.
    if unsafe { FlushInstructionCache(GetCurrentProcess(), site as *const _, bytes.len()) } == 0 {
        bail!("flushing title code: {}", std::io::Error::last_os_error());
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows_sys::Win32::System::Diagnostics::Debug::{CONTEXT, EXCEPTION_RECORD};

    #[test]
    fn data_paging_errors_keep_their_underlying_status_and_parameters() {
        let stack = [0x24u64, 0x202, 0x3030];
        let sites = [FaultSite {
            stub: 0x1000..0x1008,
            title: 0x2000,
            saved: &[Saved::Rax, Saved::Flags, Saved::ReturnAddress],
        }];
        // SAFETY: an exception record may be zero-initialized.
        let mut record: EXCEPTION_RECORD = unsafe { std::mem::zeroed() };
        // SAFETY: the context uses zero as its default value.
        let mut context: CONTEXT = unsafe { std::mem::zeroed() };
        record.ExceptionCode = STATUS_IN_PAGE_ERROR;
        record.NumberParameters = 3;
        record.ExceptionAddress = 0x1004 as *mut _;
        record.ExceptionInformation[..3].copy_from_slice(&[8, 0x9870, 0xc000_009c]);
        context.Rip = 0x1004;
        context.Rsp = stack.as_ptr() as u64;
        let mut info = EXCEPTION_POINTERS {
            ExceptionRecord: &mut record,
            ContextRecord: &mut context,
        };
        // SAFETY: the exception structures and stack above are live.
        unsafe { remap(&mut info, &sites) };
        assert_eq!(context.Rip, 0x1004, "instruction fetch is not a data fault");

        record.ExceptionInformation[0] = 0;
        // SAFETY: same live structures, now describing a data read.
        unsafe { remap(&mut info, &sites) };
        assert_eq!(record.ExceptionCode, STATUS_IN_PAGE_ERROR);
        assert_eq!(record.ExceptionAddress as usize, 0x2000);
        assert_eq!(record.NumberParameters, 3);
        assert_eq!(&record.ExceptionInformation[..3], &[0, 0x9870, 0xc000_009c]);
        assert_eq!(context.Rip, 0x2000);
        assert_eq!(context.Rax, stack[0]);
        assert_eq!(context.EFlags, stack[1] as u32);
        assert_eq!(
            context.Rsp,
            stack.as_ptr() as u64 + size_of_val(&stack) as u64
        );
    }
}
