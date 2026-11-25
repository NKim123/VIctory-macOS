use std::sync::OnceLock;

use anyhow::{Result, bail};
use windows_sys::Win32::System::Diagnostics::Debug::{
    AddVectoredExceptionHandler, EXCEPTION_POINTERS, FlushInstructionCache,
};
use windows_sys::Win32::System::Memory::{
    MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, MEMORY_BASIC_INFORMATION, PAGE_EXECUTE_READ,
    PAGE_READWRITE, VirtualAlloc, VirtualFree, VirtualProtect, VirtualQuery,
};
use windows_sys::Win32::System::Threading::GetCurrentProcess;

use super::*;
use crate::LoadedImage;
use crate::startup_patch::{self, FaultSite, Saved};

static FAULT_SITES: OnceLock<Vec<FaultSite>> = OnceLock::new();

unsafe extern "system" fn remap_zero_fault(info: *mut EXCEPTION_POINTERS) -> i32 {
    if let Some(sites) = FAULT_SITES.get() {
        // SAFETY: Windows is dispatching the exception. GPU guards get first chance.
        unsafe { startup_patch::remap(info, sites) };
    }
    0 // Continue to the title's usual exception dispatch.
}

/// The opted-in instruction sites from the original main-image build.
pub struct Patches(Vec<Site>);

impl Patches {
    pub fn from_env(prepared: &PreparedImage) -> Self {
        let mut selected = Vec::new();
        for (key, zero) in [
            ("VICTORY_PATCH_CLZERO", true),
            ("VICTORY_PATCH_WAITS", false),
        ] {
            let Some(value) = std::env::var_os(key) else {
                continue;
            };
            let Some(list) = value
                .to_str()
                .and_then(|v| sites(v, prepared.original_hash))
            else {
                tracing::warn!(
                    key,
                    "AMD patch requires the original BLAKE3 and a hex RVA list"
                );
                continue;
            };
            for rva in list {
                if let Some(site) = verify(prepared, rva, zero) {
                    selected.push(site);
                } else {
                    tracing::warn!(
                        key,
                        rva = format_args!("{rva:#x}"),
                        "unsupported AMD patch site"
                    );
                }
            }
        }
        Self(selected)
    }

    /// Applies these rewrites after the other main-image startup patches.
    ///
    /// # Safety
    /// This plan must come from the preparation of `image`. The caller
    /// must own its idle code exclusively and abort startup on error.
    /// No title code may have run. Apply this after all other code rewrites.
    pub unsafe fn apply(self, image: &LoadedImage) -> Result<()> {
        // The main image runs for the life of the process. Bound the retained
        // executable allocation to one pool even if a second load is attempted.
        static STARTUP: OnceLock<()> = OnceLock::new();
        if !self.0.is_empty() && STARTUP.set(()).is_ok() {
            // SAFETY: the caller owns the idle image.
            let (waits, zeroes) = unsafe { install(image.base(), &self.0) }?;
            tracing::info!(waits, zeroes, "startup AMD instruction patches");
        }
        Ok(())
    }
}

/// A private writable allocation, released unless code in the main image
/// has been prepared to reference it. Never writable and executable at once.
struct Pool {
    address: usize,
    size: usize,
}

impl Pool {
    fn near(base: usize, count: usize) -> Option<Self> {
        let base = base & !0xffff;
        let size = (count * 128).next_multiple_of(PAGE);
        for step in (1..=1024).map(|i| i << 20) {
            for at in [base.checked_sub(step), base.checked_add(step)]
                .into_iter()
                .flatten()
            {
                // SAFETY: requests an unoccupied range, or fails.
                let address = unsafe {
                    VirtualAlloc(at as _, size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE)
                } as usize;
                if address != 0 {
                    return Some(Self { address, size });
                }
            }
        }
        None
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        // SAFETY: this owns a still private reservation.
        unsafe { VirtualFree(self.address as _, 0, MEM_RELEASE) };
    }
}

fn flush(at: usize, size: usize) -> Result<()> {
    // SAFETY: single-process cache flush, with a valid process pseudo-handle.
    if unsafe { FlushInstructionCache(GetCurrentProcess(), at as _, size) } == 0 {
        bail!(
            "flushing AMD patch code: {}",
            std::io::Error::last_os_error()
        );
    }
    Ok(())
}

fn protect(at: usize, size: usize, access: u32) -> Result<u32> {
    let mut old = 0;
    // SAFETY: the caller provides a current-process address; Windows checks
    // ownership and protections for the range, and old is a valid out pointer.
    if unsafe { VirtualProtect(at as _, size, access, &mut old) } == 0 {
        bail!(
            "protecting AMD patch code: {}",
            std::io::Error::last_os_error()
        );
    }
    Ok(old)
}

/// # Safety
/// The caller owns an idle mapped image matching the candidate sites.
unsafe fn install(base: usize, sites: &[Site]) -> Result<(usize, usize)> {
    let mut ready = Vec::new();
    for (i, site) in sites.iter().enumerate() {
        // SAFETY: the caller owns the idle image and its protection state.
        if rejects_overlap(sites, i) || !unsafe { unchanged(base, site) } {
            tracing::warn!(
                rva = format_args!("{:#x}", site.rva),
                "skipping AMD patch at overlapping, modified or inaccessible site"
            );
        } else {
            ready.push(site);
        }
    }

    let count = ready
        .iter()
        .filter(|s| matches!(s.change, Change::Zero { .. }))
        .count();
    let pool = (count != 0).then(|| Pool::near(base, count)).flatten();
    let mut used = 0;
    let mut edits = Vec::new();
    let mut faults = Vec::new();
    for site in ready {
        let at = base + site.rva;
        let bytes = match site.change {
            Change::Wait(bytes) => bytes.to_vec(),
            Change::Zero { next, len } => {
                let code = pool.as_ref().and_then(|pool| {
                    let stub = (pool.address + used) as u64;
                    // At the original PC an out-of-function JMP looks
                    // like an epilogue to Windows' unwind decoder.
                    let mut branch = vec![0x90];
                    branch.extend(jump(at as u64 + 1, len - 1, stub)?);
                    Some((branch, zero_stub(at as u64, stub, next, len)?))
                });
                let Some((branch, body)) = code else {
                    tracing::warn!(
                        rva = format_args!("{:#x}", site.rva),
                        "no reachable CLZERO stub"
                    );
                    continue;
                };
                let pool = pool.as_ref().unwrap();
                assert!(body.len() <= 128 && used + body.len() <= pool.size);
                // SAFETY: the allocation is private, with 128 bytes per stub.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        body.as_ptr(),
                        (pool.address + used) as *mut u8,
                        body.len(),
                    );
                }
                let stub = (pool.address + used) as u64;
                faults.push(FaultSite {
                    stub: stub + ZERO_STORES_START as u64..stub + ZERO_STORES_END as u64,
                    title: at as u64,
                    saved: &[Saved::Rax, Saved::Flags],
                });
                used += body.len();
                branch
            }
        };
        edits.push((at, bytes));
    }
    if let Some(pool) = pool.filter(|_| used != 0) {
        let available = protect(pool.address, pool.size, PAGE_EXECUTE_READ)
            .and_then(|_| flush(pool.address, used))
            .and_then(|_| {
                // SAFETY: the handler uses only immutable metadata after startup.
                if unsafe { AddVectoredExceptionHandler(0, Some(remap_zero_fault)) }.is_null() {
                    bail!("registering CLZERO stub fault translation");
                }
                if FAULT_SITES.set(faults).is_err() {
                    bail!("CLZERO stub fault translation already registered");
                }
                Ok(())
            });
        if let Err(error) = available {
            tracing::warn!(%error, "CLZERO stub unavailable");
            edits.retain(|(_, bytes)| bytes.len() == 3);
        } else {
            // The main image and its fault map live for the process lifetime.
            std::mem::forget(pool);
        }
    }

    let (mut waits, mut zeroes) = (0, 0);
    for (at, bytes) in edits {
        // SAFETY: the caller owns this idle image and the site lies in one page.
        if !unsafe { startup_patch::publish(at, &bytes) }? {
            tracing::warn!(
                rva = format_args!("{:#x}", at - base),
                "AMD patch site could not be written"
            );
            continue;
        }
        if bytes.len() == 3 {
            waits += 1;
        } else {
            zeroes += 1;
        }
    }
    Ok((waits, zeroes))
}

unsafe fn unchanged(base: usize, site: &Site) -> bool {
    let at = base + site.guard_rva;
    let end = at + site.expected.len();
    // Keep this check and each page protection within a single Windows page.
    if at / PAGE != (end - 1) / PAGE {
        return false;
    }
    let Some(region) = query(at) else {
        return false;
    };
    if region.State != MEM_COMMIT
        || region.Protect != PAGE_EXECUTE_READ
        || region.AllocationBase as usize != base
        || end > (region.BaseAddress as usize).saturating_add(region.RegionSize)
    {
        return false;
    }
    // SAFETY: the checked range is readable in the idle mapping.
    unsafe { std::slice::from_raw_parts(at as *const u8, site.expected.len()) == site.expected }
}

fn query(at: usize) -> Option<MEMORY_BASIC_INFORMATION> {
    // SAFETY: plain data, filled by VirtualQuery below.
    let mut region: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: region is a writable out structure of the requested size.
    let found = unsafe { VirtualQuery(at as _, &mut region, size_of_val(&region)) };
    (found != 0).then_some(region)
}

#[cfg(test)]
mod execution_tests {
    use super::*;
    use crate::amd_patch::tests::{BACKWARD, FORWARD};
    use std::sync::atomic::{AtomicI32, AtomicU32, AtomicUsize, Ordering::Relaxed};
    use windows_sys::Win32::Foundation::{
        EXCEPTION_ACCESS_VIOLATION, EXCEPTION_ILLEGAL_INSTRUCTION, STATUS_GUARD_PAGE_VIOLATION,
    };
    use windows_sys::Win32::System::Diagnostics::Debug::{
        IMAGE_RUNTIME_FUNCTION_ENTRY, IMAGE_RUNTIME_FUNCTION_ENTRY_0,
        RemoveVectoredExceptionHandler, RtlAddFunctionTable, RtlDeleteFunctionTable,
        RtlLookupFunctionEntry, RtlVirtualUnwind, UNW_FLAG_NHANDLER,
    };
    use windows_sys::Win32::System::Memory::{PAGE_GUARD, PAGE_READONLY};
    use windows_sys::Win32::System::Threading::GetCurrentThreadId;

    static TEST_THREAD: AtomicU32 = AtomicU32::new(0);
    static ILLEGAL: AtomicUsize = AtomicUsize::new(0);
    static GUARD: AtomicUsize = AtomicUsize::new(0);
    static STORE_RIP: AtomicUsize = AtomicUsize::new(0);
    static RETRY_PC: AtomicUsize = AtomicUsize::new(0);
    static RETRY_CODE: AtomicI32 = AtomicI32::new(0);
    static GUARD_OPERATION: AtomicUsize = AtomicUsize::new(usize::MAX);

    unsafe extern "system" fn retry_title_store(info: *mut EXCEPTION_POINTERS) -> i32 {
        // SAFETY: Windows supplies the exception records and the test
        // supplies the original PC before invoking the synthetic routine.
        unsafe {
            let record = &*(*info).ExceptionRecord;
            let context = &*(*info).ContextRecord;
            let pc = RETRY_PC.load(Relaxed);
            let expected_code = RETRY_CODE.load(Relaxed);
            let expected_operation = if expected_code == STATUS_GUARD_PAGE_VIOLATION {
                GUARD_OPERATION.load(Relaxed)
            } else {
                1
            };
            if record.ExceptionCode != expected_code
                || record.ExceptionAddress as usize != pc
                || context.Rip as usize != pc
                || context.R8 == 0
                || record.NumberParameters < 2
                || record.ExceptionInformation[0] != expected_operation
                || record.ExceptionInformation[1] != context.Rax as usize & !63
                || GetCurrentThreadId() != TEST_THREAD.load(Relaxed)
            {
                return 0;
            }
            // The original code stored its own adjusted stack pointer.
            let expected_rsp = (context.R8 as *const u64).add(2).read();
            if context.Rsp != expected_rsp || context.Rax != context.Rcx || context.EFlags & 1 == 0
            {
                return 0;
            }
            let mut image = 0;
            let entry = RtlLookupFunctionEntry(context.Rip, &mut image, std::ptr::null_mut());
            if entry.is_null() {
                return 0;
            }
            let mut caller = crate::crash::Aligned16(*context);
            let (mut handler_data, mut frame) = (std::ptr::null_mut(), 0);
            RtlVirtualUnwind(
                UNW_FLAG_NHANDLER,
                image,
                context.Rip,
                entry,
                &mut caller.0,
                &mut handler_data,
                &mut frame,
                std::ptr::null_mut(),
            );
            if caller.0.Rsp != expected_rsp + 0x30
                || caller.0.Rip != ((expected_rsp + 0x28) as *const u64).read()
            {
                return 0;
            }
            let page = record.ExceptionInformation[1] & !(PAGE - 1);
            if expected_code == EXCEPTION_ACCESS_VIOLATION
                && VirtualProtect(page as *const _, PAGE, PAGE_READWRITE, &mut 0) == 0
            {
                return 0;
            }
            RETRY_PC.store(0, Relaxed);
            -1
        }
    }

    unsafe extern "system" fn observe(info: *mut EXCEPTION_POINTERS) -> i32 {
        // SAFETY: exception records and the thread ID are supplied by Windows.
        unsafe {
            if GetCurrentThreadId() != TEST_THREAD.load(Relaxed) {
                return 0;
            }
            let record = &*(*info).ExceptionRecord;
            if record.ExceptionCode == EXCEPTION_ILLEGAL_INSTRUCTION {
                ILLEGAL.fetch_add(1, Relaxed);
            }
            if RETRY_PC.load(Relaxed) != 0 && record.ExceptionCode == STATUS_GUARD_PAGE_VIOLATION {
                GUARD_OPERATION.store(record.ExceptionInformation[0], Relaxed);
            }
            let guard = GUARD.load(Relaxed);
            if guard != 0
                && record.ExceptionCode == EXCEPTION_ACCESS_VIOLATION
                && record.ExceptionInformation[0] == 1
                && (guard..guard + PAGE).contains(&record.ExceptionInformation[1])
            {
                let mut old = 0;
                if VirtualProtect(guard as _, PAGE, PAGE_READWRITE, &mut old) != 0 {
                    STORE_RIP.store((*(*info).ContextRecord).Rip as usize, Relaxed);
                    return -1;
                }
            }
            0
        }
    }

    fn allocate() -> Pool {
        // SAFETY: asks Windows for a private writable page.
        let address = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                PAGE,
                MEM_COMMIT | MEM_RESERVE,
                PAGE_READWRITE,
            )
        } as usize;
        assert_ne!(address, 0);
        Pool {
            address,
            size: PAGE,
        }
    }

    #[test]
    fn startup_patches_run_both_loops_and_waits_and_resume_a_guarded_store() {
        crate::report_crashes();
        // SAFETY: callback reads the supplied records and changes only our test guard.
        let observer = unsafe {
            TEST_THREAD.store(GetCurrentThreadId(), Relaxed);
            AddVectoredExceptionHandler(1, Some(observe))
        };
        assert!(!observer.is_null());
        let code = allocate();
        let mut bytes = [0xcc; PAGE];
        let (forward, backward, waits, modified) = (64, 144, 229, 320);
        // The forward loop runs in a synthetic non-leaf stack frame.
        let prologue = [
            0x48, 0x83, 0xec, 0x28, // sub rsp, 0x28
            0x49, 0x89, 0x60, 0x10, // mov [r8+16], rsp
            0x48, 0x89, 0xc8, // mov rax,rcx
            0xf9, // stc
        ];
        let forward_zero = forward + prologue.len();
        bytes[forward..forward_zero].copy_from_slice(&prologue);
        bytes[forward_zero..forward_zero + FORWARD.len()].copy_from_slice(&FORWARD);
        bytes[backward..backward + BACKWARD.len()].copy_from_slice(&BACKWARD);
        for at in [forward_zero + FORWARD.len(), backward + BACKWARD.len()] {
            // Copy RFLAGS and the fourth argument to the output record (r8).
            bytes[at..at + 10]
                .copy_from_slice(&[0x9c, 0x41, 0x5a, 0x4d, 0x89, 0x10, 0x4d, 0x89, 0x48, 0x08]);
        }
        let epilogue = forward_zero + FORWARD.len() + 10;
        bytes[epilogue..epilogue + 5].copy_from_slice(&[0x48, 0x83, 0xc4, 0x28, 0xc3]);
        bytes[backward + BACKWARD.len() + 10] = 0xc3;
        // Version 1, a 4-byte prologue and one UWOP_ALLOC_SMALL (40 bytes).
        const UNWIND: u32 = 512;
        bytes[UNWIND as usize..UNWIND as usize + 8].copy_from_slice(&[1, 4, 1, 0, 4, 0x42, 0, 0]);
        bytes[waits..waits + 19].copy_from_slice(&[
            0x48, 0x89, 0xc8, // mov rax,rcx
            0x31, 0xc9, 0x31, 0xd2, 0x39, 0xd1, 0xf9, // ecx=edx=0; cmp ecx,edx; stc
            0x0f, 0x01, 0xfa, 0x0f, 0x01, 0xfb, // monitorx; mwaitx
            0x9c, 0x58, 0xc3, // pushfq; pop rax; ret
        ]);
        bytes[modified..modified + 3].copy_from_slice(&[0x0f, 0x01, 0xfa]);

        let make_site = |rva, zero| {
            let (change, span) = Change::verify(&bytes, rva, zero).unwrap();
            let guard_rva = span.start - 14;
            Site {
                rva,
                expected: bytes[guard_rva..span.end + 14].to_vec(),
                span,
                guard_rva,
                change,
            }
        };
        let selected = [
            (forward_zero, true),
            (backward + 7, true),
            (waits + 10, false),
            (waits + 13, false),
            (modified, false),
        ]
        .map(|(rva, zero)| make_site(rva, zero));
        bytes[modified - 1] = 0x90; // Models an earlier neighboring code rewrite.
        // SAFETY: the synthesized code is copied to a private, idle page.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), code.address as *mut u8, PAGE) };
        protect(code.address, PAGE, PAGE_EXECUTE_READ).unwrap();
        flush(code.address, PAGE).unwrap();

        type Zero = unsafe extern "system" fn(usize, usize, *mut u64, u64) -> usize;
        type Wait = unsafe extern "system" fn(usize) -> u64;
        // SAFETY: the byte sequences above follow these ABI signatures.
        let (a, b, wait): (Zero, Zero, Wait) = unsafe {
            (
                std::mem::transmute::<usize, Zero>(code.address + forward),
                std::mem::transmute::<usize, Zero>(code.address + backward),
                std::mem::transmute::<usize, Wait>(code.address + waits),
            )
        };
        let data = allocate();
        let forward_pc = code.address + forward_zero;
        let run = |zero: Zero,
                   misalignment: usize,
                   result: usize,
                   guard: bool,
                   title_fault: Option<u32>| {
            // Leave a sentinel line below both zeroed lines and one above.
            let start = data.address + PAGE - 256;
            let at = start + 64 + misalignment;
            let mut out = [0u64; 3];
            // SAFETY: page is writable before the call, and the observer
            // opens the optional read-only page on the first scalar store.
            unsafe { (start as *mut u8).write_bytes(0x7f, 256) };
            if guard {
                protect(data.address, PAGE, PAGE_READONLY).unwrap();
                GUARD.store(data.address, Relaxed);
            }
            if let Some(protection) = title_fault {
                protect(data.address, PAGE, protection).unwrap();
                GUARD_OPERATION.store(usize::MAX, Relaxed);
                let code = if protection & PAGE_GUARD != 0 {
                    STATUS_GUARD_PAGE_VIOLATION
                } else {
                    EXCEPTION_ACCESS_VIOLATION
                };
                RETRY_CODE.store(code, Relaxed);
                RETRY_PC.store(forward_pc, Relaxed);
            }
            // SAFETY: the code and its two output ranges are live.
            let returned = unsafe { zero(at, at + 128, out.as_mut_ptr(), 0x1234) };
            assert_eq!(returned, at + result);
            GUARD.store(0, Relaxed);
            assert_eq!(RETRY_PC.load(Relaxed), 0);
            assert_eq!(out[0] & 0x8d5, 0x44);
            assert_eq!(out[1], 0x1234);
            // SAFETY: the synthetic call is done and the test page is readable.
            let bytes = unsafe { std::slice::from_raw_parts(start as *const u8, 256) };
            assert_eq!(bytes[..64], [0x7f; 64]);
            assert_eq!(bytes[64..192], [0; 128]);
            assert_eq!(bytes[192..], [0x7f; 64]);
        };

        let clzero_native = std::arch::x86_64::__cpuid(0x8000_0000).eax >= 0x8000_0008
            && std::arch::x86_64::__cpuid(0x8000_0008).ebx & 1 != 0;
        let waits_native = std::arch::x86_64::__cpuid(0x8000_0000).eax >= 0x8000_0001
            && std::arch::x86_64::__cpuid(0x8000_0001).ecx & (1 << 29) != 0;
        // SAFETY: this test owns the idle synthetic mapping.
        assert_eq!(unsafe { install(code.address, &[]) }.unwrap(), (0, 0));
        ILLEGAL.store(0, Relaxed);
        run(a, 0, 128, false, None);
        run(b, 7, 64, false, None);
        if !clzero_native {
            assert!(ILLEGAL.load(Relaxed) >= 2);
        }
        if !waits_native {
            ILLEGAL.store(0, Relaxed);
            // SAFETY: the monitored address is mapped and the waits fault on this CPU.
            assert_eq!(unsafe { wait(data.address) } & 0x8d5, 0x45);
            assert_eq!(ILLEGAL.load(Relaxed), 2);
        }

        // SAFETY: this test still owns the idle mapping.
        assert_eq!(unsafe { install(code.address, &selected) }.unwrap(), (2, 2));

        // SAFETY: checking the still-readable code, after its protections were restored.
        let patched = unsafe { std::slice::from_raw_parts(code.address as *const u8, PAGE) };
        assert_eq!(
            &patched[modified..modified + 3],
            &bytes[modified..modified + 3]
        );
        assert_eq!(
            query(code.address + forward).unwrap().Protect,
            PAGE_EXECUTE_READ
        );
        assert_eq!(&patched[forward_zero..forward_zero + 2], &[0x90, 0xe9]);
        assert_eq!(
            &patched[waits + 10..waits + 16],
            &[0x0f, 0x1f, 0x00, 0xf3, 0x90, 0x90]
        );

        ILLEGAL.store(0, Relaxed);
        run(a, 0, 128, true, None);
        run(b, 0, 64, false, None);
        // SAFETY: the patched wait function takes a mapped address.
        assert_eq!(unsafe { wait(data.address) } & 0x8d5, 0x45);
        assert_eq!(ILLEGAL.load(Relaxed), 0);
        let rip = STORE_RIP.load(Relaxed);
        assert_ne!(rip, 0);
        assert_eq!(query(rip).unwrap().Protect, PAGE_EXECUTE_READ);
        // SAFETY: RIP is the resumed store in the pool, which remains mapped.
        let store = unsafe { (rip as *const [u8; 4]).read() };
        assert_eq!(store, [0x48, 0xc7, 0x40, 0]);
        run(a, 5, 128, false, None);
        run(b, 7, 64, false, None);
        assert_eq!(ILLEGAL.load(Relaxed), 0);
        let function = IMAGE_RUNTIME_FUNCTION_ENTRY {
            BeginAddress: forward as u32,
            EndAddress: (epilogue + 5) as u32,
            Anonymous: IMAGE_RUNTIME_FUNCTION_ENTRY_0 { UnwindData: UNWIND },
        };
        // SAFETY: the table and the mapped unwind info both outlive the call.
        assert!(unsafe { RtlAddFunctionTable(&function, 1, code.address as u64) });
        // SAFETY: registered after the mapper, so this receives the title context.
        let retry = unsafe { AddVectoredExceptionHandler(0, Some(retry_title_store)) };
        assert!(!retry.is_null());
        run(a, 5, 128, false, Some(PAGE_READONLY));
        run(a, 5, 128, false, Some(PAGE_READWRITE | PAGE_GUARD));

        // SAFETY: unregister before the test releases its data pages. The
        // registered stub pool itself stays reserved for its fault map.
        unsafe {
            RemoveVectoredExceptionHandler(retry);
            assert!(RtlDeleteFunctionTable(&function));
            RemoveVectoredExceptionHandler(observer);
        }
    }
}
