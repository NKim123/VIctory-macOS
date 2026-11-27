//! Opt-in look-first rewriting of byte flag updates observed on GPU guards.
//! The profile identifies an exact executable build and instructions by digest.
//! All executable bytes are derived from a fixed template and the mapped image.

use std::fs::{File, OpenOptions};
use std::os::windows::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::sync::atomic::{
    AtomicU64, AtomicUsize,
    Ordering::{Acquire, Relaxed, Release},
};
use std::sync::mpsc::{self, RecvTimeoutError, Sender};
use std::time::Duration;

use windows_sys::Win32::Foundation::{
    ERROR_CALL_NOT_IMPLEMENTED, ERROR_INVALID_PARAMETER, ERROR_NOT_SUPPORTED, GetLastError,
};
use windows_sys::Win32::System::Diagnostics::Debug::{
    AddVectoredExceptionHandler, CONTEXT, EXCEPTION_POINTERS, FlushInstructionCache,
    ReadProcessMemory, RemoveVectoredExceptionHandler,
};
use windows_sys::Win32::System::Memory::{
    MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_EXECUTE_READ, PAGE_READWRITE, VirtualAlloc,
    VirtualFree, VirtualProtect,
};
use windows_sys::Win32::System::Threading::{
    GetCurrentProcess, GetProcessMitigationPolicy, ProcessUserShadowStackPolicy,
};

use crate::startup_patch::{self, FaultSite, Saved};
use crate::{LoadedImage, PreparedImage, masked};

const MAX_SITES: usize = 512;
const MIN_SAME: u64 = 16;
const MAX_PROFILE: usize = 64 * 1024;

struct FlagOr<'a> {
    code: &'a [u8],
    rex: u8,
    modrm: u8,
    displacement: &'a [u8],
    bit: u8,
}

impl<'a> FlagOr<'a> {
    /// Accept only a single-bit OR on a byte at [reg+disp], with one
    /// instruction large enough to replace with a rel32 jump.
    fn decode(code: &'a [u8]) -> Option<Self> {
        let (rex, prefix) = match *code.first()? {
            0x40 | 0x41 => (code[0], 1),
            _ => (0, 0),
        };
        let modrm = *code.get(prefix + 1)?;
        if code[prefix] != 0x80 || modrm & 0x38 != 0x08 || modrm & 7 == 4 {
            return None;
        }
        let size = match modrm >> 6 {
            1 => 1,
            2 => 4,
            _ => return None,
        };
        let end = prefix + 2 + size;
        let bit = *code.get(end)?;
        (end >= 4 && bit.is_power_of_two()).then(|| Self {
            code: &code[..=end],
            rex,
            modrm,
            displacement: &code[prefix + 2..end],
            bit,
        })
    }

    fn address(&self, c: &CONTEXT) -> usize {
        let reg = (self.modrm & 7) | ((self.rex & 1) << 3);
        let base = [
            c.Rax, c.Rcx, c.Rdx, c.Rbx, c.Rsp, c.Rbp, c.Rsi, c.Rdi, c.R8, c.R9, c.R10, c.R11,
            c.R12, c.R13, c.R14, c.R15,
        ][reg as usize];
        let displacement = match self.displacement {
            [byte] => i64::from(*byte as i8),
            bytes => i64::from(i32::from_le_bytes(bytes.try_into().expect("disp32"))),
        };
        base.wrapping_add_signed(displacement) as usize
    }

    fn trampoline(
        &self,
        site: u64,
        stub: u64,
        guards: &[AtomicUsize],
        shift: u8,
    ) -> (Vec<u8>, [FaultSite; 2]) {
        let mut out = vec![0x9c, 0x50, 0x51, 0x52]; // pushfq, push rax/rcx/rdx
        out.extend([0x48 | (self.rex & 1), 0x8d, self.modrm & 0xc7]); // lea rax, [m]
        out.extend(self.displacement);
        out.extend([0x48, 0x89, 0xc1, 0x48, 0xc1, 0xe9, shift]); // rcx = rax >> page shift
        out.extend([0x81, 0xe1]); // and ecx, table mask
        out.extend(((guards.len() - 1) as u32).to_le_bytes());
        out.extend([0x48, 0xba]); // mov rdx, guard table
        out.extend((guards.as_ptr() as u64).to_le_bytes());
        out.extend([0x48, 0x8b, 0x14, 0xca]); // mov rdx, [rdx + rcx*8]
        out.extend([0x48, 0x85, 0xd2]); // empty table slot
        let empty = out.len();
        out.extend([0x74, 0]); // jz original
        out.extend([0x48, 0x31, 0xc2, 0x48, 0xc1, 0xea, shift]); // compare page addresses
        let unguarded = out.len();
        out.extend([0x75, 0]); // jne original
        let read_pc = stub + out.len() as u64;
        out.extend([0x0f, 0xb6, 0x08]); // movzx ecx, byte [rax]
        out.extend([0xf6, 0xc1, self.bit]); // test cl, bit
        let different = out.len();
        out.extend([0x74, 0]); // jz original
        out.extend([0x80, 0xc9, self.bit]); // or cl, bit: defined flags of the original OR
        out.extend([0x5a, 0x59, 0x58, 0x48, 0x8d, 0x64, 0x24, 8]); // pop regs, discard saved flags
        out.push(0xc3); // ret to the title continuation
        for branch in [empty, unguarded, different] {
            out[branch + 1] = (out.len() - branch - 2) as u8;
        }
        out.extend([0x5a, 0x59, 0x58, 0x9d]); // pop regs and flags before the original store
        let write_pc = stub + out.len() as u64;
        out.extend(self.code);
        out.push(0xc3); // ret to the title continuation
        (
            out,
            [
                FaultSite {
                    stub: read_pc..read_pc + 3,
                    title: site,
                    saved: &[
                        Saved::Rdx,
                        Saved::Rcx,
                        Saved::Rax,
                        Saved::Flags,
                        Saved::ReturnAddress,
                    ],
                },
                FaultSite {
                    stub: write_pc..write_pc + self.code.len() as u64,
                    title: site,
                    saved: &[Saved::ReturnAddress],
                },
            ],
        )
    }
}

type Sites = std::collections::BTreeMap<u32, blake3::Hash>;

fn read_profile(path: &Path, image: blake3::Hash, allow_missing: bool) -> anyhow::Result<Sites> {
    use std::io::{ErrorKind, Read};
    let file = match std::fs::File::open(path) {
        Ok(file) => file,
        Err(error) if allow_missing && error.kind() == ErrorKind::NotFound => {
            return Ok(Sites::new());
        }
        Err(error) => return Err(error.into()),
    };
    let mut text = String::new();
    file.take(MAX_PROFILE as u64 + 1)
        .read_to_string(&mut text)?;
    anyhow::ensure!(text.len() <= MAX_PROFILE, "write-site profile is too large");
    let mut lines = text.lines();
    let header = format!("victory-write-sites-v1 {image}");
    anyhow::ensure!(
        lines.next() == Some(header.as_str()),
        "write-site profile is for another image or format"
    );
    let mut sites = Sites::new();
    for line in lines {
        let (rva, hash) = line
            .split_once(' ')
            .ok_or_else(|| anyhow::anyhow!("invalid write-site record"))?;
        let rva = u32::from_str_radix(rva, 16)?;
        let hash = blake3::Hash::from_hex(hash)?;
        anyhow::ensure!(
            sites.insert(rva, hash).is_none(),
            "duplicate write-site record"
        );
        anyhow::ensure!(sites.len() <= MAX_SITES, "too many write-site records");
    }
    Ok(sites)
}

fn write_profile(path: &Path, image: blake3::Hash, sites: &Sites) -> std::io::Result<()> {
    use std::fmt::Write;
    let mut text = format!("victory-write-sites-v1 {image}\n");
    for (rva, hash) in sites {
        writeln!(text, "{rva:08x} {hash}").expect("writing a string");
    }
    crate::publish::write_if_changed(path, text.as_bytes())
}

/// Reads only mapped memory, with an OS-checked copy so a disappearing
/// mapping cannot cause a nested read fault in the write handler.
fn read(at: usize, out: &mut [u8]) -> bool {
    let mut done = 0;
    // SAFETY: the destination is ours and the kernel validates the source.
    unsafe {
        ReadProcessMemory(
            GetCurrentProcess(),
            at as *const _,
            out.as_mut_ptr().cast(),
            out.len(),
            &mut done,
        ) != 0
            && done == out.len()
    }
}

struct Recorder {
    base: usize,
    sections: Vec<(std::ops::Range<u32>, u32)>,
    samples: [Sample; MAX_SITES],
}

#[derive(Default)]
struct Sample {
    counts: AtomicU64,
    instruction: AtomicU64,
}

static RECORDER: OnceLock<Recorder> = OnceLock::new();

impl Recorder {
    fn site_bytes(&self, rva: u32) -> Option<(usize, usize)> {
        let (range, file) = self
            .sections
            .iter()
            .find(|(range, _)| range.contains(&rva))?;
        let count = (range.end - rva).min(8) as usize;
        Some(((*file + (rva - range.start)) as usize, count))
    }

    fn instruction<'a>(&self, rva: u32, bytes: &'a mut [u8; 8]) -> Option<FlagOr<'a>> {
        let (_, len) = self.site_bytes(rva)?;
        let bytes = &mut bytes[..len];
        read(self.base + rva as usize, bytes).then_some(())?;
        FlagOr::decode(bytes)
    }

    fn unchanged(&self, rva: u32, len: usize, prepared: &[u8]) -> bool {
        let Some((range, file)) = self.sections.iter().find(|(range, _)| range.contains(&rva))
        else {
            return false;
        };
        // An earlier startup patch can start up to 14 bytes before the
        // candidate and still overlap it. Check the surrounding bytes too.
        let first = range.start.max(rva.saturating_sub(14));
        let last = range.end.min(rva.saturating_add(len as u32 + 14));
        let count = (last - first) as usize;
        let start = (*file + (first - range.start)) as usize;
        let mut mapped = [0; 36];
        let mapped = &mut mapped[..count];
        read(self.base + first as usize, mapped)
            && prepared.get(start..start + count) == Some(mapped)
    }

    fn observe(&self, context: &CONTEXT, target: usize) {
        let Some(rva) = (context.Rip as usize)
            .checked_sub(self.base)
            .and_then(|rva| u32::try_from(rva).ok())
        else {
            return;
        };
        let slot = &self.samples[(rva.wrapping_mul(0x9e37_79b9) >> 23) as usize];
        let previous = slot.counts.load(Relaxed);
        let same = (previous >> 32) & 0xffff;
        let different = previous >> 48;
        if previous != 0
            && (previous as u32 != rva || same >= MIN_SAME || different >= MIN_SAME / 4)
        {
            return;
        }
        let mut bytes = [0; 8];
        let Some(op) = self.instruction(rva, &mut bytes) else {
            return;
        };
        let mut byte = [0];
        if op.address(context) != target || !read(target, &mut byte) {
            return;
        }
        let mut exact = [0; 8];
        exact[..op.code.len()].copy_from_slice(op.code);
        let exact = u64::from_le_bytes(exact);
        if previous == 0 {
            slot.instruction.store(exact, Relaxed);
        } else if slot.instruction.load(Relaxed) != exact {
            return;
        }
        let increment = if byte[0] & op.bit != 0 {
            1 << 32
        } else {
            1 << 48
        };
        // Guarded faults are serialized by the GPU guard lock.
        slot.counts
            .store((previous | u64::from(rva)) + increment, Release);
    }

    fn collect(&self, sites: &mut Sites) -> bool {
        let old = sites.len();
        for slot in &self.samples {
            let sample = slot.counts.load(Acquire);
            if (sample >> 32) & 0xffff == MIN_SAME && sites.len() < MAX_SITES {
                let rva = sample as u32;
                let bytes = slot.instruction.load(Relaxed).to_le_bytes();
                if let Some(op) = FlagOr::decode(&bytes) {
                    sites.entry(rva).or_insert_with(|| blake3::hash(op.code));
                }
            }
        }
        sites.len() != old
    }
}

/// Called by the GPU guard handler before it opens a guarded, write-watched page.
pub fn observe(context: &CONTEXT, target: usize) {
    if let Some(recorder) = RECORDER.get() {
        recorder.observe(context, target);
    }
}

/// An active recording or application run. Dropping a recording flushes
/// remaining candidates.
#[derive(Default)]
pub struct Session(Option<(Sender<()>, std::thread::JoinHandle<()>)>);

impl Drop for Session {
    fn drop(&mut self) {
        if let Some((stop, worker)) = self.0.take() {
            let _ = stop.send(());
            let _ = worker.join();
        }
    }
}

fn lock_recording(profile: &Path) -> anyhow::Result<File> {
    use anyhow::Context;

    let mut name = profile
        .file_name()
        .context("write-site profile must have a file name")?
        .to_os_string();
    name.push(".lock");
    let sidecar = profile.with_file_name(name);
    victory_core::ensure_outside_repo(&sidecar)?;
    // Keep the sidecar in place when the handle closes. The OS releases
    // the sharing restriction even if a recording process is killed.
    OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(false)
        .share_mode(0)
        .open(&sidecar)
        .with_context(|| {
            format!(
                "acquiring the exclusive write-site recording lock {}",
                sidecar.display()
            )
        })
}

fn record(
    path: PathBuf,
    image: blake3::Hash,
    mut sites: Sites,
    writer: File,
) -> std::io::Result<Session> {
    let (stop, wait) = mpsc::channel();
    let worker = std::thread::Builder::new().name("write-sites".into()).spawn(move || {
        let _writer = writer;
        let recorder = RECORDER.get().expect("write-site recorder installed");
        let mut pending = false;
        loop {
            let stopping = wait.recv_timeout(Duration::from_secs(2)) != Err(RecvTimeoutError::Timeout);
            pending |= recorder.collect(&mut sites);
            if pending {
                match write_profile(&path, image, &sites) {
                    Ok(()) => pending = false,
                    Err(error) => tracing::warn!(%error, path = %path.display(), "saving write-site profile"),
                }
            }
            if stopping {
                break;
            }
        }
    })?;
    Ok(Session(Some((stop, worker))))
}

fn code_section(
    s: &victory_pe::SectionInfo,
    image_size: u32,
) -> Option<(std::ops::Range<u32>, u32)> {
    const SECTION_CODE: u32 = 0x6000_0000; // readable and executable
    const SECTION_WRITE: u32 = 0x8000_0000;
    if s.characteristics & (SECTION_CODE | SECTION_WRITE) != SECTION_CODE {
        return None;
    }
    let size = match s.virtual_size {
        0 => s.file_size,
        size => size.min(s.file_size),
    };
    let end = s.rva.checked_add(size)?;
    (size != 0 && end <= image_size && s.file_offset.checked_add(size).is_some())
        .then_some((s.rva..end, s.file_offset))
}

/// Record candidates, or apply a runtime profile explicitly selected for use.
/// The caller registers [`observe`] when this returns a session.
///
/// # Safety
/// The main image stays mapped while the returned session runs, and title
/// threads have not started when this function patches it.
pub unsafe fn start(
    image: &LoadedImage,
    prepared: &PreparedImage,
    guards: &'static [AtomicUsize],
    page_shift: u8,
) -> anyhow::Result<Option<Session>> {
    let apply_to_title = std::env::var_os("VICTORY_WRITE_SITES");
    let record_to = std::env::var_os("VICTORY_WRITE_SITES_RECORD");
    if apply_to_title.is_some() && record_to.is_some() {
        tracing::warn!("VICTORY_WRITE_SITES takes precedence over VICTORY_WRITE_SITES_RECORD");
    }
    let recording = apply_to_title.is_none();
    let Some(path) = apply_to_title.or(record_to) else {
        return Ok(None);
    };
    let path = PathBuf::from(path);
    let image_hash = prepared.original_hash;
    let loaded = (|| -> anyhow::Result<_> {
        victory_core::ensure_outside_repo(&path)?;
        // Read the starting snapshot only after declaring this writer.
        let writer = recording.then(|| lock_recording(&path)).transpose()?;
        Ok((read_profile(&path, image_hash, recording)?, writer))
    })();
    let (sites, writer) = match loaded {
        Ok(loaded) => loaded,
        Err(error) => {
            tracing::warn!(%error, path = %path.display(), "disabling guarded write sites");
            return Ok(None);
        }
    };
    let sections = image
        .info()
        .sections
        .iter()
        .filter_map(|s| code_section(s, image.info().size_of_image))
        .collect();
    let recorder = Recorder {
        base: image.base(),
        sections,
        samples: std::array::from_fn(|_| Sample::default()),
    };
    if let Some(writer) = writer {
        let result = RECORDER
            .set(recorder)
            .map_err(|_| anyhow::anyhow!("write-site recorder is already installed"))
            .and_then(|_| record(path.clone(), image_hash, sites, writer).map_err(Into::into));
        return Ok(match result {
            Ok(session) => Some(session),
            Err(error) => {
                tracing::warn!(%error, path = %path.display(), "disabling guarded write sites");
                None
            }
        });
    }
    // SAFETY: mapped image at startup; no title threads are running.
    let count = unsafe { apply(&recorder, &prepared.bytes, &sites, guards, page_shift) }?;
    tracing::info!(count, path = %path.display(), "applied guarded write sites");
    Ok((count != 0).then(Session::default))
}

static FAULT_SITES: OnceLock<Vec<FaultSite>> = OnceLock::new();

unsafe extern "system" fn remap_write_fault(info: *mut EXCEPTION_POINTERS) -> i32 {
    if let Some(sites) = FAULT_SITES.get() {
        // SAFETY: Windows is dispatching the exception. GPU guards get first chance.
        unsafe { startup_patch::remap(info, sites) };
    }
    0 // Continue to the title's usual exception dispatch.
}

fn shadow_stack_allows_fault_retry() -> bool {
    let mut flags = 0u32;
    // SAFETY: this process handle and a four-byte policy output are valid.
    unsafe {
        if GetProcessMitigationPolicy(
            GetCurrentProcess(),
            ProcessUserShadowStackPolicy,
            (&mut flags as *mut u32).cast(),
            size_of_val(&flags),
        ) != 0
        {
            return flags & 1 == 0; // EnableUserShadowStack
        }
        matches!(
            GetLastError(),
            ERROR_INVALID_PARAMETER | ERROR_NOT_SUPPORTED | ERROR_CALL_NOT_IMPLEMENTED
        )
    }
}

unsafe fn apply(
    recorder: &Recorder,
    prepared: &[u8],
    sites: &Sites,
    guards: &[AtomicUsize],
    page_shift: u8,
) -> anyhow::Result<usize> {
    if !guards.len().is_power_of_two() || sites.is_empty() {
        return Ok(0);
    }
    if !shadow_stack_allows_fault_retry() {
        tracing::warn!("write-site patches require a process without a hardware shadow stack");
        return Ok(0);
    }
    let mut end = 0;
    let candidates: Vec<_> = sites
        .iter()
        .filter_map(|(&rva, &hash)| {
            let mut bytes = [0; 8];
            let op = recorder.instruction(rva, &mut bytes)?;
            let len = op.code.len();
            let site = recorder.base + rva as usize;
            if site < end
                || site / 4096 != (site + len - 1) / 4096
                || !recorder.unchanged(rva, len, prepared)
                || blake3::hash(op.code) != hash
            {
                return None;
            }
            end = site + len;
            Some((site as u64, op.code.to_vec()))
        })
        .collect();
    if candidates.is_empty() {
        return Ok(0);
    }

    // Enough for MAX_SITES template bodies. The reservation can be
    // discarded until every body is executable and flushed.
    const POOL_SIZE: usize = 64 << 10;
    let base = recorder.base & !(POOL_SIZE - 1);
    let pool = (1..2048)
        .flat_map(|step| [base.checked_sub(step << 20), base.checked_add(step << 20)])
        .flatten()
        .filter(|&hint| hint != 0)
        .find_map(|hint| {
            // SAFETY: reserves a fresh range, never modifies a mapped view.
            let pool = unsafe {
                VirtualAlloc(
                    hint as *const _,
                    POOL_SIZE,
                    MEM_RESERVE | MEM_COMMIT,
                    PAGE_READWRITE,
                )
            };
            (!pool.is_null()).then_some(pool)
        });
    let Some(pool) = pool else { return Ok(0) };
    let mut used = 0;
    let mut sites = Vec::new();
    let mut faults = Vec::new();
    for (site, code) in &candidates {
        let op = FlagOr::decode(code).expect("validated instruction");
        let stub = pool as u64 + used as u64;
        let (body, fault_sites) = op.trampoline(*site, stub, guards, page_shift);
        let Some(mut branch) = masked::jump(*site, code.len(), stub) else {
            continue;
        };
        // CALL shares the rel32 displacement with JMP. A JMP out of the
        // title function at the original PC can be mistaken for an epilogue
        // when Windows dispatches an exception there.
        branch[0] = 0xe8;
        if used + body.len() > POOL_SIZE {
            break;
        }
        // SAFETY: fresh writable allocation with room for this stub.
        unsafe {
            std::ptr::copy_nonoverlapping(body.as_ptr(), pool.cast::<u8>().add(used), body.len())
        };
        used += body.len().next_multiple_of(16);
        sites.push((*site, branch));
        faults.extend(fault_sites);
    }
    if sites.is_empty() {
        // SAFETY: no branch was installed.
        unsafe { VirtualFree(pool, 0, MEM_RELEASE) };
        return Ok(0);
    }
    let mut old = 0;
    // SAFETY: pool is ours and no live code points to it yet.
    if unsafe {
        VirtualProtect(pool, POOL_SIZE, PAGE_EXECUTE_READ, &mut old) == 0
            || FlushInstructionCache(GetCurrentProcess(), pool, used) == 0
    } {
        // SAFETY: no branch was installed.
        unsafe { VirtualFree(pool, 0, MEM_RELEASE) };
        return Ok(0);
    }
    // SAFETY: the handler consults only immutable metadata after startup.
    let handler = unsafe { AddVectoredExceptionHandler(0, Some(remap_write_fault)) };
    if handler.is_null() || FAULT_SITES.set(faults).is_err() {
        // SAFETY: the handler, if registered, belongs to this attempt.
        // No branch points to the private pool.
        unsafe {
            if !handler.is_null() {
                RemoveVectoredExceptionHandler(handler);
            }
            VirtualFree(pool, 0, MEM_RELEASE);
        }
        tracing::warn!("write-site stub fault translation unavailable");
        return Ok(0);
    }
    // Once registered, the bounded pool and fault map live for the process.
    let mut count = 0;
    for (site, branch) in sites {
        // SAFETY: verified instruction on one page in the idle title image.
        if unsafe { startup_patch::publish(site as usize, &branch) }? {
            count += 1;
        }
    }
    Ok(count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicI32, AtomicU32};
    use windows_sys::Win32::Foundation::{
        ERROR_SHARING_VIOLATION, EXCEPTION_ACCESS_VIOLATION, STATUS_GUARD_PAGE_VIOLATION,
    };
    use windows_sys::Win32::System::Diagnostics::Debug::{
        EXCEPTION_CONTINUE_EXECUTION, EXCEPTION_CONTINUE_SEARCH, IMAGE_RUNTIME_FUNCTION_ENTRY,
        IMAGE_RUNTIME_FUNCTION_ENTRY_0, RtlAddFunctionTable, RtlDeleteFunctionTable,
        RtlLookupFunctionEntry, RtlVirtualUnwind, UNW_FLAG_NHANDLER,
    };
    use windows_sys::Win32::System::Memory::{PAGE_GUARD, PAGE_NOACCESS, PAGE_READONLY};
    use windows_sys::Win32::System::Threading::GetCurrentThreadId;

    // or byte [r9-1], 80h; used only as part of our synthetic function.
    const OR: [u8; 5] = [0x41, 0x80, 0x49, 0xff, 0x80];

    #[test]
    fn only_record_mode_accepts_a_missing_profile() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("missing.txt");
        let hash = blake3::hash(b"synthetic profile image");
        assert!(read_profile(&path, hash, true).unwrap().is_empty());
        let error = read_profile(&path, hash, false).unwrap_err();
        assert_eq!(
            error.downcast_ref::<std::io::Error>().unwrap().kind(),
            std::io::ErrorKind::NotFound
        );
    }

    #[test]
    fn executable_sections_with_no_virtual_size_use_file_size() {
        let mut section = victory_pe::SectionInfo {
            name: ".text".into(),
            rva: 0x1000,
            virtual_size: 0,
            file_offset: 0x200,
            file_size: 0x400,
            characteristics: 0x6000_0000,
        };
        assert_eq!(
            code_section(&section, 0x2000),
            Some((0x1000..0x1400, 0x200))
        );
        assert_eq!(code_section(&section, 0x1200), None);
        section.virtual_size = 0x100;
        assert_eq!(
            code_section(&section, 0x2000),
            Some((0x1000..0x1100, 0x200))
        );
        section.characteristics |= 0x8000_0000;
        assert_eq!(code_section(&section, 0x2000), None);
    }

    #[test]
    fn only_single_bit_flags_with_simple_addresses_are_eligible() {
        assert!(FlagOr::decode(&OR).is_some());
        // no REX, displacement of 32 bits
        assert!(FlagOr::decode(&[0x80, 0x8a, 1, 0, 0, 0, 2]).is_some());
        for bytes in [
            &[0xf0, 0x41, 0x80, 0x49, 0xff, 0x80][..], // lock
            &[0x67, 0x41, 0x80, 0x49, 0xff, 0x80],     // address-size prefix
            &[0x41, 0x80, 0x49, 0xff, 0x81],           // multiple bits
            &[0x41, 0x80, 0xc9, 0x80, 0x90],           // register
            &[0x80, 0x0d, 1, 0, 0, 0, 2],              // RIP-relative
            &[0x41, 0x80, 0x4c, 0x24, 0x05, 1],        // SIB
            &[0x80, 0x49, 0x01, 1],                    // too short
            &[0xc4, 0xc1, 0x7c, 0x11, 0x40, 0x34],     // vector store
            &OR[..4],
        ] {
            assert!(FlagOr::decode(bytes).is_none(), "{bytes:x?}");
        }
    }

    fn recorder(base: usize, size: u32) -> Recorder {
        Recorder {
            base,
            sections: vec![(0..size, 0)],
            samples: std::array::from_fn(|_| Sample::default()),
        }
    }

    #[test]
    fn learns_only_matching_faults_and_binds_them_to_an_image() {
        let byte = 0x80u8;
        let recorder = recorder(OR.as_ptr() as usize, OR.len() as u32);
        let address = &byte as *const u8 as usize;
        // SAFETY: plain data. Set the only fields the observer needs.
        let mut context: CONTEXT = unsafe { std::mem::zeroed() };
        context.Rip = recorder.base as u64;
        context.R9 = (address + 1) as u64;
        let mut sites = Sites::new();
        for _ in 0..MIN_SAME {
            recorder.observe(&context, address + 1);
        }
        assert!(!recorder.collect(&mut sites));
        for _ in 0..MIN_SAME {
            recorder.observe(&context, address);
        }
        assert!(recorder.collect(&mut sites));
        assert_eq!(sites.get(&0), Some(&blake3::hash(&OR)));
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("sites.txt");
        let image = blake3::hash(b"synthetic image");
        write_profile(&file, image, &sites).unwrap();
        assert_eq!(read_profile(&file, image, false).unwrap(), sites);
        assert!(read_profile(&file, blake3::hash(b"other image"), false).is_err());
        let text = std::fs::read_to_string(file).unwrap();
        assert_eq!(text.lines().count(), 2);
    }

    #[test]
    fn profile_writer_child() {
        let Some(path) = std::env::var_os("VICTORY_TEST_WRITE_SITE_LOCK_PROFILE") else {
            return;
        };
        let result = lock_recording(Path::new(&path));
        if std::env::var_os("VICTORY_TEST_WRITE_SITE_LOCK_BUSY").is_some() {
            let error = result.unwrap_err();
            let cause = error.downcast_ref::<std::io::Error>().unwrap();
            assert_eq!(cause.raw_os_error(), Some(ERROR_SHARING_VIOLATION as i32));
        } else {
            result.unwrap();
        }
    }

    #[test]
    fn only_one_process_can_record_a_profile() {
        let dir = tempfile::tempdir().unwrap();
        let profile = dir.path().join("sites.txt");
        let run_child = |busy: bool| {
            let mut command = std::process::Command::new(std::env::current_exe().unwrap());
            command
                .args(["--exact", "write_sites::tests::profile_writer_child"])
                .env("VICTORY_TEST_WRITE_SITE_LOCK_PROFILE", &profile)
                .env_remove("VICTORY_TEST_WRITE_SITE_LOCK_BUSY");
            if busy {
                command.env("VICTORY_TEST_WRITE_SITE_LOCK_BUSY", "1");
            }
            let output = command.output().unwrap();
            assert!(output.status.success(), "{output:?}");
            assert!(String::from_utf8_lossy(&output.stdout).contains("1 passed"));
        };
        let writer = lock_recording(&profile).unwrap();
        run_child(true);
        // Publication still works while the stable sidecar is held.
        let hash = blake3::hash(b"synthetic profile image");
        write_profile(&profile, hash, &Sites::new()).unwrap();
        assert!(read_profile(&profile, hash, false).unwrap().is_empty());
        drop(writer);
        run_child(false);
    }

    const PAGE: usize = 1 << 16;
    static TEST_GUARDS: [AtomicUsize; 8] = [const { AtomicUsize::new(0) }; 8];
    static TEST_THREAD: AtomicU32 = AtomicU32::new(0);
    static TEST_PAGE: AtomicUsize = AtomicUsize::new(0);
    static FAULTS: AtomicUsize = AtomicUsize::new(0);
    static TITLE_INSTRUCTION: AtomicUsize = AtomicUsize::new(0);
    static TITLE_FAULT_CODE: AtomicI32 = AtomicI32::new(0);
    static GUARD_OPERATION: AtomicUsize = AtomicUsize::new(usize::MAX);

    unsafe extern "system" fn retry_title_write(info: *mut EXCEPTION_POINTERS) -> i32 {
        // SAFETY: the system returns the current test thread's numeric identifier.
        if unsafe { GetCurrentThreadId() } != TEST_THREAD.load(Relaxed) {
            return EXCEPTION_CONTINUE_SEARCH;
        }
        // SAFETY: Windows supplies both exception structures.
        let (fault, context) = unsafe { (&*(*info).ExceptionRecord, &*(*info).ContextRecord) };
        let site = TITLE_INSTRUCTION.load(Relaxed);
        let page = TEST_PAGE.load(Relaxed);
        let expected_code = TITLE_FAULT_CODE.load(Relaxed);
        let expected_operation = if expected_code == STATUS_GUARD_PAGE_VIOLATION {
            GUARD_OPERATION.load(Relaxed)
        } else {
            1
        };
        if fault.ExceptionCode != expected_code
            || fault.ExceptionAddress as usize != site
            || context.Rip as usize != site
            || fault.NumberParameters < 2
            || fault.ExceptionInformation[0] != expected_operation
            || fault.ExceptionInformation[1] != page + PAGE - 1
            || context.R9 as usize != page + PAGE
            || context.Rcx != context.R9
        {
            return EXCEPTION_CONTINUE_SEARCH;
        }
        // SAFETY: the routine passes its stack pointer to the test via RDX
        // and keeps this synthetic non-leaf function registered.
        unsafe {
            let expected_rsp = (context.Rdx as *const u64).read();
            let mut image = 0;
            let entry = RtlLookupFunctionEntry(context.Rip, &mut image, std::ptr::null_mut());
            if entry.is_null() || context.Rsp != expected_rsp {
                return EXCEPTION_CONTINUE_SEARCH;
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
                return EXCEPTION_CONTINUE_SEARCH;
            }
        }
        if expected_code == EXCEPTION_ACCESS_VIOLATION && {
            // SAFETY: publish write access only for the synthetic test page.
            unsafe { VirtualProtect(page as *const _, PAGE, PAGE_READWRITE, &mut 0) == 0 }
        } {
            return EXCEPTION_CONTINUE_SEARCH;
        }
        TITLE_INSTRUCTION.store(0, Relaxed);
        EXCEPTION_CONTINUE_EXECUTION
    }

    unsafe extern "system" fn test_guard(info: *mut EXCEPTION_POINTERS) -> i32 {
        // SAFETY: the system returns the current test thread's numeric identifier.
        if unsafe { GetCurrentThreadId() } != TEST_THREAD.load(Relaxed) {
            return EXCEPTION_CONTINUE_SEARCH;
        }
        // SAFETY: the OS provides this exception record.
        let fault = unsafe { &*(*info).ExceptionRecord };
        let page = TEST_PAGE.load(Relaxed);
        if TITLE_INSTRUCTION.load(Relaxed) != 0 {
            if fault.ExceptionCode == STATUS_GUARD_PAGE_VIOLATION {
                GUARD_OPERATION.store(fault.ExceptionInformation[0], Relaxed);
            }
            return EXCEPTION_CONTINUE_SEARCH;
        }
        if fault.ExceptionCode == EXCEPTION_ACCESS_VIOLATION
            && fault.NumberParameters >= 2
            && fault.ExceptionInformation[0] == 1
            && fault.ExceptionInformation[1] == page + PAGE - 1
        {
            FAULTS.fetch_add(1, Relaxed);
            TEST_GUARDS[page / PAGE % TEST_GUARDS.len()].store(0, Relaxed);
            // SAFETY: our test page and only this expected byte-write fault.
            if unsafe { VirtualProtect(page as *const _, PAGE, PAGE_READWRITE, &mut 0) } != 0 {
                return EXCEPTION_CONTINUE_EXECUTION;
            }
        }
        EXCEPTION_CONTINUE_SEARCH
    }

    #[test]
    fn guarded_unchanged_flags_skip_the_store_and_changes_fault_as_before() {
        const SITE: u32 = 10;
        const PATCHED: u32 = 32;
        const UNWIND: u32 = 128;
        let code = [
            0x48, 0x83, 0xec, 0x28, // sub rsp, 0x28
            0x48, 0x89, 0x22, // mov [rdx], rsp
            0x49, 0x89, 0xc9, // mov r9, rcx
            OR[0], OR[1], OR[2], OR[3], OR[4], 0x9c, 0x58, // pushfq; pop rax
            0x48, 0x83, 0xc4, 0x28, 0xc3, // add rsp, 0x28; ret
        ];
        // SAFETY: fresh allocations contain our synthetic routines and a test page.
        unsafe {
            let image = VirtualAlloc(
                std::ptr::null(),
                PAGE,
                MEM_RESERVE | MEM_COMMIT,
                PAGE_READWRITE,
            )
            .cast::<u8>();
            let data = VirtualAlloc(
                std::ptr::null(),
                2 * PAGE,
                MEM_RESERVE | MEM_COMMIT,
                PAGE_READWRITE,
            )
            .cast::<u8>();
            assert!(!image.is_null() && !data.is_null());
            assert_ne!(
                VirtualProtect(data.add(PAGE).cast(), PAGE, PAGE_NOACCESS, &mut 0),
                0
            );
            let patched = image.add(PATCHED as usize);
            std::ptr::copy_nonoverlapping(code.as_ptr(), image, code.len());
            std::ptr::copy_nonoverlapping(code.as_ptr(), patched, code.len());
            // Version 1, a 4-byte prologue and one UWOP_ALLOC_SMALL (40 bytes).
            let unwind = [1, 4, 1, 0, 4, 0x42, 0, 0];
            std::ptr::copy_nonoverlapping(
                unwind.as_ptr(),
                image.add(UNWIND as usize),
                unwind.len(),
            );
            assert_ne!(
                VirtualProtect(image.cast(), PAGE, PAGE_EXECUTE_READ, &mut 0),
                0
            );

            let recorder = recorder(patched as usize, code.len() as u32);
            let wrong_digest = [(SITE, blake3::hash(b"old operation"))]
                .into_iter()
                .collect();
            assert_eq!(
                apply(&recorder, &code, &wrong_digest, &TEST_GUARDS, 16).unwrap(),
                0
            );
            let valid = [(SITE, blake3::hash(&OR))].into_iter().collect();
            let mut different_original = code;
            different_original[0] = 0x90; // Earlier rewrite immediately before the site.
            assert_eq!(
                apply(&recorder, &different_original, &valid, &TEST_GUARDS, 16).unwrap(),
                0
            );
            assert_eq!(
                apply(&recorder, &code, &valid, &TEST_GUARDS, 16).unwrap(),
                1
            );
            assert_eq!(patched.add(SITE as usize).read(), 0xe8);

            type Routine = unsafe extern "system" fn(usize, *mut u64) -> u64;
            let original: Routine = std::mem::transmute(image);
            let rewritten: Routine = std::mem::transmute(patched);
            let mut title_rsp = 0;
            let page = data as usize;
            let end = page + PAGE;
            let target = data.add(PAGE - 1);
            let neighbor = data.add(PAGE - 2);
            let slot = &TEST_GUARDS[page / PAGE % TEST_GUARDS.len()];
            TEST_THREAD.store(GetCurrentThreadId(), Relaxed);
            TEST_PAGE.store(page, Relaxed);
            let handler = AddVectoredExceptionHandler(1, Some(test_guard));
            assert!(!handler.is_null());
            neighbor.write(0xa6);
            // Check both same and changed bytes. Put the target at the edge
            // of our guard so an accidental wide access cannot be harmless.
            const FLAGS: u64 = 0x8c5; // OF, SF, ZF, PF, CF; AF is undefined
            let mut changes = 0;
            for before in 0..=255u8 {
                target.write(before);
                let expected = original(end, &mut title_rsp) & FLAGS;
                target.write(before);
                assert_ne!(VirtualProtect(data.cast(), PAGE, PAGE_READONLY, &mut 0), 0);
                slot.store(page, Relaxed);
                assert_eq!(rewritten(end, &mut title_rsp) & FLAGS, expected);
                assert_eq!(target.read(), before | 0x80);
                assert_eq!(neighbor.read(), 0xa6);
                changes += usize::from(before < 0x80);
                assert_eq!(FAULTS.load(Relaxed), changes);
                slot.store(0, Relaxed);
                assert_ne!(VirtualProtect(data.cast(), PAGE, PAGE_READWRITE, &mut 0), 0);
            }
            // A same-value write still faults on someone else's read-only page.
            assert_ne!(VirtualProtect(data.cast(), PAGE, PAGE_READONLY, &mut 0), 0);
            rewritten(end, &mut title_rsp);
            assert_eq!(FAULTS.load(Relaxed), changes + 1);

            // Leave the observer installed to capture the original guard
            // notification, while the title handles the retry.
            let function = IMAGE_RUNTIME_FUNCTION_ENTRY {
                BeginAddress: PATCHED,
                EndAddress: PATCHED + code.len() as u32,
                Anonymous: IMAGE_RUNTIME_FUNCTION_ENTRY_0 { UnwindData: UNWIND },
            };
            assert!(RtlAddFunctionTable(&function, 1, image as u64));
            let retry = AddVectoredExceptionHandler(0, Some(retry_title_write));
            assert!(!retry.is_null());
            for (protection, cached, status) in [
                (PAGE_READONLY, false, EXCEPTION_ACCESS_VIOLATION),
                (PAGE_NOACCESS, true, EXCEPTION_ACCESS_VIOLATION),
                (
                    PAGE_READWRITE | PAGE_GUARD,
                    false,
                    STATUS_GUARD_PAGE_VIOLATION,
                ),
                (
                    PAGE_READWRITE | PAGE_GUARD,
                    true,
                    STATUS_GUARD_PAGE_VIOLATION,
                ),
            ] {
                assert_ne!(VirtualProtect(data.cast(), PAGE, protection, &mut 0), 0);
                slot.store(if cached { page } else { 0 }, Relaxed);
                GUARD_OPERATION.store(usize::MAX, Relaxed);
                TITLE_FAULT_CODE.store(status, Relaxed);
                TITLE_INSTRUCTION.store(patched as usize + SITE as usize, Relaxed);
                rewritten(end, &mut title_rsp);
                assert_eq!(TITLE_INSTRUCTION.load(Relaxed), 0);
            }
            slot.store(0, Relaxed);
            assert_ne!(RemoveVectoredExceptionHandler(retry), 0);
            assert_ne!(RemoveVectoredExceptionHandler(handler), 0);
            assert!(RtlDeleteFunctionTable(&function));
            TEST_PAGE.store(0, Relaxed);
            assert_ne!(VirtualFree(data.cast(), 0, MEM_RELEASE), 0);
            assert_ne!(VirtualFree(image.cast(), 0, MEM_RELEASE), 0);
        }
    }
}
