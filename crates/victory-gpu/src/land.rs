//! Post-fence buffer write-back. Only one submission lands at a time, in
//! declaration order, including writes to different views of one section.

use std::cell::Cell;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};

/// The runtime holds its mapping lock for the duration of the callback.
pub enum LandingPage {
    /// The title address is no longer the destination of this submission.
    Replaced,
    /// Use the checked kernel copy at the title address.
    Direct,
    /// A committed writable view of the same bytes, valid for the callback.
    Beside(usize),
}

type Lookup = fn(usize, usize, u64, bool, &mut dyn FnMut(LandingPage, usize));
static LOOKUP: OnceLock<Lookup> = OnceLock::new();

/// Installs the physical page lookup used while write-back lands. The lookup
/// must call its callback once with the destination and length of a stable
/// prefix of the requested range. The prefix ends at the requested end or
/// a 64 KiB boundary and must cover at least the first requested block.
/// Mapping changes must advance the epoch before publishing the new mapping.
///
/// # Safety
/// Any address returned as `Beside` must be committed and writable and
/// must represent the requested bytes for the entire callback. The lookup
/// must also exclude runtime writes through that same alternate view.
pub unsafe fn use_landing_pages(lookup: Lookup) {
    let _ = LOOKUP.set(lookup);
}

static MAPPING_EPOCH: AtomicU64 = AtomicU64::new(0);

/// Called by the runtime while changing its physical mappings.
pub fn next_mapping_epoch() -> u64 {
    MAPPING_EPOCH.fetch_add(1, Ordering::AcqRel) + 1
}

/// Whether guarded write-back is enabled for this process.
pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_LAND").is_some_and(|v| v == "1"))
}

pub(crate) fn begin() -> Option<u64> {
    enabled().then(|| MAPPING_EPOCH.load(Ordering::Acquire))
}

thread_local! {
    static SUBMISSION: Cell<Option<u64>> = const { Cell::new(None) };
}

pub(crate) fn during<R>(epoch: Option<u64>, work: impl FnOnce() -> R) -> R {
    struct Restore(Option<u64>);
    impl Drop for Restore {
        fn drop(&mut self) {
            SUBMISSION.set(self.0);
        }
    }
    let _restore = Restore(SUBMISSION.replace(epoch));
    work()
}

pub(crate) fn active() -> bool {
    SUBMISSION.get().is_some()
}

pub(crate) fn prefix(at: usize, len: usize, mut apply: impl FnMut(LandingPage, usize)) {
    static BESIDE: OnceLock<bool> = OnceLock::new();
    if let (Some(epoch), Some(lookup)) = (SUBMISSION.get(), LOOKUP.get()) {
        let beside = *BESIDE
            .get_or_init(|| std::env::var_os("VICTORY_LAND_BESIDE").is_none_or(|v| v != "0"));
        lookup(at, len, epoch, beside, &mut apply);
    } else {
        apply(LandingPage::Direct, len);
    }
}

pub(crate) fn echo() -> bool {
    static ECHO: OnceLock<bool> = OnceLock::new();
    *ECHO.get_or_init(|| std::env::var_os("VICTORY_LAND_ECHO").is_none_or(|v| v != "0"))
}

type Job = Box<dyn FnOnce() + Send>;

pub(crate) fn async_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var_os("VICTORY_RELAY").is_none_or(|v| v != "0")
            && (crate::switch::on("VICTORY_RELAY_ASYNC")
                || std::env::var_os("VICTORY_RELAY").is_some_and(|v| v == "2"))
    })
}

type ResultReceiver<R> = Receiver<std::thread::Result<R>>;

pub(crate) struct Landing<R> {
    ready: Option<R>,
    result: Option<ResultReceiver<R>>,
}

impl<R> Landing<R> {
    pub(crate) fn finish(mut self) -> R {
        if let Some(value) = self.ready.take() {
            return value;
        }
        match self
            .result
            .take()
            .expect("pending landing")
            .recv()
            .expect("landing completed")
        {
            Ok(value) => value,
            Err(panic) => std::panic::resume_unwind(panic),
        }
    }
}

impl<R> Drop for Landing<R> {
    fn drop(&mut self) {
        // Even on unwind the submission cannot give up its lock while the
        // worker can change the shared shadows or publish title bytes.
        if let Some(result) = self.result.take() {
            let _ = result.recv();
        }
    }
}

/// Start one post-fence merge. The ticket must be joined under the
/// submission lock before any conflicting refresh or completion write.
pub(crate) fn start<R: Send + 'static>(
    epoch: Option<u64>,
    work: impl FnOnce() -> R + Send + 'static,
) -> Landing<R> {
    static WORKER: OnceLock<Option<SyncSender<Job>>> = OnceLock::new();
    let worker = epoch.and_then(|_| {
        WORKER
            .get_or_init(|| {
                if std::env::var_os("VICTORY_RELAY").is_some_and(|v| v == "0") {
                    return None;
                }
                let (send, recv) = sync_channel::<Job>(1);
                std::thread::Builder::new()
                    .name("victory-land".into())
                    .spawn(move || {
                        while let Ok(job) = recv.recv() {
                            job();
                        }
                    })
                    .ok()
                    .map(|_| send)
            })
            .as_ref()
    });
    let Some(worker) = worker else {
        return Landing {
            ready: Some(during(epoch, work)),
            result: None,
        };
    };
    let (done, result) = sync_channel(1);
    let job = Box::new(move || {
        let ran = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| during(epoch, work)));
        let _ = done.send(ran);
    });
    if let Err(disconnected) = worker.send(job) {
        // The worker is gone, so execute the work here.
        disconnected.0();
    }
    Landing {
        ready: None,
        result: Some(result),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::HostGpu;
    use crate::watch::PAGE;
    use std::sync::atomic::AtomicUsize;
    use windows::core::s;
    use windows_sys::Win32::Foundation::CloseHandle;
    use windows_sys::Win32::System::Memory::{
        CreateFileMappingW, FILE_MAP_WRITE, MEMORY_BASIC_INFORMATION, MapViewOfFile, PAGE_READONLY,
        PAGE_READWRITE, UnmapViewOfFile, VirtualQuery,
    };

    #[test]
    fn landing_ticket_allows_work_before_join_and_preserves_epoch() {
        if std::env::var_os("VICTORY_RELAY").is_some_and(|v| v == "0") {
            return;
        }
        let (resume, blocked) = sync_channel(1);
        let ticket = start(Some(37), move || {
            assert_eq!(SUBMISSION.get(), Some(37));
            matches!(
                blocked.recv_timeout(std::time::Duration::from_secs(2)),
                Ok(42)
            )
        });
        // The worker cannot finish until after start returns.
        resume.send(42).unwrap();
        assert!(ticket.finish());
        assert!(!active());
    }

    static VIEWS: std::sync::Mutex<Option<[usize; 3]>> = std::sync::Mutex::new(None);
    static REPLACEMENT: std::sync::Mutex<Option<(usize, u64)>> = std::sync::Mutex::new(None);
    static LOOKUPS: AtomicUsize = AtomicUsize::new(0);
    static ON_WORKER: AtomicUsize = AtomicUsize::new(0);

    thread_local! {
        static BATCH_VIEWS: Cell<Option<(usize, usize)>> = const { Cell::new(None) };
        static BATCH_LOOKUPS: Cell<usize> = const { Cell::new(0) };
    }

    fn physical_test_page(
        at: usize,
        len: usize,
        epoch: u64,
        beside: bool,
        apply: &mut dyn FnMut(LandingPage, usize),
    ) {
        if let Some((base, writable)) = BATCH_VIEWS.get()
            && (base..base + 5 * PAGE).contains(&at)
        {
            BATCH_LOOKUPS.set(BATCH_LOOKUPS.get() + 1);
            let page = (at - base) / PAGE;
            let (destination, end) = match page {
                0..=1 if beside => (LandingPage::Beside(writable + at - base), 2),
                0..=1 => (LandingPage::Direct, 2),
                2..=3 => (LandingPage::Direct, 4),
                _ => (LandingPage::Replaced, 5),
            };
            apply(destination, len.min(base + end * PAGE - at));
            return;
        }
        if REPLACEMENT
            .lock()
            .unwrap()
            .is_some_and(|(base, mapped)| (base..base + PAGE).contains(&at) && mapped > epoch)
        {
            apply(LandingPage::Replaced, len.min(PAGE - at % PAGE));
            return;
        }
        let views = VIEWS.lock().unwrap();
        if let Some([a, _, writable]) = *views
            && (a..a + 2 * PAGE).contains(&at)
        {
            LOOKUPS.fetch_add(1, Ordering::Relaxed);
            if std::thread::current().name() == Some("victory-land") {
                ON_WORKER.fetch_add(1, Ordering::Relaxed);
            }
            if beside {
                apply(
                    LandingPage::Beside(writable + at - a),
                    len.min(a + 2 * PAGE - at),
                );
                return;
            }
        }
        apply(LandingPage::Direct, len);
    }

    fn protection(at: usize) -> u32 {
        let mut info = MEMORY_BASIC_INFORMATION::default();
        // SAFETY: query of an address in this process into our structure.
        let size = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
        assert_ne!(size, 0);
        info.Protect
    }

    /// Computed writes use the same mapping epoch for guarded copies and
    /// raw title-address copies. Run with each fault-tracking setting.
    #[test]
    fn computed_landing_rejects_a_replacement() {
        let Some(epoch) = begin() else {
            return;
        };
        assert!(
            crate::host::host().is_some(),
            "the landing fixture needs a GPU"
        );
        let p = crate::watch::tests::pool(1);
        crate::shadow::register(p, p as u64, PAGE as u64);
        let copy = crate::shadow::find(p as u64, PAGE as u64).expect("a host shadow");
        // SAFETY: the test callback only supplies another view while the
        // fixture owning it is live.
        unsafe { use_landing_pages(physical_test_page) };
        assert_eq!(copy.update(p, PAGE), PAGE);
        assert_eq!(copy.clean(), crate::watch::fault_tracking());
        let baseline = copy
            .dirty_bit()
            .is_none()
            .then(|| copy.computed_baseline(0, PAGE));
        copy.write_host(p as u64, &[1]);
        if baseline.is_some() {
            assert_eq!(during(Some(epoch), || copy.copy_back_page(p, None)), 0);
            // SAFETY: the fixture owns the still-current destination.
            assert_eq!(unsafe { (p as *const u8).read_volatile() }, 0);
        }
        assert_eq!(
            during(Some(epoch), || copy.copy_back_page(p, baseline.as_deref())),
            PAGE
        );
        // SAFETY: the fixture owns the still-current destination.
        assert_eq!(unsafe { (p as *const u8).read_volatile() }, 1);

        let copied = copy.update(p, PAGE);
        let echoes = !crate::watch::fault_tracking() || !echo();
        assert_eq!(copied, if echoes { PAGE } else { 0 });
        eprintln!("Direct pool write-back: origin refreshed {copied} bytes");

        let baseline = copy
            .dirty_bit()
            .is_none()
            .then(|| copy.computed_baseline(0, PAGE));
        copy.write_host(p as u64, &[2]);
        crate::watch::title_memory_remapped(p, PAGE);
        *REPLACEMENT.lock().unwrap() = Some((p, next_mapping_epoch()));
        // SAFETY: the pool is still committed and its former guard is open.
        unsafe { (p as *mut u8).write_volatile(3) };
        assert_eq!(
            during(Some(epoch), || copy.copy_back_page(p, baseline.as_deref())),
            0
        );
        // SAFETY: still the fixture's live pool page.
        assert_eq!(unsafe { (p as *const u8).read_volatile() }, 3);
        eprintln!(
            "replacement kept with fault tracking {}",
            crate::watch::fault_tracking()
        );

        *REPLACEMENT.lock().unwrap() = None;
        drop(copy);
        crate::shadow::unregister(p, p as u64);
        crate::watch::tests::free(p);
    }

    #[test]
    fn missed_landing_mark_withholds_shared_shadow_and_completes_title() {
        if !crate::watch::fault_tracking() {
            return;
        }
        let require_marks =
            enabled() || crate::descriptor::host::show_after() || crate::texback::enabled();
        let host = crate::host::host().expect("the landing fixture needs a GPU");
        let at = crate::watch::tests::pool(1);
        crate::shadow::register(at, at as u64, PAGE as u64);
        let shadow = crate::shadow::find(at as u64, PAGE as u64).unwrap();
        let marks = crate::shadow::view_writes(false).unwrap();
        let (mark, _) = shadow.mark_entry(at as u64).unwrap();
        let read = |address| {
            // SAFETY: aligned words held by the fixture through the fence.
            unsafe { (address as *const u32).read_volatile() }
        };
        let word = |address, value| {
            let _writing = crate::watch::writing(address, size_of::<u32>());
            // SAFETY: aligned words held by the fixture across all writes.
            unsafe { (address as *mut u32).write_volatile(value) };
        };
        let mut completed = Box::new([0u32; 3]);
        let notification = completed.as_mut_ptr() as usize;

        for fail in [false, true] {
            word(at, 11);
            word(at + 4, 22);
            for offset in [0, 4, 8] {
                word(notification + offset, 0);
            }
            let lists = [(0, 100), (4, 200)].map(|(offset, value)| {
                let shader = format!(
                    "RWByteAddressBuffer output : register(u0);
                     RWByteAddressBuffer marks : register(u1);
                     [numthreads(1,1,1)] void main() {{
                         output.Store({offset}, {value});
                         marks.Store({}, {});
                     }}",
                    mark / 32 * 4,
                    1u32 << (mark % 32),
                );
                let params = [0, 1].map(|register| crate::compute::Param::Uav { register });
                let (root, pipeline) = crate::compute::pipeline(
                    &shader,
                    s!("confirmed landing groups test"),
                    0,
                    &params,
                )
                .unwrap();
                let list = host.new_list().unwrap();
                shadow.refresh(&list, at as u64, PAGE as u64);
                shadow.gpu_writes(&list, at as u64, PAGE as u64, false);
                list.keep(&root);
                list.keep(&pipeline);
                // SAFETY: the list is open and the allocations outlive execution.
                unsafe {
                    list.list.SetComputeRootSignature(&root);
                    list.list.SetPipelineState(&pipeline);
                    list.list
                        .SetComputeRootUnorderedAccessView(0, shadow.gpu(at as u64));
                    list.list.SetComputeRootUnorderedAccessView(1, marks);
                    list.list.Dispatch(1, 1, 1);
                }
                list.defer_after_with_baselines(move |_, accepted| {
                    assert_eq!(accepted, !(require_marks && fail));
                    assert_eq!(read(notification), 0);
                });
                for destination in [notification, notification + 4 + offset as usize] {
                    list.defer_write(
                        destination as *mut u8,
                        crate::queue::Written::Value(value),
                        size_of::<u32>(),
                    );
                }
                list.close();
                list
            });
            crate::host::MISS_NEXT_SUBMISSION_MARK.set(fail);
            host.execute(&[&lists[0], &lists[1]]);
            assert_eq!(
                *completed,
                [200, 100, 200],
                "both title completions are ordered"
            );
            if require_marks && fail {
                assert_eq!((read(at), read(at + 4)), (11, 22));
                assert!(!shadow.clean(), "the tainted host page must be refreshed");
            } else {
                assert_eq!((read(at), read(at + 4)), (100, 200));
            }
        }
        drop(shadow);
        crate::shadow::unregister(at, at as u64);
        crate::watch::tests::free(at);
    }

    #[test]
    fn marked_shadow_coalesces_beside_and_direct_and_skips_replaced_pages() {
        use crate::watch::{self, tests::PROTECTS};
        if !watch::fault_tracking() {
            return;
        }

        const LEN: usize = 5 * PAGE;
        // SAFETY: a fresh section mapped at three addresses. This test owns
        // the views until the shadow and its trackers are dropped.
        let views = unsafe {
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                LEN as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            let views = [(); 3].map(|()| MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, LEN));
            assert!(views.iter().all(|v| !v.Value.is_null()));
            CloseHandle(section);
            views
        };
        let [a, b, writable] = views.map(|v| v.Value as usize);
        let pages = std::array::from_fn::<_, 5, _>(|p| a + p * PAGE);
        for at in pages {
            watch::title_memory_aliased(b + at - a, &[at]);
        }
        let other = watch::track_copy(b, LEN, true).unwrap();
        assert_eq!(other.take(b, b + LEN), [(b, b + LEN)]);
        crate::shadow::register(a, a as u64, LEN as u64);
        let copy = crate::shadow::find(a as u64, LEN as u64).expect("a host shadow");
        assert_eq!(copy.update(a, LEN), LEN);
        for (i, at) in pages.into_iter().enumerate() {
            copy.write_host((at + 31) as u64, &[i as u8 + 1, 0, 42]);
        }
        // SAFETY: this thread exposes the writable view only while the
        // fixture owns it and no section writer is active.
        unsafe { use_landing_pages(physical_test_page) };
        BATCH_VIEWS.set(Some((a, writable)));
        BATCH_LOOKUPS.set(0);
        let protects = PROTECTS.get();
        assert_eq!(
            during(Some(0), || copy.copy_back_marked(&pages, None)),
            4 * PAGE
        );
        assert_eq!(BATCH_LOOKUPS.get(), 3);
        let beside = std::env::var_os("VICTORY_LAND_BESIDE").is_none_or(|v| v != "0");
        if beside {
            for at in [a, a + PAGE, b, b + PAGE] {
                assert_eq!(protection(at), PAGE_READONLY);
            }
            assert_eq!(PROTECTS.get() - protects, if echo() { 2 } else { 0 });
        }
        for i in 0..5 {
            let expected = if i < 4 { [i as u8 + 1, 0, 42] } else { [0; 3] };
            for at in [a, b, writable] {
                // SAFETY: a view owned by the test with the landing joined.
                let bytes =
                    unsafe { std::slice::from_raw_parts((at + i * PAGE + 31) as *const u8, 3) };
                assert_eq!(bytes, expected);
            }
        }
        assert_eq!(other.take(b, b + LEN), [(b, b + 4 * PAGE)]);

        // An explicit unchanged baseline skips the physical lookup entirely.
        let mut same = [0; 32];
        assert_eq!(
            during(Some(0), || watch::land(
                a,
                &mut same,
                Some(&[0; 32]),
                None,
                &mut []
            )),
            Some(0)
        );
        assert_eq!(BATCH_LOOKUPS.get(), 3);
        BATCH_VIEWS.set(None);
        drop((copy, other));
        crate::shadow::unregister(a, a as u64);
        watch::title_memory_remapped(a, LEN);
        watch::title_memory_remapped(b, LEN);
        for view in views {
            // SAFETY: this fixture owns the view and its trackers are done.
            assert_ne!(unsafe { UnmapViewOfFile(view) }, 0);
        }
    }

    /// Run on Wine with VICTORY_FAULT_TRACKING=1, once for each switch
    /// configuration. The shader writes one snapshot range and one marked
    /// page. A CPU store through another title alias runs after the GPU
    /// fence but before landing checks the result.
    #[test]
    fn fenced_shader_write_lands_beside_guards() {
        let Some(host) = crate::host::host() else {
            assert!(begin().is_none(), "the focused landing test needs a GPU");
            return;
        };
        if !crate::watch::fault_tracking() {
            return; // The guard and write-mark assertions require tracking.
        }
        // SAFETY: a fresh section, all three views owned by this test.
        let views = unsafe {
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (2 * PAGE) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            let views = [(); 3].map(|()| MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, 2 * PAGE));
            assert!(views.iter().all(|v| !v.Value.is_null()));
            CloseHandle(section);
            views
        };
        let [a, b, writable] = views.map(|v| v.Value as usize);
        *VIEWS.lock().unwrap() = Some([a, b, writable]);
        // SAFETY: the test owns the third committed view until all landings finish.
        unsafe { use_landing_pages(physical_test_page) };
        for k in [0, PAGE] {
            crate::watch::title_memory_aliased(b + k, &[a + k]);
        }
        let copy = |at| {
            crate::shadow::register(at, at as u64, PAGE as u64);
            crate::shadow::find(at as u64, PAGE as u64).expect("a host shadow")
        };
        let (exact, marked, other) = (copy(a), copy(a + PAGE), copy(b));
        assert_eq!(other.update(b, PAGE), PAGE);
        let exact_view = exact
            .widened(a as u64, 12, 1)
            .expect("widened snapshot view");
        let marked_view = marked
            .widened((a + PAGE) as u64, 36, 1)
            .expect("widened marked view");
        let marks = crate::shadow::view_writes(false).expect("the GPU write bitmap");
        let (bit, _) = marked
            .mark_entry((a + PAGE) as u64)
            .expect("a tracked shadow");
        let shader = format!(
            "RWByteAddressBuffer exact : register(u0);
             RWByteAddressBuffer marked : register(u1);
             RWByteAddressBuffer marks : register(u2);
             [numthreads(1,1,1)] void main() {{
                 exact.Store(4, 0x11223344);
                 marked.Store(4, 0x55667788);
                 marks.Store({}, {});
             }}",
            bit / 32 * 4,
            1u32 << (bit % 32),
        );
        let params = [0, 1, 2].map(|register| crate::compute::Param::Uav { register });
        let (root, pso) =
            crate::compute::pipeline(&shader, s!("landing test"), 0, &params).unwrap();
        let list = host.new_list().unwrap();
        exact.refresh(&list, a as u64, PAGE as u64);
        exact.gpu_writes(&list, a as u64, PAGE as u64, false);
        marked.refresh(&list, (a + PAGE) as u64, PAGE as u64);
        marked.gpu_writes(&list, (a + PAGE) as u64, PAGE as u64, true);
        list.defer_missed(move |_| {
            assert_eq!(protection(b + PAGE), PAGE_READONLY);
            // SAFETY: our title alias; the guard handler opens it and saves
            // the old bytes for the armed GPU copy.
            unsafe { ((b + PAGE + 24) as *mut u8).write_volatile(0x9a) };
            false
        });
        let word = |at| {
            // SAFETY: these are aligned words of our mapped section.
            unsafe { (at as *const u32).read_volatile() }
        };
        let mut notification = Box::new(0u32);
        let notify_at = (&raw mut *notification) as usize;
        list.defer_after(move || {
            assert_eq!(word(a + 4), 0x11223344);
            assert_eq!(word(b + PAGE + 4), 0x55667788);
            assert_eq!(word(a + PAGE + 24), 0x9a);
            assert_eq!(
                word(notify_at),
                0,
                "bottom-of-pipe notification follows landing"
            );
        });
        list.defer_write(notify_at as *mut u8, crate::queue::Written::Value(1), 4);
        // SAFETY: open list and live pipeline, shadows and bitmap.
        unsafe {
            list.list.SetComputeRootSignature(&root);
            list.list.SetPipelineState(&pso);
            list.list
                .SetComputeRootUnorderedAccessView(0, exact.gpu(a as u64));
            list.list
                .SetComputeRootUnorderedAccessView(1, marked.gpu((a + PAGE) as u64));
            list.list.SetComputeRootUnorderedAccessView(2, marks);
            list.list.Dispatch(1, 1, 1);
        }
        list.close();
        host.execute(&[&list]);
        assert_eq!(word(notify_at), 1);
        assert_eq!(exact_view.read_host(4), [0, 0x11223344, 0, 1]);
        assert_eq!(marked_view.read_host(4), [0, 0x55667788, 0, 1]);
        assert_eq!(protection(writable), PAGE_READWRITE);
        let on = begin().is_some();
        assert_eq!(LOOKUPS.load(Ordering::Relaxed), if on { 2 } else { 0 });
        let relay_on = std::env::var_os("VICTORY_RELAY").is_none_or(|v| v != "0");
        assert_eq!(
            ON_WORKER.load(Ordering::Relaxed),
            if on && relay_on { 2 } else { 0 }
        );
        let beside_on = std::env::var_os("VICTORY_LAND_BESIDE").is_none_or(|v| v != "0");
        let mut echo_bytes = None;
        if on && beside_on {
            assert_eq!(protection(a), PAGE_READONLY);
            assert_eq!(protection(b), PAGE_READONLY);
            assert_eq!(protection(a + PAGE), PAGE_READONLY);
            assert_eq!(
                protection(b + PAGE),
                PAGE_READWRITE,
                "only the CPU store opened this alias"
            );
        }
        if on {
            let copied = exact.update(a, PAGE);
            assert_eq!(copied, if echo() { 0 } else { PAGE });
            echo_bytes = Some(copied);
        }
        assert_eq!(
            other.update(b, PAGE),
            PAGE,
            "another shadow must see the GPU write"
        );
        assert_eq!(other.read_host((b + 4) as u64, 1), [0x11223344]);
        // A later CPU store still dirties the originating shadow.
        // SAFETY: the alias is mapped through the test.
        unsafe { ((b + 20) as *mut u8).write_volatile(0x7c) };
        assert_eq!(exact.update(a, PAGE), PAGE);
        assert_eq!(exact.read_host((a + 20) as u64, 1), [0x7c]);

        eprintln!(
            "two GPU stores: {} landing lookups, {} on the relay, origin refresh bytes {echo_bytes:?}",
            LOOKUPS.load(Ordering::Relaxed),
            ON_WORKER.load(Ordering::Relaxed),
        );
        drop((list, exact_view, marked_view, exact, marked, other));
        for at in [a, a + PAGE, b] {
            crate::shadow::unregister(at, at as u64);
        }
        crate::watch::title_memory_remapped(a, 2 * PAGE);
        crate::watch::title_memory_remapped(b, 2 * PAGE);
        *VIEWS.lock().unwrap() = None;
        for v in views {
            // SAFETY: this test owns each view; the host and trackers are done.
            assert_ne!(unsafe { UnmapViewOfFile(v) }, 0);
        }
    }
}
