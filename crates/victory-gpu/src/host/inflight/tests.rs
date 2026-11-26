use super::*;
use crate::object::{HRESULT, S_OK, This, slot};

#[test]
fn depth_defaults_and_caps_at_eight() {
    assert_eq!(parse_limit("1"), 4);
    assert_eq!(parse_limit("2"), 2);
    assert_eq!(parse_limit(" 8 "), 8);
    assert_eq!(parse_limit("22"), 8);
    assert_eq!(parse_limit("0"), 0);
    assert_eq!(parse_limit(""), 0);
    assert_eq!(parse_limit("broken"), 0);
}

/// A native queue wait holds the GPU while CPU submission proceeds.
/// A watchdog opens it on a regression so the test fails without a hang.
struct Gate {
    fence: ID3D12Fence,
    watchdog: Option<(SyncSender<()>, std::thread::JoinHandle<()>)>,
}

impl Gate {
    fn new(host: &Host) -> Self {
        // SAFETY: a new D3D12 fence and the host's direct queue.
        let fence: ID3D12Fence =
            unsafe { host.device.CreateFence(0, D3D12_FENCE_FLAG_NONE) }.unwrap();
        let (done, wait) = std::sync::mpsc::sync_channel(1);
        let wake = fence.clone();
        let watchdog = std::thread::spawn(move || {
            let _ = wait.recv_timeout(Duration::from_secs(5));
            // SAFETY: the CPU may signal this live fence from any thread.
            unsafe { wake.Signal(1) }.unwrap();
        });
        let gate = Self {
            fence,
            watchdog: Some((done, watchdog)),
        };
        // SAFETY: a native queue wait, released by the CPU.
        unsafe { host.queue().Wait(&gate.fence, 1) }.unwrap();
        gate
    }

    fn open(&mut self) {
        if let Some((done, watchdog)) = self.watchdog.take() {
            let _ = done.try_send(());
            watchdog.join().unwrap();
        }
    }
}

impl Drop for Gate {
    fn drop(&mut self) {
        self.open();
    }
}

fn host_for_ring() -> Option<&'static Host> {
    // Run explicitly under the opt-in, including on patched Wine with
    // the hardware backend. Other workspace tests can run by default.
    (limit() != 0).then(|| host().expect("in-flight tests need a host GPU"))
}

fn record_copy(host: &Host, out: &ID3D12Resource, offset: u64, value: u32) -> Arc<HostList> {
    let list = host.new_list().unwrap();
    let (upload, at, data) = list.upload(4).unwrap();
    // SAFETY: four bytes of private upload and two live output ranges.
    unsafe {
        data.cast::<u32>().write(value);
        list.list.CopyBufferRegion(out, offset, &upload, at, 4);
        list.list.CopyBufferRegion(out, 0, &upload, at, 4);
    }
    list.close();
    Arc::new(list)
}

#[test]
fn flights_return_before_gpu_completion_and_reuse_bounded_storage() {
    let Some(host) = host_for_ring() else { return };
    host.present_boundary();
    let n = limit();
    let (out, data) = host.mapped_buffer((8 * n + 1) as u64 * 4, true).unwrap();
    let mut gate = Gate::new(host);
    let first = {
        let s = host.submitted.lock().unwrap();
        s.value + 1
    };
    for i in 1..=n * 8 {
        let list = record_copy(host, &out, (i * 4) as u64, i as u32);
        assert!(host.execute_title(std::slice::from_ref(&list)));
        if i <= n {
            let s = host.submitted.lock().unwrap();
            assert_eq!(s.flights.len(), i);
            // SAFETY: a live native fence holding this queue.
            assert!(unsafe { host.fence.GetCompletedValue() } < first);
            assert_eq!(
                Arc::strong_count(&list),
                2,
                "the old allocator is still pinned"
            );
        }
        if i == n {
            gate.open();
        }
        crate::list::recycle(list);
    }
    let mut s = host.submitted.lock().unwrap();
    let _holding = Holding::enter();
    host.drain_inflight(&mut s, Boundary::Synchronous);
    assert_eq!(s.marks.len(), n);
    assert!(s.marks.iter().all(|slot| slot.len() == 1));
    assert_eq!((s.flights.len(), s.lists, s.bytes), (0, 0, 0));
    // SAFETY: all writes to this persistently mapped readback finished.
    let read = unsafe { std::slice::from_raw_parts(data.cast::<u32>(), n * 8 + 1) };
    assert_eq!(read[0], (n * 8) as u32);
    assert!(read[1..].iter().copied().eq(1..=(n * 8) as u32));
}

#[test]
fn capacity_waits_for_the_oldest_fence_only() {
    let Some(host) = host_for_ring() else { return };
    if limit() < 2 {
        return;
    }
    host.present_boundary();
    let (out, _) = host.mapped_buffer(8, true).unwrap();
    let mut oldest = Gate::new(host);
    assert!(host.execute_title(&[record_copy(host, &out, 4, 1)]));
    let mut later = Gate::new(host);
    for value in 2..=limit() {
        assert!(host.execute_title(&[record_copy(host, &out, 4, value as u32)]));
    }
    let release = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(20));
        oldest.open();
    });
    assert!(host.execute_title(&[record_copy(host, &out, 4, 19)]));
    {
        let s = host.submitted.lock().unwrap();
        assert_eq!(s.flights.len(), limit());
        // SAFETY: a live host fence. The second gate is still closed.
        assert!(unsafe { host.fence.GetCompletedValue() } < s.flights[0].value);
    }
    release.join().unwrap();
    later.open();
    host.present_boundary();
}

#[test]
fn reset_uses_another_allocator_while_the_title_recording_is_in_flight() {
    use crate::object::guid;
    let Some(host) = host_for_ring() else { return };
    host.present_boundary();
    let null = std::ptr::null();
    let mut allocator = null;
    let mut title = null;
    let allocator_iid = guid(
        0x6102dee4,
        0xaf59,
        0x4b09,
        [0xb9, 0x99, 0xb4, 0x4d, 0x73, 0xf0, 0x9b, 0x24],
    );
    let list_iid = guid(
        0x5b160d0f,
        0xac1b,
        0x4185,
        [0x8b, 0xa8, 0xb3, 0xae, 0x42, 0xa5, 0xa4, 0x55],
    );
    assert_eq!(
        crate::list::create_allocator(null, 0, &allocator_iid, &mut allocator),
        S_OK
    );
    assert_eq!(
        crate::list::create_list(null, 0, 0, allocator, null, &list_iid, &mut title),
        S_OK
    );
    let close = slot::<extern "system" fn(This) -> HRESULT>(title, 0x48);
    let reset = slot::<extern "system" fn(This, This, This) -> HRESULT>(title, 0x50);
    let (out, data) = host.mapped_buffer(8, true).unwrap();
    let record = |offset, value: u32| {
        let list = crate::list::host_list(title).unwrap();
        let (upload, at, src) = list.upload(4).unwrap();
        // SAFETY: four bytes of private upload and a live output range.
        unsafe {
            src.cast::<u32>().write(value);
            list.list.CopyBufferRegion(&out, offset, &upload, at, 4);
        }
        assert_eq!(close(title), S_OK);
        list.allocator.as_raw()
    };
    let mut gate = Gate::new(host);
    let before = record(0, 41);
    assert!(host.execute_title(&crate::list::segments(title)));
    assert_eq!(reset(title, allocator, null), S_OK);
    assert_ne!(record(4, 73), before);
    assert!(host.execute_title(&crate::list::segments(title)));
    gate.open();
    crate::with_inflight_mapping(|| {
        // SAFETY: mapping retirement is a GPU completion boundary.
        assert_eq!(unsafe { data.cast::<[u32; 2]>().read() }, [41, 73]);
    });
    crate::object::release_ref(title);
    crate::object::release_ref(allocator);
}

#[test]
fn a_title_event_and_raw_fence_value_follow_gpu_writes() {
    use windows::Win32::Foundation::{CloseHandle, WAIT_OBJECT_0, WAIT_TIMEOUT};
    let Some(host) = host_for_ring() else { return };
    host.present_boundary();
    let (out, data) = host.shared_buffer(64 << 10).unwrap();
    // SAFETY: the shared buffer is idle and mapped.
    unsafe { data.cast::<u32>().write(0) };
    let null = std::ptr::null();
    let mut fence = null;
    assert_eq!(
        crate::fence::create(null, 0, 0, &crate::fence::FENCE_IIDS[0], &mut fence),
        S_OK
    );
    let wait = slot::<extern "system" fn(This, u64, *mut std::ffi::c_void) -> HRESULT>(fence, 0x48);
    let raw = slot::<extern "system" fn(This) -> u64>(fence, 0x60)(fence);
    // SAFETY: the live emulated fence exposes its aligned AtomicU64 here.
    let raw = unsafe { &*(raw as *const AtomicU64) };
    // SAFETY: plain auto-reset event and the fence's valid value address.
    let event = unsafe { CreateEventW(None, false, false, None) }.unwrap();
    let mut gate = Gate::new(host);
    let list = record_copy(host, &out, 4, 0x4422);
    host.execute_title(&[list]);
    host.signal_title(fence, 9);
    assert_eq!(wait(fence, 9, event.0), S_OK);
    // SAFETY: a valid event and a CPU-visible title fence value.
    assert_eq!(unsafe { WaitForSingleObject(event, 0) }, WAIT_TIMEOUT);
    assert_eq!(raw.load(Ordering::Acquire), 0);
    gate.open();
    // SAFETY: a valid event. No other queue call publishes this fence.
    assert_eq!(unsafe { WaitForSingleObject(event, 10_000) }, WAIT_OBJECT_0);
    assert_eq!(raw.load(Ordering::Acquire), 9);
    // SAFETY: the event signalled after the GPU write became visible.
    assert_eq!(unsafe { data.cast::<u32>().read() }, 0x4422);

    let mut gate = Gate::new(host);
    host.execute_title(&[record_copy(host, &out, 4, 0x9944)]);
    host.signal_title(fence, 10);
    let open = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(20));
        gate.open();
    });
    assert_eq!(wait(fence, 10, std::ptr::null_mut()), S_OK);
    open.join().unwrap();
    // SAFETY: the blocking title fence wait completed.
    assert_eq!(unsafe { data.cast::<u32>().read() }, 0x9944);
    // SAFETY: the event and emulated fence are ours.
    unsafe { CloseHandle(event) }.unwrap();
    crate::fence::let_go(fence);
}

#[test]
fn ambient_versions_stay_frozen_through_the_native_fence() {
    use crate::ambient::{Binding, Candidate, CheckedViews, Watch};
    use crate::descriptor::host::TextureView;
    let Some(host) = host_for_ring() else { return };
    if !crate::ambient::enabled() {
        return;
    }
    host.present_boundary();
    let (out, _) = host.mapped_buffer(8, true).unwrap();
    let list = record_copy(host, &out, 4, 73);
    let (_, _, count) = list.upload(4).unwrap();
    let watch = Watch::new();
    let views = CheckedViews {
        image: Binding {
            resource: out.clone(),
            view: TextureView::Uav(None),
            handle: D3D12_GPU_DESCRIPTOR_HANDLE::default(),
            watch: watch.clone(),
        },
        depth: None,
        size: [1, 1, 1],
    };
    list.deferred()
        .ambient
        .push(Candidate::new(&views, count as usize));
    let mut gate = Gate::new(host);
    assert!(host.execute_title(&[list]));
    // SAFETY: the recording owns the live count word while the gate is closed.
    assert_eq!(unsafe { count.cast::<u32>().read() }, 1);
    let (started, attempting) = std::sync::mpsc::sync_channel(0);
    let (done, rewritten) = std::sync::mpsc::sync_channel(1);
    let writer = std::thread::spawn(move || {
        started.send(()).unwrap();
        watch.invalidate();
        done.send(()).unwrap();
    });
    attempting.recv().unwrap();
    assert_eq!(
        rewritten.recv_timeout(Duration::from_millis(20)),
        Err(std::sync::mpsc::RecvTimeoutError::Timeout)
    );
    gate.open();
    rewritten.recv_timeout(Duration::from_secs(5)).unwrap();
    writer.join().unwrap();
    host.present_boundary();
}

#[cfg(feature = "diagnostics")]
#[test]
fn consecutive_uploads_keep_their_own_staging_contents() {
    let Some(host) = host_for_ring() else { return };
    host.present_boundary();
    let memory = crate::watch::tests::pool(1);
    let words = [
        3u32,
        0,
        0,
        0,
        128,
        0,
        128,
        1 | 1 << 16,
        28,
        1,
        0,
        0x100,
        0,
        0,
    ];
    let mut texture = std::ptr::null();
    assert_eq!(
        crate::resource::create_at_address(
            std::ptr::null(),
            memory as u64,
            words.as_ptr().cast(),
            0,
            std::ptr::null(),
            &crate::object::IUNKNOWN,
            &mut texture,
        ),
        S_OK
    );
    let copy = crate::resource::host_copy(texture).unwrap();
    let fill = |value: u8| {
        let _write = crate::watch::writing(memory, crate::watch::PAGE);
        // SAFETY: this test owns the title backing for the entire texture.
        unsafe { std::ptr::write_bytes(memory as *mut u8, value, crate::watch::PAGE) };
    };
    let record = |value| {
        fill(value);
        let list = host.new_list().unwrap();
        crate::texture::upload(&list, texture, &copy.texture);
        let b = list.move_texture(&copy, D3D12_RESOURCE_STATE_COPY_SOURCE.0 as u32);
        if let Some(b) = b {
            // SAFETY: an open list and a retained texture.
            unsafe { list.list.ResourceBarrier(&[b]) };
        }
        let (read, _) = crate::capture::read_back(&host.device, &list.list, &copy.texture).unwrap();
        list.keep(&read);
        list.close();
        (Arc::new(list), read)
    };

    // Complete the first fill before testing back-to-back refreshes with
    // SHOW_AFTER enabled.
    let (seed, _) = record(0x11);
    host.execute(&[&seed]);
    drop(seed);
    let mut gate = Gate::new(host);
    let (first, first_read) = record(0x22);
    assert!(host.execute_title(&[first]));
    let (second, second_read) = record(0x44);
    assert!(host.execute_title(&[second]));
    {
        let s = host.submitted.lock().unwrap();
        assert_eq!(s.flights.len(), 2);
        assert!(s.flights.iter().all(|f| f.upload.is_some()));
    }
    gate.open();
    host.present_boundary();
    for (buffer, value) in [(first_read, 0x22222222), (second_read, 0x44444444)] {
        let mut data = std::ptr::null_mut();
        // SAFETY: the queue finished writing into this live readback buffer.
        unsafe {
            buffer.Map(0, None, Some(&mut data)).unwrap();
            assert_eq!(data.cast::<u32>().read(), value);
            buffer.Unmap(0, None);
        }
    }
    crate::object::release_ref(texture);
    drop(copy);
    crate::watch::tests::free(memory);
}

#[test]
fn omitted_mark_recovers_after_all_old_work_and_reuses_the_ring() {
    let Some(host) = host_for_ring() else { return };
    host.present_boundary();
    let mut gate = Gate::new(host);
    let mut s = host.submitted.lock().unwrap();
    let _holding = Holding::enter();
    let old = host.queue();
    let (out, data) = host.mapped_buffer(12, true).unwrap();
    let mut pins = Vec::new();
    for i in 1..=2 {
        let list = record_copy(host, &out, i * 4, i as u32);
        if i == 1 {
            OMIT_NEXT_SUBMISSION_MARK.set(true);
        }
        host.execute_part(
            &[&list],
            &mut s,
            false,
            &mut pins,
            Some(std::slice::from_ref(&list)),
        );
    }
    assert_eq!(s.flights.len(), 2);
    MISS_NEXT_QUEUE_PROBE.set(true);
    gate.open();
    host.drain_inflight(&mut s, Boundary::Synchronous);
    assert_ne!(host.queue().as_raw(), old.as_raw());
    assert!(!s.busy());
    let list = record_copy(host, &out, 8, 73);
    host.execute_part(
        &[&list],
        &mut s,
        false,
        &mut pins,
        Some(std::slice::from_ref(&list)),
    );
    assert!(s.busy());
    host.drain_inflight(&mut s, Boundary::Synchronous);
    // SAFETY: the old and replacement queues finished writing.
    assert_eq!(unsafe { data.cast::<[u32; 3]>().read() }, [73, 1, 73]);
}
