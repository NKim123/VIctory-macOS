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

#[test]
fn clearing_forgotten_reads_preserves_uncollected_feedback() {
    let Some(host) = host_for_ring() else { return };
    host.present_boundary();
    let _one = host.submitted.lock().unwrap_or_else(|e| e.into_inner());
    let forgotten = victory_shader::emit::HEAP_VIEWS - 7;
    let unread = forgotten + 1;
    crate::shadow::tests::mark_read(false, forgotten);
    assert!(crate::shadow::take_reads().contains(&forgotten));
    crate::shadow::forget_read(forgotten);
    crate::shadow::tests::mark_read(true, unread);
    crate::shadow::clear_forgotten_reads();
    let next = crate::shadow::take_reads();
    assert!(!next.contains(&forgotten));
    assert!(next.contains(&unread));
    crate::shadow::tests::mark_read(false, forgotten);
    assert!(crate::shadow::take_reads().contains(&forgotten));
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
fn shared_uav_page_marks_retire_before_a_younger_flight() {
    let Some(host) = host_for_ring() else { return };
    if limit() < 2 {
        return;
    }
    use crate::watch::PAGE;
    host.present_boundary();
    let title = crate::watch::tests::pool(1);
    let tracker = crate::watch::track_copy(title, PAGE, true).expect("tracked title memory");
    let (resource, data) = host.shared_buffer(PAGE as u64).unwrap();
    // SAFETY: the mapped buffer and its native GPU address are live.
    let gpu = unsafe { resource.GetGPUVirtualAddress() };
    let alias = crate::shadow::Shadow::new_aliased(resource, data, gpu, title as u64, PAGE as u64);
    let (bit, _) = alias.mark_entry(title as u64).unwrap();
    let (root, pso) = crate::compute::pipeline(
        "cbuffer Params : register(b0) { uint address; uint bits; uint value; };
         RWByteAddressBuffer target : register(u0);
         RWByteAddressBuffer marks : register(u1);
         [numthreads(1, 1, 1)] void main(uint3 id : SV_DispatchThreadID) {
             target.Store(0, value);
             if (bits != 0) marks.Store(address, bits);
         }",
        windows::core::s!("test shared page feedback"),
        0,
        &[
            crate::compute::Param::Constants {
                register: 0,
                count: 3,
            },
            crate::compute::Param::Uav { register: 0 },
            crate::compute::Param::Uav { register: 1 },
        ],
    )
    .unwrap();
    let record = |memory, value, mark| {
        let list = host.new_list().unwrap();
        let constants = [4 * (bit / 32), u32::from(mark) << (bit % 32), value];
        let marks = crate::shadow::view_writes(memory).unwrap();
        list.note_view_writes();
        list.keep_shadow(&alias);
        // SAFETY: an open list and live resources owned through both fences.
        unsafe {
            list.list.SetComputeRootSignature(&root);
            list.list.SetPipelineState(&pso);
            list.list
                .SetComputeRoot32BitConstants(0, 3, constants.as_ptr().cast(), 0);
            list.list.SetComputeRootUnorderedAccessView(1, gpu);
            list.list.SetComputeRootUnorderedAccessView(2, marks);
            list.list.Dispatch(1, 1, 1);
        }
        list.close();
        Arc::new(list)
    };
    let reset_tracker = || {
        tracker.take(title, title + PAGE);
        assert!(tracker.clean());
    };
    reset_tracker();
    let before = FALLBACKS[Boundary::MarkedBack as usize].load(Ordering::Relaxed);
    let overlaps = OVERLAPPED.load(Ordering::Relaxed);
    for memory in [false, true] {
        let mut first = Gate::new(host);
        assert!(host.execute_title(&[record(memory, 11, true)]));
        let mut second = Gate::new(host);
        assert!(host.execute_title(&[record(memory, 22, true)]));
        first.open();
        {
            let mut submitted = host.submitted.lock().unwrap();
            let _holding = Holding::enter();
            let younger = submitted.flights.back().unwrap().value;
            host.wait_inflight(&mut submitted, younger - 1, Boundary::Fence);
            assert_eq!(submitted.flights.len(), 1);
            assert!(
                !tracker.clean(),
                "the first flight dirtied the page before the second ran"
            );
            // SAFETY: the first GPU write completed; the next is gated.
            assert_eq!(unsafe { data.cast::<u32>().read() }, 11);
        }
        reset_tracker();
        second.open();
        host.present_boundary();
        assert!(
            !tracker.clean(),
            "the same bit was marked again by the second flight"
        );
        // SAFETY: both GPU writes completed before the present boundary.
        assert_eq!(unsafe { data.cast::<u32>().read() }, 22);
        reset_tracker();
        host.execute_title(&[record(memory, 33, false)]);
        host.present_boundary();
        assert!(tracker.clean(), "readback must not report a stale mark");
    }
    assert_eq!(
        FALLBACKS[Boundary::MarkedBack as usize].load(Ordering::Relaxed),
        before
    );
    assert!(OVERLAPPED.load(Ordering::Relaxed) >= overlaps + 2);
    let widened = alias.widened(title as u64, 12, 1).unwrap();
    assert!(!alias.clean());
    let ordered = ORDERED[Boundary::MarkedBack as usize].load(Ordering::Relaxed);
    let mut gate = Gate::new(host);
    assert!(host.execute_title(&[record(false, 44, true)]));
    assert_eq!(widened.read_host(4)[0], 33);
    gate.open();
    host.present_boundary();
    assert_eq!(widened.read_host(4)[0], 44);
    assert_eq!(
        ORDERED[Boundary::MarkedBack as usize].load(Ordering::Relaxed),
        ordered + 1
    );
    assert_eq!(
        FALLBACKS[Boundary::MarkedBack as usize].load(Ordering::Relaxed),
        before
    );
    drop(widened);
    assert!(alias.clean(), "only live widened views require retirement");
    assert!(host.execute_title(&[record(false, 55, true)]));
    host.present_boundary();
    drop((tracker, alias));
    crate::watch::tests::free(title);
}

#[test]
fn retained_recording_storage_returns_after_the_fence() {
    let Some(host) = host_for_ring() else { return };
    let storage = Arc::new(());
    let weak = Arc::downgrade(&storage);
    let list = host.new_list().unwrap();
    let size = 128;
    list.keep_recording_storage(storage, size);
    assert!(list.inflight_bytes() >= size);
    list.close();
    let mut gate = Gate::new(host);
    assert!(host.execute_title(&[Arc::new(list)]));
    assert!(weak.upgrade().is_some());
    gate.open();
    host.present_boundary();
    assert!(weak.upgrade().is_none());
}

#[test]
fn copied_shadow_lands_before_the_bottom_label() {
    let Some(host) = host_for_ring() else { return };
    use crate::watch::PAGE;
    let title = crate::watch::tests::pool(1);
    let label = AtomicU32::new(0);
    crate::shadow::register(title, title as u64, PAGE as u64);
    let copy = crate::shadow::find(title as u64, PAGE as u64).unwrap();
    assert!(!copy.is_aliased());
    let list = host.new_list().unwrap();
    copy.refresh(&list, title as u64, 4);
    copy.gpu_writes(&list, title as u64, 4, false);
    let (upload, at, data) = list.upload(4).unwrap();
    // SAFETY: the upload range and the host shadow hold at least four bytes.
    unsafe {
        data.cast::<u32>().write(91);
        list.list
            .CopyBufferRegion(&copy.buffer, copy.offset(title as u64), &upload, at, 4);
    }
    list.defer_write(label.as_ptr().cast(), crate::queue::Written::Value(1), 4);
    list.close();

    let mut gate = Gate::new(host);
    let before = ORDERED[Boundary::ShadowBack as usize].load(Ordering::Relaxed);
    assert!(host.execute_title(&[Arc::new(list)]));
    assert_eq!(label.load(Ordering::Acquire), 0);
    gate.open();
    host.present_boundary();
    assert_eq!(label.load(Ordering::Acquire), 1);
    // SAFETY: the registered pool is live and the copy-back has finished.
    assert_eq!(unsafe { (title as *const u32).read() }, 91);
    assert_eq!(
        ORDERED[Boundary::ShadowBack as usize].load(Ordering::Relaxed),
        before + 1
    );
    drop(copy);
    crate::shadow::unregister(title, title as u64);
    crate::watch::tests::free(title);
}

#[test]
fn texture_fill_is_published_only_after_its_fence() {
    let Some(host) = host_for_ring() else { return };
    if !crate::descriptor::host::show_after() {
        return;
    }
    let mut texture = None;
    // SAFETY: the description is a valid tiny color texture and output is ours.
    unsafe {
        host.device.CreateCommittedResource(
            &D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_DEFAULT,
                ..Default::default()
            },
            D3D12_HEAP_FLAG_NONE,
            &D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                Width: 4,
                Height: 4,
                DepthOrArraySize: 1,
                MipLevels: 1,
                Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                ..Default::default()
            },
            D3D12_RESOURCE_STATE_COMMON,
            None,
            &mut texture,
        )
    }
    .unwrap();
    let copy = Arc::new(crate::resource::HostTexture::new(
        texture.unwrap(),
        0,
        Vec::new(),
    ));
    let list = host.new_list().unwrap();
    list.defer_texture_fill(crate::texture::Fill::Upload(copy.clone()));
    list.close();
    let before = ORDERED[Boundary::Descriptors as usize].load(Ordering::Relaxed);
    let mut gate = Gate::new(host);
    assert!(host.execute_title(&[Arc::new(list)]));
    assert!(!copy.initialized.load(Ordering::Acquire));
    gate.open();
    host.present_boundary();
    assert!(copy.initialized.load(Ordering::Acquire));
    assert_eq!(
        ORDERED[Boundary::Descriptors as usize].load(Ordering::Relaxed),
        before + 1
    );
}

#[test]
fn additive_mapping_can_proceed_while_a_destructive_edit_waits() {
    let Some(host) = host_for_ring() else { return };
    host.present_boundary();
    let (out, data) = host.mapped_buffer(8, true).unwrap();
    let before = DRAINS[Boundary::Mapping as usize].load(Ordering::Relaxed);
    let mut gate = Gate::new(host);
    assert!(host.execute_title(&[record_copy(host, &out, 4, 0x1234)]));
    crate::tiled::with_physical_mapping(
        || false,
        || {
            assert_eq!(
                DRAINS[Boundary::Mapping as usize].load(Ordering::Relaxed),
                before
            );
        },
    );
    assert!(host.submitted.lock().unwrap().busy());
    let release = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(20));
        gate.open();
    });
    crate::tiled::with_physical_mapping(
        || true,
        || {
            // SAFETY: the destructive boundary has fenced the GPU write.
            assert_eq!(unsafe { data.cast::<u32>().add(1).read() }, 0x1234);
        },
    );
    release.join().unwrap();
    assert_eq!(
        DRAINS[Boundary::Mapping as usize].load(Ordering::Relaxed),
        before + 1
    );
}

#[test]
fn computed_writes_to_tiles_retire_with_their_captured_layout() {
    computed_writes_to_tiles(false);
}

#[test]
fn computed_writes_to_tiles_retire_with_a_mixed_layout() {
    computed_writes_to_tiles(true);
}

fn computed_writes_to_tiles(mixed: bool) {
    let Some(host) = host_for_ring() else { return };
    if !crate::switch::on("VICTORY_PHYSICAL_TILES") {
        return;
    }
    assert!(
        crate::alias::physical_tiles_enabled(),
        "physical tile support"
    );
    use crate::compute::Param;
    use crate::watch::PAGE;
    host.present_boundary();
    let ordinary = mixed.then(|| {
        let title = crate::watch::tests::pool(1);
        crate::shadow::register(title, title as u64, PAGE as u64);
        let copy = crate::shadow::find(title as u64, PAGE as u64).unwrap();
        assert!(!copy.is_aliased());
        (title, copy)
    });
    let title = crate::watch::tests::pool(1);
    let heap = Arc::new(crate::alias::physical_heap(PAGE).expect("CPU-visible physical tiles"));
    let tiled = crate::tiled::TiledAlias::new(title, PAGE).expect("a tiled title reservation");
    tiled.map(title, PAGE, Some((&heap, 0)));
    let copy = crate::shadow::find(title as u64, PAGE as u64).unwrap();
    assert!(copy.is_tiled());
    let tracker = crate::watch::track_copy(title, PAGE, true).unwrap();
    tracker.take(title, title + PAGE);
    assert!(tracker.clean());
    let (root, pso) = crate::compute::pipeline(
        &format!(
            "StructuredBuffer<uint> table : register(t0);
             RWByteAddressBuffer data : register(u0);
             RWByteAddressBuffer dirty : register(u1);
             [numthreads(1, 1, 1)] void main() {{
                 uint entry = (table[{}] << 12) | {};
                 data.Store(0, 0x55667788);
                 dirty.Store((entry >> 5) * 4, 1u << (entry & 31));
             }}",
            title >> 28,
            (title >> 16) & 0xfff,
        ),
        windows::core::s!("tiled computed feedback"),
        0,
        &[
            Param::Srv {
                register: 0,
                space: 0,
            },
            Param::Uav { register: 0 },
            Param::Uav { register: 1 },
        ],
    )
    .unwrap();
    let list = host.new_list().unwrap();
    let map = crate::memory::bind(&list, crate::memory::OURS).unwrap();
    // SAFETY: a fresh recording with a live page table, tile VA and bitmap.
    unsafe {
        list.list.SetComputeRootSignature(&root);
        list.list.SetPipelineState(&pso);
        list.list.SetComputeRootShaderResourceView(0, map.table);
        list.list
            .SetComputeRootUnorderedAccessView(1, copy.gpu(title as u64));
        list.list.SetComputeRootUnorderedAccessView(2, map.dirty);
        list.list.Dispatch(1, 1, 1);
    }
    list.close();
    let before = QUEUED_COMPUTED.load(Ordering::Relaxed);
    let ordered = ORDERED[Boundary::ComputedBack as usize].load(Ordering::Relaxed);
    let mut gate = Gate::new(host);
    assert!(host.execute_title(&[Arc::new(list)]));
    assert!(tracker.clean());
    gate.open();
    host.present_boundary();
    assert_eq!(QUEUED_COMPUTED.load(Ordering::Relaxed), before + 1);
    assert_eq!(
        ORDERED[Boundary::ComputedBack as usize].load(Ordering::Relaxed),
        ordered + u64::from(mixed)
    );
    assert!(
        !tracker.clean(),
        "computed tile marks notified the title tracker"
    );
    // SAFETY: the native fence completed; this is the physical tile backing.
    let physical_value = unsafe { (heap.cpu_address() as *const u32).read() };
    assert_eq!(physical_value, 0x55667788);
    drop((tiled, copy, tracker));
    crate::watch::tests::free(title);
    if let Some((address, copy)) = ordinary {
        drop(copy);
        crate::shadow::unregister(address, address as u64);
        crate::watch::tests::free(address);
    }
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
fn bottom_of_pipe_labels_and_title_signals_retire_without_another_submit() {
    let Some(host) = host_for_ring() else { return };
    host.present_boundary();
    let (out, _) = host.mapped_buffer(8, true).unwrap();
    let first = AtomicU64::new(0);
    let second = AtomicU32::new(0);
    let mut fence = std::ptr::null();
    assert_eq!(
        crate::fence::create(
            std::ptr::null(),
            0,
            0,
            &crate::fence::FENCE_IIDS[0],
            &mut fence
        ),
        S_OK
    );
    let raw = slot::<extern "system" fn(This) -> u64>(fence, 0x60)(fence);
    // SAFETY: the emulated fence exposes its value for its entire lifetime.
    let raw = unsafe { &*(raw as *const AtomicU64) };
    let prior_overlaps = OVERLAPPED.load(Ordering::Relaxed);
    let prior_fallbacks = FALLBACKS[Boundary::BottomOfPipe as usize].load(Ordering::Relaxed);
    let mut gate = Gate::new(host);
    let list = record_copy(host, &out, 4, 1);
    list.defer_write(
        first.as_ptr().cast(),
        crate::queue::Written::Value(0xfedc_ba98_7654_3210),
        8,
    );
    assert!(host.execute_title(&[list]));
    host.signal_title(fence, 1);
    let mut second_gate = Gate::new(host);
    let list = record_copy(host, &out, 4, 2);
    list.defer_write(
        second.as_ptr().cast(),
        crate::queue::Written::Value(0xfedc_ba98),
        4,
    );
    assert!(host.execute_title(&[list]));
    host.signal_title(fence, 2);
    assert!(OVERLAPPED.load(Ordering::Relaxed) > prior_overlaps);
    assert_eq!(
        FALLBACKS[Boundary::BottomOfPipe as usize].load(Ordering::Relaxed),
        prior_fallbacks
    );
    assert_eq!(first.load(Ordering::Acquire), 0);
    assert_eq!(raw.load(Ordering::Acquire), 0);

    let spin_for = |value| {
        let until = Instant::now() + Duration::from_secs(3);
        while raw.load(Ordering::Acquire) != value {
            assert!(
                Instant::now() < until,
                "the completion worker must publish flight {value}"
            );
            std::hint::spin_loop();
        }
    };
    gate.open();
    spin_for(1);
    assert_eq!(first.load(Ordering::Acquire), 0xfedc_ba98_7654_3210);
    assert_eq!(second.load(Ordering::Acquire), 0);
    second_gate.open();
    spin_for(2);
    assert_eq!(second.load(Ordering::Acquire), 0xfedc_ba98);
    crate::fence::let_go(fence);
}

#[test]
fn fifo_queue_waits_use_the_latest_label_before_the_cpu_can_read_it() {
    let Some(host) = host_for_ring() else { return };
    host.present_boundary();
    let (out, data) = host.mapped_buffer(16, true).unwrap();
    // SAFETY: a new readback allocation, with no GPU work using it.
    unsafe { data.write_bytes(0, 16) };
    let label = AtomicU32::new(4);
    let mut gate = Gate::new(host);
    for value in [0x104, 0x105] {
        let producer = record_copy(host, &out, 4, 1);
        producer.defer_write(
            label.as_ptr().cast(),
            crate::queue::Written::Value(value),
            4,
        );
        crate::queue::submit_test_segments(0, [producer]);
    }
    let waiter = |offset, reference| {
        let list = record_copy(host, &out, offset, 73);
        list.defer_wait(crate::hold::ValueWait {
            address: label.as_ptr() as u64,
            bytes: 4,
            comparison: 3,
            reference,
            mask: 0xff,
        });
        list
    };
    // Queue 2 must wait, even though the current CPU value and the older
    // submitted write both satisfy it. Queue 3 can run after the newer write.
    crate::queue::submit_test_segments(2, [waiter(8, 4)]);
    crate::queue::submit_test_segments(3, [waiter(12, 5)]);
    assert_eq!(label.load(Ordering::Acquire), 4);
    assert_eq!(host.submitted.lock().unwrap().flights.len(), 3);
    gate.open();
    host.present_boundary();
    assert_eq!(label.load(Ordering::Acquire), 0x105);
    // SAFETY: all submitted GPU work has finished.
    assert_eq!(unsafe { data.cast::<[u32; 4]>().read() }, [73, 1, 0, 73]);
    label.store(4, Ordering::Release);
    crate::queue::pump_held(crate::queue::Why::Fence);
    host.present_boundary();
    // SAFETY: the CPU write released the wait and the GPU work has finished.
    assert_eq!(unsafe { data.cast::<u32>().add(2).read() }, 73);
}

#[test]
fn fifo_queue_waits_see_earlier_segments_in_the_same_piece() {
    let Some(host) = host_for_ring() else { return };
    host.present_boundary();
    let (out, data) = host.mapped_buffer(12, true).unwrap();
    // SAFETY: a new readback allocation, with no GPU work using it.
    unsafe { data.write_bytes(0, 12) };
    let label = AtomicU64::new(0);
    let mut gate = Gate::new(host);
    let producer = |value| {
        let list = record_copy(host, &out, 4, 1);
        list.defer_write(
            label.as_ptr().cast(),
            crate::queue::Written::Value(value),
            8,
        );
        list
    };
    crate::queue::submit_test_segments(0, [producer(7)]);
    let waiter = record_copy(host, &out, 8, 73);
    waiter.defer_wait(crate::hold::ValueWait {
        address: label.as_ptr() as u64,
        bytes: 8,
        comparison: 3,
        reference: 7,
        mask: u64::MAX,
    });
    crate::queue::submit_test_segments(2, [producer(8), waiter]);
    assert_eq!(host.submitted.lock().unwrap().flights.len(), 2);
    gate.open();
    host.present_boundary();
    assert_eq!(label.load(Ordering::Acquire), 8);
    // SAFETY: the wait is still held and the other GPU work has finished.
    assert_eq!(unsafe { data.cast::<u32>().add(2).read() }, 0);
    label.store(7, Ordering::Release);
    crate::queue::pump_held(crate::queue::Why::Fence);
    host.present_boundary();
    // SAFETY: the CPU write released the wait and the GPU work has finished.
    assert_eq!(unsafe { data.cast::<u32>().add(2).read() }, 73);
}

#[test]
fn a_deferred_upload_reads_the_earlier_label() {
    let Some(host) = host_for_ring() else { return };
    host.present_boundary();
    let (out, data) = host.mapped_buffer(8, true).unwrap();
    let label = AtomicU32::new(0);
    let gate = Gate::new(host);
    let first = record_copy(host, &out, 4, 1);
    first.defer_write(label.as_ptr().cast(), crate::queue::Written::Value(73), 4);
    assert!(host.execute_title(&[first]));

    let next = host.new_list().unwrap();
    let (upload, at, dst) = next.upload(4).unwrap();
    next.defer_fill(dst, label.as_ptr().cast(), 4);
    // SAFETY: the closed recording owns the upload; the test owns the readback.
    unsafe { next.list.CopyBufferRegion(&out, 4, &upload, at, 4) };
    next.close();
    let release = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(20));
        drop(gate);
    });
    host.execute_title(&[Arc::new(next)]);
    release.join().unwrap();
    host.present_boundary();
    // SAFETY: the native copy and its flight have finished.
    assert_eq!(unsafe { data.cast::<u32>().add(1).read() }, 73);
}

#[test]
fn draining_a_newer_flight_publishes_the_earlier_label() {
    let Some(host) = host_for_ring() else { return };
    host.present_boundary();
    let (out, _) = host.mapped_buffer(8, true).unwrap();
    let label = AtomicU32::new(0);
    let mut first_gate = Gate::new(host);
    let first = record_copy(host, &out, 4, 1);
    first.defer_write(label.as_ptr().cast(), crate::queue::Written::Value(17), 4);
    assert!(host.execute_title(&[first]));
    let mut second_gate = Gate::new(host);
    assert!(host.execute_title(&[record_copy(host, &out, 4, 2)]));

    let counter = &DRAINS[Boundary::Present as usize];
    let before = counter.load(Ordering::Relaxed);
    std::thread::scope(|scope| {
        let drain = scope.spawn(|| host.present_boundary());
        let until = Instant::now() + Duration::from_secs(3);
        while counter.load(Ordering::Relaxed) == before {
            assert!(
                Instant::now() < until,
                "the concurrent drain must enter its fence wait"
            );
            std::hint::spin_loop();
        }
        first_gate.open();
        while label.load(Ordering::Acquire) == 0 {
            assert!(
                Instant::now() < until,
                "the first label must land before the second fence"
            );
            std::hint::spin_loop();
        }
        assert_eq!(label.load(Ordering::Acquire), 17);
        second_gate.open();
        drain.join().unwrap();
    });
}

#[test]
fn a_native_copy_sees_an_earlier_label_through_a_zero_copy_buffer() {
    let Some(host) = host_for_ring() else { return };
    host.present_boundary();
    let (source, src) = host.shared_buffer(64 << 10).unwrap();
    let (out, dst) = host.mapped_buffer(8, true).unwrap();
    // SAFETY: an idle mapped resource and the address of that resource.
    let gpu = unsafe {
        src.cast::<u32>().write(0);
        source.GetGPUVirtualAddress()
    };
    let shadow = crate::shadow::Shadow::new_aliased(source.clone(), src, gpu, src as u64, 64 << 10);
    crate::shadow::register_aliased(shadow.clone());

    let gate = Gate::new(host);
    let first = record_copy(host, &out, 4, 1);
    first.defer_gpu_write(&shadow, src as usize, 4, false);
    first.defer_write(src, crate::queue::Written::Value(91), 4);
    assert_eq!(first.inflight_barrier(), None);
    assert!(host.execute_title(&[first]));
    let next = host.new_list().unwrap();
    // SAFETY: both buffers are retained through the submission's fence.
    unsafe { next.list.CopyBufferRegion(&out, 4, &source, 0, 4) };
    next.close();
    let release = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(20));
        drop(gate);
    });
    host.execute_title(&[Arc::new(next)]);
    release.join().unwrap();
    host.present_boundary();
    crate::shadow::unregister_aliased(src as u64);
    // SAFETY: the GPU has completed the readback.
    assert_eq!(unsafe { dst.cast::<u32>().add(1).read() }, 91);
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
            Boundary::Synchronous,
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
        Boundary::Synchronous,
    );
    assert!(s.busy());
    host.drain_inflight(&mut s, Boundary::Synchronous);
    // SAFETY: the old and replacement queues finished writing.
    assert_eq!(unsafe { data.cast::<[u32; 3]>().read() }, [73, 1, 73]);
}
