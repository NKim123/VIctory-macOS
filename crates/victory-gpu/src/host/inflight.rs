//! Fence-ordered ownership for title submissions. Only the submission lock
//! holder can move a flight or reuse its recording and completion marks.

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{SyncSender, TrySendError};

use super::*;

const MAX_FLIGHT_LISTS: usize = 128;
const MAX_FLIGHT_BYTES: u64 = 128 << 20;

pub(super) fn limit() -> usize {
    static LIMIT: OnceLock<usize> = OnceLock::new();
    *LIMIT.get_or_init(|| {
        let value = std::env::var("VICTORY_INFLIGHT").unwrap_or_default();
        parse_limit(&value)
    })
}

fn parse_limit(value: &str) -> usize {
    match value.trim().parse::<usize>() {
        Ok(1) => 4,
        Ok(n) => n.min(8),
        Err(_) => 0,
    }
}

#[derive(Default)]
pub(super) struct Submission {
    pub(super) value: u64,
    flights: VecDeque<InFlight>,
    marks: Vec<Vec<Mark>>,
    upload_list: Option<Box<HostList>>,
    bytes: u64,
    lists: usize,
}

pub(super) struct InFlight {
    value: u64,
    marks: Vec<Mark>,
    active_marks: usize,
    native_lists: Vec<Option<ID3D12CommandList>>,
    lists: Vec<Arc<HostList>>,
    upload: Option<crate::bindless::OwnedUpload>,
    _tiles: crate::tiled::Flushed,
    _ambient: Option<Arc<crate::ambient::Frozen>>,
    _batch: Batch,
    _pins: Vec<crate::descriptor::ViewPin>,
    _retired: Vec<crate::descriptor::host::Held>,
    bytes: u64,
    writes: Vec<BottomWrite>,
    shared_writes: bool,
    signals: Vec<TitleSignal>,
}

#[derive(Clone, Copy)]
pub(super) struct BottomWrite {
    pub(super) address: usize,
    pub(super) value: crate::queue::Written,
    pub(super) bytes: usize,
}

impl BottomWrite {
    pub(super) fn publish(self) {
        self.value.write(self.address as *mut u8, self.bytes);
    }

    pub(super) fn overlaps(&self, start: usize, len: usize) -> bool {
        len != 0
            && self.address < start.saturating_add(len)
            && start < self.address.saturating_add(self.bytes)
    }

    fn may_alias(&self) -> bool {
        crate::shadow::has_aliased_overlap(
            self.address as u64,
            self.address.saturating_add(self.bytes) as u64,
        ) || crate::watch::aliased(self.address, self.bytes)
    }
}

struct TitleSignal {
    fence: usize,
    value: u64,
}

impl TitleSignal {
    fn publish(self) {
        crate::fence::set(self.fence as crate::object::This, self.value);
        crate::fence::let_go(self.fence as crate::object::This);
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Boundary {
    Capacity,
    ReusedList,
    TextureBack,
    ComputedBack,
    MarkedBack,
    ShadowBack,
    BottomOfPipe,
    CpuAccess,
    Descriptors,
    Quiet,
    Present,
    Mapping,
    Synchronous,
    Diagnostics,
    Recovery,
    Fence,
}

static QUEUED: AtomicU64 = AtomicU64::new(0);
static OVERLAPPED: AtomicU64 = AtomicU64::new(0);
static FALLBACKS: [AtomicU64; Boundary::Fence as usize + 1] =
    [const { AtomicU64::new(0) }; Boundary::Fence as usize + 1];
static DRAINS: [AtomicU64; Boundary::Fence as usize + 1] =
    [const { AtomicU64::new(0) }; Boundary::Fence as usize + 1];

pub(super) fn log(frames: f64) {
    if limit() == 0 {
        return;
    }
    let count = |why: Boundary| DRAINS[why as usize].swap(0, Ordering::Relaxed);
    let sync = |why: Boundary| FALLBACKS[why as usize].swap(0, Ordering::Relaxed);
    let back = [
        Boundary::TextureBack,
        Boundary::ComputedBack,
        Boundary::MarkedBack,
        Boundary::ShadowBack,
    ];
    let [texture_back, computed_back, marked_back, shadow_back] = back.map(sync);
    tracing::info!(
        frames,
        queued = QUEUED.swap(0, Ordering::Relaxed),
        overlapped = OVERLAPPED.swap(0, Ordering::Relaxed),
        synchronous_copy_back = texture_back + computed_back + marked_back + shadow_back,
        synchronous_texture_back = texture_back,
        synchronous_computed_back = computed_back,
        synchronous_marked_back = marked_back,
        synchronous_shadow_back = shadow_back,
        synchronous_bottom_of_pipe = sync(Boundary::BottomOfPipe),
        synchronous_descriptors = sync(Boundary::Descriptors),
        synchronous_diagnostics = sync(Boundary::Diagnostics),
        synchronous_replay = sync(Boundary::ReusedList),
        synchronous_present = sync(Boundary::Present),
        synchronous_other = sync(Boundary::Synchronous),
        capacity = count(Boundary::Capacity),
        reused_list = count(Boundary::ReusedList),
        copy_back = back.into_iter().map(count).sum::<u64>(),
        bottom_of_pipe = count(Boundary::BottomOfPipe),
        diagnostics = count(Boundary::Diagnostics),
        cpu_access = count(Boundary::CpuAccess),
        descriptors = count(Boundary::Descriptors),
        quiet = count(Boundary::Quiet),
        present = count(Boundary::Present),
        mapping = count(Boundary::Mapping),
        synchronous_boundary = count(Boundary::Synchronous),
        recovery = count(Boundary::Recovery),
        fence = count(Boundary::Fence),
        "inflight GPU submissions"
    );
}

impl Submission {
    pub(super) fn busy(&self) -> bool {
        !self.flights.is_empty()
    }

    pub(super) fn has_bottom_writes(&self) -> bool {
        self.flights.iter().any(|f| !f.writes.is_empty())
    }

    pub(super) fn uses(&self, list: &HostList) -> bool {
        self.flights
            .iter()
            .any(|f| f.lists.iter().any(|l| std::ptr::eq(&**l, list)))
    }

    pub(super) fn upload_list(&mut self, host: &Host) -> Option<Box<HostList>> {
        if let Some(list) = self.upload_list.take() {
            return Some(list);
        }
        let list = host.new_list().ok()?;
        list.set_ours();
        Some(Box::new(list))
    }

    pub(super) fn recycle_upload(&mut self, list: Box<HostList>) {
        // Upload spikes can take a lot of scratch. Keep at most one idle
        // list and only small recordings after their fence.
        if self.upload_list.is_none() && list.inflight_bytes() <= 8 << 20 {
            list.reset();
            self.upload_list = Some(list);
        }
    }
}

impl Host {
    pub(super) fn drain_inflight(&self, s: &mut Submission, why: Boundary) {
        self.retire_inflight(s);
        if let Some(last) = s.flights.back() {
            self.wait_inflight(s, last.value, why);
        }
    }

    /// A CPU touch may need the old label value to have landed first.
    /// Unknown ranges include indirect reads such as computed addresses.
    pub(super) fn wait_bottom_writes(
        &self,
        s: &mut Submission,
        touches: impl Fn(&BottomWrite, bool) -> bool,
    ) -> bool {
        if let Some(flight) = s
            .flights
            .iter()
            .rev()
            .find(|f| f.writes.iter().any(|write| touches(write, f.shared_writes)))
        {
            self.wait_inflight(s, flight.value, Boundary::BottomOfPipe);
            return true;
        }
        false
    }

    /// A native GPU list can access a CPU label through a directly mapped
    /// resource. Its CPU preparation can overlap, but queue it only once
    /// those stores have been published. A raw GPU wait on the CPU label
    /// here would prevent synchronous callers from retiring the ring.
    pub(super) fn order_shared_bottom_writes(&self, s: &mut Submission) {
        if let Some(flight) = s.flights.iter().rev().find(|f| f.shared_writes) {
            self.wait_inflight(s, flight.value, Boundary::BottomOfPipe);
        }
    }

    /// An emulated queue wait must not compare a stale label that still
    /// satisfies its predicate before the previous flight overwrites it.
    pub(crate) fn order_bottom_read(&self, address: usize, bytes: usize) {
        if limit() == 0 || HOLDS_SUBMISSIONS.with(std::cell::Cell::get) {
            return;
        }
        let mut s = self.submitted.lock().unwrap_or_else(|e| e.into_inner());
        let _holding = Holding::enter();
        self.wait_bottom_writes(&mut s, |write, may_alias| {
            may_alias || write.overlaps(address, bytes)
        });
    }

    fn wait_inflight(&self, s: &mut Submission, value: u64, why: Boundary) {
        // SAFETY: the queue has signalled this fence value.
        let completed = unsafe { self.fence.GetCompletedValue() };
        if completed == u64::MAX {
            self.flight_device_removed(s);
        }
        if completed < value {
            DRAINS[why as usize].fetch_add(1, Ordering::Relaxed);
            let started = Instant::now();
            if !self.wait_for_submission(value) {
                self.flight_device_removed(s);
            }
            SUBMISSION_US[1].fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
        }
        self.retire_inflight(s);
    }

    fn flight_device_removed(&self, s: &Submission) -> ! {
        let lists: Vec<_> = s
            .flights
            .iter()
            .flat_map(|f| f.lists.iter().map(AsRef::as_ref))
            .collect();
        fail(format_args!(
            "host GPU device removed during an in-flight submission{}{}",
            self.removal_data(&lists),
            last_notes(&lists)
        ))
    }

    /// Observe completed flights in FIFO order. A missed mark forces the
    /// whole old queue idle before recovery touches the global atomic locks,
    /// rearms a mark, or replaces the native queue.
    pub(super) fn retire_inflight(&self, s: &mut Submission) -> bool {
        if !s.busy() {
            return false;
        }
        // SAFETY: a live fence used by the direct queue.
        let mut completed = unsafe { self.fence.GetCompletedValue() };
        if completed == u64::MAX {
            self.flight_device_removed(s);
        }
        let mut recovery_mark = None;
        let mut completions = Vec::new();
        while s.flights.front().is_some_and(|f| f.value <= completed) {
            let f = s.flights.pop_front().expect("completed flight");
            let ran = f.marks[..f.active_marks]
                .iter()
                .all(|mark| submission_ran(mark, f.value));
            if !ran && recovery_mark.is_none() {
                DRAINS[Boundary::Recovery as usize].fetch_add(1, Ordering::Relaxed);
                if let Some(last) = s.flights.back()
                    && last.value > completed
                {
                    let started = Instant::now();
                    if !self.wait_for_submission(last.value) {
                        self.flight_device_removed(s);
                    }
                    SUBMISSION_US[1]
                        .fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
                    completed = last.value;
                }
                recovery_mark = Some(f.marks);
            } else {
                s.marks.push(f.marks);
            }
            s.bytes -= f.bytes;
            s.lists -= f.lists.len();
            #[cfg(feature = "diagnostics")]
            if ran {
                crate::resource::life::note(0, crate::resource::life::RAN, f.value, 0);
            }
            drop(f.native_lists);
            if let Some(upload) = f.upload {
                if !ran {
                    for staging in upload.sources {
                        staging.retry_upload();
                    }
                }
                s.recycle_upload(upload.list);
            }
            for list in f.lists {
                list.notify_aliased_writes();
                crate::list::recycle(list);
            }
            // Last releases of title resources can acquire other runtime
            // locks. The work using these references has completed.
            if !f._pins.is_empty() || !f._retired.is_empty() {
                release_later(Box::new((f._pins, f._retired)));
            }
            if !f.writes.is_empty() || !f.signals.is_empty() {
                completions.push((f.writes, f.signals));
            }
            crate::memory::note_submission(false);
        }
        if let Some(marks) = recovery_mark {
            self.lost(&mut s.value, &marks[0]);
            s.marks.push(marks);
        }
        if !s.busy() {
            // Flights cannot produce copy-back through this shared bitmap.
            // Consume read feedback and clear the alias marks only while
            // there is no writer, independently of which flight produced it.
            crate::shadow::copy_back_aliased_marks();
            log_gave_up();
            let read = crate::shadow::take_reads();
            if !read.is_empty() {
                crate::descriptor::make_read(&read);
            }
            release_retired();
            self.log_messages();
        }
        let published = !completions.is_empty();
        for (writes, signals) in completions {
            writes.into_iter().for_each(BottomWrite::publish);
            signals.into_iter().for_each(TitleSignal::publish);
        }
        published
    }

    /// When budget is exhausted, retire only enough of the oldest work to
    /// admit this submission. A completed fence never blocks the caller.
    pub(super) fn make_flight_room(&self, s: &mut Submission, lists: usize, bytes: u64) -> bool {
        if lists > MAX_FLIGHT_LISTS || bytes > MAX_FLIGHT_BYTES {
            return false;
        }
        self.retire_inflight(s);
        while s.flights.len() >= limit()
            || s.lists + lists > MAX_FLIGHT_LISTS
            || s.bytes + bytes > MAX_FLIGHT_BYTES
        {
            let Some(first) = s.flights.front() else {
                return false;
            };
            self.wait_inflight(s, first.value, Boundary::Capacity);
        }
        true
    }

    pub(super) fn flight_marks(&self, s: &mut Submission, count: usize) -> Option<Vec<Mark>> {
        let mut marks = s.marks.pop().unwrap_or_default();
        while marks.len() < count {
            match Mark::new(self) {
                Ok(mark) => marks.push(mark),
                Err(error) => {
                    tracing::warn!(%error, "could not allocate in-flight submission marks");
                    s.marks.push(marks);
                    return None;
                }
            }
        }
        Some(marks)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn enqueue_flight(
        &self,
        s: &mut Submission,
        marks: Vec<Mark>,
        native_lists: Vec<Option<ID3D12CommandList>>,
        boundaries: &[usize],
        lists: &[Arc<HostList>],
        upload: Option<crate::bindless::OwnedUpload>,
        batch: Batch,
        pins: Vec<crate::descriptor::ViewPin>,
        bytes: u64,
    ) {
        let started = Instant::now();
        let mut writes = Vec::new();
        for list in lists {
            writes.extend_from_slice(&list.deferred().writes);
        }
        let shared_writes = writes.iter().any(BottomWrite::may_alias);
        s.value += 1;
        let value = s.value;
        let queue = self.queue();
        let tiles = crate::tiled::flush(&queue);
        let ambient = if crate::ambient::enabled() {
            let candidates: Vec<_> = lists
                .iter()
                .flat_map(|list| list.deferred().ambient.clone())
                .collect();
            crate::ambient::freeze(&candidates, &self.fence, value)
        } else {
            None
        };
        let active_marks = boundaries.len() + 1;
        let mut marked = Vec::with_capacity(native_lists.len() + active_marks);
        let mut first = 0;
        for (&end, mark) in boundaries.iter().chain([&native_lists.len()]).zip(&marks) {
            mark.arm(value);
            let here = marked.len();
            marked.extend_from_slice(&native_lists[first..end]);
            #[cfg(test)]
            let omit = OMIT_NEXT_SUBMISSION_MARK.replace(false);
            #[cfg(not(test))]
            let omit = false;
            if !omit {
                marked.push(mark.list());
            }
            // SAFETY: all COM lists, allocators, and referenced objects
            // remain owned by this call and then by the flight.
            unsafe { queue.ExecuteCommandLists(&marked[here..]) };
            first = end;
        }
        #[cfg(feature = "diagnostics")]
        crate::resource::life::note(0, crate::resource::life::SUBMITTED, value, 0);
        s.lists += lists.len();
        s.bytes += bytes;
        s.flights.push_back(InFlight {
            value,
            marks,
            active_marks,
            native_lists: marked,
            lists: lists.to_vec(),
            upload,
            _tiles: tiles,
            _ambient: ambient,
            _batch: batch,
            _pins: pins,
            _retired: take_retired(),
            bytes,
            writes,
            shared_writes,
            signals: Vec::new(),
        });
        if !self.signal_submission(&queue, value) {
            self.flight_device_removed(s);
        }
        QUEUED.fetch_add(1, Ordering::Relaxed);
        SUBMISSION_US[1].fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
    }

    pub(super) fn note_sync(&self, s: &mut Submission, why: Boundary) {
        if limit() != 0 {
            FALLBACKS[why as usize].fetch_add(1, Ordering::Relaxed);
            self.drain_inflight(s, why);
        }
    }

    pub(super) fn note_overlap(&self, s: &Submission) {
        if let Some(last) = s.flights.back() {
            // SAFETY: a live fence. An earlier GPU submission was still
            // pending when this caller acquired the lock to prepare work.
            if unsafe { self.fence.GetCompletedValue() } < last.value {
                OVERLAPPED.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// Publish the emulated fence, including its raw title-visible address,
    /// only after the native fence that precedes it. The reaper does this
    /// even if the title waits on a Win32 event with no further D3D12 calls.
    pub(crate) fn signal_title(&'static self, fence: crate::object::This, value: u64) {
        let mut s = self.submitted.lock().unwrap_or_else(|e| e.into_inner());
        let _holding = Holding::enter();
        self.retire_inflight(&mut s);
        if s.flights.back().is_some_and(|f| f.signals.len() >= 64) {
            self.drain_inflight(&mut s, Boundary::Fence);
        }
        if let Some(last) = s.flights.back_mut() {
            crate::fence::keep(fence);
            last.signals.push(TitleSignal {
                fence: fence as usize,
                value,
            });
            self.wake_reaper();
        } else {
            crate::fence::set(fence, value);
        }
    }

    pub(crate) fn reap_title_fences(&self) {
        if let Ok(mut s) = self.submitted.try_lock() {
            let _holding = Holding::enter();
            self.retire_inflight(&mut s);
        }
    }

    pub(crate) fn wait_title(&self, fence: crate::object::This, value: u64) -> bool {
        if limit() == 0 || HOLDS_SUBMISSIONS.with(std::cell::Cell::get) {
            return false;
        }
        let mut s = self.submitted.lock().unwrap_or_else(|e| e.into_inner());
        let _holding = Holding::enter();
        if let Some(f) = s.flights.iter().find(|f| {
            f.signals
                .iter()
                .any(|signal| signal.fence == fence as usize && signal.value >= value)
        }) {
            let pending = f.value;
            self.wait_inflight(&mut s, pending, Boundary::Fence);
            return true;
        }
        false
    }

    pub(super) fn wake_reaper(&'static self) {
        let sender = self.reaper.get_or_init(|| {
            let (sender, receiver) = std::sync::mpsc::sync_channel::<()>(1);
            let worker = std::thread::Builder::new().name("victory GPU completion".into());
            worker
                .spawn(move || {
                    // SAFETY: an auto-reset event private to this process-lifetime
                    // worker. Synchronous waits use the host's separate event.
                    let event = unsafe { CreateEventW(None, false, false, None) };
                    while receiver.recv().is_ok() {
                        loop {
                            let (published, next) = {
                                let mut s =
                                    self.submitted.lock().unwrap_or_else(|e| e.into_inner());
                                let _holding = Holding::enter();
                                let published = self.retire_inflight(&mut s);
                                (published, s.flights.front().map(|f| f.value))
                            };
                            if published {
                                crate::queue::pump_held_if_free(crate::queue::Why::Fence);
                            }
                            let Some(next) = next else {
                                break;
                            };
                            // Never wait for the native fence with the submission lock.
                            // Retirement checks for removal after the wait.
                            if let Ok(event) = event {
                                self.wait_for_submission_event(next, event);
                            } else {
                                poll_fence(&self.fence, next);
                            }
                        }
                    }
                })
                .unwrap_or_else(|error| {
                    fail(format_args!(
                        "could not start GPU completion worker: {error}"
                    ))
                });
            sender
        });
        if let Err(TrySendError::Disconnected(())) = sender.try_send(()) {
            fail(format_args!("GPU completion worker stopped"));
        }
    }
}

/// The completion worker is unique to the process host.
pub(super) type Reaper = OnceLock<SyncSender<()>>;

#[cfg(test)]
mod tests;
