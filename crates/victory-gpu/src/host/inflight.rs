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
    pub(super) completion_us: u64,
    flights: VecDeque<InFlight>,
    marks: Vec<Vec<Mark>>,
    feedback: Vec<MarkFeedback>,
    upload_list: Option<Box<HostList>>,
    bytes: u64,
    lists: usize,
}

pub(super) struct InFlight {
    value: u64,
    marks: Vec<Mark>,
    active_marks: usize,
    feedback: Option<MarkFeedback>,
    native_lists: Vec<Option<ID3D12CommandList>>,
    lists: Vec<Arc<HostList>>,
    upload: Option<crate::bindless::OwnedUpload>,
    _tiles: crate::tiled::Flushed,
    _ambient: Option<Arc<crate::ambient_wave::Frozen>>,
    _batch: Batch,
    _pins: Vec<crate::descriptor::ViewPin>,
    _retired: Vec<crate::descriptor::host::Held>,
    computed: Option<crate::memory::ComputedBaselines>,
    ordered: Option<OrderedCompletion>,
    bytes: u64,
    writes: Vec<(BottomWrite, bool)>,
    gpu_writes: crate::ranges::Ranges,
    indirect_gpu_writes: bool,
    shared_writes: bool,
    signals: Vec<TitleSignal>,
}

/// Work that can outlive ExecuteCommandLists but must land before the
/// next submission reuses its mapped shadows, marks or shared views.
#[derive(Default)]
pub(super) struct OrderedCompletion {
    pub(super) landing: Option<u64>,
    pub(super) show_after: bool,
    pub(super) title_memory: bool,
    pub(super) descriptors: bool,
    pub(super) computed: bool,
    pub(super) marked: bool,
    pub(super) shadow: bool,
}

impl OrderedCompletion {
    pub(super) fn reason(&self) -> Option<Boundary> {
        self.reasons().next()
    }

    fn reasons(&self) -> impl Iterator<Item = Boundary> {
        [
            (self.descriptors, Boundary::Descriptors),
            (self.computed, Boundary::ComputedBack),
            (self.marked, Boundary::MarkedBack),
            (self.shadow, Boundary::ShadowBack),
        ]
        .into_iter()
        .filter_map(|(needed, why)| needed.then_some(why))
    }
}

/// A slot's readback of the page bits. Read marks and atomic lock words stay
/// in the shared UAV. Only this slot's fence allows the CPU to see its copy.
pub(super) struct MarkFeedback {
    list: ID3D12GraphicsCommandList,
    _allocator: ID3D12CommandAllocator,
    _readback: ID3D12Resource,
    data: usize,
    mark: Mark,
}

impl MarkFeedback {
    fn new(host: &Host) -> Option<Self> {
        use crate::shadow::WRITE_MARK_BYTES as BYTES;
        let sources = crate::shadow::write_mark_buffers()?;
        static ZERO: OnceLock<Option<ID3D12Resource>> = OnceLock::new();
        let zero = ZERO
            .get_or_init(|| {
                let (buffer, data) = host.mapped_buffer(BYTES, false)?;
                // SAFETY: new upload storage of BYTES bytes, only read after
                // publication by the OnceLock.
                unsafe { data.write_bytes(0, BYTES as usize) };
                Some(buffer)
            })
            .as_ref()?;
        let (readback, data) = host.mapped_buffer(2 * BYTES, true)?;
        // SAFETY: a fresh allocator and a separate ExecuteCommandLists
        // call. D3D12 buffers decay to COMMON between calls, so the page
        // copies follow all writes in this flight and precede the next.
        unsafe {
            let allocator: ID3D12CommandAllocator = host
                .device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                .ok()?;
            let list: ID3D12GraphicsCommandList = host
                .device
                .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)
                .ok()?;
            for (index, &source) in sources.iter().enumerate() {
                list.ResourceBarrier(&[transition(
                    source,
                    D3D12_RESOURCE_STATE_COMMON,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                )]);
                list.CopyBufferRegion(&readback, index as u64 * BYTES, source, 0, BYTES);
                list.ResourceBarrier(&[transition(
                    source,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                )]);
                list.CopyBufferRegion(source, 0, zero, 0, BYTES);
            }
            list.Close().ok()?;
            name(&list, || "in-flight UAV page feedback".into());
            Some(Self {
                list,
                _allocator: allocator,
                _readback: readback,
                data: data as usize,
                mark: Mark::new(host).ok()?,
            })
        }
    }

    fn notify(&self) {
        let words = (crate::shadow::WRITE_MARK_BYTES / 4) as usize;
        // SAFETY: a completed fence for both sections of this slot.
        let bits = unsafe { std::slice::from_raw_parts(self.data as *const u32, 2 * words) };
        for map in bits.chunks_exact(words) {
            crate::shadow::notify_aliased_marks(map);
        }
    }
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

    fn directly_mapped(&self) -> bool {
        crate::shadow::has_aliased_overlap(
            self.address as u64,
            self.address.saturating_add(self.bytes) as u64,
        )
    }

    fn value_for_wait(&self, address: usize, bytes: usize) -> Option<u64> {
        if self.address == address
            && self.bytes == bytes
            && let crate::queue::Written::Value(value) = self.value
        {
            return match bytes {
                4 => Some(u64::from(value as u32)),
                8 => Some(value),
                _ => None,
            };
        }
        None
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
    NativeBottomOfPipe,
    BottomOfPipeRead,
    CpuAccess,
    TextureUploadLabel,
    TextureUploadGpuWrite,
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
static QUEUED_COMPUTED: AtomicU64 = AtomicU64::new(0);
static OVERLAPPED: AtomicU64 = AtomicU64::new(0);
static QUEUED_OVERLAPPED: AtomicU64 = AtomicU64::new(0);
static BOTTOM_FIFO_READS: AtomicU64 = AtomicU64::new(0);
static FALLBACKS: [AtomicU64; Boundary::Fence as usize + 1] =
    [const { AtomicU64::new(0) }; Boundary::Fence as usize + 1];
static ORDERED: [AtomicU64; Boundary::Fence as usize + 1] =
    [const { AtomicU64::new(0) }; Boundary::Fence as usize + 1];
static DRAINS: [AtomicU64; Boundary::Fence as usize + 1] =
    [const { AtomicU64::new(0) }; Boundary::Fence as usize + 1];

pub(super) fn log(frames: f64) {
    if limit() == 0 {
        return;
    }
    let count = |why: Boundary| DRAINS[why as usize].swap(0, Ordering::Relaxed);
    let sync = |why: Boundary| FALLBACKS[why as usize].swap(0, Ordering::Relaxed);
    let ordered = |why: Boundary| ORDERED[why as usize].swap(0, Ordering::Relaxed);
    let back = [
        Boundary::TextureBack,
        Boundary::ComputedBack,
        Boundary::MarkedBack,
        Boundary::ShadowBack,
    ];
    let [texture_back, computed_back, marked_back, shadow_back] = back.map(sync);
    let (mark_copies, mark_widened) = crate::shadow::mark_landing_owners();
    let texture_upload_label = count(Boundary::TextureUploadLabel);
    let texture_upload_gpu_write = count(Boundary::TextureUploadGpuWrite);
    let bottom_cpu = count(Boundary::BottomOfPipe) + texture_upload_label;
    let bottom_native = count(Boundary::NativeBottomOfPipe);
    let bottom_read = count(Boundary::BottomOfPipeRead);
    tracing::info!(
        frames,
        queued = QUEUED.swap(0, Ordering::Relaxed),
        queued_computed = QUEUED_COMPUTED.swap(0, Ordering::Relaxed),
        queued_ordered_computed = ordered(Boundary::ComputedBack),
        queued_ordered_marked = ordered(Boundary::MarkedBack),
        queued_ordered_shadow = ordered(Boundary::ShadowBack),
        queued_ordered_descriptors = ordered(Boundary::Descriptors),
        overlapped = OVERLAPPED.swap(0, Ordering::Relaxed),
        queued_overlapped = QUEUED_OVERLAPPED.swap(0, Ordering::Relaxed),
        synchronous_copy_back = texture_back + computed_back + marked_back + shadow_back,
        synchronous_texture_back = texture_back,
        synchronous_computed_back = computed_back,
        synchronous_marked_back = marked_back,
        mark_copies,
        mark_widened,
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
        bottom_of_pipe = bottom_cpu + bottom_native + bottom_read,
        bottom_cpu,
        bottom_native,
        bottom_read,
        bottom_fifo_reads = BOTTOM_FIFO_READS.swap(0, Ordering::Relaxed),
        diagnostics = count(Boundary::Diagnostics),
        cpu_access = count(Boundary::CpuAccess) + texture_upload_gpu_write,
        texture_upload_label,
        texture_upload_gpu_write,
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

    pub(super) fn first_flight(&self) -> Option<u64> {
        self.flights.front().map(|flight| flight.value)
    }

    pub(super) fn has_bottom_writes(&self) -> bool {
        self.flights.iter().any(|f| !f.writes.is_empty())
    }

    pub(super) fn bottom_write_ranges(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.flights.iter().flat_map(|flight| {
            flight
                .writes
                .iter()
                .map(|(write, _)| (write.address, write.bytes))
        })
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

    pub(super) fn wait_ordered_completion(
        &self,
        s: &mut Submission,
        touches: impl FnOnce(&OrderedCompletion) -> bool,
    ) {
        // An ordered completion is always the youngest flight. Ordinary
        // flights before it have isolated feedback or no CPU write-back.
        if let Some(flight) = s.flights.back()
            && let Some(work) = &flight.ordered
            && touches(work)
        {
            self.wait_inflight(
                s,
                flight.value,
                work.reason().expect("ordered completion reason"),
            );
        }
    }

    /// A CPU touch may need the old label value to have landed first.
    /// Unknown ranges include indirect reads such as computed addresses.
    pub(super) fn wait_bottom_writes(
        &self,
        s: &mut Submission,
        why: Boundary,
        touches: impl Fn(&BottomWrite, bool) -> bool,
    ) -> bool {
        if let Some(flight) = s.flights.iter().rev().find(|f| {
            f.writes
                .iter()
                .any(|(write, may_alias)| touches(write, *may_alias))
        }) {
            self.wait_inflight(s, flight.value, why);
            return true;
        }
        false
    }

    pub(super) fn wait_texture_upload_sources(
        &self,
        s: &mut Submission,
        sources: &[crate::texture::UploadSource],
    ) {
        if sources.is_empty() || !s.busy() {
            return;
        }
        let ranges = || {
            sources
                .iter()
                .map(|(staging, _)| staging.source_range())
                .filter(|&(_, len)| len != 0)
        };
        let mut physical_alias = None;
        let mut gpu_mapping = None;
        let mut has_physical_alias = || {
            *physical_alias
                .get_or_insert_with(|| ranges().any(|(at, len)| crate::watch::aliased(at, len)))
        };
        let mut touches_gpu_mapping = || {
            *gpu_mapping.get_or_insert_with(|| {
                ranges().any(|(at, len)| {
                    crate::shadow::has_aliased_overlap(at as u64, at.saturating_add(len) as u64)
                })
            })
        };
        let pending = s.flights.iter().rev().find_map(|flight| {
            // Copied title-memory outputs are ordered before preparation.
            // Bindless and computed writes can target a mapped GPU buffer
            // at an address only the shader knows until it completes.
            if flight.indirect_gpu_writes && (touches_gpu_mapping() || has_physical_alias())
                || flight.gpu_writes.iter().any(|(from, to)| {
                    ranges().any(|(at, len)| at < to && from < at.saturating_add(len))
                        || has_physical_alias() && crate::watch::aliased(from, to - from)
                })
            {
                return Some((flight.value, Boundary::TextureUploadGpuWrite));
            }
            flight
                .writes
                .iter()
                .any(|(write, aliased)| {
                    ranges().any(|(at, len)| write.overlaps(at, len))
                        || *aliased && has_physical_alias()
                })
                .then_some((flight.value, Boundary::TextureUploadLabel))
        });
        if let Some((value, why)) = pending {
            self.wait_inflight(s, value, why);
        }
    }

    /// A native GPU list can access a CPU label through a directly mapped
    /// resource. Its CPU preparation can overlap, but queue it only once
    /// those stores have been published. A raw GPU wait on the CPU label
    /// here would prevent synchronous callers from retiring the ring.
    pub(super) fn order_shared_bottom_writes(&self, s: &mut Submission) {
        if let Some(flight) = s.flights.iter().rev().find(|f| f.shared_writes) {
            self.wait_inflight(s, flight.value, Boundary::NativeBottomOfPipe);
        }
    }

    /// Evaluate a held GPU queue's wait against a label already ordered
    /// before it on our single native queue. The held scheduler will still
    /// apply the mask and comparison, and CPU publication stays fenced.
    pub(crate) fn queued_wait_value(&self, address: usize, bytes: usize) -> Option<u64> {
        if limit() == 0 || HOLDS_SUBMISSIONS.with(std::cell::Cell::get) {
            return None;
        }
        let mut s = self.submitted.lock().unwrap_or_else(|e| e.into_inner());
        let _holding = Holding::enter();
        let mut younger_title_writes = false;
        for flight in s.flights.iter().rev() {
            if let Some((write, _)) = flight
                .writes
                .iter()
                .rev()
                .find(|(write, may_alias)| *may_alias || write.overlaps(address, bytes))
            {
                if !younger_title_writes && let Some(value) = write.value_for_wait(address, bytes) {
                    // A shared label still lands before native execution,
                    // via order_shared_bottom_writes. This only lets the
                    // held queue start preparing its consumer.
                    BOTTOM_FIFO_READS.fetch_add(1, Ordering::Relaxed);
                    return Some(value);
                }
                // Timestamps, partial overlaps and possible physical
                // aliases still require reading the published bytes.
                let value = flight.value;
                self.wait_inflight(&mut s, value, Boundary::BottomOfPipeRead);
                break;
            }
            younger_title_writes |= flight
                .ordered
                .as_ref()
                .is_some_and(|work| work.title_memory);
        }
        None
    }

    fn wait_inflight(&self, s: &mut Submission, value: u64, why: Boundary) {
        let mut waited = false;
        while let Some(first) = s.flights.front().filter(|first| first.value <= value) {
            // SAFETY: the queue has signalled this fence value.
            let completed = unsafe { self.fence.GetCompletedValue() };
            if completed == u64::MAX {
                self.flight_device_removed(s);
            }
            if completed < first.value {
                if !waited {
                    DRAINS[why as usize].fetch_add(1, Ordering::Relaxed);
                    waited = true;
                }
                let started = Instant::now();
                // This caller holds the submission lock. Publish older labels
                // and signals before waiting for later flights that may rely
                // on the title reacting to those earlier completions.
                if !self.wait_for_submission(first.value) {
                    self.flight_device_removed(s);
                }
                let waited_us = started.elapsed().as_micros() as u64;
                s.completion_us += waited_us;
                SUBMISSION_US[1].fetch_add(waited_us, Ordering::Relaxed);
            }
            self.retire_inflight(s);
        }
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
        let mut finished_feedback = false;
        while s.flights.front().is_some_and(|f| f.value <= completed) {
            let mut f = s.flights.pop_front().expect("completed flight");
            let mut feedback_ran = true;
            if let Some(feedback) = f.feedback {
                let _timed = crate::queue::CpuTimer::new(crate::queue::CpuPhase::FlightFeedback);
                feedback_ran = feedback.mark.saw(f.value);
                if feedback_ran {
                    feedback.notify();
                }
                s.feedback.push(feedback);
            }
            let ran = f.marks[..f.active_marks]
                .iter()
                .all(|mark| submission_ran(mark, f.value))
                && feedback_ran;
            if !ran && recovery_mark.is_none() {
                DRAINS[Boundary::Recovery as usize].fetch_add(1, Ordering::Relaxed);
                if let Some(last) = s.flights.back()
                    && last.value > completed
                {
                    let started = Instant::now();
                    if !self.wait_for_submission(last.value) {
                        self.flight_device_removed(s);
                    }
                    let waited_us = started.elapsed().as_micros() as u64;
                    s.completion_us += waited_us;
                    SUBMISSION_US[1].fetch_add(waited_us, Ordering::Relaxed);
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
            if let Some(work) = f.ordered {
                assert!(
                    !s.busy(),
                    "a younger flight must wait for ordered completion"
                );
                let runs: Vec<_> = f
                    .upload
                    .iter()
                    .map(|upload| &*upload.list)
                    .chain(f.lists.iter().map(AsRef::as_ref))
                    .collect();
                s.completion_us += self.finish_title_submission(
                    &runs,
                    f.computed.take().expect("ordered baselines"),
                    CompletionStatus {
                        landing: work.landing,
                        show_after: work.show_after,
                        verified: true,
                        incomplete: if ran { &[] } else { &[0] },
                        individual: false,
                    },
                    &mut f._pins,
                );
                finished_feedback = true;
            } else {
                let missed = f.computed.as_ref().is_some_and(|baselines| {
                    f.lists.iter().any(|list| list.deferred().missed(baselines))
                });
                crate::memory::note_submission(missed);
                if let Some(mut baselines) = f.computed {
                    for list in &f.lists {
                        for work in &list.deferred().after {
                            if let AfterWork::Computed(feedback) = work {
                                feedback.finish(&mut baselines, ran, true);
                            }
                        }
                    }
                }
                for list in &f.lists {
                    list.notify_aliased_writes();
                }
            }
            if let Some(upload) = f.upload {
                if !ran {
                    for staging in upload.sources {
                        staging.retry_upload();
                    }
                }
                s.recycle_upload(upload.list);
            }
            for list in f.lists {
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
        }
        if let Some(marks) = recovery_mark {
            self.lost(&mut s.value, &marks[0]);
            s.marks.push(marks);
        }
        if !s.busy() {
            if finished_feedback {
                // The ordered completion consumed all GPU feedback. A
                // descriptor published during it can still reset a slot.
                crate::shadow::clear_forgotten_reads();
            } else {
                // Ordinary flights leave shared read feedback and alias
                // marks for this idle boundary.
                crate::shadow::copy_back_aliased_marks();
                log_gave_up();
                let read = crate::shadow::take_reads();
                if !read.is_empty() {
                    crate::descriptor::make_read(&read);
                }
            }
            release_retired();
            self.log_messages();
        }
        let published = !completions.is_empty();
        for (writes, signals) in completions {
            writes.into_iter().for_each(|(write, _)| write.publish());
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

    pub(super) fn flight_feedback(&self, s: &mut Submission) -> Option<MarkFeedback> {
        s.feedback.pop().or_else(|| MarkFeedback::new(self))
    }

    pub(super) fn recycle_flight_marks(&self, s: &mut Submission, marks: Vec<Mark>) {
        s.marks.push(marks);
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
        feedback: Option<MarkFeedback>,
        computed: Option<crate::memory::ComputedBaselines>,
        ordered: Option<OrderedCompletion>,
        overlapped: bool,
    ) {
        let started = Instant::now();
        if computed.is_some()
            && lists
                .iter()
                .any(|list| list.deferred().has_computed_feedback())
        {
            QUEUED_COMPUTED.fetch_add(1, Ordering::Relaxed);
        }
        if let Some(work) = &ordered {
            for why in work.reasons() {
                ORDERED[why as usize].fetch_add(1, Ordering::Relaxed);
            }
        }
        let mut writes = Vec::new();
        let mut gpu_writes = crate::ranges::Ranges::default();
        let mut indirect_gpu_writes = false;
        for list in lists {
            let deferred = list.deferred();
            writes.extend(
                deferred
                    .writes
                    .iter()
                    .map(|&write| (write, crate::watch::aliased(write.address, write.bytes))),
            );
            indirect_gpu_writes |=
                list.view_writes.load(Ordering::Relaxed) || deferred.has_computed_feedback();
            for step in &deferred.copies {
                if let Step::GpuWrite(copy, from, len, ..) = step
                    && copy.is_aliased()
                {
                    gpu_writes.insert(*from, from.saturating_add(*len), 0);
                }
            }
        }
        let shared_writes = writes
            .iter()
            .any(|(write, may_alias)| *may_alias || write.directly_mapped());
        s.value += 1;
        let value = s.value;
        let queue = self.queue();
        let tiles = crate::tiled::flush(&queue);
        let ambient = if crate::ambient_wave::enabled() {
            let candidates: Vec<_> = lists
                .iter()
                .flat_map(|list| list.deferred().ambient.clone())
                .collect();
            crate::ambient_wave::freeze(&candidates, &self.fence, value)
        } else {
            None
        };
        let active_marks = boundaries.len() + 1;
        let mut marked = Vec::with_capacity(native_lists.len() + active_marks + 2);
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
        if let Some(feedback) = &feedback {
            feedback.mark.arm(value);
            let here = marked.len();
            marked.push(feedback.list.cast().ok());
            marked.push(feedback.mark.list());
            // SAFETY: follows the title work on the same queue, in a
            // separate call. Both lists are retained by the flight.
            unsafe { queue.ExecuteCommandLists(&marked[here..]) };
        }
        #[cfg(feature = "diagnostics")]
        crate::resource::life::note(0, crate::resource::life::SUBMITTED, value, 0);
        s.lists += lists.len();
        s.bytes += bytes;
        s.flights.push_back(InFlight {
            value,
            marks,
            active_marks,
            feedback,
            native_lists: marked,
            lists: lists.to_vec(),
            upload,
            _tiles: tiles,
            _ambient: ambient,
            _batch: batch,
            _pins: pins,
            _retired: take_retired(),
            computed,
            ordered,
            bytes,
            writes,
            gpu_writes,
            indirect_gpu_writes,
            shared_writes,
            signals: Vec::new(),
        });
        if !self.signal_submission(&queue, value) {
            self.flight_device_removed(s);
        }
        QUEUED.fetch_add(1, Ordering::Relaxed);
        if overlapped {
            QUEUED_OVERLAPPED.fetch_add(1, Ordering::Relaxed);
        }
        SUBMISSION_US[1].fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
    }

    pub(super) fn note_sync(&self, s: &mut Submission, why: Boundary) {
        if limit() != 0 {
            FALLBACKS[why as usize].fetch_add(1, Ordering::Relaxed);
            self.drain_inflight(s, why);
        }
    }

    pub(super) fn note_overlap(&self, s: &Submission) -> bool {
        if let Some(last) = s.flights.back() {
            // SAFETY: a live fence. An earlier GPU submission was still
            // pending when this caller acquired the lock to prepare work.
            if unsafe { self.fence.GetCompletedValue() } < last.value {
                OVERLAPPED.fetch_add(1, Ordering::Relaxed);
                return true;
            }
        }
        false
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
            self.wake_reaper(s.first_flight().expect("pending title signal"));
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

    pub(super) fn wake_reaper(&'static self, first: u64) {
        let sender = self.reaper.get_or_init(|| {
            let (sender, receiver) = std::sync::mpsc::sync_channel::<u64>(1);
            let worker = std::thread::Builder::new().name("victory GPU completion".into());
            worker
                .spawn(move || {
                    // SAFETY: an auto-reset event private to this process-lifetime
                    // worker. Synchronous waits use the host's separate event.
                    let event = unsafe { CreateEventW(None, false, false, None) };
                    while let Ok(mut next) = receiver.recv() {
                        loop {
                            // Past hints are harmless. The fence is monotonic
                            // and retirement always checks the current front.
                            if let Ok(event) = event {
                                self.wait_for_submission_event(next, event);
                            } else {
                                poll_fence(&self.fence, next);
                            }
                            let (published, front) = {
                                let lock =
                                    crate::queue::CpuTimer::new(crate::queue::CpuPhase::ReaperLock);
                                let mut s =
                                    self.submitted.lock().unwrap_or_else(|e| e.into_inner());
                                drop(lock);
                                let _holding = Holding::enter();
                                let _retire = crate::queue::CpuTimer::new(
                                    crate::queue::CpuPhase::ReaperRetire,
                                );
                                let published = self.retire_inflight(&mut s);
                                (published, s.first_flight())
                            };
                            if published {
                                crate::queue::pump_held_if_free(crate::queue::Why::Fence);
                            }
                            let Some(front) = front else {
                                break;
                            };
                            next = front;
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
        if let Err(TrySendError::Disconnected(_)) = sender.try_send(first) {
            fail(format_args!("GPU completion worker stopped"));
        }
    }
}

/// The completion worker is unique to the process host.
pub(super) type Reaper = OnceLock<SyncSender<u64>>;

#[cfg(test)]
mod tests;
