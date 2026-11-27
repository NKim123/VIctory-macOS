//! Opt-in command recording. The title copies call arguments into a word
//! stream. A pool translates chunks from different lists in parallel, with
//! at most one worker recording any one list and its segments at a time.

use super::*;
use crate::descriptor::{RtvRecord, ViewPin};
use crate::draw::{IndexView, VertexView};
use std::cell::Cell;
use std::collections::VecDeque;
use std::mem::MaybeUninit;
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock};
use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D12::*;

const CHUNK_WORDS: usize = 1024;
const QUEUED_CHUNKS: usize = 3;
const WORKERS: usize = 8;

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|error| error.into_inner())
}

pub(super) fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| crate::switch::on("VICTORY_THIN") && pool().is_some())
}

/// Raw title objects in the stream and in the state these commands set.
/// A list pins each object once, including across segment cuts.
#[derive(Default)]
pub(super) struct Pins(AddressSet<usize>);

impl Pins {
    pub(super) fn pin(&mut self, object: This) {
        if !object.is_null() && self.0.insert(object as usize) {
            crate::object::add_ref(object);
        }
    }

    fn pin_resource(&mut self, resource: This) {
        if !self.0.contains(&(resource as usize)) && crate::resource::is_resource(resource) {
            self.pin(resource);
        }
    }

    pub(super) fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl Drop for Pins {
    fn drop(&mut self) {
        for &object in &self.0 {
            crate::object::release_ref(object as This);
        }
    }
}

#[derive(Default)]
struct Chunk {
    // Typed payloads can have padding. Only headers and scalar arguments
    // are read as initialized integers.
    words: Vec<MaybeUninit<u64>>,
    views: Vec<Option<ViewPin>>,
}

impl Chunk {
    fn scalar(&mut self, value: u64) {
        self.words.push(MaybeUninit::new(value));
    }

    fn copy<T: Copy>(&mut self, values: &[T]) {
        assert!(align_of::<T>() <= align_of::<u64>());
        let bytes = size_of_val(values);
        let start = self.words.len();
        self.words
            .resize_with(start + bytes.div_ceil(8), MaybeUninit::uninit);
        // SAFETY: the destination has room for the complete slice, including
        // any padding. The packet reads these bytes back using the same type.
        unsafe {
            std::ptr::copy_nonoverlapping(
                values.as_ptr().cast::<MaybeUninit<u8>>(),
                self.words.as_mut_ptr().add(start).cast(),
                bytes,
            );
        }
    }

    fn replay(&self, this: This) {
        PLAYING.with(|playing| {
            let previous = playing.replace(this);
            let mut words = self.words.as_slice();
            while let Some((header, rest)) = words.split_first() {
                // SAFETY: `capture` initializes the header of every packet.
                let header = unsafe { header.assume_init() };
                let (args, remaining) = rest.split_at((header >> 8) as usize);
                let packet = Packet(args);
                // SAFETY: `Front::put` encodes every opcode from this enum.
                let op = unsafe { std::mem::transmute::<u8, Op>(header as u8) };
                replay(this, op, packet, &self.views);
                words = remaining;
            }
            playing.set(previous);
        });
    }
}

thread_local! {
    static PLAYING: Cell<This> = const { Cell::new(std::ptr::null()) };
}

struct Packet<'a>(&'a [MaybeUninit<u64>]);

impl<'a> Packet<'a> {
    fn word(&mut self) -> u64 {
        let (word, rest) = self.0.split_first().expect("thin scalar argument");
        self.0 = rest;
        // SAFETY: each decoder reads its scalar arguments in capture order.
        unsafe { word.assume_init() }
    }

    fn take<T: Copy>(&mut self, count: usize) -> &'a [T] {
        assert!(align_of::<T>() <= align_of::<u64>());
        let (words, rest) = self.0.split_at((count * size_of::<T>()).div_ceil(8));
        self.0 = rest;
        // SAFETY: the matching encoder copied this many values of T into
        // aligned storage in the packet, which is borrowed for replay.
        unsafe { std::slice::from_raw_parts(words.as_ptr().cast(), count) }
    }

    fn array<T: Copy>(&mut self) -> &'a [T] {
        let count = self.word() as usize;
        self.take(count)
    }

    fn one<T: Copy>(&mut self) -> &'a T {
        &self.take(1)[0]
    }
}

#[derive(Default)]
pub(super) struct Front {
    chunk: Chunk,
    lane: Arc<Lane>,
    pub(super) pins: Pins,
    calling_thread: bool,
}

#[derive(Default)]
struct Lane {
    pending: Mutex<Pending>,
    changed: Condvar,
}

#[derive(Default)]
struct Pending {
    chunks: VecDeque<Chunk>,
    spare: Option<Chunk>,
    scheduled: bool,
}

impl Lane {
    fn wait(&self) {
        drop(
            self.changed
                .wait_while(lock(&self.pending), |p| p.scheduled)
                .unwrap_or_else(|error| error.into_inner()),
        );
    }

    fn replay(&self, this: This) {
        let mut pending = lock(&self.pending);
        while let Some(mut chunk) = pending.chunks.pop_front() {
            self.changed.notify_all();
            drop(pending);
            chunk.replay(this);
            chunk.words.clear();
            chunk.views.clear();
            pending = lock(&self.pending);
            pending.spare = Some(chunk);
        }
        pending.scheduled = false;
        self.changed.notify_all();
    }
}

#[derive(Default)]
struct Pool {
    ready: Mutex<VecDeque<(usize, Arc<Lane>)>>,
    work: Condvar,
}

fn pool() -> Option<&'static Pool> {
    static POOL: OnceLock<Option<Arc<Pool>>> = OnceLock::new();
    POOL.get_or_init(|| {
        let pool = Arc::new(Pool::default());
        let mut started = 0;
        for n in 0..WORKERS {
            let pool = pool.clone();
            let worker = std::thread::Builder::new()
                .name(format!("victory-record-{n}"))
                .spawn(move || {
                    loop {
                        let (address, lane) = {
                            let mut ready = pool
                                .work
                                .wait_while(lock(&pool.ready), |ready| ready.is_empty())
                                .unwrap_or_else(|error| error.into_inner());
                            ready.pop_front().expect("a thin recording was scheduled")
                        };
                        // The title list drains all jobs before reset or release,
                        // keeping its address and draw state live during replay.
                        lane.replay(address as This);
                    }
                });
            started += usize::from(worker.is_ok());
        }
        (started != 0).then_some(pool)
    })
    .as_deref()
}

impl Front {
    fn schedule(&mut self, this: This) {
        if self.chunk.words.is_empty() {
            return;
        }
        let mut pending = self
            .lane
            .changed
            .wait_while(lock(&self.lane.pending), |p| {
                p.chunks.len() == QUEUED_CHUNKS
            })
            .unwrap_or_else(|error| error.into_inner());
        let spare = pending.spare.take().unwrap_or_default();
        pending
            .chunks
            .push_back(std::mem::replace(&mut self.chunk, spare));
        if !std::mem::replace(&mut pending.scheduled, true) {
            let pool = pool().expect("thin list requires a recording worker");
            lock(&pool.ready).push_back((this as usize, self.lane.clone()));
            pool.work.notify_one();
        }
    }

    fn put(&mut self, this: This, op: Op, args: &[u64], fill: impl FnOnce(&mut Front)) {
        let at = self.chunk.words.len();
        self.chunk.scalar(0);
        self.chunk
            .words
            .extend(args.iter().copied().map(MaybeUninit::new));
        fill(self);
        let len = self.chunk.words.len() - at - 1;
        self.chunk.words[at].write(op as u64 | ((len as u64) << 8));
        if self.calling_thread {
            self.chunk.replay(this);
            self.chunk.words.clear();
            self.chunk.views.clear();
        } else if self.chunk.words.len() >= CHUNK_WORDS {
            self.schedule(this);
        }
    }

    fn array<T: Copy>(&mut self, values: &[T]) {
        self.chunk.scalar(values.len() as u64);
        self.chunk.copy(values);
    }
}

fn capture(this: This, op: Op, args: &[u64], fill: impl FnOnce(&mut Front)) {
    if let Some(front) = title_list(this).and_then(|list| list.thin.as_ref()) {
        lock(front).put(this, op, args, fill);
    }
}

pub(super) fn wait(list: &List) {
    if let Some(front) = &list.thin {
        lock(front).lane.wait();
    }
}

pub(super) fn discard(list: &List) -> Pins {
    let Some(front) = &list.thin else {
        return Pins::default();
    };
    let mut front = lock(front);
    front.lane.wait();
    front.chunk.words.clear();
    front.chunk.views.clear();
    std::mem::take(&mut front.pins)
}

/// Synchronous callers that need the open native list form a boundary.
/// Replay can itself look up that list, including after changing segments.
pub(super) fn flush(this: This, list: &List) {
    if let Some(front) = &list.thin
        && PLAYING.with(|playing| playing.get() != this)
    {
        let mut front = lock(front);
        front.schedule(this);
        front.lane.wait();
    }
}

/// Callers with the raw native list may keep using it between title calls.
/// Drain first, then replay subsequent packets on their calling thread.
pub(super) fn use_calling_thread(this: This, list: &List) {
    if let Some(front) = &list.thin {
        let mut front = lock(front);
        front.schedule(this);
        front.lane.wait();
        front.calling_thread = true;
    }
}

fn close_recording(this: This) {
    with_state(this, |host, state| {
        state.flush_hdr_probe(host);
        host.close();
    });
}

pub(super) fn close(this: This, list: &List) {
    if let Some(front) = &list.thin {
        let mut front = lock(front);
        front.put(this, Op::Close, &[], |_| {});
        front.schedule(this);
        front.lane.wait();
    } else {
        close_recording(this);
    }
}

// Plain commands are generated from the same argument list for capture and
// replay. The opt-in table and the normal table share the replay methods.
macro_rules! commands {
    ($( $op:ident $slot:literal $front:ident => $back:path, ( $( $arg:ident: $ty:ty ),* ) $(pin [$($pin:ident),*])?; )*) => {
        #[repr(u8)]
        enum Op {
            $($op,)*
            Close, Viewports, Scissors, BlendFactor, Heaps, ComputeConstants, GraphicsConstants,
            IndexBuffer, VertexBuffers, Targets, ClearRtv, ClearDsv, ClearUav,
            Barrier, Groups, CopyTexture,
        }

        $(extern "system" fn $front(this: This, $($arg: $ty),*) {
            capture(this, Op::$op, &[$($arg as u64),*], |_front: &mut Front| {
                $($(_front.pins.pin($pin);)*)?
            });
        })*

        const PLAIN: &[(usize, *const ())] = &[$(($slot, $front as *const ()),)*];

        fn replay(this: This, op: Op, mut packet: Packet<'_>, views: &[Option<ViewPin>]) {
            match op {
                $(Op::$op => {
                    $(let $arg = packet.word() as $ty;)*
                    $back(this, $($arg),*);
                })*
                _ => replay_complex(this, op, packet, views),
            }
        }
    }
}

commands! {
    Draw 12 draw => state::draw, (vertices: u32, instances: u32, first: u32, first_instance: u32);
    DrawIndexed 13 draw_indexed => state::draw_indexed, (indices: u32, instances: u32, first: u32, base: i32, first_instance: u32);
    Dispatch 14 dispatch => state::dispatch, (x: u32, y: u32, z: u32);
    CopyBuffer 15 copy_buffer => super::copy_buffer_region, (dst: This, dst_offset: u64, src: This, src_offset: u64, len: u64) pin [dst, src];
    CopyResource 17 copy_resource => super::copy_resource, (dst: This, src: This) pin [dst, src];
    Resolve 19 resolve => super::resolve, (dst: This, dst_sub: u32, src: This, src_sub: u32, format: u32) pin [dst, src];
    Topology 20 topology => state::topology, (topology: u32);
    StencilRef 24 stencil_ref => state::stencil_ref, (reference: u32);
    Pipeline 25 pipeline => super::set_pipeline, (pipeline: This) pin [pipeline];
    Queries 30 queries => super::resolve_queries, (heap: This, kind: u32, first: u32, count: u32, buffer: This, offset: u64) pin [buffer];
    ComputeTable 35 compute_table => state::compute_table, (slot: u32, handle: u64);
    GraphicsTable 36 graphics_table => state::graphics_table, (slot: u32, handle: u64);
    ComputeConstant 37 compute_constant => state::compute_constant, (slot: u32, value: u32, offset: u32);
    GraphicsConstant 38 graphics_constant => state::graphics_constant, (slot: u32, value: u32, offset: u32);
    ComputeView 41 compute_view => state::compute_view, (slot: u32, address: u64);
    GraphicsView 42 graphics_view => state::graphics_view, (slot: u32, address: u64);
    Indirect 59 indirect => super::execute_indirect, (signature: This, max: u32, args: This, offset: u64, count: This, count_offset: u64) pin [signature, args, count];
    CopyMemory 60 copy_memory => super::copy_memory, (dst: u64, src: u64, len: u64, flags: u32);
    DepthBounds 68 depth_bounds => replay_depth_bounds, (min: u32, max: u32);
    Write32 69 write32 => super::write32, (address: u64, value: u32, flush: u32, flags: u32);
    Write64 70 write64 => super::write64, (address: u64, value: u64, flush: u32, flags: u32);
    Wait32 71 wait32 => super::wait32, (address: u64, comparison: u32, reference: u32, mask: u32, flags: u32);
    Wait64 72 wait64 => super::wait64, (address: u64, comparison: u32, reference: u64, flags: u32);
    DispatchX 78 dispatch_x => super::dispatch_x, (x: u32, y: u32, z: u32, flags: u32);
    CopyResourceX 84 copy_resource_x => super::copy_resource_x, (dst: This, src: This, flags: u32) pin [dst, src];
    MeshX 98 mesh_x => state::dispatch_mesh_x, (x: u32, flags: u64, indirect: u64, payload: u64);
    Mesh 127 mesh => state::dispatch_mesh, (x: u32, y: u32, z: u32);
    Sync 129 sync => super::sync, (flags: u32, control: u32, address: u64, size: u64);
}

fn replay_depth_bounds(this: This, min: u32, max: u32) {
    super::depth_bounds(this, f32::from_bits(min), f32::from_bits(max));
}

extern "system" fn capture_depth_bounds(this: This, min: f32, max: f32) {
    depth_bounds(this, min.to_bits(), max.to_bits());
}

extern "system" fn viewports(this: This, count: u32, values: *const D3D12_VIEWPORT) {
    capture(this, Op::Viewports, &[], |front| {
        front.array(state::items(values, count))
    });
}

extern "system" fn scissors(this: This, count: u32, rects: *const RECT) {
    capture(this, Op::Scissors, &[], |front| {
        front.array(state::items(rects, count))
    });
}

extern "system" fn blend_factor(this: This, factor: *const [f32; 4]) {
    capture(this, Op::BlendFactor, &[], |front| {
        front.array(state::items(factor, 1))
    });
}

extern "system" fn heaps(this: This, count: u32, heaps: *const This) {
    capture(this, Op::Heaps, &[], |front| {
        let heaps = state::items(heaps, count);
        for &heap in heaps {
            front.pins.pin(heap);
        }
        front.array(heaps);
    });
}

extern "system" fn compute_constants(
    this: This,
    slot: u32,
    count: u32,
    values: *const u32,
    offset: u32,
) {
    capture(
        this,
        Op::ComputeConstants,
        &[slot.into(), offset.into()],
        |front| {
            front.array(state::items(values, count));
        },
    );
}

extern "system" fn graphics_constants(
    this: This,
    slot: u32,
    count: u32,
    values: *const u32,
    offset: u32,
) {
    capture(
        this,
        Op::GraphicsConstants,
        &[slot.into(), offset.into()],
        |front| {
            front.array(state::items(values, count));
        },
    );
}

extern "system" fn index_buffer(this: This, view: *const IndexView) {
    capture(this, Op::IndexBuffer, &[], |front| {
        front.array(state::items(view, 1))
    });
}

extern "system" fn vertex_buffers(this: This, start: u32, count: u32, views: *const VertexView) {
    capture(
        this,
        Op::VertexBuffers,
        &[start.into(), count.into()],
        |front| {
            front.array(state::items(views, count));
        },
    );
}

extern "system" fn targets(
    this: This,
    count: u32,
    handles: *const u64,
    range: i32,
    depth: *const u64,
) {
    capture(this, Op::Targets, &[], |front| {
        let handles = state::target_handles(count, handles, range);
        front.chunk.scalar(handles.len() as u64);
        for handle in handles {
            let view = crate::descriptor::rtv_record(handle);
            if let Some(record) = view {
                front.pins.pin(record.resource);
            }
            front.chunk.copy(&[(handle, view)]);
        }
        let depth = state::items(depth, 1).first().copied().unwrap_or(0);
        let view = crate::descriptor::dsv_record(depth);
        if let Some(record) = view {
            front.pins.pin(record.resource);
        }
        front.chunk.copy(&[view]);
    });
}

extern "system" fn clear_rtv(
    this: This,
    handle: u64,
    color: *const [f32; 4],
    count: u32,
    rects: *const Rect,
) {
    let (Some(view), &[color]) = (
        crate::descriptor::rtv_record(handle),
        state::items(color, 1),
    ) else {
        return;
    };
    capture(this, Op::ClearRtv, &[], |front| {
        front.pins.pin(view.resource);
        front.chunk.copy(&[view]);
        front.chunk.copy(&[color]);
        front.array(state::items(rects, count));
    });
}

extern "system" fn clear_dsv(
    this: This,
    handle: u64,
    flags: u32,
    depth: f32,
    stencil: u8,
    count: u32,
    rects: *const Rect,
) {
    let Some(view) = crate::descriptor::dsv_record(handle) else {
        return;
    };
    capture(
        this,
        Op::ClearDsv,
        &[flags.into(), depth.to_bits().into(), stencil.into()],
        |front| {
            front.pins.pin(view.resource);
            front.chunk.copy(&[view]);
            front.array(state::items(rects, count));
        },
    );
}

extern "system" fn clear_uav_uint(
    this: This,
    _gpu: u64,
    cpu: u64,
    _resource: This,
    values: *const [u32; 4],
    count: u32,
    rects: *const RECT,
) {
    clear_uav(this, cpu, values, count, rects, false);
}

extern "system" fn clear_uav_float(
    this: This,
    _gpu: u64,
    cpu: u64,
    _resource: This,
    values: *const [u32; 4],
    count: u32,
    rects: *const RECT,
) {
    clear_uav(this, cpu, values, count, rects, true);
}

fn clear_uav(
    this: This,
    cpu: u64,
    values: *const [u32; 4],
    count: u32,
    rects: *const RECT,
    float: bool,
) {
    let &[values] = state::items(values, 1) else {
        return;
    };
    capture(this, Op::ClearUav, &[float.into()], |front| {
        front.chunk.scalar(front.chunk.views.len() as u64);
        front.chunk.views.push(crate::descriptor::view(cpu));
        front.chunk.copy(&[values]);
        front.array(state::items(rects, count));
    });
}

extern "system" fn barrier(this: This, count: u32, entries: *const Barrier) -> u32 {
    if !entries.is_null() {
        capture(this, Op::Barrier, &[], |front| {
            let entries = state::items(entries, count);
            for entry in entries {
                if entry.kind <= 2 {
                    front.pins.pin(entry.first);
                    if entry.kind == 1 {
                        front.pins.pin(entry.second as This);
                    }
                }
            }
            front.array(entries);
        });
    }
    0
}

extern "system" fn barrier_groups(this: This, count: u32, groups: *const u8) {
    capture(
        this,
        Op::Groups,
        &[count.into(), u64::from(!groups.is_null())],
        |front| {
            for group in state::items(groups.cast::<enhanced::Group>(), count) {
                front.chunk.scalar(group.kind.0 as u64);
                front.chunk.scalar(group.count.into());
                front.chunk.scalar(u64::from(!group.entries.is_null()));
                match group.kind {
                    D3D12_BARRIER_TYPE_GLOBAL => front.chunk.copy(state::items(
                        group.entries.cast::<D3D12_GLOBAL_BARRIER>(),
                        group.count,
                    )),
                    D3D12_BARRIER_TYPE_TEXTURE => {
                        let entries =
                            state::items(group.entries.cast::<enhanced::Texture>(), group.count);
                        for entry in entries {
                            front.pins.pin_resource(entry.resource);
                        }
                        front.chunk.copy(entries);
                    }
                    D3D12_BARRIER_TYPE_BUFFER => {
                        let entries =
                            state::items(group.entries.cast::<enhanced::Buffer>(), group.count);
                        for entry in entries {
                            front.pins.pin_resource(entry.resource);
                        }
                        front.chunk.copy(entries);
                    }
                    _ => {}
                }
            }
        },
    );
}

extern "system" fn copy_texture(
    this: This,
    dst: *const CopyLocation,
    x: u32,
    y: u32,
    z: u32,
    src: *const CopyLocation,
    bounds: *const [u32; 6],
) {
    let (&[dst], &[src]) = (state::items(dst, 1), state::items(src, 1)) else {
        return;
    };
    capture(
        this,
        Op::CopyTexture,
        &[x.into(), y.into(), z.into()],
        |front| {
            front.pins.pin(dst.resource);
            front.pins.pin(src.resource);
            front.chunk.copy(&[dst, src]);
            front.array(state::items(bounds, 1));
        },
    );
}

#[allow(clippy::too_many_arguments)] // Console method signature.
extern "system" fn copy_texture_x(
    this: This,
    dst: *const CopyLocation,
    x: u32,
    y: u32,
    z: u32,
    src: *const CopyLocation,
    bounds: *const [u32; 6],
    _flags: u32,
) {
    copy_texture(this, dst, x, y, z, src, bounds);
}

fn replay_complex(this: This, op: Op, mut packet: Packet<'_>, views: &[Option<ViewPin>]) {
    match op {
        Op::Close => close_recording(this),
        Op::Viewports => {
            let values = packet.array();
            state::viewports(this, values.len() as u32, values.as_ptr());
        }
        Op::Scissors => {
            let rects = packet.array();
            state::scissors(this, rects.len() as u32, rects.as_ptr());
        }
        Op::BlendFactor => state::blend_factor(this, optional_ptr(packet.array())),
        Op::Heaps => {
            let heaps = packet.array();
            state::heaps(this, heaps.len() as u32, heaps.as_ptr());
        }
        Op::ComputeConstants | Op::GraphicsConstants => {
            let slot = packet.word() as u32;
            let offset = packet.word() as u32;
            let values = packet.array();
            let apply = if matches!(op, Op::ComputeConstants) {
                state::compute_constants
            } else {
                state::graphics_constants
            };
            apply(this, slot, values.len() as u32, values.as_ptr(), offset);
        }
        Op::IndexBuffer => state::index_buffer(this, optional_ptr(packet.array())),
        Op::VertexBuffers => {
            let start = packet.word() as u32;
            let count = packet.word() as u32;
            state::vertex_buffers(this, start, count, optional_ptr(packet.array()));
        }
        Op::Targets => {
            let rtvs = packet.array();
            let dsv = *packet.one();
            with_state(this, |list, state| {
                state.set_target_records(list, rtvs, dsv)
            });
        }
        Op::ClearRtv => {
            let view: &RtvRecord = packet.one();
            let color = packet.one();
            let rects = packet.array();
            super::clear_rtv(
                this,
                std::ptr::from_ref(view) as u64,
                color,
                rects.len() as u32,
                rects.as_ptr(),
            );
        }
        Op::ClearDsv => {
            let flags = packet.word() as u32;
            let depth = f32::from_bits(packet.word() as u32);
            let stencil = packet.word() as u8;
            let view: &RtvRecord = packet.one();
            let rects = packet.array();
            super::clear_dsv(
                this,
                std::ptr::from_ref(view) as u64,
                flags,
                depth,
                stencil,
                rects.len() as u32,
                rects.as_ptr(),
            );
        }
        Op::ClearUav => {
            let float = packet.word() != 0;
            let view = &views[packet.word() as usize];
            let values = *packet.one();
            let rects = packet.array();
            with_state(this, |list, state| {
                state.clear_unordered_view(list, view.as_deref(), values, float, rects)
            });
        }
        Op::Barrier => {
            let barriers = packet.array();
            super::barrier(this, barriers.len() as u32, barriers.as_ptr());
        }
        Op::Groups => replay_groups(this, packet),
        Op::CopyTexture => {
            let x = packet.word() as u32;
            let y = packet.word() as u32;
            let z = packet.word() as u32;
            let locations: &[CopyLocation] = packet.take(2);
            super::copy_texture_region(
                this,
                &locations[0],
                x,
                y,
                z,
                &locations[1],
                optional_ptr(packet.array()),
            );
        }
        _ => unreachable!("plain opcode handled by replay"),
    }
}

fn optional_ptr<T>(items: &[T]) -> *const T {
    items.first().map_or(std::ptr::null(), std::ptr::from_ref)
}

fn replay_groups(this: This, mut packet: Packet<'_>) {
    let count = packet.word() as u32;
    if packet.word() == 0 {
        super::barrier_groups(this, count, std::ptr::null());
        return;
    }
    let mut groups = Vec::with_capacity(count as usize);
    for _ in 0..count {
        let kind = D3D12_BARRIER_TYPE(packet.word() as i32);
        let count = packet.word() as u32;
        let present = packet.word() != 0;
        let entries = if !present {
            std::ptr::null()
        } else {
            let len = count as usize;
            match kind {
                D3D12_BARRIER_TYPE_GLOBAL => {
                    packet.take::<D3D12_GLOBAL_BARRIER>(len).as_ptr().cast()
                }
                D3D12_BARRIER_TYPE_TEXTURE => packet.take::<enhanced::Texture>(len).as_ptr().cast(),
                D3D12_BARRIER_TYPE_BUFFER => packet.take::<enhanced::Buffer>(len).as_ptr().cast(),
                // Unknown groups are counted but their entries are never read.
                _ => std::ptr::NonNull::dangling().as_ptr(),
            }
        };
        groups.push(enhanced::Group {
            kind,
            count,
            entries,
        });
    }
    super::barrier_groups(this, count, groups.as_ptr().cast());
}

pub(super) const fn table(newer: bool) -> [*const (); 160] {
    let mut table = list_table(newer);
    const ARRAY: &[(usize, *const ())] = &[
        (16, copy_texture as *const ()),
        (21, viewports as *const ()),
        (22, scissors as *const ()),
        (23, blend_factor as *const ()),
        (26, barrier as *const ()),
        (32, heaps as *const ()),
        (39, compute_constants as *const ()),
        (40, graphics_constants as *const ()),
        (43, compute_view as *const ()),
        (44, graphics_view as *const ()),
        (45, compute_view as *const ()),
        (46, graphics_view as *const ()),
        (47, index_buffer as *const ()),
        (48, vertex_buffers as *const ()),
        (50, targets as *const ()),
        (51, clear_dsv as *const ()),
        (52, clear_rtv as *const ()),
        (53, clear_uav_uint as *const ()),
        (54, clear_uav_float as *const ()),
        (68, capture_depth_bounds as *const ()),
        (83, copy_texture_x as *const ()),
        (130, sync as *const ()),
    ];
    let mut i = 0;
    while i < PLAIN.len() + ARRAY.len() {
        let (slot, method) = if i < PLAIN.len() {
            PLAIN[i]
        } else {
            ARRAY[i - PLAIN.len()]
        };
        table[if newer { newer_slot(slot) } else { slot }] = method;
        i += 1;
    }
    if newer {
        table[146] = barrier_groups as *const ();
    }
    table
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::slot;
    use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
    use windows::core::s;

    fn method<F: Copy>(list: This, index: usize) -> F {
        let index = if has_newer_table(list) {
            newer_slot(index)
        } else {
            index
        };
        slot(list, index * 8)
    }

    fn make_list(pipeline: This) -> (This, This) {
        let null = std::ptr::null();
        let mut allocator = null;
        let mut list = null;
        assert_eq!(
            create_allocator(null, 0, &ALLOCATOR_IIDS[0], &mut allocator),
            S_OK
        );
        assert_eq!(
            create_list(null, 0, 0, allocator, pipeline, &LIST_IIDS[0], &mut list),
            S_OK
        );
        (allocator, list)
    }

    #[test]
    fn thin_tables_cover_every_recording_method() {
        // Lifecycle, object methods, and methods that do not record any work
        // have the same callbacks. Allocation returns a pointer immediately.
        let immediate = [0, 2, 7, 8, 9, 10, 28, 29, 33, 34, 55, 73, 99, 124, 137];
        for newer in [false, true] {
            let normal = list_table(newer);
            let thin = table(newer);
            for &(index, _) in METHODS.iter().chain(&MORE_METHODS) {
                let at = if newer { newer_slot(index) } else { index };
                assert_eq!(
                    thin[at] == normal[at],
                    immediate.contains(&index),
                    "slot {index}"
                );
            }
            if newer {
                assert_ne!(thin[146], normal[146]);
                assert_eq!(thin[148], normal[148]);
            }
        }
    }

    #[test]
    fn thin_recording_copies_arguments_and_drains_at_close() {
        let Some(_) = crate::host::host() else { return };
        let null = std::ptr::null();
        let (allocator, list) = make_list(null);
        let mut constants = [3, 5];
        method::<extern "system" fn(This, u32, u32, *const u32, u32)>(list, 39)(
            list,
            0,
            2,
            constants.as_ptr(),
            0,
        );
        constants.fill(19);
        let constant = method::<extern "system" fn(This, u32, u32, u32)>(list, 37);
        for _ in 0..CHUNK_WORDS / 2 {
            constant(list, 0, 13, 2);
        }
        constant(list, 0, 7, 3);
        if enabled() {
            let front = lock(
                title_list(list)
                    .unwrap()
                    .thin
                    .as_ref()
                    .expect("opt-in table"),
            );
            assert!(!front.chunk.words.is_empty() && front.chunk.words.len() < CHUNK_WORDS);
        }
        assert_eq!(super::super::close(list), S_OK);
        let recording = title_list(list).unwrap();
        assert_eq!(
            lock(&recording.state).compute_constants_for_test(0),
            [3, 5, 13, 7]
        );
        if let Some(front) = &recording.thin {
            assert!(!lock(&lock(front).lane.pending).scheduled);
        }

        assert_eq!(reset(list, allocator, null), S_OK);
        assert!(
            lock(&recording.state)
                .compute_constants_for_test(0)
                .is_empty()
        );
        assert!(!owns_submission_storage(list));
        assert!(!allocate_upload(list, 16, 16).is_null());
        assert!(owns_submission_storage(list));
        assert_eq!(super::super::close(list), S_OK);
        assert_eq!(reset(list, allocator, null), S_OK);
        assert!(!owns_submission_storage(list));
        assert_eq!(list_release(list), 0);
        assert_eq!(allocator_release(allocator), 0);
    }

    fn buffer() -> This {
        committed(&[1, 0, 0, 0, 512, 0, 1, 1 | 1 << 16, 0, 1, 0, 1, 4, 0])
    }

    fn committed(desc: &[u32; 14]) -> This {
        let null = std::ptr::null();
        let properties = [1u32, 0, 0, 0, 0];
        let mut out = null;
        assert_eq!(
            crate::resource::create_committed(
                null,
                properties.as_ptr().cast(),
                0,
                desc.as_ptr().cast(),
                0,
                null.cast(),
                &PAGEABLE_IIDS[0],
                &mut out
            ),
            S_OK
        );
        out
    }

    #[test]
    fn thin_waits_cut_the_replayed_segments_in_order() {
        let Some(host) = crate::host::host() else {
            return;
        };
        let (allocator, list) = make_list(std::ptr::null());
        let mut flag = 0u32;
        let address = (&raw mut flag) as u64;
        let write = method::<extern "system" fn(This, u64, u32, u32, u32)>(list, 69);
        write(list, address, 1, 0, 0);
        let wait = method::<extern "system" fn(This, u64, u32, u32, u32, u32)>(list, 71);
        wait(list, address, 7 /* greater or equal */, 1, u32::MAX, 0);
        write(list, address, 2, 0, 0);
        assert_eq!(super::super::close(list), S_OK);
        let runs = segments(list);
        assert_eq!(runs.len(), 2);
        assert_eq!(flag, 0);
        crate::queue::submit_test_segments(0, runs);
        drop(host.quiet());
        assert_eq!(flag, 2);
        assert_eq!(list_release(list), 0);
        assert_eq!(allocator_release(allocator), 0);
    }

    #[test]
    fn thin_clear_keeps_the_original_view_across_submission_and_reset() {
        let Some(host) = crate::host::host() else {
            return;
        };
        let null = std::ptr::null();
        let (allocator, list) = make_list(null);
        let texture = [3, 0, 0, 0, 1, 0, 1, 1 | 1 << 16, 28, 1, 0, 0, 1, 0];
        let target = committed(&texture);
        let replacement = committed(&texture);
        let mut descriptor = [0u64; 16];
        let descriptor = descriptor.as_mut_ptr() as u64;
        crate::descriptor::create_rtv(null, target, null.cast(), descriptor);
        let clear =
            method::<extern "system" fn(This, u64, *const [f32; 4], u32, *const Rect)>(list, 52);
        clear(list, descriptor, &[1.0, 0.0, 0.0, 1.0], 0, null.cast());
        crate::descriptor::create_rtv(null, replacement, null.cast(), descriptor);
        assert_eq!(super::super::close(list), S_OK);
        let submitted = segments(list);
        let queued = host.execute_title(&submitted);
        assert_eq!(queued, crate::host::inflight_enabled());
        assert_eq!(reset(list, allocator, null), S_OK);
        assert!(!Arc::ptr_eq(&submitted[0], &segments(list)[0]));
        drop(submitted);

        let output = buffer();
        let dst = CopyLocation::placed(
            output,
            crate::resource::PlacedFootprint::new(0, 28, 1, 1, 256),
        );
        let src = CopyLocation::whole(target);
        type Copy = extern "system" fn(
            This,
            *const CopyLocation,
            u32,
            u32,
            u32,
            *const CopyLocation,
            *const [u32; 6],
        );
        method::<Copy>(list, 16)(list, &dst, 0, 0, 0, &src, null.cast());
        assert_eq!(super::super::close(list), S_OK);
        host.execute_title(&segments(list));
        drop(host.quiet());
        let (bytes, _) = crate::resource::buffer_bytes(output).unwrap();
        // SAFETY: the readback has completed and the first pixel is in the buffer.
        assert_eq!(unsafe { bytes.cast::<[u8; 4]>().read() }, [255, 0, 0, 255]);
        for object in [list, allocator, target, replacement, output] {
            assert_eq!(crate::object::release_ref(object), 0);
        }
    }

    fn dispatch_signature() -> This {
        #[repr(C)]
        struct Desc {
            stride: u32,
            count: u32,
            arguments: *const [u32; 4],
            nodes: u32,
        }
        let null = std::ptr::null();
        let args = [2, 0, 0, 0];
        let desc = Desc {
            stride: 12,
            count: 1,
            arguments: &args,
            nodes: 0,
        };
        let mut out = null;
        assert_eq!(
            crate::signature::create(
                null,
                (&raw const desc).cast(),
                null,
                &PAGEABLE_IIDS[0],
                &mut out
            ),
            S_OK
        );
        out
    }

    fn compute_pipeline() -> This {
        use crate::pipeline::{Shader, Translated};
        use victory_shader::{HwStage, emit::HostShader};
        let source = b"
            RWByteAddressBuffer dst : register(u0, space1);
            [numthreads(1, 1, 1)]
            void main(uint3 group : SV_GroupID) { dst.Store(group.x * 4, 29); }
        ";
        let mut code = None;
        // SAFETY: the shader source and out pointer remain live during the call.
        unsafe {
            D3DCompile(
                source.as_ptr().cast(),
                source.len(),
                s!("thin test"),
                None,
                None,
                s!("main"),
                s!("cs_5_1"),
                0,
                0,
                &mut code,
                None,
            )
        }
        .unwrap();
        let code = code.unwrap();
        // SAFETY: the live compiler blob owns this many bytes.
        let dxil = unsafe {
            std::slice::from_raw_parts(code.GetBufferPointer().cast(), code.GetBufferSize())
        }
        .to_vec();
        crate::pipeline::test_compute(Shader::for_test(
            "compute",
            Translated {
                host: HostShader {
                    stage: HwStage::Cs,
                    dxil,
                    user_data: vec![],
                    samplers: vec![],
                    static_samplers: vec![],
                    root_cbvs: vec![],
                    root_srvs: vec![],
                    root_uavs: vec![0],
                    root_srv_bytes: vec![],
                    memory: false,
                    memory_writes: false,
                    view_addresses: false,
                    heap: false,
                    heap_writes: false,
                    view_marks: false,
                    half_views: Vec::new(),
                    heap_index_sources: Vec::new(),
                },
                used: vec![],
                used_written: Vec::new(),
                write_unknown: true,
            },
        ))
    }

    #[test]
    fn thin_gpu_replays_draw_dispatch_indirect_and_barriers_before_submit() {
        let Some(host) = crate::host::host() else {
            return;
        };
        let null = std::ptr::null();
        let (allocator, list) = make_list(null);
        let pipeline = compute_pipeline();
        let (args, output) = (buffer(), buffer());
        let (args_bytes, _) = crate::resource::buffer_bytes(args).unwrap();
        let (out, _) = crate::resource::buffer_bytes(output).unwrap();
        // SAFETY: both committed buffers are writable and at least 512 bytes.
        unsafe {
            args_bytes.cast::<[u32; 3]>().write([2, 1, 1]);
        }
        let signature = dispatch_signature();

        // With no graphics pipeline these calls still have to reach the
        // original draw entry points in order. The synthetic list benchmark
        // exercises graphics pipeline creation and rasterization separately.
        let draws = DRAWS.load(std::sync::atomic::Ordering::Relaxed);
        method::<extern "system" fn(This, u32, u32, u32, u32)>(list, 12)(list, 3, 1, 0, 0);
        method::<extern "system" fn(This, u32, u32, u32, i32, u32)>(list, 13)(list, 3, 1, 0, 0, 0);
        method::<extern "system" fn(This, u32, u32, u32)>(list, 127)(list, 1, 1, 1);
        method::<extern "system" fn(This, This)>(list, 25)(list, pipeline);
        let uav = method::<extern "system" fn(This, u32, u64)>(list, 45);
        uav(list, 0, out as u64);
        method::<extern "system" fn(This, u32, u32, u32)>(list, 14)(list, 1, 1, 1);
        let mut barrier = UAV_BARRIER;
        method::<extern "system" fn(This, u32, *const Barrier) -> u32>(list, 26)(list, 1, &barrier);
        barrier.kind = u32::MAX;
        assert_eq!(barrier.kind, u32::MAX);

        let mut entry = enhanced::Buffer {
            scope: D3D12_GLOBAL_BARRIER {
                SyncBefore: D3D12_BARRIER_SYNC_COMPUTE_SHADING,
                SyncAfter: D3D12_BARRIER_SYNC_COMPUTE_SHADING,
                AccessBefore: D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
                AccessAfter: D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
            },
            resource: output,
            offset: 0,
            size: 512,
        };
        let mut group = enhanced::Group {
            kind: D3D12_BARRIER_TYPE_BUFFER,
            count: 1,
            entries: (&raw const entry).cast(),
        };
        let recording = title_list(list).unwrap();
        if recording.thin.is_some() {
            barrier_groups(list, 1, (&raw const group).cast());
        } else {
            super::super::barrier_groups(list, 1, (&raw const group).cast());
        }
        entry.resource = null;
        group.entries = null.cast();
        assert!(entry.resource.is_null() && group.entries.is_null());
        uav(list, 0, out as u64 + 256);
        method::<extern "system" fn(This, This, u32, This, u64, This, u64)>(list, 59)(
            list, signature, 1, args, 0, null, 0,
        );
        assert_eq!(super::super::close(list), S_OK);
        assert!(DRAWS.load(std::sync::atomic::Ordering::Relaxed) >= draws + 4);
        assert_eq!(segments(list).len(), 1);
        host.execute_title(&segments(list));
        drop(host.quiet());
        // SAFETY: submission has copied declared root-UAV writes back to
        // these positions in the still-live output buffer.
        unsafe {
            assert_eq!(out.cast::<u32>().read(), 29);
            assert_eq!(out.add(256).cast::<[u32; 2]>().read(), [29, 29]);
        }
        for object in [list, allocator, pipeline, signature, args, output] {
            assert_eq!(crate::object::release_ref(object), 0);
        }
    }
}
