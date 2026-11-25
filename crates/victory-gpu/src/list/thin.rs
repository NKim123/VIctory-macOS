//! Record commands whose inputs are fully captured at the API boundary.
//! Color clears capture an owned host texture and a CPU view description before
//! entering the stream. Draws and other commands that need live descriptor
//! or resource lookups flush it before translating on the calling thread.

use super::{List, Rect, This, host_list, title_list, with_state};
use crate::{draw::State, host::HostList, resource::HostTexture};
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::sync::{Arc, Mutex, OnceLock};
use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D12::{
    D3D12_RENDER_TARGET_VIEW_DESC, D3D12_RESOURCE_STATES, D3D12_VIEWPORT,
};

const CHUNK: usize = 32;

pub(super) fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| crate::switch::on("VICTORY_THIN") && crate::host::host().is_some())
}

pub(super) enum Command {
    Topology(u32),
    Viewports(Vec<D3D12_VIEWPORT>),
    Scissors(Vec<RECT>),
    BlendFactor(Option<[f32; 4]>),
    StencilRef(u32),
    Table(bool, u32, u64),
    View(bool, u32, u64),
    Constant(bool, u32, u32, u32),
    Constants(bool, u32, Vec<u32>, u32),
    Clear(Arc<HostTexture>, ColorClear, Vec<RECT>),
}

impl Command {
    fn replay(self, list: &HostList, state: &mut State) {
        match self {
            Self::Topology(value) => state.set_topology(list, value),
            Self::Viewports(values) => state.viewports(list, &values),
            Self::Scissors(rects) => state.scissors(list, &rects),
            Self::BlendFactor(value) => state.blend_factor(list, value.as_ref()),
            Self::StencilRef(value) => state.stencil_ref(list, value),
            Self::Table(compute, slot, handle) => state.set_table(compute, slot, handle),
            Self::View(compute, slot, address) => state.set_view(compute, slot, address),
            Self::Constant(compute, slot, value, offset) => {
                state.set_constants(compute, slot, &[value], offset);
            }
            Self::Constants(compute, slot, values, offset) => {
                state.set_constants(compute, slot, &values, offset);
            }
            Self::Clear(texture, clear, rects) => clear.replay(list, &texture, &rects),
        }
    }
}

pub(super) struct ColorClear {
    desc: Option<D3D12_RENDER_TARGET_VIEW_DESC>,
    color: [f32; 4],
}

const RENDER_TARGET: u32 = 0x4;

impl ColorClear {
    fn replay(self, list: &HostList, texture: &Arc<HostTexture>, rects: &[RECT]) {
        let host = crate::host::host().expect("a recording has a host GPU");
        // SAFETY: the list is open, and the owned texture stays live until
        // recording finishes. Its state entry retains it for submission.
        unsafe {
            if let Some((resource, before)) = list.transition_copy(texture, RENDER_TARGET) {
                list.list.ResourceBarrier(&[crate::host::transition(
                    &resource,
                    D3D12_RESOURCE_STATES(before as i32),
                    D3D12_RESOURCE_STATES(RENDER_TARGET as i32),
                )]);
            }
            let view = host.rtv(&texture.texture, self.desc.as_ref());
            list.list
                .ClearRenderTargetView(view, &self.color, Some(rects));
        }
    }
}

pub(super) fn clear_color(
    this: This,
    resource: This,
    desc: Option<D3D12_RENDER_TARGET_VIEW_DESC>,
    color: [f32; 4],
    count: u32,
    rects: *const Rect,
) {
    let Some(texture) = crate::resource::host_copy(resource) else {
        return;
    };
    let clear = ColorClear { desc, color };
    let rects = if count == 0 || rects.is_null() {
        &[]
    } else {
        // SAFETY: the title passes count rectangles in the host layout.
        unsafe { std::slice::from_raw_parts(rects.cast::<RECT>(), count as usize) }
    };
    if enabled() && rects.len() <= 16 {
        // As with immediate recording, a write makes the chosen host copy
        // ineligible for eviction while this command is waiting to replay.
        texture.note_state(RENDER_TARGET);
        record(this, Command::Clear(texture, clear, rects.to_vec()));
    } else if let Some(list) = host_list(this) {
        clear.replay(list, &texture, rects);
    }
}

#[derive(Default)]
pub(super) struct Front {
    commands: Vec<Command>,
    pending: Option<Receiver<()>>,
}

impl Front {
    fn wait(&mut self) {
        if let Some(done) = self.pending.take() {
            // A job owns the state and the host list until it has replayed.
            done.recv().expect("thin command recording stopped");
        }
    }
}

struct Job {
    state: Arc<Mutex<State>>,
    host: Arc<HostList>,
    commands: Vec<Command>,
    done: SyncSender<()>,
    close: bool,
}

fn worker() -> Option<&'static SyncSender<Job>> {
    static WORKER: OnceLock<Option<SyncSender<Job>>> = OnceLock::new();
    WORKER
        .get_or_init(|| {
            let (send, receive) = mpsc::sync_channel::<Job>(2);
            std::thread::Builder::new()
                .name("victory-record".into())
                .spawn(move || {
                    for Job {
                        state,
                        host,
                        commands,
                        done,
                        close,
                    } in receive
                    {
                        replay(commands, &host, &state);
                        if close {
                            host.close();
                        }
                        // Drop our references before releasing the recording thread.
                        drop((host, state));
                        let _ = done.send(());
                    }
                })
                .ok()?;
            Some(send)
        })
        .as_ref()
}

fn replay(commands: Vec<Command>, host: &HostList, state: &Mutex<State>) {
    let mut state = state.lock().unwrap_or_else(|e| e.into_inner());
    for command in commands {
        command.replay(host, &mut state);
    }
}

#[inline]
pub(super) fn record(this: This, command: Command) {
    if this.is_null() {
        return;
    }
    if !enabled() {
        with_state(this, |host, state| command.replay(host, state));
        return;
    }
    let Some(list) = title_list(this) else {
        return;
    };
    let mut front = list.thin.lock().unwrap_or_else(|e| e.into_inner());
    front.commands.push(command);
    if front.commands.len() == CHUNK {
        hand_over(list, &mut front, false);
    }
}

pub(super) fn close(list: &List, host: &HostList) {
    if enabled() {
        let mut front = list.thin.lock().unwrap_or_else(|e| e.into_inner());
        if !front.commands.is_empty() {
            hand_over(list, &mut front, true);
            return;
        }
        front.wait();
    }
    host.close();
}

fn hand_over(list: &List, front: &mut Front, close: bool) {
    front.wait();
    let commands = std::mem::take(&mut front.commands);
    let segment = list.segment.load(std::sync::atomic::Ordering::Relaxed);
    let Some(host) = list
        .hosts
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get(segment)
        .cloned()
    else {
        return;
    };
    let (done, receive) = mpsc::sync_channel(1);
    let job = Job {
        state: list.state.clone(),
        host,
        commands,
        done,
        close,
    };
    let job = if let Some(worker) = worker() {
        match worker.try_send(job) {
            Ok(()) => {
                front.pending = Some(receive);
                return;
            }
            Err(mpsc::TrySendError::Full(job) | mpsc::TrySendError::Disconnected(job)) => job,
        }
    } else {
        job
    };
    replay(job.commands, &job.host, &job.state);
    if close {
        job.host.close();
    }
}

pub(super) fn wait(list: &List) {
    if enabled() {
        list.thin.lock().unwrap_or_else(|e| e.into_inner()).wait();
    }
}

/// A direct operation needs the preceding state; reset and release can
/// discard unplayed state after any running chunk has let go of the list.
#[inline]
pub(super) fn flush(list: &List, host: Option<&HostList>) {
    if !enabled() {
        return;
    }
    let mut front = list.thin.lock().unwrap_or_else(|e| e.into_inner());
    front.wait();
    if !front.commands.is_empty() {
        let commands = std::mem::take(&mut front.commands);
        if let Some(host) = host {
            replay(commands, host, &list.state);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::list::{
        ALLOCATOR_IIDS, LIST_IIDS, close, create_allocator, create_list, list_release, reset, state,
    };
    use crate::object::S_OK;

    #[test]
    fn thin_recording_copies_arguments_flushes_chunks_and_clears_state_on_reset() {
        let null = std::ptr::null();
        let mut allocator = null;
        let mut command_list = null;
        assert_eq!(
            create_allocator(null, 0, &ALLOCATOR_IIDS[0], &mut allocator),
            S_OK
        );
        assert_eq!(
            create_list(
                null,
                0,
                0,
                allocator,
                null,
                &LIST_IIDS[0],
                &mut command_list
            ),
            S_OK
        );

        let mut constants = [3, 5];
        state::compute_constants(command_list, 0, 2, constants.as_ptr(), 0);
        constants.fill(19);
        for _ in 1..CHUNK {
            state::compute_constant(command_list, 0, 13, 2);
        }
        state::compute_constant(command_list, 0, 7, 3);
        if enabled() {
            // SAFETY: this is the list created above.
            let list = unsafe { &*command_list.cast::<List>() };
            assert_eq!(list.thin.lock().unwrap().commands.len(), 1);
        }
        assert_eq!(close(command_list), S_OK);
        with_state(command_list, |_, state| {
            assert_eq!(state.compute_constants_for_test(0), [3, 5, 13, 7]);
        });

        assert_eq!(reset(command_list, allocator, null), S_OK);
        state::compute_constant(command_list, 0, 11, 0);
        assert_eq!(close(command_list), S_OK);
        assert_eq!(reset(command_list, allocator, null), S_OK);
        with_state(command_list, |_, state| {
            assert!(state.compute_constants_for_test(0).is_empty());
        });
        assert!(!crate::list::owns_submission_storage(command_list));
        assert!(!crate::list::allocate_upload(command_list, 16, 16).is_null());
        assert!(crate::list::owns_submission_storage(command_list));
        assert_eq!(close(command_list), S_OK);
        assert_eq!(reset(command_list, allocator, null), S_OK);
        assert!(!crate::list::owns_submission_storage(command_list));

        list_release(command_list);
        crate::list::allocator_release(allocator);
    }
}
