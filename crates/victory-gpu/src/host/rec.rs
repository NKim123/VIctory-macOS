//! Collect adjacent barrier calls on a host list until its next use.
//! Adapted from avatarmanz/VIctory-7900XTX (GPL-3.0).

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, Mutex, OnceLock};

use windows::{Win32::Graphics::Direct3D12::*, core::Interface};

use super::{HostList, WidenedEvent};
use crate::{resource::HostTexture, shadow::Shadow};

static CALLS: AtomicU64 = AtomicU64::new(0);
static SUBMITTED: AtomicU64 = AtomicU64::new(0);

struct Held(Vec<D3D12_RESOURCE_BARRIER>);

impl Held {
    fn extend(&mut self, barriers: &[D3D12_RESOURCE_BARRIER]) {
        for barrier in barriers {
            if is_global_uav(barrier) && self.0.last().is_some_and(is_global_uav) {
                continue;
            }
            // SAFETY: the ManuallyDrop fields borrow the original pointers.
            self.0.push(unsafe { std::ptr::read(barrier) });
        }
    }
}

// SAFETY: the barrier records borrow pointers to free-threaded resources
// held by the command list until execution finishes.
unsafe impl Send for Held {}

pub(crate) struct Rec {
    list: ID3D12GraphicsCommandList,
    join: bool,
    pending: AtomicBool,
    held: Mutex<Held>,
    indirect: Mutex<Option<crate::indirect::BatchOutput>>,
}

impl Rec {
    pub(crate) fn new(list: ID3D12GraphicsCommandList) -> Self {
        static JOIN: OnceLock<bool> = OnceLock::new();
        Self {
            list,
            join: *JOIN.get_or_init(|| crate::switch::on("VICTORY_JOIN_BARRIERS")),
            pending: false.into(),
            held: Mutex::new(Held(Vec::new())),
            indirect: Mutex::new(None),
        }
    }

    /// Same contract as the native command. Resources must stay alive until
    /// the list has finished executing. An intervening use flushes the queue.
    #[allow(non_snake_case)]
    pub(crate) unsafe fn ResourceBarrier(&self, barriers: &[D3D12_RESOURCE_BARRIER]) {
        const MAX_HELD: usize = 64;
        if !self.join {
            // SAFETY: the caller's.
            unsafe { self.list.ResourceBarrier(barriers) };
            return;
        }
        if barriers.is_empty() {
            return;
        }
        CALLS.fetch_add(1, Relaxed);
        if barriers.len() >= MAX_HELD {
            self.flush();
            // SAFETY: the caller's.
            unsafe { self.list.ResourceBarrier(barriers) };
            SUBMITTED.fetch_add(1, Relaxed);
            return;
        }
        let mut held = self.held.lock().unwrap_or_else(|e| e.into_inner());
        let repeats_transition = || {
            barriers.iter().filter_map(transitioned).any(|r| {
                held.0
                    .iter()
                    .filter_map(transitioned)
                    .any(|previous| previous == r)
            })
        };
        if held.0.len() + barriers.len() > MAX_HELD || repeats_transition() {
            self.submit_held(&mut held);
        }
        held.extend(barriers);
        self.pending.store(true, Relaxed);
    }

    /// Same contract as the native command. The caller has waited for all
    /// submissions of this recording. A successful reset discards it.
    #[allow(non_snake_case)]
    pub(crate) unsafe fn Reset(
        &self,
        allocator: &ID3D12CommandAllocator,
        pipeline: Option<&ID3D12PipelineState>,
    ) -> windows::core::Result<()> {
        self.flush();
        // SAFETY: the caller's.
        unsafe { self.list.Reset(allocator, pipeline)? };
        self.indirect
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take();
        Ok(())
    }

    fn submit_held(&self, held: &mut Held) {
        if !held.0.is_empty() {
            // SAFETY: the callers supplied valid barriers on this open list.
            unsafe { self.list.ResourceBarrier(&held.0) };
            held.0.clear();
            SUBMITTED.fetch_add(1, Relaxed);
        }
    }

    pub(crate) fn flush(&self) {
        if self.join && self.pending.load(Relaxed) {
            let mut held = self.held.lock().unwrap_or_else(|e| e.into_inner());
            self.submit_held(&mut held);
            self.pending.store(false, Relaxed);
        }
    }
}

impl HostList {
    pub(crate) fn indirect_batch_output(&self) -> Option<ID3D12Resource> {
        let mut output = self.list.indirect.lock().unwrap_or_else(|e| e.into_inner());
        if output.is_none() {
            let lease = crate::indirect::BatchOutput::take()?;
            // Count the buffer against the in-flight storage budget.
            self.deferred().uploads.push(lease.buffer().clone());
            *output = Some(lease);
        }
        output.as_ref().map(|lease| lease.buffer().clone())
    }

    /// The tail state of a texture already transitioned on this list.
    pub(crate) fn recorded_texture_state(&self, texture: &Arc<HostTexture>) -> Option<u32> {
        self.states
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&(Arc::as_ptr(texture) as usize))
            .map(|(_, last, _)| *last)
    }

    /// The widened copy pass samples all pending writers. A scoped UAV
    /// barrier can precede it only if it orders every source that pass reads.
    pub(crate) fn orders_widened_writes(&self, buffers: &[Arc<Shadow>]) -> bool {
        self.widened_pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .all(|event| match event {
                WidenedEvent::Write((shadow, _, _)) => {
                    buffers.iter().any(|buffer| buffer.buffer == shadow.buffer)
                }
                WidenedEvent::Heap(_) => false,
            })
    }

    /// Records a resource-specific UAV dependency. Callers retain the
    /// resource and any title backing allocation for this recording.
    pub(crate) fn scoped_uav(&self, resource: &ID3D12Resource) {
        let barrier = D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                    // SAFETY: as in `transition`, the resource pointer is borrowed.
                    pResource: unsafe { std::mem::transmute_copy(resource) },
                }),
            },
        };
        // SAFETY: the list is open and the caller holds the resource alive.
        unsafe { self.list.ResourceBarrier(&[barrier]) };
    }
}

fn transitioned(barrier: &D3D12_RESOURCE_BARRIER) -> Option<usize> {
    if barrier.Type != D3D12_RESOURCE_BARRIER_TYPE_TRANSITION {
        return None;
    }
    // SAFETY: Type identifies this union member. Read its borrowed pointer.
    unsafe { barrier.Anonymous.Transition.pResource.as_ref() }
        .map(|resource| resource.as_raw() as usize)
}

fn is_global_uav(barrier: &D3D12_RESOURCE_BARRIER) -> bool {
    barrier.Type == D3D12_RESOURCE_BARRIER_TYPE_UAV
        && barrier.Flags == D3D12_RESOURCE_BARRIER_FLAG_NONE
        // SAFETY: Type identifies this union member.
        && unsafe { barrier.Anonymous.UAV.pResource.is_none() }
}

impl std::ops::Deref for Rec {
    type Target = ID3D12GraphicsCommandList;

    fn deref(&self) -> &Self::Target {
        self.flush();
        &self.list
    }
}

#[cfg(test)]
impl PartialEq<ID3D12GraphicsCommandList> for Rec {
    fn eq(&self, other: &ID3D12GraphicsCommandList) -> bool {
        self.list == *other
    }
}

pub(super) fn report(frames: f64) {
    #[cfg(feature = "diagnostics")]
    crate::list::report_enhanced_barriers(frames);
    let calls = CALLS.swap(0, Relaxed);
    let made = SUBMITTED.swap(0, Relaxed);
    if calls > 0 {
        tracing::info!(
            "joined barriers per frame: {:.0} calls into {:.0}",
            calls as f64 / frames,
            made as f64 / frames
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::HostGpu;
    use crate::host::{host, uav_barrier};

    #[test]
    fn joins_only_adjacent_global_uav_barriers() {
        let mut held = Held(Vec::new());
        let alias = D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_ALIASING,
            ..Default::default()
        };
        held.extend(&[uav_barrier()]);
        held.extend(&[uav_barrier(), alias, uav_barrier(), uav_barrier()]);
        assert_eq!(held.0.len(), 3);
        assert!(is_global_uav(&held.0[0]));
        assert_eq!(held.0[1].Type, D3D12_RESOURCE_BARRIER_TYPE_ALIASING);
        assert!(is_global_uav(&held.0[2]));
    }

    #[test]
    fn an_indirect_batch_buffer_stays_leased_until_reset() {
        let Some(host) = host() else { return };
        let first = host.new_list().unwrap();
        let first_output = first.indirect_batch_output().unwrap();
        assert_eq!(first.indirect_batch_output().unwrap(), first_output);
        first.close();

        let second = host.new_list().unwrap();
        let second_output = second.indirect_batch_output().unwrap();
        assert_ne!(second_output, first_output);
        first.reset();
        assert!(first.list.indirect.lock().unwrap().is_none());
        assert_ne!(first.indirect_batch_output().unwrap(), second_output);
        first.close();
        second.close();
    }
}
