//! Move self-contained submissions to one worker. Queue operations that can
//! touch title memory, waits, signals and presents drain it in call order.
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

use crate::host::HostList;

pub(super) const LIST_LIMIT: usize = 4;
const BYTE_LIMIT: u64 = 16 * 1024 * 1024;

pub(super) struct Lane {
    // Only one caller admits work or crosses a synchronous queue boundary.
    admission: Mutex<()>,
    sender: SyncSender<Message>,
    outstanding: Arc<Outstanding>,
}

#[derive(Default)]
struct Outstanding {
    lists: AtomicU64,
    bytes: AtomicU64,
}

enum Message {
    Submit(Submission),
    Drain(SyncSender<()>),
}

struct Submission {
    queue: usize,
    lists: Vec<Arc<HostList>>,
    bytes: u64,
}

impl Lane {
    pub(super) fn get() -> Option<&'static Self> {
        if crate::host::inflight_enabled() {
            return None;
        }
        static LANE: OnceLock<Option<Lane>> = OnceLock::new();
        LANE.get_or_init(|| {
            let join = std::env::var("VICTORY_JOIN_PARTS")
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(if crate::host::pipeline_submissions() {
                    2
                } else {
                    1
                })
                .clamp(1, LIST_LIMIT);
            if join == 1
                && !crate::host::pipeline_submissions()
                && !crate::switch::on("VICTORY_ASYNC_SUBMIT")
            {
                return None;
            }
            let host = crate::host::host()?;
            Lane::start(
                move |parts| {
                    host.execute_host_parts(parts);
                    let mut account = super::account();
                    account.pieces += parts.len() as u32;
                    account.segments += parts.iter().map(|part| part.len() as u32).sum::<u32>();
                },
                crate::list::recycle,
                join,
            )
        })
        .as_ref()
    }

    fn start(
        execute: impl Fn(&[&[&HostList]]) + Send + 'static,
        recycle: impl Fn(Arc<HostList>) + Send + 'static,
        join: usize,
    ) -> Option<Self> {
        let (sender, receiver) = mpsc::sync_channel(LIST_LIMIT);
        let outstanding = Arc::new(Outstanding::default());
        let worker_outstanding = outstanding.clone();
        match std::thread::Builder::new()
            .name("victory-submit".into())
            .spawn(move || run(execute, recycle, receiver, &worker_outstanding, join))
        {
            Ok(_) => Some(Self {
                admission: Mutex::new(()),
                sender,
                outstanding,
            }),
            Err(error) => {
                tracing::warn!(%error, "running GPU submissions on the calling thread");
                None
            }
        }
    }

    pub(super) fn order(&self) -> MutexGuard<'_, ()> {
        self.admission.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Returns the lists for a synchronous submission if it cannot go ahead.
    /// The caller must own the admission lock.
    pub(super) fn submit(
        &self,
        queue: usize,
        lists: Vec<Arc<HostList>>,
    ) -> Result<(), Vec<Arc<HostList>>> {
        if lists.is_empty()
            || lists.len() > LIST_LIMIT
            || self.outstanding.lists.load(Ordering::Acquire) as usize + lists.len() > LIST_LIMIT
        {
            return Err(lists);
        }
        let mut bytes = 0u64;
        for list in &lists {
            let Some(estimate) = list.submit_ahead_bytes() else {
                return Err(lists);
            };
            bytes = bytes.saturating_add(estimate);
        }
        if self
            .outstanding
            .bytes
            .load(Ordering::Acquire)
            .saturating_add(bytes)
            > byte_limit()
        {
            return Err(lists);
        }
        let count = lists.len() as u64;
        self.outstanding.lists.fetch_add(count, Ordering::Relaxed);
        self.outstanding.bytes.fetch_add(bytes, Ordering::Relaxed);
        match self.sender.try_send(Message::Submit(Submission {
            queue,
            lists,
            bytes,
        })) {
            Ok(()) => Ok(()),
            Err(
                mpsc::TrySendError::Full(Message::Submit(job))
                | mpsc::TrySendError::Disconnected(Message::Submit(job)),
            ) => {
                self.outstanding.bytes.fetch_sub(bytes, Ordering::Release);
                self.outstanding.lists.fetch_sub(count, Ordering::Release);
                Err(job.lists)
            }
            Err(_) => unreachable!("only submitted a command list"),
        }
    }

    /// Waits for work admitted before this call, including host completion
    /// and release of the worker's references. Called with admission held.
    pub(super) fn drain(&self) {
        if self.outstanding.lists.load(Ordering::Acquire) == 0 {
            return;
        }
        let (done, result) = mpsc::sync_channel(0);
        self.sender
            .send(Message::Drain(done))
            .expect("GPU submission worker stopped");
        result
            .recv()
            .expect("GPU submission worker stopped during drain");
    }
}

/// Use the same override and units as texture eviction, with a conservative
/// fallback when DXGI cannot report a budget.
fn byte_limit() -> u64 {
    static LIMIT: OnceLock<u64> = OnceLock::new();
    *LIMIT.get_or_init(|| {
        let device_mb =
            crate::host::video_memory().map_or(f64::INFINITY, |[(.., local_mb), _]| local_mb);
        BYTE_LIMIT.min((crate::resource::budget_mb(device_mb) * 10_000.0) as u64)
    })
}

fn run(
    execute: impl Fn(&[&[&HostList]]),
    recycle: impl Fn(Arc<HostList>),
    receiver: Receiver<Message>,
    outstanding: &Outstanding,
    join: usize,
) {
    let mut next = None;
    while let Some(message) = next.take().or_else(|| receiver.recv().ok()) {
        let first = match message {
            Message::Submit(job) => job,
            Message::Drain(done) => {
                let _ = done.send(());
                continue;
            }
        };
        let queue = first.queue;
        let mut parts = vec![first];
        while parts.len() < join {
            match receiver.try_recv() {
                Ok(Message::Submit(job)) if job.queue == queue => parts.push(job),
                Ok(message) => {
                    next = Some(message);
                    break;
                }
                Err(_) => break,
            }
        }
        let lists: Vec<Vec<&HostList>> = parts
            .iter()
            .map(|part| part.lists.iter().map(Arc::as_ref).collect())
            .collect();
        let runs: Vec<&[&HostList]> = lists.iter().map(Vec::as_slice).collect();
        execute(&runs);
        for part in parts {
            let count = part.lists.len() as u64;
            // Reclaim uniquely held lists after the fence. Recycling clears
            // their texture references before returning the allocators to
            // the existing bounded spare pool.
            part.lists.into_iter().for_each(&recycle);
            outstanding.bytes.fetch_sub(part.bytes, Ordering::Release);
            outstanding.lists.fetch_sub(count, Ordering::Release);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::HostGpu;

    #[test]
    fn submission_worker_recycles_an_early_reset_list() {
        use crate::object::{HRESULT, S_OK, This, slot};
        use crate::resource::PAGEABLE_IIDS;

        let host = crate::host::host().expect("test needs a host GPU");
        let (started, running) = mpsc::channel();
        let (resume, blocked) = mpsc::channel();
        let gate = std::cell::Cell::new(Some(blocked));
        let spare = Arc::new(Mutex::new(Vec::new()));
        let worker_spare = spare.clone();
        let lane = Lane::start(
            move |parts| {
                if let Some(blocked) = gate.take() {
                    started.send(()).unwrap();
                    blocked.recv().unwrap();
                }
                host.execute_host_parts(parts);
            },
            move |list| crate::list::recycle_into(list, &worker_spare),
            1,
        )
        .unwrap();

        let null: This = std::ptr::null();
        let mut allocator = null;
        let mut list = null;
        let iid = &PAGEABLE_IIDS[1];
        assert_eq!(
            crate::list::create_allocator(null, 0, iid, &mut allocator),
            S_OK
        );
        assert_eq!(
            crate::list::create_list(null, 0, 0, allocator, null, iid, &mut list),
            S_OK
        );
        let close = slot::<extern "system" fn(This) -> HRESULT>(list, 0x48);
        let reset = slot::<extern "system" fn(This, This, This) -> HRESULT>(list, 0x50);
        assert_eq!(close(list), S_OK);

        let _ordered = lane.order();
        let original = crate::list::segments(list);
        let handle = original[0].list.clone();
        assert!(lane.submit(1, original).is_ok());
        running.recv().unwrap();

        assert_eq!(reset(list, allocator, null), S_OK);
        assert!(crate::list::host_list(list).unwrap().list != handle);
        resume.send(()).unwrap();
        lane.drain();

        let recycled = Arc::new(
            spare
                .lock()
                .unwrap()
                .pop()
                .expect("early-reset host list in the spare pool")
                .list,
        );
        assert!(recycled.list == handle);
        recycled.close();
        assert_eq!(recycled.submit_ahead_bytes(), Some(0));
        assert!(lane.submit(1, vec![recycled.clone()]).is_ok());
        lane.drain();

        for object in [list, allocator] {
            assert_eq!(
                slot::<extern "system" fn(This) -> u32>(object, 0x10)(object),
                0
            );
        }
    }

    #[test]
    fn submission_worker_bounds_work_and_joins_only_adjacent_same_queue_parts() {
        let host = crate::host::host().expect("test needs a host GPU");
        let (groups, observed) = mpsc::channel();
        let (resume, blocked) = mpsc::channel();
        let first = std::cell::Cell::new(true);
        let lane = Lane::start(
            move |parts| {
                groups.send(parts.len()).unwrap();
                if first.replace(false) {
                    blocked.recv().unwrap();
                }
                host.execute_host_parts(parts);
            },
            crate::list::recycle,
            LIST_LIMIT,
        )
        .unwrap();
        let _ordered = lane.order();
        let make = || {
            let list = Arc::new(host.new_list().unwrap());
            list.close();
            list
        };

        let first = make();
        assert!(lane.submit(1, vec![first.clone()]).is_ok());
        assert_eq!(observed.recv().unwrap(), 1);
        assert_eq!(
            Arc::strong_count(&first),
            2,
            "the running job retains its list"
        );
        assert!(lane.submit(1, vec![make()]).is_ok());
        assert!(lane.submit(1, vec![make()]).is_ok());
        assert!(lane.submit(2, vec![make()]).is_ok());
        assert!(
            lane.submit(2, vec![make()]).is_err(),
            "submission depth includes running work"
        );
        resume.send(()).unwrap();
        lane.drain();
        assert_eq!(
            Arc::strong_count(&first),
            1,
            "drain releases the worker's list"
        );
        assert_eq!(observed.try_iter().collect::<Vec<_>>(), [2, 1]);

        let outputs = Arc::new(host.new_list().unwrap());
        let mut output = 0u64;
        outputs.defer_write(
            (&raw mut output).cast(),
            crate::queue::Written::Value(42),
            8,
        );
        outputs.close();
        assert!(
            lane.submit(2, vec![outputs.clone()]).is_err(),
            "a CPU-visible completion stays synchronous"
        );
        host.execute(&[&outputs]);
        assert_eq!(output, 42);
    }
}
