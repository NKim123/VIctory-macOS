//! A bounded wall-time report for drivers whose GPU timestamps do not advance.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

use super::HostList;

pub(super) fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| crate::switch::on("VICTORY_SUBMISSION_PROFILE"))
}

/// Per recorded host pipeline. An indirect command counts as one call
/// because the number of records actually executed can be GPU-dependent.
#[derive(Default)]
pub(super) struct Commands(HashMap<usize, (String, u64, u64)>);

impl Commands {
    pub(super) fn add(&mut self, pipeline: usize, compute: bool, label: impl FnOnce() -> String) {
        let (_, draws, dispatches) = self.0.entry(pipeline).or_insert_with(|| (label(), 0, 0));
        if compute {
            *dispatches += 1;
        } else {
            *draws += 1;
        }
    }
}

struct Sample {
    submission: u64,
    native_lists: usize,
    encode: Duration,
    wait: Duration,
    draws: u64,
    dispatches: u64,
    pipelines: Vec<(String, u64)>,
}

#[derive(Default)]
struct Report {
    count: u64,
    encode: Duration,
    wait: Duration,
    slowest: Vec<Sample>,
}

const TOP: usize = 5;
static REPORT: Mutex<Report> = Mutex::new(Report {
    count: 0,
    encode: Duration::ZERO,
    wait: Duration::ZERO,
    slowest: Vec::new(),
});

pub(super) fn record(
    submission: u64,
    native_lists: usize,
    encode: Duration,
    wait: Duration,
    runs: &[&HostList],
) {
    let mut report = REPORT.lock().unwrap_or_else(|e| e.into_inner());
    report.record(submission, native_lists, encode, wait, runs);
}

impl Report {
    fn record(
        &mut self,
        submission: u64,
        native_lists: usize,
        encode: Duration,
        wait: Duration,
        runs: &[&HostList],
    ) {
        self.count += 1;
        self.encode += encode;
        self.wait += wait;
        let cost = encode + wait;
        if self.slowest.len() == TOP
            && self
                .slowest
                .last()
                .is_some_and(|s| s.encode + s.wait >= cost)
        {
            return;
        }

        let mut pipelines = HashMap::<String, u64>::new();
        let (mut draws, mut dispatches) = (0, 0);
        for run in runs {
            for (name, d, c) in run
                .profile_commands
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .0
                .values()
            {
                draws += d;
                dispatches += c;
                *pipelines.entry(name.clone()).or_default() += d + c;
            }
        }
        let mut pipelines: Vec<_> = pipelines.into_iter().collect();
        pipelines.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        pipelines.truncate(3);
        let at = self.slowest.partition_point(|s| s.encode + s.wait >= cost);
        self.slowest.insert(
            at,
            Sample {
                submission,
                native_lists,
                encode,
                wait,
                draws,
                dispatches,
                pipelines,
            },
        );
        self.slowest.truncate(TOP);
    }
}

pub(super) fn log(frames: f64) {
    if !enabled() {
        return;
    }
    let report = std::mem::take(&mut *REPORT.lock().unwrap_or_else(|e| e.into_inner()));
    if report.count == 0 {
        return;
    }
    let ms = |time: Duration| time.as_secs_f64() * 1000.0;
    tracing::info!(
        submissions = report.count,
        encode_ms_per_frame = ms(report.encode) / frames,
        wait_ms_per_frame = ms(report.wait) / frames,
        "host submission profile"
    );
    for s in report.slowest {
        tracing::info!(
            submission = s.submission,
            lists = s.native_lists,
            encode_ms = ms(s.encode),
            wait_ms = ms(s.wait),
            draws = s.draws,
            dispatches = s.dispatches,
            pipelines = ?s.pipelines,
            "slowest host submissions"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn report_keeps_only_the_slowest_submissions_by_combined_time() {
        let mut report = Report::default();
        for n in [6, 2, 5, 4, 7, 1, 3] {
            report.record(
                n,
                1,
                Duration::from_millis(n),
                Duration::from_millis(n * 2),
                &[],
            );
        }
        assert_eq!(report.count, 7);
        assert_eq!(report.encode, Duration::from_millis(28));
        assert_eq!(report.wait, Duration::from_millis(56));
        let slowest: Vec<_> = report
            .slowest
            .iter()
            .map(|sample| sample.submission)
            .collect();
        assert_eq!(slowest, [7, 6, 5, 4, 3]);
    }
}
