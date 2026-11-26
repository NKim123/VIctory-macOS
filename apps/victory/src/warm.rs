//! Replay captured title pipelines in short-lived compiler processes.

use std::collections::{HashSet, VecDeque};
use std::io::{BufRead, BufReader, Write};
use std::os::windows::fs::OpenOptionsExt;
use std::os::windows::process::CommandExt;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail, ensure};
use victory_gpu::pipeline::{self, Prebuild};

pub(crate) const RECORDS: &str = "pipelines/known-v1";
const BATCH_SIZE: usize = 40;
const CHECKPOINT_EVERY: Duration = Duration::from_secs(20);
const TIMEOUT: Duration = Duration::from_secs(180);

pub fn run(paths: Vec<PathBuf>, jobs: Option<usize>) -> Result<()> {
    if !paths.is_empty() {
        // Use the existing shader pack path and its cache.
        return crate::translate::warm(paths, Some(jobs.unwrap_or(2)), true, false);
    }
    ensure!(jobs.is_none(), "--jobs requires shader pack paths");
    let dir = victory_core::Config::load()?.cache_subdir(RECORDS)?;
    let mut keys = Vec::new();
    for entry in std::fs::read_dir(&dir)? {
        let name = entry?.file_name();
        if let Some(key) = name.to_str().and_then(|n| n.strip_suffix(".pso"))
            && pipeline::is_pipeline_key(key)
        {
            keys.push(key.to_owned());
        }
    }
    ensure!(
        !keys.is_empty(),
        "no pipeline records in {}. Set VICTORY_PIPELINE_BOOK=1 for a title run, or pass shader pack paths to warm",
        dir.display()
    );
    let early = std::fs::read_to_string(dir.join("../early-pipelines.txt")).unwrap_or_default();
    let early: HashSet<_> = early.lines().collect();
    keys.sort_unstable_by(|a, b| {
        (!early.contains(a.as_str()), a).cmp(&(!early.contains(b.as_str()), b))
    });

    // A second warm command must not publish a competing copy of the library.
    let _lease = std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(false)
        .share_mode(0)
        .open(dir.join("warm.lock"))
        .context("opening the exclusive warm-up lock")?;
    eprintln!("warming {} recorded pipelines", keys.len());

    let total = keys.len();
    let mut pending = VecDeque::from(keys);
    let mut counts = Counts::default();
    let mut list_cached = true;
    while !pending.is_empty() {
        let mut batch: Vec<_> = pending.drain(..BATCH_SIZE.min(pending.len())).collect();
        let (progress, outcome) = child(&batch, list_cached);
        if list_cached {
            let before = pending.len();
            pending.retain(|key| !progress.known.contains(key));
            counts.cached += before - pending.len();
            list_cached = false;
        }
        counts.add(&progress.saved);
        match outcome {
            Ok(()) => {}
            Err(error) if progress.active.is_some() => {
                let at = progress.active.unwrap();
                eprintln!("pipeline {}: {error:#}", batch[at]);
                counts.refused += 1;
                batch.remove(at);
                // A checkpoint covers the prefix whose library and key list
                // both reached disk. Replay only the uncommitted survivors.
                for key in batch.into_iter().skip(progress.saved.total()).rev() {
                    pending.push_front(key);
                }
            }
            Err(error) => return Err(error),
        }
    }
    println!(
        "pipelines    {} recorded: {} built, {} already cached, {} failed",
        total, counts.built, counts.cached, counts.refused
    );
    ensure!(
        counts.refused == 0,
        "{} recorded pipelines could not be warmed",
        counts.refused
    );
    Ok(())
}

#[derive(Clone, Copy, Default)]
struct Counts {
    built: usize,
    cached: usize,
    refused: usize,
}

impl Counts {
    fn total(self) -> usize {
        self.built + self.cached + self.refused
    }

    fn add(&mut self, other: &Self) {
        self.built += other.built;
        self.cached += other.cached;
        self.refused += other.refused;
    }
}

/// Monitor one child. Attribute failure only to a pipeline the child
/// announced, so startup, protocol and cache-save errors stop the command.
fn child(keys: &[String], list_cached: bool) -> (Progress, Result<()>) {
    let mut progress = Progress::default();
    let mut process = match (|| -> Result<_> {
        const IDLE_PRIORITY_CLASS: u32 = 0x40;
        let mut command = Command::new(std::env::current_exe()?);
        command.arg("warm-batch");
        if list_cached {
            command.arg("--list-cached");
        }
        Ok(command
            .args(keys)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .creation_flags(IDLE_PRIORITY_CLASS)
            .spawn()?)
    })()
    .context("starting the warm-up worker")
    {
        Ok(process) => process,
        Err(error) => return (progress, Err(error)),
    };
    let result = (|| -> Result<()> {
        let stdout = process
            .stdout
            .take()
            .context("opening warm-up worker output")?;
        let (send, receive) = mpsc::channel();
        std::thread::Builder::new()
            .name("victory-warm-output".into())
            .spawn(move || {
                for line in BufReader::new(stdout).lines() {
                    let failed = line.is_err();
                    if send.send(line).is_err() || failed {
                        break;
                    }
                }
            })?;

        let mut heard = Instant::now();
        loop {
            match receive.recv_timeout(TIMEOUT.saturating_sub(heard.elapsed())) {
                Ok(line) => {
                    if let Err(error) = line
                        .context("reading warm-up worker output")
                        .and_then(|line| progress.update(keys, &line))
                    {
                        progress.active = None;
                        return Err(error);
                    }
                    heard = Instant::now();
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    bail!("warm-up worker timed out after {} s", TIMEOUT.as_secs())
                }
                Err(mpsc::RecvTimeoutError::Disconnected) => break,
            }
        }
        // EOF can precede process exit. Keep the timeout in force.
        loop {
            if let Some(status) = process.try_wait()? {
                ensure!(status.success(), "warm-up worker exited with {status}");
                ensure!(
                    progress.saved.total() == keys.len(),
                    "warm-up worker exited before reporting all results"
                );
                return Ok(());
            }
            ensure!(
                heard.elapsed() < TIMEOUT,
                "warm-up worker timed out after {} s",
                TIMEOUT.as_secs()
            );
            std::thread::sleep(Duration::from_millis(20));
        }
    })();
    if result.is_err() {
        let _ = process.kill();
        let _ = process.wait();
    }
    (progress, result)
}

#[derive(Default)]
struct Progress {
    next: usize,
    active: Option<usize>,
    counts: Counts,
    saved: Counts,
    known: HashSet<String>,
}

impl Progress {
    fn update(&mut self, keys: &[String], line: &str) -> Result<()> {
        let (status, key) = line
            .split_once(' ')
            .context("invalid warm-up worker output")?;
        match status {
            "known"
                if self.next == 0 && self.active.is_none() && pipeline::is_pipeline_key(key) =>
            {
                self.known.insert(key.to_owned());
                return Ok(());
            }
            "saved" if self.active.is_none() && key.parse::<usize>() == Ok(self.next) => {
                self.saved = self.counts;
                return Ok(());
            }
            _ => {}
        }
        ensure!(
            keys.get(self.next).is_some_and(|k| k == key),
            "unexpected warm-up worker pipeline key"
        );
        match status {
            "start" if self.active.is_none() => self.active = Some(self.next),
            "built" | "cached" | "failed" if self.active.take().is_some() => {
                self.next += 1;
                match status {
                    "built" => self.counts.built += 1,
                    "cached" => self.counts.cached += 1,
                    _ => self.counts.refused += 1,
                }
            }
            _ => bail!("unexpected warm-up worker status {status}"),
        }
        Ok(())
    }
}

/// Only the child opens and writes the host pipeline library.
pub fn batch(keys: Vec<String>, list_cached: bool) -> Result<()> {
    ensure!(
        !keys.is_empty()
            && keys.len() <= BATCH_SIZE
            && keys.iter().all(|k| pipeline::is_pipeline_key(k)),
        "invalid warm-up batch"
    );
    let cfg = victory_core::Config::load()?;
    let dir = cfg.cache_subdir(RECORDS)?;
    let shaders = cfg.cache_subdir("shaders")?;
    victory_gpu::shader_cache::use_shader_cache(shaders)
        .ok_or_else(|| anyhow!("cannot open shader cache"))?;
    pipeline::use_pipeline_cache(cfg.cache_subdir("pipelines")?);
    pipeline::require_pipeline_cache()?;
    let mut output = std::io::stdout().lock();
    if list_cached {
        for key in pipeline::prebuilt_keys() {
            writeln!(output, "known {key}")?;
        }
        output.flush()?;
    }
    let len = keys.len();
    let mut last_saved = Instant::now();
    for (index, key) in keys.into_iter().enumerate() {
        writeln!(output, "start {key}")?;
        output.flush()?;
        let status = if pipeline::prebuilt(&key) {
            "cached"
        } else {
            match Prebuild::recorded(&dir, &key)
                .map_err(|e| e.to_string())
                .and_then(|p| p.make_cached())
            {
                Ok(true) => "cached",
                Ok(false) => "built",
                Err(e) => {
                    eprintln!("pipeline {key}: {e}");
                    "failed"
                }
            }
        };
        writeln!(output, "{status} {key}")?;
        output.flush()?;
        let completed = index + 1;
        if completed == len || last_saved.elapsed() >= CHECKPOINT_EVERY {
            pipeline::flush_pipeline_cache()?;
            writeln!(output, "saved {completed}")?;
            output.flush()?;
            last_saved = Instant::now();
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn worker_progress_attributes_only_the_pipeline_in_flight() {
        let keys = ["a".repeat(32), "b".repeat(32), "c".repeat(32)];
        let mut p = Progress::default();
        assert!(p.update(&keys, &format!("cached {}", keys[0])).is_err());
        assert!(p.update(&keys, &format!("start {}", keys[1])).is_err());
        assert_eq!(p.active, None);
        p.update(&keys, &format!("start {}", keys[0])).unwrap();
        p.update(&keys, &format!("cached {}", keys[0])).unwrap();
        assert!(p.update(&keys, "saved 2").is_err());
        p.update(&keys, "saved 1").unwrap();
        p.update(&keys, &format!("start {}", keys[1])).unwrap();
        assert_eq!(p.active, Some(1));
        assert!(p.update(&keys, "saved 1").is_err());
        assert_eq!(p.saved.cached, 1);
        assert_eq!(p.saved.total(), 1);
        p.update(&keys, &format!("failed {}", keys[1])).unwrap();
        p.update(&keys, &format!("start {}", keys[2])).unwrap();
        p.update(&keys, &format!("built {}", keys[2])).unwrap();
        assert_eq!(p.active, None);
        assert_eq!(p.next, keys.len());
        assert_eq!(p.saved.total(), 1);
        p.update(&keys, "saved 3").unwrap();
        assert_eq!(p.saved.total(), keys.len());
        assert_eq!(
            (p.counts.cached, p.counts.refused, p.counts.built),
            (1, 1, 1)
        );
    }
}
