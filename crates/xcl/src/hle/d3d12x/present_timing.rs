//! Optional CPU timestamps of DXGI presents accepted by the swap chain.

use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{mpsc, Arc, OnceLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const CAPACITY: usize = 1024;
const HEARTBEAT: Duration = Duration::from_secs(1);
static RECORDER: OnceLock<Recorder> = OnceLock::new();

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Sample {
    sequence: u64,
    time_us: u64,
    present_x: bool,
}

enum Message {
    Present(Sample),
    Flush(mpsc::Sender<()>),
}

struct Recorder {
    send: mpsc::SyncSender<Message>,
    sequence: AtomicU64,
    dropped: Arc<AtomicU64>,
}

impl Recorder {
    fn note(&self, time_us: u64, present_x: bool) {
        let sequence = self.sequence.fetch_add(1, Ordering::Relaxed) + 1;
        let sample = Sample { sequence, time_us, present_x };
        if self.send.try_send(Message::Present(sample)).is_err() {
            self.dropped.fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// Call from the host before installing VFS hooks or creating a presenter.
pub(crate) fn start(logs: &Path) {
    if !std::env::var("XCL_PRESENT_TIMING").is_ok_and(|v| v.trim() == "1") {
        return;
    }
    let wall_ms = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis();
    let path = logs.join(format!("present-timing-{}-{wall_ms}.csv", std::process::id()));
    let result = (|| -> io::Result<Recorder> {
        let file = File::create_new(&path)?;
        let (send, receive) = mpsc::sync_channel(CAPACITY);
        let dropped = Arc::new(AtomicU64::new(0));
        let lost = dropped.clone();
        std::thread::Builder::new().name("xcl-present-timing".into()).spawn(move || {
            if let Err(error) = write_timeline(BufWriter::new(file), receive, &lost) {
                crate::xlog_warn!("present timing: writing the timeline failed: {error}");
            }
        })?;
        Ok(Recorder { send, sequence: AtomicU64::new(0), dropped })
    })();
    match result {
        Ok(recorder) => {
            let _ = RECORDER.set(recorder);
            crate::xlog_info!("present timing: {}", path.display());
        }
        Err(error) => crate::xlog_warn!("present timing: {}: {error}", path.display()),
    }
}

/// Called under the presenter lock. The writer owns the file and this send never waits for it.
pub(crate) fn note(at: Instant, present_x: bool) {
    if let Some(recorder) = RECORDER.get() {
        recorder.note(crate::log::uptime_us_at(at), present_x);
    }
}

/// Best effort from a host exit path outside the presenter lock.
pub(crate) fn flush() {
    if let Some(recorder) = RECORDER.get() {
        let (send, receive) = mpsc::channel();
        if recorder.send.try_send(Message::Flush(send)).is_ok() {
            let _ = receive.recv_timeout(Duration::from_millis(250));
        }
    }
}

fn write_sample(writer: &mut impl Write, sample: Sample) -> io::Result<()> {
    let source = if sample.present_x { "presentx" } else { "fallback" };
    writeln!(writer, "present,{},{},{source},", sample.sequence, sample.time_us)
}

fn write_heartbeat(writer: &mut impl Write, sequence: u64, dropped: &AtomicU64) -> io::Result<()> {
    let at = crate::log::uptime_us_at(Instant::now());
    writeln!(writer, "heartbeat,{sequence},{at},,{}", dropped.load(Ordering::Relaxed))?;
    writer.flush()
}

fn write_timeline(mut writer: impl Write, receive: mpsc::Receiver<Message>, dropped: &AtomicU64) -> io::Result<()> {
    writeln!(writer, "row,sequence,time_us,source,dropped_total")?;
    let mut deadline = Instant::now() + HEARTBEAT;
    let mut last_sequence = 0;
    loop {
        match receive.recv_timeout(deadline.saturating_duration_since(Instant::now())) {
            Ok(Message::Present(sample)) => {
                write_sample(&mut writer, sample)?;
                last_sequence = sample.sequence;
            }
            Ok(Message::Flush(reply)) => {
                write_heartbeat(&mut writer, last_sequence, dropped)?;
                let _ = reply.send(());
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => return write_heartbeat(&mut writer, last_sequence, dropped),
        }
        let now = Instant::now();
        if now >= deadline {
            write_heartbeat(&mut writer, last_sequence, dropped)?;
            deadline = now + HEARTBEAT;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn present_timing_writes_present_x_and_fallback_separately() {
        let mut csv = Vec::new();
        write_sample(&mut csv, Sample { sequence: 1, time_us: 1000, present_x: false }).unwrap();
        write_sample(&mut csv, Sample { sequence: 2, time_us: 34333, present_x: true }).unwrap();
        assert_eq!(csv, b"present,1,1000,fallback,\npresent,2,34333,presentx,\n");
    }

    #[test]
    fn present_timing_full_buffer_leaves_an_explicit_sequence_gap() {
        let (send, receive) = mpsc::sync_channel(1);
        let recorder = Recorder { send, sequence: AtomicU64::new(0), dropped: Arc::new(AtomicU64::new(0)) };
        recorder.note(100, false);
        recorder.note(120, true);
        assert!(matches!(receive.try_recv(), Ok(Message::Present(Sample { sequence: 1, .. }))));
        recorder.note(130, true);
        assert!(matches!(receive.try_recv(), Ok(Message::Present(Sample { sequence: 3, time_us: 130, present_x: true }))));
        assert_eq!(recorder.dropped.load(Ordering::Relaxed), 1);
    }
}
