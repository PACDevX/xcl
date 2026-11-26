//! Opt-in recording and playback of the menu buttons seen by GameInput.

use std::collections::HashSet;
use std::fs::{File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::SystemTime;

use super::{button_named, title_has_focus, A, DPAD_DOWN, DPAD_LEFT, DPAD_RIGHT, DPAD_UP};

const MASK: u32 = A | DPAD_UP | DPAD_DOWN | DPAD_LEFT | DPAD_RIGHT;
const BUTTONS: [(u32, &str); 5] = [(A, "A"), (DPAD_UP, "UP"), (DPAD_DOWN, "DOWN"), (DPAD_LEFT, "LEFT"), (DPAD_RIGHT, "RIGHT")];
const VERSION: &str = "# xcl-menu-input-v1";
const COLUMNS: &str = "time_us,buttons";
/// A stopped recording may end before the release was observed.
const FINAL_PRESS_US: u64 = 200_000;
const FOLLOWUP_POLL_US: u64 = 225_000;

enum Mode {
    Off,
    Record(Mutex<Option<Recorder<File>>>),
    Replay { timeline: Replay, frame_delay_us: Option<u64>, followup: Option<Followup> },
}

fn mode() -> &'static Mode {
    static MODE: OnceLock<Mode> = OnceLock::new();
    MODE.get_or_init(|| {
        let record = std::env::var_os("XCL_MENU_INPUT_RECORD").filter(|path| !path.is_empty());
        let replay = std::env::var_os("XCL_MENU_INPUT_REPLAY").filter(|path| !path.is_empty());
        match (record.as_deref().map(Path::new), replay.as_deref().map(Path::new)) {
            (None, None) => Mode::Off,
            (Some(_), Some(_)) => {
                crate::xlog_warn!("GameInput: set only one of XCL_MENU_INPUT_RECORD and XCL_MENU_INPUT_REPLAY");
                Mode::Off
            }
            (Some(path), None) => match OpenOptions::new().write(true).create_new(true).open(path).and_then(Recorder::new) {
                Ok(writer) => {
                    crate::xlog_info!("GameInput: recording foreground menu buttons to {}", path.display());
                    Mode::Record(Mutex::new(Some(writer)))
                }
                Err(e) => {
                    crate::xlog_warn!("GameInput: cannot create menu recording {}: {e}", path.display());
                    Mode::Off
                }
            },
            (None, Some(path)) => match std::fs::read_to_string(path).and_then(|text| Replay::parse(&text)) {
                Ok(timeline) => {
                    let frame_delay_us = std::env::var("XCL_MENU_INPUT_FIRST_FRAME_DELAY_MS").ok().and_then(|delay| {
                        let micros = delay.trim().parse::<u64>().ok().and_then(|ms| ms.checked_mul(1_000));
                        if micros.is_none() {
                            crate::xlog_warn!("GameInput: invalid XCL_MENU_INPUT_FIRST_FRAME_DELAY_MS");
                        }
                        micros
                    });
                    crate::xlog_info!("GameInput: replaying {} menu button changes from {}", timeline.changes.len(), path.display());
                    if let Some(delay) = frame_delay_us {
                        crate::xlog_info!("GameInput: wait at least {} ms after the first game frame", delay / 1_000);
                    }
                    let followup = std::env::var_os("XCL_MENU_INPUT_FOLLOWUP").filter(|path| !path.is_empty()).map(|path| {
                        let path = PathBuf::from(path);
                        crate::xlog_info!("GameInput: accepting menu followups from {}", path.display());
                        Followup { path, state: Mutex::new(FollowupState::default()) }
                    });
                    Mode::Replay { timeline, frame_delay_us, followup }
                }
                Err(e) => {
                    crate::xlog_warn!("GameInput: cannot open menu replay {}: {e}", path.display());
                    Mode::Off
                }
            },
        }
    })
}

fn time_us() -> u64 {
    (crate::log::uptime() * 1_000_000.0) as u64
}

/// Record when a reading is produced for the title. An unfocused title contributes at most a
/// release; input directed at another process is never stored.
pub(super) fn record(buttons: u32) {
    let Mode::Record(writer) = mode() else { return };
    let mut writer = writer.lock().unwrap_or_else(|e| e.into_inner());
    let Some(recorder) = writer.as_mut() else { return };
    let buttons = if buttons & MASK != 0 && title_has_focus() { buttons & MASK } else { 0 };
    if let Err(e) = recorder.observe(time_us(), buttons) {
        *writer = None;
        crate::xlog_warn!("GameInput: menu recording stopped after a write error: {e}");
    }
}

/// Replace only the five recorded buttons. Run before callback change detection.
pub(super) fn replay_buttons(buttons: u32) -> u32 {
    match mode() {
        Mode::Replay { timeline, frame_delay_us, followup } => {
            let now = time_us();
            let first_frame = frame_delay_us.and_then(|_| crate::hle::d3d12x::present::first_frame_time_us());
            let replayed = if let Some(followup) = followup {
                timeline.initial_buttons(now, first_frame, *frame_delay_us).or_else(|| followup.buttons(now))
            } else {
                Some(match frame_delay_us {
                    Some(delay) => timeline.buttons_after_frame(now, first_frame, *delay),
                    None => timeline.buttons_at(now),
                })
            };
            replayed.map_or(buttons, |pressed| (buttons & !MASK) | pressed)
        }
        _ => buttons,
    }
}

struct Followup {
    path: PathBuf,
    state: Mutex<FollowupState>,
}

impl Followup {
    fn buttons(&self, now: u64) -> Option<u32> {
        self.state.lock().unwrap_or_else(|e| e.into_inner()).buttons(now, || {
            // Obtain the timestamp and contents from the same open file across an atomic replacement.
            let mut file = File::open(&self.path).ok()?;
            let modified = file.metadata().and_then(|info| info.modified()).ok()?;
            let mut text = String::new();
            file.read_to_string(&mut text).ok()?;
            Some((time_us(), FollowupVersion { modified, text }))
        })
    }
}

#[derive(PartialEq, Eq, Hash)]
struct FollowupVersion {
    modified: SystemTime,
    text: String,
}

#[derive(Default)]
struct FollowupState {
    next_poll_us: u64,
    seen_versions: HashSet<FollowupVersion>,
    active: Option<(u64, Replay)>,
}

impl FollowupState {
    fn buttons(&mut self, now: u64, read: impl FnOnce() -> Option<(u64, FollowupVersion)>) -> Option<u32> {
        if let Some((start, timeline)) = &self.active {
            if let Some(buttons) = timeline.active_buttons_at(now.saturating_sub(*start)) {
                return Some(buttons);
            }
            self.active = None;
            return None;
        }
        if now < self.next_poll_us {
            return None;
        }
        self.next_poll_us = now.saturating_add(FOLLOWUP_POLL_US);
        let (start, version) = read()?;
        if self.seen_versions.contains(&version) {
            return None;
        }
        let timeline = Replay::parse(&version.text).ok()?;
        self.seen_versions.insert(version);
        let buttons = timeline.active_buttons_at(0)?;
        self.active = Some((start, timeline));
        Some(buttons)
    }
}

struct Recorder<W> {
    writer: W,
    previous: u32,
}

impl<W: Write> Recorder<W> {
    fn new(mut writer: W) -> io::Result<Self> {
        writeln!(writer, "{VERSION}\n{COLUMNS}")?;
        Ok(Self { writer, previous: 0 })
    }

    fn observe(&mut self, at_us: u64, buttons: u32) -> io::Result<()> {
        let buttons = buttons & MASK;
        if buttons == self.previous {
            return Ok(());
        }
        let held = if buttons == 0 {
            "-".to_owned()
        } else {
            BUTTONS.iter().filter_map(|&(bit, name)| (buttons & bit != 0).then_some(name)).collect::<Vec<_>>().join("+")
        };
        // Write each complete row directly so a title failure leaves a usable recording.
        let line = format!("{at_us},{held}\n");
        self.writer.write_all(line.as_bytes())?;
        self.previous = buttons;
        Ok(())
    }
}

struct Replay {
    changes: Vec<(u64, u32)>,
}

impl Replay {
    fn parse(text: &str) -> io::Result<Self> {
        let mut lines = text.lines();
        if lines.next() != Some(VERSION) || lines.next() != Some(COLUMNS) {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "expected an xcl-menu-input-v1 CSV"));
        }
        let mut changes = Vec::new();
        for (line_no, line) in lines.enumerate() {
            let bad = || io::Error::new(io::ErrorKind::InvalidData, format!("invalid menu input row {}", line_no + 3));
            let (at_us, held) = line.split_once(',').ok_or_else(bad)?;
            let at_us = at_us.parse::<u64>().map_err(|_| bad())?;
            if changes.last().is_some_and(|&(previous, _)| at_us < previous) {
                return Err(bad());
            }
            let mut buttons = 0;
            if held != "-" {
                for name in held.split('+') {
                    let bit = button_named(name).filter(|bit| bit & MASK != 0).ok_or_else(bad)?;
                    buttons |= bit;
                }
            }
            changes.push((at_us, buttons));
        }
        Ok(Self { changes })
    }

    fn buttons_after_frame(&self, at_us: u64, first_frame_us: Option<u64>, delay_us: u64) -> u32 {
        self.elapsed_after_frame(at_us, first_frame_us, delay_us).map_or(0, |elapsed| self.buttons_at(elapsed))
    }

    fn elapsed_after_frame(&self, at_us: u64, first_frame_us: Option<u64>, delay_us: u64) -> Option<u64> {
        let start = self.changes.first().map_or(0, |&(start, _)| start);
        let shift = first_frame_us?.saturating_add(delay_us).saturating_sub(start);
        at_us.checked_sub(shift)
    }

    /// Before the initial frame anchor the tape still owns the five menu buttons.
    fn initial_buttons(&self, now: u64, first_frame_us: Option<u64>, frame_delay_us: Option<u64>) -> Option<u32> {
        let elapsed = match frame_delay_us {
            Some(delay) => match self.elapsed_after_frame(now, first_frame_us, delay) {
                Some(elapsed) => elapsed,
                None => return Some(0),
            },
            None => now,
        };
        self.active_buttons_at(elapsed)
    }

    /// Keep the terminal release in the tape. After that instant the real input owns the buttons.
    fn active_buttons_at(&self, at_us: u64) -> Option<u32> {
        let &(last_at, last_buttons) = self.changes.last()?;
        let end = if last_buttons == 0 { last_at } else { last_at.saturating_add(FINAL_PRESS_US) };
        (at_us <= end).then(|| self.buttons_at(at_us))
    }

    fn buttons_at(&self, at_us: u64) -> u32 {
        let count = self.changes.partition_point(|&(time, _)| time <= at_us);
        let Some(&(time, buttons)) = count.checked_sub(1).and_then(|i| self.changes.get(i)) else { return 0 };
        if count == self.changes.len() && at_us - time >= FINAL_PRESS_US {
            return 0;
        }
        buttons
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hle::gameinput::B;

    #[test]
    fn menu_recording_keeps_only_button_changes_and_replays_holds() {
        let mut rec = Recorder::new(Vec::new()).unwrap();
        rec.observe(10, B).unwrap();
        rec.observe(20, A | B).unwrap();
        rec.observe(30, A).unwrap();
        rec.observe(40, A | DPAD_RIGHT).unwrap();
        rec.observe(50, DPAD_RIGHT).unwrap();
        rec.observe(60, 0).unwrap();
        let csv = String::from_utf8(rec.writer).unwrap();
        assert_eq!(csv, "# xcl-menu-input-v1\ntime_us,buttons\n20,A\n40,A+RIGHT\n50,RIGHT\n60,-\n");
        let replay = Replay::parse(&csv).unwrap();
        assert_eq!([19, 20, 39, 40, 50, 60].map(|t| replay.buttons_at(t)), [0, A, A, A | DPAD_RIGHT, DPAD_RIGHT, 0]);
    }

    #[test]
    fn menu_replay_rejects_other_buttons_and_backward_time() {
        for rows in ["20,X", "20,BOGUS", "20,A\n19,-"] {
            let csv = format!("{VERSION}\n{COLUMNS}\n{rows}\n");
            assert!(Replay::parse(&csv).is_err());
        }
    }

    #[test]
    fn menu_replay_waits_for_the_frame_and_shifts_all_changes_if_it_arrives_late() {
        let replay = Replay::parse(&format!("{VERSION}\n{COLUMNS}\n75000000,A\n75200000,-\n76000000,UP\n")).unwrap();
        let delay = 10_000_000;
        assert_eq!(replay.buttons_after_frame(100_000_000, None, delay), 0);
        assert_eq!(replay.buttons_after_frame(74_999_999, Some(10_000_000), delay), 0);
        assert_eq!(replay.buttons_after_frame(75_000_000, Some(10_000_000), delay), A);
        assert_eq!(replay.buttons_after_frame(89_999_999, Some(80_000_000), delay), 0);
        assert_eq!(replay.buttons_after_frame(90_000_000, Some(80_000_000), delay), A);
        assert_eq!(replay.buttons_after_frame(90_200_000, Some(80_000_000), delay), 0);
        assert_eq!(replay.buttons_after_frame(91_000_000, Some(80_000_000), delay), DPAD_UP);
    }

    #[test]
    fn menu_replay_releases_a_press_if_the_recording_stopped_mid_hold() {
        let replay = Replay::parse(&format!("{VERSION}\n{COLUMNS}\n200,UP\n")).unwrap();
        assert_eq!(replay.buttons_at(200 + FINAL_PRESS_US - 1), DPAD_UP);
        assert_eq!(replay.buttons_at(200 + FINAL_PRESS_US), 0);
    }

    #[test]
    fn menu_followup_waits_for_initial_frame_and_releases_the_tape() {
        let replay = Replay::parse(&format!("{VERSION}\n{COLUMNS}\n200,A\n300,RIGHT\n")).unwrap();
        assert_eq!(replay.initial_buttons(1_000_000, None, Some(20)), Some(0));
        assert_eq!(replay.initial_buttons(1019, Some(1000), Some(20)), Some(0));
        assert_eq!(replay.initial_buttons(1020, Some(1000), Some(20)), Some(A));
        let end = 1120 + FINAL_PRESS_US;
        assert_eq!(replay.initial_buttons(end - 1, Some(1000), Some(20)), Some(DPAD_RIGHT));
        assert_eq!(replay.initial_buttons(end, Some(1000), Some(20)), Some(0));
        assert_eq!(replay.initial_buttons(end + 1, Some(1000), Some(20)), None);
    }

    #[test]
    fn menu_followup_polls_when_idle_and_plays_each_valid_revision_once() {
        use std::time::{Duration, UNIX_EPOCH};

        let mut state = FollowupState::default();
        let csv = format!("{VERSION}\n{COLUMNS}\n0,A\n100,DOWN\n200,-\n");
        let revision = |generation, text: &str| FollowupVersion {
            modified: UNIX_EPOCH + Duration::from_secs(generation),
            text: text.to_owned(),
        };
        let poll = FOLLOWUP_POLL_US;
        let unexpected_read = || panic!("polled a followup too soon");
        assert_eq!(state.buttons(0, || None), None);
        assert_eq!(state.buttons(poll - 1, unexpected_read), None);
        assert_eq!(state.buttons(poll, || Some((poll, revision(0, "invalid CSV")))), None);
        let start = 2 * poll + 50;
        assert_eq!(state.buttons(2 * poll, || Some((start, revision(0, &csv)))), Some(A));
        assert_eq!(state.buttons(start + 99, unexpected_read), Some(A));
        assert_eq!(state.buttons(start + 100, unexpected_read), Some(DPAD_DOWN));
        assert_eq!(state.buttons(start + 200, unexpected_read), Some(0));
        assert_eq!(state.buttons(start + 201, unexpected_read), None);
        assert_eq!(state.buttons(3 * poll, || Some((3 * poll, revision(0, &csv)))), None);
        assert_eq!(state.buttons(4 * poll, || Some((4 * poll, revision(1, &csv)))), Some(A));
        assert_eq!(state.buttons(4 * poll + 201, unexpected_read), None);
        assert_eq!(state.buttons(5 * poll, || Some((5 * poll, revision(0, &csv)))), None);
    }
}
