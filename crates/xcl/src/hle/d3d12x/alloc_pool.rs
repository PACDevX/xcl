//! A title's command allocator Reset does not take recordings the title still submits.
//!
//! On the PC a list's recording lives in its allocator: after `allocator.Reset()` a list that
//! recorded on it must not be submitted again ("The command allocator was reset after the
//! command list was recorded", debug layer id 541 – the runtime removes the device,
//! 0x887a0001). A retail title does exactly that, which its console lets it: one recording,
//! submitted at 61.6 s, at 63.2 s on a second queue and at 67.8 s again, with the allocator
//! it had been given reset at 67.2 s.
//!
//! So the title's allocator is a name for a PC allocator that changes: at the title's Reset,
//! when lists still hold recordings on the PC allocator in use (or the GPU has not finished
//! them), that one is put aside as it is and the title's allocator goes on with another – one
//! put aside earlier whose lists have all been reset since and whose submits are done, or a new
//! one. A list reset on the title's allocator records on the PC allocator in use. A list holds
//! one recording, so no more allocators are put aside than there are lists.
use std::collections::{HashMap, HashSet};
use std::ffi::c_void;
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;

use crate::util::HRESULT;

#[derive(Default)]
struct Backing {
    /// PC lists whose recording is on this allocator
    users: HashSet<usize>,
    /// Native mesh submissions that may use this backing and have no accepted mark yet.
    mesh_pending: usize,
    /// queue → (fence, value after the last submit of one of them)
    marks: HashMap<usize, (ID3D12Fence, u64)>,
}

impl Backing {
    unsafe fn free(&self) -> bool {
        self.users.is_empty() && self.mesh_pending == 0 && self.marks.values().all(|(f, v)| f.GetCompletedValue() >= *v)
    }
}

#[derive(Default)]
struct State {
    /// title allocator → the PC allocator in use (absent: itself)
    cur: HashMap<usize, usize>,
    /// title allocator → PC allocators put aside
    aside: HashMap<usize, Vec<usize>>,
    backings: HashMap<usize, Backing>,
    of_list: HashMap<usize, usize>,
    /// title allocator → the type of the lists on it
    types: HashMap<usize, i32>,
    fences: HashMap<usize, (ID3D12Fence, u64)>,
    made: u64,
    swaps: u64,
    /// PC list → its last submit (queue, fence, value after it, recording number then)
    last_submit: HashMap<usize, LastSubmit>,
    /// PC list → recordings so far (a Reset or creation starts the next one)
    recordings: HashMap<usize, u64>,
    resubmits: u64,
    mesh_recordings: HashMap<usize, Arc<MeshProgress>>,
    mesh_reserved: HashSet<usize>,
}

struct LastSubmit {
    queue: usize,
    fence: ID3D12Fence,
    value: u64,
    recording: u64,
}

unsafe impl Send for State {}

static STATE: Mutex<Option<State>> = Mutex::new(None);
static MESH_SUBMIT_AVAILABLE: Condvar = Condvar::new();

#[derive(Clone)]
pub(super) struct MeshCompletion {
    pub(super) fence: ID3D12Fence,
    pub(super) value: u64,
}

#[derive(Default)]
pub(super) struct MeshProgress {
    submission: Mutex<MeshSubmission>,
}

#[derive(Default)]
struct MeshSubmission {
    reserved: bool,
    completion: Option<MeshCompletion>,
}

impl MeshProgress {
    /// Only ended recordings may use this. None means the native submit has not yet been marked.
    pub(super) fn retirement_state(&self) -> Option<Option<MeshCompletion>> {
        let submit = self.submission.lock().unwrap();
        (!submit.reserved).then(|| submit.completion.clone())
    }
}

/// Scratch follows this recording even when the same native list is reset or released.
pub(super) fn track_mesh_recording(list: usize) -> Option<Arc<MeshProgress>> {
    let mut state = STATE.lock().unwrap();
    let state = state.as_mut()?;
    if !state.of_list.contains_key(&list) {
        return None;
    }
    Some(state.mesh_recordings.entry(list).or_default().clone())
}

pub(super) fn end_mesh_recording(list: usize, progress: &Arc<MeshProgress>) {
    let mut state = STATE.lock().unwrap();
    let Some(state) = state.as_mut() else { return };
    if state.mesh_recordings.get(&list).is_some_and(|other| Arc::ptr_eq(progress, other)) {
        state.mesh_recordings.remove(&list);
    }
}

/// `XCL_ALLOC_SWAP=0` turns it off.
pub fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("XCL_ALLOC_SWAP").is_ok_and(|v| v.trim() == "0"))
}

fn move_list(s: &mut State, list: usize, backing: usize) {
    if let Some(old) = s.of_list.insert(list, backing) {
        if old != backing {
            if let Some(b) = s.backings.get_mut(&old) {
                b.users.remove(&list);
            }
        }
    }
    s.backings.entry(backing).or_default().users.insert(list);
}

/// The PC list's final Release ended its recording. Submitted work still holds the backing
/// through `marks`; removing the list must not remove those fences or the other users.
pub(super) fn forget_list(list: usize) {
    let mut g = STATE.lock().unwrap();
    let Some(s) = g.as_mut() else { return };
    s.last_submit.remove(&list);
    s.recordings.remove(&list);
    s.mesh_recordings.remove(&list);
    if let Some(backing) = s.of_list.remove(&list) {
        if let Some(b) = s.backings.get_mut(&backing) {
            b.users.remove(&list);
        }
    }
}

/// Reset `list` on the title's allocator `alloc`, committing its new recording association
/// only after native Reset succeeds. No pool lock is held across the driver call.
pub fn reset_list(list: usize, alloc: usize, ty: i32, reset: impl FnOnce(usize) -> HRESULT) -> HRESULT {
    let backing = current(alloc);
    let hr = reset(backing);
    if hr < 0 || !enabled() || alloc == 0 {
        return hr;
    }
    let mut g = STATE.lock().unwrap();
    let s = g.get_or_insert_with(State::default);
    if ty >= 0 {
        s.types.insert(alloc, ty);
    }
    move_list(s, list, backing);
    s.mesh_recordings.remove(&list);
    *s.recordings.entry(list).or_default() += 1;
    hr
}

#[cfg(test)]
pub(super) use recording_backing as recording_backing_for_test;

pub(super) fn recording_backing(list: usize) -> Option<usize> {
    STATE.lock().unwrap().as_ref()?.of_list.get(&list).copied()
}

/// The PC allocator the title's allocator `alloc` stands for now.
pub fn current(alloc: usize) -> usize {
    if !enabled() {
        return alloc;
    }
    STATE.lock().unwrap().as_ref().and_then(|s| s.cur.get(&alloc).copied()).unwrap_or(alloc)
}

/// `list` (of type `ty`) was made recording on the PC allocator `raw` itself
/// (CreateCommandList).
pub fn created_on(list: usize, raw: usize, ty: i32) {
    if !enabled() || raw == 0 {
        return;
    }
    let mut g = STATE.lock().unwrap();
    let s = g.get_or_insert_with(State::default);
    s.types.entry(raw).or_insert(ty);
    move_list(s, list, raw);
    s.mesh_recordings.remove(&list);
    *s.recordings.entry(list).or_default() += 1;
}

/// Diagnostic (#1452): `lists` are about to be submitted to `queue`. A PC list whose previous
/// submit the GPU has not finished must not be submitted again on the PC (debug layer id 553,
/// "the command queue fence has not advanced past previous executions of the command list");
/// a console lets a title do it. Logged (the first 8, then at powers of two) with whether it is
/// the same recording and the same queue. Every submitted PC list is seen (`executed` marks them all).
pub unsafe fn note_resubmits(queue: *mut c_void, lists: &[*mut c_void]) -> usize {
    if !enabled() {
        return 0;
    }
    let mut g = STATE.lock().unwrap();
    let Some(s) = g.as_mut() else { return 0 };
    let mut found = 0;
    for &l in lists {
        let l = l as usize;
        let Some(last) = s.last_submit.get(&l) else { continue };
        if last.fence.GetCompletedValue() >= last.value {
            continue;
        }
        let same_recording = s.recordings.get(&l).copied().unwrap_or(0) == last.recording;
        let same_queue = last.queue == queue as usize;
        found += 1;
        s.resubmits += 1;
        let n = s.resubmits;
        if n <= 8 || n.is_power_of_two() {
            crate::xlog_warn!(
                "D3D12.X: a command list is submitted while its previous submit is still on the GPU ({}, {}; #{n} so far) – the PC does not allow it (debug layer id 553)",
                if same_recording { "the same recording" } else { "a new recording" },
                if same_queue { "the same queue" } else { "another queue" },
            );
        }
    }
    found
}

/// Whether the GPU has not finished the last submit of the PC list `list` (#1452: it must not be
/// submitted again before that; `title_list` records the title's next recording elsewhere).
pub fn in_flight(list: usize) -> bool {
    let g = STATE.lock().unwrap();
    g.as_ref().is_some_and(|s| {
        s.mesh_reserved.contains(&list) || s.last_submit.get(&list)
            .is_some_and(|l| unsafe { l.fence.GetCompletedValue() } < l.value)
    })
}

/// `lists` were just submitted to `queue`.
pub unsafe fn executed(queue: *mut c_void, lists: &[*mut c_void]) {
    let _ = mark_executed(queue, lists);
}

unsafe fn mark_executed(queue: *mut c_void, lists: &[*mut c_void]) -> Option<MeshCompletion> {
    if !enabled() || queue.is_null() {
        return None;
    }
    // (every submit is marked, also of lists this module does not track – the PC lists behind
    // console lists – so note_resubmits sees them all)
    let mut g = STATE.lock().unwrap();
    let s = g.get_or_insert_with(State::default);
    let q = ID3D12CommandQueue::from_raw_borrowed(&queue)?;
    let key = queue as usize;
    if !s.fences.contains_key(&key) {
        let mut dev: Option<ID3D12Device> = None;
        let f = q.GetDevice(&mut dev).ok().and(dev).and_then(|d| d.CreateFence::<ID3D12Fence>(0, D3D12_FENCE_FLAG_NONE).ok())?;
        s.fences.insert(key, (f, 0));
    }
    let (fence, value) = {
        let (f, v) = s.fences.get_mut(&key).unwrap();
        let next = v.checked_add(1).filter(|&v| v != u64::MAX)?;
        q.Signal(&*f, next).ok()?;
        *v = next;
        (f.clone(), *v)
    };
    for &l in lists {
        if let Some(&b) = s.of_list.get(&(l as usize)) {
            s.backings.entry(b).or_default().marks.insert(key, (fence.clone(), value));
        }
        let recording = s.recordings.get(&(l as usize)).copied().unwrap_or(0);
        s.last_submit.insert(l as usize, LastSubmit { queue: key, fence: fence.clone(), value, recording });
    }
    Some(MeshCompletion { fence, value })
}

fn fence_finished(done: u64, value: u64) -> bool {
    done != u64::MAX && done >= value
}

/// Completion of the reservation protects the scratch of the exact submit, even if a Reset
/// or Release removes the recording from the native-list map meanwhile.
struct MeshReservation {
    progress: Vec<MeshRecording>,
    previous: Vec<(usize, MeshCompletion)>,
}

struct MeshRecording {
    list: usize,
    backing: usize,
    progress: Arc<MeshProgress>,
}

#[derive(Debug)]
enum MeshReserveError {
    UnknownBacking,
    Timeout,
}

fn reserve_mesh(lists: &[*mut c_void], until: Instant) -> Result<MeshReservation, MeshReserveError> {
    let mut g = STATE.lock().unwrap();
    loop {
        let s = g.get_or_insert_with(State::default);
        if lists.iter().all(|&list| !s.mesh_reserved.contains(&(list as usize))) {
            let progress = lists.iter().filter_map(|&list| {
                let list = list as usize;
                s.mesh_recordings.get(&list).map(|p| {
                    let backing = s.of_list.get(&list).copied().ok_or(MeshReserveError::UnknownBacking)?;
                    Ok(MeshRecording { list, backing, progress: p.clone() })
                })
            }).collect::<Result<Vec<_>, MeshReserveError>>()?;
            let previous = progress.iter().filter_map(|r| {
                s.last_submit.get(&r.list).map(|mark| (r.list, MeshCompletion {
                    fence: mark.fence.clone(), value: mark.value,
                }))
            }).collect();
            for recording in &progress {
                recording.progress.submission.lock().unwrap().reserved = true;
                s.mesh_reserved.insert(recording.list);
                s.backings.get_mut(&recording.backing).unwrap().mesh_pending += 1;
            }
            return Ok(MeshReservation { progress, previous });
        }
        let left = until.saturating_duration_since(Instant::now());
        if left.is_zero() {
            return Err(MeshReserveError::Timeout);
        }
        (g, _) = MESH_SUBMIT_AVAILABLE.wait_timeout(g, left).unwrap();
    }
}

fn stop_mesh_submission(reason: String, status: u32) -> ! {
    crate::xlog_error!("D3D12.X: two-pass mesh submit stopped: {reason}");
    super::real::stop_if_device_removed();
    crate::report::write_final(Some(reason));
    crate::log::flush();
    std::process::exit(status as i32);
}

unsafe fn submit_mesh_chunk(
    queue: *mut c_void, lists: &[*mut c_void],
    submit: &mut impl FnMut(&[*mut c_void]),
    after: &mut impl FnMut(&[*mut c_void]),
) {
    let until = Instant::now() + Duration::from_secs(2);
    let reservation = reserve_mesh(lists, until).unwrap_or_else(|error| {
        let (reason, status) = match error {
            MeshReserveError::UnknownBacking => ("a mesh recording lost its native allocator", 0x8000_4005),
            MeshReserveError::Timeout => ("could not reserve the mesh recordings in time", 0x8007_05b4),
        };
        stop_mesh_submission(format!("queue {queue:p}: {reason}"), status)
    });
    for (list, mark) in reservation.previous {
        loop {
            let done = mark.fence.GetCompletedValue();
            if done == u64::MAX {
                stop_mesh_submission(format!("device removed waiting for mesh list {list:#x}"), 0x887a_0005);
            }
            if fence_finished(done, mark.value) {
                break;
            }
            let left = until.saturating_duration_since(Instant::now());
            if left.is_zero() {
                stop_mesh_submission(format!("mesh list {list:#x} fence has {done:#x}, expected {:#x}", mark.value), 0x8007_05b4);
            }
            std::thread::sleep(left.min(Duration::from_millis(1)));
        }
    }
    submit(lists);
    if reservation.progress.is_empty() {
        executed(queue, lists);
        after(lists);
        return;
    }
    let completion = mark_executed(queue, lists).unwrap_or_else(|| {
        stop_mesh_submission(format!("queue {queue:p} did not accept a mesh completion signal"), 0x8000_4005)
    });
    after(lists);
    let mut g = STATE.lock().unwrap();
    let state = g.as_mut().unwrap();
    finish_mesh_marks(state, reservation.progress, queue as usize, &completion);
    drop(g);
    MESH_SUBMIT_AVAILABLE.notify_all();
}

fn finish_mesh_marks(state: &mut State, recordings: Vec<MeshRecording>, queue: usize, mark: &MeshCompletion) {
    for recording in recordings {
        let backing = state.backings.entry(recording.backing).or_default();
        backing.mesh_pending = backing.mesh_pending.saturating_sub(1);
        // Another list on this backing may have accepted a later signal while this
        // submission was in its post-submit callbacks.
        backing.marks.entry(queue).and_modify(|(fence, value)| {
            if fence.as_raw() == mark.fence.as_raw() {
                *value = (*value).max(mark.value);
            }
        }).or_insert_with(|| (mark.fence.clone(), mark.value));
        let mut submit = recording.progress.submission.lock().unwrap();
        submit.completion = Some(mark.clone());
        submit.reserved = false;
        state.mesh_reserved.remove(&recording.list);
    }
}

/// Leave ordinary submissions alone. Actual mesh recordings cannot reuse scratch while their
/// prior native execution is live, including across queues or twice in one caller's batch.
pub(super) unsafe fn submit_with_mesh_tracking(
    queue: *mut c_void, lists: &[*mut c_void],
    mut submit: impl FnMut(&[*mut c_void]), mut after: impl FnMut(&[*mut c_void]),
) {
    let mesh: HashSet<_> = if super::bind::mesh_indirect::on() {
        let g = STATE.lock().unwrap();
        lists.iter().filter_map(|&list| {
            let list = list as usize;
            g.as_ref().is_some_and(|s| s.mesh_recordings.contains_key(&list) || s.mesh_reserved.contains(&list)).then_some(list)
        }).collect()
    } else {
        HashSet::new()
    };
    if mesh.is_empty() {
        submit(lists);
        executed(queue, lists);
        after(lists);
        return;
    }
    let mut seen = HashSet::new();
    let mut first = 0;
    for (i, &list) in lists.iter().enumerate() {
        let list = list as usize;
        if mesh.contains(&list) && !seen.insert(list) {
            submit_mesh_chunk(queue, &lists[first..i], &mut submit, &mut after);
            first = i;
            seen.clear();
            seen.insert(list);
        }
    }
    submit_mesh_chunk(queue, &lists[first..], &mut submit, &mut after);
}

/// Whether the GPU may still run a recording made on the title's allocator `alloc`.
pub(super) unsafe fn allocator_in_flight(alloc: usize) -> bool {
    let g = STATE.lock().unwrap();
    let Some(s) = g.as_ref() else { return false };
    let mine = s.cur.get(&alloc).into_iter().chain(s.aside.get(&alloc).into_iter().flatten()).chain([&alloc]);
    let busy = mine.filter_map(|b| s.backings.get(b))
        .any(|b| b.mesh_pending != 0 || b.marks.values().any(|(f, v)| f.GetCompletedValue() < *v));
    busy
}

/// The title's allocator `alloc` is gone (its last Release): nothing of it stays here – a later
/// allocator may get its address, and must not go on with this one's PC allocators. The ones
/// made here go with it (those lists still record on are left alone: not freed, not tracked).
pub unsafe fn forget(alloc: usize) {
    let made: Vec<usize> = {
        let mut g = STATE.lock().unwrap();
        let Some(s) = g.as_mut() else { return };
        s.types.remove(&alloc);
        let mut mine: Vec<usize> = s.aside.remove(&alloc).unwrap_or_default();
        mine.extend(s.cur.remove(&alloc));
        mine.push(alloc);
        let mut made = Vec::new();
        for b in mine {
            let free = s.backings.get(&b).is_none_or(|x| x.free());
            if let Some(gone) = s.backings.remove(&b) {
                for l in gone.users {
                    s.of_list.remove(&l);
                }
            }
            if b != alloc && free {
                made.push(b);
            }
        }
        made
    };
    // (outside the lock: their Release comes back here)
    for b in made {
        drop(ID3D12CommandAllocator::from_raw(b as *mut c_void));
    }
}

/// The title resets its allocator `alloc`. `None`: reset it as it is (nothing holds a
/// recording on it, or this is off). `Some(hr)`: done here – the PC allocator in use was put
/// aside with its recordings and another one took its place.
pub unsafe fn title_reset(alloc: *mut c_void) -> Option<HRESULT> {
    if !enabled() {
        return None;
    }
    let mut g = STATE.lock().unwrap();
    let s = g.get_or_insert_with(State::default);
    title_reset_in(s, alloc, |raw| super::real::reset_pc_allocator(raw))
}

/// The allocator transition, with the PC Reset supplied so failure/retry needs no GPU in tests.
unsafe fn title_reset_in(s: &mut State, alloc: *mut c_void, mut reset: impl FnMut(*mut c_void) -> HRESULT) -> Option<HRESULT> {
    let a = alloc as usize;
    let c = s.cur.get(&a).copied().unwrap_or(a);
    if s.backings.get(&c).is_none_or(|b| b.free()) {
        if let Some(b) = s.backings.get_mut(&c) {
            b.marks.clear();
        }
        if c == a {
            return None;
        }
        return Some(reset(c as *mut c_void));
    }
    let ty = *s.types.get(&a)?;
    // one put aside earlier that nothing uses any more, or a new one
    let reuse = s.aside.get(&a).and_then(|v| v.iter().copied().find(|f| s.backings.get(f).is_none_or(|b| b.free())));
    let next = match reuse {
        Some(f) => {
            // Keep ownership until Reset succeeds so a failure can be retried.
            let hr = reset(f as *mut c_void);
            if hr < 0 {
                return Some(hr);
            }
            s.aside.get_mut(&a).unwrap().retain(|&x| x != f);
            if let Some(b) = s.backings.get_mut(&f) {
                b.marks.clear();
            }
            f
        }
        None => {
            let this = ID3D12CommandAllocator::from_raw_borrowed(&alloc)?;
            let mut dev: Option<ID3D12Device> = None;
            this.GetDevice(&mut dev).ok()?;
            let made: ID3D12CommandAllocator = dev?.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE(ty)).ok()?;
            let raw = made.into_raw();
            // (made through the title's device it is adopted already)
            if !super::real::is_adopted(raw) {
                super::real::adopt(raw, super::real::Kind::CommandAllocator);
            }
            s.made += 1;
            raw as usize
        }
    };
    s.aside.entry(a).or_default().push(c);
    s.cur.insert(a, next);
    s.swaps += 1;
    if s.swaps.is_power_of_two() {
        let aside: usize = s.aside.values().map(|v| v.len()).sum();
        crate::xlog_info!(
            "D3D12.X: {} allocator resets so far with recordings still held or on the GPU – another PC allocator took its place ({} made, {aside} put aside now)",
            s.swaps,
            s.made
        );
    }
    Some(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_mesh_completion_is_unavailable_while_the_recording_is_reserved() {
        let progress = MeshProgress::default();
        assert!(matches!(progress.retirement_state(), Some(None)));
        progress.submission.lock().unwrap().reserved = true;
        assert!(progress.retirement_state().is_none());
        assert!(fence_finished(14, 14));
        assert!(!fence_finished(13, 14));
        assert!(!fence_finished(u64::MAX, 14));
    }

    #[test]
    fn a_mesh_reservation_keeps_its_backing_through_native_address_reuse() {
        use std::cell::Cell;
        use std::mem::{offset_of, size_of};

        #[repr(C)]
        struct Fence {
            vtable: *const usize,
            value: Cell<u64>,
            refs: Cell<u32>,
        }
        unsafe extern "system" fn completed(this: *mut c_void) -> u64 {
            (*this.cast::<Fence>()).value.get()
        }
        unsafe extern "system" fn add(this: *mut c_void) -> u32 {
            let refs = &(*this.cast::<Fence>()).refs;
            refs.set(refs.get() + 1);
            refs.get()
        }
        unsafe extern "system" fn release(this: *mut c_void) -> u32 {
            let refs = &(*this.cast::<Fence>()).refs;
            refs.set(refs.get() - 1);
            refs.get()
        }
        extern "system" fn unexpected_call() { std::process::abort(); }

        const SLOT: usize = size_of::<usize>();
        let mut vtable = [unexpected_call as usize; size_of::<ID3D12Fence_Vtbl>() / SLOT];
        vtable[offset_of!(ID3D12Fence_Vtbl, GetCompletedValue) / SLOT] = completed as usize;
        vtable[offset_of!(ID3D12Fence_Vtbl, base__.base__.base__.base__.AddRef) / SLOT] = add as usize;
        vtable[offset_of!(ID3D12Fence_Vtbl, base__.base__.base__.base__.Release) / SLOT] = release as usize;
        let mut fence = Fence { vtable: vtable.as_ptr(), value: Cell::new(0), refs: Cell::new(1) };
        let mark = MeshCompletion {
            fence: unsafe { ID3D12Fence::from_raw(std::ptr::from_mut(&mut fence).cast()) },
            value: 42,
        };
        let newer = MeshCompletion { fence: mark.fence.clone(), value: 43 };
        let mut ids = Box::new([0u8; 4]);
        let list = ids.as_mut_ptr() as usize;
        let (old_alloc, new_alloc, other) = (list + 1, list + 2, list + 3);
        created_on(other, old_alloc, D3D12_COMMAND_LIST_TYPE_DIRECT.0);
        let newer_recording = track_mesh_recording(other).unwrap();
        created_on(list, old_alloc, D3D12_COMMAND_LIST_TYPE_DIRECT.0);
        let original = track_mesh_recording(list).unwrap();
        let submit = [list as *mut c_void];
        let reservation = reserve_mesh(&submit, Instant::now()).unwrap();

        assert!(original.retirement_state().is_none(), "submit has no accepted completion yet");
        assert!(in_flight(list), "do not re-use a PC list while the driver is accepting it");
        assert!(reserve_mesh(&submit, Instant::now()).is_err());
        created_on(list, new_alloc, D3D12_COMMAND_LIST_TYPE_DIRECT.0);
        let replacement = track_mesh_recording(list).unwrap();
        end_mesh_recording(list, &original);
        let newer_submission = reserve_mesh(&[other as *mut c_void], Instant::now()).unwrap();
        {
            let mut g = STATE.lock().unwrap();
            finish_mesh_marks(g.as_mut().unwrap(), newer_submission.progress, 1, &newer);
        }
        forget_list(other);
        let (reserved, new_unmarked, old_mark, pending) = {
            let mut g = STATE.lock().unwrap();
            let s = g.as_mut().unwrap();
            let reserved = !unsafe { s.backings[&old_alloc].free() };
            finish_mesh_marks(s, reservation.progress, 1, &mark);
            let old = &s.backings[&old_alloc];
            (reserved, s.backings[&new_alloc].marks.is_empty(), old.marks.get(&1).map(|m| m.1), !unsafe { old.free() })
        };
        let original_mark = original.retirement_state().flatten().map(|m| m.value);
        let replacement_unmarked = matches!(replacement.retirement_state(), Some(None));
        let value = mark.value;
        let newer_value = newer.value;
        let newer_mark = newer_recording.retirement_state().flatten().map(|m| m.value);
        fence.value.set(value);
        let newer_still_pending = !unsafe { STATE.lock().unwrap().as_ref().unwrap().backings[&old_alloc].free() };
        fence.value.set(newer_value);
        let completed = unsafe { STATE.lock().unwrap().as_ref().unwrap().backings[&old_alloc].free() };

        // Release the synthetic COM object from the globals before checking the result.
        forget_list(list);
        unsafe { forget(old_alloc); forget(new_alloc); }
        drop(original);
        drop(newer_recording);
        drop(mark);
        drop(newer);
        assert!(reserved, "old backing must remain reserved");
        assert!(pending, "the old backing now waits for its native fence");
        assert!(new_unmarked && replacement_unmarked);
        assert_eq!(old_mark, Some(newer_value), "the older callback finished last on the same fence");
        assert_eq!(original_mark, Some(value));
        assert_eq!(newer_mark, Some(newer_value));
        assert!(newer_still_pending, "the newer native submit can still use the old allocator");
        assert!(completed);
        assert_eq!(fence.refs.get(), 0);
    }

    #[test]
    fn a_spare_whose_reset_fails_stays_available_for_retry() {
        let (alloc, spare, list) = (0x1000usize, 0x2000usize, 0x3000usize);
        let mut s = State::default();
        s.types.insert(alloc, D3D12_COMMAND_LIST_TYPE_DIRECT.0);
        move_list(&mut s, list, alloc);
        s.aside.insert(alloc, vec![spare]);
        s.backings.insert(spare, Backing::default());
        let mut calls = Vec::new();

        for _ in 0..2 {
            let hr = unsafe {
                title_reset_in(&mut s, alloc as *mut c_void, |raw| {
                    calls.push(raw as usize);
                    crate::util::E_FAIL
                })
            };
            assert_eq!(hr, Some(crate::util::E_FAIL));
            assert_eq!(s.aside.get(&alloc), Some(&vec![spare]), "the failed candidate is still owned and reusable");
            assert!(!s.cur.contains_key(&alloc), "the current backing did not change");
            assert_eq!(s.of_list.get(&list), Some(&alloc));
            assert!(s.backings[&alloc].users.contains(&list), "the old recording is still held");
            assert_eq!((s.made, s.swaps), (0, 0));
        }

        let hr = unsafe {
            title_reset_in(&mut s, alloc as *mut c_void, |raw| {
                calls.push(raw as usize);
                0
            })
        };
        assert_eq!(hr, Some(0));
        assert_eq!(calls, vec![spare; 3], "all attempts used the same spare without creating another");
        assert_eq!(s.cur.get(&alloc), Some(&spare));
        assert_eq!(s.aside.get(&alloc), Some(&vec![alloc]), "only the successful reset switches backings");
        assert_eq!(s.of_list.get(&list), Some(&alloc), "the earlier recording stays on its original backing");
        assert_eq!((s.made, s.swaps), (0, 1));
    }

    /// Final-release cleanup drops only the recording's hold: another live list and a pending
    /// GPU fence each still prevent reuse, then completion makes the backing reusable.
    #[test]
    fn released_lists_leave_only_live_recordings_and_gpu_work_holding_the_backing() {
        use std::cell::Cell;
        use std::mem::{offset_of, size_of};

        #[repr(C)]
        struct Fence {
            vtable: *const usize,
            value: Cell<u64>,
            releases: Cell<u32>,
        }
        unsafe extern "system" fn completed(this: *mut c_void) -> u64 {
            (*this.cast::<Fence>()).value.get()
        }
        unsafe extern "system" fn release(this: *mut c_void) -> u32 {
            let f = &*this.cast::<Fence>();
            f.releases.set(f.releases.get() + 1);
            0
        }
        extern "system" fn unexpected_call() { std::process::abort(); }

        const SLOT: usize = size_of::<usize>();
        let mut vtable = [unexpected_call as usize; size_of::<ID3D12Fence_Vtbl>() / SLOT];
        vtable[offset_of!(ID3D12Fence_Vtbl, GetCompletedValue) / SLOT] = completed as usize;
        vtable[offset_of!(ID3D12Fence_Vtbl, base__.base__.base__.base__.Release) / SLOT] = release as usize;
        let mut fence = Fence { vtable: vtable.as_ptr(), value: Cell::new(0), releases: Cell::new(0) };
        let mut ids = Box::new([0u8; 3]);
        let alloc = ids.as_mut_ptr() as usize;
        let (first, second) = (alloc + 1, alloc + 2);
        created_on(first, alloc, D3D12_COMMAND_LIST_TYPE_DIRECT.0);
        created_on(second, alloc, D3D12_COMMAND_LIST_TYPE_DIRECT.0);
        {
            let mut g = STATE.lock().unwrap();
            let b = g.as_mut().unwrap().backings.get_mut(&alloc).unwrap();
            // The map owns one synthetic fence reference, dropped by forget below.
            let f = unsafe { ID3D12Fence::from_raw(std::ptr::from_mut(&mut fence).cast()) };
            b.marks.insert(1, (f, 7));
        }

        super::super::list_x::on_release(first);
        let first_removed = {
            let g = STATE.lock().unwrap();
            let s = g.as_ref().unwrap();
            !s.of_list.contains_key(&first) && s.of_list.get(&second) == Some(&alloc)
                && s.backings[&alloc].users == HashSet::from([second])
                && !unsafe { s.backings[&alloc].free() }
        };
        super::super::list_x::on_release(second);
        super::super::list_x::on_release(first); // duplicate cleanup is harmless
        let pending_still_held = {
            let g = STATE.lock().unwrap();
            let s = g.as_ref().unwrap();
            !s.of_list.contains_key(&second) && s.backings[&alloc].users.is_empty()
                && s.backings[&alloc].marks.len() == 1 && !unsafe { s.backings[&alloc].free() }
        };
        fence.value.set(7);
        let reusable = unsafe { STATE.lock().unwrap().as_ref().unwrap().backings[&alloc].free() };
        // Clean up the synthetic COM pointer before assertions, including on the old code.
        unsafe { forget(alloc) };
        assert!(first_removed, "another live recording must remain while the released list goes");
        assert!(pending_still_held, "final list Release must retain the submitted-work fence");
        assert!(reusable, "a released list cannot keep a completed backing busy");
        assert_eq!(fence.releases.get(), 1);
    }

    /// A list recorded once and submitted again after the title reset its allocator: the
    /// recording is still there (the device is not removed), the title's allocator goes on with
    /// another PC allocator, and the first one is used again once the list has moved on.
    #[test]
    fn a_recording_outlives_the_reset_of_the_titles_allocator() {
        unsafe {
            let Ok(dev) = crate::gpu::agility::create_device(None, windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0) else {
                return crate::gpu::recomp::runner::skip_test("no GPU");
            };
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).unwrap();
            let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
            let a = alloc.as_raw();
            if !super::super::real::is_adopted(a) {
                super::super::real::adopt(a, super::super::real::Kind::CommandAllocator);
            }
            let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).unwrap();
            let l = list.as_raw();
            created_on(l as usize, a as usize, D3D12_COMMAND_LIST_TYPE_DIRECT.0);
            list.Close().unwrap();
            let lists = [Some(list.cast::<ID3D12CommandList>().unwrap())];
            let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let done = |v: u64| {
                queue.Signal(&fence, v).unwrap();
                let start = std::time::Instant::now();
                while fence.GetCompletedValue() < v && start.elapsed().as_secs() < 10 {
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
                assert!(fence.GetCompletedValue() >= v, "the queue finishes");
            };

            queue.ExecuteCommandLists(&lists);
            executed(queue.as_raw(), &[l]);
            done(1);

            // the title resets its allocator (through the hook: `real::alloc_reset`) …
            alloc.Reset().unwrap();
            let second = current(a as usize);
            assert_ne!(second, a as usize, "the list still holds its recording: another PC allocator takes the place");
            // … and submits the recording again
            queue.ExecuteCommandLists(&lists);
            executed(queue.as_raw(), &[l]);
            done(2);
            assert!(dev.GetDeviceRemovedReason().is_ok(), "the recording was still there");

            // the list is reset on the title's allocator: it records on the PC allocator in use
            let second_alloc = ID3D12CommandAllocator::from_raw_borrowed(&(second as *mut c_void)).unwrap().clone();
            assert_eq!(reset_list(l as usize, a as usize, D3D12_COMMAND_LIST_TYPE_DIRECT.0, |pc| {
                assert_eq!(pc, second);
                list.Reset(&second_alloc, None).map_or_else(|e| e.code().0, |_| 0)
            }), 0);
            list.Close().unwrap();
            queue.ExecuteCommandLists(&lists);
            executed(queue.as_raw(), &[l]);
            done(3);

            // the next reset: the first PC allocator holds nothing now and is used again
            alloc.Reset().unwrap();
            assert_eq!(current(a as usize), a as usize, "the allocator put aside is free again");
            assert!(dev.GetDeviceRemovedReason().is_ok());

            // the title's allocator goes while another PC allocator stands for it: nothing of
            // it stays (the next allocator at its address is its own)
            assert_eq!(reset_list(l as usize, a as usize, D3D12_COMMAND_LIST_TYPE_DIRECT.0, |pc| {
                assert_eq!(pc, a as usize);
                list.Reset(&alloc, None).map_or_else(|e| e.code().0, |_| 0)
            }), 0);
            list.Close().unwrap();
            alloc.Reset().unwrap();
            assert_ne!(current(a as usize), a as usize);
            drop(lists);
            drop(list);
            drop(alloc);
            assert_eq!(current(a as usize), a as usize, "a released allocator leaves nothing behind");
            assert!(!STATE.lock().unwrap().as_ref().unwrap().types.contains_key(&(a as usize)));
        }
    }

    /// A list submitted again while its previous submit is still on the GPU is counted (and
    /// logged); once that submit is done, or after the list is forgotten, it is not.
    #[test]
    fn a_resubmit_while_the_previous_submit_runs_is_noticed() {
        unsafe {
            let Ok(dev) = crate::gpu::agility::create_device(None, windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0) else {
                return crate::gpu::recomp::runner::skip_test("no GPU");
            };
            // a fence the GPU has not reached stands for a submit still running
            let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let mut ids = Box::new([0u8; 2]);
            let (alloc, list) = (ids.as_mut_ptr() as usize, ids.as_mut_ptr() as usize + 1);
            let queue = 0x10usize as *mut c_void;
            created_on(list, alloc, D3D12_COMMAND_LIST_TYPE_DIRECT.0);
            {
                let mut g = STATE.lock().unwrap();
                let s = g.as_mut().unwrap();
                let recording = s.recordings[&list];
                s.last_submit.insert(list, LastSubmit { queue: queue as usize, fence: fence.clone(), value: 1, recording });
            }
            let while_running = note_resubmits(queue, &[list as *mut c_void]);
            fence.Signal(1).unwrap();
            let after_done = note_resubmits(queue, &[list as *mut c_void]);
            {
                // running again, then the list goes: nothing is left to compare with
                let mut g = STATE.lock().unwrap();
                g.as_mut().unwrap().last_submit.get_mut(&list).unwrap().value = 2;
            }
            forget_list(list);
            let after_forget = note_resubmits(queue, &[list as *mut c_void]);
            forget(alloc);
            assert_eq!(while_running, 1, "a submit while the previous one runs must be noticed");
            assert_eq!(after_done, 0, "a finished previous submit is not a resubmit");
            assert_eq!(after_forget, 0, "a forgotten list has no previous submit");
        }
    }
}
