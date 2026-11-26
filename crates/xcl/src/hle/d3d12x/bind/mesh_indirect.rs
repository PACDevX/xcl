//! Single-command subset of XCL PR #2092 (Cattn and PracticalStudios).
//! The title's GPU-written arguments and the existing two-pass MS ABI stay in their
//! original memory windows. Per-recording D3D12 buffers carry the rewritten commands.

use std::collections::{HashMap, HashSet};
use std::ffi::c_void;
use std::mem::ManuallyDrop;
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc, Mutex, OnceLock,
};

use std::time::Instant;

use windows::core::{Interface, GUID};
use windows::Win32::Graphics::{Direct3D12::*, Dxgi::Common::DXGI_SAMPLE_DESC};

use crate::gpu::recomp::{self, IO_UMA_BASE_HI, KERNEL_SGPRS};

use super::super::{alloc_pool, console_heap, console_list, list_x, real, title_textures};
use super::with;

const STRIDE: u32 = 12;
const RECORD_BYTES: u64 = 32 << 20;
const SCRATCH_BYTES: u64 = 256;
const PREP_GROUPS: u32 = 4;
const BUDGET_BYTES: u64 = 1024 << 20;
const BUDGET_EXCEEDED: &str = "live two-pass buffers exceed the resident budget";
const RECORDING_BYTES: u64 = RECORD_BYTES + D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT as u64;
const COMPANION: GUID = GUID::from_u128(0x344dad60_78d3_4c19_8654_eed01aa91897);

pub(in crate::hle::d3d12x) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("XCL_MESH_INDIRECT_SINGLE").is_ok_and(|v| v == "1"))
}

pub(in crate::hle::d3d12x) fn eligible_signature(stride: u32, root: usize, args: &[[u32; 4]]) -> bool {
    stride == STRIDE && root == 0
        && matches!(args, [a] if a[0] == D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH_MESH.0 as u32)
}

pub(super) fn eligible_call(kind: &str, max: u32, count: usize) -> bool {
    kind == "MS" && max == 1 && count == 0
}

/// The original PC signature owns its companion through D3D12 private data.
pub(in crate::hle::d3d12x) unsafe fn attach_companion(device: *mut c_void, sig: *mut c_void, node: u32) {
    let result = (|| -> windows::core::Result<()> {
        let dev = ID3D12Device::from_raw_borrowed(&device).ok_or(windows::core::Error::empty())?;
        let original = ID3D12CommandSignature::from_raw_borrowed(&sig).ok_or(windows::core::Error::empty())?;
        let argument = D3D12_INDIRECT_ARGUMENT_DESC { Type: D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH, ..Default::default() };
        let desc = D3D12_COMMAND_SIGNATURE_DESC {
            ByteStride: STRIDE, NumArgumentDescs: 1, pArgumentDescs: &argument, NodeMask: node,
        };
        let mut dispatch: Option<ID3D12CommandSignature> = None;
        dev.CreateCommandSignature(&desc, None::<&ID3D12RootSignature>, &mut dispatch)?;
        original.SetPrivateDataInterface(&COMPANION, dispatch.as_ref().ok_or(windows::core::Error::empty())?)
    })();
    if let Err(error) = result {
        warn(format!("create the DISPATCH companion: {error}"));
    }
}

unsafe fn companion(sig: &ID3D12CommandSignature) -> Option<ID3D12CommandSignature> {
    let mut size = std::mem::size_of::<*mut c_void>() as u32;
    let mut raw: *mut c_void = std::ptr::null_mut();
    sig.GetPrivateData(&COMPANION, &mut size, Some((&mut raw as *mut *mut c_void).cast())).ok()?;
    // GetPrivateData AddRefs an interface installed by SetPrivateDataInterface.
    (!raw.is_null() && size as usize == std::mem::size_of::<*mut c_void>())
        .then(|| ID3D12CommandSignature::from_raw(raw))
}

struct Budget<'a> {
    used: &'a AtomicU64,
}

impl<'a> Budget<'a> {
    fn acquire(used: &'a AtomicU64) -> Option<Self> {
        used.try_update(Ordering::AcqRel, Ordering::Relaxed, |now| {
            now.checked_add(RECORDING_BYTES).filter(|&next| next <= BUDGET_BYTES)
        }).ok()?;
        Some(Self { used })
    }
}

impl Drop for Budget<'_> {
    fn drop(&mut self) {
        self.used.fetch_sub(RECORDING_BYTES, Ordering::Release);
    }
}

struct Preprocess {
    pso: ID3D12PipelineState,
    root: ID3D12RootSignature,
}

struct Buffers {
    _device: ID3D12Device,
    records: ID3D12Resource,
    args: ID3D12Resource,
    prep: Arc<Preprocess>,
    _budget: Budget<'static>,
}

#[derive(Default)]
pub(super) struct Recording {
    buffers: Option<Arc<Buffers>>,
    progress: Option<Arc<alloc_pool::MeshProgress>>,
    arena: crate::gpu::uma::Arena,
    metadata: Vec<usize>,
    signatures: HashMap<usize, (ID3D12CommandSignature, ID3D12CommandSignature)>,
    inputs: HashMap<usize, ID3D12Resource>,
    pipelines: HashMap<usize, (ID3D12PipelineState, Arc<real::TwoPassPso>)>,
    roots: HashMap<usize, ID3D12RootSignature>,
}

impl Drop for Recording {
    fn drop(&mut self) {
        if !self.metadata.is_empty() {
            if let Some(uma) = crate::gpu::uma::get() {
                for &block in &self.metadata {
                    uma.free(block);
                }
            }
        }
    }
}

static USED_BYTES: AtomicU64 = AtomicU64::new(0);
static RETIRED: Mutex<Vec<Recording>> = Mutex::new(Vec::new());

pub(super) fn retire(list: usize, recording: Recording) {
    if let Some(progress) = &recording.progress {
        alloc_pool::end_mesh_recording(list, progress);
        RETIRED.lock().unwrap().push(recording);
    }
}

fn marker_completed(value: u64, done: u64) -> bool {
    done != u64::MAX && done >= value
}

fn take_retired<T>(pending: &mut Vec<T>, mut ready: impl FnMut(&T) -> bool) -> Vec<T> {
    let mut released = Vec::new();
    let mut i = 0;
    while i < pending.len() {
        if ready(&pending[i]) {
            released.push(pending.swap_remove(i));
        } else {
            i += 1;
        }
    }
    released
}

/// Recover budget only from ended recordings. A mesh recording can still be executed again
/// after Close, so its resources and argument tables are not retired until Reset or final Release.
fn reap_retired_for_budget() {
    static START: OnceLock<Instant> = OnceLock::new();
    static LAST: AtomicU64 = AtomicU64::new(0);
    let now = START.get_or_init(Instant::now).elapsed().as_millis() as u64 + 1;
    if LAST.try_update(Ordering::Relaxed, Ordering::Relaxed, |last| {
        (last == 0 || now.saturating_sub(last) >= 2).then_some(now)
    }).is_err() {
        return;
    }
    let pending: Vec<_> = {
        let Ok(retired) = RETIRED.try_lock() else { return };
        retired.iter().filter_map(|r| r.progress.clone()).collect()
    };
    let mut sampled = HashMap::new();
    let ready: HashSet<_> = pending.iter().filter_map(|progress| {
        let completion = progress.retirement_state()?;
        let done = completion.is_none_or(|mark| {
            let value = *sampled.entry(mark.fence.as_raw() as usize)
                .or_insert_with(|| unsafe { mark.fence.GetCompletedValue() });
            marker_completed(mark.value, value)
        });
        done.then_some(Arc::as_ptr(progress))
    }).collect();
    if ready.is_empty() {
        return;
    }
    let released = {
        let Ok(mut retired) = RETIRED.try_lock() else { return };
        take_retired(&mut retired, |r| {
            r.progress.as_ref().is_some_and(|p| ready.contains(&Arc::as_ptr(p)))
        })
    };
    let buffers = released.len();
    drop(released);
    if buffers != 0 {
        static REAPED: AtomicU64 = AtomicU64::new(0);
        let n = REAPED.fetch_add(1, Ordering::Relaxed) + 1;
        if n <= 4 || n.is_power_of_two() {
            crate::xlog_info!("D3D12.X: mesh scratch reclamation #{n}: released {buffers} ended recordings");
        }
    }
}

fn record_groups(bytes: u64, split: u32) -> Option<u32> {
    let x = (RECORD_BYTES.checked_div(bytes)?).min(65_535u64.checked_div(split as u64)?);
    (x != 0).then_some(x as u32)
}

/// #2092's capacity and Y/Z folding, specialized to one 12-byte command and two copies.
/// The translated record ABI and pass-through mesh shader both index by group X.
/// The actual MS can return without writing a record header. Clear the admitted headers
/// here, so an empty group cannot inherit a preceding draw's vertices and primitives.
const PREPROCESS_HLSL: &str = r#"RWByteAddressBuffer io : register(u0);
RWByteAddressBuffer records : register(u1);
RWByteAddressBuffer src : register(u2);
RWByteAddressBuffer dst : register(u3);
[numthreads(64, 1, 1)]
void main(uint tid : SV_DispatchThreadID) {
    uint cap = io.Load(0), split = io.Load(4);
    uint recordBytes = io.Load(8), threads = io.Load(12);
    uint3 g = src.Load3(io.Load(16));
    uint n = 0;
    if (g.x != 0 && g.y != 0 && g.z != 0) {
        n = min(g.x, cap);
        n = g.y > cap / n ? cap : n * g.y;
        n = g.z > cap / n ? cap : n * g.z;
    }
    if (tid == 0) {
        dst.Store3(0, uint3(n, 1, 1));
        dst.Store3(12, uint3(n * split, 1, 1));
    }
    for (uint i = tid; i < n; i += threads) {
        records.Store2(i * recordBytes, uint2(0, 0));
    }
}
"#;

unsafe fn prep_pipeline(dev: &ID3D12Device) -> Result<Arc<Preprocess>, String> {
    static CACHE: Mutex<Option<(ID3D12Device, Arc<Preprocess>)>> = Mutex::new(None);
    let mut cache = CACHE.lock().unwrap();
    if let Some((_, p)) = cache.as_ref().filter(|(d, _)| d == dev) {
        return Ok(p.clone());
    }
    let code = recomp::dxc::compile_cached(PREPROCESS_HLSL, "main", "cs_6_6")?.0;
    let root = recomp::pso::abi_root_signature(dev, false)?;
    let mut desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
        pRootSignature: ManuallyDrop::new(Some(root.clone())),
        CS: D3D12_SHADER_BYTECODE { pShaderBytecode: code.as_ptr().cast(), BytecodeLength: code.len() },
        ..Default::default()
    };
    let pso = dev.CreateComputePipelineState(&desc);
    ManuallyDrop::drop(&mut desc.pRootSignature);
    let p = Arc::new(Preprocess { pso: pso.map_err(|e| format!("preprocess pipeline: {e}"))?, root });
    *cache = Some((dev.clone(), p.clone()));
    Ok(p)
}

unsafe fn create_buffer(dev: &ID3D12Device, width: u64, state: D3D12_RESOURCE_STATES) -> Result<ID3D12Resource, String> {
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: width, Height: 1, DepthOrArraySize: 1, MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        ..Default::default()
    };
    let heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() };
    let mut res = None;
    dev.CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, None, &mut res)
        .map_err(|e| format!("allocate mesh buffer of {width} bytes: {e}"))?;
    res.ok_or_else(|| "mesh buffer was null".to_string())
}

unsafe fn buffers(this: *mut c_void, list: &ID3D12GraphicsCommandList) -> Result<Arc<Buffers>, String> {
    if let Some(buffers) = with(this, |s| s.mesh_indirect.buffers.clone()) {
        return Ok(buffers);
    }
    let budget = Budget::acquire(&USED_BYTES).or_else(|| {
        reap_retired_for_budget();
        Budget::acquire(&USED_BYTES)
    }).ok_or(BUDGET_EXCEEDED)?;
    let mut device = None;
    list.GetDevice(&mut device).map_err(|e| format!("get device: {e}"))?;
    let device: ID3D12Device = device.ok_or("no D3D12 device")?;
    let buffers = title_textures::own_work(|| -> Result<_, String> {
        Ok(Arc::new(Buffers {
            records: create_buffer(&device, RECORD_BYTES, D3D12_RESOURCE_STATE_COMMON)?,
            args: create_buffer(&device, SCRATCH_BYTES, D3D12_RESOURCE_STATE_COMMON)?,
            prep: prep_pipeline(&device)?,
            _device: device,
            _budget: budget,
        }))
    })?;
    with(this, |s| s.mesh_indirect.buffers = Some(buffers.clone()));
    Ok(buffers)
}

unsafe fn bound_uma(bytes: &[u8], this: *mut c_void, uma: &crate::gpu::uma::Uma) -> Result<(usize, u64), String> {
    let at = with(this, |s| {
        let recording = &mut s.mesh_indirect;
        recording.arena.take(uma, bytes.len().max(16), 256, &mut recording.metadata)
    }).ok_or("two-pass unified memory is full")?;
    let va = real::uma_va_of(at as usize).filter(|&va| va != 0).ok_or("two-pass UMA window is not open")?;
    std::ptr::copy_nonoverlapping(bytes.as_ptr(), at, bytes.len());
    Ok((at as usize, va))
}

fn stage_words(
    layout: Option<&console_list::Layout>, args: &[u8], table: u64,
    heaps: [Option<u32>; 2], uma: usize, mesh_compute: bool,
) -> Result<Vec<u32>, String> {
    let empty = console_list::Layout::default();
    let layout = layout.unwrap_or(&empty);
    let mut io = if mesh_compute {
        console_list::user_data_ms(layout, args, table)
    } else {
        console_list::user_data_gfx(layout, args, table)
    };
    for (indexed, base) in layout.mesh_heap_flags.into_iter().zip(heaps) {
        if indexed {
            io.push(base.ok_or("a directly indexed console descriptor heap is not tracked")?);
        }
    }
    io.resize(KERNEL_SGPRS, 0);
    io[IO_UMA_BASE_HI] = (uma >> 32) as u32;
    Ok(io)
}

unsafe fn stage_io(
    this: *mut c_void, uma: &crate::gpu::uma::Uma, layout: Option<&console_list::Layout>,
    args: &[u8], heaps: [Option<u32>; 2], mesh_compute: bool,
) -> Result<u64, String> {
    let inline = layout.map_or(0, |l| if mesh_compute { l.inline_ms } else { l.inline_gfx }) as usize;
    let (table, _) = bound_uma(&args[inline.min(args.len())..], this, uma)?;
    let io = stage_words(layout, args, table as u64, heaps, uma.base(), mesh_compute)?;
    let bytes = io.into_iter().flat_map(u32::to_le_bytes).collect::<Vec<_>>();
    Ok(bound_uma(&bytes, this, uma)?.1)
}

unsafe fn uav_barrier(list: &ID3D12GraphicsCommandList, res: Option<&ID3D12Resource>) {
    let barrier = D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 { UAV: ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
            pResource: ManuallyDrop::new(res.cloned()),
        }) },
    };
    list.ResourceBarrier(std::slice::from_ref(&barrier));
    let mut uav = ManuallyDrop::into_inner(barrier.Anonymous.UAV);
    ManuallyDrop::drop(&mut uav.pResource);
}

fn title_source_offset(base: usize, len: usize, offset: u64) -> Option<usize> {
    let off = usize::try_from(offset).ok()?;
    if off.checked_add(STRIDE as usize)? > len {
        return None;
    }
    base.checked_add(off).filter(|p| p.is_multiple_of(4))
}

type ExecuteIndirect = unsafe extern "system" fn(*mut c_void, usize, u32, usize, u64, usize, u64);

/// None: the title's signature was not the supported shape or its companion could not be made.
pub(super) unsafe fn record(this: *mut c_void, sig: usize, pso: usize, tp: Arc<real::TwoPassPso>, args: usize, offset: u64) -> Option<Result<(), String>> {
    let sig_raw = sig as *mut c_void;
    let original = ID3D12CommandSignature::from_raw_borrowed(&sig_raw)?;
    let dispatch = with(this, |s| s.mesh_indirect.signatures.get(&sig).map(|(_, dispatch)| dispatch.clone()))
        .or_else(|| companion(original))?;
    Some(record_pair(this, original, dispatch, pso, tp, args, offset))
}

unsafe fn record_pair(
    this: *mut c_void, original: &ID3D12CommandSignature, dispatch: ID3D12CommandSignature,
    pso: usize, tp: Arc<real::TwoPassPso>, args: usize, offset: u64,
) -> Result<(), String> {
    if alloc_pool::recording_backing(this as usize).is_none() {
        return Err("requires a tracked native command allocator".into());
    }
    if !list_x::mesh_indirect_can_record(this) {
        return Err("requires a tracked direct list without predication".into());
    }
    let list = ID3D12GraphicsCommandList::from_raw_borrowed(&this).ok_or("no command list")?;
    let uma = crate::gpu::uma::get().ok_or("no UMA")?;
    let windows = real::uma_window_vas();
    let first = windows.first().copied().filter(|&v| v != 0).ok_or("GPU window 0 not open")?;
    if recomp::pso::heap_windows() && console_heap::global_heap().is_none() {
        return Err("the shared GPU view heap is not open".into());
    }
    let (base, len) = real::console_buffer(args).ok_or("argument buffer has no tracked UMA backing")?;
    let source = title_source_offset(base, len, offset).ok_or("argument range is invalid")?;
    // Root raw UAV addresses are aligned without changing the dispatch record read by Load3.
    let align = D3D12_RAW_UAV_SRV_BYTE_ALIGNMENT as usize;
    let aligned = source & !(align - 1);
    let source_offset = (source - aligned) as u32;
    if !real::uma_window_buffer(aligned, STRIDE as usize + source_offset as usize)
        .is_some_and(|(buffer, _)| buffer != 0)
    {
        return Err("arguments do not fit in a backed UMA window".into());
    }
    let source = real::uma_va_of(aligned).filter(|v| *v != 0 && v.is_multiple_of(align as u64))
        .ok_or("argument GPU address is invalid")?;
    let (input_raw, mesh_raw, compute_raw) = (args as *mut c_void, pso as *mut c_void, tp.compute as *mut c_void);
    let (compute_root_raw, mesh_root_raw) = (tp.compute_rs as *mut c_void, tp.mesh_rs as *mut c_void);
    let input = ID3D12Resource::from_raw_borrowed(&input_raw).ok_or("missing argument buffer")?;
    let mesh = ID3D12PipelineState::from_raw_borrowed(&mesh_raw).ok_or("no mesh pipeline")?;
    let compute = ID3D12PipelineState::from_raw_borrowed(&compute_raw).ok_or("no compute pipeline")?;
    let compute_root = ID3D12RootSignature::from_raw_borrowed(&compute_root_raw).ok_or("no compute root")?;
    let mesh_root = ID3D12RootSignature::from_raw_borrowed(&mesh_root_raw).ok_or("no mesh root")?;
    let record_bytes = tp.record.record_dwords() as u64 * 4;
    let cap = record_groups(record_bytes, tp.mesh_groups_per_record).ok_or("two-pass mesh record or split exceeds its capacity")?;
    let buffers = buffers(this, list)?;
    let (layout, arguments, heaps) = with(this, |s| (s.layout_gfx.clone(), s.args_gfx.clone(), s.mesh_heap_bases));
    let io_cs = stage_io(this, uma, layout.as_deref(), &arguments, heaps, true)?;
    let io_ms = stage_io(this, uma, layout.as_deref(), &arguments, heaps, false)?;
    let params: Vec<_> = [cap, tp.mesh_groups_per_record, record_bytes as u32, PREP_GROUPS * 64, source_offset]
        .into_iter().flat_map(u32::to_le_bytes).collect();
    let (_, io_prep) = bound_uma(&params, this, uma)?;
    let progress = alloc_pool::track_mesh_recording(this as usize).ok_or("native recording no longer tracked")?;
    with(this, |s| {
        let r = &mut s.mesh_indirect;
        r.progress = Some(progress);
        r.roots.entry(tp.compute_rs).or_insert_with(|| compute_root.clone());
        r.roots.entry(tp.mesh_rs).or_insert_with(|| mesh_root.clone());
        r.signatures.entry(original.as_raw() as usize).or_insert_with(|| (original.clone(), dispatch.clone()));
        r.inputs.entry(args).or_insert_with(|| input.clone());
        r.pipelines.entry(pso).or_insert_with(|| (mesh.clone(), tp.clone()));
    });

    // All fallible setup is complete. The title argument resource stays in the title's state.
    list_x::ensure_view_heap(this);
    uav_barrier(list, None);
    let scratch = buffers.args.as_raw() as usize;
    let records = buffers.records.as_raw() as usize;
    let record_va = buffers.records.GetGPUVirtualAddress();
    list_x::transition(this, records, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    list_x::transition(this, scratch, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    list.SetComputeRootSignature(&buffers.prep.root);
    list_x::bind_mesh_internal_pipeline(this, buffers.prep.pso.as_raw());
    list.SetComputeRootUnorderedAccessView(0, io_prep);
    list.SetComputeRootUnorderedAccessView(1, record_va);
    list.SetComputeRootUnorderedAccessView(2, source);
    list.SetComputeRootUnorderedAccessView(3, buffers.args.GetGPUVirtualAddress());
    let dispatch_preprocess: unsafe extern "system" fn(*mut c_void, u32, u32, u32) =
        std::mem::transmute(real::orig_fn(this, 14));
    dispatch_preprocess(this, PREP_GROUPS, 1, 1);
    list_x::transition(this, scratch, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
    uav_barrier(list, Some(&buffers.records));

    let views = |io| [io, record_va].into_iter().chain(
        (0..4).map(|k| windows.get(k).copied().filter(|&v| v != 0).unwrap_or(first))
    );
    let execute: ExecuteIndirect = std::mem::transmute(real::orig_fn(this, 59));
    list.SetComputeRootSignature(compute_root);
    list_x::bind_mesh_internal_pipeline(this, compute.as_raw());
    for (k, va) in views(io_cs).enumerate() {
        list.SetComputeRootUnorderedAccessView(k as u32, va);
    }
    execute(this, dispatch.as_raw() as usize, 1, scratch, 0, 0, 0);
    uav_barrier(list, Some(&buffers.records));
    list.SetGraphicsRootSignature(mesh_root);
    list_x::bind_mesh_internal_pipeline(this, mesh.as_raw());
    for (k, va) in views(io_ms).enumerate() {
        list.SetGraphicsRootUnorderedAccessView(k as u32, va);
    }
    execute(this, original.as_raw() as usize, 1, scratch, STRIDE as u64, 0, 0);
    list_x::transition(this, records, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
    list_x::transition(this, scratch, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT, D3D12_RESOURCE_STATE_COMMON);
    static RECORDED: AtomicU64 = AtomicU64::new(0);
    let n = RECORDED.fetch_add(1, Ordering::Relaxed) + 1;
    if n <= 8 || n.is_power_of_two() {
        crate::xlog_info!("D3D12.X: single two-pass ExecuteIndirect(DispatchMesh) #{n} recorded, cap {cap}, split {}", tp.mesh_groups_per_record);
    }
    Ok(())
}

pub(super) fn warn(error: String) {
    static LOGGED: AtomicU64 = AtomicU64::new(0);
    let n = LOGGED.fetch_add(1, Ordering::Relaxed) + 1;
    if error == BUDGET_EXCEEDED && n.is_power_of_two() {
        let used = USED_BYTES.load(Ordering::Acquire);
        let retired = RETIRED.try_lock().ok().map(|list| list.len());
        crate::xlog_warn!("D3D12.X: single two-pass ExecuteIndirect skipped #{n}: {error}; {used} bytes reserved, {retired:?} recordings retired");
        return;
    }
    if n <= 8 || n.is_power_of_two() {
        crate::xlog_warn!("D3D12.X: single two-pass ExecuteIndirect skipped #{n}: {error}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_only_one_plain_dispatch_mesh() {
        let mesh = [D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH_MESH.0 as u32, 0, 0, 0];
        assert!(eligible_signature(12, 0, &[mesh]));
        let root = [D3D12_INDIRECT_ARGUMENT_TYPE_CONSTANT.0 as u32, 0, 0, 0];
        for (stride, root_sig, args) in [
            (16, 0, vec![mesh]), (12, 1, vec![mesh]), (12, 0, vec![root, mesh]),
            (12, 0, vec![[107, 0, 0, 0]]), (12, 0, Vec::new()),
        ] {
            assert!(!eligible_signature(stride, root_sig, &args));
        }
        assert!(eligible_call("MS", 1, 0));
        assert!(!eligible_call("MS", 2, 0));
        assert!(!eligible_call("MS", 1, 42));
        assert!(!eligible_call("GS", 1, 0));
        assert_eq!(title_source_offset(0x1000, 32, 20), Some(0x1014));
        assert_eq!(title_source_offset(0x1000, 32, 21), None);
        assert_eq!(title_source_offset(0x1000, 32, u64::MAX), None);
        assert_eq!(title_source_offset(usize::MAX - 3, 32, 4), None);
    }

    #[test]
    fn mesh_reap_keeps_other_recordings_and_ignores_removal() {
        let mut pending = vec![1, 2, 3, 4];
        let completed = [4, u64::MAX, 2, 5];
        let values = [4, 1, 3, 5];
        let mut released = take_retired(&mut pending, |id| {
            marker_completed(values[*id - 1], completed[*id - 1])
        });
        released.sort_unstable();
        pending.sort_unstable();
        assert_eq!(released, [1, 4]);
        assert_eq!(pending, [2, 3]);
    }

    #[test]
    fn held_budget_survives_other_recording_completions() {
        let used = AtomicU64::new(BUDGET_BYTES - RECORDING_BYTES);
        let mut pending = vec![Budget::acquire(&used).unwrap()];
        assert!(Budget::acquire(&used).is_none());
        assert!(take_retired(&mut pending, |_| false).is_empty());
        assert!(Budget::acquire(&used).is_none());
        drop(take_retired(&mut pending, |_| true));
        assert!(Budget::acquire(&used).is_some());
        assert_eq!(used.load(Ordering::Acquire), BUDGET_BYTES - RECORDING_BYTES);
    }

    #[test]
    fn preprocess_shader_compiles_without_a_device() {
        let dxil = recomp::dxc::compile(PREPROCESS_HLSL, "main", "cs_6_6", &[]).expect("mesh argument shader");
        assert!(dxil.starts_with(b"DXBC"));
    }

    #[test]
    fn caps_groups_and_preserves_mesh_user_data() {
        assert_eq!(record_groups(256, 2), Some(32_767));
        assert_eq!(record_groups(RECORD_BYTES / 2, 1), Some(2));
        assert_eq!(record_groups(RECORD_BYTES + 1, 1), None);
        assert_eq!(record_groups(0, 1), None);
        assert_eq!(record_groups(256, 0), None);
        let l = console_list::Layout {
            inline_ms: 4, inline_gfx: 4, mesh_heap_flags: [true, true], ..Default::default()
        };
        let args = 7u32.to_le_bytes();
        let (table, uma) = (0x8_1234_5678u64, 0x8_0000_0000usize);
        let heaps = [Some(11), Some(22)];
        let cs = stage_words(Some(&l), &args, table, heaps, uma, true).unwrap();
        let ms = stage_words(Some(&l), &args, table, heaps, uma, false).unwrap();
        assert!(stage_words(Some(&l), &args, table, [None, Some(22)], uma, true).is_err());
        assert_eq!(&cs[8..10], &[0x1234_5678, 8]);
        assert_eq!(&cs[16..19], &[7, 11, 22]);
        assert_eq!(&ms[0..2], &[0x1234_5678, 8]);
        assert_eq!(&ms[8..11], &[7, 11, 22]);
        assert_eq!(cs[IO_UMA_BASE_HI], 8);
        assert_eq!(ms[IO_UMA_BASE_HI], 8);
    }
}
