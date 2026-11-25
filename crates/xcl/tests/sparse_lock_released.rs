//! A sparse upload waiting for either a command slot or staging space keeps the mappings
//! accessible. Keep the test in its own process because it initializes the sparse state.

use std::sync::mpsc::{self, RecvTimeoutError};
use std::time::{Duration, Instant};

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::{Common::DXGI_SAMPLE_DESC, CreateDXGIFactory2, DXGI_CREATE_FACTORY_FLAGS, IDXGIFactory6};
use windows_sys::Win32::System::Memory::{VirtualAlloc, VirtualFree, MEM_COMMIT, MEM_DECOMMIT, PAGE_READWRITE};

use xcl::gpu::{adapter, agility, recomp::runner, uma};
use xcl::hle::d3d12x::sparse;

const PAGE: usize = 4096;
const TILE: usize = 64 << 10;
const UPLOAD: usize = 96 << 20;

fn device() -> Result<ID3D12Device, String> {
    unsafe {
        let factory: IDXGIFactory6 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).map_err(|e| e.to_string())?;
        let selector = adapter::Selector::from_env_or("auto");
        let (adapter, _) = adapter::candidates(&factory, &selector).into_iter().next().ok_or_else(|| adapter::no_match(&factory, &selector))?;
        agility::create_device(&adapter, D3D_FEATURE_LEVEL_12_0).map_err(|e| e.to_string())
    }
}

unsafe fn drain(queue: &ID3D12CommandQueue, dev: &ID3D12Device) {
    let done: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("completion fence");
    queue.Signal(&done, 1).expect("signal completion");
    let deadline = Instant::now() + Duration::from_secs(10);
    while done.GetCompletedValue() == 0 && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(5));
    }
    dev.GetDeviceRemovedReason().expect("device stayed alive");
    assert_eq!(done.GetCompletedValue(), 1, "the sparse queue drained");
}

/// Read the GPU buffer directly so a readback cannot retry the CPU writes under test.
unsafe fn check_pages(dev: &ID3D12Device, queue: &ID3D12CommandQueue, window: &ID3D12Resource, offset: usize, expected: &[(usize, u8)]) {
    let len = expected.len() * PAGE;
    let mut readback = None;
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: len as u64,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        ..Default::default()
    };
    dev.CreateCommittedResource(
        &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() },
        D3D12_HEAP_FLAG_NONE,
        &desc,
        D3D12_RESOURCE_STATE_COPY_DEST,
        None,
        &mut readback,
    )
    .expect("readback buffer");
    let readback: ID3D12Resource = readback.expect("readback resource");
    let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).expect("readback allocator");
    let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).expect("readback list");
    for (i, &(page, _)) in expected.iter().enumerate() {
        list.CopyBufferRegion(&readback, (i * PAGE) as u64, window, (offset + page) as u64, PAGE as u64);
    }
    list.Close().expect("close readback list");
    queue.ExecuteCommandLists(&[Some(list.cast().expect("command list"))]);
    drain(queue, dev);

    let mut mapped = std::ptr::null_mut();
    readback.Map(0, Some(&D3D12_RANGE { Begin: 0, End: len }), Some(&mut mapped)).expect("map readback");
    let bytes = std::slice::from_raw_parts(mapped.cast::<u8>(), len);
    for (i, &(page, want)) in expected.iter().enumerate() {
        assert!(bytes[i * PAGE..(i + 1) * PAGE].iter().all(|&byte| byte == want), "GPU page {page:#x} must contain {want:#04x}");
    }
    readback.Unmap(0, Some(&D3D12_RANGE { Begin: 0, End: 0 }));
}

unsafe fn check_wait(
    dev: &ID3D12Device,
    queue: &ID3D12CommandQueue,
    addr: usize,
    hold: Duration,
    submit: impl FnOnce() + Send + 'static,
) {
    let gate: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("gate fence");
    queue.Wait(&gate, 1).expect("hold the sparse queue");
    let (started, wait_started) = mpsc::channel();
    let (done, wait_done) = mpsc::channel();
    let worker = std::thread::spawn(move || {
        let _ = started.send(());
        submit();
        let _ = done.send(());
    });
    let began = wait_started.recv_timeout(Duration::from_secs(2)).is_ok();
    let start = Instant::now();
    let initially_held = matches!(wait_done.recv_timeout(Duration::from_millis(500)), Err(RecvTimeoutError::Timeout));

    let (ready, wait_ready) = mpsc::channel();
    let mapper = std::thread::spawn(move || {
        let mapped = sparse::mapped_ranges().iter().any(|&(begin, end)| (begin..end).contains(&addr));
        let _ = ready.send(mapped);
    });
    let mapping = wait_ready.recv_timeout(Duration::from_secs(2));
    let held = initially_held && matches!(wait_done.recv_timeout(hold.saturating_sub(start.elapsed())), Err(RecvTimeoutError::Timeout));

    // Let both workers finish before asserting, even if the assertions will fail.
    gate.Signal(1).expect("release the sparse queue");
    if held {
        wait_done.recv_timeout(Duration::from_secs(10)).expect("upload finished after the fence");
    }
    if mapping.is_err() {
        wait_ready.recv_timeout(Duration::from_secs(10)).expect("mapping query finished after the fence");
    }
    worker.join().expect("upload worker");
    mapper.join().expect("mapping worker");
    assert!(began && held, "the upload waited for the closed fence");
    assert!(matches!(mapping, Ok(true)), "mapped_ranges found the allocation within two seconds while the queue was held");
}

#[test]
fn a_wait_for_the_sparse_queue_does_not_hold_the_state() {
    let dev = match device() {
        Ok(dev) => dev,
        Err(e) => {
            runner::skip_test(&format!("no D3D12 adapter: {e}"));
            return;
        }
    };
    unsafe {
        let windows: Vec<_> = sparse::init(&dev)
            .into_iter()
            .map(|(index, raw, _)| (index, ID3D12Resource::from_raw(raw as *mut _)))
            .collect();
        let Some((index, window)) = windows.first() else {
            runner::skip_test("no sparse VRAM windows");
            return;
        };
        let queue = sparse::queue().expect("sparse queue");
        let title: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).expect("title queue");
        let (addr, len) = sparse::alloc_in(*index, UPLOAD, TILE).expect("96 MiB sparse allocation");
        assert_eq!(len, UPLOAD);
        let offset = addr - uma::get().expect("GPU UMA").window(*index).0;
        let pages: Vec<_> = (addr..addr + len).step_by(PAGE).collect();
        std::ptr::write_bytes(addr as *mut u8, 0x11, len);
        sparse::before_execute(&title, &pages, PAGE);
        check_pages(&dev, &queue, window, offset, &[(0, 0x11), (64 << 20, 0x11), (len - PAGE, 0x11)]);

        // Partial-tile decommits fill 16 command slots with zero uploads, using no staging space.
        let small_title = title.clone();
        let holes: Vec<_> = pages[1..40].iter().step_by(2).copied().collect();
        let small = holes.clone();
        check_wait(&dev, &queue, addr, Duration::from_secs(3), move || {
            for page in small {
                sparse::decommitted(page, PAGE);
                assert_ne!(VirtualFree(page as *mut _, PAGE, MEM_DECOMMIT), 0, "decommit one page");
                sparse::before_execute(&small_title, &[], PAGE);
            }
        });
        let mut expected: Vec<_> = holes.iter().map(|&page| (page - addr, 0)).collect();
        expected.push((0, 0x11));
        check_pages(&dev, &queue, window, offset, &expected);
        for page in holes {
            assert_eq!(VirtualAlloc(page as *const _, PAGE, MEM_COMMIT, PAGE_READWRITE) as usize, page, "recommit one page");
            sparse::committed(page, PAGE);
        }

        // The second batch cannot fit in the 64 MiB staging ring. Hold it through the
        // five-second fence warning and verify that both halves reach the GPU afterwards.
        let zero = 64 << 20;
        std::ptr::write_bytes(addr as *mut u8, 0x22, len - PAGE);
        std::ptr::write_bytes((addr + zero) as *mut u8, 0, PAGE);
        let dirty = pages[..pages.len() - 1].to_vec();
        check_wait(&dev, &queue, addr, Duration::from_secs(6), move || sparse::before_execute(&title, &dirty, PAGE));
        check_pages(&dev, &queue, window, offset, &[(0, 0x22), (zero - PAGE, 0x22), (zero, 0), (len - 2 * PAGE, 0x22), (len - PAGE, 0x11)]);
        assert!(sparse::free(addr));
    }
}
