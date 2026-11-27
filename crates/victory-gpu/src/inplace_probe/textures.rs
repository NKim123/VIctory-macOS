//! Texture allocation probes on the same direct queue as their transfers.

use super::*;
use windows::Win32::Graphics::Dxgi::Common::*;

fn texture_desc(size: u32, mips: u16, format: DXGI_FORMAT) -> D3D12_RESOURCE_DESC {
    D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Width: u64::from(size),
        Height: size,
        DepthOrArraySize: 1,
        MipLevels: mips,
        Format: format,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        ..Default::default()
    }
}

fn heap(device: &ID3D12Device, size: u64) -> ID3D12Heap {
    let desc = D3D12_HEAP_DESC {
        SizeInBytes: size,
        Alignment: K64 as u64,
        Properties: D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            ..Default::default()
        },
        Flags: D3D12_HEAP_FLAG_ALLOW_ONLY_NON_RT_DS_TEXTURES,
    };
    let mut heap = None;
    // SAFETY: the size is aligned and the output is ours.
    unsafe { device.CreateHeap(&desc, &mut heap) }.unwrap();
    heap.unwrap()
}

struct Transfer<'a> {
    texture: &'a ID3D12Resource,
    mip: u32,
    footprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
    rows: usize,
    row_bytes: usize,
    pattern: Vec<u8>,
    upload: ID3D12Resource,
    readback: ID3D12Resource,
}

impl<'a> Transfer<'a> {
    fn new(rig: &Rig, texture: &'a ID3D12Resource, mip: u32, seed: u8) -> Self {
        let mut footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
        let (mut rows, mut row_bytes, mut bytes) = (0, 0, 0);
        // SAFETY: all outputs are local and the mip exists.
        unsafe {
            rig.device.GetCopyableFootprints(
                &texture.GetDesc(),
                mip,
                1,
                0,
                Some(&mut footprint),
                Some(&mut rows),
                Some(&mut row_bytes),
                Some(&mut bytes),
            );
        }
        let upload = rig
            .committed(
                D3D12_HEAP_TYPE_UPLOAD,
                bytes,
                D3D12_RESOURCE_STATE_GENERIC_READ,
            )
            .unwrap();
        let readback = rig
            .committed(
                D3D12_HEAP_TYPE_READBACK,
                bytes,
                D3D12_RESOURCE_STATE_COPY_DEST,
            )
            .unwrap();
        let pattern = pattern(rows as usize * row_bytes as usize, seed);
        let pitch = footprint.Footprint.RowPitch as usize;
        let mut data = std::ptr::null_mut();
        // SAFETY: each row is inside the D3D12 copy footprint.
        unsafe {
            upload.Map(0, None, Some(&mut data)).unwrap();
            for (row, bytes) in pattern.chunks_exact(row_bytes as usize).enumerate() {
                write(data.cast::<u8>().add(row * pitch), bytes);
            }
            upload.Unmap(0, None);
        }
        Self {
            texture,
            mip,
            footprint,
            rows: rows as usize,
            row_bytes: row_bytes as usize,
            pattern,
            upload,
            readback,
        }
    }

    fn copy(&self, list: &ID3D12GraphicsCommandList, upload: bool) {
        let buffer = if upload { &self.upload } else { &self.readback };
        // SAFETY: these descriptors borrow the live resources for recording.
        unsafe {
            let buffer = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(buffer),
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    PlacedFootprint: self.footprint,
                },
            };
            let texture = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(self.texture),
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    SubresourceIndex: self.mip,
                },
            };
            let (to, from) = if upload {
                (&texture, &buffer)
            } else {
                (&buffer, &texture)
            };
            list.CopyTextureRegion(to, 0, 0, 0, from, None);
        }
    }

    fn verify(&self) {
        let mut data = std::ptr::null_mut();
        let pitch = self.footprint.Footprint.RowPitch as usize;
        // SAFETY: the queue has completed the copy and each row fits the buffer.
        unsafe {
            self.readback.Map(0, None, Some(&mut data)).unwrap();
            let equal = (0..self.rows).all(|row| {
                same(
                    data.cast::<u8>().add(row * pitch),
                    &self.pattern[row * self.row_bytes..(row + 1) * self.row_bytes],
                )
            });
            let first_byte = *data.cast::<u8>();
            self.readback.Unmap(0, None);
            assert!(
                equal,
                "mip {} readback differed ({} bytes, first={first_byte:#04x} expected={:#04x})",
                self.mip,
                self.pattern.len(),
                self.pattern[0]
            );
        }
    }
}

#[test]
fn small_texture_placement_probe() {
    let Some(host) = crate::host::host().filter(|_| crate::watch::rosetta()) else {
        return;
    };
    let device = &host.device;
    let mut rig = Rig::with_queue(device, host.queue()).unwrap();
    let small = u64::from(D3D12_SMALL_RESOURCE_PLACEMENT_ALIGNMENT);
    let heap = heap(device, K64 as u64);
    let mut textures = Vec::new();
    let mut offset = 0;
    for format in [DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_BC7_UNORM] {
        let mut desc = texture_desc(32, 1, format);
        // SAFETY: querying two descriptions of the same texture shape.
        let default = unsafe { device.GetResourceAllocationInfo(0, &[desc]) };
        desc.Alignment = small;
        // SAFETY: the same texture with a documented small alignment request.
        let info = unsafe { device.GetResourceAllocationInfo(0, &[desc]) };
        eprintln!(
            "PLACEMENT {format:?} 32x32: default {}/{} bytes, requested 4096 => {}/{} bytes (size/alignment)",
            default.SizeInBytes, default.Alignment, info.SizeInBytes, info.Alignment
        );
        if info.Alignment != small || !(small..=K64 as u64).contains(&info.SizeInBytes) {
            continue;
        }
        for _ in 0..3 {
            let mut texture = None;
            // SAFETY: disjoint aligned ranges of a live default texture heap.
            let created = unsafe {
                device.CreatePlacedResource(
                    &heap,
                    offset,
                    &desc,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    None,
                    &mut texture,
                )
            };
            eprint!("PLACEMENT {format:?} at {offset} bytes ");
            report("CreatePlacedResource", &created);
            created.unwrap();
            textures.push(texture.unwrap());
            offset += info.SizeInBytes;
        }
    }
    assert_eq!(
        textures.len(),
        6,
        "host did not report 4 KiB placement for both formats"
    );

    let transfers: Vec<_> = textures
        .iter()
        .enumerate()
        .map(|(i, t)| Transfer::new(&rig, t, 0, i as u8 + 1))
        .collect();
    rig.run(|list| {
        for transfer in &transfers {
            transfer.copy(list, true);
        }
        for transfer in &transfers {
            // SAFETY: the prior copy wrote this subresource.
            unsafe {
                list.ResourceBarrier(&[crate::host::transition(
                    transfer.texture,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                )]);
            }
            transfer.copy(list, false);
        }
    })
    .unwrap();
    for transfer in &transfers {
        transfer.verify();
    }
    eprintln!("PLACEMENT six textures verified, offsets 0..{offset} bytes");
}

fn map(rig: &Rig, texture: &ID3D12Resource, heap: &ID3D12Heap, mip: u32, offset: u32, count: u32) {
    let at = D3D12_TILED_RESOURCE_COORDINATE {
        Subresource: mip,
        ..Default::default()
    };
    let size = D3D12_TILE_REGION_SIZE {
        NumTiles: count,
        ..Default::default()
    };
    // SAFETY: one region and one range obtained from GetResourceTiling.
    unsafe {
        rig.queue.UpdateTileMappings(
            texture,
            1,
            Some(&at),
            Some(&size),
            heap,
            1,
            Some(&D3D12_TILE_RANGE_FLAG_NONE),
            Some(&offset),
            Some(&count),
            D3D12_TILE_MAPPING_FLAG_NONE,
        );
    }
}

fn transfer_mips(rig: &mut Rig, write: &[Transfer<'_>], read: &[&Transfer<'_>]) {
    rig.run(|list| {
        crate::tiles::activate(list, write[0].texture);
        for transfer in write {
            transfer.copy(list, true);
        }
        for transfer in write {
            let mut barrier = crate::host::transition(
                transfer.texture,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            );
            // SAFETY: this is a transition barrier and only this mip was written.
            unsafe {
                let transition = &mut *barrier.Anonymous.Transition;
                transition.Subresource = transfer.mip;
                list.ResourceBarrier(&[barrier]);
            }
        }
        for transfer in read {
            transfer.copy(list, false);
        }
    })
    .unwrap();
    for transfer in read {
        transfer.verify();
    }
}

/// Compare the first texel exposed by a mip-restricted SRV with an
/// identically initialized committed texture.
fn sample_mip(rig: &mut Rig, transfer: &Transfer<'_>, mut desc: D3D12_RESOURCE_DESC) {
    use crate::compute::{self, Param};
    use windows::core::s;

    let mip = transfer.mip;
    let mut reference = None;
    desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    // SAFETY: ordinary committed texture, the descriptor and output are local.
    unsafe {
        rig.device.CreateCommittedResource(
            &D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_DEFAULT,
                ..Default::default()
            },
            D3D12_HEAP_FLAG_NONE,
            &desc,
            D3D12_RESOURCE_STATE_COPY_DEST,
            None,
            &mut reference,
        )
    }
    .unwrap();
    let reference = reference.unwrap();
    let reference_upload = Transfer {
        texture: &reference,
        ..Transfer::new(rig, transfer.texture, mip, transfer.pattern[0])
    };

    let (root, pso) = compute::pipeline(
        "Texture2D<float4> image : register(t0);
         RWByteAddressBuffer output : register(u0);
         [numthreads(1, 1, 1)] void main() {
             output.Store4(0, asuint(image.Load(int3(0, 0, 0))));
         }",
        s!("reserved_texture_probe"),
        0,
        &[
            Param::Table(&[compute::srv(0, 1)]),
            Param::Uav { register: 0 },
        ],
    )
    .unwrap();
    // SAFETY: a plain descriptor-heap creation with a local description.
    let descriptors: ID3D12DescriptorHeap = unsafe {
        rig.device
            .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: 2,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                ..Default::default()
            })
    }
    .unwrap();
    let srv = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: desc.Format,
        ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Texture2D: D3D12_TEX2D_SRV {
                MostDetailedMip: mip,
                MipLevels: 1,
                ..Default::default()
            },
        },
    };
    let mut outputs = Vec::<ID3D12Resource>::new();
    // SAFETY: two descriptors in the heap and two independent 16-byte UAVs.
    unsafe {
        let step = rig
            .device
            .GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        let cpu = descriptors.GetCPUDescriptorHandleForHeapStart();
        let gpu = descriptors.GetGPUDescriptorHandleForHeapStart();
        for (i, texture) in [transfer.texture, &reference].into_iter().enumerate() {
            rig.device.CreateShaderResourceView(
                texture,
                Some(&srv),
                D3D12_CPU_DESCRIPTOR_HANDLE {
                    ptr: cpu.ptr + i * step as usize,
                },
            );
            let mut output: Option<ID3D12Resource> = None;
            let mut d = buffer_desc(16, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
            d.Alignment = 0;
            rig.device
                .CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES {
                        Type: D3D12_HEAP_TYPE_DEFAULT,
                        ..Default::default()
                    },
                    D3D12_HEAP_FLAG_NONE,
                    &d,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    None,
                    &mut output,
                )
                .unwrap();
            outputs.push(output.unwrap());
        }
        let transition = |texture, from, to| {
            let mut barrier = crate::host::transition(texture, from, to);
            let transition = &mut *barrier.Anonymous.Transition;
            transition.Subresource = mip;
            barrier
        };
        let shader = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        let copy = D3D12_RESOURCE_STATE_COPY_SOURCE;
        rig.run(|list| {
            reference_upload.copy(list, true);
            list.ResourceBarrier(&[
                transition(&reference, D3D12_RESOURCE_STATE_COPY_DEST, shader),
                transition(transfer.texture, copy, shader),
            ]);
            list.SetDescriptorHeaps(&[Some(descriptors.clone())]);
            list.SetComputeRootSignature(&root);
            list.SetPipelineState(&pso);
            for (i, output) in outputs.iter().enumerate() {
                list.SetComputeRootDescriptorTable(
                    0,
                    D3D12_GPU_DESCRIPTOR_HANDLE {
                        ptr: gpu.ptr + i as u64 * u64::from(step),
                    },
                );
                list.SetComputeRootUnorderedAccessView(1, output.GetGPUVirtualAddress());
                list.Dispatch(1, 1, 1);
                list.ResourceBarrier(&[crate::host::transition(
                    output,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    copy,
                )]);
            }
            list.ResourceBarrier(&[transition(transfer.texture, shader, copy)]);
        })
        .unwrap();
    }
    let sparse = rig.fetch(&outputs[0], 0, 16).unwrap();
    let committed = rig.fetch(&outputs[1], 0, 16).unwrap();
    assert_ne!(
        committed, [0; 16],
        "reference sample must exercise the view"
    );
    assert_eq!(
        sparse, committed,
        "shader read of mip {mip} differs from committed texture"
    );
}

#[test]
fn reserved_texture_tier_two_probe() {
    let Some(host) = crate::host::host().filter(|_| crate::watch::rosetta()) else {
        return;
    };
    let device = &host.device;
    let mut rig = Rig::with_queue(device, host.queue()).unwrap();
    let mut features = D3D12_FEATURE_DATA_D3D12_OPTIONS::default();
    // SAFETY: feature and output type match.
    unsafe {
        device.CheckFeatureSupport(
            D3D12_FEATURE_D3D12_OPTIONS,
            (&raw mut features).cast(),
            size_of_val(&features) as u32,
        )
    }
    .unwrap();
    eprintln!(
        "SPARSE tiled resources tier {}",
        features.TiledResourcesTier.0
    );
    if features.TiledResourcesTier.0 < D3D12_TILED_RESOURCES_TIER_2.0 {
        return;
    }
    let mut supported = 0;
    for format in [
        DXGI_FORMAT_BC7_UNORM,
        DXGI_FORMAT_BC1_UNORM,
        DXGI_FORMAT_R8G8B8A8_UNORM,
    ] {
        let mut desc = texture_desc(512, 10, format);
        desc.Layout = D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE;
        let mut texture = None;
        // SAFETY: standard single-layer reserved texture and typed output.
        let created = unsafe {
            device.CreateReservedResource(&desc, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut texture)
        };
        eprint!("SPARSE {format:?} 512x512 mips 10 ");
        report("CreateReservedResource", &created);
        if created.is_err() {
            continue;
        }
        let texture = texture.unwrap();
        let (mut total, mut packed, mut shape) = (
            0,
            D3D12_PACKED_MIP_INFO::default(),
            D3D12_TILE_SHAPE::default(),
        );
        let mut tilings = [D3D12_SUBRESOURCE_TILING::default(); 10];
        let mut count = tilings.len() as u32;
        // SAFETY: outputs are local, and tilings has room for count entries.
        unsafe {
            device.GetResourceTiling(
                &texture,
                Some(&mut total),
                Some(&mut packed),
                Some(&mut shape),
                Some(&mut count),
                0,
                tilings.as_mut_ptr(),
            );
        }
        eprintln!("SPARSE {format:?} {total} tiles, {packed:?}, {shape:?}, {count} subresources");
        assert!(
            total > 0
                && packed.NumStandardMips > 0
                && packed.NumPackedMips > 0
                && packed.NumTilesForPackedMips > 0
        );
        let heap = heap(device, u64::from(total) * K64 as u64);
        let tail = u32::from(packed.NumStandardMips);
        map(
            &rig,
            &texture,
            &heap,
            tail,
            packed.StartTileIndexInOverallResource,
            packed.NumTilesForPackedMips,
        );
        let coarse: Vec<_> = (tail..10)
            .map(|mip| Transfer::new(&rig, &texture, mip, mip as u8 + 1))
            .collect();
        transfer_mips(&mut rig, &coarse, &coarse.iter().collect::<Vec<_>>());
        sample_mip(&mut rig, &coarse[0], desc);
        let mip = tail - 1;
        let tiling = tilings[mip as usize];
        let tiles =
            tiling.WidthInTiles * u32::from(tiling.HeightInTiles) * u32::from(tiling.DepthInTiles);
        assert!(tiles > 0);
        map(
            &rig,
            &texture,
            &heap,
            mip,
            tiling.StartTileIndexInOverallResource,
            tiles,
        );
        let fine = [Transfer::new(&rig, &texture, mip, 32)];
        let read = fine.iter().chain(&coarse).collect::<Vec<_>>();
        transfer_mips(&mut rig, &fine, &read);
        sample_mip(&mut rig, &fine[0], desc);
        eprintln!("SPARSE {format:?} verified tail mip {tail} and finer mip {mip}");
        supported += 1;
    }
    assert_eq!(supported, 3, "host refused reserved texture creation");
}

#[test]
fn placed_srv_uploads_only_its_backed_mips() {
    use crate::backend::{HostGpu, HostMemory};
    use crate::descriptor::View;

    let Some(host) =
        crate::host::host().filter(|_| crate::watch::rosetta() && crate::tiles::enabled())
    else {
        return;
    };
    let rig = Rig::with_queue(&host.device, host.queue()).unwrap();
    // Linear RGBA8, 512 square with 10 mips, in independent title memory.
    let memory = vec![0x59u8; 2 << 20];
    let address = memory.as_ptr() as u64;
    let words = [3, 0, 0, 0, 512, 0, 512, 1 | 10 << 16, 28, 1, 0, 1, 0, 0];
    let get_copy = |first| {
        let view = View::Placed(
            false,
            words,
            [
                28,
                4,
                0x1688,
                0,
                first,
                10 - first,
                0,
                0,
                0,
                0,
                address as u32,
                (address >> 32) as u32,
                0,
                0,
                0,
                0,
            ],
        );
        let View::Shader(resource, _) = crate::descriptor::host::placed_texture(&view).unwrap()
        else {
            panic!("expected a 2D SRV");
        };
        let resource = resource as crate::object::This;
        let copy = crate::resource::host_copy(resource).unwrap();
        let staging = crate::texture::staging(resource, &copy).unwrap();
        (copy, staging)
    };
    let (coarse, coarse_staging) = get_copy(3);
    let (fine, fine_staging) = get_copy(2);
    let (full, full_staging) = get_copy(0);
    assert_eq!(
        [coarse.first_mip(), fine.first_mip(), full.first_mip()],
        [3, 2, 0]
    );
    assert_eq!(coarse.tiles.as_ref().unwrap().bytes(), K64 as u64);
    assert_eq!(fine.tiles.as_ref().unwrap().bytes(), 2 * K64 as u64);
    assert!(coarse_staging.size_bytes() < fine_staging.size_bytes());
    assert!(fine_staging.size_bytes() < full_staging.size_bytes());
    let list = host.new_list().unwrap();
    let mut reads = Vec::new();
    for (copy, staging) in [(&coarse, &coarse_staging), (&fine, &fine_staging)] {
        let crate::texture::WorkerRefresh::Uploaded { upload, .. } =
            staging.refresh_for_worker(&mut |sz| {
                host.upload_buffer(sz).map(|(b, p)| (b, 0, p))
            })
        else {
            panic!("available title mip bytes");
        };
        assert_eq!(upload.2.len(), (10 - copy.first_mip()) as usize);
        eprintln!(
            "SPARSE production: mip {}, {} upload bytes, {} tiles",
            copy.first_mip(),
            staging.size_bytes(),
            copy.tiles.as_ref().unwrap().bytes() / K64 as u64
        );
        if let Some(barrier) = crate::texture::record_upload(&list, copy, staging, &upload) {
            // SAFETY: the list is open, and owns the texture and upload.
            unsafe { list.list.ResourceBarrier(&[barrier]) };
        }
        if let Some(barrier) = list.move_texture(copy, D3D12_RESOURCE_STATE_COPY_SOURCE.0 as u32) {
            // SAFETY: the list is open and owns the resource.
            unsafe { list.list.ResourceBarrier(&[barrier]) };
        }
        for mip in [copy.first_mip(), 9] {
            let mut transfer = Transfer::new(&rig, &copy.texture, mip, 0);
            transfer.pattern.fill(0x59);
            transfer.copy(&list.list, false);
            reads.push(transfer);
        }
    }
    list.close();
    host.execute(&[&list]);
    for transfer in &reads {
        transfer.verify();
    }
}
