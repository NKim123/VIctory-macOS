use super::*;

const COPY_SRC: D3D12_RESOURCE_STATES = D3D12_RESOURCE_STATE_COPY_SOURCE;
const COPY_DST: D3D12_RESOURCE_STATES = D3D12_RESOURCE_STATE_COPY_DEST;

const WIDTH: u32 = 9;
const HEIGHT: u32 = 9;
const ROW: u32 = 256;

struct Image {
    texture: ID3D12Resource,
    footprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
    state: D3D12_RESOURCE_STATES,
}

impl Image {
    fn new(
        list: &HostList,
        format: DXGI_FORMAT,
        state: D3D12_RESOURCE_STATES,
        slices: u32,
        words: &[u32],
    ) -> Self {
        let host = host().unwrap();
        let mut texture = None;
        // SAFETY: plain creation with a valid descriptor and heap.
        unsafe {
            host.device.CreateCommittedResource(
                &D3D12_HEAP_PROPERTIES {
                    Type: D3D12_HEAP_TYPE_DEFAULT,
                    ..Default::default()
                },
                D3D12_HEAP_FLAG_NONE,
                &D3D12_RESOURCE_DESC {
                    Dimension: if slices == 1 {
                        D3D12_RESOURCE_DIMENSION_TEXTURE2D
                    } else {
                        D3D12_RESOURCE_DIMENSION_TEXTURE3D
                    },
                    Width: u64::from(WIDTH),
                    Height: HEIGHT,
                    DepthOrArraySize: slices as u16,
                    MipLevels: 1,
                    Format: format,
                    SampleDesc: DXGI_SAMPLE_DESC {
                        Count: 1,
                        Quality: 0,
                    },
                    Flags: if state == WRITE {
                        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
                    } else {
                        D3D12_RESOURCE_FLAG_NONE
                    },
                    ..Default::default()
                },
                COPY_DST,
                None,
                &mut texture,
            )
        }
        .unwrap();
        let image = Self {
            texture: texture.unwrap(),
            footprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                Offset: 0,
                Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                    Format: format,
                    Width: WIDTH,
                    Height: HEIGHT,
                    Depth: slices,
                    RowPitch: ROW,
                },
            },
            state,
        };
        let (upload, data) = host.upload_buffer(image.staging_bytes()).unwrap();
        assert_eq!(words.len(), (WIDTH * HEIGHT * slices) as usize);
        for (row, words) in words.as_chunks::<{ WIDTH as usize }>().0.iter().enumerate() {
            // SAFETY: the mapped upload has every padded row.
            unsafe {
                data.add(row * ROW as usize)
                    .cast::<u32>()
                    .copy_from_nonoverlapping(words.as_ptr(), words.len())
            };
        }
        // SAFETY: the open list copies initialized texels, then changes the image state.
        unsafe {
            list.list
                .CopyTextureRegion(&image.whole(), 0, 0, 0, &image.placed(&upload), None);
            list.list
                .ResourceBarrier(&[transition(&image.texture, COPY_DST, state)]);
        }
        list.keep(&upload);
        image
    }

    fn staging_bytes(&self) -> u64 {
        u64::from(ROW * HEIGHT * self.footprint.Footprint.Depth)
    }

    fn whole(&self) -> D3D12_TEXTURE_COPY_LOCATION {
        D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: a borrowed pointer the location never releases.
            pResource: unsafe { std::mem::transmute_copy(&self.texture) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: 0,
            },
        }
    }

    fn placed(&self, buffer: &ID3D12Resource) -> D3D12_TEXTURE_COPY_LOCATION {
        D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: as above.
            pResource: unsafe { std::mem::transmute_copy(buffer) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: self.footprint,
            },
        }
    }

    fn read(&self, list: &HostList) -> Snapshot {
        let (buffer, data) = list
            .readback(host().unwrap(), self.staging_bytes())
            .unwrap();
        // SAFETY: a live image in its known state; a mapped, live readback destination.
        unsafe {
            list.list
                .ResourceBarrier(&[transition(&self.texture, self.state, COPY_SRC)]);
            list.list
                .CopyTextureRegion(&self.placed(&buffer), 0, 0, 0, &self.whole(), None);
            list.list
                .ResourceBarrier(&[transition(&self.texture, COPY_SRC, self.state)]);
        }
        Snapshot {
            _buffer: buffer,
            data,
            rows: HEIGHT * self.footprint.Footprint.Depth,
        }
    }
}

struct Snapshot {
    _buffer: ID3D12Resource,
    data: *mut u8,
    rows: u32,
}

impl Snapshot {
    fn texels(&self) -> Vec<u32> {
        (0..self.rows)
            .flat_map(|row| {
                // SAFETY: tests call after the queue finishes, for the initialized part of each mapped row.
                unsafe {
                    std::slice::from_raw_parts(
                        self.data.add((row * ROW) as usize).cast::<u32>(),
                        WIDTH as usize,
                    )
                }
                .iter()
                .copied()
            })
            .collect()
    }
}

fn shared_exp(exponent: u32, r: u32, g: u32, b: u32) -> u32 {
    (exponent << 27) | (b << 18) | (g << 9) | r
}

fn float_rgb(red_green: u32, blue: u32) -> u32 {
    red_green | (red_green << 11) | (blue << 22)
}

#[test]
fn off_and_constant_settings() {
    for value in ["", "off", "0", "NaN", "-1", "constant:inf"] {
        assert_eq!(Mode::parse(value), Mode::Off);
    }
    assert_eq!(Mode::parse(" 0.8 "), Mode::Estimate(0.8));
    assert_eq!(Mode::parse("constant:0.25"), Mode::Constant(0.25));
}

#[test]
fn a_shared_slot_is_frozen_once_and_a_rewrite_disarms_every_pair() {
    let image = Watch::new();
    let depth = Watch::new();
    let mut first = 0u32;
    let mut second = 0u32;
    let pairs = [
        Candidate {
            image: image.clone(),
            depth: Some(depth.clone()),
            count: (&raw mut first) as usize,
        },
        Candidate {
            image: image.clone(),
            depth: None,
            count: (&raw mut second) as usize,
        },
    ];
    let frozen = arm(&pairs);
    assert_eq!((first, second, frozen.len()), (1, 1, 2));
    assert!(image.0.try_write().is_err());
    assert!(depth.0.try_write().is_err());
    drop(frozen);

    depth.invalidate();
    drop(arm(&pairs));
    assert_eq!((first, second), (0, 1));

    image.invalidate();
    drop(arm(&pairs));
    assert_eq!((first, second), (0, 0));
}

#[test]
fn only_expected_compute_roles_are_enabled() {
    let resolve = b"CS_ScreenProbes_TemporalResolve_Synthetic";
    let radiance = b"CS_CacheScreenSpaceRadiance_Synthetic";
    let air = b"CS_AccumulateLocalLightInscatteringVolume_Synthetic";
    let settings = Settings {
        mode: Mode::Estimate(0.8),
        shade: [1.0, 0.6],
        air: 2.0,
    };
    assert_eq!(Role::in_container(resolve, settings), Some(Role::Resolve));
    assert_eq!(Role::in_container(radiance, settings), Some(Role::Radiance));
    assert_eq!(Role::in_container(air, settings), Some(Role::Air));
    assert_eq!(Role::in_container(b"CS_TemporalResolve", settings), None);
    assert_eq!(
        Role::in_container(
            air,
            Settings {
                air: 0.0,
                ..settings
            }
        ),
        None
    );
    let constant = Settings {
        mode: Mode::Constant(0.25),
        ..settings
    };
    assert_eq!(Role::in_container(resolve, constant), Some(Role::Resolve));
    assert_eq!(Role::in_container(radiance, constant), None);
    assert_eq!(Role::in_container(air, constant), None);
    assert_eq!(
        Role::in_container(
            resolve,
            Settings {
                mode: Mode::Off,
                ..settings
            }
        ),
        None
    );
}

#[test]
fn title_indirect_light_and_air_read_back() {
    let Some(host) = host() else {
        eprintln!("no D3D12 host for the pixel test");
        return;
    };
    let run = host.new_list().unwrap();
    let len = (WIDTH * HEIGHT) as usize;
    let one = float_rgb(0x3c0, 0x1e0);
    let quarter = float_rgb(0x340, 0x1a0);
    let dimmer = float_rgb(0x300, 0x180);
    let radiance = Image::new(&run, DXGI_FORMAT_R11G11B10_FLOAT, WRITE, 1, &vec![one; len]);
    let depth = Image::new(
        &run,
        DXGI_FORMAT_R32_FLOAT,
        READ,
        1,
        &vec![0.5_f32.to_bits(); len],
    );
    let mut original_light = vec![0; len];
    original_light[1] = shared_exp(16, 256, 256, 256);
    original_light[2] = shared_exp(19, 256, 256, 256);
    let light = Image::new(&run, DXGI_FORMAT_R32_UINT, WRITE, 1, &original_light);
    let mut original_air = vec![one; len * 3];
    original_air[1] = dimmer;
    let air = Image::new(&run, DXGI_FORMAT_R11G11B10_FLOAT, WRITE, 3, &original_air);

    // SAFETY: the heap and all descriptors remain live until execution finishes.
    let (light_view, radiance_view, air_view, heap) = unsafe {
        let heap: ID3D12DescriptorHeap = host
            .device
            .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: 4,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                NodeMask: 0,
            })
            .unwrap();
        let cpu = heap.GetCPUDescriptorHandleForHeapStart().ptr;
        let gpu = heap.GetGPUDescriptorHandleForHeapStart().ptr;
        let stride = host
            .device
            .GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        let cpu_at = |i: u32| D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: cpu + (i * stride) as usize,
        };
        let gpu_at = |i: u32| D3D12_GPU_DESCRIPTOR_HANDLE {
            ptr: gpu + u64::from(i * stride),
        };
        host.device
            .CreateShaderResourceView(&depth.texture, None, cpu_at(0));
        for (slot, image) in [(1, &light), (2, &radiance), (3, &air)] {
            host.device.CreateUnorderedAccessView(
                &image.texture,
                None::<&ID3D12Resource>,
                None,
                cpu_at(slot),
            );
        }
        let views = |image: &Image, index, depth, slices| Views {
            resource: image.texture.clone(),
            image: gpu_at(index),
            depth,
            size: [WIDTH, HEIGHT, slices],
        };
        run.list.SetDescriptorHeaps(&[Some(heap.clone())]);
        (
            views(&light, 1, Some(gpu_at(0)), 1),
            views(&radiance, 2, None, 1),
            views(&air, 3, None, 3),
            heap,
        )
    };

    let settings = Settings {
        mode: Mode::Estimate(0.5),
        shade: [1.0, 0.6],
        air: 0.25,
    };
    let off = Recorder::new(Settings {
        mode: Mode::Off,
        ..settings
    });
    assert!(!off.after(&run, Role::Resolve, &light_view, None));
    assert!(!off.after(&run, Role::Air, &air_view, None));

    let estimate = Recorder::new(settings);
    assert!(estimate.after(&run, Role::Resolve, &light_view, None));
    assert!(estimate.after(&run, Role::Air, &air_view, None));
    let startup_light = light.read(&run);
    let startup_air = air.read(&run);
    assert!(estimate.after(&run, Role::Radiance, &radiance_view, None));
    run.close();
    host.execute(&[&run]);
    assert_eq!(startup_light.texels(), original_light);
    assert_eq!(startup_air.texels(), original_air);

    let resolve = host.new_list().unwrap();
    // SAFETY: a live shader-visible heap and open command list.
    unsafe { resolve.list.SetDescriptorHeaps(&[Some(heap.clone())]) };
    assert!(estimate.after(&resolve, Role::Resolve, &light_view, None));
    assert!(estimate.after(&resolve, Role::Air, &air_view, None));
    let resolved = light.read(&resolve);
    let capped = air.read(&resolve);
    let retained = radiance.read(&resolve);
    let retained_depth = depth.read(&resolve);
    let constant = Recorder::new(Settings {
        mode: Mode::Constant(0.25),
        ..settings
    });
    assert!(constant.after(&resolve, Role::Resolve, &light_view, None));
    assert!(!constant.after(&resolve, Role::Air, &air_view, None));
    let fixed = light.read(&resolve);
    resolve.close();
    host.execute(&[&resolve]);

    let mut expected_light = vec![shared_exp(15, 256, 256, 256); len];
    expected_light[1] = original_light[1];
    expected_light[2] = shared_exp(17, 256, 256, 256);
    assert_eq!(resolved.texels(), expected_light);
    let mut expected_air = vec![quarter; len * 3];
    expected_air[1] = dimmer;
    assert_eq!(capped.texels(), expected_air);
    assert_eq!(retained.texels(), vec![one; len]);
    assert_eq!(retained_depth.texels(), vec![0.5_f32.to_bits(); len]);
    assert_eq!(fixed.texels(), vec![shared_exp(14, 256, 251, 241); len]);
}
