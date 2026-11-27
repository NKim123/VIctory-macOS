//! DispatchX dimensions and the optional root words emitted by the shader.

const THREAD_COUNT: u32 = 0x20;
const MAX_GROUPS: u32 = 65_535;

#[cfg(windows)]
use crate::pipeline::host::HostPipeline;
#[cfg(windows)]
use windows::Win32::Graphics::Direct3D12::ID3D12GraphicsCommandList;

struct Dispatch {
    groups: [u32; 3],
    threads_x: Option<(u32, u32)>, // (logical threads, group width)
}

impl Dispatch {
    fn new(counts: [u32; 3], flags: u32, size: Option<[u32; 3]>) -> Option<Self> {
        if flags & THREAD_COUNT == 0 {
            return Some(Self {
                groups: counts,
                threads_x: None,
            });
        }
        let size = size.filter(|size| !size.contains(&0))?;
        Some(Self {
            groups: std::array::from_fn(|i| counts[i].div_ceil(size[i])),
            threads_x: Some((counts[0], size[0])),
        })
    }

    fn parts(&self, has_group_base: bool) -> Option<impl Iterator<Item = (u32, [u32; 3], u32)>> {
        let [x, y, z] = self.groups;
        if y > MAX_GROUPS || z > MAX_GROUPS || (x > MAX_GROUPS && !has_group_base) {
            return None;
        }
        Some((0..x).step_by(MAX_GROUPS as usize).map(move |base| {
            let groups = [(x - base).min(MAX_GROUPS), y, z];
            let tail = self
                .threads_x
                .map_or(0, |(threads, width)| threads - base * width);
            (base, groups, tail)
        }))
    }
}

// RootType::Other codes from the shader emitter's DispatchX ABI.
#[cfg(windows)]
const USER_GROUP_BASE: u32 = 0x104;
#[cfg(windows)]
const USER_DISPATCH_THREADS_X: u32 = 0x106;

#[cfg(windows)]
pub(crate) fn uses_root_word(root: victory_shader::RootType) -> bool {
    matches!(
        root,
        victory_shader::RootType::Other(USER_GROUP_BASE | USER_DISPATCH_THREADS_X)
    )
}

#[cfg(windows)]
pub(crate) fn record(
    list: &ID3D12GraphicsCommandList,
    pipeline: &HostPipeline,
    counts: [u32; 3],
    flags: u32,
) -> bool {
    let stage = &pipeline.stages[0];
    let shader = &stage.translated().host;
    let word = |kind| {
        let (param, _) = stage.user_data?;
        let word = shader
            .user_data
            .iter()
            .find(|u| u.root == victory_shader::RootType::Other(kind))?;
        Some((param, word.index))
    };
    let size = if flags & THREAD_COUNT != 0 {
        compute_size(&shader.dxil)
    } else {
        None
    };
    let Some(dispatch) = Dispatch::new(counts, flags, size) else {
        return false;
    };
    let group_base = word(USER_GROUP_BASE);
    let tail_word = word(USER_DISPATCH_THREADS_X);
    let Some(parts) = dispatch.parts(group_base.is_some()) else {
        return false;
    };
    // SAFETY: the open list has the pipeline bound. Each optional word
    // belongs to the pipeline's compute root signature.
    unsafe {
        for (base, [x, y, z], tail) in parts {
            if let Some((param, index)) = group_base {
                list.SetComputeRoot32BitConstant(param, base, index);
            }
            if let Some((param, index)) = tail_word {
                list.SetComputeRoot32BitConstant(param, tail, index);
            }
            list.Dispatch(x, y, z);
        }
        for (param, index) in [group_base, tail_word].into_iter().flatten() {
            list.SetComputeRoot32BitConstant(param, 0, index);
        }
    }
    true
}

/// The host compute shader's dimensions in PSV runtime info version 2+.
fn compute_size(dxil: &[u8]) -> Option<[u32; 3]> {
    use victory_shader::{Container, FourCC};
    let container = Container::parse(dxil).ok()?;
    let data = container.part(FourCC::PSV0)?.data;
    let word = |at| Some(u32::from_le_bytes(data.get(at..at + 4)?.try_into().ok()?));
    if word(0)? < 48 || *data.get(28)? != 5 {
        return None;
    }
    Some([word(40)?, word(44)?, word(48)?])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thread_counts_round_up_in_each_dimension_and_group_counts_stay_as_given() {
        let counts = [65, 3, 5];
        let size = Some([64, 2, 4]);
        assert_eq!(Dispatch::new(counts, 0x21, size).unwrap().groups, [2, 2, 2]);
        assert_eq!(Dispatch::new(counts, 0x01, None).unwrap().groups, counts);
        assert!(Dispatch::new(counts, 0x20, None).is_none());
        assert!(Dispatch::new(counts, 0x20, Some([64, 0, 4])).is_none());
    }

    #[test]
    fn oversized_x_dispatch_preserves_group_ids_and_adjusts_tail() {
        let threads = 64 * MAX_GROUPS + 65;
        let dispatch = Dispatch::new([threads, 2, 1], 0x20, Some([64, 1, 1])).unwrap();
        assert_eq!(
            dispatch.parts(true).unwrap().collect::<Vec<_>>(),
            [
                (0, [MAX_GROUPS, 2, 1], threads),
                (MAX_GROUPS, [2, 2, 1], 65)
            ]
        );
        assert!(dispatch.parts(false).is_none());
        let dispatch = Dispatch::new([1, MAX_GROUPS + 1, 1], 0, None).unwrap();
        assert!(dispatch.parts(true).is_none());
    }

    #[test]
    fn reads_group_dimensions_from_host_psv() {
        let mut container = [0; 96];
        container[..4].copy_from_slice(b"DXBC");
        for (offset, word) in [
            (24, 96u32),
            (28, 1),
            (32, 36),
            (40, 52),
            (44, 48),
            (84, 64),
            (88, 2),
            (92, 1),
        ] {
            container[offset..offset + 4].copy_from_slice(&word.to_le_bytes());
        }
        container[36..40].copy_from_slice(b"PSV0");
        container[72] = 5;
        assert_eq!(compute_size(&container), Some([64, 2, 1]));
        container[72] = 0;
        assert_eq!(compute_size(&container), None);
    }
}
