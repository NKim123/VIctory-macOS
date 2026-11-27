//! Classify ray passes from their original containers, before translation or
//! shader spilling. The default keeps the frame-based builder and ray-skip
//! controls. Static BVH mode allows traversal of an existing structure.

use victory_shader::{Container, FourCC, NativeProgram, native::StageIo};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Work {
    Builder,
    Traversal,
    Activate,
    RayBins,
    Resolve,
    Other,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Policy {
    Default,
    Enabled,
    Disabled,
    StaticBvh,
}

impl Policy {
    fn parse(value: Option<&str>) -> Self {
        match value {
            Some("0") => Self::Disabled,
            Some("1") => Self::Enabled,
            Some("static-bvh") => Self::StaticBvh,
            _ => Self::Default,
        }
    }

    pub(super) fn classifies(self) -> bool {
        matches!(self, Self::Disabled | Self::StaticBvh)
    }

    pub(super) fn omission(self, pass: Work) -> Option<&'static str> {
        match (self, pass) {
            (Self::StaticBvh, Work::Builder) => Some("static BVH mode: dynamic builder omitted"),
            (Self::Disabled, Work::Builder) => Some("ray tracing disabled: BVH builder"),
            (Self::Disabled, Work::Traversal) => Some("ray tracing disabled: BVH traversal"),
            (Self::Disabled, Work::Activate) => Some("ray tracing disabled: ray activation"),
            (Self::Disabled, Work::RayBins) => Some("ray tracing disabled: ray bin preparation"),
            (Self::Disabled, Work::Resolve) => Some("ray tracing disabled: ray hit resolve"),
            _ => None,
        }
    }
}

#[cfg(windows)]
pub(super) fn policy() -> Policy {
    static POLICY: std::sync::OnceLock<Policy> = std::sync::OnceLock::new();
    *POLICY.get_or_init(|| {
        let value = std::env::var("VICTORY_RAY_TRACING").ok();
        let policy = Policy::parse(value.as_deref());
        if value.is_some() {
            tracing::info!(?policy, "configured ray-tracing passes");
        }
        policy
    })
}

pub(super) fn classify(stage: &str, container: &[u8], mode: Policy) -> Work {
    if !mode.classifies() || (mode == Policy::StaticBvh && stage != "compute") {
        return Work::Other;
    }
    if mode == Policy::Disabled && stage == "compute" {
        let named = |name: &[u8]| container.windows(name.len()).any(|w| w == name);
        if named(b"CS_RaytracingResolve") {
            return Work::Resolve;
        }
        if named(b"CS_ScreenProbes_ActivateRays") {
            return Work::Activate;
        }
    }
    let Ok(container) = Container::parse(container) else {
        return Work::Other;
    };
    if mode == Policy::Disabled
        && stage == "compute"
        && container.part(FourCC::PSV0).is_some_and(|psv| {
            psv.data.split(|b| *b == 0).any(|entry| {
                [
                    b"CS_ScreenProbes_EvaluateRayRequests_RayBinning_".as_slice(),
                    b"CS_RayBudgeting_Cull_RayBinning_",
                    b"CS_ScreenProbes_GenerateRayBins_",
                    b"CS_GenerateRayBins_",
                    b"CS_GenerateRayBinOffsets_",
                ]
                .iter()
                .any(|name| entry.starts_with(name))
            })
        })
    {
        return Work::RayBins;
    }
    let programs = container
        .part(FourCC::DXIL)
        .or_else(|| container.part(FourCC::ACCC))
        .and_then(|p| victory_shader::native_programs(p.data).ok());
    let Some(programs) = programs else {
        return Work::Other;
    };
    if stage == "compute" && programs.iter().any(is_spilled) {
        return Work::Builder;
    }
    if mode == Policy::Disabled && programs.iter().any(|p| traverses(&p.code)) {
        return Work::Traversal;
    }
    Work::Other
}

fn traverses(code: &[u8]) -> bool {
    // The fast scan excludes programs without either MIMG header. Only the
    // decoder can distinguish a real instruction from a literal or unreachable data.
    code.as_chunks::<4>().0.iter().any(|&bytes| {
        let word = u32::from_le_bytes(bytes);
        word >> 26 == 0x3c && (word >> 18) & 0x7e == 0x66 && word & 1 != 0
    }) && victory_shader::gfx10::decode(code).is_ok_and(|instructions| {
        instructions.iter().any(|i| {
            matches!(
                i.name(),
                Some("image_bvh_intersect_ray" | "image_bvh64_intersect_ray")
            )
        })
    })
}

// The source builder's compute programs use a user-data spill table that
// starts at dword 16. Runtime creation also identifies them by DLL origin.
fn is_spilled(program: &NativeProgram) -> bool {
    let StageIo::Cs(cs) = &program.io else {
        return false;
    };
    let registers = u32::from(cs.user_sgprs());
    let first = program
        .layout
        .iter()
        .filter(|entry| entry.sgprs > 0 && u32::from(entry.user_data) >= registers)
        .map(|entry| entry.user_data)
        .min();
    registers >= 2 && first == Some(16)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_policy_omits_only_its_selected_passes() {
        let passes = [
            Work::Builder,
            Work::Traversal,
            Work::Activate,
            Work::RayBins,
            Work::Resolve,
        ];
        let off = Policy::parse(Some("0"));
        let partial = Policy::parse(Some("static-bvh"));
        assert!(partial.classifies());
        for pass in passes {
            assert!(off.omission(pass).is_some());
            assert_eq!(partial.omission(pass).is_some(), pass == Work::Builder);
            assert!(Policy::parse(Some("1")).omission(pass).is_none());
            assert!(Policy::parse(None).omission(pass).is_none());
        }
        assert!(off.omission(Work::Other).is_none());
        assert_eq!(
            classify("compute", b"CS_RaytracingResolve", off),
            Work::Resolve
        );
        assert_eq!(
            classify("compute", b"CS_ScreenProbes_ActivateRays", off),
            Work::Activate
        );
        assert_eq!(
            classify("compute", b"CS_ScreenProbes_Solve", off),
            Work::Other
        );
        assert_eq!(classify("pixel", b"CS_RaytracingResolve", off), Work::Other);
    }

    #[test]
    fn ray_bin_labels_are_matched_in_psv() {
        let entry = b"CS_GenerateRayBinOffsets_Wrapped\0";
        let mut container = vec![0; 44];
        container[..4].copy_from_slice(b"DXBC");
        container[24..28].copy_from_slice(&(44 + entry.len() as u32).to_le_bytes());
        container[28..32].copy_from_slice(&1u32.to_le_bytes());
        container[32..36].copy_from_slice(&36u32.to_le_bytes());
        container[36..40].copy_from_slice(b"PSV0");
        container[40..44].copy_from_slice(&(entry.len() as u32).to_le_bytes());
        container.extend_from_slice(entry);
        assert_eq!(
            classify("compute", &container, Policy::Disabled),
            Work::RayBins
        );
        assert_eq!(
            classify("compute", &container, Policy::StaticBvh),
            Work::Other
        );
    }

    #[test]
    fn only_executed_bvh_opcodes_identify_a_traversal_shader() {
        let run = |words: &[u32]| {
            traverses(
                &words
                    .iter()
                    .flat_map(|w| w.to_le_bytes())
                    .collect::<Vec<_>>(),
            )
        };
        for header in [0xf198_9f07, 0xf19c_9f07] {
            assert!(run(&[
                header, 0x020d01, 0x05040302, 0x09080706, 0x000c0b0a, 0xbf810000
            ]));
            assert!(!run(&[0xbe98_03ff, header, 0xbf810000])); // scalar literal
            assert!(!run(&[0xbf810000, header])); // unreachable tail
        }
    }
}
