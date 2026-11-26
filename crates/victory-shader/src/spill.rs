//! Identifies the spilled user-data table used by XboxBvhBuilder compute shaders.

use crate::native::{HwStage, NativeProgram, StageIo};

/// Whether the native compute layout starts a spill table at dword 16 and
/// leaves at least two user SGPRs for its address.
pub fn is_spilled(native: &NativeProgram) -> bool {
    let StageIo::Cs(cs) = &native.io else {
        return false;
    };
    let registers = cs.user_sgprs();
    native.stage == HwStage::Cs
        && registers >= 2
        && native
            .layout
            .iter()
            .filter(|e| e.sgprs > 0 && e.user_data >= registers)
            .map(|e| e.user_data)
            .min()
            == Some(16)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{RootParam, RootType, native::ComputeIo};

    #[test]
    fn spill_requires_a_compute_table_starting_at_dword_16() {
        let mut native = NativeProgram {
            stage: HwStage::Cs,
            wave64: false,
            vgprs: 0,
            layout: vec![RootParam {
                root: RootType::Descriptor,
                table_offset_dwords: 0,
                sgprs: 2,
                base_register: 0,
                user_data: 16,
                slot: 0,
            }],
            used: Vec::new(),
            io: StageIo::Cs(ComputeIo {
                rsrc2: 2 << 1,
                threads: [1; 3],
            }),
            code: Vec::new(),
        };
        assert!(is_spilled(&native));
        native.layout[0].user_data = 17;
        assert!(!is_spilled(&native));
        native.layout[0].user_data = 16;
        native.stage = HwStage::Ps;
        assert!(!is_spilled(&native));
        native.stage = HwStage::Cs;
        for registers in [1, 18] {
            native.io = StageIo::Cs(ComputeIo {
                rsrc2: registers << 1,
                threads: [1; 3],
            });
            assert!(!is_spilled(&native));
        }
    }
}
