//! A scalar bitwise instruction can pack a number drawn from a join of masks
//! and numbers. Preserve the numeric operation when its inputs have exact words.
use super::{Bin, Bool, Fields, LaneExit, Lower, MaskData, Operand, Reg, V, Val, Value};

impl Lower<'_> {
    pub(super) fn packed_mask_word(&mut self, value: Value) -> Option<V> {
        if self.mask_data != MaskData::Console {
            return None;
        }
        // A failed proof must not make an unrelated join live.
        let mut work = Vec::new();
        let result = self.scalar_bitwise_word(value, &mut 64, &mut work)?;
        // Once numeric phis are being filled, earlier phis may already have
        // substituted unused inputs. New dependencies can no longer be retained.
        if self.data_phis.is_empty() && work.iter().any(|v| !self.packed_phis.contains(v)) {
            return None;
        }
        while let Some(input) = work.pop() {
            if let Some(&(b, i)) = self.phi_at.get(&input)
                && self.packed_phis.insert(input)
            {
                self.read_phis.insert(input);
                let block = &self.x.p.blocks[b];
                for (&arg, &from) in block.phis[i].args.iter().zip(&block.preds) {
                    if !self.impossible_forward_edge(from, b) {
                        work.push(arg);
                    }
                }
            }
        }
        self.vals.insert(value, Val::I32(result));
        Some(result)
    }

    fn scalar_bitwise_word(
        &mut self,
        value: Value,
        budget: &mut u32,
        phis: &mut Vec<Value>,
    ) -> Option<V> {
        *budget = budget.checked_sub(1)?;
        if let Some(&Val::I32(word)) = self.vals.get(&value) {
            if self.phi_at.contains_key(&value) {
                phis.push(value);
            }
            return Some(word);
        }
        let &block = self.def_block.get(&value)?;
        let inst = self.x.p.blocks[block]
            .insts
            .iter()
            .find(|i| i.writes.iter().any(|w| w.1 == value))?;
        let Fields::Sop {
            dst: Some(Operand::Scalar { code: dest, .. }),
            src,
            ..
        } = &inst.fields
        else {
            return None;
        };
        if inst.write(Reg::S(*dest)) != Some(value) {
            return None;
        }
        let op = match inst.name() {
            "s_and_b32" => Bin::And,
            "s_or_b32" => Bin::Or,
            "s_xor_b32" => Bin::Xor,
            _ => return None,
        };
        let [a, b] = src.as_slice() else { return None };
        let mut arg = |this: &mut Self, op: &Operand| match *op {
            Operand::Literal(k) => Some(this.s.m.i32(k)),
            Operand::Const(k) => crate::resource::inline(k).map(|v| this.s.m.i32(v)),
            Operand::Scalar { code, .. } => {
                this.scalar_bitwise_word(inst.read(Reg::S(code))?, budget, phis)
            }
            _ => None,
        };
        let at = if block == self.cur_ir {
            self.cur
        } else {
            self.tail[block]?
        };
        let x = arg(self, a)?;
        let y = arg(self, b)?;
        Some(self.s.m.bin(at, op, x, y))
    }

    /// A scalar word at a join need not come from an edge no lane can take.
    /// Only ordinary forward branches qualify: a loop limit can force an exit.
    pub(super) fn impossible_forward_edge(&self, from: usize, to: usize) -> bool {
        let LaneExit::Branch {
            when,
            taken,
            not_taken,
        } = self.x.st.exits[from]
        else {
            return false;
        };
        if taken == not_taken
            || self.rpo[taken] <= self.rpo[from]
            || self.rpo[not_taken] <= self.rpo[from]
        {
            return false;
        }
        (when == Bool::FALSE && to == taken) || (when == Bool::TRUE && to == not_taken)
    }
}
