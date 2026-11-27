//! Wave ORs that only steer a loop over each lane's own bits.
//!
//! The title's light loops are "scalarised": the lanes' light masks are ORed over the wave and
//! the wave walks the set bits of the result together, each lane working only where its own
//! mask has the bit:
//!
//! ```text
//! merged = WaveActiveBitOr(mask);            // over the lanes that run
//! while (merged != 0) {
//!     bit = firstbitlow(merged);
//!     if (mask & (1 << bit)) { ... }         // the body
//!     merged ^= 1 << bit;
//! }
//! ```
//!
//! A console wave64 program in two host waves of 32 cannot exchange the halves' ORs where not
//! every lane runs (a group barrier must be reached by every thread). It does not have to: as
//! one thread runs it, the loop visits the bits of `merged` in rising order and does something
//! only at the bits of the thread's own mask, so any `merged` that holds the thread's own bits
//! gives that thread the same run. The host wave's own OR is such a value.
//!
//! That holds only if the OR is used for nothing else, and if a round of the loop at a bit the
//! thread does not have leaves nothing behind. [`find`] proves both for each wave OR, with no
//! instruction sequence taken on trust:
//!
//! - the OR and what derives from it as "the bits still to visit" (copies, the loop's join, the
//!   first-lane read of it, the set less the visited bit) are read only by the loop's own
//!   steps: the test for an empty set, the lowest set bit, the removal of that bit, the test
//!   that ends the loop;
//! - the lane's test is its own bit of the very value that was ORed, and "the body" is what a
//!   lane only reaches with that test true in the same round;
//! - outside the body the loop has no side effect and no cross-lane step, writes nothing that
//!   is read after the loop or carried into a later round, and is left only by the test for
//!   "no bits left";
//! - no lane mask that depends on the test is used outside the round that made it.
//!
//! Sites outside this proof use the normal cross-half lowering.
//! `VICTORY_OWN_BITS=0` disables this optimization when translating. Use a
//! fresh shader cache for each setting when comparing the two forms.

use std::collections::{HashMap, HashSet};

use super::wave::{self, Op, Shape};
use crate::gfx10::{Fields, NULL, Operand};
use crate::ir::{EXEC, Exit, Inst, Program, Reg, Value};
use crate::structure::{Atom, Bool, LaneExit, Set, Structured};

type Why = &'static str;

const USED: Why = "a wave OR used as data";
const NO_LOOP: Why = "a wave OR that steers no loop over its bits";
const SHAPE: Why = "a loop over a wave OR's bits of another shape";
const TEST: Why = "a loop over a wave OR's bits whose lanes do not test their own bit";
const LEAVES: Why = "a value of a round over a wave OR's bits that outlives the round";
const EFFECT: Why = "an effect in a loop over a wave OR's bits outside the lanes' own bits";
const CROSS: Why = "a cross-lane step in a loop over a wave OR's bits";
const EXIT: Why = "a loop over a wave OR's bits left another way";
const STALE: Why = "a lane's own-bit test used outside its round";
const MASK: Why = "a wave OR over a mask a running lane may be outside";

pub(super) fn enabled() -> bool {
    #[cfg(test)]
    if let Some(enabled) = TEST_ENABLED.get() {
        return enabled;
    }
    std::env::var_os("VICTORY_OWN_BITS").is_none_or(|value| value != "0")
}

#[cfg(test)]
thread_local! {
    static TEST_ENABLED: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

#[cfg(test)]
pub(super) fn with_enabled<T>(enabled: bool, f: impl FnOnce() -> T) -> T {
    struct Restore(Option<bool>);
    impl Drop for Restore {
        fn drop(&mut self) {
            TEST_ENABLED.set(self.0);
        }
    }
    let _restore = Restore(TEST_ENABLED.replace(Some(enabled)));
    f()
}

/// What [`find`] proved.
#[derive(Debug, Default)]
pub(super) struct OwnBits {
    /// Wave ORs (the values their sections leave) a host wave may take over its own lanes.
    pub totals: HashSet<Value>,
    /// Results of first-lane reads that can use the lane's own remaining bits.
    pub first_reads: HashSet<Value>,
    /// Loop and preheader blocks proved free of cross-half operations.
    pub local_blocks: Set<usize>,
    /// Whole-wave sections whose only live output is the proven OR.
    pub local_sections: Set<(usize, usize)>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum User {
    Inst(usize, usize),
    Phi(usize, usize),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Arg {
    Val(Value),
    K(u32),
    Other,
}

struct Ctx<'a> {
    p: &'a Program,
    vgprs: u16,
    st: &'a Structured,
    users: HashMap<Value, Vec<User>>,
    /// The instruction writing each value.
    def: HashMap<Value, (usize, usize)>,
    /// Values an instruction reads, directly or through joins.
    live: HashSet<Value>,
    idom: Vec<usize>,
    sections: Vec<HashMap<usize, wave::Section>>,
}

/// Values read by an instruction, including EXEC for mask liveness. Ignore a
/// VGPR's old destination value when every lane reaching the write runs it.
fn used(p: &Program, vgprs: u16, st: &Structured, b: usize, i: usize) -> Vec<Value> {
    let inst = &p.blocks[b].insts[i];
    let ops = crate::ir::operand_reads(p, vgprs, inst);
    let guarded = st.guards[b][i].is_some();
    inst.reads
        .iter()
        .filter(|&&(r, _)| match r {
            Reg::V(_) => ops.contains(&r) || guarded,
            _ => true,
        })
        .map(|r| r.1)
        .collect()
}

fn value(inst: &Inst, op: &Operand) -> Option<Value> {
    match *op {
        Operand::Scalar { code, .. } if code != NULL => inst.read(Reg::S(code)),
        Operand::Vector { reg, .. } => inst.read(Reg::V(reg)),
        _ => None,
    }
}

/// The sources of a scalar or unmodified vector operation.
fn args(inst: &Inst) -> Vec<Arg> {
    let src: &[Operand] = match &inst.fields {
        Fields::Sop { src, .. } => src,
        Fields::Valu(v)
            if v.dpp.is_none()
                && v.sdwa.is_none()
                && v.abs == 0
                && v.neg == 0
                && !v.clamp
                && v.omod == 0
                && v.opsel == 0 =>
        {
            &v.src
        }
        _ => &[],
    };
    src.iter()
        .map(|o| match (value(inst, o), o) {
            (Some(v), _) => Arg::Val(v),
            (None, &Operand::Const(c)) => crate::resource::inline(c).map_or(Arg::Other, Arg::K),
            (None, &Operand::Literal(x)) => Arg::K(x),
            _ => Arg::Other,
        })
        .collect()
}

/// The other source of a two-source operation that reads `a`.
fn other(inst: &Inst, a: Value) -> Option<Arg> {
    match args(inst)[..] {
        [Arg::Val(x), o] if x == a => Some(o),
        [o, Arg::Val(x)] if x == a => Some(o),
        _ => None,
    }
}

/// The scalar (not SCC) or vector dword an instruction writes, if it writes one.
fn result(inst: &Inst) -> Option<Value> {
    let mut w = inst.writes.iter().filter(|w| w.0 != Reg::Scc);
    match (w.next(), w.next()) {
        (Some(&(_, v)), None) => Some(v),
        _ => None,
    }
}

/// The same test as `wave_local`'s and `Lower::inst`'s `effect`, and the messages a program
/// sends.
fn effect(inst: &Inst) -> bool {
    let name = inst.name();
    matches!(inst.fields, Fields::Exp { .. })
        || matches!(inst.fields, Fields::Flat { .. }) && !name.starts_with("global_load_")
        || name.contains("store")
        || name.contains("atomic")
        || matches!(inst.fields, Fields::Ds { .. })
            && !(name.starts_with("ds_read")
                || name.starts_with("ds_swizzle")
                || name.contains("permute"))
        || matches!(name, "s_barrier" | "s_sendmsg" | "s_sendmsghalt")
}

impl Ctx<'_> {
    fn new<'a>(p: &'a Program, vgprs: u16, st: &'a Structured) -> Ctx<'a> {
        let mut users: HashMap<Value, Vec<User>> = HashMap::new();
        let mut def = HashMap::new();
        let mut live: HashSet<Value> = HashSet::new();
        for (b, block) in p.blocks.iter().enumerate() {
            for (k, phi) in block.phis.iter().enumerate() {
                for &a in &phi.args {
                    users.entry(a).or_default().push(User::Phi(b, k));
                }
            }
            for (i, inst) in block.insts.iter().enumerate() {
                for v in used(p, vgprs, st, b, i) {
                    users.entry(v).or_default().push(User::Inst(b, i));
                    live.insert(v);
                }
                for &(_, w) in &inst.writes {
                    def.insert(w, (b, i));
                }
            }
        }
        // through joins
        let mut changed = true;
        while changed {
            changed = false;
            for block in &p.blocks {
                for phi in &block.phis {
                    if live.contains(&phi.value) {
                        for &a in &phi.args {
                            changed |= live.insert(a);
                        }
                    }
                }
            }
        }
        let order = crate::ir::reverse_postorder(&p.blocks);
        let mut index = vec![usize::MAX; p.blocks.len()];
        for (n, &b) in order.iter().enumerate() {
            index[b] = n;
        }
        let idom = crate::ir::idoms(&p.blocks, &order, &index);
        let sections = (0..p.blocks.len()).map(|b| wave::sections(p, b)).collect();
        Ctx {
            p,
            vgprs,
            st,
            users,
            def,
            live,
            idom,
            sections,
        }
    }

    fn inst(&self, u: (usize, usize)) -> &Inst {
        &self.p.blocks[u.0].insts[u.1]
    }

    fn at(&self, u: (usize, usize)) -> usize {
        self.inst(u).machine.offset
    }

    /// Whether block `a` dominates block `b`.
    fn dominates(&self, a: usize, mut b: usize) -> bool {
        loop {
            if a == b {
                return true;
            }
            let d = self.idom.get(b).copied().unwrap_or(usize::MAX);
            if d == b || d == usize::MAX {
                return false;
            }
            b = d;
        }
    }

    fn readers(&self, v: Value) -> Vec<(usize, usize)> {
        let mut out: Vec<(usize, usize)> = self
            .users
            .get(&v)
            .into_iter()
            .flatten()
            .filter_map(|u| match *u {
                User::Inst(b, i) => Some((b, i)),
                User::Phi(..) => None,
            })
            .collect();
        out.sort_unstable();
        out.dedup();
        out
    }

    /// The joins reading `v` whose value something reads.
    fn joins(&self, v: Value) -> Vec<(usize, usize)> {
        let mut out: Vec<(usize, usize)> = self
            .users
            .get(&v)
            .into_iter()
            .flatten()
            .filter_map(|u| match *u {
                User::Phi(b, k) if self.live.contains(&self.p.blocks[b].phis[k].value) => {
                    Some((b, k))
                }
                _ => None,
            })
            .collect();
        out.sort_unstable();
        out.dedup();
        out
    }

    /// The loop of header `h`: the blocks that reach one of its back edges without passing it.
    fn natural_loop(&self, h: usize) -> Vec<bool> {
        let mut inside = vec![false; self.p.blocks.len()];
        let mut work: Vec<usize> = self.p.blocks[h]
            .preds
            .iter()
            .copied()
            .filter(|&q| self.dominates(h, q))
            .collect();
        if work.is_empty() {
            return inside;
        }
        inside[h] = true;
        while let Some(b) = work.pop() {
            if inside[b] {
                continue;
            }
            inside[b] = true;
            work.extend(self.p.blocks[b].preds.iter().copied());
        }
        inside
    }

    /// Whether `b` holds only in lanes where `a` does.
    fn implied(&self, b: Bool, a: Atom, memo: &mut HashMap<Bool, bool>) -> bool {
        if b == Bool::FALSE {
            return true;
        }
        if let Some(&r) = memo.get(&b) {
            return r;
        }
        let r = match self.st.bools.node(b) {
            None => false,
            Some((x, lo, _)) if x == a => lo == Bool::FALSE,
            Some((_, lo, hi)) => self.implied(lo, a, memo) && self.implied(hi, a, memo),
        };
        memo.insert(b, r);
        r
    }

    /// The conditional branch reading `scc`, as (its place, the block it goes to when SCC is
    /// set, the other one).
    fn branch(&self, scc: Value) -> Option<((usize, usize), usize, usize)> {
        let [u] = self.readers(scc)[..] else {
            return None;
        };
        let Exit::Branch { taken, not_taken } = self.p.blocks[u.0].exit else {
            return None;
        };
        if u.1 + 1 != self.p.blocks[u.0].insts.len()
            || !matches!(self.st.exits[u.0], LaneExit::Branch {
                taken: t, not_taken: f, ..
            } if t == taken && f == not_taken)
        {
            return None;
        }
        match self.inst(u).name() {
            "s_cbranch_scc1" => Some((u, taken, not_taken)),
            "s_cbranch_scc0" => Some((u, not_taken, taken)),
            _ => None,
        }
    }

    /// Prove the loop fed by a wave OR `t` of `x` over `mask` in block `sb`.
    fn site(
        &self,
        sb: usize,
        t: Value,
        x: Value,
        mask: Value,
    ) -> Result<(Option<Value>, Vec<usize>), (usize, Why)> {
        let p = self.p;
        let st = self.st;
        let n = p.blocks.len();
        let here = p.blocks[sb].offset;
        let is_mask = |v: Value| st.masks.contains_key(&v) || st.highs.contains_key(&v);
        // every lane that runs the section is one of the lanes ORed
        let &mb = st.masks.get(&mask).ok_or((here, MASK))?;
        if st.bools.meet(&[(st.reach[sb], true), (mb, false)]) {
            return Err((here, MASK));
        }
        // ---- the bits still to visit, and the loop's own steps ----
        let mut rem: HashSet<Value> = HashSet::from([t]);
        let mut pattern: HashSet<(usize, usize)> = HashSet::new();
        let mut first_read = None;
        // an unguarded copy
        let copy = |u: (usize, usize)| -> Option<Value> {
            let i = self.inst(u);
            (matches!(i.name(), "v_mov_b32" | "s_mov_b32")
                && args(i).len() == 1
                && st.guards[u.0][u.1].is_none())
            .then(|| result(i))
            .flatten()
        };
        let mut init = vec![t];
        // (SCC, whether it is set when the OR is empty)
        let mut zero: Option<(Value, bool)> = None;
        for u in self.readers(t) {
            let i = self.inst(u);
            let name = i.name();
            if matches!(
                name,
                "s_cmp_eq_u32" | "s_cmp_lg_u32" | "s_cmp_eq_i32" | "s_cmp_lg_i32"
            ) && other(i, t) == Some(Arg::K(0))
                && zero.is_none()
                && let Some(scc) = i.write(Reg::Scc)
            {
                zero = Some((scc, name.contains("_eq_")));
                rem.insert(scc);
            } else if let Some(w) = copy(u) {
                init.push(w);
                rem.insert(w);
            } else {
                return Err((self.at(u), USED));
            }
            pattern.insert(u);
        }
        // the loop's join of them
        let mut join: Option<(usize, usize)> = None;
        for &v in &init {
            if v != t
                && let Some(&u) = self.readers(v).first()
            {
                return Err((self.at(u), USED));
            }
            for j in self.joins(v) {
                if join.is_some_and(|q| q != j) {
                    return Err((p.blocks[j.0].offset, USED));
                }
                join = Some(j);
            }
        }
        let (h, k) = join.ok_or((here, NO_LOOP))?;
        let phi = &p.blocks[h].phis[k];
        let r = phi.value;
        rem.insert(r);
        let head = p.blocks[h].offset;
        let lp = self.natural_loop(h);
        if !lp[h] || lp[sb] {
            return Err((head, NO_LOOP));
        }
        // the set as a scalar: the first lane's copy of it
        let cur = if matches!(phi.reg, Reg::V(_)) {
            let [u] = self.readers(r)[..] else {
                return Err((head, USED));
            };
            let i = self.inst(u);
            if i.name() != "v_readfirstlane_b32"
                || args(i).as_slice() != [Arg::Val(r)]
                || u.0 != h
                || st.guards[u.0][u.1].is_some()
            {
                return Err((self.at(u), USED));
            }
            pattern.insert(u);
            let c = result(i).ok_or((self.at(u), SHAPE))?;
            first_read = Some(c);
            if !self.joins(r).is_empty() {
                return Err((head, USED));
            }
            c
        } else {
            r
        };
        rem.insert(cur);
        // its lowest bit, the set without it, the tests for "that was the last"
        let mut idx: Option<Value> = None;
        let mut next: Option<(Value, Value)> = None;
        // (SCC, whether it is set while bits remain)
        let mut more: Vec<(Value, bool)> = Vec::new();
        let mut bits: Vec<Value> = Vec::new();
        for u in self.readers(cur) {
            let i = self.inst(u);
            let name = i.name();
            if !lp[u.0] {
                return Err((self.at(u), USED));
            }
            match (name, &args(i)[..]) {
                ("s_ff1_i32_b32", [Arg::Val(_)]) if idx.is_none() => idx = result(i),
                ("s_xor_b32", _) | ("s_andn2_b32", [Arg::Val(_), Arg::Val(_)])
                    if next.is_none() =>
                {
                    let bit = match (name, &args(i)[..]) {
                        ("s_andn2_b32", [Arg::Val(c), Arg::Val(b)]) if *c == cur && *b != cur => *b,
                        ("s_xor_b32", _) => match other(i, cur) {
                            Some(Arg::Val(b)) if b != cur => b,
                            _ => return Err((self.at(u), USED)),
                        },
                        _ => return Err((self.at(u), USED)),
                    };
                    let w = result(i).ok_or((self.at(u), SHAPE))?;
                    next = Some((w, bit));
                    bits.push(bit);
                    if let Some(scc) = i.write(Reg::Scc) {
                        more.push((scc, true));
                        rem.insert(scc);
                    }
                }
                ("s_cmp_eq_u32" | "s_cmp_lg_u32" | "s_cmp_eq_i32" | "s_cmp_lg_i32", _) => {
                    let (Some(Arg::Val(b)), Some(scc)) = (other(i, cur), i.write(Reg::Scc)) else {
                        return Err((self.at(u), USED));
                    };
                    bits.push(b);
                    more.push((scc, name.contains("_lg_")));
                    rem.insert(scc);
                }
                _ => return Err((self.at(u), USED)),
            }
            pattern.insert(u);
        }
        if cur != r && !self.joins(cur).is_empty() {
            return Err((head, USED));
        }
        let (Some(idx), Some((nx, bit))) = (idx, next) else {
            return Err((head, SHAPE));
        };
        if bits.iter().any(|&b| b != bit) {
            return Err((head, SHAPE));
        }
        // the bit: 1 shifted by the lowest set bit's number (a shift takes five bits of it)
        let made = |v: Value| self.def.get(&v).map(|&u| (u, self.inst(u)));
        let shifted = match made(bit) {
            Some((u, i)) if i.name() == "s_lshl_b32" && lp[u.0] => match args(i)[..] {
                [Arg::K(1), Arg::Val(sh)] => Some(sh),
                _ => None,
            },
            _ => None,
        };
        let by_idx = shifted.is_some_and(|sh| {
            sh == idx
                || made(sh).is_some_and(|(_, i)| {
                    i.name() == "s_and_b32" && other(i, idx) == Some(Arg::K(31))
                })
        });
        if !by_idx {
            return Err((head, SHAPE));
        }
        // the set less the bit: back into the join, or tested for empty
        rem.insert(nx);
        let mut nexts = vec![nx];
        for u in self.readers(nx) {
            let i = self.inst(u);
            let name = i.name();
            if matches!(
                name,
                "s_cmp_eq_u32" | "s_cmp_lg_u32" | "s_cmp_eq_i32" | "s_cmp_lg_i32"
            ) && other(i, nx) == Some(Arg::K(0))
                && let Some(scc) = i.write(Reg::Scc)
            {
                more.push((scc, name.contains("_lg_")));
                rem.insert(scc);
            } else if let Some(w) = copy(u) {
                nexts.push(w);
                rem.insert(w);
            } else {
                return Err((self.at(u), USED));
            }
            pattern.insert(u);
        }
        for &v in &nexts {
            if v != nx
                && let Some(&u) = self.readers(v).first()
            {
                return Err((self.at(u), USED));
            }
            if self.joins(v).iter().any(|&j| j != (h, k)) {
                return Err((head, USED));
            }
        }
        for (q, a) in p.blocks[h].preds.iter().zip(&phi.args) {
            let ok = if lp[*q] {
                nexts.contains(a)
            } else {
                init.contains(a)
            };
            if !ok {
                return Err((head, SHAPE));
            }
        }
        // the loop goes on while bits remain and is left when none does
        let mut latch = None;
        for &(scc, set_while_more) in &more {
            if !self.joins(scc).is_empty() {
                return Err((head, USED));
            }
            if self.readers(scc).is_empty() {
                continue;
            }
            let (u, set, clear) = self.branch(scc).ok_or((head, USED))?;
            let (on, out) = if set_while_more {
                (set, clear)
            } else {
                (clear, set)
            };
            if !lp[u.0] || !lp[on] || lp[out] || latch.is_some() {
                return Err((self.at(u), SHAPE));
            }
            latch = Some((u.0, out));
            pattern.insert(u);
        }
        let (latch, exit) = latch.ok_or((head, SHAPE))?;
        // Both an empty initial set and exhaustion must reach the same continuation.
        let (scc, set_when_empty) = zero.ok_or((here, SHAPE))?;
        if !self.joins(scc).is_empty() {
            return Err((here, USED));
        }
        let (u, set, clear) = self.branch(scc).ok_or((here, USED))?;
        let (none, some) = if set_when_empty {
            (set, clear)
        } else {
            (clear, set)
        };
        if none != exit || lp[u.0] || !self.dominates(sb, u.0) || !self.dominates(u.0, h) {
            return Err((self.at(u), SHAPE));
        }
        // The nonempty path leading to the header, before the first iteration.
        let pre: Vec<bool> = (0..n)
            .map(|b| self.dominates(some, b) && !self.dominates(h, b))
            .collect();
        if some != h && (p.blocks[some].preds[..] != [u.0] || !self.dominates(some, h)) {
            return Err((self.at(u), SHAPE));
        }
        pattern.insert(u);
        // ---- the lane's test of its own bit ----
        // (the atom of the compare, its block, its index)
        let mut tests: Vec<(Atom, usize, usize)> = Vec::new();
        for u in self.readers(bit) {
            let i = self.inst(u);
            if i.name() != "v_and_b32"
                || other(i, bit) != Some(Arg::Val(x))
                || st.guards[u.0][u.1].is_some()
            {
                continue;
            }
            let Some(y) = result(i) else { continue };
            for c in self.readers(y) {
                let ci = self.inst(c);
                if matches!(
                    ci.name(),
                    "v_cmp_ne_u32"
                        | "v_cmpx_ne_u32"
                        | "v_cmp_lg_u32"
                        | "v_cmpx_lg_u32"
                        | "v_cmp_ne_i32"
                        | "v_cmpx_ne_i32"
                        | "v_cmp_lg_i32"
                        | "v_cmpx_lg_i32"
                ) && other(ci, y) == Some(Arg::K(0))
                    && lp[c.0]
                {
                    tests.extend(ci.writes.iter().map(|w| (Atom::Lane(w.1), c.0, c.1)));
                }
            }
        }
        if tests.is_empty() {
            return Err((head, TEST));
        }
        let mut memo: Vec<HashMap<Bool, bool>> = vec![HashMap::new(); tests.len()];
        // what a lane only reaches with its own bit set, in the round of that test
        let mut body_block = vec![false; n];
        for b in (0..n).filter(|&b| lp[b]) {
            body_block[b] = tests.iter().enumerate().any(|(j, &(a, tb, _))| {
                b != tb && self.dominates(tb, b) && self.implied(st.reach[b], a, &mut memo[j])
            });
        }
        let body_inst = |b: usize, i: usize, memo: &mut Vec<HashMap<Bool, bool>>| {
            body_block[b]
                || lp[b]
                    && st.guards[b][i].is_some_and(|g| {
                        tests.iter().enumerate().any(|(j, &(a, tb, ti))| {
                            (b == tb && i > ti || b != tb && self.dominates(tb, b))
                                && self.implied(g, a, &mut memo[j])
                        })
                    })
        };
        // Each path enters once, reaches the latch once per round, and rejoins at
        // the common exit. In particular no nested cycle can hit a loop cap.
        let back_edges: Vec<usize> = p.blocks[h]
            .preds
            .iter()
            .copied()
            .filter(|&b| lp[b])
            .collect();
        let [back] = back_edges[..] else {
            return Err((head, SHAPE));
        };
        if back != latch && p.blocks[back].preds[..] != [latch] {
            return Err((head, SHAPE));
        }
        let local_condition = |g| {
            st.bools
                .support(g)
                .iter()
                .all(|a| matches!(a, Atom::Lane(_) | Atom::Uniform(_) | Atom::Constant(_)))
        };
        for b in (0..n).filter(|&b| lp[b] || pre[b]) {
            if st.guards[b].iter().flatten().any(|&g| !local_condition(g))
                || matches!(st.exits[b], LaneExit::Branch { when, .. } if !local_condition(when))
            {
                return Err((p.blocks[b].offset, CROSS));
            }
            let successors = match st.exits[b] {
                LaneExit::Jump(t) => [Some(t), None],
                LaneExit::Branch {
                    taken, not_taken, ..
                } => [Some(taken), Some(not_taken)],
                LaneExit::End | LaneExit::Indirect => [None, None],
            };
            if successors[0].is_none()
                || successors.into_iter().flatten().any(|s| {
                    !(lp[b] && (lp[s] || b == latch && s == exit) || pre[b] && (pre[s] || s == h))
                        || self.dominates(s, b) && (b != back || s != h)
                })
            {
                return Err((p.blocks[b].offset, EXIT));
            }
            for (i, inst) in p.blocks[b].insts.iter().enumerate() {
                let u = (b, i);
                let known = pattern.contains(&u);
                let name = inst.name();
                // A shortened loop must never require its peers, even in the own-bit body.
                let cross = matches!(inst.fields, Fields::Ds { .. })
                    || matches!(name, "s_barrier" | "s_sendmsg" | "s_sendmsghalt")
                    || name.starts_with("v_mbcnt")
                    || name.contains("atomic")
                    || name.contains("lane")
                    || matches!(&inst.fields, Fields::Valu(v) if v.dpp.is_some())
                    || super::lower::reads_mask_as_data(st, inst)
                    || self.sections[b]
                        .iter()
                        .any(|(&s, sec)| i >= s && i <= sec.end);
                if !known && cross {
                    return Err((self.at(u), CROSS));
                }
                if !known
                    && used(p, self.vgprs, st, b, i)
                        .iter()
                        .any(|v| rem.contains(v))
                {
                    return Err((self.at(u), USED));
                }
                if lp[b] && body_inst(b, i, &mut memo) {
                    continue;
                }
                if effect(inst) {
                    return Err((self.at(u), EFFECT));
                }
                for &(reg, w) in &inst.writes {
                    if rem.contains(&w)
                        || is_mask(w)
                        || matches!(reg, Reg::S(c) if c == EXEC || c == EXEC + 1)
                    {
                        continue;
                    }
                    for user in self.users.get(&w).into_iter().flatten() {
                        let stays = match *user {
                            User::Inst(b2, _) => lp[b2] || pre[b2],
                            User::Phi(b2, k2) => {
                                body_block[b2] || !self.live.contains(&p.blocks[b2].phis[k2].value)
                            }
                        };
                        if !stays {
                            return Err((self.at(u), LEAVES));
                        }
                    }
                }
            }
        }
        // Every compare in the loop can depend on the round, through its
        // operands or through EXEC. Follow mask phis through both dwords.
        let dep: HashSet<Atom> = p
            .blocks
            .iter()
            .enumerate()
            .filter(|(b, _)| lp[*b] || pre[*b])
            .flat_map(|(_, block)| {
                block
                    .insts
                    .iter()
                    .flat_map(|inst| inst.writes.iter().map(|&(_, w)| Atom::Lane(w)))
                    .chain(block.phis.iter().map(|phi| Atom::Phi(phi.value)))
            })
            .collect();
        let mut seen: HashMap<Bool, bool> = HashMap::new();
        let mut tainted = |b: Bool| -> bool {
            *seen
                .entry(b)
                .or_insert_with(|| st.bools.support(b).iter().any(|a| dep.contains(a)))
        };
        for (b, &in_loop) in lp.iter().enumerate() {
            if in_loop && tests.iter().any(|&(_, tb, _)| self.dominates(tb, b)) {
                continue;
            }
            let stale = st.guards[b].iter().flatten().any(|&g| tainted(g))
                || matches!(st.exits[b], LaneExit::Branch { when, .. } if tainted(when));
            if stale {
                return Err((p.blocks[b].offset, STALE));
            }
        }
        for (v, m) in st.masks.iter().map(|(&v, &m)| (v, m)).chain(
            st.highs
                .iter()
                .filter_map(|(&v, low)| st.masks.get(low).map(|&m| (v, m))),
        ) {
            if !tainted(m) {
                continue;
            }
            for user in self.users.get(&v).into_iter().flatten() {
                let outside = match *user {
                    User::Inst(b2, _) => !lp[b2],
                    User::Phi(b2, k2) => {
                        !lp[b2] && self.live.contains(&p.blocks[b2].phis[k2].value)
                    }
                };
                if outside {
                    return Err((head, STALE));
                }
            }
        }
        Ok((first_read, (0..n).filter(|&b| lp[b] || pre[b]).collect()))
    }
}

/// The wave ORs of `p` that only steer a loop over each lane's own bits (see the module
/// notes). `why`: one line on stderr for each wave OR that is not one.
pub(super) fn find(p: &Program, vgprs: u16, st: &Structured, why: bool) -> OwnBits {
    let mut out = OwnBits::default();
    let mut ctx: Option<Ctx> = None;
    for b in 0..p.blocks.len() {
        let sections = wave::sections(p, b);
        let mut starts: Vec<_> = sections.keys().copied().collect();
        starts.sort_unstable();
        for s in starts {
            let sec = &sections[&s];
            let mut left: Vec<_> = sec.shapes.iter().collect();
            left.sort_unstable_by_key(|(v, _)| v.0);
            for (&t, &shape) in left {
                let Shape::Active(Op::Or, x, mask, 0) = shape else {
                    continue;
                };
                let ctx = ctx.get_or_insert_with(|| Ctx::new(p, vgprs, st));
                // (values only the section itself reads are not results)
                let inside =
                    |u: &User| matches!(*u, User::Inst(b2, i) if b2 == b && i > s && i < sec.end);
                if ctx.users.get(&t).is_none_or(|us| us.iter().all(inside)) {
                    continue;
                }
                match ctx.site(b, t, x, mask) {
                    Ok((first_read, blocks)) => {
                        out.totals.insert(t);
                        out.first_reads.extend(first_read);
                        out.local_blocks.extend(blocks);
                    }
                    Err((offset, what)) => {
                        if why {
                            eprintln!(
                                "WAVE64-OWN-BITS the wave OR at {:#x}: not a loop over the lanes' own bits at {offset:#x}: {what}",
                                p.blocks[b].insts[s].machine.offset
                            );
                        }
                    }
                }
            }
            if let Some(ctx) = &ctx
                && p.blocks[b].insts[s + 1..sec.end]
                    .iter()
                    .flat_map(|i| &i.writes)
                    .all(|&(_, w)| {
                        out.totals.contains(&w)
                            || ctx.users.get(&w).is_none_or(|users| {
                                users.iter().all(|u| match *u {
                                    User::Inst(b2, i) => b2 == b && i > s && i < sec.end,
                                    User::Phi(b2, k) => {
                                        !ctx.live.contains(&p.blocks[b2].phis[k].value)
                                    }
                                })
                            })
                    })
                && sec.shapes.keys().any(|v| out.totals.contains(v))
            {
                out.local_sections.extend((s..=sec.end).map(|i| (b, i)));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    /// `WaveActiveBitOr(v7)` into s7 as the console compiler expands it: every lane takes part,
    /// those outside the saved EXEC (s[4:5]) with 0.
    const WAVE_OR_V7_TO_S7: [u32; 20] = [
        0xBE84_25C1, // s_or_saveexec_b64 s[4:5], -1
        0xD501_0001,
        0x0012_0E80, // v_cndmask_b32 v1, 0, v7, s[4:5]
        0x3802_02E9,
        0xDE54_C101, // v_or_b32 v1, v1, v1 dpp8:[1,0,3,2,5,4,7,6]
        0x3802_02E9,
        0x9360_1201, // v_or_b32 v1, v1, v1 dpp8:[2,2,0,0,6,6,4,4]
        0x3802_02E9,
        0x0009_2401, // v_or_b32 v1, v1, v1 dpp8:[4,4,4,4,0,0,0,0]
        0x3802_02FA,
        0xFF09_4001, // v_or_b32 v1, v1, v1 row_mirror bound_ctrl:1
        0xD778_0002,
        0x0201_0101, // v_permlanex16_b32 v2, v1, 0, 0
        0x3802_0302, // v_or_b32 v1, v2, v1
        0xD760_0007,
        0x0001_4101, // v_readlane_b32 s7, v1, 32
        0x3802_0207, // v_or_b32 v1, s7, v1
        0xD760_0007,
        0x0001_0101, // v_readlane_b32 s7, v1, 0
        0xBEFE_0404, // s_mov_b64 exec, s[4:5]
    ];

    const ADD_BIT_NUMBER: u32 = 0x4A0A_0A0A; // v_add_nc_u32 v5, s10, v5

    /// The title's scalarised light loop (`indirect_light_probes:CS_IrradianceCache_Shade`,
    /// 0x71f8 on), in small: the wave OR of v7 in s7, a loop over its bits with the lowest
    /// one's number in s10 and the bit in s11, and `body` for the lanes whose v7 has the bit
    /// (`own`) or for every lane; `latch` runs in every lane after the body, `after` after
    /// the loop.
    fn light_loop_words(own: bool, body: &[u32], latch: &[u32], after: &[u32]) -> Vec<u32> {
        let mut w = WAVE_OR_V7_TO_S7.to_vec();
        let test: &[u32] = if own {
            &[
                0x3608_0E0B, // v_and_b32 v4, s11, v7
                0xBE8C_047E, // s_mov_b64 s[12:13], exec
                0x7DAA_0880, // v_cmpx_ne_u32 exec, 0, v4
            ]
        } else {
            &[0xBE8C_047E] // s_mov_b64 s[12:13], exec
        };
        // from the loop's head to its last branch
        let head = 3 + test.len() + usize::from(own) + body.len() + 1 + latch.len() + 3;
        w.extend_from_slice(&[
            0xBF06_8007,                     // s_cmp_eq_u32 s7, 0
            0xBF85_0000 | (3 + head) as u32, // s_cbranch_scc1 past the loop
            0x7E06_0207,                     // v_mov_b32 v3, s7
            0xBF82_0001,                     // s_branch to the loop's head
            0x7E06_0209,                     // again: v_mov_b32 v3, s9
            0x7E10_0503,                     // head: v_readfirstlane_b32 s8, v3
            0xBE8A_1308,                     // s_ff1_i32_b32 s10, s8
            0x8F0B_0A81,                     // s_lshl_b32 s11, 1, s10
        ]);
        w.extend_from_slice(test);
        if own {
            w.push(0xBF88_0000 | body.len() as u32); // s_cbranch_execz past the body
        }
        w.extend_from_slice(body);
        w.push(0xBEFE_040C); // s_mov_b64 exec, s[12:13]
        w.extend_from_slice(latch);
        w.extend_from_slice(&[
            0x8909_080B,                                // s_xor_b32 s9, s11, s8
            0xBF06_0B08,                                // s_cmp_eq_u32 s8, s11
            0xBF84_0000 | (0x1_0000 - head - 1) as u32, // s_cbranch_scc0 again
        ]);
        w.extend_from_slice(after);
        w.push(0xBF81_0000); // s_endpgm
        w
    }

    fn light_loop(own: bool, body: &[u32], latch: &[u32], after: &[u32]) -> Program {
        let code: Vec<u8> = light_loop_words(own, body, latch, after)
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect();
        crate::ir::build(&code, true, 256).unwrap()
    }

    fn found(p: &Program) -> OwnBits {
        let st = crate::structure::structure(p).unwrap();
        find(p, 256, &st, false)
    }

    #[test]
    fn a_wave_or_that_only_steers_a_loop_over_own_bits_is_found() {
        let p = light_loop(true, &[ADD_BIT_NUMBER], &[], &[]);
        let insts = || p.blocks.iter().flat_map(|b| &b.insts);
        let names: Vec<&str> = insts().map(|i| i.name()).collect();
        for n in [
            "s_ff1_i32_b32",
            "s_lshl_b32",
            "v_and_b32",
            "v_cmpx_ne_u32",
            "s_xor_b32",
            "v_readfirstlane_b32",
            "s_cbranch_execz",
        ] {
            assert!(names.contains(&n), "{n} in {names:?}");
        }
        // a loop: its last branch goes back to the copy of the bits still to visit
        assert!(
            p.blocks.iter().enumerate().any(|(b, blk)| blk
                .exit
                .successors()
                .iter()
                .any(|&s| s < b))
        );
        let own = found(&p);
        // the OR is the section's last lane read, and the bits still to visit (v3) are alike
        let total = insts()
            .rfind(|i| i.name() == "v_readlane_b32")
            .and_then(|i| i.write(Reg::S(7)))
            .unwrap();
        assert_eq!(own.totals, HashSet::from([total]));
        let first = insts().find(|i| i.name() == "v_readfirstlane_b32").unwrap();
        assert!(own.first_reads.contains(&first.write(Reg::S(8)).unwrap()));
    }

    #[test]
    fn a_loop_whose_lanes_all_work_at_every_bit_needs_the_whole_waves_or() {
        // (the same title's volume loops, 0xa668 on: no test of the lane's own mask)
        let p = light_loop(false, &[ADD_BIT_NUMBER], &[], &[]);
        assert!(found(&p).totals.is_empty());
    }

    #[test]
    fn a_bit_number_read_after_the_loop_is_not_the_lanes_own() {
        // v_mov_b32 v6, s10: the last bit of the OR, which another lane may have set
        let p = light_loop(true, &[ADD_BIT_NUMBER], &[], &[0x7E0C_020A]);
        assert!(found(&p).totals.is_empty());
    }

    #[test]
    fn the_bits_still_to_visit_read_in_the_body_are_not_the_lanes_own() {
        // v_add_nc_u32 v5, s8, v5: s8 holds the other lanes' bits too
        let p = light_loop(true, &[0x4A0A_0A08], &[], &[]);
        assert!(found(&p).totals.is_empty());
    }

    #[test]
    fn a_count_of_the_rounds_is_not_the_lanes_own() {
        // s_add_i32 s14, s14, 1 where every lane passes, read after the loop
        // (v_mov_b32 v6, s14): one round for each bit any lane has
        let p = light_loop(true, &[ADD_BIT_NUMBER], &[0x810E_810E], &[0x7E0C_020E]);
        assert!(
            p.blocks
                .iter()
                .flat_map(|b| &b.insts)
                .any(|i| i.name() == "s_add_i32")
        );
        assert!(found(&p).totals.is_empty());
        // (reading s14 there without counting in it is nothing of the loop's)
        let p = light_loop(true, &[ADD_BIT_NUMBER], &[], &[0x7E0C_020E]);
        assert_eq!(found(&p).totals.len(), 1);
    }

    #[test]
    fn an_effect_where_every_lane_passes_is_not_the_lanes_own() {
        // ds_write_b32 v5, v5 after the body: once for each bit any lane has
        let p = light_loop(true, &[ADD_BIT_NUMBER], &[0xD834_0000, 0x0000_0505], &[]);
        assert!(
            p.blocks
                .iter()
                .flat_map(|b| &b.insts)
                .any(|i| i.name() == "ds_write_b32")
        );
        assert!(found(&p).totals.is_empty());
    }
    #[test]
    fn an_own_bit_body_with_group_sync_or_lane_reads_is_refused() {
        for inst in [0xBF8A_0000, 0x7E1C_0505] {
            // s_barrier / v_readfirstlane_b32 s14, v5.
            let p = light_loop(true, &[inst, ADD_BIT_NUMBER], &[], &[]);
            assert!(found(&p).totals.is_empty());
        }
    }

    #[test]
    fn testing_a_different_mask_is_refused() {
        let mut words = light_loop_words(true, &[ADD_BIT_NUMBER], &[], &[]);
        let test = words.iter_mut().find(|w| **w == 0x3608_0E0B).unwrap();
        *test = 0x3608_0C0B; // v_and_b32 v4, s11, v6
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let p = crate::ir::build(&code, true, 256).unwrap();
        assert!(found(&p).totals.is_empty());
    }

    fn lower_light_loop(
        enabled: bool,
        threads: u32,
        nonempty_from: u32,
        test_own_bit: bool,
        first_read: bool,
        divergent: bool,
    ) -> crate::emit::lower::Lowered {
        use crate::emit::{lower, with_thread_ticket};
        use crate::native::{
            ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo,
            UsedResource,
        };
        use crate::{io, ir, resource, structure};

        let mut words = vec![
            0xBE84_0300, // s_mov_b32 s4, s0
            0xBE85_0380, // s_mov_b32 s5, 0
            0xF408_0A02, // s_load_dwordx4 s[40:43], s[4:5], 0
            0xFA00_0000,
            0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
            0x3420_0085, // v_lshlrev_b32 v16, 5, v0 (UAV byte offset)
            0x2C0E_0081, // v_lshrrev_b32 v7, 1, v0
            0x7E0C_0281, // v_mov_b32 v6, 1
            0x340E_0D07, // v_lshlrev_b32 v7, v7, v6
            0x380E_0E81, // v_or_b32 v7, 1, v7 (bit zero shared)
            0x7D86_00FF, // v_cmp_le_u32 vcc, nonempty_from, v0
            nonempty_from,
            0xD501_0007, // v_cndmask_b32 v7, 0, v7, vcc
            0x01AA_0E80,
            0x7E0A_0281, // v_mov_b32 v5, 1
        ];
        let mut body = vec![
            0x340A_0A85, // v_lshlrev_b32 v5, 5, v5 (encode visit order)
            ADD_BIT_NUMBER,
        ];
        let store = [0xE070_1000, 0x800A_0510]; // buffer_store_dword v5, v16, s[40:43], 0 offen
        let after = if divergent {
            body.extend(store);
            &[][..]
        } else {
            &store[..]
        };
        let mut loop_words = light_loop_words(test_own_bit, &body, &[], after);
        if !first_read {
            let start = WAVE_OR_V7_TO_S7.len();
            loop_words[start + 2] = 0xBE88_0307; // s_mov_b32 s8, s7
            loop_words[start + 4] = 0xBE88_0309; // s_mov_b32 s8, s9
            loop_words[start + 5] = 0xBF80_0000; // s_nop 0
        }
        if divergent {
            words.extend([
                0x7DA6_00FF, // v_cmpx_le_u32 exec, 16, v0
                16,
                0xBF88_0000 | (loop_words.len() - 1) as u32, // s_cbranch_execz end
            ]);
        }
        words.extend(loop_words);
        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64: true,
            vgprs: 32,
            layout: vec![RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 0,
                slot: 0,
            }],
            used: vec![UsedResource {
                kind: ResourceType::Uav,
                root: RootType::Table,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 0,
                dwords: 4,
                user_data: 0,
                slot: 0,
            }],
            io: StageIo::Cs(ComputeIo {
                rsrc2: 1 << 1,
                threads: [threads, 1, 1],
            }),
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let p = ir::build(&native.code, true, native.vgprs).unwrap();
        let st = structure::structure(&p).unwrap();
        let io = io::raise(&p, &native).unwrap();
        let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        with_thread_ticket(false, || {
            with_enabled(enabled, || {
                lower::lower_pruned(
                    &lower::Parts {
                        native: &native,
                        p: &p,
                        st: &st,
                        io: &io,
                        res: &res,
                        inputs: Vec::new(),
                        outputs: Vec::new(),
                        prims: Vec::new(),
                        mesh_stride: 0,
                        view_marks: false,
                        atomic_locks: false,
                        atomic_lock_rounds: false,
                        atomic_umax: false,
                        atomic_early_exit: false,
                        bias_as_gradients: false,
                        narrow_waves: true,
                        uav_size_level0: false,
                        halves_rendezvous: true,
                        host_wave32: true,
                    },
                    None,
                    false,
                    false,
                    false,
                )
            })
        })
        .expect("light loop lowers")
    }

    fn both_forms(threads: u32, nonempty_from: u32) -> [crate::emit::lower::Lowered; 2] {
        let exchange = lower_light_loop(false, threads, nonempty_from, true, false, false);
        let local = lower_light_loop(true, threads, nonempty_from, true, false, false);
        assert_eq!(exchange.wave_size, Some(32));
        assert_eq!(local.wave_size, Some(32));
        assert_eq!(exchange.barriers, 2);
        assert!(exchange.half_exchange);
        assert_eq!(local.barriers, 0);
        assert!(!local.half_exchange);
        [exchange, local]
    }

    #[test]
    fn a_proven_light_loop_eliminates_the_emitted_exchange() {
        both_forms(64, 0);
        let first_read = lower_light_loop(true, 64, 0, true, true, false);
        assert_eq!(first_read.barriers, 0);
        assert!(!first_read.half_exchange);
        let divergent = lower_light_loop(true, 64, 0, true, true, true);
        assert_eq!(divergent.barriers, 0);
        assert!(!divergent.half_exchange);
    }

    #[test]
    fn an_unproven_light_loop_keeps_the_same_dxil() {
        let exchange = lower_light_loop(false, 64, 0, false, false, false);
        let enabled = lower_light_loop(true, 64, 0, false, false, false);
        assert_eq!(exchange.barriers, 2);
        assert!(exchange.half_exchange);
        assert_eq!(exchange.dxil, enabled.dxil);
    }

    #[cfg(windows)]
    #[test]
    fn local_and_cross_half_light_loops_write_the_same_bits_on_hardware() {
        use crate::emit::HostShader;
        use crate::emit::halves_tests::shader;
        use crate::emit::host::{Dispatch, Host, Root, View};

        let _one = crate::emit::valu_tests::ONE_AT_A_TIME
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let hosts = Host::each_hardware();
        if hosts.is_empty() {
            eprintln!("no SM 6.6 hardware device; skipped");
            return;
        }
        let divergent = shader(lower_light_loop(true, 64, 0, true, true, true));
        for (threads, nonempty_from) in [(64, 0), (64, 32), (64, 64), (96, 0), (128, 0)] {
            let [exchange, local] = both_forms(threads, nonempty_from);
            let first_read = lower_light_loop(true, threads, nonempty_from, true, true, false);
            assert_eq!(first_read.barriers, 0);
            assert!(!first_read.half_exchange);
            let forms = [exchange, local, first_read].map(shader);
            for (name, host) in &hosts {
                let run = |shader: &HostShader| {
                    let (root, pso) = host
                        .compute_pipeline(shader)
                        .unwrap_or_else(|e| panic!("{name}: {e}"));
                    host.dispatch(
                        &root,
                        &pso,
                        &Dispatch {
                            roots: vec![(
                                0,
                                Root::Constants(vec![0; shader.user_data_dwords().max(1) as usize]),
                            )],
                            views: &[View::RawUav(threads * 8)],
                            samplers: &[],
                            groups: [1, 1, 1],
                        },
                    )
                    .unwrap_or_else(|e| panic!("{name}: {e}"))
                };
                let [exchange, local, first_read] = forms.each_ref().map(run);
                assert_eq!(
                    exchange, local,
                    "{name}, {threads} threads, first nonempty {nonempty_from}"
                );
                assert_eq!(exchange, first_read, "{name}: first-lane form");
                let want: Vec<u32> = (0..threads)
                    .map(|lane| {
                        if lane < nonempty_from {
                            1
                        } else {
                            let bit = (lane / 2) % 32;
                            if bit == 0 { 32 } else { 1024 + bit }
                        }
                    })
                    .collect();
                assert_eq!(
                    exchange[0]
                        .as_chunks::<8>()
                        .0
                        .iter()
                        .map(|row| row[0])
                        .collect::<Vec<_>>(),
                    want,
                    "{name}, {threads} threads, first nonempty {nonempty_from}"
                );
                if threads == 64 && nonempty_from == 0 {
                    let masked = run(&divergent);
                    let rows = masked[0].as_chunks::<8>().0;
                    assert_eq!(
                        rows[16..].iter().map(|row| row[0]).collect::<Vec<_>>(),
                        want[16..],
                        "{name}: lanes reaching a divergent wave OR"
                    );
                }
                eprintln!(
                    "{name}: {threads} lanes match with exchange on and off (first nonempty lane {nonempty_from})"
                );
            }
        }
    }
}
