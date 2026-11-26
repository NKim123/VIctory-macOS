//! Where group barriers can go when the host splits a compute group into
//! waves the console's lockstep did not have: the blocks every lane of the
//! group runs together ([`in_step`]), and the barriers a group that is one
//! console wave needs between its lanes' LDS accesses
//! ([`split_wave_barriers`]).

use super::*;

/// Where a compute group that is one console wave needs group barriers
/// when the host splits it into narrower waves: (block, instruction) to
/// sync before, instruction index = the block's length for its end.
///
/// The console runs such a group in lockstep, so its compiler leaves out
/// the barriers between one lane's LDS access and another's. Here a read
/// (or atomic) of LDS that a write or atomic may have changed since the last
/// barrier gets one before it, and so does a write (or atomic) after a read
/// or atomic since then, so that it cannot overwrite what another lane has
/// yet to read. (A write after a write, which the console orders too, gets
/// none.) Barriers need every lane of the group, so they go only in blocks
/// every lane runs together, the same number of times ([`in_step`]): once,
/// or in a loop on a count every host wave computes alike. Such a block that
/// leads into code only some lanes run syncs at its end where that code
/// could otherwise read a write, or overwrite a read, made before it.
///
/// A read may see a write from divergent code only where the region's
/// only observable effect is an idempotent flag store. Other such reads
/// return `None` because a group barrier cannot go in that region.
pub(crate) fn split_wave_barriers(
    p: &Program,
    st: &Structured,
    waterfall_reads: &[Value],
) -> Option<Set<(usize, usize)>> {
    /// LDS accesses since the last barrier.
    #[derive(Clone, Copy, Default, PartialEq)]
    struct Since {
        read: bool,
        written: bool,
    }
    impl Since {
        fn or(self, o: Since) -> Since {
            Since {
                read: self.read || o.read,
                written: self.written || o.written,
            }
        }
    }
    // (reads, writes) of an LDS access; an atomic does both.
    let access = |inst: &Inst| {
        let name = inst.name();
        if !matches!(inst.fields, Fields::Ds { .. })
            || name.starts_with("ds_swizzle")
            || name.contains("permute")
        {
            return None;
        }
        Some((
            !name.starts_with("ds_write"),
            !name.starts_with("ds_read"),
        ))
    };
    let n = p.blocks.len();
    let succ = |b: usize| p.blocks[b].exit.successors();
    let step = in_step(p, st, waterfall_reads);
    let divergent_writes = (0..n).any(|b| {
        !step[b]
            && p.blocks[b]
                .insts
                .iter()
                .any(|i| access(i).is_some_and(|(_, w)| w))
    });
    let mut flag_region = HashMap::new();
    let order = crate::ir::reverse_postorder(&p.blocks);
    let mut out = vec![Since::default(); n];
    loop {
        let mut at = Set::default();
        let mut changed = false;
        for &b in &order {
            let mut since = p.blocks[b]
                .preds
                .iter()
                .fold(Since::default(), |a, &q| a.or(out[q]));
            for (i, inst) in p.blocks[b].insts.iter().enumerate() {
                if inst.name() == "s_barrier" {
                    since = Since::default();
                }
                let Some((reads, writes)) = access(inst) else {
                    continue;
                };
                // A read of what another lane may have written; a write over
                // what another lane may have yet to read.
                let stale = reads && since.written;
                let early = writes && since.read;
                if stale || early {
                    if step[b] {
                        at.insert((b, i));
                        since = Since::default();
                    } else if stale {
                        if !flag_region.contains_key(&b) {
                            let mut seen = HashSet::from([b]);
                            let mut pending = vec![b];
                            while let Some(q) = pending.pop() {
                                for t in p.blocks[q].preds.iter().copied().chain(succ(q)) {
                                    if !step[t] && seen.insert(t) {
                                        pending.push(t);
                                    }
                                }
                            }
                            let blocks: Vec<_> = seen.into_iter().collect();
                            let safe = lds_flag_region(p, st, &blocks);
                            flag_region.extend(blocks.into_iter().map(|b| (b, safe)));
                        }
                        if !flag_region[&b] {
                            return None;
                        }
                    }
                }
                since = since.or(Since {
                    read: reads,
                    written: writes,
                });
            }
            if step[b]
                && succ(b).iter().any(|&s| !step[s])
                && (since.written || since.read && divergent_writes)
            {
                at.insert((b, p.blocks[b].insts.len()));
                since = Since::default();
            }
            // What reaches a block only grows, so this ends; barriers placed
            // for more than reaches it are still right.
            let since = out[b].or(since);
            changed |= std::mem::replace(&mut out[b], since) != since;
        }
        if !changed {
            return Some(at);
        }
    }
}

/// Per block, whether every lane of the group runs it together, the same
/// number of times, however the host splits the group into waves, so that a
/// group barrier can go there: outside the region of every branch whose
/// condition may differ between waves ([`halves::branch_regions`]). That is
/// a per-lane branch, or one on a scalar not computed alike in every wave.
/// Every block each lane runs once ([`once_per_lane`]) is one; so is a loop
/// whose count every wave computes alike, from the group's inputs,
/// constants and scalar loads through scalar code (not from lane masks,
/// other lanes' values or hardware state). Such a scalar is the same in
/// every console wave of the group too.
///
/// Scalars are assumed alike until shown otherwise (a loop counter needs
/// itself), and the regions and the scalars are narrowed together to a
/// fixed point. With an indirect branch, the blocks each lane runs once.
pub(crate) fn in_step(p: &Program, st: &Structured, waterfall_reads: &[Value]) -> Vec<bool> {
    let Some(regions) = halves::branch_regions(st) else {
        return once_per_lane(p);
    };
    let n = p.blocks.len();
    let scalar = |r: Reg| matches!(r, Reg::S(_) | Reg::Scc);
    let group_branches = !st.group_branches.is_empty();
    // Not EXEC or VCC: lane masks.
    let mask = |r: Reg| matches!(r, Reg::S(k) if [EXEC, EXEC + 1, VCC, VCC + 1].contains(&k));
    // Scalar writes no host wave can compute alike: other lanes' values,
    // hardware state, atomics, coherent loads (the group may write them).
    let own = |inst: &Inst| {
        let name = inst.name();
        match &inst.fields {
            Fields::Sop { .. } => {
                name.starts_with("s_getreg")
                    || name.starts_with("s_sendmsg")
                    || name.contains("waveid")
            }
            Fields::Smem { cache, .. } => {
                !(name.starts_with("s_load") || name.starts_with("s_buffer_load")) || cache.glc
            }
            _ => true,
        }
    };
    let mut alike: Set<Value> = p
        .inputs
        .iter()
        .filter(|(r, _)| scalar(*r))
        .map(|(_, v)| *v)
        .chain(p.blocks.iter().flat_map(|b| {
            b.phis
                .iter()
                .filter(|f| scalar(f.reg))
                .map(|f| f.value)
                .chain(
                    b.insts
                        .iter()
                        .flat_map(|i| i.writes.iter().filter(|w| scalar(w.0)).map(|w| w.1)),
                )
        }))
        .collect();
    loop {
        let mut step = vec![true; n];
        for (d, region) in &regions {
            let LaneExit::Branch { when, .. } = st.exits[*d] else {
                continue;
            };
            let together = st.bools.support(when).iter().all(|a| {
                matches!(a, Atom::GroupAny(_)) || matches!(a, Atom::Uniform(v) if alike.contains(v))
            });
            if !together {
                for (b, &r) in region.iter().enumerate() {
                    step[b] &= !r;
                }
            }
        }
        let mut next: Set<Value> = p
            .inputs
            .iter()
            .filter(|(r, _)| scalar(*r))
            .map(|(_, v)| *v)
            .collect();
        for (b, block) in p.blocks.iter().enumerate() {
            // Every host wave enters by the same edge.
            let entered = step[b] && block.preds.iter().all(|&q| step[q]);
            for phi in &block.phis {
                if scalar(phi.reg) && entered && phi.args.iter().all(|a| alike.contains(a)) {
                    next.insert(phi.value);
                }
            }
            for inst in &block.insts {
                let from = inst
                    .reads
                    .iter()
                    .all(|&(r, v)| scalar(r) && (group_branches || !mask(r)) && alike.contains(&v));
                // The one-console-wave retry broadcasts lane reads across
                // the group where it is in step. A waterfall's per-lane
                // replacement or a restored SGPR can still vary by lane.
                let broadcast = group_branches
                    && step[b]
                    && matches!(inst.name(), "v_readfirstlane_b32" | "v_readlane_b32")
                    && inst.writes.iter().all(|&(_, v)| {
                        !waterfall_reads.contains(&v)
                            && st.restored.get(&v).is_none_or(|a| alike.contains(a))
                    })
                    && inst
                        .reads
                        .iter()
                        .all(|&(r, v)| !scalar(r) || mask(r) || alike.contains(&v));
                if (from && !own(inst)) || broadcast {
                    next.extend(inst.writes.iter().filter(|w| scalar(w.0)).map(|w| w.1));
                }
                if group_branches && step[b] {
                    // Active mask dwords read as scalars are materialized
                    // by a ballot across both halves at their definition.
                    next.extend(inst.writes.iter().filter_map(|&(r, v)| {
                        let low = st.highs.get(&v).copied().unwrap_or(v);
                        (scalar(r) && st.active.contains(&low)).then_some(v)
                    }));
                }
            }
        }
        if next.len() == alike.len() {
            return step;
        }
        alike = next;
    }
}

/// Per block, whether each lane of the group runs it exactly once: no cycle
/// passes through it, and every path from the entry to an end does. A group
/// barrier needs every lane of the group, so it can go only in these.
pub(crate) fn once_per_lane(p: &Program) -> Vec<bool> {
    let n = p.blocks.len();
    let succ = |b: usize| p.blocks[b].exit.successors();
    // Blocks reachable from `from`'s successors (or from `from` itself when
    // `start`), never entering `avoid`.
    let reach = |from: usize, start: bool, avoid: Option<usize>| {
        let mut seen = vec![false; n];
        let mut stack = if start { vec![from] } else { succ(from) };
        while let Some(b) = stack.pop() {
            if Some(b) == avoid || std::mem::replace(&mut seen[b], true) {
                continue;
            }
            stack.extend(succ(b));
        }
        seen
    };
    (0..n)
        .map(|b| {
            let ends_around = b != 0
                && reach(0, true, Some(b))
                    .iter()
                    .enumerate()
                    .any(|(q, &r)| r && succ(q).is_empty());
            !reach(b, false, None)[b] && !ends_around
        })
        .collect()
}
/// Whether a lane-branch region (`blocks`) that both reads and writes
/// LDS only raises flags, so that it may run as two host waves with group barriers at its
/// edges alone. The console runs such a region as one wave, where a lane's read sees every
/// write before it; split in two, a read may miss the other half's write. That changes nothing
/// the region leaves behind when:
/// - every LDS access in it is a `ds_read_b32` or `ds_write_b32` at a constant address, and
///   every write stores the same constant;
/// - one address at most is both read and written in it (the flag), and every write a lane
///   can reach after one of its reads goes to that flag;
/// - it has no other effect (no store, atomic, export, message or barrier);
/// - no value it computes is used after it and the region is acyclic.
///
/// A missed write only makes a lane go on as if the flag were still down; whatever that lane
/// writes is the flag's own constant, which the write it missed has already put there. So the
/// flag ends up raised exactly when some lane raises it, as on the console, and no other word
/// or register depends on when it was seen.
fn lds_flag_region(p: &Program, st: &Structured, blocks: &[usize]) -> bool {
    let inside: HashSet<usize> = blocks.iter().copied().collect();
    // A read of the flag cannot govern progress around a loop. The other
    // host wave may already be waiting at the region's exit barrier.
    let mut rank = vec![usize::MAX; p.blocks.len()];
    for (i, b) in crate::ir::reverse_postorder(&p.blocks)
        .into_iter()
        .enumerate()
    {
        rank[b] = i;
    }
    if blocks.iter().any(|&b| {
        p.blocks[b]
            .exit
            .successors()
            .iter()
            .any(|t| inside.contains(t) && rank[*t] <= rank[b])
    }) {
        return false;
    }
    // where each value is written: (block, instruction)
    let mut def: HashMap<Value, (usize, usize)> = HashMap::new();
    for (b, block) in p.blocks.iter().enumerate() {
        for (i, inst) in block.insts.iter().enumerate() {
            for &(_, v) in &inst.writes {
                def.insert(v, (b, i));
            }
        }
    }
    // a VGPR value a `v_mov_b32` of a constant wrote, unguarded or under the same guard as `at`
    let constant = |v: Option<Value>, at: (usize, usize)| -> Option<u32> {
        let (b, i) = *def.get(&v?)?;
        let inst = &p.blocks[b].insts[i];
        if inst.name() != "v_mov_b32" {
            return None;
        }
        let guard = st.guards.get(b).and_then(|g| g.get(i)).copied().flatten();
        let same = guard.is_none() || (b == at.0 && guard == st.guards[at.0][at.1]);
        let Fields::Valu(valu) = &inst.fields else {
            return None;
        };
        if !same || valu.dpp.is_some() || valu.sdwa.is_some() {
            return None;
        }
        scalar_constant(*valu.src.first()?)
    };
    // every LDS access: (block, index, is_read, address)
    let mut access: Vec<(usize, usize, bool, u32)> = Vec::new();
    let mut stored: Option<u32> = None;
    for &b in blocks {
        for (i, inst) in p.blocks[b].insts.iter().enumerate() {
            let name = inst.name();
            if name.contains("store")
                || name.contains("atomic")
                || name.starts_with("exp")
                || name.starts_with("s_sendmsg")
                || matches!(name, "s_barrier" | "s_setpc_b64" | "s_swappc_b64")
                || name.starts_with("s_trap")
                || name.starts_with("s_setreg")
            {
                return false;
            }
            let Fields::Ds {
                addr,
                data,
                offset0,
                offset1,
                gds,
                ..
            } = &inst.fields
            else {
                if name.starts_with("ds_") {
                    return false;
                }
                continue;
            };
            if name.starts_with("ds_swizzle") || name.contains("permute") {
                continue;
            }
            let is_read = match name {
                "ds_read_b32" => true,
                "ds_write_b32" => false,
                _ => return false,
            };
            if *gds {
                return false;
            }
            let Some(Operand::Vector { reg, .. }) = *addr else {
                return false;
            };
            let Some(base) = constant(inst.read(Reg::V(reg)), (b, i)) else {
                return false;
            };
            let at = base.wrapping_add(u32::from(*offset0) | u32::from(*offset1) << 8);
            if !at.is_multiple_of(4) {
                return false;
            }
            if !is_read {
                let Some(&Operand::Vector { reg, .. }) = data.first() else {
                    return false;
                };
                let Some(k) = constant(inst.read(Reg::V(reg)), (b, i)) else {
                    return false;
                };
                if stored.is_some_and(|s| s != k) {
                    return false;
                }
                stored = Some(k);
            }
            access.push((b, i, is_read, at));
        }
    }
    let read_at: HashSet<u32> = access.iter().filter(|a| a.2).map(|a| a.3).collect();
    let written_at: HashSet<u32> = access.iter().filter(|a| !a.2).map(|a| a.3).collect();
    let flags: Vec<u32> = read_at.intersection(&written_at).copied().collect();
    if flags.len() > 1 {
        return false;
    }
    if let Some(&flag) = flags.first() {
        // the blocks of the region a lane can reach from `b` along one edge or more
        let succ = |b: usize| -> Vec<usize> {
            match st.exits[b] {
                LaneExit::Jump(t) => vec![t],
                LaneExit::Branch {
                    taken, not_taken, ..
                } => vec![taken, not_taken],
                LaneExit::End | LaneExit::Indirect => Vec::new(),
            }
        };
        for &(rb, ri, is_read, _) in &access {
            if !is_read {
                continue;
            }
            let mut seen: HashSet<usize> = HashSet::new();
            let mut work: Vec<usize> = succ(rb)
                .into_iter()
                .filter(|t| inside.contains(t))
                .collect();
            while let Some(b) = work.pop() {
                if seen.insert(b) {
                    work.extend(succ(b).into_iter().filter(|t| inside.contains(t)));
                }
            }
            let after = access.iter().any(|&(wb, wi, w_read, at)| {
                !w_read && at != flag && ((wb == rb && wi > ri) || seen.contains(&wb))
            });
            if after {
                return false;
            }
        }
    }
    // Nothing the region computes is used after it: no register it writes is live, lane by
    // lane, where a lane leaves it. A lane's own write kills a register for that lane (an
    // unguarded instruction that writes the whole dword); what an instruction reads of a VGPR
    // only for the lanes outside EXEC (the old value of the one it writes) is no use.
    let succ = |b: usize| -> Option<Vec<usize>> {
        match st.exits[b] {
            LaneExit::Jump(t) => Some(vec![t]),
            LaneExit::Branch {
                taken, not_taken, ..
            } => Some(vec![taken, not_taken]),
            LaneExit::End => Some(Vec::new()),
            LaneExit::Indirect => None,
        }
    };
    let n = st.exits.len().min(p.blocks.len());
    if (0..n).any(|b| succ(b).is_none()) {
        return false;
    }
    let exec = |r: Reg| matches!(r, Reg::S(c) if c == EXEC || c == EXEC + 1);
    let covers = |ops: &[Operand], r: Reg| {
        let Reg::V(x) = r else { return false };
        ops.iter().any(|op| {
            matches!(*op, Operand::Vector { reg, dwords }
                if (u16::from(reg)..u16::from(reg) + u16::from(dwords.max(1))).contains(&u16::from(x)))
        })
    };
    // per instruction: the registers it reads for its lanes, and those it writes in full
    let effect = |b: usize, i: usize| -> (Vec<Reg>, Vec<Reg>) {
        let inst = &p.blocks[b].insts[i];
        let name = inst.name();
        let partial = name.contains("16")
            || name.contains("mac")
            || name.contains("lane")
            || name.contains("swap")
            || name.contains("movrel")
            || name.contains("acc")
            || matches!(&inst.fields, Fields::Valu(v) if v.dpp.is_some() || v.sdwa.is_some());
        // the operands it reads explicitly, where its old destination is not one of them
        let sources: Option<Vec<Operand>> = match &inst.fields {
            Fields::Valu(v) if !partial => Some(v.src.clone()),
            Fields::Ds { addr, data, .. } if !partial => {
                Some(addr.iter().copied().chain(data.iter().copied()).collect())
            }
            Fields::Buffer { addr, .. }
                if !partial && !name.contains("store") && !name.contains("atomic") =>
            {
                Some(vec![*addr])
            }
            Fields::Image { addr, .. }
                if !partial && !name.contains("store") && !name.contains("atomic") =>
            {
                Some(addr.clone())
            }
            _ => None,
        };
        let mut reads = Vec::new();
        for &(r, _) in &inst.reads {
            if exec(r) {
                continue;
            }
            let old_only = matches!(r, Reg::V(_))
                && inst.write(r).is_some()
                && sources.as_ref().is_some_and(|src| !covers(src, r));
            if !old_only {
                reads.push(r);
            }
        }
        let whole = st
            .guards
            .get(b)
            .and_then(|g| g.get(i))
            .copied()
            .flatten()
            .is_none();
        let kills: Vec<Reg> = if whole
            && (sources.is_some()
                || !matches!(
                    inst.fields,
                    Fields::Valu(_)
                        | Fields::Ds { .. }
                        | Fields::Buffer { .. }
                        | Fields::Image { .. }
                )) {
            inst.writes
                .iter()
                .map(|w| w.0)
                .filter(|&r| !exec(r) && !(partial && matches!(r, Reg::V(_))))
                .collect()
        } else {
            Vec::new()
        };
        (reads, kills)
    };
    let mut live_in: Vec<HashSet<Reg>> = vec![HashSet::new(); n];
    loop {
        let mut changed = false;
        for b in (0..n).rev() {
            let mut live: HashSet<Reg> = HashSet::new();
            for t in succ(b).unwrap_or_default() {
                if let Some(l) = live_in.get(t) {
                    live.extend(l.iter().copied());
                }
            }
            for i in (0..p.blocks[b].insts.len()).rev() {
                let (reads, kills) = effect(b, i);
                for r in kills {
                    live.remove(&r);
                }
                live.extend(reads);
            }
            if live != live_in[b] {
                live_in[b] = live;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    let written: HashSet<Reg> = blocks
        .iter()
        .flat_map(|&b| {
            p.blocks[b]
                .insts
                .iter()
                .flat_map(|i| i.writes.iter().map(|w| w.0))
        })
        .filter(|&r| !exec(r))
        .collect();
    !blocks.iter().any(|&b| {
        succ(b)
            .unwrap_or_default()
            .into_iter()
            .any(|t| !inside.contains(&t) && live_in[t].iter().any(|r| written.contains(r)))
    })
}
