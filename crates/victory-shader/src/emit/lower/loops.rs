//! Send capped loops through their existing exit when it can also serve as
//! their iteration check. Giving a loop a return makes any later group
//! barrier unsafe if some threads can reach that return alone.

use super::*;

#[derive(Default)]
pub(super) struct LoopCaps {
    /// Existing exit branch -> (loop header, exit successor).
    pub(super) exits: Map<usize, (usize, usize)>,
    /// Headers whose back edges increment the count without testing it.
    pub(super) checked_at_exit: Set<usize>,
}

impl LoopCaps {
    pub(super) fn new(p: &Program, st: &Structured) -> Self {
        let order = crate::ir::reverse_postorder(&p.blocks);
        let mut rank = vec![usize::MAX; p.blocks.len()];
        for (i, &b) in order.iter().enumerate() {
            rank[b] = i;
        }
        let mut caps = Self::default();
        let mut checked = Set::default();
        let dom = OnceCell::new();
        for &latch in &order {
            let LaneExit::Jump(header) = st.exits[latch] else {
                continue;
            };
            if rank[header] > rank[latch] || !checked.insert(header) {
                continue;
            }
            let back_edges: Vec<_> = p.blocks[header]
                .preds
                .iter()
                .copied()
                .filter(|&b| rank[b] != usize::MAX && rank[b] >= rank[header])
                .collect();
            // IR construction makes every back edge reducible, so walking
            // toward the header recovers the natural loop with this header.
            let mut inside = Set::default();
            inside.insert(header);
            let mut pending = back_edges.clone();
            while let Some(b) = pending.pop() {
                if inside.insert(b) {
                    pending.extend(p.blocks[b].preds.iter().copied());
                }
            }
            let mut leaving = inside.iter().flat_map(|&b| {
                p.blocks[b]
                    .exit
                    .successors()
                    .into_iter()
                    .filter(|t| !inside.contains(t))
                    .map(move |t| (b, t))
            });
            let Some((branch, exit)) = leaving.next() else {
                continue;
            };
            let LaneExit::Branch {
                taken, not_taken, ..
            } = st.exits[branch]
            else {
                continue;
            };
            if leaving.next().is_some()
                || caps.exits.contains_key(&branch)
                || (taken != exit && not_taken != exit)
                || [taken, not_taken]
                    .into_iter()
                    .any(|t| t != header && rank[t] <= rank[branch])
            {
                continue;
            }
            // The normal exit must be checked before every path back to
            // the header. The original exit edge already supplies the
            // destination's SSA inputs.
            let dom = dom.get_or_init(|| crate::ir::idoms(&p.blocks, &order, &rank));
            if !back_edges.iter().all(|&b| {
                let mut at = b;
                while at != branch && at != header {
                    at = dom[at];
                }
                at == branch
            }) {
                continue;
            }
            caps.exits.insert(branch, (header, exit));
            caps.checked_at_exit.insert(header);
        }
        caps
    }
}
