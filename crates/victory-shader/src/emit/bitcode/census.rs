//! Counts of a module's instructions by kind (a diagnostic): what a
//! translated program costs before the driver's compiler sees it.
//!
//! `VICTORY_SHADER_CENSUS` (any value) turns the counting on; the counts of
//! the emitted module are read with [`take`].

use super::{Cast, Module, Op, Plan};
use crate::{Container, DxilProgram, FourCC};
use std::cell::RefCell;

/// The kinds, in the order of [`Census::counts`].
pub const KINDS: [&str; 23] = [
    "total",
    "arith",
    "math_call",
    "bitcast",
    "cast",
    "select",
    "phi",
    "extract",
    "buffer_load",
    "cbuffer_load",
    "texture_read",
    "lds_load",
    "buffer_store",
    "lds_store",
    "atomic",
    "handle",
    "annotate",
    "dimensions",
    "barrier",
    "wave_op",
    "io_other",
    "gep",
    "blocks",
];

/// Instruction counts of one written module, by [`KINDS`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Census {
    pub counts: [u32; KINDS.len()],
}

thread_local! {
    /// The modules this thread wrote since the last [`take`]: the length and
    /// a hash of each one's bitcode, with its counts (a program may be
    /// emitted in several forms before one is kept).
    static WRITTEN: RefCell<Vec<(usize, u64, Census)>> = const { RefCell::new(Vec::new()) };
}

fn fnv(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |h, &b| {
        (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
    })
}

/// Whether modules are counted (`VICTORY_SHADER_CENSUS` set).
pub fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_SHADER_CENSUS").is_some())
}

/// The counts of the module whose bitcode the container `dxil` holds, if
/// this thread wrote it since the last call (which forgets the others).
pub fn take(dxil: &[u8]) -> Option<Census> {
    let written = WRITTEN.with(|l| std::mem::take(&mut *l.borrow_mut()));
    let container = Container::parse(dxil).ok()?;
    let bitcode = DxilProgram::parse(container.part(FourCC::DXIL)?.data)
        .ok()?
        .bitcode;
    let hash = fnv(bitcode);
    written
        .into_iter()
        .rev()
        .find(|&(len, h, _)| len == bitcode.len() && h == hash)
        .map(|w| w.2)
}

/// Counts `m`, whose written form is `bitcode`, and keeps the result for
/// [`take`].
pub(super) fn note(m: &Module, plan: &Plan, bitcode: &[u8]) {
    if !enabled() {
        return;
    }
    let c = count(m, plan);
    WRITTEN.with(|l| l.borrow_mut().push((bitcode.len(), fnv(bitcode), c)));
}

fn kind(name: &str) -> usize {
    let n = name.strip_prefix("dx.op.").unwrap_or(name);
    let n = n.split('.').next().unwrap_or(n);
    let k = |s: &str| KINDS.iter().position(|k| *k == s).expect("kind");
    match n {
        "unary" | "binary" | "tertiary" | "unaryBits" | "legacyF16ToF32" | "legacyF32ToF16"
        | "isSpecialFloat" | "dot2" | "dot3" | "dot4" | "dot4AddPacked" | "quaternary" => {
            k("math_call")
        }
        "bufferLoad" | "rawBufferLoad" => k("buffer_load"),
        "cbufferLoadLegacy" | "cbufferLoad" => k("cbuffer_load"),
        "bufferStore" | "rawBufferStore" | "textureStore" => k("buffer_store"),
        "atomicBinOp" | "atomicCompareExchange" => k("atomic"),
        "createHandle" | "createHandleFromBinding" | "createHandleFromHeap" => k("handle"),
        "annotateHandle" => k("annotate"),
        "getDimensions" => k("dimensions"),
        "barrier" => k("barrier"),
        _ if n.starts_with("wave") || n.starts_with("quad") => k("wave_op"),
        _ if n.starts_with("textureLoad")
            || n.starts_with("sample")
            || n.starts_with("textureGather")
            || n.starts_with("calculateLOD") =>
        {
            k("texture_read")
        }
        _ => k("io_other"),
    }
}

/// The instructions the writer actually emits, including its block folding.
fn count(m: &Module, plan: &Plan) -> Census {
    let k = |s: &str| KINDS.iter().position(|k| *k == s).expect("kind");
    let call_kind: Vec<_> = m.funcs.iter().map(|f| kind(&f.name)).collect();
    let mut c = Census::default();
    c.counts[k("blocks")] = plan.layout.len() as u32;
    for &b in &plan.layout {
        let blk = &m.blocks[b];
        for &i in blk
            .phis
            .iter()
            .chain(&blk.body)
            .filter(|&&i| plan.live[i as usize])
        {
            c.counts[0] += 1;
            let at = match &m.insts[i as usize].op {
                Op::Bin(..) | Op::Cmp(..) => k("arith"),
                Op::Cast(Cast::Bitcast, _) => k("bitcast"),
                Op::Cast(..) => k("cast"),
                Op::Select(..) => k("select"),
                Op::Extract(..) => k("extract"),
                Op::Phi(_) => k("phi"),
                Op::Gep(..) => k("gep"),
                Op::Load(_) => k("lds_load"),
                Op::Store(..) => k("lds_store"),
                Op::AtomicRmw(..) | Op::CmpXchg(..) => k("atomic"),
                Op::Call(f, _) => call_kind[*f as usize],
            };
            c.counts[at] += 1;
        }
    }
    c
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::emit::bitcode::{Attr, Bin, Term};
    use crate::emit::{
        container,
        shader::{Shader, flags},
    };

    #[test]
    fn counts_and_hashes_follow_optimized_bitcode() {
        let mut s = Shader::new(false);
        s.threads = Some([32, 1, 1]);
        s.lds_dwords = 1;
        let b = s.entry;
        let i32t = s.t.i32;
        let zero = s.m.i32(0);
        s.op(
            b,
            "waveReadLaneFirst.i32",
            Attr::NoUnwind,
            i32t,
            118,
            &[zero],
        );
        s.flags |= flags::WAVE_OPS;
        let x = s.op(b, "threadId.i32", Attr::ReadNone, i32t, 93, &[zero]);
        let a = s.m.bin(b, Bin::Add, x, zero);
        let b_or = s.m.bin(b, Bin::Or, a, zero);
        let lds = s.lds();
        let dst = s.m.element(b, lds, zero);
        s.m.store(b, dst, b_or);
        s.m.terminate(b, Term::Ret);
        let before = count(&s.m, &s.m.plan());
        assert_eq!(before.counts[0], 5);

        let bytes = s.finish();
        let dxbc = Container::parse(&bytes).unwrap();
        let dxil = dxbc.part(FourCC::DXIL).unwrap().data;
        DxilProgram::parse(dxil).unwrap();
        assert_eq!(
            dxbc.part(FourCC::HASH).unwrap().data,
            container::shader_hash(dxil)
        );
        assert_eq!(bytes[4..20], container::hash(&bytes[20..]));
        let features = dxbc.part(FourCC::SFI0).unwrap().data.try_into().unwrap();
        assert_eq!(u64::from_le_bytes(features) & 0x4000, 0);
        if enabled() {
            let after = take(&bytes).unwrap();
            assert_eq!(after.counts[0], 3);
            assert!(take(&bytes).is_none());
        }
    }
}
