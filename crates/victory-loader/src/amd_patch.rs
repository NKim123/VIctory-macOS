//! Startup-only replacements for locally verified console instruction sites.

use std::ops::Range;

use crate::PreparedImage;
use crate::masked::jump;

const CLZERO: [u8; 3] = [0x0f, 0x01, 0xfc];
const ADD_LINE: [u8; 4] = [0x48, 0x83, 0xc0, 0x40];
const MAX_SITES: usize = 256;
const ZERO_STORES_START: usize = 6;
const ZERO_STORES_END: usize = ZERO_STORES_START + 8 * 8;
#[cfg(all(windows, target_arch = "x86_64"))]
const PAGE: usize = 4096;
const EXECUTE: u32 = 0x2000_0000;
const WRITE: u32 = 0x8000_0000;

fn sites(value: &str, hash: blake3::Hash) -> Option<Vec<usize>> {
    if matches!(value, "" | "0") {
        return Some(Vec::new());
    }
    let (build, list) = value.split_once(':')?;
    if blake3::Hash::from_hex(build).ok()? != hash {
        return None;
    }
    let mut sites: Vec<_> = list
        .split(',')
        .take(MAX_SITES + 1)
        .map(|item| {
            let item = item.trim();
            let digits = item
                .strip_prefix("0x")
                .or_else(|| item.strip_prefix("0X"))
                .unwrap_or(item);
            u32::from_str_radix(digits, 16).map(|rva| rva as usize)
        })
        .collect::<Result<_, _>>()
        .ok()?;
    if sites.len() > MAX_SITES {
        return None;
    }
    sites.sort_unstable();
    sites.dedup();
    Some(sites)
}

#[derive(Clone, Copy, Debug)]
enum Change {
    Wait([u8; 3]),
    Zero { next: [u8; 4], len: usize },
}

impl Change {
    fn verify(code: &[u8], at: usize, zero: bool) -> Option<(Self, Range<usize>)> {
        let instruction = code.get(at..at + 3)?;
        if !zero {
            let bytes = match instruction {
                [0x0f, 0x01, 0xfa] => [0x0f, 0x1f, 0x00], // monitorx -> nop
                [0x0f, 0x01, 0xfb] => [0xf3, 0x90, 0x90], // mwaitx -> pause; nop
                _ => return None,
            };
            return Some((Self::Wait(bytes), at..at + 3));
        }
        if instruction != CLZERO {
            return None;
        }

        let after = at + CLZERO.len();
        if let Some(code) = code.get(after..at + 12)
            && clzero_forward_limit(code).is_some()
        {
            return Some((
                Self::Zero {
                    next: ADD_LINE,
                    len: 7,
                },
                at..at + 12,
            ));
        }

        let start = at.checked_sub(7)?;
        let before = code.get(start..at)?;
        let after = code.get(after..at + 8)?;
        let (cursor, _) = clzero_backward_registers(before, after)?;
        if cursor == 4 {
            // A stub must sit outside any stack adjustment.
            return None;
        }
        let [rex, op, modrm, ..] = *after else {
            return None;
        };
        Some((
            Self::Zero {
                next: [rex, op, modrm, 0],
                len: 6,
            },
            start..at + 8,
        ))
    }
}

/// Limit of `clzero; add rax, 64; cmp rax, limit; jb` back to clzero.
pub(crate) fn clzero_forward_limit(after: &[u8]) -> Option<u8> {
    let [0x48, 0x83, 0xc0, 0x40, cmp @ .., 0x72, 0xf4] = after else {
        return None;
    };
    let (cursor, limit) = compare_registers(cmp)?;
    (cursor == 0 && limit != 0).then_some(limit)
}

/// Registers of `mov rax, cursor; add cursor, 64; clzero; cmp cursor, limit; jb` back to mov.
pub(crate) fn clzero_backward_registers(before: &[u8], after: &[u8]) -> Option<(u8, u8)> {
    let [
        rex @ (0x48 | 0x4c),
        0x89,
        load,
        add_rex @ (0x48 | 0x49),
        0x83,
        add,
        0x40,
    ] = *before
    else {
        return None;
    };
    let [cmp @ .., 0x72, 0xf1] = after else {
        return None;
    };
    let (cursor, limit) = compare_registers(cmp)?;
    (cursor != 0
        && limit != 0
        && load & 0xc7 == 0xc0
        && cursor == (load >> 3) & 7 | (rex & 4) << 1
        && add & 0xf8 == 0xc0
        && cursor == add & 7 | (add_rex & 1) << 3)
        .then_some((cursor, limit))
}

/// `cmp rm, reg` between 64-bit registers (`REX.W 39 /r`).
pub(crate) fn compare_registers(code: &[u8]) -> Option<(u8, u8)> {
    let &[rex @ 0x48..=0x4f, 0x39, modrm] = code else {
        return None;
    };
    (modrm >> 6 == 3).then_some((
        modrm & 7 | (rex & 1) << 3,
        (modrm >> 3) & 7 | (rex & 4) << 1,
    ))
}

#[derive(Debug)]
struct Site {
    rva: usize,
    span: Range<usize>,
    guard_rva: usize,
    expected: Vec<u8>,
    change: Change,
}

fn verify(image: &PreparedImage, rva: usize, zero: bool) -> Option<Site> {
    let mut sections = image.info.sections.iter().filter_map(|s| {
        let base = s.rva as usize;
        let size = match s.virtual_size {
            0 => s.file_size,
            size => size.min(s.file_size),
        } as usize;
        let at = rva.checked_sub(base)?;
        (at < size).then_some((s, base, size, at))
    });
    let (section, base, size, at) = sections.next()?;
    if sections.next().is_some() || section.characteristics & (EXECUTE | WRITE) != EXECUTE {
        return None;
    }
    let file = section.file_offset as usize;
    let code = image.bytes.get(file..file.checked_add(size)?)?;
    let (change, span) = Change::verify(code, at, zero)?;
    let span = base + span.start..base + span.end;
    if span.end > image.info.size_of_image as usize {
        return None;
    }
    // Include the surrounding bytes a previous x86 rewrite (at most 15 bytes
    // per instruction) could have changed even if its common bytes match.
    let start = base.max(span.start.saturating_sub(14));
    let end = (base + size)
        .min(span.end + 14)
        .min(image.info.size_of_image as usize);
    Some(Site {
        rva,
        span,
        guard_rva: start,
        expected: code.get(start - base..end - base)?.to_vec(),
        change,
    })
}

fn intersects(a: &Range<usize>, b: &Range<usize>) -> bool {
    a.start < b.end && b.start < a.end
}

fn rejects_overlap(sites: &[Site], i: usize) -> bool {
    sites
        .iter()
        .enumerate()
        .any(|(j, other)| i != j && intersects(&sites[i].span, &other.span))
}

fn zero_stub(site: u64, stub: u64, next: [u8; 4], len: usize) -> Option<Vec<u8>> {
    // CLZERO ignores the low six address bits. Keep the original address
    // and flags so an unhandled store fault can retry at the title PC.
    let mut code = vec![0x9c, 0x50, 0x48, 0x83, 0xe0, 0xc0]; // pushfq; push rax; and rax,-64
    for off in (0..64).step_by(8) {
        code.extend([0x48, 0xc7, 0x40, off, 0, 0, 0, 0]); // mov qword [rax+off], 0
    }
    debug_assert_eq!(code.len(), ZERO_STORES_END);
    code.extend([0x58, 0x9d]); // pop rax; popfq
    code.extend_from_slice(&next[..len - CLZERO.len()]);
    code.extend(jump(stub + code.len() as u64, 5, site + len as u64)?);
    Some(code)
}

#[cfg(all(windows, target_arch = "x86_64"))]
mod windows;
#[cfg(all(windows, target_arch = "x86_64"))]
pub use windows::Patches;

#[cfg(test)]
mod tests {
    use super::*;

    pub(super) const FORWARD: [u8; 12] = [
        0x0f, 0x01, 0xfc, 0x48, 0x83, 0xc0, 0x40, 0x48, 0x39, 0xd0, 0x72, 0xf4,
    ];
    pub(super) const BACKWARD: [u8; 15] = [
        0x48, 0x89, 0xc8, 0x48, 0x83, 0xc1, 0x40, 0x0f, 0x01, 0xfc, 0x48, 0x39, 0xd1, 0x72, 0xf1,
    ];

    #[test]
    fn opt_in_requires_the_build_and_explicit_sites() {
        let hash = blake3::hash(b"synthetic image");
        assert_eq!(sites("0", hash), Some(vec![]));
        assert_eq!(sites("", hash), Some(vec![]));
        for value in ["all", "1", "1000", "wrong:1000"] {
            assert_eq!(sites(value, hash), None);
        }
        assert_eq!(
            sites(&format!("{hash}:ff, 0x10, 0XFF"), hash),
            Some(vec![0x10, 0xff])
        );
        assert_eq!(sites(&format!("{hash}:ff,zz"), hash), None);
        let other = blake3::hash(b"other build");
        assert_eq!(sites(&format!("{other}:ff"), hash), None);
    }

    #[test]
    fn checks_shape_and_section_and_excludes_overlaps_in_both_directions() {
        use victory_pe::{ImageInfo, SectionInfo};

        let mut bytes = vec![0xcc; 100];
        bytes[32..44].copy_from_slice(&FORWARD);
        bytes[52..67].copy_from_slice(&BACKWARD);
        bytes[80..83].copy_from_slice(&[0x0f, 0x01, 0xfa]);
        let mut image = PreparedImage {
            info: ImageInfo {
                machine: victory_pe::MACHINE_AMD64,
                characteristics: 0,
                timestamp: 0,
                subsystem: 0,
                dll_characteristics: 0,
                image_base: 0,
                size_of_image: 8192,
                entry_rva: 4096,
                stack_reserve: 0,
                stack_commit: 0,
                sections: vec![SectionInfo {
                    name: ".text".into(),
                    rva: 4096,
                    virtual_size: 100,
                    file_offset: 0,
                    file_size: 100,
                    characteristics: EXECUTE,
                }],
                imports: vec![],
                delay_imports: vec![],
                exports: vec![],
                tls: None,
                has_relocations: false,
                has_exception_table: false,
                has_load_config: false,
                codeview: None,
            },
            original_hash: blake3::hash(&bytes),
            bytes,
        };
        let first = verify(&image, 4096 + 32, true).unwrap();
        assert_eq!(first.rva, 4128);
        assert_eq!(first.span, 4128..4140);
        assert_eq!(first.guard_rva, 4114);
        assert_eq!(first.expected, image.bytes[18..58]);
        let second = verify(&image, 4096 + 59, true).unwrap();
        assert_eq!(second.span, 4148..4163);
        let wait = verify(&image, 4096 + 80, false).unwrap();
        assert!(matches!(wait.change, Change::Wait([0x0f, 0x1f, 0])));
        let conflicts = [
            first,
            second,
            verify(&image, 4096 + 32, true).unwrap(),
            wait,
        ];
        assert!(rejects_overlap(&conflicts, 0));
        assert!(!rejects_overlap(&conflicts, 1));
        assert!(rejects_overlap(&conflicts, 2));
        assert!(!rejects_overlap(&conflicts, 3));

        assert!(verify(&image, 4096 + 33, true).is_none());
        assert!(verify(&image, 4096 + 80, true).is_none());
        image.bytes[43] = 0xf3; // Different branch displacement.
        assert!(verify(&image, 4096 + 32, true).is_none());
        image.info.sections[0].virtual_size = 82; // Last byte is unbacked.
        assert!(verify(&image, 4096 + 80, false).is_none());
        image.info.sections[0].characteristics |= WRITE;
        assert!(verify(&image, 4096 + 59, true).is_none());

        let Change::Zero { next, len } = conflicts[1].change else {
            panic!("clzero expected")
        };
        let stub = zero_stub(0x1000, 0x2000, next, len).unwrap();
        assert_eq!(&stub[70..75], &[0x58, 0x9d, 0x48, 0x39, 0xd1]);
        assert!(zero_stub(0x1000, 0x1_0000_0000, next, len).is_none());
    }
}
