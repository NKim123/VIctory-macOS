//! Byte-checked frame-rate reads for Series 1.0.2.1. The two limiter reads
//! and the timer's minimum-step read change together so the simulation can
//! follow the requested pace.
// Ported from avatarmanz/VIctory-7900XTX (08b88b82), GPL-3.0-or-later.

/// The frame rate the title is given for an interval: `1000 / ms` rounded
/// (16.667 → 60, 8.333 → 120, 33.333 → 30), 0 for no wait (the title's own
/// "no limit"). `VICTORY_TITLE_FPS=<rate>` gives it outright, `off` leaves the
/// title as it is (only our clock changes).
pub fn title_rate(interval_ms: f64, title_fps: Option<&str>) -> Option<u32> {
    match title_fps.map(str::trim) {
        Some("off") => None,
        Some(rate) => rate.parse::<u32>().ok().filter(|r| *r <= 1000),
        None if interval_ms <= 0.0 => Some(0),
        None => Some(((1000.0 / interval_ms).round() as u32).clamp(1, 1000)),
    }
}

/// A read of the setting to change: its offset in the image, what it is,
/// and the bytes there (compared first; the replacement goes at the start).
#[cfg(any(windows, test))]
struct Site {
    #[cfg(windows)]
    at: usize,
    kind: Kind,
    bytes: &'static [u8],
}

#[cfg(any(windows, test))]
#[derive(Clone, Copy)]
enum Kind {
    /// `mov eax, [reg+210h]` before the frame time is made of it: becomes
    /// `mov eax, rate ; nop` (6 bytes).
    Load,
    /// The timer's minimum step is 1/rate. Load the new rate and jump to
    /// the existing float conversion. For no limit, load 1000.
    SimRate,
}

/// The three reads (game_scarlett_master.exe 1.0.2.1, image offsets).
#[cfg(any(windows, test))]
const SITES: [Site; 3] = [
    // Load and subsequent clamp in the device settings.
    Site {
        #[cfg(windows)]
        at: 0x1e0fc62,
        kind: Kind::Load,
        bytes: &[
            0x8b, 0x87, 0x10, 0x02, 0x00, 0x00, 0xba, 0xe8, 0x03, 0x00, 0x00, 0xc5, 0xfa, 0x10,
            0x0d, 0x9f, 0xa1, 0x34, 0x04, 0xb9, 0xe8, 0x03, 0x00, 0x00, 0x3d, 0xe8, 0x03, 0x00,
            0x00, 0x0f, 0x42, 0xd0, 0x85, 0xc0, 0x0f, 0x44, 0xd1, 0xc5, 0xea, 0x2a, 0xc2, 0xc5,
            0xf2, 0x5e, 0xc0, 0xc4, 0xe1, 0xfa, 0x2c, 0xc0, 0x89, 0x05, 0x12, 0x88, 0xae, 0x07,
        ],
    },
    // Load and subsequent clamp when graphics settings are applied.
    Site {
        #[cfg(windows)]
        at: 0x224359a,
        kind: Kind::Load,
        bytes: &[
            0x8b, 0x86, 0x10, 0x02, 0x00, 0x00, 0xba, 0xe8, 0x03, 0x00, 0x00, 0xc5, 0xfa, 0x10,
            0x0d, 0x67, 0x68, 0xf1, 0x03, 0xb9, 0xe8, 0x03, 0x00, 0x00, 0x3d, 0xe8, 0x03, 0x00,
            0x00, 0x0f, 0x42, 0xd0, 0x85, 0xc0, 0x0f, 0x44, 0xd1, 0xb1, 0xff, 0xc5, 0xea, 0x2a,
            0xc2, 0xc5, 0xf2, 0x5e, 0xc0, 0xc4, 0xe1, 0xfa, 0x2c, 0xc0, 0x89, 0x05, 0xd8, 0x4e,
            0x6b, 0x07,
        ],
    },
    // Check and load the setting, followed by its float conversion.
    Site {
        #[cfg(windows)]
        at: 0x2241fdb,
        kind: Kind::SimRate,
        bytes: &[
            0x83, 0xb8, 0x10, 0x02, 0x00, 0x00, 0x00, 0x74, 0x12, 0xe8, 0xc7, 0xa1, 0xe0, 0x02,
            0x8b, 0x80, 0x10, 0x02, 0x00, 0x00, 0xc4, 0xe1, 0xfa, 0x2a, 0xc0,
        ],
    },
];

/// What goes at a site of `kind` for `rate`.
#[cfg(any(windows, test))]
fn replacement(kind: Kind, rate: u32) -> ([u8; 7], usize) {
    match kind {
        // mov eax, rate ; nop
        Kind::Load => {
            let r = rate.to_le_bytes();
            ([0xb8, r[0], r[1], r[2], r[3], 0x90, 0], 6)
        }
        // mov eax, rate ; jmp +13 (to the vcvtsi2ss, 20 bytes on)
        Kind::SimRate => {
            let r = (if rate == 0 { 1000 } else { rate }).to_le_bytes();
            ([0xb8, r[0], r[1], r[2], r[3], 0xeb, 0x0d], 7)
        }
    }
}

/// Apply all reads together so the title's limiter and simulation agree.
///
/// # Safety
/// The image stays mapped and its code has yet to run.
#[cfg(windows)]
pub unsafe fn apply(base: usize, size: usize, rate: u32) -> crate::resolution::Patched {
    // SAFETY: the caller supplies the mapped, idle image.
    unsafe { apply_sites(base, size, rate, &SITES) }
}

#[cfg(windows)]
unsafe fn apply_sites(
    base: usize,
    size: usize,
    rate: u32,
    sites: &[Site; 3],
) -> crate::resolution::Patched {
    use crate::resolution::{self, Site};
    let windows = sites.each_ref().map(|site| Site {
        at: site.at,
        what: "frame rate setting",
        bytes: site.bytes,
    });
    let codes = sites.each_ref().map(|site| replacement(site.kind, rate));
    let parts = std::array::from_fn::<_, 3, _>(|i| (&windows[i], &codes[i].0[..codes[i].1]));
    // SAFETY: the caller supplies the mapped, idle image.
    unsafe { resolution::write_together(base, size, &parts) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_rate_follows_the_clock_by_default() {
        assert_eq!(title_rate(16.667, None), Some(60));
        assert_eq!(title_rate(8.333, None), Some(120));
        assert_eq!(title_rate(33.333, None), Some(30));
        assert_eq!(title_rate(0.0, None), Some(0));
        assert_eq!(title_rate(16.667, Some("off")), None);
        assert_eq!(title_rate(16.667, Some("40")), Some(40));
    }

    #[test]
    fn replacements_preserve_the_simulation_conversion() {
        let (code, len) = replacement(SITES[0].kind, 60);
        assert_eq!(&code[..len], &[0xb8, 60, 0, 0, 0, 0x90]);
        let (code, len) = replacement(SITES[2].kind, 0);
        assert_eq!(code[1..5], 1000u32.to_le_bytes());
        let target = len + usize::from(code[6]);
        assert_eq!(&SITES[2].bytes[target..], &[0xc4, 0xe1, 0xfa, 0x2a, 0xc0]);
    }

    #[cfg(windows)]
    #[test]
    fn a_mismatch_leaves_all_three_reads_alone() {
        use crate::resolution::Patched;
        let mut image = vec![0; 256];
        let sites = std::array::from_fn(|i| Site {
            at: i * 64,
            kind: SITES[i].kind,
            bytes: SITES[i].bytes,
        });
        for site in &sites {
            image[site.at..site.at + site.bytes.len()].copy_from_slice(site.bytes);
        }
        image[sites[2].at] ^= 1;
        let original = image.clone();
        let size = image.len();
        let at = image.as_mut_ptr() as usize;
        // SAFETY: image is still allocated and nothing executes its bytes.
        let patched = unsafe { apply_sites(at, size, 60, &sites) };
        assert_eq!(patched, Patched::Differs);
        assert_eq!(image, original);
    }
}
