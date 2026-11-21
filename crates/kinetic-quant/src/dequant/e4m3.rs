//! The formats built on E4M3 floats: NVFP4, whose block scales are E4M3,
//! and F8_E4M3, which stores one E4M3 value per element.

use crate::grids::{e4m3, ue4m3_half, MXFP4_VALUES};

// block_nvfp4: u8 d[4] (unsigned E4M3), u8 qs[32] of E2M1 codes. Group g of
// 16 values has scale d[g] and bytes 8g..8g + 8 of qs, low nibbles first.
pub(super) fn nvfp4_block(src: &[u8], dst: &mut [f32]) {
    let (scales, qs) = src.split_at(4);
    for ((&s, q), out) in scales
        .iter()
        .zip(qs.chunks_exact(8))
        .zip(dst.chunks_exact_mut(16))
    {
        let d = ue4m3_half(s);
        let (lo, hi) = out.split_at_mut(8);
        for (j, &q) in q.iter().enumerate() {
            lo[j] = MXFP4_VALUES[(q & 0xF) as usize] as f32 * d;
            hi[j] = MXFP4_VALUES[(q >> 4) as usize] as f32 * d;
        }
    }
}

pub(super) fn f8_e4m3_block(src: &[u8], dst: &mut [f32]) {
    dst[0] = e4m3(src[0]);
}
