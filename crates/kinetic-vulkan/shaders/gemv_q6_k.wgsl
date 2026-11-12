// block_q6_K is 210 bytes, so blocks are only 2-byte aligned. The device copy
// is planar: all ql[128], then all qh[64], all scales[16], all f16 d.
//
// A block is two halves of 128 elements; in half n, element 32g + l (g < 4,
// l < 32) takes its low four bits from ql[64n + l + 32(g & 1)] (low nibble
// for g < 2, high for g >= 2) and its top two from bits 2g..2g+1 of
// qh[32n + l]. Each run of 16 elements has its own int8 scale.
//
// Eight invocations share a block. Invocation j = 4n + k loads ql bytes
// 64n + 16k, which hold elements l = 16(k & 1) .. +16 of group g = k / 2
// (low nibbles) and of group g + 2 (high nibbles).

const LANES: u32 = 8u;

// Signed byte `i` of a word.
fn sbyte(v: vec4<u32>, i: u32) -> f32 {
    return f32(extractBits(bitcast<i32>(v[i / 4u]), 8u * (i % 4u), 8u));
}

// One invocation's share of a block for the first `m` of four input blocks: the weights are loaded and unpacked
// once. Quants are stored as q + 32; dotting 32 with the input takes it
// back out.
fn dots(total: u32, blk: u32, xs: vec4<u32>, j: u32, m: u32) -> vec4<f32> {
    let n = j / 4u;
    let k = j % 4u;
    let g = k / 2u;

    let lq = w[blk * 8u + j];
    let hq = w[total * 8u + blk * 4u + 2u * n + (k & 1u)];
    let low2 = vec4<u32>(0x03030303u);
    let lo = (lq & vec4<u32>(NIBBLES)) | (((hq >> vec4<u32>(2u * g)) & low2) << vec4<u32>(4u));
    let hi = ((lq >> vec4<u32>(4u)) & vec4<u32>(NIBBLES))
        | (((hq >> vec4<u32>(2u * g + 4u)) & low2) << vec4<u32>(4u));
    // Quants are stored as q + 32. The sums are of at most 16 products
    // below 32 * 127, exact in f32, so they are taken in float, which the
    // GPU multiplies faster than int8.
    let off = vec4<f32>(32.0);
    let l0 = vec4<f32>(unpack4xU8(lo.x)) - off;
    let l1 = vec4<f32>(unpack4xU8(lo.y)) - off;
    let l2 = vec4<f32>(unpack4xU8(lo.z)) - off;
    let l3 = vec4<f32>(unpack4xU8(lo.w)) - off;
    let h0 = vec4<f32>(unpack4xU8(hi.x)) - off;
    let h1 = vec4<f32>(unpack4xU8(hi.y)) - off;
    let h2 = vec4<f32>(unpack4xU8(hi.z)) - off;
    let h3 = vec4<f32>(unpack4xU8(hi.w)) - off;
    let scales = w[total * 12u + blk];
    let sc = 8u * n + (k & 1u) + 2u * g;
    let sc0 = sbyte(scales, sc);
    let sc1 = sbyte(scales, sc + 4u);
    let d = f16_at(total * 13u, blk);
    var out = vec4<f32>(0.0);
    for (var i = 0u; i < m; i++) {
        let e = xs[i] * 256u + 128u * n + 32u * g + 16u * (k & 1u);
        let u = xq[e / 16u];
        let v = xq[e / 16u + 4u];
        let s0 = dot(l0, vec4<f32>(unpack4xI8(u.x))) + dot(l1, vec4<f32>(unpack4xI8(u.y)))
            + dot(l2, vec4<f32>(unpack4xI8(u.z))) + dot(l3, vec4<f32>(unpack4xI8(u.w)));
        let s1 = dot(h0, vec4<f32>(unpack4xI8(v.x))) + dot(h1, vec4<f32>(unpack4xI8(v.y)))
            + dot(h2, vec4<f32>(unpack4xI8(v.z))) + dot(h3, vec4<f32>(unpack4xI8(v.w)));
        let r = sc0 * xd[e / 32u].x * s0 + sc1 * xd[e / 32u + 2u].x * s1;
        out[i] = d * r;
    }
    return out;
}
