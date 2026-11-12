// block_q4_K: d, dmin, scales[12], qs[128]; 144 bytes, nine words.

// One invocation's share of a block for the first `m` of four input blocks.
// The weights are loaded and their scales unpacked once for all of them;
// each input's share is then computed as `k4_combine` computes it.
fn dots(total: u32, blk: u32, xs: vec4<u32>, j: u32, m: u32) -> vec4<f32> {
    let head = w[blk * 9u];
    let q = w[blk * 9u + 1u + j];
    // The sums are of at most 16 products below 15 * 127, exact in f32, so
    // they are taken in float, which the GPU multiplies faster than int8.
    let l0 = vec4<f32>(unpack4xU8(q.x & NIBBLES));
    let l1 = vec4<f32>(unpack4xU8(q.y & NIBBLES));
    let l2 = vec4<f32>(unpack4xU8(q.z & NIBBLES));
    let l3 = vec4<f32>(unpack4xU8(q.w & NIBBLES));
    let h0 = vec4<f32>(unpack4xU8((q.x >> 4u) & NIBBLES));
    let h1 = vec4<f32>(unpack4xU8((q.y >> 4u) & NIBBLES));
    let h2 = vec4<f32>(unpack4xU8((q.z >> 4u) & NIBBLES));
    let h3 = vec4<f32>(unpack4xU8((q.w >> 4u) & NIBBLES));
    let s = 2u * (j / 2u);
    let a = k4_scale_min(s, head.y, head.z, head.w);
    let c = k4_scale_min(s + 1u, head.y, head.z, head.w);
    let dm = unpack2x16float(head.x);
    var out = vec4<f32>(0.0);
    for (var i = 0u; i < m; i++) {
        let b = xs[i];
        let xi = k4_input(b, j);
        let x0 = xd[b * 8u + s];
        let x1 = xd[b * 8u + s + 1u];
        let u = xq[xi];
        let v = xq[xi + 2u];
        let lo = dot(l0, vec4<f32>(unpack4xI8(u.x))) + dot(l1, vec4<f32>(unpack4xI8(u.y)))
            + dot(l2, vec4<f32>(unpack4xI8(u.z))) + dot(l3, vec4<f32>(unpack4xI8(u.w)));
        let hi = dot(h0, vec4<f32>(unpack4xI8(v.x))) + dot(h1, vec4<f32>(unpack4xI8(v.y)))
            + dot(h2, vec4<f32>(unpack4xI8(v.z))) + dot(h3, vec4<f32>(unpack4xI8(v.w)));
        var r = dm.x * (a.x * x0.x * lo + c.x * x1.x * hi);
        if j % 2u == 0u {
            r -= dm.y * (a.y * x0.y + c.y * x1.y);
        }
        out[i] = r;
    }
    return out;
}
