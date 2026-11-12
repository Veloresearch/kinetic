// block_q4_K: d, dmin, scales[12], qs[128]; 144 bytes, nine words.

// One invocation's share of a block for the first `m` of four input blocks.
// The weights are loaded and their scales unpacked once for all of them;
// each input's share is then computed as `k4_combine` computes it.
fn dots(total: u32, blk: u32, xs: vec4<u32>, j: u32, m: u32) -> vec4<f32> {
    let head = w[blk * 9u];
    let q = w[blk * 9u + 1u + j];
    let lo = q & vec4<u32>(NIBBLES);
    let hi = (q >> vec4<u32>(4u)) & vec4<u32>(NIBBLES);
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
        var r = dm.x * (a.x * x0.x * f32(dot16(lo, xq[xi])) + c.x * x1.x * f32(dot16(hi, xq[xi + 2u])));
        if j % 2u == 0u {
            r -= dm.y * (a.y * x0.y + c.y * x1.y);
        }
        out[i] = r;
    }
    return out;
}
