// `dots` for formats that define one input block's share as `block_dot`.
fn dots(total: u32, blk: u32, xs: vec4<u32>, part: u32, m: u32) -> vec4<f32> {
    var out = vec4<f32>(0.0);
    for (var i = 0u; i < m; i++) {
        out[i] = block_dot(total, blk, xs[i], part);
    }
    return out;
}
