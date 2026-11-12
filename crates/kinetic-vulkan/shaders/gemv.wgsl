// Shared part of every GEMV kernel; build.rs appends one weight format to it,
// which defines `LANES` and `dots`: one invocation's share of a weight block
// for the first `m` of four input blocks (`gemv_dots.wgsl` builds it from a
// format's `block_dot`).
//
// y = W x for some rows x of the input, with one subgroup of 32 invocations
// per row of W (the host pins the subgroup size). `LANES` consecutive
// invocations share a weight block, so a subgroup covers 32 / LANES blocks
// per step. `block_dot` returns one invocation's share of a block; `blk`
// indexes the block in the whole matrix and `b` the block of the input, and
// `total` is the matrix's block count, which planar layouts need to find
// their planes.
//
// The batch says which input rows: with `head` zero, its first `batch[0]`
// rows; otherwise `batch[3]` rows from row `batch[2]`. Output row t of y
// starts at `t * stride + offset`. Each row's sums run in the same order
// whatever else is in the batch.

struct Shape {
    rows: u32,
    blocks_per_row: u32,
    stride: u32,
    offset: u32,
    head: u32,
}

var<immediate> shape: Shape;

@group(0) @binding(0) var<storage, read> w: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;
// The input quantized to int8, four per word, in blocks of 32 elements.
@group(0) @binding(2) var<storage, read> xq: array<vec4<u32>>;
// Per block of 32: (d, d * sum of its int8 values).
@group(0) @binding(3) var<storage, read> xd: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
@group(0) @binding(5) var<storage, read> batch: array<u32>;

const ROWS_PER_GROUP: u32 = 4u;
const NIBBLES: u32 = 0x0F0F0F0Fu;

@compute @workgroup_size(128)
fn main(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
    @builtin(subgroup_id) slot: u32,
    @builtin(subgroup_invocation_id) lane: u32,
) {
    let row = (group.y * groups.x + group.x) * ROWS_PER_GROUP + slot;
    if row >= shape.rows {
        return;
    }
    let part = lane % LANES;
    let total = shape.rows * shape.blocks_per_row;
    let first = row * shape.blocks_per_row;
    var start = 0u;
    var count = batch[0];
    if shape.head != 0u {
        start = batch[2];
        count = batch[3];
    }

    let n = shape.blocks_per_row;
    let step = 32u / LANES;
    let out = shape.offset + row;
    // Up to four tokens at a time share each weight load. One token alone
    // takes the same path with `m` = 1, so every token's sum runs through
    // the same code in the same order and batching never changes a result.
    for (var t = 0u; t < count; t += 4u) {
        let m = min(4u, count - t);
        let xb = (start + t) * n;
        let xs = vec4<u32>(xb, xb + n, xb + 2u * n, xb + 3u * n);
        var sum = vec4<f32>(0.0);
        var b = lane / LANES;
        // two blocks per step keep two weight loads in flight
        for (; b + step < n; b += 2u * step) {
            sum += dots(total, first + b, xs + b, part, m)
                + dots(total, first + b + step, xs + b + step, part, m);
        }
        if b < n {
            sum += dots(total, first + b, xs + b, part, m);
        }
        // `m` is the same for the whole subgroup, so the reductions stay in
        // uniform control flow; a lone token pays for one, not four.
        for (var i = 0u; i < m; i++) {
            let all = subgroupAdd(sum[i]);
            if lane == 0u {
                y[(t + i) * shape.stride + out] = all;
            }
        }
    }
}

// Four signed int8 dot products of 16 values.
fn dot16(a: vec4<u32>, b: vec4<u32>) -> i32 {
    return dot4I8Packed(a.x, b.x) + dot4I8Packed(a.y, b.y) + dot4I8Packed(a.z, b.z)
        + dot4I8Packed(a.w, b.w);
}

// Element `i` of an f16 plane that starts at word `base` of the weights.
fn f16_at(base: u32, i: u32) -> f32 {
    let pair = unpack2x16float(w[base + i / 8u][(i / 2u) % 4u]);
    return select(pair.x, pair.y, (i & 1u) == 1u);
}
