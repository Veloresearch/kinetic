// threadgroup bytes: 16384
// y[t][offset + r] = sum over k of w[r][k] x[t][k], for the 16 tokens of a
// full batch and every row r of w, with Metal's SIMD-group matrices, which
// the GPU multiplies 8x8 at a time. Hand-written: WGSL has no such type.
//
// A threadgroup takes 32 rows and all 16 tokens. Each of its SPLITS SIMD
// groups takes its share of the columns: per step of eight it loads four
// tiles of w and two of x and multiplies each pair, so a load serves
// several products. The shares are then added in threadgroup memory.
// Without the split a matrix of few rows would keep few SIMD groups busy
// and most of the GPU idle.
//
// The weights are f16 and the sums f32. Unlike the GEMV kernels this does
// not give a token the sum it gets alone, so it only runs prompt batches
// (`Commands::set_prompt`). A sum is the same whenever it is computed:
// the order of its terms depends on the matrix alone.
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// Tiles of eight rows and of eight tokens per SIMD group. More of either
// no longer fits the registers and runs several times slower.
#define ROW_TILES 4
#define TOKEN_TILES 2
// SIMD groups of a threadgroup; `gemv.rs` checks the columns divide.
#define SPLITS 8
// Sums one SIMD group holds.
#define SUMS (ROW_TILES * TOKEN_TILES * 64)

struct Shape {
    uint rows;
    uint cols;
    uint stride;
    uint offset;
};

[[max_total_threads_per_threadgroup(256)]] kernel void main_(
    device const half *w [[buffer(0)]],
    device const float *x [[buffer(1)]],
    device float *y [[buffer(2)]],
    constant Shape &s [[buffer(3)]],
    threadgroup float *shares [[threadgroup(0)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint split [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    const ulong r0 = (ulong)group.x * 8 * ROW_TILES;
    const ulong span = s.cols / SPLITS;
    const ulong first = split * span;
    simdgroup_float8x8 sum[ROW_TILES][TOKEN_TILES];
    for (int r = 0; r < ROW_TILES; r++) {
        for (int t = 0; t < TOKEN_TILES; t++) {
            sum[r][t] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }
    simdgroup_half8x8 a[ROW_TILES];
    simdgroup_float8x8 b[TOKEN_TILES];
    for (ulong k = first; k < first + span; k += 8) {
        for (int r = 0; r < ROW_TILES; r++) {
            simdgroup_load(a[r], w, s.cols, ulong2(k, r0 + 8 * r), false);
        }
        // x holds a token per row; the product wants a token per column.
        for (int t = 0; t < TOKEN_TILES; t++) {
            simdgroup_load(b[t], x, s.cols, ulong2(k, 8 * t), true);
        }
        for (int r = 0; r < ROW_TILES; r++) {
            for (int t = 0; t < TOKEN_TILES; t++) {
                simdgroup_multiply_accumulate(sum[r][t], a[r], b[t], sum[r][t]);
            }
        }
    }
    // Tile (r, t) of this split as 64 floats, a row of the matrix per eight.
    for (int r = 0; r < ROW_TILES; r++) {
        for (int t = 0; t < TOKEN_TILES; t++) {
            simdgroup_store(sum[r][t], shares + split * SUMS + (r * TOKEN_TILES + t) * 64, 8);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (split != 0) {
        return;
    }
    // The first SIMD group adds the shares, each thread its part of the
    // sums, always in split order.
    for (uint e = lane * (SUMS / 32); e < (lane + 1) * (SUMS / 32); e++) {
        float total = 0.0f;
        for (uint p = 0; p < SPLITS; p++) {
            total += shares[p * SUMS + e];
        }
        const uint tile = e / 64;
        const uint row = 8 * (tile / TOKEN_TILES) + (e % 64) / 8;
        const uint token = 8 * (tile % TOKEN_TILES) + e % 8;
        y[token * s.stride + s.offset + r0 + row] = total;
    }
}
