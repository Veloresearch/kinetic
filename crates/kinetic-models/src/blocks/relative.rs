//! Attention with a learned bias by relative distance and no rotation
//! (Inkling 2). Query and key heads and each head's relative terms are
//! clipped to a largest norm, value heads are normalized, and every query
//! has a sink logit computed from its input. Causal short convolutions with
//! a residual run over the keys, the values, the attention output and the
//! layer's feed-forward output, which keep their inputs in this layer's
//! state.

use kinetic_cpu::{dot, matmul, matmul_many, pool, rms_norm, sigmoid};
use kinetic_quant::Matrix;

use super::attention::KvCache;
use super::conv_window::ConvWindow;
use super::tree::Tree;

pub struct RelAttention<'a> {
    /// RMS norms of the query, key and value inputs.
    pub norms: [Vec<f32>; 3],
    pub q: Matrix<'a>,
    pub k: Matrix<'a>,
    pub v: Matrix<'a>,
    pub o: Matrix<'a>,
    /// Gates every value of the attention output through a sigmoid.
    pub gate: Matrix<'a>,
    /// The sink logit of each query head.
    pub sink: Matrix<'a>,
    /// Multiplies the queries before they are clipped.
    pub q_scale: Vec<f32>,
    /// `n_rel` relative terms per query head.
    pub rel: Matrix<'a>,
    /// Multiplies the relative terms before they are clipped.
    pub rel_scale: Vec<f32>,
    /// `n_rel` weights per distance below the extent; farther keys get no
    /// bias.
    pub rel_bias: Vec<f32>,
    pub n_rel: usize,
    /// Taps of the key, value, attention output and feed-forward output
    /// convolutions, `taps` per channel, oldest input first.
    pub convs: [Vec<f32>; 4],
    pub taps: usize,
    /// On full-attention layers: the key and value projections to `taps`
    /// more taps per key-value head that depend on the input, newest
    /// input first.
    pub dynamic: Option<[Matrix<'a>; 2]>,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub window: Option<usize>,
    /// The largest norm of a query or key head.
    pub qk_clip: f32,
    /// The largest norm of a head's relative terms.
    pub rel_clip: f32,
    /// On full-attention layers, the floor and alpha of
    /// `1 + alpha ln(max(1, (pos + 1) / floor))`, which scales the queries
    /// and relative terms at position `pos`.
    pub log_scale: Option<(f32, f32)>,
    pub eps: f32,
}

/// The keys and values of every position attended, and the last inputs of
/// each convolution.
#[derive(Clone)]
pub struct RelState {
    pub(super) kv: KvCache,
    /// In the order of [`RelAttention::convs`].
    pub(super) convs: [ConvWindow; 4],
}

impl RelState {
    pub fn checkpoint(&mut self) {
        self.kv.checkpoint();
        self.convs.iter_mut().for_each(ConvWindow::checkpoint);
    }
}

impl RelAttention<'_> {
    fn kv_dim(&self) -> usize {
        self.n_head_kv * self.head_dim
    }

    fn conv_channels(&self) -> [usize; 4] {
        let d = self.o.rows;
        [self.kv_dim(), self.kv_dim(), d, d]
    }

    pub fn new_state(&self) -> RelState {
        RelState {
            kv: KvCache::default(),
            convs: self.conv_channels().map(|c| ConvWindow::new(c, self.taps)),
        }
    }

    /// Floats of the state of a sequence `len` tokens long.
    pub fn state_floats(&self, len: usize) -> usize {
        2 * self.kv_dim() * self.window.map_or(len, |w| w.min(len))
            + self.conv_channels().iter().sum::<usize>() * (self.taps - 1)
    }

    /// Lengths of the buffers of a state after `pos` tokens, in the order
    /// of [`RelState::buffers`].
    pub fn buffer_lens(&self, pos: usize) -> Vec<usize> {
        let rows = self.window.map_or(pos, |w| pos.min(w));
        let mut lens = vec![rows * self.kv_dim(); 2];
        lens.extend(self.conv_channels().map(|c| c * (self.taps - 1)));
        lens
    }

    pub fn rollback(&self, s: &mut RelState, rows: &[usize]) {
        s.kv.rollback(rows, self.window);
        s.convs.iter_mut().for_each(|c| c.rollback(rows));
    }

    /// Runs a batch from position `pos0` on the query, key and value
    /// inputs, before their norms, and returns the output after its
    /// convolution. With `tree`, each row continues from its parent row.
    pub fn forward(
        &self,
        inputs: [&[f32]; 3],
        pos0: usize,
        tree: Option<&Tree>,
        s: &mut RelState,
    ) -> Vec<f32> {
        let d = self.q.cols;
        let n = inputs[0].len() / d;
        let (hd, kv_dim) = (self.head_dim, self.kv_dim());
        let depth = |t: usize| tree.map_or(t, |tree| tree.depth(t));
        let [x0, x1, x2] = std::array::from_fn(|j| {
            let mut x = inputs[j].to_vec();
            pool::for_each_chunk(&mut x, d, |_, row| rms_norm(row, &self.norms[j], self.eps));
            x
        });

        let mut q = vec![0.0; n * self.q.rows];
        let mut r = vec![0.0; n * self.rel.rows];
        let mut sink = vec![0.0; n * self.n_head];
        let mut gate = vec![0.0; n * self.gate.rows];
        matmul_many(
            &[&self.q, &self.rel, &self.sink, &self.gate],
            &x0,
            &mut [&mut q, &mut r, &mut sink, &mut gate],
        );
        let taps = self
            .dynamic
            .as_ref()
            .map(|[k, v]| [project(k, &x1), project(v, &x2)]);
        let mut k = project(&self.k, &x1);
        let mut v = project(&self.v, &x2);
        let [k_taps, v_taps] = match &taps {
            Some([k, v]) => [Some(&k[..]), Some(&v[..])],
            None => [None, None],
        };
        k = residual_conv(&k, &self.convs[0], self.taps, k_taps, &mut s.convs[0], tree);
        v = residual_conv(&v, &self.convs[1], self.taps, v_taps, &mut s.convs[1], tree);

        let ones = vec![1.0; hd];
        for t in 0..n {
            let tau = self.log_scale.map_or(1.0, |(floor, alpha)| {
                let pos = (pos0 + depth(t)) as f32;
                1.0 + alpha * ((pos + 1.0) * (1.0 / floor)).max(1.0).ln()
            });
            let q = q[t * self.q.rows..][..self.q.rows].chunks_exact_mut(hd);
            for (q, scale) in q.zip(self.q_scale.chunks_exact(hd)) {
                q.iter_mut().zip(scale).for_each(|(q, s)| *q *= s);
                clip(q, self.qk_clip, tau);
            }
            let r = r[t * self.rel.rows..][..self.rel.rows].chunks_exact_mut(self.n_rel);
            for (r, scale) in r.zip(self.rel_scale.chunks_exact(self.n_rel)) {
                r.iter_mut().zip(scale).for_each(|(r, s)| *r *= s);
                clip(r, self.rel_clip, tau);
            }
            for k in k[t * kv_dim..][..kv_dim].chunks_exact_mut(hd) {
                clip(k, self.qk_clip, 1.0);
            }
            for v in v[t * kv_dim..][..kv_dim].chunks_exact_mut(hd) {
                rms_norm(v, &ones, self.eps);
            }
        }

        let extent = self.rel_bias.len() / self.n_rel;
        let stored = s.kv.k.len() / kv_dim;
        let cached: Vec<usize> = (0..stored)
            .map(|i| match self.window {
                Some(w) if stored == w => pos0 - 1 - (pos0 - 1 - i) % w,
                _ => i,
            })
            .collect();
        let scale = 1.0 / (hd as f32).sqrt();
        let group = self.n_head / self.n_head_kv;
        let mut out = vec![0.0; n * self.n_head * hd];
        pool::for_each_chunk(&mut out, hd, |i, out| {
            let (t, head) = (i / self.n_head, i % self.n_head);
            let pos = pos0 + depth(t);
            let q = &q[t * self.q.rows + head * hd..][..hd];
            let r = &r[t * self.rel.rows + head * self.n_rel..][..self.n_rel];
            let sink = sink[t * self.n_head + head];
            let bias = |dist: usize| {
                if dist < extent {
                    dot(&self.rel_bias[dist * self.n_rel..][..self.n_rel], r) - sink
                } else {
                    -sink
                }
            };
            let kv = head / group * hd;
            let path = tree.map_or_else(|| (0..=t).collect(), |tree| tree.path(t));
            let keys: Vec<(usize, &[f32], &[f32])> = cached
                .iter()
                .enumerate()
                .map(|(i, &p)| {
                    (
                        p,
                        &s.kv.k[i * kv_dim..][..kv_dim],
                        &s.kv.v[i * kv_dim..][..kv_dim],
                    )
                })
                .chain(path.iter().map(|&j| {
                    let row = j * kv_dim..(j + 1) * kv_dim;
                    (pos0 + depth(j), &k[row.clone()], &v[row])
                }))
                .filter(|&(p, _, _)| self.window.is_none_or(|w| pos - p < w))
                .collect();
            // The sink is one more logit of zero: each bias already has the
            // query's sink logit taken off.
            let mut w: Vec<f32> = keys
                .iter()
                .map(|&(p, k, _)| dot(q, &k[kv..][..hd]) * scale + bias(pos - p))
                .chain([0.0])
                .collect();
            kinetic_cpu::softmax(&mut w);
            for (&(_, _, v), w) in keys.iter().zip(&w) {
                out.iter_mut()
                    .zip(&v[kv..][..hd])
                    .for_each(|(o, v)| *o += w * v);
            }
        });
        for t in 0..n {
            let row = t * kv_dim..(t + 1) * kv_dim;
            s.kv.store(self.window, pos0 + depth(t), &k[row.clone()], &v[row]);
        }

        out.iter_mut()
            .zip(&gate)
            .for_each(|(o, g)| *o *= sigmoid(*g));
        let mut y = vec![0.0; n * self.o.rows];
        matmul(&self.o, &out, &mut y);
        residual_conv(&y, &self.convs[2], self.taps, None, &mut s.convs[2], tree)
    }

    /// The feed-forward output of this layer after its convolution.
    pub fn ffn_conv(&self, y: &[f32], tree: Option<&Tree>, s: &mut RelState) -> Vec<f32> {
        residual_conv(y, &self.convs[3], self.taps, None, &mut s.convs[3], tree)
    }
}

/// `m` applied to every row of `x`.
fn project(m: &Matrix, x: &[f32]) -> Vec<f32> {
    let mut y = vec![0.0; x.len() / m.cols * m.rows];
    matmul(m, x, &mut y);
    y
}

/// Scales `x` down to a norm of at most `max`, then by `tau`.
fn clip(x: &mut [f32], max: f32, tau: f32) {
    let norm = x.iter().map(|v| v * v).sum::<f32>().sqrt() + 1e-6;
    let f = (max / norm).clamp(0.0, 1.0);
    x.iter_mut().for_each(|v| *v = *v * f * tau);
}

/// `(x + c) / sqrt(k + 1)` for every channel of every row of `x`, where `c`
/// is the causal convolution of the channel with its `taps` weights of `w`,
/// oldest input first, and `k` the number of taps. With `dynamic`, every
/// row also has `taps` weights per head, newest input first, that add to
/// `c` and count in `k` too. `window` holds the inputs before the batch.
fn residual_conv(
    x: &[f32],
    w: &[f32],
    taps: usize,
    dynamic: Option<&[f32]>,
    window: &mut ConvWindow,
    tree: Option<&Tree>,
) -> Vec<f32> {
    let c = w.len() / taps;
    let n = x.len() / c;
    let k = if dynamic.is_some() { 2 * taps } else { taps };
    let scale = 1.0 / ((k + 1) as f32).sqrt();
    let mut y = vec![0.0; x.len()];
    window.run(n, tree, |t, past| {
        let x = &x[t * c..][..c];
        let row = dynamic.map(|d| &d[t * (d.len() / n)..][..d.len() / n]);
        let head_len = row.map_or(c, |r| c / (r.len() / taps));
        for ch in 0..c {
            let past = past.past(ch);
            let w = &w[ch * taps..][..taps];
            let mut acc = dot(&w[..taps - 1], past) + w[taps - 1] * x[ch];
            if let Some(row) = row {
                let lags = &row[ch / head_len * taps..][..taps];
                acc += lags[0] * x[ch];
                for (lag, l) in lags.iter().enumerate().skip(1) {
                    acc += l * past[taps - 1 - lag];
                }
            }
            y[t * c + ch] = (x[ch] + acc) * scale;
        }
        x.to_vec()
    });
    y
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blocks::depth::tests::{bytes, f32_matrix, pattern};

    const D: usize = 8;
    const T: usize = 4;

    fn plus_one(v: Vec<f32>) -> Vec<f32> {
        v.into_iter().map(|v| v + 1.0).collect()
    }

    /// Weights as in the NumPy script the expected values come from.
    fn weights() -> Vec<Vec<u8>> {
        [
            pattern(64, 3, 7, 0.2),
            pattern(32, 5, 9, 0.15),
            pattern(32, 7, 11, 0.1),
            pattern(64, 5, 13, 0.1),
            pattern(64, 2, 5, 0.3),
            pattern(16, 3, 5, 0.2),
            pattern(32, 3, 11, 0.25),
            pattern(24, 5, 11, 0.1),
            pattern(24, 7, 13, 0.1),
        ]
        .iter()
        .map(|w| bytes(w))
        .collect()
    }

    fn attention(w: &[Vec<u8>], full: bool) -> RelAttention<'_> {
        let m = |i: usize, rows| f32_matrix(&w[i], rows, D);
        let extent = if full { 3 } else { 2 };
        RelAttention {
            norms: [
                plus_one(pattern(D, 1, 3, 0.25)),
                plus_one(pattern(D, 2, 5, 0.2)),
                plus_one(pattern(D, 3, 7, 0.1)),
            ],
            q: m(0, 8),
            k: m(1, 4),
            v: m(2, 4),
            o: m(3, 8),
            gate: m(4, 8),
            sink: m(5, 2),
            q_scale: plus_one(pattern(8, 1, 3, 0.5)),
            rel: m(6, 4),
            rel_scale: plus_one(pattern(4, 1, 3, 0.5)),
            rel_bias: pattern(2 * extent, 2, 7, 0.5),
            n_rel: 2,
            convs: [
                pattern(12, 5, 7, 0.2),
                pattern(12, 3, 5, 0.3),
                pattern(24, 7, 9, 0.1),
                pattern(24, 5, 11, 0.2),
            ],
            taps: 3,
            dynamic: full.then(|| [m(7, 3), m(8, 3)]),
            n_head: 2,
            n_head_kv: 1,
            head_dim: 4,
            window: (!full).then_some(2),
            qk_clip: 1.0,
            rel_clip: 0.5,
            log_scale: full.then_some((2.0, 0.5)),
            eps: 1e-6,
        }
    }

    fn inputs() -> [Vec<f32>; 3] {
        [
            pattern(T * D, 3, 17, 0.25),
            pattern(T * D, 5, 19, 0.2),
            pattern(T * D, 7, 23, 0.15),
        ]
    }

    /// Runs the inputs in batches of `sizes` rows on one state.
    fn run(a: &RelAttention, sizes: &[usize]) -> Vec<f32> {
        let x = inputs();
        let mut s = a.new_state();
        let mut out = Vec::new();
        let mut at = 0;
        for &n in sizes {
            let rows = |x: &Vec<f32>| x[at * D..(at + n) * D].to_vec();
            let [x0, x1, x2] = [rows(&x[0]), rows(&x[1]), rows(&x[2])];
            out.extend(a.forward([&x0, &x1, &x2], at, None, &mut s));
            at += n;
        }
        out
    }

    fn close(got: &[f32], want: &[f32]) {
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert!((g - w).abs() < 2e-5, "value {i}: {g} against {w}");
        }
    }

    // Expected values computed in NumPy (float64) from the graph of
    // llama.cpp's src/models/inkling2.cpp: clips, the log scale, the sink,
    // the bias gathered by distance and the four short convolutions.
    #[test]
    fn full_attention_matches_the_reference_in_any_batches() {
        let w = weights();
        let a = attention(&w, true);
        close(&run(&a, &[4]), &FULL);
        close(&run(&a, &[2, 1, 1]), &FULL);
    }

    #[test]
    fn windowed_attention_matches_the_reference_past_the_window() {
        let w = weights();
        let a = attention(&w, false);
        close(&run(&a, &[4]), &WINDOWED);
        close(&run(&a, &[1, 2, 1]), &WINDOWED);
    }

    #[test]
    fn feed_forward_convolution_matches_the_reference() {
        let w = weights();
        let a = attention(&w, true);
        let mut s = a.new_state();
        let x = &inputs()[0];
        let mut out = a.ffn_conv(&x[..D], None, &mut s);
        out.extend(a.ffn_conv(&x[D..], None, &mut s));
        close(&out, &FFN_CONV);
    }

    /// A row of a tree batch gets what its path gets alone, and keeping
    /// that path leaves the state the path alone leaves.
    #[test]
    fn tree_rows_and_rollback_match_their_path() {
        let w = weights();
        for full in [true, false] {
            let a = attention(&w, full);
            let x = inputs();
            let rows = |r: &[usize], j: usize| -> Vec<f32> {
                r.iter()
                    .flat_map(|&r| &x[j][r * D..][..D])
                    .copied()
                    .collect()
            };
            let input = |r: &[usize]| [rows(r, 0), rows(r, 1), rows(r, 2)];

            let mut alone = a.new_state();
            let [x0, x1, x2] = input(&[0, 2]);
            let want = a.forward([&x0, &x1, &x2], 0, None, &mut alone);
            let want_ffn = a.ffn_conv(&want, None, &mut alone);
            let [x0, x1, x2] = input(&[3]);
            let next = a.forward([&x0, &x1, &x2], 2, None, &mut alone);

            let mut s = a.new_state();
            s.checkpoint();
            let tree = Tree::new(&[None, Some(0), Some(0)]);
            let [x0, x1, x2] = input(&[0, 1, 2]);
            let got = a.forward([&x0, &x1, &x2], 0, Some(&tree), &mut s);
            close(&got[..D], &want[..D]);
            close(&got[2 * D..], &want[D..]);
            let ffn = a.ffn_conv(&got, Some(&tree), &mut s);
            close(&ffn[2 * D..], &want_ffn[D..]);
            a.rollback(&mut s, &[0, 2]);
            let [x0, x1, x2] = input(&[3]);
            close(&a.forward([&x0, &x1, &x2], 2, None, &mut s), &next);
        }
    }

    const FULL: [f32; 32] = [
        0.06097218,
        0.01768052,
        -0.02540816,
        0.07235946,
        0.08472995,
        0.0129058,
        -0.09258032,
        -0.1799743,
        0.03598015,
        0.001043498,
        0.01687951,
        0.1699478,
        -0.1350884,
        -0.05270002,
        -0.07823755,
        -0.004166865,
        -0.02864988,
        -0.08861532,
        -0.09491281,
        0.03972951,
        0.1734703,
        0.0522557,
        0.01972895,
        -0.04599692,
        -0.00618418,
        -0.1025415,
        -0.01840313,
        -0.02910783,
        -0.06701671,
        0.009015496,
        0.1234997,
        0.2312075,
    ];
    const WINDOWED: [f32; 32] = [
        0.06528252,
        0.01992776,
        -0.02561612,
        0.07361707,
        0.0858443,
        0.01205048,
        -0.08889826,
        -0.1778992,
        0.02333336,
        0.0159972,
        0.02791755,
        0.1948711,
        -0.1180045,
        -0.04058346,
        -0.09591377,
        -0.02450528,
        -0.03072583,
        -0.042867,
        -0.04434765,
        0.009184782,
        0.03756359,
        -0.0009589844,
        0.0115504,
        -0.00473943,
        -0.03958326,
        0.03833917,
        0.05249184,
        0.01219036,
        0.005725971,
        0.01911726,
        0.1455029,
        0.2113411,
    ];
    const FFN_CONV: [f32; 32] = [
        -2.0, -0.375, -0.35, 0.0, 0.4, 1.4, -0.175, -0.5, -0.25, -0.35, 1.025, 0.025, -0.1, -0.95,
        -0.35, 0.875, 2.5, -0.275, -1.4, 0.1, -0.65, 1.65, 0.7, -1.35, -0.875, -0.925, 1.1, -0.85,
        0.925, -1.7, 0.275, 0.95,
    ];
}
