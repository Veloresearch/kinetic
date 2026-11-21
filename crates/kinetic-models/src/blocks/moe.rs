//! Mixture of experts: a router picks a few expert feed-forward networks per
//! token from many, and their outputs are mixed by the router's weights.

mod kernels;

use std::cell::RefCell;

use kinetic_cpu::{matmul, pool, rms_norm, sigmoid, softmax, Q8K};
use kinetic_quant::{Matrix, TensorType};
use rayon::prelude::*;

use super::ACTIVATION_CHUNK;
#[cfg(all(test, feature = "cuda"))]
pub(crate) use kernels::scale_offsets;
use kernels::Unpacked;

/// Equally shaped expert matrices stored one after another, as in a 3-D
/// GGUF tensor.
#[derive(Clone, Copy, Debug)]
pub struct Experts<'a> {
    pub ty: TensorType,
    pub data: &'a [u8],
    pub rows: usize,
    pub cols: usize,
    /// Bytes from the start of one expert's matrix to the next.
    pub stride: usize,
}

impl<'a> Experts<'a> {
    /// Bytes of one expert's matrix.
    pub fn expert_bytes(&self) -> usize {
        self.rows * self.cols / self.ty.block_len() * self.ty.block_bytes()
    }

    pub fn expert(&self, e: usize) -> Matrix<'a> {
        let n = self.expert_bytes();
        Matrix::new(
            self.ty,
            &self.data[e * self.stride..][..n],
            self.rows,
            self.cols,
        )
    }

    /// The top and bottom halves of the rows of every expert, as the gate
    /// and up projections stored together in one tensor.
    pub fn halves(&self) -> (Experts<'a>, Experts<'a>) {
        let half = Experts {
            rows: self.rows / 2,
            ..*self
        };
        let bottom = Experts {
            data: &self.data[half.expert_bytes()..],
            ..half
        };
        (half, bottom)
    }
}

/// The routed part of a mixture-of-experts feed-forward layer, with the gate
/// of the layer's shared expert, which runs as a plain [`super::SwiGlu`].
#[derive(Clone)]
pub struct Moe<'a> {
    /// One row of logits weights per expert.
    pub router: Matrix<'a>,
    pub gate: Experts<'a>,
    pub up: Experts<'a>,
    pub down: Experts<'a>,
    /// Experts each token runs through.
    pub n_used: usize,
    /// Weights whose dot product with the input, through a sigmoid, scales
    /// the shared expert's output.
    pub shared_gate: Option<Vec<f32>>,
    pub gating: Gating,
    pub glu: Glu,
    /// Multiplies the routed weights after they are normalized.
    pub weight_scale: f32,
    /// Per-expert multipliers of the gate, up and down products.
    pub scales: [Option<Vec<f32>>; 3],
    /// Set when the experts and their router read the layer input through
    /// norms of their own.
    pub norms: Option<ExpertNorms>,
}

/// How an expert combines its gate and up projections.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum Glu {
    /// `silu(gate) * up`.
    #[default]
    Silu,
    /// `beta tanh(gate / beta) sigmoid(gate) * linear tanh(up / linear)`,
    /// with `up` left as is when `linear` is zero (Inkling 2).
    Situ { beta: f32, linear: f32 },
    /// `gelu(gate) * up` (Gemma 4).
    Gelu,
}

impl Glu {
    fn apply(self, gate: f32, up: f32) -> f32 {
        match self {
            Glu::Silu => kernels::silu(gate) * up,
            Glu::Situ { beta, linear } => {
                let up = if linear == 0.0 {
                    up
                } else {
                    linear * (up / linear).tanh()
                };
                beta * (gate / beta).tanh() * sigmoid(gate) * up
            }
            Glu::Gelu => super::gelu(gate) * up,
        }
    }
}

/// The norms of a layer whose routed and shared experts each normalize
/// their own input and output, and whose router reads the layer input
/// through a norm of its own (Gemma 4).
#[derive(Clone)]
pub struct ExpertNorms {
    pub input: Vec<f32>,
    /// Folds in the router's input scale.
    pub router: Vec<f32>,
    pub output: Vec<f32>,
    /// On the shared expert's output.
    pub shared: Vec<f32>,
}

/// How router logits become expert probabilities.
#[derive(Clone, Debug, Default, PartialEq)]
pub enum Gating {
    #[default]
    Softmax,
    /// A sigmoid per expert; the experts are chosen by probability plus a
    /// per-expert bias, which does not enter their weights (LFM2 MoE).
    Sigmoid { bias: Vec<f32> },
}

impl Gating {
    /// The `k` experts these router `logits` choose, largest first, weighted
    /// by their share of the `k` probabilities; `logits` become the
    /// probabilities in place.
    pub fn pick(&self, logits: &mut [f32], k: usize) -> Vec<Pick> {
        match self {
            Gating::Softmax => {
                softmax(logits);
                top_k(logits, None, k)
            }
            Gating::Sigmoid { bias } => {
                logits.iter_mut().for_each(|x| *x = sigmoid(*x));
                top_k(logits, Some(bias), k)
            }
        }
    }
}

/// One expert picked for a token, with its share of the output.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pick {
    pub expert: u32,
    pub weight: f32,
}

/// One expert to run on one input row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Job {
    pub row: u32,
    pub expert: u32,
}

impl Moe<'_> {
    pub fn n_experts(&self) -> usize {
        self.router.rows
    }

    /// The experts for each row of `h`, [`Moe::n_used`] per row in order of
    /// falling probability, with weights normalized to sum to one.
    pub fn route(&self, h: &[f32]) -> Vec<Pick> {
        let n_exp = self.n_experts();
        let mut probs = vec![0.0; h.len() / self.router.cols * n_exp];
        matmul(&self.router, h, &mut probs);
        probs
            .chunks_exact_mut(n_exp)
            .flat_map(|p| self.gating.pick(p, self.n_used))
            .map(|p| Pick {
                weight: p.weight * self.weight_scale,
                ..p
            })
            .collect()
    }

    /// Whether the block uses only what the GPU backends run: SiLU experts
    /// on the shared feed-forward input, with no scales or norms of their
    /// own.
    #[cfg(feature = "cuda")]
    pub fn plain(&self) -> bool {
        self.glu == Glu::Silu
            && self.weight_scale == 1.0
            && self.scales.iter().all(Option::is_none)
            && self.norms.is_none()
    }

    /// [`Moe::forward`] in a layer whose input is `x` and whose normalized
    /// feed-forward input is `h`; with [`Moe::norms`] the experts and the
    /// router read `x` through their own norms, and each branch's output is
    /// normalized before the two are summed into `shared`.
    pub fn forward_layer(&self, x: &[f32], h: &[f32], shared: &mut [f32], eps: f32) {
        let Some(n) = &self.norms else {
            return self.forward(h, shared);
        };
        let d = self.router.cols;
        let norm = |y: &mut [f32], w: &[f32]| {
            pool::for_each_chunk(y, d, |_, r| rms_norm(r, w, eps));
        };
        let normed = |w: &[f32]| {
            let mut y = x.to_vec();
            norm(&mut y, w);
            y
        };
        let mut routed = vec![0.0; x.len()];
        let picks = self.route(&normed(&n.router));
        self.add_routed(&normed(&n.input), &picks, &mut routed);
        norm(&mut routed, &n.output);
        norm(shared, &n.shared);
        shared.iter_mut().zip(&routed).for_each(|(s, r)| *s += r);
    }

    /// Adds the routed experts' output for `h` to `shared`, the shared
    /// expert's output for the same rows, after scaling it by its gate.
    pub fn forward(&self, h: &[f32], shared: &mut [f32]) {
        self.scale_shared(h, shared);
        self.add_routed(h, &self.route(h), shared);
    }

    /// Adds to `out` the output of the experts `picks` choose for each row
    /// of `h`, by their weights.
    pub fn add_routed(&self, h: &[f32], picks: &[Pick], out: &mut [f32]) {
        let d = self.router.cols;
        let k = self.n_used;
        let jobs: Vec<Job> = picks
            .iter()
            .enumerate()
            .map(|(s, p)| Job {
                row: (s / k) as u32,
                expert: p.expert,
            })
            .collect();
        let routed = self.run_experts(h, &jobs);
        for (t, out) in out.chunks_exact_mut(d).enumerate() {
            for (slot, pick) in picks[t * k..][..k].iter().enumerate() {
                let y = &routed[(t * k + slot) * d..][..d];
                out.iter_mut()
                    .zip(y)
                    .for_each(|(o, y)| *o += pick.weight * y);
            }
        }
    }

    /// Reads every page of the experts' weights, in parallel. Tokens touch
    /// a few experts each, so otherwise every page of the file is faulted in
    /// on its own while tokens wait, even when the file is in the system's
    /// cache.
    pub fn page_in(&self) {
        let touch = |data: &[u8]| {
            data.par_chunks(1 << 20)
                .map(|c| c.iter().step_by(4096).fold(0u8, |a, &b| a ^ b))
                .reduce(|| 0, |a, b| a ^ b)
        };
        for e in [&self.gate, &self.up, &self.down] {
            std::hint::black_box(touch(e.data));
        }
    }

    /// Multiplies each row of the shared expert's output by its gate.
    pub fn scale_shared(&self, h: &[f32], shared: &mut [f32]) {
        let Some(w) = &self.shared_gate else {
            return;
        };
        let d = self.router.cols;
        for (h, out) in h.chunks_exact(d).zip(shared.chunks_exact_mut(d)) {
            let g = sigmoid(kinetic_cpu::dot(w, h));
            out.iter_mut().for_each(|v| *v *= g);
        }
    }

    /// Runs every job's expert on its row of `h` and returns one output row
    /// per job, in order. Each expert's weights are read once for all its
    /// jobs.
    pub fn run_experts(&self, h: &[f32], jobs: &[Job]) -> Vec<f32> {
        let groups = group_by_expert(jobs.iter().map(|j| j.expert), self.n_experts());
        let xq: Vec<Q8K> = if kernels::supported(self.gate.ty) || kernels::supported(self.up.ty) {
            let d = self.router.cols;
            pool::map(h.len() / d, |t| Q8K::quantize(&h[t * d..][..d]))
        } else {
            Vec::new()
        };
        let input = Input {
            x: h,
            xq: &xq,
            jobs: Some(jobs),
        };
        let [mut act, mut up] = products([&self.gate, &self.up], &groups, &input, jobs.len());
        self.scale(0, &mut act, jobs);
        self.scale(1, &mut up, jobs);
        pool::for_each_chunk(&mut act, ACTIVATION_CHUNK, |i, g| {
            g.iter_mut()
                .zip(&up[i * ACTIVATION_CHUNK..])
                .for_each(|(g, u)| *g = self.glu.apply(*g, *u))
        });

        let actq: Vec<Q8K> = if kernels::supported(self.down.ty) {
            let d = self.down.cols;
            pool::map(act.len() / d, |t| Q8K::quantize(&act[t * d..][..d]))
        } else {
            Vec::new()
        };
        let input = Input {
            x: &act,
            xq: &actq,
            jobs: None,
        };
        let [mut out] = products([&self.down], &groups, &input, jobs.len());
        self.scale(2, &mut out, jobs);
        out
    }

    /// Multiplies each job's row of `y` by its expert's scale of product
    /// `i` (gate, up, down), if the file has those scales.
    fn scale(&self, i: usize, y: &mut [f32], jobs: &[Job]) {
        let Some(scales) = &self.scales[i] else {
            return;
        };
        let width = y.len() / jobs.len();
        for (row, job) in y.chunks_exact_mut(width).zip(jobs) {
            let s = scales[job.expert as usize];
            row.iter_mut().for_each(|v| *v *= s);
        }
    }
}

/// The input rows of one expert product.
struct Input<'a> {
    x: &'a [f32],
    /// `x` quantized row by row, for the codebook kernels.
    xq: &'a [Q8K],
    /// The job of every slot, whose row it reads; without them slot `s`
    /// reads row `s`.
    jobs: Option<&'a [Job]>,
}

impl Input<'_> {
    fn row_of(&self, slot: usize) -> usize {
        self.jobs.map_or(slot, |j| j[slot].row as usize)
    }
}

thread_local! {
    /// The unpacked weight row each thread reuses.
    static ROW: RefCell<Vec<Unpacked>> = RefCell::default();
}

/// Weight rows one parallel task computes for every pick of one expert.
const ROWS_PER_TASK: usize = 64;

/// `W_e x` for every pick routed to expert `e` by `groups`, as `slots` rows
/// of `m.rows` values in slot order, for each of the equally shaped `ms`.
/// One parallel task computes the same rows of all of them.
fn products<const N: usize>(
    ms: [&Experts; N],
    groups: &[(usize, Vec<usize>)],
    x: &Input,
    slots: usize,
) -> [Vec<f32>; N] {
    let m = ms[0];
    if !ms.iter().all(|m| kernels::supported(m.ty)) {
        return ms.map(|m| unpacked_product(m, groups, x, slots));
    }
    let tasks: Vec<(usize, &[usize], usize)> = groups
        .iter()
        .flat_map(|(e, picks)| {
            (0..m.rows)
                .step_by(ROWS_PER_TASK)
                .map(move |r| (*e, &picks[..], r))
        })
        .collect();
    let blocks = m.cols / m.ty.block_len();
    let done: Vec<[Vec<f32>; N]> = pool::map(tasks.len(), |i| {
        let (e, picks, r0) = tasks[i];
        let rows = r0..m.rows.min(r0 + ROWS_PER_TASK);
        ROW.with_borrow_mut(|row| {
            row.resize(blocks, Unpacked::default());
            ms.map(|m| {
                let w = m.expert(e);
                let mut out = Vec::with_capacity(rows.len() * picks.len());
                for r in rows.clone() {
                    if let [s] = picks {
                        out.push(kernels::packed_dot(m.ty, w.row(r), &x.xq[x.row_of(*s)]));
                        continue;
                    }
                    kernels::unpack(m.ty, w.row(r), row);
                    out.extend(picks.iter().map(|&s| kernels::dot(row, &x.xq[x.row_of(s)])));
                }
                out
            })
        })
    });
    std::array::from_fn(|i| {
        let mut y = vec![0.0; slots * m.rows];
        for (&(_, picks, r0), out) in tasks.iter().zip(&done) {
            for (k, values) in out[i].chunks_exact(picks.len()).enumerate() {
                for (&s, &v) in picks.iter().zip(values) {
                    y[s * m.rows + r0 + k] = v;
                }
            }
        }
        y
    })
}

/// [`products`] for weights the codebook kernels do not read.
fn unpacked_product(
    m: &Experts,
    groups: &[(usize, Vec<usize>)],
    x: &Input,
    slots: usize,
) -> Vec<f32> {
    let mut y = vec![0.0; slots * m.rows];
    for (e, picks) in groups {
        let rows: Vec<f32> = picks
            .iter()
            .flat_map(|&s| &x.x[x.row_of(s) * m.cols..][..m.cols])
            .copied()
            .collect();
        let mut out = vec![0.0; picks.len() * m.rows];
        matmul(&m.expert(*e), &rows, &mut out);
        for (&s, out) in picks.iter().zip(out.chunks_exact(m.rows)) {
            y[s * m.rows..][..m.rows].copy_from_slice(out);
        }
    }
    y
}

/// The `k` largest of `probs` plus `bias`, largest first, weighted by their
/// share of the sum of the `k` probabilities.
fn top_k(probs: &[f32], bias: Option<&[f32]>, k: usize) -> Vec<Pick> {
    let score = |e: u32| probs[e as usize] + bias.map_or(0.0, |b| b[e as usize]);
    let mut order: Vec<u32> = (0..probs.len() as u32).collect();
    order.select_nth_unstable_by(k - 1, |&a, &b| {
        score(b).total_cmp(&score(a)).then(a.cmp(&b))
    });
    order.truncate(k);
    order.sort_by(|&a, &b| score(b).total_cmp(&score(a)).then(a.cmp(&b)));
    let mut picks: Vec<Pick> = order
        .iter()
        .map(|&e| Pick {
            expert: e,
            weight: probs[e as usize],
        })
        .collect();
    // The floor is the smallest normal f16, as in the reference.
    let sum = picks
        .iter()
        .map(|p| p.weight)
        .sum::<f32>()
        .max(6.103_515_6e-5);
    picks.iter_mut().for_each(|p| p.weight /= sum);
    picks
}

/// The positions in `experts` of each expert that appears, by expert.
fn group_by_expert(
    experts: impl Iterator<Item = u32>,
    n_experts: usize,
) -> Vec<(usize, Vec<usize>)> {
    let mut slots = vec![Vec::new(); n_experts];
    for (s, e) in experts.enumerate() {
        slots[e as usize].push(s);
    }
    slots
        .into_iter()
        .enumerate()
        .filter(|(_, s)| !s.is_empty())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn top_k_picks_the_largest_and_normalizes() {
        let probs = [0.1, 0.4, 0.05, 0.3, 0.15];
        let picks = top_k(&probs, None, 2);
        assert_eq!(picks.iter().map(|p| p.expert).collect::<Vec<_>>(), [1, 3]);
        assert!((picks[0].weight - 0.4 / 0.7).abs() < 1e-6);
        assert!((picks[1].weight - 0.3 / 0.7).abs() < 1e-6);
    }

    #[test]
    fn sigmoid_gating_chooses_with_the_bias_and_weights_without() {
        let gating = Gating::Sigmoid {
            bias: vec![0.0, 0.0, 1.0],
        };
        let picks = gating.pick(&mut [2.0, 1.0, 0.0], 2);
        assert_eq!(picks.iter().map(|p| p.expert).collect::<Vec<_>>(), [2, 0]);
        let (a, b) = (sigmoid(0.0), sigmoid(2.0));
        assert!((picks[0].weight - a / (a + b)).abs() < 1e-6);
    }

    // Expected values computed in NumPy (float64) from llama.cpp's
    // LLM_FFN_SITU with beta 4 and linear beta 25, as in Inkling 2.
    #[test]
    fn situ_matches_the_reference() {
        let glu = Glu::Situ {
            beta: 4.0,
            linear: 25.0,
        };
        let want = [0.6179198, -2.511176, -5.410969];
        for ((g, u), w) in [(0.5, 2.0), (-3.0, 30.0), (6.0, -1.5)]
            .into_iter()
            .zip(want)
        {
            assert!((glu.apply(g, u) - w).abs() < 1e-5);
        }
    }

    /// Per-expert scales give what weights multiplied by them give, as
    /// llama.cpp's `build_lora_mm_id` scales each expert's product.
    #[test]
    fn expert_scales_multiply_each_product() {
        let bytes = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let pattern = |n: usize, k: usize| -> Vec<f32> {
            (0..n).map(|i| ((i * k) % 7) as f32 * 0.3 - 0.9).collect()
        };
        let (gate, up, down) = (pattern(12, 3), pattern(12, 5), pattern(12, 2));
        let scales = [[0.5, 2.0, -1.5], [3.0, 0.25, 1.0], [-2.0, 0.5, 4.0]];
        let scaled = |w: &[f32], s: &[f32; 3]| -> Vec<f32> {
            w.chunks(4)
                .zip(s)
                .flat_map(|(e, s)| e.iter().map(move |w| w * s))
                .collect()
        };
        let router = bytes(&pattern(6, 4));
        let plain = [bytes(&gate), bytes(&up), bytes(&down)];
        let folded = [
            bytes(&scaled(&gate, &scales[0])),
            bytes(&scaled(&up, &scales[1])),
            bytes(&scaled(&down, &scales[2])),
        ];
        let sets = [plain, folded];
        let experts = |data| Experts {
            ty: TensorType::F32,
            data,
            rows: 2,
            cols: 2,
            stride: 16,
        };
        let moe = |k: usize, scales: [Option<Vec<f32>>; 3]| Moe {
            router: Matrix::new(TensorType::F32, &router, 3, 2),
            gate: experts(&sets[k][0]),
            up: experts(&sets[k][1]),
            down: experts(&sets[k][2]),
            n_used: 2,
            shared_gate: None,
            gating: Gating::Softmax,
            glu: Glu::Gelu,
            weight_scale: 1.0,
            scales,
            norms: None,
        };
        let h = [0.7, -1.2, 0.3, 0.9];
        let (mut got, mut want) = ([0.0; 4], [0.0; 4]);
        moe(0, scales.map(|s| Some(s.to_vec()))).forward(&h, &mut got);
        moe(1, Default::default()).forward(&h, &mut want);
        for (g, w) in got.iter().zip(want) {
            assert!((g - w).abs() < 1e-6, "{got:?} against {want:?}");
        }
    }

    #[test]
    fn halves_split_each_expert_into_gate_and_up() {
        let values: Vec<f32> = (0..16).map(|v| v as f32).collect();
        let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let fused = Experts {
            ty: TensorType::F32,
            data: &data,
            rows: 4,
            cols: 2,
            stride: 32,
        };
        let (gate, up) = fused.halves();
        let row = |m: Matrix, r: usize| m.row(r).to_vec();
        assert_eq!(row(gate.expert(1), 1), data[40..48]);
        assert_eq!(row(up.expert(1), 0), data[48..56]);
        assert_eq!(row(up.expert(0), 1), data[24..32]);
    }

    #[test]
    fn groups_keep_slot_order_per_expert() {
        let groups = group_by_expert([2, 0, 2, 1].into_iter(), 4);
        assert_eq!(groups, [(0, vec![1]), (1, vec![3]), (2, vec![0, 2])]);
    }
}
