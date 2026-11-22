//! DSpark, the block drafter Inkling 2 ships: a few layers that read hidden
//! states of the target at some of its layer visits and propose the tokens
//! after an anchor token in one pass, then a recurrent head that chains
//! those positions into a sequence through the target's own output head.
//!
//! Every committed position of the target leaves, in each drafter layer,
//! keys and values projected from a softmax-weighted mix of its hidden
//! states. A draft runs a block of the anchor's normalized target embedding
//! followed by mask embeddings; every row of the block sees the committed
//! entries, the windowed layers only those within the window before the
//! anchor, and every other row of the block.

use kinetic_cpu::{matmul, matmul_many, pool, rms_norm, sigmoid};
use kinetic_gguf::TensorSource;
use kinetic_quant::{dequantize_row, Matrix};

use crate::blocks::{Attention, Gate, KvCache, Mixer, Rope, RopeLayout, SharedKv};
use crate::loader::Loader;
use crate::model::Layer;
use crate::LoadError;

/// Names one file dialect gives the drafter's own keys, under the
/// architecture it declares.
struct Dialect {
    arch: &'static str,
    block_size: &'static str,
    target_layers: &'static str,
    markov_rank: &'static str,
}

const DIALECTS: &[Dialect] = &[
    // Unsloth's conversion for llama.cpp
    Dialect {
        arch: "dflash",
        block_size: "block_size",
        target_layers: "target_layers",
        markov_rank: "dspark.markov_rank",
    },
];

/// Keys the drafter has no use for: its context is the target's.
const SKIPPED: &[&str] = &["context_length"];

/// A DSpark drafter with weights borrowed from its file.
pub struct Dspark<'a> {
    d: usize,
    eps: f32,
    layers: Vec<Layer<'a>>,
    contexts: Vec<Context<'a>>,
    /// Per drafter layer, then for the head's context: the softmax weights
    /// of the target's hidden states, in the order of `target_layers`.
    mixes: Vec<Vec<f32>>,
    target_layers: Vec<usize>,
    /// Rows of a draft block: the anchor and the masks after it.
    block: usize,
    mask: Vec<f32>,
    output_norm: Vec<f32>,
    context_norm: Vec<f32>,
    rank: usize,
    /// One row of `rank` values per token.
    markov: Matrix<'a>,
    joint: Matrix<'a>,
    joint_bias: Vec<f32>,
    correction: Matrix<'a>,
    correction_bias: Vec<f32>,
    /// Tokens the target scores.
    vocab: usize,
    cap: f32,
}

/// How a drafter layer turns the target's hidden states at a position into
/// the keys and values its block rows attend to.
struct Context<'a> {
    norm: Vec<f32>,
    k: Matrix<'a>,
    v: Matrix<'a>,
    k_norm: Vec<f32>,
}

impl<'a> Dspark<'a> {
    /// Reads the drafter in `file`. Fails on any tensor or key of the file
    /// it does not use.
    pub fn load(file: &'a dyn TensorSource) -> Result<Self, LoadError> {
        let arch = file.header().architecture().unwrap_or_default();
        let names = DIALECTS
            .iter()
            .find(|n| n.arch == arch && file.header().arch_key(n.markov_rank).is_some())
            .ok_or_else(|| LoadError(format!("{arch} is not a DSpark drafter")))?;
        let l = Loader::new(file);
        let flag = |key: &str| -> Result<bool, LoadError> {
            l.hparam(key)?
                .as_bool()
                .ok_or_else(|| LoadError(format!("{key} is not a boolean")))
        };
        if flag("sample_from_anchor")? || flag("has_confidence_head")? || flag("attention.causal")?
        {
            return Err(LoadError(
                "only DSpark drafters with a noncausal block after the anchor and no confidence head are supported yet"
                    .into(),
            ));
        }
        let d = l.usize("embedding_length")?;
        let eps = l.f32("attention.layer_norm_rms_epsilon")?;
        let n_layers = l.usize("block_count")?;
        let (nh, kv, hd) = (
            l.usize("attention.head_count")?,
            l.usize("attention.head_count_kv")?,
            l.usize("attention.key_length")?,
        );
        let n_ff = l.usize("feed_forward_length")?;
        let window = l.usize("attention.sliding_window")?;
        let windowed = l.array("attention.sliding_window_pattern", n_layers, |v| {
            v.as_bool()
        })?;
        let (rope_dims, rope_base) = (l.usize("rope.dimension_count")?, l.f32("rope.freq_base")?);
        let block = l.usize(names.block_size)?;
        let rank = l.usize(names.markov_rank)?;
        let padded = l.usize("vocab_size")?;
        let vocab = l.usize("vocab_size_unpadded")?;
        let cap = l.f32("final_logit_softcapping")?;
        let key = names.target_layers;
        let n_targets = l.hparam(key)?.as_array().map_or(0, |a| a.len());
        let target_layers = l.array(key, n_targets, |v| v.as_u64().map(|v| v as usize))?;
        if !(eps.is_finite() && eps > 0.0 && cap.is_finite() && cap > 0.0)
            || n_layers == 0
            || kv == 0
            || !nh.is_multiple_of(kv)
            || l.usize("attention.value_length")? != hd
            || rope_dims != hd
            || window == 0
            || block < 2
            || rank == 0
            || rank > d
            || vocab == 0
            || vocab > padded
            || target_layers.len() < 2
            || target_layers.windows(2).any(|w| w[0] >= w[1])
        {
            return Err(LoadError("DSpark hyperparameters are out of range".into()));
        }

        let mut layers = Vec::with_capacity(n_layers);
        let mut contexts = Vec::with_capacity(n_layers);
        for (i, &windowed) in windowed.iter().enumerate() {
            let t = |s: &str| format!("blk.{i}.{s}");
            let attention = Attention {
                q: l.matrix(&t("attn_q.weight"), d, nh * hd)?,
                k: l.matrix(&t("attn_k.weight"), d, kv * hd)?,
                v: l.matrix(&t("attn_v.weight"), d, kv * hd)?,
                o: l.matrix(&t("attn_output.weight"), nh * hd, d)?,
                gate: Gate::None,
                q_bias: None,
                k_bias: None,
                v_bias: None,
                q_norm: Some(l.vector(&t("attn_q_norm.weight"), hd)?),
                k_norm: Some(l.vector(&t("attn_k_norm.weight"), hd)?),
                norm_after_rope: false,
                n_head: nh,
                n_head_kv: kv,
                head_dim: hd,
                rope: Rope {
                    dims: rope_dims,
                    base: rope_base,
                    factors: None,
                    layout: RopeLayout::Neox,
                    mscale: 1.0,
                },
                window: windowed.then_some(window),
                scale: None,
                v_norm: None,
                shared_kv: SharedKv::No,
                eps,
                folding: None,
            };
            layers.push(Layer {
                mixer_norm: Some(l.vector(&t("attn_norm.weight"), d)?),
                mixer: Mixer::Attention(attention),
                mixer_out_norm: None,
                ffn_norm: Some(l.vector(&t("ffn_norm.weight"), d)?),
                ffn: Some(l.swiglu(i, d, n_ff)?),
                experts: None,
                ffn_out_norm: None,
                residual_scale: None,
                output_scale: None,
            });
            contexts.push(Context {
                norm: l.vector(&t("dflare_norm.weight"), d)?,
                k: l.matrix(&t("attn_k_ctx.weight"), d, kv * hd)?,
                v: l.matrix(&t("attn_v_ctx.weight"), d, kv * hd)?,
                k_norm: l.vector(&t("attn_k_ctx_norm.weight"), hd)?,
            });
        }
        let n = target_layers.len();
        let fuse = l.vector("dspark_fuse.weight", n * (n_layers + 1))?;
        let mixes = fuse
            .chunks_exact(n)
            .map(|row| {
                let mut w = row.to_vec();
                kinetic_cpu::softmax(&mut w);
                w
            })
            .collect();
        let dspark = Dspark {
            d,
            eps,
            layers,
            contexts,
            mixes,
            target_layers,
            block,
            mask: l.vector("mask_embd.weight", d)?,
            output_norm: l.vector("output_norm.weight", d)?,
            context_norm: l.vector("dspark_norm.weight", d)?,
            rank,
            markov: l.matrix("markov_w1.weight", rank, padded)?,
            joint: l.matrix("joint_proj.weight", 2 * rank + 2 * d, 3 * rank)?,
            joint_bias: l.vector("joint_proj.bias", 3 * rank)?,
            correction: l.matrix("dspark_hidden_out.weight", rank, d)?,
            correction_bias: l.vector("dspark_hidden_out.bias", d)?,
            vocab,
            cap,
        };
        l.check_all_read(SKIPPED)?;
        l.refuse_weight_scales()?;
        Ok(dspark)
    }

    /// The target's layer visits whose input hidden states the drafter
    /// reads, in the order it mixes them; the number of visits stands for
    /// the target's normalized final hidden state.
    pub fn target_layers(&self) -> &[usize] {
        &self.target_layers
    }

    /// Most tokens one draft proposes.
    pub fn max_draft(&self) -> usize {
        self.block - 1
    }

    /// The width of the target's hidden states the drafter reads.
    pub fn n_embd(&self) -> usize {
        self.d
    }

    /// Tokens of the target's vocabulary the drafter scores.
    pub fn vocab(&self) -> usize {
        self.vocab
    }

    pub fn new_state(&self) -> DsparkState {
        DsparkState {
            caches: vec![KvCache::default(); self.layers.len()],
            len: 0,
            context: Vec::new(),
        }
    }

    /// Commits the positions after those `state` holds. `features` holds,
    /// for each position, the target's hidden states at
    /// [`Dspark::target_layers`], in that order.
    pub fn commit(&self, state: &mut DsparkState, features: &[f32]) {
        let width = self.d * self.target_layers.len();
        assert!(
            !features.is_empty() && features.len().is_multiple_of(width),
            "features are not whole positions"
        );
        for (i, (layer, context)) in self.layers.iter().zip(&self.contexts).enumerate() {
            let x = self.mix(features, &self.mixes[i], &context.norm);
            let rows = x.len() / self.d;
            let kv_dim = context.k.rows;
            let (mut k, mut v) = (vec![0.0; rows * kv_dim], vec![0.0; rows * kv_dim]);
            matmul_many(&[&context.k, &context.v], &x, &mut [&mut k, &mut v]);
            attention(layer).store_keys(k, &v, &context.k_norm, state.len, &mut state.caches[i]);
        }
        let last = &features[features.len() - width..];
        state.context = self.mix(last, &self.mixes[self.layers.len()], &self.context_norm);
        state.len += features.len() / width;
    }

    /// Up to `n` tokens after the anchor, the token at the position after
    /// those `state` holds, each with the chance the drafter gives it after
    /// the ones before. `anchor` is the target's normalized embedding of the
    /// anchor token, and `head` the target's logits for a hidden state.
    /// Ties go to the lower token.
    pub fn draft(
        &self,
        state: &DsparkState,
        anchor_token: u32,
        anchor: &[f32],
        n: usize,
        head: impl Fn(&[f32]) -> Vec<f32>,
    ) -> Vec<(u32, f32)> {
        assert!(!state.context.is_empty(), "nothing committed");
        let hidden = self.backbone(state, anchor);
        let mut recurrent = vec![0.0; self.rank];
        let mut prev = anchor_token;
        let mut draft = Vec::with_capacity(n);
        for row in hidden.chunks_exact(self.d).skip(1).take(n) {
            let h = self.step(prev, &mut recurrent, row, &state.context);
            let mut logits = head(&h);
            logits
                .iter_mut()
                .for_each(|l| *l = self.cap * (*l / self.cap).tanh());
            prev = crate::speculative::argmax(&logits);
            let best = logits[prev as usize];
            let total: f32 = logits.iter().map(|l| (l - best).exp()).sum();
            draft.push((prev, 1.0 / total));
        }
        draft
    }

    /// The softmax-weighted sum of each position's hidden states in
    /// `features`, normalized with `norm`.
    fn mix(&self, features: &[f32], weights: &[f32], norm: &[f32]) -> Vec<f32> {
        let d = self.d;
        let width = d * weights.len();
        let mut x = vec![0.0; features.len() / weights.len()];
        pool::for_each_chunk(&mut x, d, |t, row| {
            for (w, f) in weights
                .iter()
                .zip(features[t * width..][..width].chunks_exact(d))
            {
                row.iter_mut().zip(f).for_each(|(x, f)| *x += w * f);
            }
            rms_norm(row, norm, self.eps);
        });
        x
    }

    /// The drafter's normalized output for every row of a block after the
    /// anchor whose normalized embedding is `anchor`.
    fn backbone(&self, state: &DsparkState, anchor: &[f32]) -> Vec<f32> {
        let d = self.d;
        let masks = self.mask.iter().cycle().take((self.block - 1) * d);
        let mut x: Vec<f32> = anchor.iter().chain(masks).copied().collect();
        for (layer, cache) in self.layers.iter().zip(&state.caches) {
            let attention = attention(layer);
            layer.forward_with(&mut x, self.eps, |h| {
                attention.forward_block(h, state.len, cache)
            });
        }
        pool::for_each_chunk(&mut x, d, |_, row| {
            rms_norm(row, &self.output_norm, self.eps)
        });
        x
    }

    /// One step of the recurrent head on the block row `hidden`, after the
    /// token `prev`: updates `recurrent` and returns the hidden state whose
    /// logits under the target's output head score the row's token.
    fn step(&self, prev: u32, recurrent: &mut [f32], hidden: &[f32], context: &[f32]) -> Vec<f32> {
        let r = self.rank;
        let mut markov = vec![0.0; r];
        dequantize_row(self.markov.ty, self.markov.data, prev as usize, &mut markov)
            .expect("markov table type");
        let input = [&recurrent[..], &markov, hidden, context].concat();
        let mut joint = vec![0.0; 3 * r];
        matmul(&self.joint, &input, &mut joint);
        joint
            .iter_mut()
            .zip(&self.joint_bias)
            .for_each(|(j, b)| *j += b);
        let (gates, rest) = joint.split_at(r);
        let (candidates, latent) = rest.split_at(r);
        for ((s, &g), &c) in recurrent.iter_mut().zip(gates).zip(candidates) {
            let c = c.tanh();
            *s = c + sigmoid(g) * (*s - c);
        }
        let latent: Vec<f32> = latent.iter().map(|v| v.tanh()).collect();
        let mut out = vec![0.0; self.d];
        matmul(&self.correction, &latent, &mut out);
        for ((o, b), h) in out.iter_mut().zip(&self.correction_bias).zip(hidden) {
            *o += b + h;
        }
        out
    }
}

/// What a drafter keeps of one sequence: per layer the keys and values of
/// the committed positions, only the window's latest in windowed layers,
/// and the head's context at the latest position.
#[derive(Clone)]
pub struct DsparkState {
    caches: Vec<KvCache>,
    len: usize,
    context: Vec<f32>,
}

impl DsparkState {
    /// Positions committed so far.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

fn attention<'l, 'a>(layer: &'l Layer<'a>) -> &'l Attention<'a> {
    match &layer.mixer {
        Mixer::Attention(a) => a,
        _ => unreachable!("DSpark layers attend"),
    }
}
