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

use kinetic_gguf::TensorSource;
use kinetic_quant::Matrix;

use crate::blocks::{Attention, Gate, Mixer, Rope, RopeLayout, SharedKv};
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
// shortcut: the fields are read by the forward pass of a later change
#[allow(dead_code)]
pub struct Dspark<'a> {
    pub(crate) d: usize,
    pub(crate) eps: f32,
    pub(crate) layers: Vec<Layer<'a>>,
    pub(crate) contexts: Vec<Context<'a>>,
    /// Per drafter layer, then for the head's context: the softmax weights
    /// of the target's hidden states, in the order of `target_layers`.
    pub(crate) mixes: Vec<Vec<f32>>,
    pub(crate) target_layers: Vec<usize>,
    /// Rows of a draft block: the anchor and the masks after it.
    pub(crate) block: usize,
    pub(crate) mask: Vec<f32>,
    pub(crate) output_norm: Vec<f32>,
    pub(crate) context_norm: Vec<f32>,
    pub(crate) rank: usize,
    /// One row of `rank` values per token.
    pub(crate) markov: Matrix<'a>,
    pub(crate) joint: Matrix<'a>,
    pub(crate) joint_bias: Vec<f32>,
    pub(crate) correction: Matrix<'a>,
    pub(crate) correction_bias: Vec<f32>,
    /// Tokens the target scores.
    pub(crate) vocab: usize,
    pub(crate) cap: f32,
}

/// How a drafter layer turns the target's hidden states at a position into
/// the keys and values its block rows attend to.
// shortcut: read by the forward pass of a later change
#[allow(dead_code)]
pub(crate) struct Context<'a> {
    pub norm: Vec<f32>,
    pub k: Matrix<'a>,
    pub v: Matrix<'a>,
    pub k_norm: Vec<f32>,
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
}
