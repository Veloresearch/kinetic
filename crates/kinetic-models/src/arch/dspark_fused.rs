//! DSpark drafters as llama.cpp loads them from a separate file beside the
//! target (`--spec-type draft-dspark`), such as the `*-DSpark-GGUF` files
//! for LFM3, Llama 3.2, Qwen3, Qwen3.5, Gemma 4 and Phi-4: a DFlash
//! backbone whose input is the target's hidden states fused by one linear
//! layer, followed by a Markov head and an optional confidence head.
//!
//! Every committed position of the target leaves, in each drafter layer,
//! keys and values projected from `fc` of its hidden states at the target
//! layers, concatenated. A draft runs a block of the anchor's embedding
//! followed by mask embeddings at the positions after those committed;
//! every row sees the committed entries (the windowed layers only the
//! latest ones) and every other row. Each row's logits, from the drafter's
//! own output head when it has one (mapped to the target's tokens through
//! `d2t`), are biased by the Markov head with the token chosen at the row
//! before, so the rows chain into a sequence.
//!
//! The reference is llama.cpp's `src/models/dflash.cpp` (`build_inject`,
//! `build_layers`, `build_logits`), `src/models/dspark.cpp`
//! (`build_dspark_markov_head`) and `common/speculative.cpp`
//! (`common_speculative_impl_draft_dspark`).

use kinetic_cpu::{matmul, matmul_many, pool, rms_norm, sigmoid};
use kinetic_gguf::TensorSource;
use kinetic_quant::{dequantize_row, Matrix};

use crate::blocks::{Attention, Gate, KvCache, Mixer, Rope, RopeLayout, SharedKv};
use crate::loader::Loader;
use crate::model::Layer;
use crate::LoadError;

/// Keys the drafter has no use for: its context is the target's.
const SKIPPED: &[&str] = &["context_length"];

/// A DSpark drafter whose target features pass through one fusion layer,
/// with weights borrowed from its file.
pub struct FusedDspark<'a> {
    pub(crate) d: usize,
    pub(crate) eps: f32,
    pub(crate) layers: Vec<Layer<'a>>,
    pub(crate) target_layers: Vec<usize>,
    /// Concatenated target features to the drafter's width.
    pub(crate) fc: Matrix<'a>,
    pub(crate) fc_norm: Vec<f32>,
    pub(crate) output_norm: Vec<f32>,
    /// The drafter's own output head over its reduced vocabulary, if any;
    /// without one it scores with the target's head.
    pub(crate) output: Option<Matrix<'a>>,
    /// The target token of each row of the drafter's vocabulary.
    pub(crate) d2t: Option<Vec<u32>>,
    /// One row of `rank` values per target token.
    pub(crate) markov_in: Matrix<'a>,
    /// From the Markov state to a bias on each token of the drafter's
    /// vocabulary.
    pub(crate) markov_out: Matrix<'a>,
    pub(crate) rank: usize,
    /// Weights over the row's hidden state and its Markov state, and bias.
    pub(crate) confidence: Option<(Vec<f32>, f32)>,
    /// Rows of the largest block it was trained on.
    pub(crate) block: usize,
    /// Whether the anchor's row predicts the first token; otherwise the
    /// first mask row does.
    pub(crate) from_anchor: bool,
    pub(crate) mask_token: u32,
    /// Tokens of the target's vocabulary.
    pub(crate) vocab: usize,
}

impl<'a> FusedDspark<'a> {
    /// Whether `file` holds a drafter of this kind.
    pub fn detect(file: &dyn TensorSource) -> bool {
        let h = file.header();
        h.architecture() == Some("dflash")
            && h.tensor("fc.weight").is_some()
            && h.tensor("markov_w2.weight").is_some()
    }

    /// Reads the drafter in `file`. Fails on any tensor or key of the file
    /// it does not use.
    pub fn load(file: &'a dyn TensorSource) -> Result<Self, LoadError> {
        if !Self::detect(file) {
            return Err(LoadError("not a DSpark drafter with a fusion layer".into()));
        }
        let l = Loader::new(file);
        let flag = |key: &str| -> Result<bool, LoadError> {
            l.hparam(key)?
                .as_bool()
                .ok_or_else(|| LoadError(format!("{key} is not a boolean")))
        };
        let from_anchor = flag("sample_from_anchor")?;
        let has_confidence = flag("has_confidence_head")?;
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
        let rope_dims = l.usize_or("rope.dimension_count", hd)?;
        let rope_base = l.f32("rope.freq_base")?;
        let block = l.usize("block_size")?;
        let n_targets = l.hparam("target_layers")?.as_array().map_or(0, |a| a.len());
        let target_layers = l.array("target_layers", n_targets, |v| {
            v.as_i64()
                .and_then(|v| usize::try_from(v).ok())
                .or_else(|| v.as_u64().map(|v| v as usize))
        })?;
        let mask_token = file
            .header()
            .get("tokenizer.ggml.mask_token_id")
            .and_then(|v| v.as_u64())
            .and_then(|v| u32::try_from(v).ok())
            .ok_or_else(|| LoadError("the drafter names no mask token".into()))?;
        let markov_dims = file
            .header()
            .tensor("markov_w1.weight")
            .map(|t| t.dims.clone())
            .unwrap_or_default();
        let [rank, vocab] = markov_dims[..] else {
            return Err(LoadError("markov_w1.weight is not a matrix".into()));
        };
        let (rank, vocab) = (rank as usize, vocab as usize);
        if !(eps.is_finite() && eps > 0.0)
            || n_layers == 0
            || kv == 0
            || !nh.is_multiple_of(kv)
            || l.usize("attention.value_length")? != hd
            || rope_dims != hd
            || window == 0
            || block < 2
            || rank == 0
            || target_layers.is_empty()
            || mask_token as usize >= vocab
        {
            return Err(LoadError("DSpark hyperparameters are out of range".into()));
        }

        let mut layers = Vec::with_capacity(n_layers);
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
        }

        let header = file.header();
        let n_draft = match header.tensor("d2t") {
            Some(t) => t.elements() as usize,
            None => vocab,
        };
        let d2t = match header.tensor("d2t") {
            Some(_) => Some(l.indices("d2t", n_draft, vocab)?),
            None => None,
        };
        let output = match header.tensor("output.weight") {
            Some(_) => Some(l.matrix("output.weight", d, n_draft)?),
            None if d2t.is_some() => {
                return Err(LoadError(
                    "a drafter with a reduced vocabulary needs its own output head".into(),
                ))
            }
            None => None,
        };
        let confidence = match has_confidence {
            true => {
                let w = l.vector("conf_proj.weight", d + rank)?;
                let b = l.vector("conf_proj.bias", 1)?[0];
                Some((w, b))
            }
            false => None,
        };
        let dspark = FusedDspark {
            d,
            eps,
            layers,
            fc: l.matrix("fc.weight", d * target_layers.len(), d)?,
            fc_norm: l.vector("enc.output_norm.weight", d)?,
            output_norm: l.vector("output_norm.weight", d)?,
            target_layers,
            output,
            d2t,
            markov_in: l.matrix("markov_w1.weight", rank, vocab)?,
            markov_out: l.matrix("markov_w2.weight", rank, n_draft)?,
            rank,
            confidence,
            block,
            from_anchor,
            mask_token,
            vocab,
        };
        l.check_all_read(SKIPPED)?;
        l.refuse_weight_scales()?;
        Ok(dspark)
    }

    /// The target's layers whose input hidden states the drafter reads, in
    /// the order it concatenates them.
    pub fn target_layers(&self) -> &[usize] {
        &self.target_layers
    }

    /// Most tokens one draft proposes.
    pub fn max_draft(&self) -> usize {
        if self.from_anchor {
            self.block
        } else {
            self.block - 1
        }
    }

    /// The width of the target's hidden states the drafter reads.
    pub fn n_embd(&self) -> usize {
        self.d
    }

    /// Tokens of the target's vocabulary.
    pub fn vocab(&self) -> usize {
        self.vocab
    }

    /// The token whose embedding fills the block after the anchor.
    pub fn mask_token(&self) -> u32 {
        self.mask_token
    }

    pub fn new_state(&self) -> FusedDsparkState {
        FusedDsparkState {
            caches: vec![KvCache::default(); self.layers.len()],
            len: 0,
        }
    }

    /// Commits the positions after those `state` holds. `features` holds,
    /// for each position, the target's hidden states at
    /// [`FusedDspark::target_layers`], in that order.
    pub fn commit(&self, state: &mut FusedDsparkState, features: &[f32]) {
        let width = self.fc.cols;
        assert!(
            !features.is_empty() && features.len().is_multiple_of(width),
            "features are not whole positions"
        );
        let rows = features.len() / width;
        let mut g = vec![0.0; rows * self.d];
        matmul(&self.fc, features, &mut g);
        pool::for_each_chunk(&mut g, self.d, |_, row| {
            rms_norm(row, &self.fc_norm, self.eps)
        });
        for (layer, cache) in self.layers.iter().zip(&mut state.caches) {
            let attention = attention(layer);
            let kv_dim = attention.k.rows;
            let (mut k, mut v) = (vec![0.0; rows * kv_dim], vec![0.0; rows * kv_dim]);
            matmul_many(&[&attention.k, &attention.v], &g, &mut [&mut k, &mut v]);
            let k_norm = attention
                .k_norm
                .as_deref()
                .expect("DSpark layers norm keys");
            attention.store_keys(k, &v, k_norm, state.len, cache);
        }
        state.len += rows;
    }

    /// Up to `n` tokens after the anchor, the token at the position after
    /// those `state` holds, each with its chance: the confidence head's
    /// when the drafter has one, otherwise the chance its logits give the
    /// token. `embed` gives the embeddings of tokens as the target's input
    /// layer holds them, and `head` the target's logits for a hidden state,
    /// used only by drafters without their own output head. Ties go to the
    /// lower token.
    pub fn draft(
        &self,
        state: &FusedDsparkState,
        anchor: u32,
        n: usize,
        embed: &dyn Fn(&[u32]) -> Vec<f32>,
        head: &dyn Fn(&[f32]) -> Vec<f32>,
    ) -> Vec<(u32, f32)> {
        assert!(state.len > 0, "nothing committed");
        let n = n.min(self.max_draft());
        if n == 0 {
            return Vec::new();
        }
        let skip = usize::from(!self.from_anchor);
        let mut tokens = vec![self.mask_token; n + skip];
        tokens[0] = anchor;
        let hidden = self.backbone(state, &embed(&tokens));
        let rows = &hidden[skip * self.d..];
        // the rows' own logits do not depend on the tokens chosen before
        // them, so the output head is read once for the whole block
        let base = match &self.output {
            Some(output) => {
                let mut logits = vec![0.0; n * output.rows];
                matmul(output, rows, &mut logits);
                logits
            }
            None => rows.chunks_exact(self.d).flat_map(head).collect(),
        };
        let n_logits = base.len() / n;
        let mut prev = anchor;
        let mut draft = Vec::with_capacity(n);
        let mut markov = vec![0.0; self.rank];
        for (row, base) in rows.chunks_exact(self.d).zip(base.chunks_exact(n_logits)) {
            dequantize_row(
                self.markov_in.ty,
                self.markov_in.data,
                prev as usize,
                &mut markov,
            )
            .expect("markov table type");
            let mut logits = base.to_vec();
            let mut bias = vec![0.0; self.markov_out.rows];
            matmul(&self.markov_out, &markov, &mut bias);
            logits.iter_mut().zip(&bias).for_each(|(l, b)| *l += b);
            let best = crate::speculative::argmax(&logits) as usize;
            let token = self.d2t.as_ref().map_or(best as u32, |d2t| d2t[best]);
            let chance = match &self.confidence {
                Some((w, b)) => {
                    let (wh, wm) = w.split_at(self.d);
                    let dot =
                        |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(a, b)| a * b).sum::<f32>();
                    sigmoid(dot(wh, row) + dot(wm, &markov) + b)
                }
                None => {
                    let top = logits[best];
                    1.0 / logits.iter().map(|l| (l - top).exp()).sum::<f32>()
                }
            };
            draft.push((token, chance));
            prev = token;
        }
        draft
    }

    /// The drafter's normalized output for every row of a block whose input
    /// embeddings are `x`, at the positions after those `state` holds.
    fn backbone(&self, state: &FusedDsparkState, x: &[f32]) -> Vec<f32> {
        let mut x = x.to_vec();
        for (layer, cache) in self.layers.iter().zip(&state.caches) {
            let attention = attention(layer);
            layer.forward_with(&mut x, self.eps, |h| {
                attention.forward_block(h, state.len, cache)
            });
        }
        pool::for_each_chunk(&mut x, self.d, |_, row| {
            rms_norm(row, &self.output_norm, self.eps)
        });
        x
    }
}

/// What a drafter keeps of one sequence: per layer the keys and values of
/// the committed positions, only the window's latest in windowed layers.
#[derive(Clone)]
pub struct FusedDsparkState {
    caches: Vec<KvCache>,
    len: usize,
}

impl FusedDsparkState {
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
