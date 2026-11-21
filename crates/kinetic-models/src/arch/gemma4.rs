//! Gemma 4. Sliding-window and full-attention layers with their own head
//! sizes and rotary bases, GeGLU feed-forwards of per-layer widths and
//! per-layer token inputs. The last layers project no keys or values of
//! their own and attend over those of the last earlier layer of their kind.
//! Image rows go in through [`crate::Input::Image`]: causal like text, at
//! consecutive positions, with the padding token's per-layer input.
//! The larger sizes have no per-layer inputs, and their full-attention
//! layers fewer key-value heads and no values of their own. In the
//! mixture-of-experts size the feed-forward is the shared expert beside the
//! routed experts, each with norms of its own.

use super::{Basics, Family, Stack};
use crate::blocks::{
    Activation, Attention, ExpertNorms, Gate, Gating, Glu, Mixer, Moe, Rope, RopeLayout, SharedKv,
    SwiGlu,
};
use crate::loader::Loader;
use crate::model::Layer;
use crate::per_layer::{LayerInput, PerLayer};
use crate::LoadError;

pub(crate) const FAMILY: Family = Family {
    sqrt_embd_scale: true,
    ..Family::decoder(&["gemma4"], build)
};

fn build<'a>(l: &Loader<'a>, b: &Basics) -> Result<Stack<'a>, LoadError> {
    let (d, eps) = (b.d, b.eps);
    let n_layer = l.usize("block_count")?;
    let n_head = l.usize("attention.head_count")?;
    let n_head_kv = l.usizes("attention.head_count_kv", n_layer)?;
    if n_head_kv
        .iter()
        .any(|&n| n == 0 || !n_head.is_multiple_of(n))
    {
        return Err(LoadError("heads do not split into groups".into()));
    }
    let n_ff = l.usizes("feed_forward_length", n_layer)?;
    let sliding = l.array("attention.sliding_window_pattern", n_layer, |v| v.as_bool())?;
    let window = l.usize("attention.sliding_window")?;
    let shared = l.usize_or("attention.shared_kv_layers", 0)?;
    let first_shared = n_layer
        .checked_sub(shared)
        .ok_or_else(|| LoadError("more shared layers than layers".into()))?;
    // a shared layer reads the last unshared layer of its kind
    let source: Vec<Option<usize>> = (0..n_layer)
        .map(|i| {
            (i >= first_shared)
                .then(|| (0..first_shared).rev().find(|&j| sliding[j] == sliding[i]))
                .flatten()
        })
        .collect();
    let head_full = l.usize("attention.key_length")?;
    let head_swa = l.usize("attention.key_length_swa")?;
    if l.usize_or("attention.value_length", head_full)? != head_full
        || l.usize_or("attention.value_length_swa", head_swa)? != head_swa
    {
        return Err(LoadError("value heads must be as long as key heads".into()));
    }
    let factors = l.optional_vector("rope_freqs.weight", head_full / 2)?;
    let width = l.usize("embedding_length_per_layer_input")?;

    let mut layers = Vec::with_capacity(n_layer);
    let mut inputs = Vec::with_capacity(n_layer);
    for i in 0..n_layer {
        let t = |s: &str| format!("blk.{i}.{s}");
        let scale = l.vector(&t("layer_output_scale.weight"), 1)?[0];
        let (head_dim, rope) = if sliding[i] {
            let rope = Rope {
                dims: l.usize_or("rope.dimension_count_swa", head_swa)?,
                base: l.f32_or("rope.freq_base_swa", 10_000.0)?,
                factors: None,
                layout: RopeLayout::Neox,
                mscale: 1.0,
            };
            (head_swa, rope)
        } else {
            let rope = Rope {
                dims: l.usize_or("rope.dimension_count", head_full)?,
                base: l.f32("rope.freq_base")?,
                factors: factors.clone(),
                layout: RopeLayout::Neox,
                mscale: 1.0,
            };
            (head_full, rope)
        };
        let kv_layer = source[i].unwrap_or(i);
        let n_head_kv = n_head_kv[kv_layer];
        let (q_dim, kv_dim) = (n_head * head_dim, n_head_kv * head_dim);
        let kv = |s: &str| format!("blk.{kv_layer}.{s}");
        // layers without values of their own take the unnormalized keys
        let v = match l.header().tensor(&kv("attn_v.weight")) {
            Some(_) => kv("attn_v.weight"),
            None => kv("attn_k.weight"),
        };
        let attention = Attention {
            q: l.matrix(&t("attn_q.weight"), d, q_dim)?,
            k: l.matrix(&kv("attn_k.weight"), d, kv_dim)?,
            v: l.matrix(&v, d, kv_dim)?,
            o: l.matrix(&t("attn_output.weight"), q_dim, d)?,
            folding: None,
            gate: Gate::None,
            q_bias: None,
            k_bias: None,
            v_bias: None,
            q_norm: Some(l.vector(&t("attn_q_norm.weight"), head_dim)?),
            k_norm: Some(l.vector(&kv("attn_k_norm.weight"), head_dim)?),
            norm_after_rope: false,
            n_head,
            n_head_kv,
            head_dim,
            rope,
            window: sliding[i].then_some(window),
            // the query norm stands in for the usual 1/sqrt(head_dim)
            scale: Some(1.0),
            v_norm: Some(vec![1.0; head_dim]),
            shared_kv: match source[i] {
                Some(j) => SharedKv::Borrows(j),
                None if source.contains(&Some(i)) => SharedKv::Lends,
                None => SharedKv::No,
            },
            eps,
        };
        layers.push(Layer {
            mixer_norm: Some(l.vector(&t("attn_norm.weight"), d)?),
            mixer: Mixer::Attention(attention),
            mixer_out_norm: Some(l.vector(&t("post_attention_norm.weight"), d)?),
            ffn_norm: Some(l.vector(&t("ffn_norm.weight"), d)?),
            ffn: Some(SwiGlu {
                activation: Activation::Gelu,
                ..l.swiglu(i, d, n_ff[i])?
            }),
            experts: match l.header().tensor(&t("ffn_gate_inp.weight")) {
                Some(_) => Some(experts(l, i, d)?),
                None => None,
            },
            ffn_out_norm: Some(l.vector(&t("post_ffw_norm.weight"), d)?),
            residual_scale: None,
            output_scale: (width == 0).then_some(scale),
        });
        if width > 0 {
            inputs.push(LayerInput {
                gate: l.matrix(&t("inp_gate.weight"), d, width)?,
                proj: l.matrix(&t("proj.weight"), width, d)?,
                norm: l.vector(&t("post_norm.weight"), d)?,
                scale,
            });
        }
    }
    if layers.iter().any(|l| l.experts.is_some()) {
        l.check_all_read(SKIPPED)?;
    }
    if width == 0 {
        return Ok(layers.into());
    }
    let n_vocab = l
        .header()
        .tensor("per_layer_token_embd.weight")
        .map(|t| t.dims[1] as usize)
        .ok_or_else(|| LoadError("missing tensor per_layer_token_embd.weight".into()))?;
    let per_layer = PerLayer {
        embd: l.matrix("per_layer_token_embd.weight", n_layer * width, n_vocab)?,
        proj: l.matrix("per_layer_model_proj.weight", d, n_layer * width)?,
        proj_norm: l.vector("per_layer_proj_norm.weight", width)?,
        width,
        layers: inputs,
    };
    Ok(Stack {
        layers,
        mtp: None,
        per_layer: Some(per_layer),
        network: None,
    })
}

/// What the shared loader reads after the layers.
const SKIPPED: &[&str] = &[
    "context_length",
    "final_logit_softcapping",
    "token_embd.weight",
    "output.weight",
    "output_norm.weight",
];

/// The routed experts of layer `i`, whose feed-forward is their shared
/// expert.
fn experts<'a>(l: &Loader<'a>, i: usize, d: usize) -> Result<Moe<'a>, LoadError> {
    let t = |s: &str| format!("blk.{i}.{s}");
    let n_ff = l.usize("expert_feed_forward_length")?;
    let n_experts = l.usize("expert_count")?;
    let (gate, up) = super::moe::gate_up(l, i, d, n_ff, n_experts)?;
    // the router reads the normalized input over sqrt(d), times its scale
    let mut router = l.vector(&t("ffn_gate_inp.scale"), d)?;
    router.iter_mut().for_each(|w| *w /= (d as f32).sqrt());
    Ok(Moe {
        router: l.matrix(&t("ffn_gate_inp.weight"), d, n_experts)?,
        gate,
        up,
        down: l.experts(&t("ffn_down_exps.weight"), n_ff, d, n_experts)?,
        n_used: l.usize("expert_used_count")?,
        shared_gate: None,
        gating: Gating::Softmax,
        glu: Glu::Gelu,
        weight_scale: 1.0,
        scales: [
            None,
            None,
            l.optional_vector(&t("ffn_down_exps.scale"), n_experts)?,
        ],
        norms: Some(ExpertNorms {
            input: l.vector(&t("pre_ffw_norm_2.weight"), d)?,
            router,
            output: l.vector(&t("post_ffw_norm_2.weight"), d)?,
            shared: l.vector(&t("post_ffw_norm_1.weight"), d)?,
        }),
    })
}

#[cfg(test)]
mod tests {
    use kinetic_gguf::{Gguf, TensorInfo, TensorSource, TensorType, Value};

    use crate::blocks::Mixer;

    const D: u64 = 8;
    const VOCAB: u64 = 16;

    struct Synthetic {
        header: Gguf,
        data: Vec<u8>,
    }

    impl TensorSource for Synthetic {
        fn header(&self) -> &Gguf {
            &self.header
        }

        fn tensor_data(&self, t: &TensorInfo) -> &[u8] {
            &self.data[t.offset as usize..][..t.size as usize]
        }
    }

    /// Two layers laid out as in the 12B and 31B files: a sliding-window
    /// layer, then a full-attention one with fewer key-value heads and no
    /// values, and no per-layer inputs; each layer also gets the `tensors`.
    fn file(extra: &[(&str, Value)], tensors: &[(&str, &[u64])]) -> Synthetic {
        let key = |k: &str, v| (format!("gemma4.{k}"), v);
        let mut metadata = vec![
            (
                "general.architecture".into(),
                Value::String("gemma4".into()),
            ),
            key("embedding_length", Value::U32(D as u32)),
            key("block_count", Value::U32(2)),
            key("feed_forward_length", Value::U32(16)),
            key("attention.head_count", Value::U32(2)),
            key(
                "attention.head_count_kv",
                Value::Array(vec![Value::U32(2), Value::U32(1)]),
            ),
            key("attention.layer_norm_rms_epsilon", Value::F32(1e-6)),
            key(
                "attention.sliding_window_pattern",
                Value::Array(vec![Value::Bool(true), Value::Bool(false)]),
            ),
            key("attention.sliding_window", Value::U32(4)),
            key("attention.key_length", Value::U32(8)),
            key("attention.key_length_swa", Value::U32(4)),
            key("rope.freq_base", Value::F32(1e6)),
            key("embedding_length_per_layer_input", Value::U32(0)),
            key("final_logit_softcapping", Value::F32(30.0)),
        ];
        metadata.extend(extra.iter().map(|(k, v)| key(k, v.clone())));
        let mut shapes = vec![
            ("token_embd.weight".to_string(), vec![D, VOCAB]),
            ("output_norm.weight".to_string(), vec![D]),
        ];
        for (i, (hd, kv, values)) in [(4, 2, true), (8, 1, false)].into_iter().enumerate() {
            let t = |s: &str| format!("blk.{i}.{s}");
            shapes.extend([
                (t("attn_norm.weight"), vec![D]),
                (t("attn_q.weight"), vec![D, 2 * hd]),
                (t("attn_k.weight"), vec![D, kv * hd]),
                (t("attn_output.weight"), vec![2 * hd, D]),
                (t("attn_q_norm.weight"), vec![hd]),
                (t("attn_k_norm.weight"), vec![hd]),
                (t("post_attention_norm.weight"), vec![D]),
                (t("ffn_norm.weight"), vec![D]),
                (t("ffn_gate.weight"), vec![D, 16]),
                (t("ffn_up.weight"), vec![D, 16]),
                (t("ffn_down.weight"), vec![16, D]),
                (t("post_ffw_norm.weight"), vec![D]),
                (t("layer_output_scale.weight"), vec![1]),
            ]);
            if values {
                shapes.push((t("attn_v.weight"), vec![D, kv * hd]));
            }
            shapes.extend(tensors.iter().map(|(n, dims)| (t(n), dims.to_vec())));
        }
        let mut data = Vec::new();
        let tensors = shapes
            .into_iter()
            .map(|(name, dims)| {
                let n: u64 = dims.iter().product();
                let offset = data.len() as u64;
                data.extend(
                    (0..n).flat_map(|j| (((j * 7 + offset) % 13) as f32 * 0.1 - 0.5).to_le_bytes()),
                );
                TensorInfo {
                    name,
                    dims,
                    ty: TensorType::F32,
                    offset,
                    size: n * 4,
                }
            })
            .collect();
        Synthetic {
            header: Gguf {
                version: 3,
                alignment: 32,
                data_offset: 0,
                metadata,
                tensors,
            },
            data,
        }
    }

    #[test]
    fn dense_sizes_load_and_run() {
        let file = file(&[], &[]);
        let model = crate::load(&file).unwrap_or_else(|e| panic!("{}", e.0));
        assert!(model.per_layer.is_none());
        let Mixer::Attention(full) = &model.layers[1].mixer else {
            panic!("an attention layer");
        };
        assert_eq!(full.n_head_kv, 1);
        assert!(std::ptr::eq(full.v.data, full.k.data));
        assert!(model.layers.iter().all(|l| l.output_scale.is_some()));
        let logits = model.run(&mut model.new_state(), &[1, 5, 3], false).logits;
        assert!(logits.iter().all(|v| v.is_finite()));
    }

    /// The 26B A4B layout: four experts of width four, two per token,
    /// gate and up stored together.
    const EXPERTS: &[(&str, &[u64])] = &[
        ("ffn_gate_inp.weight", &[D, 4]),
        ("ffn_gate_inp.scale", &[D]),
        ("ffn_gate_up_exps.weight", &[D, 8, 4]),
        ("ffn_down_exps.weight", &[4, D, 4]),
        ("ffn_down_exps.scale", &[4]),
        ("pre_ffw_norm_2.weight", &[D]),
        ("post_ffw_norm_1.weight", &[D]),
        ("post_ffw_norm_2.weight", &[D]),
    ];

    fn experts(tensors: &[(&str, &[u64])]) -> Synthetic {
        file(
            &[
                ("expert_count", Value::U32(4)),
                ("expert_used_count", Value::U32(2)),
                ("expert_feed_forward_length", Value::U32(4)),
            ],
            tensors,
        )
    }

    // Expected values from a NumPy (float64) rendering of llama.cpp's
    // Gemma 4 feed-forward on the same weights, for the second layer, as
    // the first one's output scale is zero.
    #[test]
    fn experts_match_the_reference() {
        let file = experts(EXPERTS);
        let model = crate::load(&file).unwrap_or_else(|e| panic!("{}", e.0));
        let mut x: Vec<f32> = (0..2 * D)
            .map(|i| ((i * 5) % 11) as f32 * 0.3 - 1.4)
            .collect();
        model.layers[1].forward_with(&mut x, 1e-6, |h| vec![0.0; h.len()]);
        let want = [
            -0.84, 0.218986, 0.971891, -0.0978544, 1.07974, -0.403377, 0.762813, -0.584191, 0.42,
            -0.816299, 0.291497, -0.669238, -0.160717, 1.23901, 0.00574467, 0.762909,
        ];
        for (g, w) in x.iter().zip(want) {
            assert!((g - w).abs() < 1e-4, "{x:?} against {want:?}");
        }
        let logits = model.run(&mut model.new_state(), &[1, 5, 3], false).logits;
        assert!(logits.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn experts_in_a_layout_not_read_are_refused() {
        let mut tensors = EXPERTS.to_vec();
        tensors.push(("ffn_gate_exps.weight", &[D, 4, 4]));
        let err = crate::load(&experts(&tensors)).err().expect("an error");
        assert!(err.0.contains("blk.0.ffn_gate_exps.weight"), "{}", err.0);
    }
}
