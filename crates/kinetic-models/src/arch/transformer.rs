//! Plain transformers: Llama (with Mistral and SmolLM), Qwen2, Qwen3,
//! Gemma 3, ERNIE 4.5, Hunyuan, EXAONE 4 and OLMo 2. Every layer is
//! grouped-query attention followed by a gated feed-forward. They differ in
//! the rotary layout, in optional projection biases and query/key norms
//! (per head or across heads, before or after the rotation), in whether
//! blocks normalize their input, their output or both, and in Gemma 3's
//! GeGLU and sliding-window layers with their own rotary base.

use kinetic_gguf::Value;

use super::{Basics, Body, Family, Stack};
use crate::blocks::{Activation, Attention, Gate, Mixer, Rope, RopeLayout, SharedKv, SwiGlu};
use crate::loader::Loader;
use crate::model::Layer;
use crate::LoadError;

/// Tensors some of these files carry that the blocks have no place for.
const UNSUPPORTED: &[&str] = &[
    "output.bias",
    "blk.0.attn_qkv.weight",
    "blk.0.attn_output.bias",
    "blk.0.ffn_gate.bias",
    "blk.0.ffn_up.bias",
    "blk.0.ffn_down.bias",
    "blk.0.ffn_gate_inp.weight",
];

pub(crate) const FAMILY: Family = Family::decoder(
    &[
        "llama",
        "qwen2",
        "qwen3",
        "ernie4_5",
        "hunyuan-dense",
        "exaone4",
        "olmo2",
    ],
    build,
);

pub(crate) const GEMMA3: Family = Family {
    sqrt_embd_scale: true,
    ..Family::decoder(&["gemma3"], build)
};

/// EmbeddingGemma, run by [`super::gemma_embedding`].
pub(crate) const GEMMA_EMBEDDING: Family = Family {
    names: &["gemma-embedding"],
    body: Body::Bidirectional(build),
    ..GEMMA3
};

fn build<'a>(l: &Loader<'a>, b: &Basics) -> Result<Stack<'a>, LoadError> {
    Ok(layers(l, b.arch, b.d, b.eps)?.into())
}

pub(crate) fn layers<'a>(
    l: &Loader<'a>,
    arch: &str,
    d: usize,
    eps: f32,
) -> Result<Vec<Layer<'a>>, LoadError> {
    if let Some(name) = UNSUPPORTED.iter().find(|n| l.header().tensor(n).is_some()) {
        return Err(LoadError(format!("tensor {name} is not supported yet")));
    }
    let gemma = matches!(arch, "gemma3" | "gemma-embedding");

    let n_layer = l.usize("block_count")?;
    let n_head = l.usize("attention.head_count")?;
    let n_head_kv = l.usize_or("attention.head_count_kv", n_head)?;
    if n_head_kv == 0 || !n_head.is_multiple_of(n_head_kv) {
        return Err(LoadError("heads do not split into groups".into()));
    }
    let head_dim = l.usize_or("attention.key_length", d / n_head)?;
    if l.usize_or("attention.value_length", head_dim)? != head_dim {
        return Err(LoadError(
            "different key and value head sizes are not supported".into(),
        ));
    }
    let rope_dims = l.usize_or("rope.dimension_count", head_dim)?;
    if rope_dims > head_dim || !rope_dims.is_multiple_of(2) {
        return Err(LoadError(format!("invalid rotary size {rope_dims}")));
    }
    let n_ff = l.usize("feed_forward_length")?;
    let base = match l.header().arch_key("rope.freq_base") {
        Some(_) => l.f32("rope.freq_base")?,
        // Ollama's Gemma 3 converter writes `rope.global.freq_base` or omits both.
        None if gemma => l.f32_or("rope.global.freq_base", 1_000_000.0)?,
        None => 10_000.0,
    };
    let layout = if matches!(arch, "llama" | "ernie4_5") {
        RopeLayout::Adjacent
    } else {
        RopeLayout::Neox
    };
    // Per-pair divisors on full-context layers, as Llama 3 stores them.
    let factors = l.optional_vector("rope_freqs.weight", rope_dims / 2)?;
    let output_norms_only = matches!(arch, "exaone4" | "olmo2");
    let input_norm = |name: &str| match output_norms_only {
        true => Ok(None),
        false => l.vector(name, d).map(Some),
    };

    let window = if gemma {
        l.usize_or("attention.sliding_window", 0)?
    } else {
        0
    };
    let sliding = if window == 0 {
        vec![false; n_layer]
    } else {
        sliding_layers(l, n_layer)?
    };
    let base_swa = match l.header().arch_key("rope.freq_base_swa") {
        Some(_) => l.f32("rope.freq_base_swa")?,
        None if gemma => l.f32_or("rope.local.freq_base", 10_000.0)?,
        None => 10_000.0,
    };
    // The largest Gemma 3 scales queries by the model width per head rather
    // than the head size; folding the ratio into the query norm keeps the
    // attention block unchanged.
    let query_scale =
        (gemma && n_layer == 62).then(|| (head_dim as f32).sqrt() / ((d / n_head) as f32).sqrt());

    (0..n_layer)
        .map(|i| {
            let t = |s: &str| format!("blk.{i}.{s}");
            let (q_dim, kv_dim) = (n_head * head_dim, n_head_kv * head_dim);
            // OLMo 2 normalizes all heads of a token together.
            let (q_norm_len, k_norm_len) = match arch {
                "olmo2" => (q_dim, kv_dim),
                _ => (head_dim, head_dim),
            };
            let mut q_norm = l.optional_vector(&t("attn_q_norm.weight"), q_norm_len)?;
            if let (Some(scale), Some(w)) = (query_scale, &mut q_norm) {
                w.iter_mut().for_each(|w| *w *= scale);
            }
            let rope = if sliding[i] {
                Rope {
                    dims: rope_dims,
                    base: base_swa,
                    factors: None,
                    layout,
                    mscale: 1.0,
                }
            } else {
                Rope {
                    dims: rope_dims,
                    base,
                    factors: factors.clone(),
                    layout,
                    mscale: 1.0,
                }
            };
            let attention = Attention {
                folding: None,
                q: l.matrix(&t("attn_q.weight"), d, q_dim)?,
                k: l.matrix(&t("attn_k.weight"), d, kv_dim)?,
                v: l.matrix(&t("attn_v.weight"), d, kv_dim)?,
                o: l.matrix(&t("attn_output.weight"), q_dim, d)?,
                gate: Gate::None,
                q_bias: l.optional_vector(&t("attn_q.bias"), q_dim)?,
                k_bias: l.optional_vector(&t("attn_k.bias"), kv_dim)?,
                v_bias: l.optional_vector(&t("attn_v.bias"), kv_dim)?,
                q_norm,
                k_norm: l.optional_vector(&t("attn_k_norm.weight"), k_norm_len)?,
                norm_after_rope: arch == "hunyuan-dense",
                n_head,
                n_head_kv,
                head_dim,
                rope,
                window: sliding[i].then_some(window),
                scale: None,
                v_norm: None,
                shared_kv: SharedKv::No,
                eps,
            };
            let activation = if gemma {
                Activation::Gelu
            } else {
                Activation::Silu
            };
            Ok(Layer {
                mixer_norm: input_norm(&t("attn_norm.weight"))?,
                mixer: Mixer::Attention(attention),
                mixer_out_norm: l.optional_vector(&t("post_attention_norm.weight"), d)?,
                ffn_norm: input_norm(&t("ffn_norm.weight"))?,
                ffn: Some(SwiGlu {
                    activation,
                    ..l.swiglu(i, d, n_ff)?
                }),
                experts: None,
                ffn_out_norm: l.optional_vector(&t("post_ffw_norm.weight"), d)?,
                residual_scale: None,
                output_scale: None,
            })
        })
        .collect()
}

/// Which layers attend through a sliding window: a per-layer list, or
/// every layer but each `n`-th, six by default.
fn sliding_layers(l: &Loader, n_layer: usize) -> Result<Vec<bool>, LoadError> {
    let key = "attention.sliding_window_pattern";
    if l.header().arch_key(key).and_then(Value::as_array).is_some() {
        return l.array(key, n_layer, |v: &Value| {
            v.as_bool().or_else(|| v.as_u64().map(|n| n != 0))
        });
    }
    let n = l.usize_or(key, 6)?;
    Ok((0..n_layer).map(|i| n == 0 || i % n < n - 1).collect())
}

#[cfg(test)]
mod tests {
    use kinetic_gguf::{Gguf, TensorInfo, TensorSource, TensorType, Value};

    use super::*;

    const D: u64 = 16;
    const VOCAB: u64 = 32;

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

    /// Six layers laid out as Ollama writes 128k Gemma 3 / MedGemma GGUF
    /// files: five sliding-window layers and a sixth full-attention layer,
    /// without `attention.layer_norm_rms_epsilon`, `rope.freq_base` or
    /// `rope.scaling.factor`.
    fn ollama_gemma3(arch: &str, context: u32, extra: &[(&str, Value)]) -> Synthetic {
        let key = |k: &str, v| (format!("{arch}.{k}"), v);
        let mut metadata = vec![
            ("general.architecture".into(), Value::String(arch.into())),
            key("embedding_length", Value::U32(D as u32)),
            key("block_count", Value::U32(6)),
            key("feed_forward_length", Value::U32(32)),
            key("attention.head_count", Value::U32(2)),
            key("attention.head_count_kv", Value::U32(1)),
            key("attention.key_length", Value::U32(8)),
            key("attention.value_length", Value::U32(8)),
            key("attention.sliding_window", Value::U32(8)),
            key("context_length", Value::U32(context)),
        ];
        metadata.extend(extra.iter().map(|(k, v)| key(k, v.clone())));
        let mut shapes = vec![
            ("token_embd.weight".to_string(), vec![D, VOCAB]),
            ("output_norm.weight".to_string(), vec![D]),
        ];
        for i in 0..6 {
            let t = |s: &str| format!("blk.{i}.{s}");
            shapes.extend([
                (t("attn_norm.weight"), vec![D]),
                (t("attn_q.weight"), vec![D, 16]),
                (t("attn_k.weight"), vec![D, 8]),
                (t("attn_v.weight"), vec![D, 8]),
                (t("attn_output.weight"), vec![16, D]),
                (t("attn_q_norm.weight"), vec![8]),
                (t("attn_k_norm.weight"), vec![8]),
                (t("post_attention_norm.weight"), vec![D]),
                (t("ffn_norm.weight"), vec![D]),
                (t("ffn_gate.weight"), vec![D, 32]),
                (t("ffn_up.weight"), vec![D, 32]),
                (t("ffn_down.weight"), vec![32, D]),
                (t("post_ffw_norm.weight"), vec![D]),
            ]);
        }
        let mut data = Vec::new();
        let tensors = shapes
            .into_iter()
            .map(|(name, dims)| {
                let n: u64 = dims.iter().product();
                let offset = data.len() as u64;
                data.extend(
                    (0..n).flat_map(|j| (((j * 5 + offset) % 11) as f32 * 0.1 - 0.4).to_le_bytes()),
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
    fn ollama_gemma3_files_use_upstream_defaults() {
        let f = ollama_gemma3("gemma3", 131_072, &[]);
        let model = crate::load(&f).unwrap();
        assert_eq!(model.eps, 1e-6);
        let Mixer::Attention(swa) = &model.layers[0].mixer else {
            panic!("expected attention");
        };
        assert_eq!(swa.window, Some(8));
        assert_eq!(swa.rope.base, 10_000.0);
        assert_eq!(swa.rope.factors, None);

        let Mixer::Attention(full) = &model.layers[5].mixer else {
            panic!("expected attention");
        };
        assert_eq!(full.window, None);
        assert_eq!(full.rope.base, 1_000_000.0);
        assert_eq!(full.rope.factors, Some(vec![8.0; 4]));

        let logits = model.run(&mut model.new_state(), &[1, 2, 3], false).logits;
        assert_eq!(logits.len(), VOCAB as usize);
        assert!(logits.iter().all(|v| v.is_finite()));

        // 32k 1B files keep unit scaling unless the file sets a factor.
        let small_file = ollama_gemma3("gemma3", 32_768, &[]);
        let small = crate::load(&small_file).unwrap();
        let Mixer::Attention(full_small) = &small.layers[5].mixer else {
            panic!("expected attention");
        };
        assert_eq!(full_small.rope.factors, None);

        // Non-Gemma families still require `attention.layer_norm_rms_epsilon`.
        let llama_file = ollama_gemma3("llama", 131_072, &[]);
        assert!(crate::load(&llama_file).is_err());
    }
}
