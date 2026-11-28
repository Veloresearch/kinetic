//! Phi-3 Mini 4K: full attention with NeoX rotary positions and a SwiGLU
//! feed-forward. The GGUF packs query, key and value in `attn_qkv`;
//! `ffn_up` holds the gate first and the up projection second.

use kinetic_quant::Matrix;

use super::{Basics, Family, Stack};
use crate::blocks::{Activation, Attention, Gate, Mixer, Rope, RopeLayout, SharedKv, SwiGlu};
use crate::loader::Loader;
use crate::model::Layer;
use crate::LoadError;

const UNSUPPORTED: &[&str] = &[
    "output.bias",
    "rope_freqs.weight",
    "rope_factors_long.weight",
    "rope_factors_short.weight",
];
const UNSUPPORTED_LAYER: &[&str] = &[
    "attn_qkv.bias",
    "attn_output.bias",
    "attn_q_norm.weight",
    "attn_k_norm.weight",
    "ffn_gate.weight",
    "ffn_gate.bias",
    "ffn_up.bias",
    "ffn_down.bias",
    "ffn_gate_inp.weight",
    "post_attention_norm.weight",
    "post_ffw_norm.weight",
];

pub(crate) const FAMILY: Family = Family::decoder(&["phi3"], build);

fn build<'a>(l: &Loader<'a>, b: &Basics) -> Result<Stack<'a>, LoadError> {
    let rope_scale = if l.header().arch_key("rope.scaling.factor").is_some() {
        l.f32("rope.scaling.factor")?
    } else {
        l.f32_or("rope.scale_linear", 1.0)?
    };
    if l.usize("context_length")? != 4096
        || l.usize_or("attention.sliding_window", 0)? != 0
        || !matches!(rope_scale, 0.0 | 1.0)
        || l.header()
            .arch_key("rope.scaling.type")
            .is_some_and(|v| v.as_str() != Some("none"))
    {
        return Err(LoadError(
            "only Phi-3 4K with unscaled rotary embeddings and full attention is supported".into(),
        ));
    }
    let d = b.d;
    let n_layer = l.usize("block_count")?;
    let n_head = l.usize("attention.head_count")?;
    let n_head_kv = l.usize_or("attention.head_count_kv", n_head)?;
    if n_head == 0
        || n_head_kv == 0
        || !d.is_multiple_of(n_head)
        || !n_head.is_multiple_of(n_head_kv)
    {
        return Err(LoadError("heads do not split into groups".into()));
    }
    let head_dim = d / n_head;
    if l.usize_or("attention.key_length", head_dim)? != head_dim
        || l.usize_or("attention.value_length", head_dim)? != head_dim
        || l.usize_or("rope.dimension_count", head_dim)? != head_dim
        || !head_dim.is_multiple_of(2)
    {
        return Err(LoadError("unsupported Phi-3 head or rotary size".into()));
    }
    let n_ff = l.usize("feed_forward_length")?;
    let base = l.f32_or("rope.freq_base", 10_000.0)?;
    let kv_dim = n_head_kv * head_dim;
    if let Some(tensor) = l.header().tensors.iter().find(|tensor| {
        UNSUPPORTED.contains(&tensor.name.as_str())
            || tensor
                .name
                .strip_prefix("blk.")
                .and_then(|name| name.split_once('.'))
                .is_some_and(|(_, suffix)| UNSUPPORTED_LAYER.contains(&suffix))
    }) {
        return Err(LoadError(format!(
            "tensor {} is not supported in Phi-3",
            tensor.name
        )));
    }
    let layers = (0..n_layer)
        .map(|i| {
            let t = |s: &str| format!("blk.{i}.{s}.weight");
            let qkv = l.matrix(&t("attn_qkv"), d, d + 2 * kv_dim)?;
            let gate_up = l.matrix(&t("ffn_up"), d, 2 * n_ff)?;
            let attention = Attention {
                folding: None,
                q: rows(qkv, 0, d),
                k: rows(qkv, d, kv_dim),
                v: rows(qkv, d + kv_dim, kv_dim),
                o: l.matrix(&t("attn_output"), d, d)?,
                gate: Gate::None,
                q_bias: None,
                k_bias: None,
                v_bias: None,
                q_norm: None,
                k_norm: None,
                norm_after_rope: false,
                n_head,
                n_head_kv,
                head_dim,
                rope: Rope {
                    dims: head_dim,
                    base,
                    factors: None,
                    layout: RopeLayout::Neox,
                    mscale: 1.0,
                },
                window: None,
                scale: None,
                v_norm: None,
                shared_kv: SharedKv::No,
                eps: b.eps,
            };
            Ok(Layer {
                mixer_norm: Some(l.vector(&t("attn_norm"), d)?),
                mixer: Mixer::Attention(attention),
                mixer_out_norm: None,
                ffn_norm: Some(l.vector(&t("ffn_norm"), d)?),
                ffn: Some(SwiGlu {
                    gate: rows(gate_up, 0, n_ff),
                    up: rows(gate_up, n_ff, n_ff),
                    down: l.matrix(&t("ffn_down"), n_ff, d)?,
                    activation: Activation::Silu,
                    folding: None,
                }),
                experts: None,
                ffn_out_norm: None,
                residual_scale: None,
                output_scale: None,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(layers.into())
}

/// Borrows whole output rows of a checked matrix without unpacking them.
fn rows(w: Matrix<'_>, start: usize, count: usize) -> Matrix<'_> {
    let stride = w.row_bytes();
    Matrix::new(
        w.ty,
        &w.data[start * stride..(start + count) * stride],
        count,
        w.cols,
    )
}
