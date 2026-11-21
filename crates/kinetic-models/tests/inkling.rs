//! Inkling 2: a tiny looped model against reference logits, and the
//! ignored tests against a real file. Only the real file's header matters,
//! so it may be a copy whose tensor data is a sparse hole of the same
//! length. Loading fails on any tensor or `<arch>.` key the family neither
//! reads nor names as skipped.
//!
//! KINETIC_TEST_INKLING=Inkling-2-Mini-IQ4_NL.gguf cargo test --release -p kinetic-models --test inkling -- --ignored

use kinetic_gguf::{Gguf, GgufFile, TensorInfo, TensorOut, TensorSource, TensorType, Value};

fn open() -> GgufFile {
    let path = std::env::var("KINETIC_TEST_INKLING").expect("set KINETIC_TEST_INKLING");
    GgufFile::open(path).unwrap()
}

#[test]
#[ignore]
fn reads_every_tensor_and_key_of_the_file() {
    let file = open();
    let model = kinetic_models::load(&file).unwrap();
    let key = |k: &str| {
        let arch = file.header().architecture().unwrap();
        file.header()
            .get(&format!("{arch}.{k}"))
            .and_then(|v| v.as_u64())
    };
    let n_phys = key("block_count").unwrap();
    let extra = match (key("loop.start"), key("loop.end"), key("loop.count")) {
        (Some(start), Some(end), Some(count)) => (end - start) * (count - 1),
        _ => 0,
    };
    assert_eq!(model.n_layers() as u64, n_phys + extra);
    assert_eq!(Some(model.n_vocab() as u64), key("vocab_size_unpadded"));
}

/// Runs a token through every layer and pass. With the sparse copy the
/// weights are zeros, so this checks the shapes and the order of the
/// layers, not the values.
#[test]
#[ignore]
fn runs_a_token_through_every_pass() {
    let file = open();
    let model = kinetic_models::load(&file).unwrap();
    let mut state = model.new_state();
    let logits = model.forward_batch(&mut state, &[1, 2]);
    assert_eq!(logits.len(), model.n_vocab());
    assert!(logits.iter().all(|l| l.is_finite()));
    let logits = model.forward(&mut state, 3);
    assert!(logits.iter().all(|l| l.is_finite()));
}

/// A file held in memory.
struct Memory {
    header: Gguf,
    bytes: Vec<u8>,
}

impl TensorSource for Memory {
    fn header(&self) -> &Gguf {
        &self.header
    }

    fn tensor_data(&self, t: &TensorInfo) -> &[u8] {
        &self.bytes[t.offset as usize..][..t.size as usize]
    }
}

const D: u64 = 8;
const N_PHYS: usize = 6;
const LOOP: std::ops::Range<usize> = 2..4;

/// The tensors of a tiny looped model: every other layer windowed, blocks
/// of two layers and a core of one block that runs twice.
fn tiny_tensors() -> Vec<(String, Vec<u64>)> {
    let (heads, hd, rel, taps, experts, ff, rank) = (2, 4, 2, 3, 4, 4, 3);
    let mut t: Vec<(String, Vec<u64>)> = [
        ("token_embd.weight", vec![D, 12]),
        ("token_embd_norm.weight", vec![D]),
        ("output.weight", vec![D, 12]),
        ("output_norm.weight", vec![D]),
        ("output_res_query.weight", vec![D]),
    ]
    .into_iter()
    .map(|(n, d)| (n.to_string(), d))
    .collect();
    for i in 0..N_PHYS {
        let windowed = i % 2 == 0;
        let mut add = |name: &str, dims: Vec<u64>| t.push((format!("blk.{i}.{name}"), dims));
        add("attn_q.weight", vec![D, heads * hd]);
        add("attn_k.weight", vec![D, hd]);
        add("attn_v.weight", vec![D, hd]);
        add("attn_output.weight", vec![heads * hd, D]);
        add("attn_gate.weight", vec![D, heads * hd]);
        add("attn_sink_proj.weight", vec![D, heads]);
        add("attn_q_clip.weight", vec![hd, heads]);
        add("attn_rel.weight", vec![D, rel * heads]);
        add(
            "attn_rel_bias.weight",
            vec![rel, if windowed { 2 } else { 3 }],
        );
        add("attn_rel_clip.weight", vec![rel, heads]);
        add("attn_k_conv.weight", vec![taps, hd]);
        add("attn_v_conv.weight", vec![taps, hd]);
        add("attn_post_conv.weight", vec![taps, D]);
        add("ffn_post_conv.weight", vec![taps, D]);
        if windowed {
            add("attn_norm.weight", vec![D]);
        } else {
            add("attn_q_inp_norm.weight", vec![D]);
            add("attn_k_inp_norm.weight", vec![D]);
            add("attn_v_inp_norm.weight", vec![D]);
            add("attn_k_conv_proj.weight", vec![D, taps]);
            add("attn_v_conv_proj.weight", vec![D, taps]);
        }
        if i > 0 {
            add(
                "attn_res_query.weight",
                if windowed { vec![D] } else { vec![D, 3] },
            );
        }
        add("ffn_res_query.weight", vec![D]);
        add("ffn_norm.weight", vec![D]);
        add("ffn_gate_inp.weight", vec![D, experts]);
        add("exp_probs_b.bias", vec![experts]);
        add("ffn_gate_up_exps.weight", vec![D, 2 * ff, experts]);
        add("ffn_down_exps.weight", vec![ff, D, experts]);
        if LOOP.contains(&i) {
            add("loop_msg_query.weight", vec![D, 4]);
            for x in ["q", "k", "v", "ffn"] {
                add(&format!("loop_{x}_ctrl_down.weight"), vec![2 * D, rank]);
                add(&format!("loop_{x}_ctrl_gain.weight"), vec![rank, D]);
                add(&format!("loop_{x}_ctrl_forget.weight"), vec![rank, D]);
            }
        }
    }
    t
}

fn tiny_model() -> Memory {
    let arch = "inkling2_loop";
    let per_layer = |v: Value| Value::Array(vec![v; N_PHYS]);
    let mut metadata = vec![(
        "general.architecture".to_string(),
        Value::String(arch.into()),
    )];
    let keys = [
        ("block_count", Value::U32(N_PHYS as u32)),
        ("context_length", Value::U32(64)),
        ("embedding_length", Value::U32(D as u32)),
        ("feed_forward_length", Value::U32(4)),
        ("attention.head_count", per_layer(Value::U32(2))),
        ("attention.head_count_kv", per_layer(Value::U32(1))),
        ("attention.layer_norm_rms_epsilon", Value::F32(1e-6)),
        ("expert_count", Value::U32(4)),
        ("expert_used_count", Value::U32(2)),
        ("attention.key_length", Value::U32(4)),
        ("attention.value_length", Value::U32(4)),
        ("vocab_size", Value::U32(12)),
        ("vocab_size_unpadded", Value::U32(10)),
        ("expert_feed_forward_length", Value::U32(4)),
        ("rope.dimension_count", Value::U32(0)),
        ("attention.sliding_window", Value::U32(2)),
        (
            "attention.sliding_window_pattern",
            Value::Array((0..N_PHYS).map(|i| Value::Bool(i % 2 == 0)).collect()),
        ),
        ("shortconv.l_cache", Value::U32(3)),
        ("expert_weights_scale", Value::F32(2.0)),
        ("activation.situ_beta", Value::F32(4.0)),
        ("activation.situ_linear_beta", Value::F32(25.0)),
        ("attn_res.block_size", Value::U32(2)),
        ("final_logit_softcapping", Value::F32(30.0)),
        ("attention.relative_length", Value::U32(2)),
        ("attention.relative_extent", Value::U32(3)),
        ("attention.qk_clip", Value::F32(1.5)),
        ("attention.relative_clip", Value::F32(0.8)),
        ("attention.log_scale_floor", Value::U32(2)),
        ("attention.log_scale_alpha", Value::F32(0.5)),
        ("loop.start", Value::U32(LOOP.start as u32)),
        ("loop.end", Value::U32(LOOP.end as u32)),
        ("loop.count", Value::U32(2)),
        ("loop.gate_rank", Value::U32(3)),
        ("loop.varying", Value::Bool(false)),
    ];
    metadata.extend(keys.into_iter().map(|(k, v)| (format!("{arch}.{k}"), v)));
    let tensors = tiny_tensors();
    let out: Vec<TensorOut> = tensors
        .iter()
        .map(|(name, dims)| TensorOut {
            name: name.clone(),
            dims: dims.clone(),
            ty: TensorType::F32,
        })
        .collect();
    let mut bytes = Vec::new();
    kinetic_gguf::write(&mut bytes, &metadata, &out, |k| {
        let (name, dims) = &tensors[k];
        let n: u64 = dims.iter().product();
        let shift = if name.contains("norm") || name.contains("_clip") {
            1.0
        } else {
            0.0
        };
        Ok((0..n as usize)
            .map(|i| ((i * (2 * k + 3)) % 17) as f32 - 8.0)
            .flat_map(|v| (v * 0.07 + shift).to_le_bytes())
            .collect())
    })
    .unwrap();
    Memory {
        header: Gguf::parse(&bytes).unwrap(),
        bytes,
    }
}

/// Logits of the tiny model after each of the tokens 3, 7, 1 and 9,
/// whichever way they are batched, against values computed in NumPy
/// (float64) from the graphs of llama.cpp's src/models/inkling2.cpp and
/// inkling2-loop.cpp on the same weights.
#[test]
fn tiny_model_matches_the_reference() {
    let file = tiny_model();
    let model = kinetic_models::load(&file).unwrap();
    assert_eq!(model.n_layers(), N_PHYS + LOOP.len());
    assert_eq!(model.n_vocab(), 10);
    let tokens = [3, 7, 1, 9];
    let mut first: Option<Vec<f32>> = None;
    for batches in [&[4][..], &[2, 1, 1], &[1, 1, 1, 1]] {
        let mut state = model.new_state();
        let mut logits = Vec::new();
        let mut at = 0;
        for &n in batches {
            logits.extend(model.run(&mut state, &tokens[at..at + n], true).logits);
            at += n;
        }
        for (i, (got, want)) in logits.iter().zip(LOGITS).enumerate() {
            assert!(
                (got - want).abs() < 1e-4,
                "{batches:?}, logit {i}: {got} against {want}"
            );
        }
        // a batch gives each token what it gets alone, bit for bit
        assert_eq!(first.get_or_insert_with(|| logits.clone()), &logits);
    }
}

const LOGITS: [f32; 40] = [
    1.337009, 0.8481389, -0.7500277, -1.971547, 0.6931028, 0.5891846, -2.125962, 1.414415,
    0.4340913, -0.6725019, -0.4141479, 0.8115726, -1.191427, -0.212345, 0.7890617, 0.707845,
    -0.2348708, -0.4028865, 0.6853302, -1.180181, 0.2231518, 0.5435061, -0.398955, -1.543489,
    0.460775, 0.9439149, -1.626012, 0.2645262, 0.8612356, -0.3575845, 0.08001616, 0.8101404,
    -1.594792, 0.02124436, 0.5619222, 1.058711, -0.2271008, 0.2041879, 0.8106044, -1.470942,
];
