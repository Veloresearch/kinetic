//! Inkling 2: sigmoid-routed experts in every layer and attention by
//! relative distance in windowed and full layers, with no residual stream:
//! each block of layers starts from a mix of the outputs of earlier blocks
//! (attention over depth). `inkling2_loop` runs a core of its blocks
//! several times; each pass restarts from the output before the core, has
//! its own caches, and mixes in a message from the previous pass.

use kinetic_gguf::Value;

use super::{Basics, Family, Stack};
use crate::blocks::{mix, Control, Gating, Glu, Mixer, MixerState, Moe, RelAttention, Tree};
use crate::loader::Loader;
use crate::model::Layer;
use crate::LoadError;

pub(crate) const FAMILY: Family = Family::decoder(&["inkling2", "inkling2_loop"], build);

/// Keys and tensors the layers leave to the embedding and head, or do not
/// need.
const SKIPPED: &[&str] = &[
    "context_length",
    "feed_forward_length",
    "final_logit_softcapping",
    "token_embd.weight",
    "output.weight",
    "output_norm.weight",
];

/// The controls of the query, key, value and feed-forward inputs of a looped
/// layer, as ggml-org's files and as Unsloth's name them; the files Unsloth
/// publishes in both layouts hold the same bytes under these names.
const CONTROLS: [&str; 4] = ["q", "k", "v", "ffn"];
const UNROLLED_CONTROLS: [&str; 4] = ["attn", "k", "v", "mlp"];

/// What the layers of a network share beyond their weights: where they
/// read their inputs from, and the loop.
pub(crate) struct Network<'a> {
    d: usize,
    eps: f32,
    embd_norm: Vec<f32>,
    /// The query that mixes the blocks into the input of the output norm.
    out_query: Vec<f32>,
    /// Layers per block.
    block: usize,
    /// One per physical layer.
    wiring: Vec<Wiring<'a>>,
    core: Option<Core>,
}

/// The physical layers `start..end` run `passes` times; pass `p > 0` keeps
/// its states in the layers after the physical ones.
struct Core {
    start: usize,
    end: usize,
    passes: usize,
}

struct Wiring<'a> {
    /// One query for the input of a windowed layer, one each for the
    /// query, key and value inputs of a full one; the first layer has none.
    attn_query: Option<Vec<f32>>,
    ffn_query: Vec<f32>,
    /// Layers of the core: queries for the messages to the query, key,
    /// value and feed-forward inputs, and their controls.
    looped: Option<(Vec<f32>, [Control<'a>; 4])>,
}

fn build<'a>(l: &Loader<'a>, b: &Basics) -> Result<Stack<'a>, LoadError> {
    let d = b.d;
    let block = l.usize("attn_res.block_size")?;
    let listed = l.usize("block_count")?;
    // Unsloth's files count every pass of the core in block_count and in the
    // per-layer arrays, store each physical layer once, and name the loop
    // keys otherwise.
    let unrolled = l.header().arch_key("num_loops").is_some();
    let core = if unrolled {
        Some(Core {
            start: l.usize("loop.core_start")?,
            end: l.usize("loop.core_end")?,
            passes: l.usize("num_loops")?,
        })
    } else if l.header().arch_key("loop.start").is_some() {
        Some(Core {
            start: l.usize("loop.start")?,
            end: l.usize("loop.end")?,
            passes: l.usize("loop.count")?,
        })
    } else {
        None
    };
    let n_phys = match &core {
        Some(c) if unrolled => (c.end > c.start && c.passes > 0)
            .then(|| listed.checked_sub((c.passes - 1) * (c.end - c.start)))
            .flatten()
            .ok_or_else(|| LoadError("Inkling 2 loop is out of range".into()))?,
        _ => listed,
    };
    if let Some(c) = &core {
        let varying = match l.hparam("loop.varying") {
            Ok(v) => v
                .as_bool()
                .ok_or_else(|| LoadError("loop.varying is not a boolean".into()))?,
            Err(_) => false,
        };
        if c.passes < if varying { 1 } else { 2 }
            || c.start == 0
            || c.start >= c.end
            || c.end > n_phys
            || !c.start.is_multiple_of(block)
            || !c.end.is_multiple_of(block)
        {
            return Err(LoadError("Inkling 2 loop is out of range".into()));
        }
    }
    // An unrolled array lists the later passes of the core after the
    // physical layers; they must repeat the core.
    let physical = |key: &str, values: Vec<Value>| -> Result<Vec<Value>, LoadError> {
        if let Some(c) = core.as_ref().filter(|_| unrolled) {
            let span = c.end - c.start;
            if (n_phys..listed).any(|s| values[s] != values[c.start + (s - n_phys) % span]) {
                return Err(LoadError(format!(
                    "{key} differs between passes of the loop"
                )));
            }
        }
        Ok(values[..n_phys].to_vec())
    };
    let per_layer = |key: &str| -> Result<Vec<u64>, LoadError> {
        let values = match l.hparam(key)? {
            Value::Array(_) => l.array(key, listed, |v| Some(v.clone()))?,
            v => vec![v.clone(); listed],
        };
        physical(key, values)?
            .iter()
            .map(|v| {
                v.as_u64()
                    .ok_or_else(|| LoadError(format!("{key} is not an integer")))
            })
            .collect()
    };
    let n_head = per_layer("attention.head_count")?;
    let n_head_kv = per_layer("attention.head_count_kv")?;
    let hd = l.usize("attention.key_length")?;
    let widths = [
        l.usize("attention.value_length")?,
        l.usize_or("attention.key_length_swa", hd)?,
        l.usize_or("attention.value_length_swa", hd)?,
    ];
    if widths.iter().any(|&w| w != hd) {
        return Err(LoadError(
            "Inkling 2 with heads of different widths is not supported yet".into(),
        ));
    }
    if l.usize_or("rope.dimension_count", 0)? != 0 {
        return Err(LoadError(
            "Inkling 2 with rotary positions is not supported yet".into(),
        ));
    }
    let key = "attention.sliding_window_pattern";
    let windowed = physical(key, l.array(key, listed, |v| Some(v.clone()))?)?
        .iter()
        .map(|v| {
            v.as_bool()
                .ok_or_else(|| LoadError(format!("{key} has an invalid value")))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let n_vocab = l.usize("vocab_size")?;
    let unpadded = l.usize("vocab_size_unpadded")?;
    let p = Params {
        d,
        hd,
        window: l.usize("attention.sliding_window")?,
        taps: l.usize("shortconv.l_cache")?,
        n_rel: l.usize("attention.relative_length")?,
        extent: l.usize("attention.relative_extent")?,
        qk_clip: l.f32("attention.qk_clip")?,
        rel_clip: l.f32("attention.relative_clip")?,
        log_scale: (
            l.usize("attention.log_scale_floor")? as f32,
            l.f32("attention.log_scale_alpha")?,
        ),
        n_experts: l.usize("expert_count")?,
        n_used: l.usize("expert_used_count")?,
        n_ff: l.usize("expert_feed_forward_length")?,
        glu: Glu::Situ {
            beta: l.f32("activation.situ_beta")?,
            linear: l.f32("activation.situ_linear_beta")?,
        },
        weight_scale: l.f32_or("expert_weights_scale", 1.0)?,
        eps: b.eps,
    };
    let positive = [p.qk_clip, p.rel_clip, p.eps]
        .iter()
        .all(|v| v.is_finite() && *v > 0.0);
    if !positive
        || !p.log_scale.1.is_finite()
        || p.log_scale.1 < 0.0
        || p.log_scale.0 == 0.0
        || !(2..=64).contains(&p.taps)
        || p.window == 0
        || p.n_rel == 0
        || p.extent == 0
        || block == 0
        || unpadded == 0
        || unpadded > n_vocab
        || p.n_used == 0
        || p.n_used > p.n_experts
    {
        return Err(LoadError(
            "Inkling 2 hyperparameters are out of range".into(),
        ));
    }

    let rank = core
        .as_ref()
        .map(|_| l.usize("loop.gate_rank"))
        .transpose()?;

    let layer = |i: usize| -> Result<Layer<'a>, LoadError> {
        let (nh, kv) = (n_head[i] as usize, n_head_kv[i] as usize);
        if kv == 0 || !nh.is_multiple_of(kv) {
            return Err(LoadError(format!(
                "layer {i}: heads do not split into groups"
            )));
        }
        p.layer(l, i, nh, kv, windowed[i])
    };
    let mut layers = (0..n_phys).map(layer).collect::<Result<Vec<_>, _>>()?;
    if let Some(c) = &core {
        for _ in 1..c.passes {
            for i in c.start..c.end {
                layers.push(layer(i)?);
            }
        }
    }

    // Unsloth keeps the message queries of the whole core in two tables of
    // two queries per layer: the query and feed-forward ones, and the key
    // and value ones.
    let unrolled_queries = match core.as_ref().filter(|_| unrolled) {
        Some(c) => {
            let n = 2 * (c.end - c.start);
            Some((
                table(l, "loop_res_query.weight", d, n)?,
                table(l, "loop_kv_res_query.weight", d, n)?,
            ))
        }
        None => None,
    };
    let wiring = (0..n_phys)
        .map(|i| {
            let t = |s: &str| format!("blk.{i}.{s}");
            let attn_query = match (i, windowed[i]) {
                (0, _) => None,
                (_, true) => Some(l.vector(&t("attn_res_query.weight"), d)?),
                (_, false) => Some(table(l, &t("attn_res_query.weight"), d, 3)?),
            };
            let looped = match (&core, rank) {
                (Some(c), Some(rank)) if (c.start..c.end).contains(&i) => {
                    let name = |x: usize, part: &str| {
                        if unrolled {
                            t(&format!("loop_{}_{part}.weight", UNROLLED_CONTROLS[x]))
                        } else {
                            t(&format!("loop_{}_ctrl_{part}.weight", CONTROLS[x]))
                        }
                    };
                    let control = |x: usize| -> Result<Control<'a>, LoadError> {
                        Ok(Control {
                            down: l.matrix(&name(x, "down"), 2 * d, rank)?,
                            gain: l.matrix(&name(x, "gain"), rank, d)?,
                            forget: l.matrix(&name(x, "forget"), rank, d)?,
                        })
                    };
                    let query = match &unrolled_queries {
                        Some((res, kv)) => {
                            let j = i - c.start;
                            let row = |table: &[f32], r: usize| table[r * d..][..d].to_vec();
                            [
                                row(res, 2 * j),
                                row(kv, 2 * j),
                                row(kv, 2 * j + 1),
                                row(res, 2 * j + 1),
                            ]
                            .concat()
                        }
                        None => table(l, &t("loop_msg_query.weight"), d, 4)?,
                    };
                    Some((query, [control(0)?, control(1)?, control(2)?, control(3)?]))
                }
                _ => None,
            };
            Ok(Wiring {
                attn_query,
                ffn_query: l.vector(&t("ffn_res_query.weight"), d)?,
                looped,
            })
        })
        .collect::<Result<Vec<_>, LoadError>>()?;

    let network = Network {
        d,
        eps: b.eps,
        embd_norm: l.vector("token_embd_norm.weight", d)?,
        out_query: l.vector("output_res_query.weight", d)?,
        block,
        wiring,
        core,
    };
    l.check_all_read(SKIPPED)?;
    Ok(Stack {
        network: Some(network),
        ..layers.into()
    })
}

/// A small f32 tensor of shape `[cols, rows]`, row by row.
fn table(l: &Loader, name: &str, cols: usize, rows: usize) -> Result<Vec<f32>, LoadError> {
    l.matrix(name, cols, rows)?;
    l.vector(name, cols * rows)
}

/// The hyperparameters every layer shares.
struct Params {
    d: usize,
    hd: usize,
    window: usize,
    taps: usize,
    n_rel: usize,
    extent: usize,
    qk_clip: f32,
    rel_clip: f32,
    log_scale: (f32, f32),
    n_experts: usize,
    n_used: usize,
    n_ff: usize,
    glu: Glu,
    weight_scale: f32,
    eps: f32,
}

impl Params {
    fn layer<'a>(
        &self,
        l: &Loader<'a>,
        i: usize,
        nh: usize,
        kv: usize,
        windowed: bool,
    ) -> Result<Layer<'a>, LoadError> {
        let t = |s: &str| format!("blk.{i}.{s}");
        let (d, hd, taps) = (self.d, self.hd, self.taps);
        let norms = if windowed {
            let n = l.vector(&t("attn_norm.weight"), d)?;
            [n.clone(), n.clone(), n]
        } else {
            [
                l.vector(&t("attn_q_inp_norm.weight"), d)?,
                l.vector(&t("attn_k_inp_norm.weight"), d)?,
                l.vector(&t("attn_v_inp_norm.weight"), d)?,
            ]
        };
        let dynamic = if windowed {
            None
        } else {
            Some([
                l.matrix(&t("attn_k_conv_proj.weight"), d, taps * kv)?,
                l.matrix(&t("attn_v_conv_proj.weight"), d, taps * kv)?,
            ])
        };
        let extent = if windowed { self.window } else { self.extent };
        let attention = RelAttention {
            norms,
            q: l.matrix(&t("attn_q.weight"), d, nh * hd)?,
            k: l.matrix(&t("attn_k.weight"), d, kv * hd)?,
            v: l.matrix(&t("attn_v.weight"), d, kv * hd)?,
            o: l.matrix(&t("attn_output.weight"), nh * hd, d)?,
            gate: l.matrix(&t("attn_gate.weight"), d, nh * hd)?,
            sink: l.matrix(&t("attn_sink_proj.weight"), d, nh)?,
            q_scale: table(l, &t("attn_q_clip.weight"), hd, nh)?,
            rel: l.matrix(&t("attn_rel.weight"), d, self.n_rel * nh)?,
            rel_scale: table(l, &t("attn_rel_clip.weight"), self.n_rel, nh)?,
            rel_bias: table(l, &t("attn_rel_bias.weight"), self.n_rel, extent)?,
            n_rel: self.n_rel,
            convs: [
                table(l, &t("attn_k_conv.weight"), taps, kv * hd)?,
                table(l, &t("attn_v_conv.weight"), taps, kv * hd)?,
                table(l, &t("attn_post_conv.weight"), taps, d)?,
                table(l, &t("ffn_post_conv.weight"), taps, d)?,
            ],
            taps,
            dynamic,
            n_head: nh,
            n_head_kv: kv,
            head_dim: hd,
            window: windowed.then_some(self.window),
            qk_clip: self.qk_clip,
            rel_clip: self.rel_clip,
            log_scale: (!windowed).then_some(self.log_scale),
            eps: self.eps,
        };
        let (gate, up) = l
            .experts(
                &t("ffn_gate_up_exps.weight"),
                d,
                2 * self.n_ff,
                self.n_experts,
            )?
            .halves();
        let experts = Moe {
            router: l.matrix(&t("ffn_gate_inp.weight"), d, self.n_experts)?,
            gate,
            up,
            down: l.experts(&t("ffn_down_exps.weight"), self.n_ff, d, self.n_experts)?,
            n_used: self.n_used,
            shared_gate: None,
            gating: Gating::Sigmoid {
                bias: l
                    .optional_vector(&t("exp_probs_b.bias"), self.n_experts)?
                    .unwrap_or_else(|| vec![0.0; self.n_experts]),
            },
            glu: self.glu,
            weight_scale: self.weight_scale,
            scales: {
                let gate_up = l.optional_vector(&t("ffn_gate_up_exps.scale"), self.n_experts)?;
                let down = l.optional_vector(&t("ffn_down_exps.scale"), self.n_experts)?;
                [gate_up.clone(), gate_up, down]
            },
            norms: None,
        };
        Ok(Layer {
            mixer_norm: None,
            mixer: Mixer::Relative(Box::new(attention)),
            mixer_out_norm: None,
            ffn_norm: Some(l.vector(&t("ffn_norm.weight"), d)?),
            ffn: None,
            experts: Some(experts),
            ffn_out_norm: None,
            residual_scale: None,
            output_scale: None,
        })
    }
}

impl Network<'_> {
    /// Runs the token embeddings `x` of a batch from position `pos0`
    /// through `layers`, the physical ones and then those of every later
    /// pass of the core, and returns the input of the output norm.
    pub(crate) fn forward(
        &self,
        layers: &[Layer],
        x: Vec<f32>,
        pos0: usize,
        tree: Option<&Tree>,
        states: &mut [MixerState],
    ) -> Vec<f32> {
        let mut cur = x;
        let d = self.d;
        pool_norm(&mut cur, &self.embd_norm, d, self.eps);
        let mut blocks: Vec<Vec<f32>> = Vec::new();
        let mut run =
            |slot: usize, ip: usize, cur: Vec<f32>, blocks: &mut Vec<Vec<f32>>, message| {
                self.layer(
                    &layers[slot],
                    ip,
                    cur,
                    blocks,
                    message,
                    pos0,
                    tree,
                    &mut states[slot],
                )
            };
        let (start, end) = self.core.as_ref().map_or((0, 0), |c| (c.start, c.end));
        for i in 0..start {
            cur = run(i, i, cur, &mut blocks, None);
        }
        if let Some(c) = &self.core {
            let prelude = cur.clone();
            let prelude_blocks = blocks.clone();
            for pass in 0..c.passes {
                if pass == 0 {
                    for i in c.start..c.end {
                        cur = run(i, i, cur, &mut blocks, None);
                    }
                    continue;
                }
                // The previous pass: its blocks after the first of the core,
                // and its output.
                let mut bank = blocks.split_off(prelude_blocks.len() + 1);
                bank.push(std::mem::replace(&mut cur, prelude.clone()));
                blocks = prelude_blocks.clone();
                let bank: Vec<&[f32]> = bank.iter().map(|b| &b[..]).collect();
                for i in c.start..c.end {
                    let (query, _) = self.wiring[i].looped.as_ref().expect("core layers loop");
                    let messages = mix(&bank, query, d);
                    let slot = self.wiring.len() + (pass - 1) * (c.end - c.start) + i - c.start;
                    cur = run(slot, i, cur, &mut blocks, Some(messages));
                }
            }
        }
        for i in end.max(start)..self.wiring.len() {
            cur = run(i, i, cur, &mut blocks, None);
        }
        mix_with(&blocks, &cur, &self.out_query, d).remove(0)
    }

    /// One layer on cache slot `layer` of physical layer `ip`, with the
    /// messages of the previous pass in a looped layer.
    #[allow(clippy::too_many_arguments)]
    fn layer(
        &self,
        layer: &Layer,
        ip: usize,
        cur: Vec<f32>,
        blocks: &mut Vec<Vec<f32>>,
        messages: Option<Vec<Vec<f32>>>,
        pos0: usize,
        tree: Option<&Tree>,
        state: &mut MixerState,
    ) -> Vec<f32> {
        let d = self.d;
        let (Mixer::Relative(attention), MixerState::Relative(state)) = (&layer.mixer, state)
        else {
            unreachable!("Inkling 2 layers attend by relative distance")
        };
        let wiring = &self.wiring[ip];
        let mut inputs: [Vec<f32>; 3] = match &wiring.attn_query {
            None => [cur.clone(), cur.clone(), cur.clone()],
            Some(query) => {
                let mut mixed = mix_with(blocks, &cur, query, d);
                match mixed.len() {
                    1 => [mixed[0].clone(), mixed[0].clone(), mixed.remove(0)],
                    _ => std::array::from_fn(|j| std::mem::take(&mut mixed[j])),
                }
            }
        };
        let controls = wiring.looped.as_ref().map(|(_, c)| c);
        if let (Some(messages), Some(controls)) = (&messages, controls) {
            for j in 0..3 {
                controls[j].apply(&mut inputs[j], &messages[j]);
            }
        }
        let out = attention.forward([&inputs[0], &inputs[1], &inputs[2]], pos0, tree, state);

        // The first layer of each block starts from the attention output
        // alone and hands its input to the blocks the later layers mix.
        let ffn_in = if ip.is_multiple_of(self.block) {
            blocks.push(cur);
            out
        } else {
            add(cur, &out)
        };
        let mut h = mix_with(blocks, &ffn_in, &wiring.ffn_query, d).remove(0);
        if let (Some(messages), Some(controls)) = (&messages, controls) {
            controls[3].apply(&mut h, &messages[3]);
        }
        let norm = layer
            .ffn_norm
            .as_ref()
            .expect("Inkling 2 layers norm their experts' input");
        pool_norm(&mut h, norm, d, self.eps);
        let mut y = vec![0.0; h.len()];
        layer
            .experts
            .as_ref()
            .expect("Inkling 2 layers have experts")
            .forward(&h, &mut y);
        add(ffn_in, &attention.ffn_conv(&y, tree, state))
    }
}

/// [`mix`] of the earlier blocks and `cur`.
fn mix_with(blocks: &[Vec<f32>], cur: &[f32], queries: &[f32], d: usize) -> Vec<Vec<f32>> {
    let states: Vec<&[f32]> = blocks.iter().map(|b| &b[..]).chain([cur]).collect();
    mix(&states, queries, d)
}

fn add(mut x: Vec<f32>, y: &[f32]) -> Vec<f32> {
    x.iter_mut().zip(y).for_each(|(x, y)| *x += y);
    x
}

fn pool_norm(x: &mut [f32], w: &[f32], d: usize, eps: f32) {
    kinetic_cpu::pool::for_each_chunk(x, d, |_, row| kinetic_cpu::rms_norm(row, w, eps));
}
