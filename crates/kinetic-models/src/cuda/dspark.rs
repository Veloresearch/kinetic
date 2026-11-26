//! A DSpark drafter with a fusion layer on the device; see [`FusedDspark`].
//!
//! Committing runs the fusion layer over the target's tapped hidden states
//! and stores, in each drafter layer, the keys and values projected from
//! it, through the attention kernel's own store. A draft runs the block of
//! the anchor and mask embeddings through the layers with every row seeing
//! the whole block ([`kinetic_cuda::Batch::set_block`]) and the output
//! head over all its rows at once, then chains the rows: each adds the
//! Markov bias of the token chosen at the row before and takes its best
//! token.

use std::sync::Arc;

use kinetic_cuda::{DeviceBuffer, DeviceMatrix, Flow, GemvInput, RmsNorm, MAX_BATCH};
use kinetic_quant::dequantize_row;

use super::layer::{CudaLayer, CudaMixer, LayerState};
use super::scratch::Work;
use super::{cuda_error, CudaModel};
use crate::speculative::{Drafter, Lookup};
use crate::{FusedDspark, LoadError};

/// A [`FusedDspark`] uploaded for one [`CudaModel`]; slots share it.
pub struct CudaDspark<'a> {
    host: FusedDspark<'a>,
    layers: Vec<CudaLayer>,
    fc: DeviceMatrix,
    fc_norm: DeviceBuffer<f32>,
    output_norm: DeviceBuffer<f32>,
    /// The output head over the drafter's vocabulary.
    head: DeviceMatrix,
    /// From a Markov state to a bias on each token of that vocabulary.
    markov: DeviceMatrix,
    n_draft: usize,
}

/// A sequence's drafter caches and buffers on the device.
pub struct CudaDsparkState {
    work: Work,
    /// The Markov state of the row being chosen.
    markov_in: GemvInput,
    /// The head's logits for every row of the block, and one row's.
    block_logits: DeviceBuffer<f32>,
    row_logits: DeviceBuffer<f32>,
    len: usize,
}

impl CudaDspark<'_> {
    /// The target layers whose inputs the drafter reads, those below
    /// `n_layers`, the target's count, which stands for its final state.
    pub fn taps(&self, n_layers: usize) -> Vec<usize> {
        let mut taps: Vec<usize> = self
            .host
            .target_layers()
            .iter()
            .copied()
            .filter(|&l| l < n_layers)
            .collect();
        taps.dedup();
        taps
    }

    /// Device bytes of the weights.
    pub fn size_bytes(&self) -> usize {
        let layers: usize = self.layers.iter().map(CudaLayer::size_bytes).sum();
        layers + self.fc.size_bytes() + self.head.size_bytes() + self.markov.size_bytes()
    }
}

impl<'m> CudaModel<'m> {
    /// Uploads `dspark` for drafting for this model. Fails when it was not
    /// made for the model or has no output head of its own.
    pub fn upload_dspark<'a>(&self, dspark: FusedDspark<'a>) -> Result<CudaDspark<'a>, LoadError> {
        if dspark.n_embd() != self.n_embd || dspark.vocab() != self.n_vocab() {
            return Err(LoadError("the drafter was not made for this model".into()));
        }
        let gpu = &self.gpu;
        let output = dspark.output.as_ref().ok_or_else(|| {
            LoadError("a DSpark drafter on CUDA needs its own output head".into())
        })?;
        let layers = dspark
            .layers
            .iter()
            .map(|l| CudaLayer::upload(gpu, l, None, dspark.eps))
            .collect::<Result<Vec<_>, _>>()?;
        if layers
            .iter()
            .any(|l| !matches!(l.mixer, CudaMixer::Attention(_)))
        {
            return Err(LoadError("DSpark layers attend".into()));
        }
        Ok(CudaDspark {
            fc: gpu.matrix(&dspark.fc).map_err(cuda_error)?,
            fc_norm: gpu.vector(&dspark.fc_norm).map_err(cuda_error)?,
            output_norm: gpu.vector(&dspark.output_norm).map_err(cuda_error)?,
            head: gpu.matrix(output).map_err(cuda_error)?,
            markov: gpu.matrix(&dspark.markov_out).map_err(cuda_error)?,
            n_draft: output.rows,
            layers,
            host: dspark,
        })
    }

    /// An empty drafter state for a sequence of up to the model's context.
    pub fn new_dspark_state(&self, dspark: &CudaDspark) -> kinetic_cuda::Result<CudaDsparkState> {
        let layers: Vec<&CudaLayer> = dspark.layers.iter().collect();
        let d = self.n_embd;
        Ok(CudaDsparkState {
            work: Work::new(
                &self.gpu,
                &layers,
                d,
                self.context,
                MAX_BATCH,
                dspark.n_draft,
                dspark.fc.cols(),
                false,
            )?,
            markov_in: GemvInput::new(&self.gpu.ctx, dspark.host.rank)?,
            block_logits: DeviceBuffer::new(
                &self.gpu.ctx,
                dspark.host.max_draft() * dspark.n_draft,
            )?,
            row_logits: DeviceBuffer::new(&self.gpu.ctx, dspark.n_draft)?,
            len: 0,
        })
    }

    /// Commits the positions after those `state` holds: per position, the
    /// target's hidden states at the drafter's target layers, in its order.
    pub fn dspark_commit(
        &self,
        dspark: &CudaDspark,
        state: &mut CudaDsparkState,
        features: &[f32],
    ) -> kinetic_cuda::Result<()> {
        let (k, stream) = (&self.gpu.kernels, &self.gpu.stream);
        let (d, width) = (self.n_embd, dspark.fc.cols());
        let w = &mut state.work;
        w.batch.set_block(false);
        for chunk in features.chunks(MAX_BATCH * width) {
            let n = chunk.len() / width;
            let s = &mut w.scratch;
            let joined = s
                .joined
                .as_mut()
                .expect("drafter scratch has a fusion input");
            joined.upload(k, chunk, stream)?;
            k.gemv_rows(&dspark.fc, d, joined, 0..n, &mut s.y, stream)?;
            let norm = RmsNorm {
                weight: &dspark.fc_norm,
                eps: dspark.host.eps,
            };
            k.rms_norm(&s.y, norm, &mut s.h, 0, n, stream)?;
            // the token ids are not read: only positions matter here
            w.batch.set(state.len, &vec![0; n], stream)?;
            for (layer, cache) in dspark.layers.iter().zip(&mut w.layers) {
                let (CudaMixer::Attention(a), LayerState::Attention(cache)) = (&layer.mixer, cache)
                else {
                    unreachable!("DSpark layers attend")
                };
                // the queries attend to nothing anyone reads; only the
                // stored keys and values count
                a.project(&self.gpu, s, n)?;
                a.mix(&self.gpu, s, cache, &w.batch)?;
            }
            state.len += n;
        }
        Ok(())
    }

    /// Up to `n` tokens after `anchor`, at the position after those `state`
    /// holds, each with its chance; see [`FusedDspark::draft`].
    pub fn dspark_draft(
        &self,
        dspark: &CudaDspark,
        state: &mut CudaDsparkState,
        anchor: u32,
        n: usize,
    ) -> kinetic_cuda::Result<Vec<(u32, f32)>> {
        let host = &dspark.host;
        let n = n.min(host.max_draft());
        if n == 0 || state.len == 0 {
            return Ok(Vec::new());
        }
        let (k, stream) = (&self.gpu.kernels, &self.gpu.stream);
        let d = self.n_embd;
        let skip = usize::from(!host.from_anchor);
        let mut tokens = vec![host.mask_token; n + skip];
        tokens[0] = anchor;
        let m = tokens.len();
        let w = &mut state.work;
        w.batch.set(state.len, &tokens, stream)?;
        w.batch.set_block(true);
        let s = &mut w.scratch;
        self.embed_rows(&w.batch, &tokens, &mut s.x)?;
        dspark.layers[0].input(&self.gpu, &s.x, &mut s.h, m)?;
        let run = (0..dspark.layers.len()).try_for_each(|i| {
            let next = match dspark.layers.get(i + 1) {
                Some(l) => l.mixer_norm.as_ref(),
                None => Some(&dspark.output_norm),
            };
            dspark.layers[i].forward(
                &self.gpu,
                s,
                &mut w.layers[i],
                Flow::Commit,
                &w.batch,
                next,
                None,
            )
        });
        w.batch.set_block(false);
        run?;
        let v = dspark.n_draft;
        // the rows' own logits do not depend on the tokens chosen before
        // them, so the head reads its weights once for the whole block
        k.gemv_rows(
            &dspark.head,
            v,
            &s.h,
            skip..m,
            &mut state.block_logits,
            stream,
        )?;
        let mut hidden = vec![0.0; n * d];
        if host.confidence.is_some() {
            s.h.values().copy_range_to(skip * d, &mut hidden)?;
        }

        let mut markov = vec![0.0; host.rank];
        let mut draft = Vec::with_capacity(n);
        let mut prev = anchor;
        for (i, row) in hidden.chunks_exact(d).enumerate() {
            dequantize_row(
                host.markov_in.ty,
                host.markov_in.data,
                prev as usize,
                &mut markov,
            )
            .map_err(|e| super::shape(format!("Markov table: {e}")))?;
            state.markov_in.upload(k, &markov, stream)?;
            k.gemv_rows(
                &dspark.markov,
                v,
                &state.markov_in,
                0..1,
                &mut s.logits,
                stream,
            )?;
            state.row_logits.copy_from_device(
                0,
                &state.block_logits,
                i * v..(i + 1) * v,
                stream,
            )?;
            k.add(&mut s.logits, &state.row_logits, v, stream)?;
            k.argmax(&s.logits, v, 1, &mut s.greedy, stream)?;
            let mut best = [0];
            s.greedy.best().copy_range_to(0, &mut best)?;
            let best = best[0] as usize;
            let token = host.d2t.as_ref().map_or(best as u32, |d2t| d2t[best]);
            let chance = match &host.confidence {
                Some((weights, bias)) => {
                    let input = row.iter().chain(&markov);
                    let dot: f32 = weights.iter().zip(input).map(|(a, b)| a * b).sum();
                    1.0 / (1.0 + (-(dot + bias)).exp())
                }
                None => {
                    let mut logits = vec![0.0; dspark.n_draft];
                    s.logits.copy_range_to(0, &mut logits)?;
                    let top = logits[best];
                    1.0 / logits.iter().map(|l| (l - top).exp()).sum::<f32>()
                }
            };
            draft.push((token, chance));
            prev = token;
        }
        Ok(draft)
    }

    /// Empties `state` for a new sequence; the caches are overwritten as
    /// positions are committed again.
    pub fn clear_dspark_state(&self, state: &mut CudaDsparkState) {
        state.len = 0;
    }
}

fn device<T>(r: kinetic_cuda::Result<T>) -> T {
    r.unwrap_or_else(|e| panic!("CUDA: {e}"))
}

/// Drafts with a [`CudaDspark`] for a [`CudaModel`] whose taps are the
/// drafter's ([`CudaModel::set_taps`]), like
/// [`CompanionDrafter`](crate::speculative::CompanionDrafter) does on the
/// CPU: a context lookup first, then the drafter, from the tokens it saw
/// since its last reset.
pub struct CudaDsparkDrafter<'a> {
    dspark: Arc<CudaDspark<'a>>,
    state: CudaDsparkState,
    /// Where each of the drafter's target layers sits in a row of the
    /// target's hidden states, in units of its width.
    slots: Vec<usize>,
    /// The last token committed since the last reset.
    last: Option<u32>,
    pub lookup: Option<Lookup>,
}

impl<'a> CudaDsparkDrafter<'a> {
    /// A drafter for sequences of `model`, which must return the hidden
    /// states `dspark` reads.
    pub fn new(
        model: &CudaModel,
        dspark: Arc<CudaDspark<'a>>,
        n_layers: usize,
    ) -> Result<Self, LoadError> {
        let taps = dspark.taps(n_layers);
        if model.taps() != taps {
            return Err(LoadError(
                "the model does not return the hidden states the drafter reads".into(),
            ));
        }
        let slots = dspark
            .host
            .target_layers()
            .iter()
            .map(|l| taps.iter().position(|t| t == l).map_or(0, |i| i + 1))
            .collect();
        let state = model.new_dspark_state(&dspark).map_err(cuda_error)?;
        Ok(CudaDsparkDrafter {
            dspark,
            state,
            slots,
            last: None,
            lookup: Some(Lookup {
                max_ngram: 4,
                min_ngram: 3,
            }),
        })
    }

    fn draft(&mut self, model: &CudaModel, context: &[u32], limit: usize) -> Vec<(u32, f32)> {
        let follows = context.len() >= 2 && self.last == Some(context[context.len() - 2]);
        match context.last() {
            Some(&anchor) if self.state.len > 0 && self.state.len < context.len() && follows => {
                device(model.dspark_draft(&self.dspark, &mut self.state, anchor, limit))
            }
            _ => Vec::new(),
        }
    }
}

impl<'m> Drafter<CudaModel<'m>> for CudaDsparkDrafter<'_> {
    fn observe(&mut self, model: &CudaModel<'m>, tokens: &[u32], hidden: &[f32]) {
        if tokens.is_empty() {
            return;
        }
        let d = model.n_embd();
        let width = hidden.len() / tokens.len();
        let features: Vec<f32> = hidden
            .chunks_exact(width)
            .flat_map(|row| self.slots.iter().flat_map(move |&s| &row[s * d..][..d]))
            .copied()
            .collect();
        device(model.dspark_commit(&self.dspark, &mut self.state, &features));
        self.last = tokens.last().copied();
    }

    fn propose(&mut self, model: &CudaModel<'m>, context: &[u32], limit: usize) -> Vec<u32> {
        if let Some(l) = self.lookup.as_mut() {
            let copied = Drafter::<CudaModel>::propose(l, model, context, limit);
            if !copied.is_empty() {
                return copied;
            }
        }
        self.draft(model, context, limit)
            .into_iter()
            .map(|(t, _)| t)
            .collect()
    }

    fn reset(&mut self, model: &CudaModel<'m>) {
        model.clear_dspark_state(&mut self.state);
        self.last = None;
    }
}
