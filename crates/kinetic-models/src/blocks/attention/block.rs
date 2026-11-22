use kinetic_cpu::{axpy_rows, dot, matmul, rms_norm, softmax};
use rayon::prelude::*;

use super::{Attention, Gate, KvCache};

impl Attention<'_> {
    /// A block of tokens from position `pos0` in which every token attends
    /// to every entry of `cache` and to every token of the block, as the
    /// rows of a block drafter do. Nothing of the block is stored.
    pub(crate) fn forward_block(&self, h: &[f32], pos0: usize, cache: &KvCache) -> Vec<f32> {
        assert!(
            matches!(self.gate, Gate::None) && self.folding.is_none(),
            "gated or folded block attention"
        );
        let hd = self.head_dim;
        let n = h.len() / self.q.cols;
        let (q_dim, kv_dim) = (self.q.rows, self.n_head_kv * hd);
        let mut p = self.project(h).into_iter();
        let (mut q, mut k, v) = (
            p.next().expect("queries"),
            p.next().expect("keys"),
            p.next().expect("values"),
        );
        q.par_chunks_mut(q_dim)
            .zip(k.par_chunks_mut(kv_dim))
            .enumerate()
            .for_each(|(t, (q, k))| self.prepare(q, Some(k), &self.rope.angles(pos0 + t)));

        let scale = self.scale.unwrap_or(1.0 / (hd as f32).sqrt());
        let group = self.n_head / self.n_head_kv;
        let stored = cache.k.len() / kv_dim;
        let mut by_head = vec![0.0; self.n_head * n * hd];
        by_head.par_chunks_mut(n * hd).enumerate().for_each_init(
            Vec::new,
            |scores, (head, out)| {
                let kv = head / group * hd;
                for (t, out) in out.chunks_exact_mut(hd).enumerate() {
                    let qt = &q[t * q_dim + head * hd..][..hd];
                    scores.clear();
                    scores.extend(
                        (0..stored).map(|j| dot(&cache.k[j * kv_dim + kv..][..hd], qt) * scale),
                    );
                    scores.extend((0..n).map(|j| dot(&k[j * kv_dim + kv..][..hd], qt) * scale));
                    softmax(scores);
                    if stored > 0 {
                        axpy_rows(out, &scores[..stored], &cache.v[kv..], kv_dim);
                    }
                    axpy_rows(out, &scores[stored..], &v[kv..], kv_dim);
                }
            },
        );
        let mut mixed = vec![0.0; n * q_dim];
        for (head, rows) in by_head.chunks_exact(n * hd).enumerate() {
            for (t, src) in rows.chunks_exact(hd).enumerate() {
                mixed[t * q_dim + head * hd..][..hd].copy_from_slice(src);
            }
        }
        let mut y = vec![0.0; n * self.o.rows];
        matmul(&self.o, &mixed, &mut y);
        y
    }

    /// Stores keys and values made outside the layer for the positions
    /// from `pos0`: each key head is normalized with `norm` and rotated, as
    /// the layer's own keys are.
    pub(crate) fn store_keys(
        &self,
        mut k: Vec<f32>,
        v: &[f32],
        norm: &[f32],
        pos0: usize,
        cache: &mut KvCache,
    ) {
        let kv_dim = self.n_head_kv * self.head_dim;
        for (t, (k, v)) in k
            .chunks_exact_mut(kv_dim)
            .zip(v.chunks_exact(kv_dim))
            .enumerate()
        {
            let angles = self.rope.angles(pos0 + t);
            for head in k.chunks_exact_mut(self.head_dim) {
                rms_norm(head, norm, self.eps);
                self.rope.rotate(head, &angles);
            }
            cache.store(self.window, pos0 + t, k, v);
        }
    }
}
