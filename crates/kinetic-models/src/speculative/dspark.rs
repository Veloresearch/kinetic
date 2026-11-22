use super::Companion;
use crate::{Dspark, DsparkState};

/// A [`Dspark`] drafter and the sequence it drafts for.
pub struct DsparkCompanion<'a> {
    dspark: Dspark<'a>,
    state: DsparkState,
}

impl<'a> DsparkCompanion<'a> {
    pub fn new(dspark: Dspark<'a>) -> Self {
        DsparkCompanion {
            state: dspark.new_state(),
            dspark,
        }
    }
}

impl Companion for DsparkCompanion<'_> {
    fn target_layers(&self) -> &[usize] {
        self.dspark.target_layers()
    }

    fn n_embd(&self) -> usize {
        self.dspark.n_embd()
    }

    fn vocab(&self) -> usize {
        self.dspark.vocab()
    }

    fn max_draft(&self) -> usize {
        self.dspark.max_draft()
    }

    fn committed(&self) -> usize {
        self.state.len()
    }

    fn commit(&mut self, features: &[f32]) {
        self.dspark.commit(&mut self.state, features);
    }

    fn draft(
        &self,
        anchor: u32,
        embedding: &[f32],
        n: usize,
        head: &dyn Fn(&[f32]) -> Vec<f32>,
    ) -> Vec<(u32, f32)> {
        self.dspark.draft(&self.state, anchor, embedding, n, head)
    }

    fn reset(&mut self) {
        self.state = self.dspark.new_state();
    }
}
