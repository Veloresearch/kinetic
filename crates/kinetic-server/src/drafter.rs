//! The drafts every backend decodes with.

use kinetic_models::speculative::{
    Decoder, Draft, Drafter, Ensemble, Lookup, MtpDrafter, Predictor, SessionIndex,
};

/// One of the drafters a backend lets compete.
pub enum Source<'a, M: Predictor> {
    /// Repeats from the context.
    Lookup(Lookup),
    /// Repeats from everything earlier requests saw.
    Session(SessionIndex),
    /// The model's own prediction block.
    Mtp(MtpDrafter<M>),
    /// A drafter from its own file, such as a DSpark drafter.
    Extra(Box<dyn Drafter<M> + Send + 'a>),
}

impl<M: Predictor> Drafter<M> for Source<'_, M> {
    fn observe(&mut self, model: &M, tokens: &[u32], hidden: &[f32]) {
        match self {
            Source::Lookup(d) => Drafter::<M>::observe(d, model, tokens, hidden),
            Source::Session(d) => Drafter::<M>::observe(d, model, tokens, hidden),
            Source::Mtp(d) => d.observe(model, tokens, hidden),
            Source::Extra(d) => d.observe(model, tokens, hidden),
        }
    }

    fn propose(&mut self, model: &M, context: &[u32], limit: usize) -> Vec<u32> {
        match self {
            Source::Lookup(d) => Drafter::<M>::propose(d, model, context, limit),
            Source::Session(d) => Drafter::<M>::propose(d, model, context, limit),
            Source::Mtp(d) => d.propose(model, context, limit),
            Source::Extra(d) => d.propose(model, context, limit),
        }
    }

    fn propose_tree(&mut self, model: &M, context: &[u32], limit: usize) -> Draft {
        match self {
            Source::Extra(d) => d.propose_tree(model, context, limit),
            _ => Draft::chain(self.propose(model, context, limit)),
        }
    }

    fn reset(&mut self, model: &M) {
        match self {
            Source::Lookup(d) => Drafter::<M>::reset(d, model),
            Source::Session(d) => Drafter::<M>::reset(d, model),
            Source::Mtp(d) => d.reset(model),
            Source::Extra(d) => d.reset(model),
        }
    }
}

/// A decoder whose drafts come from the context, from everything earlier
/// requests saw and from the model's prediction block when it has one,
/// whichever has lately been right most.
pub fn decoder<'a, M: Predictor>(model: &M) -> Decoder<Ensemble<Source<'a, M>>> {
    decoder_with(model, None)
}

/// [`decoder`] with `extra` competing too.
pub fn decoder_with<'a, M: Predictor>(
    model: &M,
    extra: Option<Box<dyn Drafter<M> + Send + 'a>>,
) -> Decoder<Ensemble<Source<'a, M>>> {
    let mut sources = vec![
        Source::Lookup(Lookup::default()),
        Source::Session(SessionIndex::default()),
    ];
    let mtp = MtpDrafter::new(model).map(|mut d| {
        d.lookup = None;
        Source::Mtp(d)
    });
    let cheap = mtp.is_some() || extra.is_some();
    sources.extend(mtp);
    sources.extend(extra.map(Source::Extra));
    let mut decoder = Decoder::new(Ensemble::new(sources));
    if cheap {
        // A prediction block's or a drafter's drafts are cheap and usually
        // right.
        decoder.max_pause = 0;
    }
    decoder
}
