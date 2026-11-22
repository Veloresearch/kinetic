use std::path::{Path, PathBuf};

use kinetic_gguf::{Gguf, GgufFile, TensorSource};

use super::dspark::DsparkCompanion;
use super::{Draft, Drafter, Lookup};
use crate::{Dspark, LoadError, Model};

/// A drafter that belongs to a target model: it reads hidden states the
/// target returns for every kept token and drafts the tokens after an
/// anchor, the token about to run.
pub trait Companion {
    /// The target's layer visits whose input hidden states it reads, in the
    /// order it reads them; the number of the target's layers stands for
    /// its normalized final hidden state.
    fn target_layers(&self) -> &[usize];

    /// The width of the target's hidden states.
    fn n_embd(&self) -> usize;

    /// Tokens of the target's vocabulary it scores.
    fn vocab(&self) -> usize;

    /// Most tokens one draft proposes.
    fn max_draft(&self) -> usize;

    /// Positions committed so far.
    fn committed(&self) -> usize;

    /// Commits the positions after those committed: per position, the
    /// target's hidden states at [`Companion::target_layers`].
    fn commit(&mut self, features: &[f32]);

    /// Up to `n` tokens after `anchor`, at the position after those
    /// committed, each with its chance given the ones before. `embedding` is
    /// the anchor's embedding as the target's first layer reads it, and
    /// `head` the target's logits for a hidden state.
    fn draft(
        &self,
        anchor: u32,
        embedding: &[f32],
        n: usize,
        head: &dyn Fn(&[f32]) -> Vec<f32>,
    ) -> Vec<(u32, f32)>;

    /// Forgets the sequence.
    fn reset(&mut self);
}

/// Drafts with a target's [`Companion`]. A context lookup is tried first,
/// since a match there is free and usually long; the companion fills in
/// where there is nothing to copy.
pub struct CompanionDrafter<'a> {
    companion: Box<dyn Companion + 'a>,
    /// Where each of the companion's target layers sits in a row of the
    /// target's hidden states, in units of its width.
    slots: Vec<usize>,
    pub lookup: Option<Lookup>,
}

impl<'a> CompanionDrafter<'a> {
    /// Makes `model` return the hidden states `companion` reads with every
    /// run; fails when the companion was not made for the model.
    pub fn new(companion: Box<dyn Companion + 'a>, model: &mut Model) -> Result<Self, LoadError> {
        let layers = companion.target_layers();
        let final_state = model.n_layers();
        if companion.n_embd() != model.n_embd()
            || companion.vocab() != model.n_vocab()
            || layers.iter().any(|&l| l > final_state)
        {
            return Err(LoadError("the drafter was not made for this model".into()));
        }
        let inner: Vec<usize> = layers
            .iter()
            .copied()
            .filter(|&l| l < final_state)
            .collect();
        let slots = layers
            .iter()
            .map(|l| inner.iter().position(|i| i == l).map_or(0, |i| i + 1))
            .collect();
        model.set_taps(inner);
        Ok(CompanionDrafter {
            companion,
            slots,
            // Only long matches: they beat the drafter on repeated text and
            // rarely misfire on new text.
            lookup: Some(Lookup {
                max_ngram: 4,
                min_ngram: 3,
            }),
        })
    }

    /// The companion's tokens after the last of `context` and their
    /// chances, when every token before it is committed.
    fn draft(&self, model: &Model, context: &[u32], limit: usize) -> Vec<(u32, f32)> {
        let c = &self.companion;
        match context.last() {
            Some(&anchor) if c.committed() > 0 && c.committed() + 1 == context.len() => {
                let embedding = model.first_layer_input(&[anchor]);
                let n = limit.min(c.max_draft());
                c.draft(anchor, &embedding, n, &|h| {
                    model.draft_logits(h, model.n_vocab())
                })
            }
            _ => Vec::new(),
        }
    }

    fn lookup(&mut self, model: &Model, context: &[u32], limit: usize) -> Vec<u32> {
        self.lookup.as_mut().map_or_else(Vec::new, |l| {
            Drafter::<Model>::propose(l, model, context, limit)
        })
    }
}

impl<'m> Drafter<Model<'m>> for CompanionDrafter<'_> {
    fn observe(&mut self, model: &Model<'m>, tokens: &[u32], hidden: &[f32]) {
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
        self.companion.commit(&features);
    }

    fn propose(&mut self, model: &Model<'m>, context: &[u32], limit: usize) -> Vec<u32> {
        let copied = self.lookup(model, context, limit);
        if !copied.is_empty() {
            return copied;
        }
        self.draft(model, context, limit)
            .into_iter()
            .map(|(t, _)| t)
            .collect()
    }

    fn propose_tree(&mut self, model: &Model<'m>, context: &[u32], limit: usize) -> Draft {
        let copied = self.lookup(model, context, limit);
        if !copied.is_empty() {
            return Draft::chain(copied);
        }
        let mut draft = Draft::default();
        let mut chance = 1.0;
        for (token, p) in self.draft(model, context, limit) {
            chance *= p;
            let parent = draft.len().checked_sub(1);
            draft.push(token, parent, chance);
        }
        draft
    }

    fn reset(&mut self, _model: &Model<'m>) {
        self.companion.reset();
    }
}

type LoadCompanion = for<'a> fn(&'a dyn TensorSource) -> Result<Box<dyn Companion + 'a>, LoadError>;

/// Drafters published beside their targets: the target architectures, the
/// start and end of the drafter's file name in the target's folder, the
/// drafter's architecture and how to load it.
struct Pairing {
    targets: &'static [&'static str],
    name: (&'static str, &'static str),
    arch: &'static str,
    load: LoadCompanion,
}

const PAIRINGS: &[Pairing] = &[Pairing {
    targets: &["inkling2", "inkling2_loop"],
    name: ("dspark-", ".gguf"),
    arch: "dflash",
    load: |file| Ok(Box::new(DsparkCompanion::new(Dspark::load(file)?))),
}];

/// Keys a drafter's header must share with its target's, each under its
/// own architecture.
const MATCHING: &[&str] = &["embedding_length", "vocab_size_unpadded"];

/// The companion drafter of the model file at `target`, whose header is
/// `header`: `explicit` when given, otherwise the first file beside it
/// whose name a pairing gives. `None` when there is none; an error when the
/// drafter found does not fit the target.
pub fn find_companion(
    target: &Path,
    header: &Gguf,
    explicit: Option<&Path>,
) -> Result<Option<PathBuf>, LoadError> {
    let arch = header.architecture().unwrap_or_default();
    let path = match explicit {
        Some(path) => path.to_path_buf(),
        None => {
            let Some(pairing) = PAIRINGS.iter().find(|p| p.targets.contains(&arch)) else {
                return Ok(None);
            };
            let folder = target.parent().unwrap_or(Path::new("."));
            let Ok(entries) = std::fs::read_dir(folder) else {
                return Ok(None);
            };
            let (start, end) = pairing.name;
            let mut found: Vec<PathBuf> = entries
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| {
                    p.file_name()
                        .and_then(|n| n.to_str())
                        .is_some_and(|n| n.starts_with(start) && n.ends_with(end))
                })
                .collect();
            found.sort();
            match found.into_iter().next() {
                Some(path) => path,
                None => return Ok(None),
            }
        }
    };
    let file =
        GgufFile::open(&path).map_err(|e| LoadError(format!("drafter {}: {e}", path.display())))?;
    let drafter = file.header();
    let drafter_arch = drafter.architecture().unwrap_or_default();
    let paired = PAIRINGS
        .iter()
        .any(|p| p.targets.contains(&arch) && p.arch == drafter_arch);
    let fits = MATCHING.iter().all(|k| {
        let value = |h: &Gguf, a: &str| h.get(&format!("{a}.{k}")).and_then(|v| v.as_u64());
        value(drafter, drafter_arch).is_some()
            && value(drafter, drafter_arch) == value(header, arch)
    });
    if !paired || !fits {
        return Err(LoadError(format!(
            "{} is not a drafter for this {arch} model",
            path.display()
        )));
    }
    Ok(Some(path))
}

/// Loads the companion drafter in `file`, found by [`find_companion`].
pub fn load_companion(file: &dyn TensorSource) -> Result<Box<dyn Companion + '_>, LoadError> {
    let arch = file.header().architecture().unwrap_or_default();
    let pairing = PAIRINGS
        .iter()
        .find(|p| p.arch == arch)
        .ok_or_else(|| LoadError(format!("{arch} is not a drafter Kinetic runs")))?;
    (pairing.load)(file)
}
