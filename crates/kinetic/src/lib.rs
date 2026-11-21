//! Run a local language model from Rust.
//!
//! [`Model::open`] loads a `.gguf` or `.ktc` file. A text model then
//! [generates](Model::generate) after a raw prompt or [chats](Model::chat)
//! through its own chat template, and exposes the raw
//! [logits](Model::logits) of the next token; an embedding model
//! [embeds](Model::embed) texts. Everything runs on this computer: a text model on the GPU when
//! the `cuda` feature is on and the model fits in its memory, otherwise on
//! the CPU.
//!
//! ```no_run
//! let model = kinetic::Model::open("model.gguf")?;
//! let request = model.request();
//! let messages = [kinetic::Message::new("user", "Name three rivers.")];
//! let answer = model.chat(&messages, &request, &mut |text| {
//!     print!("{text}");
//!     true
//! })?;
//! println!("\n{} tokens", answer.completion_tokens);
//! # Ok::<(), kinetic::Error>(())
//! ```

mod file;

use std::mem::ManuallyDrop;
use std::path::Path;

use kinetic_models::{Embedder, Pooling};
use kinetic_server::{Backend, CpuBackend, Engine, Prompt};
use kinetic_tokenizer::Tokenizer;
use serde_json::{json, Map, Value};

pub use kinetic_server::{Completion, Finish, Request, Sampling};

use file::ModelFile;

/// Any failure: a file that cannot be read, a model that is not supported,
/// a request the model cannot serve.
pub type Error = Box<dyn std::error::Error + Send + Sync>;

/// Tokens of context a text model gets when it was trained on more.
pub const DEFAULT_CONTEXT: usize = 32768;

/// How [`Model::open_with`] loads a model.
#[derive(Clone, Debug)]
pub struct Options {
    /// Most tokens per conversation; `None` takes what the model was
    /// trained on, up to [`DEFAULT_CONTEXT`].
    pub context: Option<usize>,
    /// Whether a text model may run on the GPU when it fits there.
    pub gpu: bool,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            context: None,
            gpu: true,
        }
    }
}

/// One message of a conversation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Message {
    /// `system`, `user` or `assistant`.
    pub role: String,
    pub content: String,
}

impl Message {
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Message {
            role: role.into(),
            content: content.into(),
        }
    }
}

/// A loaded model. It is shared freely between threads; generation
/// requests take turns.
pub struct Model {
    // Borrows from `file`, so `Drop` releases it first.
    kind: ManuallyDrop<Kind>,
    file: &'static ModelFile,
    name: String,
}

enum Kind {
    Text(Box<Text>),
    Embedding(Box<Embedding>),
}

struct Text {
    engine: Engine<'static>,
    /// Leaked from a box like the file, which it borrows; the engine
    /// borrows it.
    model: &'static kinetic_models::Model<'static>,
}

struct Embedding {
    embedder: Embedder<'static>,
    tokenizer: Tokenizer,
}

impl Model {
    /// Loads the model in a `.gguf` or `.ktc` file. Its weights stay mapped
    /// from the file rather than copied into memory.
    pub fn open(path: impl AsRef<Path>) -> Result<Model, Error> {
        Model::open_with(path, &Options::default())
    }

    /// Loads the model in a `.gguf` or `.ktc` file as `options` ask.
    pub fn open_with(path: impl AsRef<Path>, options: &Options) -> Result<Model, Error> {
        let path = path.as_ref();
        let file: &'static ModelFile = Box::leak(Box::new(ModelFile::open(path)?));
        let name = path
            .file_stem()
            .map_or("model".into(), |s| s.to_string_lossy().into_owned());
        match load(file, &name, options) {
            Ok(kind) => Ok(Model {
                kind: ManuallyDrop::new(kind),
                file,
                name,
            }),
            Err(e) => {
                // SAFETY: `file` came from `Box::leak` above and nothing
                // built from it survived the failed load.
                drop(unsafe { Box::from_raw(std::ptr::from_ref(file).cast_mut()) });
                Err(e)
            }
        }
    }

    /// The file name without its extension.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Whether the model turns text into vectors rather than generating.
    pub fn is_embedding(&self) -> bool {
        matches!(*self.kind, Kind::Embedding(_))
    }

    /// Length of the vectors [`embed`](Model::embed) returns; `None` for a
    /// text model.
    pub fn embedding_size(&self) -> Option<usize> {
        match &*self.kind {
            Kind::Embedding(e) => Some(e.embedder.n_embd()),
            Kind::Text(_) => None,
        }
    }

    /// Tokens per conversation for a text model; `None` for an embedding
    /// model.
    pub fn context_length(&self) -> Option<usize> {
        self.engine().ok().map(Engine::context_length)
    }

    /// The token ids of `text`, with BOS where the model adds one. With
    /// `special`, control tokens such as `<|im_start|>` written in the
    /// text become their ids.
    pub fn tokenize(&self, text: &str, special: bool) -> Vec<u32> {
        self.tokenizer().encode(text, special)
    }

    /// The bytes `ids` stand for, special tokens included.
    pub fn detokenize(&self, ids: &[u32]) -> Vec<u8> {
        let t = self.tokenizer();
        ids.iter().flat_map(|&id| t.piece(id)).copied().collect()
    }

    /// The beginning- and end-of-sequence tokens, where the model has them.
    pub fn special_tokens(&self) -> (Option<u32>, Option<u32>) {
        let t = self.tokenizer();
        (t.bos(), t.eos())
    }

    fn tokenizer(&self) -> &Tokenizer {
        match &*self.kind {
            Kind::Text(text) => text.engine.tokenizer(),
            Kind::Embedding(e) => &e.tokenizer,
        }
    }

    /// A request with the sampling the model file recommends and no token
    /// limit beyond the context.
    pub fn request(&self) -> Request {
        let sampling = match &*self.kind {
            Kind::Text(text) => text.engine.default_sampling().clone(),
            Kind::Embedding(_) => Sampling::default(),
        };
        Request {
            sampling,
            max_tokens: None,
            stop: Vec::new(),
            grammar: None,
        }
    }

    /// Continues `prompt` exactly as written, with no chat template.
    /// `on_text` receives each new piece of text; returning false stops.
    pub fn generate(
        &self,
        prompt: &str,
        request: &Request,
        on_text: &mut dyn FnMut(&str) -> bool,
    ) -> Result<Completion, Error> {
        let engine = self.engine()?;
        Ok(
            engine.complete(&Prompt::raw(prompt), request, &mut |part, _| {
                on_text(&part.content)
            })?,
        )
    }

    /// Answers the conversation in `messages` through the model's chat
    /// template. `on_text` receives each new piece of the answer; returning
    /// false stops.
    pub fn chat(
        &self,
        messages: &[Message],
        request: &Request,
        on_text: &mut dyn FnMut(&str) -> bool,
    ) -> Result<Completion, Error> {
        let engine = self.engine()?;
        let messages: Vec<Value> = messages
            .iter()
            .map(|m| json!({"role": m.role, "content": m.content}))
            .collect();
        let prompt = engine.chat_prompt(&messages, &Map::new())?;
        Ok(engine.complete(&prompt, request, &mut |part, _| {
            part.content.is_empty() || on_text(&part.content)
        })?)
    }

    /// One vector per text, from an embedding model.
    pub fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, Error> {
        let Kind::Embedding(e) = &*self.kind else {
            return Err(format!("{} generates text; open an embedding model", self.name).into());
        };
        let limit = e.embedder.n_ctx_train();
        let inputs = texts
            .iter()
            .map(|text| {
                if text.is_empty() {
                    return Err("texts must not be empty".into());
                }
                let ids = e.tokenizer.encode(text, true);
                if ids.len() > limit {
                    return Err(format!(
                        "a text is {} tokens; this model takes at most {limit}",
                        ids.len()
                    )
                    .into());
                }
                Ok(ids)
            })
            .collect::<Result<Vec<_>, Error>>()?;
        let refs: Vec<&[u32]> = inputs.iter().map(Vec::as_slice).collect();
        Ok(e.embedder.embed(&refs))
    }

    /// How many values [`logits`](Model::logits) returns: the vocabulary
    /// size. `None` for an embedding model.
    pub fn n_vocab(&self) -> Option<usize> {
        Some(self.as_text().ok()?.model.n_vocab())
    }

    /// The model's raw logits after `prompt`: one value per vocabulary
    /// entry, exactly the scores plain decoding's sampler would see, before
    /// any sampler, from the backend a request for them would use, so they
    /// are that device's own. Greedy decoding (`temperature` 0) picks
    /// through a fused top-1 on the device and this row comes from the
    /// head read; when the top two scores sit within the rounding of the
    /// two, the argmax and greedy's pick can differ — each is exact for
    /// its own kernel.
    ///
    /// `prompt` continues exactly as [`generate`](Model::generate) takes
    /// it, with no chat template;
    /// [`logits_chat`](Model::logits_chat) renders messages through the
    /// template first. An embedding model has no logits; use
    /// [`embed`](Model::embed).
    pub fn logits(&self, prompt: &str) -> Result<Vec<f32>, Error> {
        self.engine()?;
        let ids = self.encode_prompt(prompt)?;
        Ok(self.engine()?.logits(&ids, false)?)
    }

    /// [`Model::logits`] after `messages` rendered through the model's own
    /// chat template, the generation prompt included.
    pub fn logits_chat(&self, messages: &[Message]) -> Result<Vec<f32>, Error> {
        let engine = self.engine()?;
        let prompt = engine.chat_prompt(
            &messages
                .iter()
                .map(|m| json!({"role": m.role, "content": m.content}))
                .collect::<Vec<_>>(),
            &Map::new(),
        )?;
        let ids = self.encode_prompt(&prompt.text)?;
        Ok(engine.logits(&ids, false)?)
    }

    /// The model's raw logits after every token of `tokens`, as one flat
    /// row-major buffer of `tokens.len() *`
    /// [`n_vocab`](Model::n_vocab) values: row `i`, the `n_vocab` values
    /// at `i * n_vocab`, scores the token after `tokens[i]`. Score a token
    /// sequence by summing, for every `i < n - 1`, the log-softmax of
    /// `tokens[i + 1]` under row `i`; the first row scores a token of your
    /// own.
    ///
    /// One row per token means the buffer holds `tokens.len() * n_vocab`
    /// floats — about 1 GB per thousand tokens at a 250,000-entry
    /// vocabulary — so score long texts in chunks, as `kinetic perplexity`
    /// does.
    ///
    /// An embedding model has no logits; use [`embed`](Model::embed).
    pub fn logits_rows(&self, tokens: &[u32]) -> Result<Vec<f32>, Error> {
        let text = self.as_text()?;
        if tokens.is_empty() {
            return Err("the tokens are empty".into());
        }
        let vocab = text.model.n_vocab();
        if let Some(&id) = tokens.iter().find(|&&t| t as usize >= vocab) {
            return Err(
                format!("token {id} is past the vocabulary, which has {vocab} entries").into(),
            );
        }
        if let Some(limit) = self.context_length() {
            if tokens.len() > limit {
                return Err(format!(
                    "the tokens are {} long, over the {}-token context",
                    tokens.len(),
                    limit
                )
                .into());
            }
        }
        Ok(text.engine.logits(tokens, true)?)
    }

    /// The text model, or the error an embedding model's callers get.
    fn as_text(&self) -> Result<&Text, Error> {
        match &*self.kind {
            Kind::Text(text) => Ok(text),
            Kind::Embedding(_) => {
                Err(format!("{} is an embedding model; use embed", self.name).into())
            }
        }
    }

    /// Encodes a prompt the way generation does: special tokens written in
    /// the text become their ids, and a template that writes BOS itself is
    /// not doubled.
    fn encode_prompt(&self, text: &str) -> Result<Vec<u32>, Error> {
        let mut ids = self.tokenize(text, true);
        let (bos, _) = self.special_tokens();
        if let Some(bos) = bos {
            if ids.len() > 1 && ids[0] == ids[1] && ids[0] == bos {
                ids.remove(0);
            }
        }
        if ids.is_empty() {
            return Err("the prompt is empty".into());
        }
        if let Some(limit) = self.context_length() {
            if ids.len() > limit {
                return Err(format!(
                    "the prompt is {} tokens, over the {}-token context",
                    ids.len(),
                    limit
                )
                .into());
            }
        }
        Ok(ids)
    }

    fn engine(&self) -> Result<&Engine<'static>, Error> {
        match &*self.kind {
            Kind::Text(text) => Ok(&text.engine),
            Kind::Embedding(_) => {
                Err(format!("{} is an embedding model; use embed", self.name).into())
            }
        }
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        let model = match &*self.kind {
            Kind::Text(text) => Some(text.model),
            Kind::Embedding(_) => None,
        };
        // SAFETY: the engine in `kind` borrows `model`, which borrows
        // `file`, so they are dropped in that order, and none is used again;
        // `model` and `file` came from `Box::leak` in `load` and `open`.
        unsafe {
            ManuallyDrop::drop(&mut self.kind);
            if let Some(model) = model {
                drop(Box::from_raw(std::ptr::from_ref(model).cast_mut()));
            }
            drop(Box::from_raw(std::ptr::from_ref(self.file).cast_mut()));
        }
    }
}

fn load(file: &'static ModelFile, name: &str, options: &Options) -> Result<Kind, Error> {
    let source = file.source();
    let header = source.header();
    let encoder = header
        .architecture()
        .is_some_and(kinetic_models::is_encoder);
    if encoder || header.arch_key("pooling_type").is_some() {
        let embedder = Embedder::load(source)?;
        if embedder.pooling() == Pooling::Rank {
            return Err("ranking models are not supported yet".into());
        }
        return Ok(Kind::Embedding(Box::new(Embedding {
            embedder,
            tokenizer: Tokenizer::from_gguf(header)?,
        })));
    }
    let model: &'static kinetic_models::Model = Box::leak(Box::new(kinetic_models::load(source)?));
    let trained = header
        .arch_key("context_length")
        .and_then(|v| v.as_u64())
        .map_or(DEFAULT_CONTEXT, |n| n as usize);
    let context = options.context.unwrap_or(DEFAULT_CONTEXT).min(trained);
    let backend = if options.gpu {
        backend(model, context)
    } else {
        Box::new(CpuBackend::new(model))
    };
    match Engine::new(name, header, backend, context) {
        Ok(engine) => Ok(Kind::Text(Box::new(Text { engine, model }))),
        Err(e) => {
            // SAFETY: `model` came from `Box::leak` above, and the backend
            // that borrowed it went with the failed engine.
            drop(unsafe { Box::from_raw(std::ptr::from_ref(model).cast_mut()) });
            Err(e)
        }
    }
}

/// The GPU when it takes the model, as `kinetic run --device auto` picks,
/// else the CPU.
#[cfg(feature = "cuda")]
fn backend(model: &'static kinetic_models::Model, context: usize) -> Box<dyn Backend> {
    match kinetic_server::CudaBackend::new(model, context) {
        Ok(gpu) => Box::new(gpu),
        Err(_) => Box::new(CpuBackend::new(model)),
    }
}

#[cfg(not(feature = "cuda"))]
fn backend(model: &'static kinetic_models::Model, _context: usize) -> Box<dyn Backend> {
    Box::new(CpuBackend::new(model))
}
