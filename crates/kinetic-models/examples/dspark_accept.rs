//! Greedy speculative decoding with the drafter a model ships, on the CPU:
//! tokens accepted per verification pass, drafter time per round and target
//! time per pass. The drafter is the given file, or the one beside the
//! model when none is given.
//!
//! cargo run --release -p kinetic-models --example dspark_accept -- model.gguf prompt.txt [drafter.gguf] [n]

use std::path::Path;
use std::time::{Duration, Instant};

use kinetic_gguf::GgufFile;
use kinetic_models::speculative::{
    find_companion, load_companion, CompanionDrafter, Decoder, Draft, Drafter, Greedy,
};
use kinetic_models::Model;
use kinetic_tokenizer::Tokenizer;

/// A drafter that adds up the time it takes.
struct Timed<D> {
    drafter: D,
    spent: Duration,
}

impl<'m, D: Drafter<Model<'m>>> Drafter<Model<'m>> for Timed<D> {
    fn observe(&mut self, model: &Model<'m>, tokens: &[u32], hidden: &[f32]) {
        let start = Instant::now();
        self.drafter.observe(model, tokens, hidden);
        self.spent += start.elapsed();
    }

    fn propose(&mut self, model: &Model<'m>, context: &[u32], limit: usize) -> Vec<u32> {
        let start = Instant::now();
        let draft = self.drafter.propose(model, context, limit);
        self.spent += start.elapsed();
        draft
    }

    fn propose_tree(&mut self, model: &Model<'m>, context: &[u32], limit: usize) -> Draft {
        let start = Instant::now();
        let draft = self.drafter.propose_tree(model, context, limit);
        self.spent += start.elapsed();
        draft
    }

    fn reset(&mut self, model: &Model<'m>) {
        self.drafter.reset(model);
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let usage = "usage: dspark_accept <model.gguf> <prompt.txt> [drafter.gguf] [n]";
    let path = args.next().ok_or(usage)?;
    let prompt = std::fs::read_to_string(args.next().ok_or(usage)?)?;
    let explicit = args.next();
    let n: usize = args.next().map_or(Ok(128), |s| s.parse())?;

    let file = GgufFile::open(&path)?;
    let companion = find_companion(
        Path::new(&path),
        file.header(),
        explicit.as_deref().map(Path::new),
    )?
    .ok_or("no drafter beside the model; pass one")?;
    println!("drafter {}", companion.display());
    let drafter_file = GgufFile::open(&companion)?;
    let tokenizer = Tokenizer::from_gguf(file.header())?;
    let mut model = kinetic_models::load(&file)?;
    let mut drafter = CompanionDrafter::new(load_companion(&drafter_file)?, &mut model)?;
    // Only the drafter, so its acceptance shows.
    drafter.lookup = None;
    let mut decoder = Decoder::new(Timed {
        drafter,
        spent: Duration::ZERO,
    });
    decoder.max_pause = 0;

    let ids = tokenizer.encode(&prompt, true);
    let eos = tokenizer.eos();
    println!("prompt {} tokens", ids.len());
    let mut state = model.new_state();
    let mut context = ids.clone();
    decoder.prefill(&model, &mut Greedy, &mut state, &context, 0);
    let (mut passes, mut accepted, mut drafted) = (0, 0, 0);
    let (mut drafter_time, mut target_time) = (Duration::ZERO, Duration::ZERO);
    println!("pass accepted/drafted drafter_ms target_ms");
    while context.len() < ids.len() + n && decoder.pending().is_some() {
        let limit = ids.len() + n - context.len();
        decoder.drafter.spent = Duration::ZERO;
        let start = Instant::now();
        let step = decoder.step(&model, &mut Greedy, &mut state, &mut context, limit, |t| {
            Some(t) == eos
        });
        let (d, t) = (
            decoder.drafter.spent,
            start.elapsed() - decoder.drafter.spent,
        );
        passes += 1;
        accepted += step.tokens - 1;
        drafted += step.drafted;
        drafter_time += d;
        target_time += t;
        println!(
            "{passes:4} {}/{} {:.1} {:.1}",
            step.tokens - 1,
            step.drafted,
            d.as_secs_f64() * 1e3,
            t.as_secs_f64() * 1e3
        );
    }
    let generated = context.len() - ids.len();
    println!(
        "{generated} tokens in {passes} passes: {:.2} tokens per pass, {accepted} of {drafted} drafts accepted",
        generated as f64 / passes as f64
    );
    println!(
        "drafter {:.1} ms per round, target {:.1} ms per pass",
        drafter_time.as_secs_f64() * 1e3 / passes as f64,
        target_time.as_secs_f64() * 1e3 / passes as f64
    );
    println!("---\n{}", tokenizer.decode(&context[ids.len()..]));
    Ok(())
}
