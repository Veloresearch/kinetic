//! Compares a DSpark drafter's drafts with those another engine logged for
//! the same greedy runs, on the CPU. The input holds one run after another,
//! one line each of space-separated token ids: `prompt ...` starts a run,
//! then `generated ...`, then one `draft ...` line per draft in the order
//! the engine made them. Each draft's anchor is found
//! from the ones before it: the anchor after a draft whose first `a` tokens
//! match the generated tokens is `a + 1` tokens further on.
//!
//! cargo run --release -p kinetic-models --example dspark_conformance -- model.gguf drafter.gguf runs.txt

use kinetic_gguf::GgufFile;
use kinetic_models::speculative::{load_companion, Target};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let usage = "usage: dspark_conformance <model.gguf> <drafter.gguf> <runs.txt>";
    let model_path = args.next().ok_or(usage)?;
    let drafter_path = args.next().ok_or(usage)?;
    let runs = parse(&std::fs::read_to_string(args.next().ok_or(usage)?)?)?;

    let file = GgufFile::open(&model_path)?;
    let drafter_file = GgufFile::open(&drafter_path)?;
    let mut model = kinetic_models::load(&file)?;
    let mut companion = load_companion(&drafter_file)?;
    let final_state = model.n_layers();
    let taps: Vec<usize> = companion
        .target_layers()
        .iter()
        .copied()
        .filter(|&l| l < final_state)
        .collect();
    model.set_taps(taps.clone());
    let slots: Vec<usize> = companion
        .target_layers()
        .iter()
        .map(|l| taps.iter().position(|t| t == l).map_or(0, |i| i + 1))
        .collect();
    let d = model.n_embd();
    let (mut same, mut total, mut same_drafts, mut drafts_n) = (0, 0, 0, 0);
    for run in &runs {
        let (prompt, generated) = (&run.prompt, &run.generated);
        let tokens: Vec<u32> = prompt.iter().chain(generated).copied().collect();
        let mut state = model.new_state();
        let (_, hidden) = Target::score(&model, &mut state, &tokens, false);
        let width = hidden.len() / tokens.len();
        let features: Vec<f32> = hidden
            .chunks_exact(width)
            .flat_map(|row| slots.iter().flat_map(move |&s| &row[s * d..][..d]))
            .copied()
            .collect();
        let per = slots.len() * d;
        companion.reset();
        let mut anchor = prompt.len();
        let (mut run_same, mut run_total) = (0, 0);
        for theirs in &run.drafts {
            if anchor >= tokens.len() || theirs.is_empty() {
                break;
            }
            let committed = companion.committed();
            companion.commit(&features[committed * per..anchor * per]);
            let ours: Vec<u32> = companion
                .draft(
                    tokens[anchor],
                    &[],
                    &|t| model.embeddings(t),
                    theirs.len(),
                    &|h| model.draft_logits(h, model.n_vocab()),
                )
                .into_iter()
                .map(|(t, _)| t)
                .collect();
            let matching = ours.iter().zip(theirs).take_while(|(a, b)| a == b).count();
            run_same += matching;
            run_total += theirs.len();
            same_drafts += usize::from(&ours == theirs);
            drafts_n += 1;
            if &ours != theirs {
                println!("anchor {anchor}: ours {ours:?} theirs {theirs:?}");
            }
            let accepted = theirs
                .iter()
                .zip(&tokens[anchor + 1..])
                .take_while(|(a, b)| a == b)
                .count();
            anchor += accepted + 1;
        }
        println!("run: {run_same} of {run_total} draft tokens agree");
        same += run_same;
        total += run_total;
    }
    println!(
        "all: {same} of {total} draft tokens agree, {same_drafts} of {drafts_n} drafts identical"
    );
    Ok(())
}

struct Run {
    prompt: Vec<u32>,
    generated: Vec<u32>,
    drafts: Vec<Vec<u32>>,
}

fn parse(text: &str) -> Result<Vec<Run>, Box<dyn std::error::Error>> {
    let mut runs: Vec<Run> = Vec::new();
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let mut words = line.split_whitespace();
        let kind = words.next().unwrap_or_default();
        let ids = words.map(str::parse).collect::<Result<Vec<u32>, _>>()?;
        match (kind, runs.last_mut()) {
            ("prompt", _) => runs.push(Run {
                prompt: ids,
                generated: Vec::new(),
                drafts: Vec::new(),
            }),
            ("generated", Some(run)) => run.generated = ids,
            ("draft", Some(run)) => run.drafts.push(ids),
            _ => return Err(format!("unexpected line: {line}").into()),
        }
    }
    Ok(runs)
}
