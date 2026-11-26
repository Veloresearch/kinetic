//! [`dspark_conformance`](./dspark_conformance.rs) on a CUDA GPU: compares
//! the device drafter's drafts with those another engine logged, and times
//! them.
//!
//! cargo run --release -p kinetic-models --features cuda --example cuda_dspark -- model.gguf drafter.gguf runs.txt

use std::time::Instant;

use kinetic_gguf::GgufFile;
use kinetic_models::cuda::CudaModel;
use kinetic_models::FusedDspark;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let usage = "usage: cuda_dspark <model.gguf> <drafter.gguf> <runs.txt>";
    let model_path = args.next().ok_or(usage)?;
    let drafter_path = args.next().ok_or(usage)?;
    let runs = parse(&std::fs::read_to_string(args.next().ok_or(usage)?)?)?;

    let file = GgufFile::open(&model_path)?;
    let drafter_file = GgufFile::open(&drafter_path)?;
    let model = kinetic_models::load(&file)?;
    let gpu = CudaModel::load(&model, 4096)?;
    let dspark = gpu.upload_dspark(FusedDspark::load(&drafter_file)?)?;
    let n_layers = model.n_layers();
    let taps = dspark.taps(n_layers);
    gpu.set_taps(taps.clone())?;
    let targets = FusedDspark::load(&drafter_file)?.target_layers().to_vec();
    let slots: Vec<usize> = targets
        .iter()
        .map(|l| taps.iter().position(|t| t == l).map_or(0, |i| i + 1))
        .collect();
    let d = model.n_embd();
    let (mut same, mut total, mut same_drafts, mut drafts_n) = (0, 0, 0, 0);
    let mut spent = 0.0;
    for run in &runs {
        let tokens: Vec<u32> = run.prompt.iter().chain(&run.generated).copied().collect();
        let mut state = gpu.new_state()?;
        let (_, hidden) = gpu.run_greedy(&mut state, &tokens, false)?;
        let width = hidden.len() / tokens.len();
        assert_eq!(width, (1 + taps.len()) * d, "hidden rows carry the taps");
        let features: Vec<f32> = hidden
            .chunks_exact(width)
            .flat_map(|row| slots.iter().flat_map(move |&s| &row[s * d..][..d]))
            .copied()
            .collect();
        let per = slots.len() * d;
        let mut ds = gpu.new_dspark_state(&dspark)?;
        let mut committed = 0;
        let mut anchor = run.prompt.len();
        let (mut run_same, mut run_total) = (0, 0);
        for theirs in &run.drafts {
            if anchor >= tokens.len() || theirs.is_empty() {
                break;
            }
            gpu.dspark_commit(&dspark, &mut ds, &features[committed * per..anchor * per])?;
            committed = anchor;
            let t = Instant::now();
            let ours: Vec<u32> = gpu
                .dspark_draft(&dspark, &mut ds, tokens[anchor], theirs.len())?
                .into_iter()
                .map(|(t, _)| t)
                .collect();
            spent += t.elapsed().as_secs_f64();
            run_same += ours.iter().zip(theirs).take_while(|(a, b)| a == b).count();
            run_total += theirs.len();
            same_drafts += usize::from(&ours == theirs);
            drafts_n += 1;
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
    println!("{:.2} ms per draft", 1e3 * spent / drafts_n as f64);
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
