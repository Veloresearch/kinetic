//! The DSpark drafter on a CUDA GPU, on a real model and its drafter: taps
//! change no bit of what a run returns, and the device drafts what the CPU
//! drafter does.
//!
//! KINETIC_TEST_GGUF=model.gguf KINETIC_TEST_DSPARK=drafter.gguf cargo test --release -p kinetic-models --features cuda --test cuda_dspark -- --ignored

#![cfg(feature = "cuda")]

use kinetic_gguf::GgufFile;
use kinetic_models::cuda::CudaModel;
use kinetic_models::speculative::{load_companion, Target};
use kinetic_models::FusedDspark;

const PROMPT: [u32; 6] = [1, 400, 1000, 23, 5000, 77];
/// Tokens the model continues the prompt with; drafts within them are
/// confident, as in use, where on arbitrary tokens near ties would flip.
const GENERATED: usize = 40;

fn bits(x: &[f32]) -> Vec<u32> {
    x.iter().map(|v| v.to_bits()).collect()
}

#[test]
#[ignore = "needs KINETIC_TEST_GGUF and KINETIC_TEST_DSPARK"]
fn the_device_drafts_what_the_cpu_does() {
    let model_path = std::env::var("KINETIC_TEST_GGUF").expect("set KINETIC_TEST_GGUF");
    let drafter_path = std::env::var("KINETIC_TEST_DSPARK").expect("set KINETIC_TEST_DSPARK");
    let file = GgufFile::open(&model_path).unwrap();
    let drafter_file = GgufFile::open(&drafter_path).unwrap();
    let model = kinetic_models::load(&file).unwrap();
    let d = model.n_embd();

    let gpu = CudaModel::load(&model, 256).unwrap();
    let mut state = gpu.new_state().unwrap();
    let mut tokens = PROMPT.to_vec();
    let (mut next, _) = gpu.run_greedy(&mut state, &PROMPT, false).unwrap();
    for _ in 0..GENERATED {
        tokens.push(next[0]);
        next = gpu
            .run_greedy(&mut state, &tokens[tokens.len() - 1..], false)
            .unwrap()
            .0;
    }
    let mut state = gpu.new_state().unwrap();
    let (plain_greedy, plain) = gpu.run_greedy(&mut state, &tokens, true).unwrap();
    drop(state);
    let dspark = gpu
        .upload_dspark(FusedDspark::load(&drafter_file).unwrap())
        .unwrap();
    let taps = dspark.taps(model.n_layers());
    gpu.set_taps(taps.clone()).unwrap();
    let mut state = gpu.new_state().unwrap();
    let (greedy, hidden) = gpu.run_greedy(&mut state, &tokens, true).unwrap();
    let width = (1 + taps.len()) * d;
    assert_eq!(hidden.len(), tokens.len() * width);
    assert_eq!(greedy, plain_greedy);
    let finals: Vec<f32> = hidden
        .chunks_exact(width)
        .flat_map(|row| &row[..d])
        .copied()
        .collect();
    assert_eq!(bits(&finals), bits(&plain), "taps change the final states");

    // the CPU drafter over the CPU's own features, the device's over its own
    let mut companion = load_companion(&drafter_file).unwrap();
    let mut model = kinetic_models::load(&file).unwrap();
    model.set_taps(taps.clone());
    let mut cpu_state = model.new_state();
    let (_, cpu_hidden) = Target::score(&model, &mut cpu_state, &tokens, false);
    let slots: Vec<usize> = companion
        .target_layers()
        .iter()
        .map(|l| taps.iter().position(|t| t == l).map_or(0, |i| i + 1))
        .collect();
    let features = |hidden: &[f32]| -> Vec<f32> {
        hidden
            .chunks_exact(width)
            .flat_map(|row| slots.iter().flat_map(move |&s| &row[s * d..][..d]))
            .copied()
            .collect()
    };
    let (cpu_features, gpu_features) = (features(&cpu_hidden), features(&hidden));
    let per = slots.len() * d;
    let mut ds = gpu.new_dspark_state(&dspark).unwrap();
    let (mut same, mut total) = (0, 0);
    for anchor in [12, 20, 30] {
        let committed = companion.committed();
        companion.commit(&cpu_features[committed * per..anchor * per]);
        gpu.dspark_commit(
            &dspark,
            &mut ds,
            &gpu_features[committed * per..anchor * per],
        )
        .unwrap();
        let cpu: Vec<u32> = companion
            .draft(tokens[anchor], &[], &|t| model.embeddings(t), 7, &|h| {
                model.draft_logits(h, model.n_vocab())
            })
            .into_iter()
            .map(|(t, _)| t)
            .collect();
        let ours: Vec<u32> = gpu
            .dspark_draft(&dspark, &mut ds, tokens[anchor], 7)
            .unwrap()
            .into_iter()
            .map(|(t, _)| t)
            .collect();
        assert_eq!(ours.len(), cpu.len());
        same += ours.iter().zip(&cpu).take_while(|(a, b)| a == b).count();
        total += cpu.len();
    }
    // the two differ in rounding, which may flip a near tie
    assert!(
        same * 10 >= total * 8,
        "{same} of {total} draft tokens agree"
    );
}
