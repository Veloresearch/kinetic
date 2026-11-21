//! The logits a backend returns are the logits its requests choose from,
//! on the CPU, and with the features on, on Vulkan and CUDA.
//!
//! KINETIC_TEST_GGUF=model.gguf \
//! cargo test --release -p kinetic-server --test score -- --ignored
//!
//! KINETIC_TEST_GGUF=model.gguf \
//! cargo test --release -p kinetic-server --features vulkan --test score -- --ignored
//!
//! KINETIC_TEST_GGUF=model.gguf \
//! cargo test --release -p kinetic-server --features cuda --test score -- --ignored

use kinetic_gguf::GgufFile;
use kinetic_models::load;
use kinetic_server::{Backend, CpuBackend, Job, Sampling};

/// A prompt unlike any training data, as tests/cuda.rs uses.
const PROMPT: [u32; 12] = [1, 400, 1000, 23, 5000, 77, 901, 12, 3333, 600, 45, 2024];

/// The row after the prompt is the row a greedy request chooses from, the
/// last of the rows is the same row, and both are the same bits again.
fn parity(backend: &mut dyn Backend, vocab: usize) {
    let sampling = Sampling {
        temperature: 0.0,
        ..Sampling::default()
    };
    let job = Job {
        sampling: &sampling,
        max_tokens: 1,
        stop_tokens: &[],
        grammar: None,
        images: &[],
        logprobs: None,
    };
    let mut chosen = None;
    backend.generate(&PROMPT, &job, &mut |tokens| {
        chosen = Some(tokens[0]);
        false
    });
    let row = backend
        .logits(&PROMPT, false)
        .expect("the row after the prompt");
    assert_eq!(row.len(), vocab);
    // Greedy picks through the device's fused top-1 and this row comes
    // from the head read; when the top two scores sit within the rounding
    // of the two kernels the argmax and the pick can differ, so equality
    // is asserted only when the gap is decisive and the pick is one of
    // the top five always.
    let mut top: Vec<usize> = (0..row.len()).collect();
    top.sort_by(|a, b| row[*b].total_cmp(&row[*a]));
    let pick = chosen.expect("a chosen token") as usize;
    assert!(
        top[..5].contains(&pick),
        "the pick {pick} is in the top five"
    );
    if row[top[0]] - row[top[1]] > 0.1 {
        assert_eq!(pick, top[0], "a decisive top chooses the argmax");
    }

    let all = backend
        .logits(&PROMPT, true)
        .expect("one row per token of the prompt");
    assert_eq!(all.len(), PROMPT.len() * vocab);
    assert_eq!(
        &all[all.len() - vocab..],
        &row[..],
        "the last row is the row"
    );

    assert_eq!(
        backend.logits(&PROMPT, false).expect("the row again"),
        row,
        "a rerun gives the same bits"
    );
}

#[test]
#[ignore = "needs KINETIC_TEST_GGUF"]
fn cpu_rows_match_generation() {
    let file =
        GgufFile::open(std::env::var("KINETIC_TEST_GGUF").expect("set KINETIC_TEST_GGUF")).unwrap();
    let model = load(&file).unwrap();
    parity(&mut CpuBackend::new(&model), model.n_vocab());
}

#[cfg(feature = "vulkan")]
#[test]
#[ignore = "needs KINETIC_TEST_GGUF and a Vulkan device"]
fn vulkan_rows_match_generation() {
    let file =
        GgufFile::open(std::env::var("KINETIC_TEST_GGUF").expect("set KINETIC_TEST_GGUF")).unwrap();
    let model = load(&file).unwrap();
    let mut backend =
        kinetic_server::VulkanBackend::new(&model, 4096).unwrap_or_else(|e| panic!("Vulkan: {e}"));
    parity(&mut backend, model.n_vocab());
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs KINETIC_TEST_GGUF and a CUDA device"]
fn cuda_rows_match_generation() {
    let file =
        GgufFile::open(std::env::var("KINETIC_TEST_GGUF").expect("set KINETIC_TEST_GGUF")).unwrap();
    let model = load(&file).unwrap();
    let mut backend =
        kinetic_server::CudaBackend::new(&model, 4096).unwrap_or_else(|e| panic!("CUDA: {e}"));
    parity(&mut backend, model.n_vocab());
}
