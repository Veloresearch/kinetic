//! Compares token ids from a real GGUF vocabulary against reference ids.
//!
//! KINETIC_TEST_GGUF=model.gguf cargo test --release -p kinetic-tokenizer -- --ignored
//!
//! The bundled inputs are llama.cpp's `ggml-vocab-qwen35` test texts; the ids
//! are what llama.cpp's `llama-tokenize --no-escape` gives for Qwen3.5-4B,
//! whose vocabulary differs from the one upstream recorded. Set
//! KINETIC_TEST_VECTORS to another `.inp` file, with its `.out` next to it,
//! to check a different model or corpus.
//!
//! `tekken.inp` holds mixed-case, digit, punctuation, non-Latin, combining
//! mark, emoji and Polish texts for Mistral Nemo. Its ids, BOS included, are
//! what the Hugging Face `tokenizers` package 0.23.3 gives with the
//! `tokenizer.json` of Mistral-Nemo-Instruct-2407 and
//! `add_special_tokens=True`. A GGUF holding only the header of
//! `bartowski/Mistral-Nemo-Instruct-2407-GGUF`, with its tensor count set
//! to zero, is enough to run them.
//!
//! `inkling2.inp` holds the same texts and a few with contractions and long
//! numbers for Inkling 2. Its ids are what `tokenizers` 0.23.3 gives with
//! the `tokenizer.json` of thinkingmachines/Inkling-2-Mini and
//! `add_special_tokens=True`, which adds none. The header of
//! `ggml-org/Inkling-2-Mini-GGUF` is enough to run them, in a copy of the
//! file whose tensor data is a sparse hole of the same length.
//!
//! `phi3.inp` starts with the Inkling 2 corpus and adds whitespace after
//! EOS and role markers, non-trimming special tokens, and whitespace and
//! Unicode controls. The ids come from Hugging Face `tokenizers` 0.23.3 with
//! `add_special_tokens=True` and `microsoft/Phi-3-mini-4k-instruct`'s
//! `tokenizer.json` at `f39ac1d28e925b323eae81227eaba4464caced4e`.
//! That reference adds no BOS. Run against the vocabulary header of
//! Microsoft's `Phi-3-mini-4k-instruct-q4.gguf` from
//! `microsoft/Phi-3-mini-4k-instruct-gguf` at
//! `a64113399c2f6b8ad3e11c394733a2ddadaa7f33`.

use std::path::PathBuf;

use kinetic_gguf::GgufFile;
use kinetic_tokenizer::Tokenizer;

const SEPARATOR: &str = "\n__ggml_vocab_test__\n";

fn vectors(default: &str) -> Vec<(String, Vec<u32>)> {
    let inp = std::env::var_os("KINETIC_TEST_VECTORS").map_or_else(
        || {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("tests/data")
                .join(default)
        },
        PathBuf::from,
    );
    let inputs = std::fs::read_to_string(&inp).unwrap();
    let outputs = std::fs::read_to_string(inp.with_extension("out")).unwrap();
    let inputs = inputs.strip_suffix(SEPARATOR).unwrap_or(&inputs);
    let pairs: Vec<_> = inputs
        .split(SEPARATOR)
        .zip(outputs.lines())
        .map(|(text, ids)| {
            let ids = ids
                .split_whitespace()
                .map(|id| id.parse().unwrap())
                .collect();
            (text.to_owned(), ids)
        })
        .collect();
    assert_eq!(
        pairs.len(),
        outputs.lines().count(),
        "inputs and outputs differ in count"
    );
    pairs
}

#[test]
#[ignore = "needs KINETIC_TEST_GGUF"]
fn matches_llama_cpp_token_ids() {
    let path = std::env::var("KINETIC_TEST_GGUF").expect("set KINETIC_TEST_GGUF");
    let file = GgufFile::open(&path).unwrap();
    let tokenizer = Tokenizer::from_gguf(file.header()).unwrap();

    let phi3 = file.header().architecture() == Some("phi3");
    let bos = tokenizer.encode("", false);
    if phi3 {
        assert_eq!(bos, [1], "Phi-3 GGUF should add one BOS");
    }
    let vectors = vectors(if phi3 { "phi3.inp" } else { "qwen35.inp" });
    let mut mismatches = 0;
    for (text, expected) in &vectors {
        let ids = tokenizer.encode(text, true);
        let body = &ids[bos.len()..];
        // Upstream vectors are recorded without the BOS the model may add.
        let ids = if !phi3 && expected.starts_with(&bos) {
            &ids
        } else {
            body
        };
        if ids != expected {
            mismatches += 1;
            println!("{text:?}\n  expected {expected:?}\n  got      {ids:?}");
        }
        // Phi-3's special tokens can consume whitespace.
        if !phi3 {
            let decoded = tokenizer.decode(body);
            // SentencePiece models may put a space in front of each run of text.
            assert!(
                decoded == *text || decoded.replace(' ', "") == text.replace(' ', ""),
                "decode does not round-trip: {decoded:?}"
            );
        }
    }
    println!("{} of {} match", vectors.len() - mismatches, vectors.len());
    assert_eq!(mismatches, 0);
}
