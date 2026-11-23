# Header fixtures

Headers of published GGUF files without their weights, written by the
`fixture` example; `tests/headers.rs` builds every family from them. Values
longer than 256 items or bytes (the tokenizer's vocabulary, scores, merges,
chat template) are kept only as their length.

To add one, take the first bytes of the file and write the fixture:

```
curl -L -r 0-16777215 https://huggingface.co/<repo>/resolve/<revision>/<file> -o head.gguf
cargo run --release -p kinetic-models --example fixture -- head.gguf crates/kinetic-models/tests/headers/<name>.gguf
```

then add its row here and its digest with
`KINETIC_BLESS=1 cargo test --release -p kinetic-models --test headers`.

| Fixture | Repository | File | Revision |
|---|---|---|---|
| `qwen35.gguf` | [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) | `Qwen3.8-27B-UD-Q4_K_M.gguf` | `4af51b3542f775b0a8e4b310bfb3f17e01c306c9` |
| `qwen35moe.gguf` | [unsloth/Qwen3.6-35B-A3B-MTP-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF) | `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` | `5bc3e238d916f48a861bac2f8a1990a0e9b7e98d` |
| `lfm3.gguf` | [LiquidAI/LFM3-Omni-2B-GGUF](https://huggingface.co/LiquidAI/LFM3-Omni-2B-GGUF) | `LFM3-Omni-2B-Q4_K_M.gguf` | `d508b88f08a5b4286f2d2805a6a8d69a0ff44c72` |
| `lfm2.gguf` | [LiquidAI/LFM2.5-Audio-1.5B-JP-GGUF](https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B-JP-GGUF) | `LFM2.5-Audio-1.5B-JP-Q4_0.gguf` | `64b96718b341dbd5650f9e85627cecdcbd4ac61b` |
| `lfm2moe.gguf` | [DavidAU/LFM2.5-24B-A2B-Turbo-Brilliance-50X-NEO-MAX-Imatrix-GGUF](https://huggingface.co/DavidAU/LFM2.5-24B-A2B-Turbo-Brilliance-50X-NEO-MAX-Imatrix-GGUF) | `LFM2.5-24B-A2B-TBrill-MAX-NEO-Q8_0.gguf` | `fa309d64a136563539ec74a1c1e3f57536396a1f` |
| `llama.gguf` | [rippertnt/HyperCLOVAX-SEED-Text-Instruct-1.5B-Q4_K_M-GGUF](https://huggingface.co/rippertnt/HyperCLOVAX-SEED-Text-Instruct-1.5B-Q4_K_M-GGUF) | `hyperclovax-seed-text-instruct-1.5b-q4_k_m.gguf` | `3d2edd543d75372922aabd4ae538e6d181634b5d` |
| `qwen2.gguf` | [Qwen/Qwen2.5-1.5B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF) | `qwen2.5-1.5b-instruct-q4_k_m.gguf` | `91cad51170dc346986eccefdc2dd33a9da36ead9` |
| `qwen3.gguf` | [unsloth/Qwen3-4B-Instruct-2507-GGUF](https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF) | `Qwen3-4B-Instruct-2507-Q4_K_M.gguf` | `a06e946bb6b655725eafa393f4a9745d460374c9` |
| `ernie4_5.gguf` | [unsloth/ERNIE-4.5-0.3B-PT-GGUF](https://huggingface.co/unsloth/ERNIE-4.5-0.3B-PT-GGUF) | `ERNIE-4.5-0.3B-PT-Q4_K_M.gguf` | `32b9743ef4cb56af53d71a6ab4b822d191ccb4fc` |
| `hunyuan-dense.gguf` | [tencent/Hy-MT2-7B-GGUF](https://huggingface.co/tencent/Hy-MT2-7B-GGUF) | `Hy-MT2-7B-Q4_K_M.gguf` | `ab8472660ac61fac25f1af43fac2599d52a8a775` |
| `exaone4.gguf` | [LGAI-EXAONE/EXAONE-4.0-1.2B-GGUF](https://huggingface.co/LGAI-EXAONE/EXAONE-4.0-1.2B-GGUF) | `EXAONE-4.0-1.2B-Q4_K_M.gguf` | `162446400ea4596377a3ce1d3ddffa32971af0a6` |
| `exaone4-mtp.gguf` | [LGAI-EXAONE/EXAONE-4.5-33B-GGUF](https://huggingface.co/LGAI-EXAONE/EXAONE-4.5-33B-GGUF) | `EXAONE-4.5-33B-Q4_K_M.gguf` | `0e969634ef24db05151b435970297a6dee634b7e` |
| `olmo2.gguf` | [bartowski/OLMo-2-1124-13B-Instruct-GGUF](https://huggingface.co/bartowski/OLMo-2-1124-13B-Instruct-GGUF) | `OLMo-2-1124-13B-Instruct-Q4_K_M.gguf` | `00f2e4aed2fdcac064c1b9613ce2d520f54369e2` |
| `gemma3.gguf` | [ggml-org/gemma-3-1b-it-GGUF](https://huggingface.co/ggml-org/gemma-3-1b-it-GGUF) | `gemma-3-1b-it-Q4_K_M.gguf` | `f9c28bcd85737ffc5aef028638d3341d49869c27` |
| `gemma4.gguf` | [unsloth/gemma-4-26B-A4B-it-GGUF](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF) | `gemma-4-26B-A4B-it-UD-Q4_K_M.gguf` | `c099eb48e663fd284577b04978a94ffccb261841` |
| `gemma4-12b.gguf` | [unsloth/gemma-4-12b-it-GGUF](https://huggingface.co/unsloth/gemma-4-12b-it-GGUF) | `gemma-4-12b-it-Q4_K_M.gguf` | `fc034cfff751157913579611efad8462ac1be606` |
| `gemma4-e2b.gguf` | [unsloth/gemma-4-E2B-it-GGUF](https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF) | `gemma-4-E2B-it-Q4_K_M.gguf` | `0314792d7f1f7e229411f620751375812bb9faf2` |
| `granite.gguf` | [ibm-granite/granite-4.2-8b-GGUF](https://huggingface.co/ibm-granite/granite-4.2-8b-GGUF) | `granite-4.2-8b-Q4_K_M.gguf` | `93f3f6a8938ee922b784cf4e5b4203cd3428df8f` |
| `granitehybrid.gguf` | [ibm-granite/granite-4.0-h-tiny-GGUF](https://huggingface.co/ibm-granite/granite-4.0-h-tiny-GGUF) | `granite-4.0-h-tiny-Q4_K_M.gguf` | `240a43d5efe7bdcfa4205b0019bb95e4cc6e5315` |
| `granitehybrid-dense.gguf` | [ibm-granite/granite-4.0-h-micro-GGUF](https://huggingface.co/ibm-granite/granite-4.0-h-micro-GGUF) | `granite-4.0-h-micro-Q4_K_M.gguf` | `4a3531463f3d8d2294ee6f629e2f691f15f8f5c3` |
| `granitemoe.gguf` | [bartowski/granite-3.1-1b-a400m-instruct-GGUF](https://huggingface.co/bartowski/granite-3.1-1b-a400m-instruct-GGUF) | `granite-3.1-1b-a400m-instruct-Q4_K_M.gguf` | `940d2e1f9f65330615c7c8e980e6c5ac73d3360c` |
| `falcon-h1.gguf` | [tiiuae/Falcon-H1-Tiny-90M-Instruct-GGUF](https://huggingface.co/tiiuae/Falcon-H1-Tiny-90M-Instruct-GGUF) | `Falcon-H1-Tiny-90M-Instruct-Q4_K_M.gguf` | `578d134817c19bc48991ccff9a14709d664261cc` |
| `bert.gguf` | [mixedbread-ai/mxbai-embed-large-v1](https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1) | `gguf/mxbai-embed-large-v1-f16.gguf` | `b33106f585b9ce46904ad7443a3b52b7a63e231c` |
| `nomic-bert.gguf` | [nomic-ai/nomic-embed-text-v1.5-GGUF](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF) | `nomic-embed-text-v1.5.Q4_K_M.gguf` | `0188c9bf409793f810680a5a431e7b899c46104c` |
| `gemma-embedding.gguf` | [ggml-org/embeddinggemma-300M-GGUF](https://huggingface.co/ggml-org/embeddinggemma-300M-GGUF) | `embeddinggemma-300M-Q8_0.gguf` | `0f741b5a6585bd53aeb15cd1372c56f2a0f65e12` |
