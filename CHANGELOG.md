# Changelog

All notable changes to Kinetic are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
the rules in [Versioning](https://veloresearch.github.io/kinetic/versioning.html).

## [Unreleased]

### Models

- Microsoft Phi-3 Mini 4K Instruct (`phi3`) on the CPU, tested with
  Microsoft's Q4_K_M GGUF. The 128K variant is not supported.

## [0.1.1] - 2026-11-27

Inkling 2 Mini on NVIDIA and AMD GPUs, also on cards with 6 GB, macOS on
Apple Silicon, older NVIDIA and AMD GPUs, several GPUs for one model, and a
home screen in the terminal.

### Models

- Inkling 2 Mini, Thinking Machines' 26B mixture of experts, on NVIDIA GPUs
  through CUDA, in both published layouts (`inkling2_loop` and `inkling2`).
  Flywheel splits its experts between a cache on the GPU and the CPU, and
  shares the cache between layers by how many times a token runs each one,
  so the layers of its loop keep more of their experts; it runs on GPUs with
  6 GB. CUDA kernels for attention by relative distance, the mix over depth
  and SiTU experts with per-expert weight scales; experts of batches of 64
  tokens or more run on tensor cores with the bits of smaller batches (GPUs
  without tensor cores keep the earlier kernels), and experts that run on
  the CPU use AVX2 kernels on every core and give the GPU's bits, so where
  an expert runs never changes the answer. On a 6 GB NVIDIA GPU it generates
  about 25 tokens per second where llama.cpp gives about 18 on the same
  file, and repeated edits with long drafts reach about 100.
- Inkling 2 Mini through Vulkan, on AMD GPUs among others, in both layouts;
  see Devices.
- Inkling 2 Mini on the CPU, in both published layouts: ggml-org's looped
  `inkling2_loop` and Unsloth's unrolled `inkling2`. Attention by relative
  distance, attention over depth, SiTU experts with per-expert weight
  scales and the `inkling2` pre-tokenizer; every published quantization
  loads. A DSpark drafter file beside the model is found and drafts for
  RAD-1 on the CPU
  ([#38](https://github.com/Veloresearch/kinetic/pull/38)).
- Gemma 4 12B and 31B, and the 26B A4B mixture of experts on the CPU
  ([#38](https://github.com/Veloresearch/kinetic/pull/38)). Files that store
  a per-layer value as a single integer load
  ([#27](https://github.com/Veloresearch/kinetic/issues/27), reported by
  @Mamin8002).
- Mistral Nemo, with its `tekken` pre-tokenizer
  ([#38](https://github.com/Veloresearch/kinetic/pull/38)).
- NVFP4 and F8_E4M3 weights on the CPU. Files with per-tensor weight scales
  that nothing applies are refused instead of loading wrong
  ([#38](https://github.com/Veloresearch/kinetic/pull/38)).

### Platforms

- macOS on Apple Silicon, on the CPU: NEON kernels with SDOT for the
  K-quants, Q8_0 and Q4_0, f16 and bf16 weights, activation quantization
  and the chunked delta rule; `kinetic doctor` reports the machine; the
  default thread count is the performance cores
  ([#4](https://github.com/Veloresearch/kinetic/pull/4), @jasonkneen).
- The GPU of Apple Silicon Macs, through Metal, with no driver to install:
  the Vulkan kernels are written out as Metal shaders at build time and a
  Metal driver runs them (`--device metal`). Its GEMV kernels take a tile
  of four rows per invocation and their block sums in float, which is
  faster on a Mac's GPU and changes no result. A prompt runs in passes of
  up to 64 tokens, whose Q4_K, Q5_K and Q6_K matrices take kernels written
  for many tokens that give every token the bits it gets alone. A
  `macos-arm64-metal` archive and a Python wheel are built for every
  release ([#4](https://github.com/Veloresearch/kinetic/pull/4),
  [#10](https://github.com/Veloresearch/kinetic/pull/10), @jasonkneen).
- Built without the `metal` feature, the Vulkan backend also runs on a Mac
  through MoltenVK or KosmicKrisp: it opts into portability drivers, finds
  Homebrew's loader and accepts devices whose subgroups are 32 wide without
  being asked ([#4](https://github.com/Veloresearch/kinetic/pull/4),
  @jasonkneen).
- AMD's GCN and Vega GPUs through Vulkan. They only run subgroups of 64, so
  every kernel is also built for them, with each half of a subgroup doing
  the work of a subgroup of 32; `kinetic doctor` no longer turns them away
  ([#2](https://github.com/Veloresearch/kinetic/issues/2), reported by
  @harrier77).

### Devices

- NVIDIA Pascal (GTX 10 series, Tesla P100, sm_60 and sm_61) and Volta (sm_70)
  through CUDA. Without the tensor cores the prompt kernels use, prompts run
  in batches of up to 1024 tokens through f32 matrix products of the quantized
  weights, in f16 pairs on the P100 and V100, and attention in tiles, in f16
  pairs on the P100; generation runs the same kernels as on newer GPUs.
  `KINETIC_CUDA_ARCHS` builds the kernels for chosen GPUs only
  ([#12](https://github.com/Veloresearch/kinetic/pull/12),
  [#20](https://github.com/Veloresearch/kinetic/pull/20),
  [#22](https://github.com/Veloresearch/kinetic/pull/22), @brianporeilly).
- Several CUDA GPUs: `--gpus 0,1` on `run`, `chat`, `serve` and `bench` spreads
  a dense model's layers over them, with the same output as on one GPU.
  A prediction block drafts from the first GPU. Mixture-of-experts models,
  Gemma 4 E2B/E4B and shared passes stay on one GPU for now. `kinetic fit
  --gpus` estimates for them, `kinetic doctor` times the copies between each
  pair of GPUs, and `KINETIC_CUDA_P2P=0` turns off direct access between GPUs
  where it is slower than going through memory
  ([#13](https://github.com/Veloresearch/kinetic/pull/13),
  [#23](https://github.com/Veloresearch/kinetic/pull/23), @brianporeilly).
- CUDA: the batched GEMVs that check drafts compute two rows per warp when a
  batch has more than one input, which halves the reads of the inputs; each
  output keeps its order of sums, so batches still match single inputs bit
  for bit ([#36](https://github.com/Veloresearch/kinetic/pull/36),
  @brianporeilly).
- CUDA: IQ3_XXS and IQ4_XS matrices have kernels of their own, and weights
  of a type without a CUDA kernel run expanded to Q8_0 on the GPU, ordinary
  matrices and experts alike, instead of keeping the model on the CPU; the
  load says in one line which types were expanded and how much more memory
  that takes. CUDA now refuses only experts with a GELU gate and expert
  norms.
- CUDA: the memory plan measures the state it builds instead of adding up
  its buffers, keeps a reserve free (300 MiB or a twentieth of the GPU,
  whichever is larger) and gives the expert cache the largest size that
  leaves it, so a load no longer fails when the state takes more memory
  than its buffers add up to.
- Mixture-of-experts models through Vulkan, Inkling 2 among them, with
  kernels for the weight types of its published files (IQ4_NL, NVFP4,
  IQ3_XXS, IQ3_S, IQ2_S, IQ2_XS and Q3_K among them); experts of other
  types run on the CPU and other matrices are expanded to f32. When the
  experts do not fit, Flywheel keeps every expert of the layers a token runs
  most often on the GPU, the looped layers of Inkling 2 first, and runs the
  others on the CPU from the mapped file, copying nothing while the model
  runs. The plan starts from the memory the driver reports free and keeps a
  reserve.
- Vulkan prompts on the matrix units of GPUs with cooperative f16 matrices
  (`VK_KHR_cooperative_matrix`, such as AMD's RDNA3 and NVIDIA's RTX cards):
  Q8_0, Q4_K, Q5_K and Q6_K products of prompt passes of up to 512 tokens
  run as 16x16x16 f16 products into f32, about three times as fast on long
  prompts. As on CUDA, a prompt's state differs slightly from the same
  tokens decoded one at a time; where a prompt is cut changes no bit, and
  decoding and drafts stay exact. Warm Start keeps such states apart, also
  by driver version, receipts record them, and `--off coopmat` runs prompts
  as before, as does a GPU whose memory has no room for the larger buffers
  ([#34](https://github.com/Veloresearch/kinetic/pull/34), closes
  [#30](https://github.com/Veloresearch/kinetic/issues/30), @brianporeilly)
  or whose limit for one buffer is below them
  ([#43](https://github.com/Veloresearch/kinetic/pull/43), @brianporeilly).
- Vulkan: a Warm Start mark reads only the convolution slots it keeps, so a
  mark costs the same whatever the size of the prompt pass
  ([#32](https://github.com/Veloresearch/kinetic/pull/32), @brianporeilly).
- `Dockerfile.cuda` takes `CUDA_ARCHS` to build for chosen GPUs only and
  names its base images in full, so that Podman builds it without asking
  for a registry; the install guide shows Podman with the NVIDIA Container
  Toolkit's CDI devices
  ([#18](https://github.com/Veloresearch/kinetic/pull/18), @brianporeilly).

### Speed

- RAD-1 cuts each draft to the length that pays at the step costs it
  measures on the model as it is placed. It times its steps by the rows
  each pass runs and measures how far drafts come true; a draft is verified
  only up to the length that gives the most tokens per second expected, and
  not at all when a plain step gives as many. Every draft is judged against
  the output at no cost, depth by depth, and verified only once its measured
  chances pay, so drafts that don't pay cost no verify pass. Drafting no
  longer slows down a model split between the GPU and the CPU, by layers or
  by experts. The output does not change.
- RAD-1 on CUDA: when Inkling 2 runs with all its layers on one GPU, a draft
  of up to 127 tokens is verified in one pass, its experts run once for the
  whole pass, and drafts grow to fill it while they keep coming true. Every
  row keeps the bits it has in one-token decoding. Other models and
  placements verify drafts of up to 15 tokens a pass.
- Agent loops on the CPU: each request resumes from the prompt cache where
  the previous prompt ended, rounded down to 32 tokens, instead of at a
  multiple of 256, so an agent's short, repeated tool calls no longer run
  their whole tail again. `--cache-spacing N` on `run`, `chat` and `serve`
  sets the tokens between saved states
  ([#37](https://github.com/Veloresearch/kinetic/pull/37)).
- Agent loops that send the history back: tool call arguments reach the
  chat template in the order the request sent them, so the history matches
  the reply as it was generated and the next request resumes where the two
  part instead of inside the first tool call. On CUDA, a prompt that leaves
  the cached sequence resumes from the latest saved state before that point,
  as on the CPU, instead of from its first token, and of several slots a
  request goes to the one that holds its conversation.
- A request that continues a conversation on CUDA no longer runs the last
  reply again as a prompt first: the server settles the reply in pieces of
  64 tokens while the backend is idle, and a request that comes sooner
  settles what is left. Results do not change
  ([#19](https://github.com/Veloresearch/kinetic/pull/19), @brianporeilly).

### Command line

- `kinetic` alone in a terminal opens a home screen: Chat, Models, API,
  Benchmark, Doctor and Settings, and every command with a form of all its
  options. The model list shows every model on the computer (Ollama's, LM
  Studio's and Hugging Face's included), offers two to download and browses
  folders for any other `.gguf` file. Settings are saved and taken by
  `chat`, `serve`, `run` and `bench` as if typed. `chat`, `serve` and
  `bench` given no model open the same list at a terminal, and a model name
  typed slightly wrong is matched. On Windows, a double-clicked `kinetic`
  opens the home screen, then a terminal to type in.
- In `kinetic chat`, `/model` switches models, `/settings` opens Settings
  for the next answers, and `/` lists the commands.
- `kinetic update` installs the newest release for this computer and build,
  checked against its `SHA256SUMS`; `--nightly` follows the nightly build.
  A daily check for a new release, which says in one line when there is
  one, is off by default; Settings turns it on.
- `-hf` and `-hff` download a model from Hugging Face as `kinetic pull`
  does, then run it, as llama.cpp's flags do
  ([#38](https://github.com/Veloresearch/kinetic/pull/38)).
- `kinetic bench` reports a first edit and a repeated edit separately, also
  in `--json` and on the `--compare` card, and runs every edit repetition on
  a fresh backend, so no run replays an earlier answer
  ([#38](https://github.com/Veloresearch/kinetic/pull/38)).
- `kinetic pull --file` selects the file to download again; the llama.cpp
  flag translator had rejected `-f`/`--file` for every command
  ([#29](https://github.com/Veloresearch/kinetic/pull/29), @orphytype).
- `kinetic doctor` leaves software Vulkan renderers such as llvmpipe out of
  the GPUs, `kinetic fit` shows a model that fits a Vulkan-only GPU as
  running on the GPU, and `kinetic chat <repo>` no longer lists a model's
  separate MTP file as one of its quantizations
  ([#38](https://github.com/Veloresearch/kinetic/pull/38)).

### Server

- A tool call that starts while reasoning is still open ends the reasoning
  and comes back as a tool call. Qwen3.5 with thinking on often writes
  `<tool_call>` without closing `</think>` first; the call used to stay in
  `reasoning_content` and the reply ended with `finish_reason: "stop"`
  ([#21](https://github.com/Veloresearch/kinetic/pull/21), closes
  [#5](https://github.com/Veloresearch/kinetic/issues/5), @brianporeilly).
- Chat templates write `tojson` as Python's `json.dumps` does, with spaces
  after separators, `<`, `>`, `&` and `'` as they are and keys in the order
  the request sent them, as Hugging Face and llama.cpp render them. LFM3's
  template writes every tool definition through `tojson`, and OpenCode's
  tool calls with LFM3 now work.
- Chat completions give the same logprobs streamed and not streamed when a
  reply ends at a stop string or in the middle of a UTF-8 character
  ([#33](https://github.com/Veloresearch/kinetic/pull/33), closes
  [#31](https://github.com/Veloresearch/kinetic/issues/31), @Heironghuncheng).
- With reasoning off, a stray `</think>` on a line of its own is dropped
  from the reply ([#38](https://github.com/Veloresearch/kinetic/pull/38)).
- `kinetic serve` exits on SIGTERM and SIGINT, also as a container's PID 1,
  so `docker stop` ends it at once; Warm Start renames each entry into
  place, so a stop leaves none half written
  ([#16](https://github.com/Veloresearch/kinetic/pull/16), closes
  [#11](https://github.com/Veloresearch/kinetic/issues/11), reported by
  @brianporeilly).

### Rust, C and Python API

- `model.logits(prompt)` and `model.logits_chat(messages)` in the Rust
  crate return the raw full-vocabulary logits of the next token, exactly
  as plain decoding scores it, from the backend a request for them would
  use, so they are the device's own; `model.n_vocab()` is how many values
  they hold. The C API gains `kinetic_logits` and `kinetic_vocab_size`,
  and Python `Model.logits` and `Model.n_vocab`
  ([#15](https://github.com/Veloresearch/kinetic/pull/15), @yagaltd).
- `model.logits_rows(tokens)` returns the raw logits after every token of
  a token sequence as one flat row-major buffer, one `n_vocab`-value row
  per token, with token ids past the vocabulary rejected; Python returns
  it as little-endian float32 bytes, so sequences can be scored token by
  token as perplexity does
  ([#15](https://github.com/Veloresearch/kinetic/pull/15), @yagaltd).

### Changed

- On Vulkan GPUs with cooperative matrices, a prompt's state is no longer
  bit for bit the state of decoding the same tokens one at a time (see
  Devices); such states carry their own Warm Start and receipt class
  (`vulkan-coopmat`), and `--off coopmat` restores 0.1.0's behavior.
- Chat templates render `tojson` and mappings as Hugging Face and llama.cpp
  do (see Server), so a prompt with tools can differ from the one 0.1.0
  built, and so can the answer.
- `kinetic bench` prints a first and a repeated edit as two results.
- Warm Start entries are keyed by the release, so the prompt cache of 0.1.0
  starts over.
- `kinetic` alone opens the home screen instead of the short list of
  suggested models; with input or output redirected it prints the help, as
  before.

### Fixed

- CUDA, a model split between the GPU and the CPU: resetting a state zeroed
  the GPU layers but kept the CPU layers' recurrent and convolution states,
  so every request after the first gave another answer on Qwen3.5 and LFM,
  and `kinetic bench` reported drafted output not identical to plain
  decoding.
- Vulkan: a recorded command graph kept resetting a timestamp query pool
  that had been destroyed with its recording. Drivers on Metal crashed on
  it, and RADV lost the device
  ([#7](https://github.com/Veloresearch/kinetic/pull/7), @jasonkneen; fixes
  [#25](https://github.com/Veloresearch/kinetic/issues/25), reported by
  @microhobby).
- CUDA: the state a recurrent layer builds for a new sequence zeroed its
  ring and buffers on the legacy default stream, which no thread may touch
  while a CUDA graph records; with two models on one device, one recording
  a step and the other opening a sequence, the second failed with
  CUDA_ERROR_STREAM_CAPTURE_IMPLICIT. The zeros are queued on the model's
  own stream now, and a test builds recurrent state while a graph records
  ([#17](https://github.com/Veloresearch/kinetic/pull/17), @yagaltd).
- A model whose weights include a type the GPU has no kernel for says that
  the type comes from the file, that the file runs on the CPU and that
  K-quants such as Q4_K_M have kernels on that GPU, instead of reading like
  a driver problem ([#16](https://github.com/Veloresearch/kinetic/pull/16),
  [#1](https://github.com/Veloresearch/kinetic/issues/1)). On a GPU that is
  not NVIDIA's, the CUDA backend says that it needs one and that other GPUs
  run through Vulkan ([#38](https://github.com/Veloresearch/kinetic/pull/38)).
- CUDA: the expert cache is planned around the sequence the load builds,
  which the first request then takes, instead of leaving room for another
  one, so the reserve the load prints is what stays free and
  `kinetic bench -d 8192` runs on GPUs with 6 GB.

### Known issues

- On Vulkan, the logits of LFM2.5-8B-A1B (`lfm2moe`) differ from the CPU's
  by up to about 14% of the largest logit, and the first tokens of an answer
  can differ. Dense models and Inkling 2 match the CPU in our tests; use
  `--device cpu` or CUDA for LFM2.5 MoE until this is fixed.
- Gemma 4 26B A4B runs on the CPU only: its experts have a GELU gate, which
  the GPU paths do not run yet.
- With a very long context on a small GPU, plain generation falls behind
  llama.cpp: at 32,000 tokens on a 6 GB GPU, Inkling 2 Mini generates about
  10 tokens per second where llama.cpp gives about 14. The context is kept on
  the GPU at full precision and takes room from the expert cache. Up to
  8,000 tokens Kinetic is ahead.
- Inkling 2 has no image input yet; its vision projector is not read.
- When Inkling 2 verifies a long draft in one pass, the tokens of the pass
  reach the client when the pass ends, so a stream can come in bursts.
- The kernels for Pascal and Volta are built with CUDA 12, as the release
  archives and `Dockerfile.cuda` are; a build from source with CUDA 13 has
  none for them.
- `--gpus` spreads dense models only; mixture-of-experts models and Gemma 4
  E2B/E4B stay on one GPU.

Thanks to @brianporeilly, @jasonkneen, @yagaltd, @Heironghuncheng and
@orphytype for their pull requests, and to everyone who reported what broke.

## [0.1.0] - 2026-11-05

The first public release.

### Command line

- `kinetic run`, `chat`, `serve`, `bench`, `info` and `verify`.
- Common llama.cpp flags work unchanged (`-m`, `-p`, `-n`, `-c`, `-t`, `-s`,
  `--temp`, `-ngl`, `-dev` and others); the ones Kinetic does not need are
  explained instead of rejected.
- `kinetic bench` runs llama-bench's pp512 and tg128 tests, plus an edit test
  that checks drafted output against plain decoding.

### Server

- OpenAI-compatible `/v1/chat/completions`, `/v1/completions`, `/v1/models`
  and `/v1/health`, with streaming, stop strings, seeds, tool calls,
  `response_format` (JSON and JSON Schema), GBNF grammars and reasoning output.
- `/v1/embeddings` for embedding models and `/v1/rerank` for rerankers.
- Images for Qwen3.5 with its `mmproj` projector.
- Listens on 127.0.0.1 by default and refuses requests from web pages.

### Models

- Gemma 4 (`gemma4`) on the CPU and CUDA, with images through its `gemma4v`
  projector.
- LFM2.5 mixture of experts (`lfm2moe`), experimental, on the CPU and CUDA,
  with experts kept in system memory when they do not fit the GPU.
- Architectures: `qwen35`, `qwen35moe`, `lfm3`, `lfm2`, `llama`, `qwen2`,
  `qwen3`, `gemma3`, `ernie4_5`, `hunyuan-dense`, `exaone4`, `olmo2`,
  `granite`, `granitehybrid`, `granitemoe`, `falcon-h1`, `bert`, `nomic-bert`
  and `gemma-embedding`, and decoder embedding and reranking models built on
  them.
- Every GGUF weight type in common use: F32, F16, BF16, the legacy and
  K-quants, the IQ formats, TQ1_0, TQ2_0 and MXFP4.
- The `.ktc` format, version 1: one file with weights, tokenizer, chat
  template and a hashed manifest, optionally signed with Ed25519.

### Devices

- x86-64 CPUs, with AVX2 kernels.
- NVIDIA GPUs from Turing (RTX 20 series) on, through CUDA. The driver is
  loaded at runtime, so the CUDA build also starts on machines without one.
- Mixture-of-experts models larger than GPU memory (Flywheel).

### Speed

- RAD-1: speculative decoding that is exact and automatic, drafting from the
  context, from earlier requests and from a model's own multi-token
  prediction block.
- Warm Start: prompts that begin like an earlier one resume from its exact
  state, in memory and on disk.
- Shortlist: exact top tokens without reading the whole output layer.

### Libraries

- Python package `kinetic`, C API (`kinetic.h`, `kinetic_capi`) and the Rust
  crate `kinetic`.
- Release archives for Windows and Linux, CPU and CUDA, and Dockerfiles.

[Unreleased]: https://github.com/Veloresearch/kinetic/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/Veloresearch/kinetic/releases/tag/v0.1.1
[0.1.0]: https://github.com/Veloresearch/kinetic/releases/tag/v0.1.0
