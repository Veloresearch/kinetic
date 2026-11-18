# Devices and offload

Kinetic runs on x86-64 and Apple Silicon CPUs, on NVIDIA GPUs through CUDA
and on AMD, Intel and NVIDIA GPUs through Vulkan, on Windows, Linux and
macOS. A model larger than the
GPU can still use it: a dense model splits its layers between the GPU and the
CPU, and a mixture-of-experts model keeps its experts in system memory.

## Choosing the device

`run`, `chat`, `serve` and `bench` take `--device`:

| `--device` | Does |
|---|---|
| `auto` (default) | CUDA, then Vulkan, then the CPU: the first that can run the model |
| `cuda` | the first NVIDIA GPU through CUDA, or fail with the reason |
| `vulkan` | the first GPU through Vulkan, or fail with the reason |
| `cpu` | the CPU only |

`none` is accepted for `cpu`, `cuda0` for `cuda` and `vulkan0` for `vulkan`,
as llama.cpp spells them. `-ngl 0` means `--device cpu`, and any other
`-ngl N` means `--gpu-layers N` (see [Dense models larger than the
GPU](#dense-models-larger-than-the-gpu)).

With `auto`, Kinetic skips a GPU it cannot use with one line that says why:

```text
kinetic: running on the CPU; CUDA is not usable: <reason>; Vulkan is not usable: <reason>
```

The reasons are: no NVIDIA driver, a build without the CUDA kernels, a model
whose weights and caches do not fit in the GPU's free memory, or a model
that uses something the GPU code does not have yet (see the CUDA column in
[Models and formats](models.md)). The start-up line of `kinetic serve` and
the status line of `kinetic run` name the device that is used:

```text
Serving Qwen3.5-4B-MTP-Q4_K_M on NVIDIA GeForce RTX 3060 Laptop GPU at http://127.0.0.1:8080/v1
[4 tokens at 273.9 tok/s; prompt 13 tokens at 1727.7 tok/s on NVIDIA GeForce RTX 3060 Laptop GPU]
```

Some work always runs on the CPU, whatever `--device` says: prompts that
carry images, and embedding and reranking models.

## CPU

Any x86-64 CPU runs Kinetic. The fast kernels need AVX2, FMA and F16C, which
Intel CPUs have since Haswell (2013) and AMD CPUs since Excavator and every
Ryzen; Kinetic detects them at runtime. Without them it still runs, on a
much slower path.

On Apple Silicon the kernels use NEON and the SDOT instruction, which every
M-series chip has.

## Apple GPUs

On a Mac, `--device metal` (or `vulkan`, the same device) runs the Vulkan
kernels through Metal itself, with nothing to install. `auto` picks it.

`-t` sets the number of threads; by default Kinetic uses every logical CPU,
except on macOS, where it uses the performance cores: a step that waits for
an efficiency core is slower than one that leaves it idle.
Generation is limited by memory bandwidth, so more threads than physical
cores rarely help, and on some machines fewer threads are faster. Measure
with `kinetic bench model.gguf --device cpu -t N`.

## NVIDIA GPUs

GPUs from the Pascal generation (GTX 10 series, Tesla P100) on are
supported, with kernels for compute capability 6.0, 6.1, 7.0, 7.5, 8.0, 8.6,
8.9, 9.0 and 12.0, and PTX that newer drivers compile for later GPUs.

Pascal and Volta (compute capability 6.x and 7.0) have none of the tensor
cores the prompt kernels use. Prompts run there in batches of up to 1024
tokens through f32 matrix products of the quantized weights and f32
attention in tiles of 64 queries, with the delta rule in chunks of 32
tokens in f32 (and attention to heads of 64 taken 16 tokens at a time), so
prompts take longer than on Turing and newer; generation runs the same
kernels as on newer GPUs. On the Tesla P100 and V100 (6.0 and 7.0), whose
f16 arithmetic runs at twice the f32 rate, the products multiply weights and
inputs in pairs of f16 values and apply the scales in f32, about 1.5 times
as fast; their results differ from the f32 products in the last bits, as a
different order of summation would. Batches of 256 tokens or more expand
each matrix to f16 once and multiply that, faster again, with the same
results. The P100's attention tiles take queries, keys, weights and
values in f16 pairs too, adding each run of terms in f32, which rounds them
to about one part in a thousand. The Tesla P100 (6.0) also lacks the int8
dot product instruction, which its kernels do in plain integer arithmetic,
with the same results and less speed. The release binaries and `Dockerfile.cuda` build with CUDA 12
and have kernels for them; a build from source with CUDA 13 does not, see
[Building](building.md).

Kinetic opens the driver library at runtime (`nvcuda.dll` on Windows,
`libcuda.so.1` on Linux) and links nothing from the CUDA toolkit, so:

- you need the NVIDIA driver, not the CUDA toolkit;
- the `cuda` build starts on machines without a GPU and runs on the CPU.

Use a recent driver; the kernels are built with CUDA 12.8, and `nvidia-smi`
shows the highest CUDA version your driver supports in its header. The first
GPU is used unless `--gpus` names others; see [Several GPUs](#several-gpus).

## Vulkan

`--device vulkan` runs models on any GPU with a Vulkan driver: AMD, Intel and
NVIDIA. Today it runs `llama`, `qwen2`, `qwen3`, `lfm3` and `qwen35` models,
with drafting and Warm Start, and mixture-of-experts models such as
Inkling 2 (see [Mixture-of-experts offload](#mixture-of-experts-offload));
the other architectures are [coming](roadmap.md). Under `auto`, Vulkan is tried after CUDA, so an NVIDIA
GPU uses CUDA and an AMD or Intel GPU uses Vulkan.

The Vulkan kernels run in subgroups of 32 where the GPU offers them, as
NVIDIA, Intel and AMD's RDNA GPUs do. AMD's older GCN and Vega GPUs, among
them the integrated graphics of many Ryzen APUs, only run subgroups of 64;
there each half of a subgroup does the work of a subgroup of 32, and models
run on them as on the others.

LFM2.5 mixture of experts (`lfm2moe`) runs through Vulkan but does not match
the CPU yet: its logits differ by up to about 14% of the largest one, and
the first tokens of an answer can differ. Use `--device cpu`, or CUDA on an
NVIDIA GPU, for it until that is fixed.

On GPUs with cooperative f16 matrices (`VK_KHR_cooperative_matrix`, such as
AMD's RDNA3 cards and NVIDIA's RTX cards), prompts run on the matrix units:
Q8_0 and K-quant products of passes of up to 512 tokens take 16x16x16 f16
products into f32, three to four times as fast as the decode kernels on
long prompts. The status line says so, `(Vulkan, cooperative matrices)`. As
on CUDA, a prompt's state then differs slightly from the same tokens decoded
one at a time; where the prompt is cut changes no bit, and decoding and
drafts give exactly what they give without it. So that a chat's next turn
gives what it gives run cold, it runs the last reply again as prompt, from
the prompt cache when that is on. With `--no-prompt-cache` there is nothing
to resume from, so each turn runs the whole chat again and a long chat's
turns grow slow; keep Warm Start on for chats, or turn the matrix units off.
Warm Start keeps states made this way apart from the others and from other
driver versions, receipts record them, and `--off coopmat` runs prompts with
the decode kernels. Prompt passes keep buffers for 512 tokens in every
state; when those don't fit in the GPU's memory, or one is larger than the
GPU allows for a single buffer, Kinetic says so and runs prompts with the
decode kernels instead.

## Dense models larger than the GPU

When a dense model does not fit in GPU memory, Kinetic puts as many of its
last layers on the GPU as fit, with the cache, and runs the first layers on
the CPU, exactly as the CPU path runs them. Only one hidden state per token
crosses between the two. The status line says how the model was split:

```text
[... on NVIDIA GeForce RTX 3060 Laptop GPU, 24 of 33 layers, the rest on the CPU]
```

`--gpu-layers N` puts at most `N` layers on the GPU, as llama.cpp's `-ngl N`
does; use it to leave GPU memory for other programs. This works through
CUDA; on Vulkan a model runs wholly on the GPU or on the CPU.

## Several GPUs

`--gpus 0,1` spreads a dense model's layers over CUDA GPUs 0 and 1 (in CUDA's
order, which `kinetic doctor` lists): the first GPU holds the embeddings, the
output head and the first run of layers, each further GPU the run after, each
taking a share of the layers as large as its share of the free memory. Per
token only the hidden state and its norm, a few kilobytes, cross from one GPU
to the next and back to the first for the head, so the GPUs need no NVLink;
where they have direct access to each other's memory it is used, otherwise
the driver copies through system memory.

```sh
kinetic run model.gguf --gpus 0,1 "Explain this file."
kinetic serve model.gguf --gpus 0,1
kinetic bench model.gguf --gpus 0,1
```

The status line names each GPU with the number of layers it holds. The GPUs run one after the other, so a single answer is as fast as one GPU
holding the whole model would make it: the gain is room. A model that would
spill layers to the CPU on one GPU runs wholly on two, and the answer is the
same, bit for bit, as on one GPU of the same kind. Layers that still do not
fit run on the CPU, and `--gpu-layers` counts the layers over all GPUs.

`kinetic doctor` numbers the CUDA GPUs and, with more than one, times the
copies between each pair: a token-sized copy and a large one, through
system memory and with direct access where the driver offers it.
`kinetic fit --gpus 0,1` estimates where models go and how fast they run
over those GPUs together.

**If a model is much slower over several GPUs than on one,** direct access
between them is the usual cause: on some hosts, often with an IOMMU or PCIe
access control between the GPUs, the driver offers it but direct copies
crawl or stall. `kinetic doctor` shows it, and `KINETIC_CUDA_P2P=0` turns
direct access off so copies go through system memory, which costs a few
microseconds a token. With it set, `kinetic doctor` leaves out the direct
copies too, in case they stall:

```sh
KINETIC_CUDA_P2P=0 kinetic run model.gguf --gpus 0,1 "Hello"
podman run -e KINETIC_CUDA_P2P=0 ...        # in a container
```

A model's own prediction block goes to the first GPU, with the embeddings
and the output head, and drafts as on one GPU.

For now, `--gpus` takes dense models without per-layer inputs or shared
attention caches (not mixture-of-experts models or Gemma 4 E2B/E4B), and
runs the requests of `--parallel` one after another rather than in shared
passes.

### Every layer over two GPUs

`--split-mode row` (llama.cpp's `-sm row`; `tensor` means the same) splits
every layer over two GPUs instead of giving each a run of layers. Each GPU
holds a share of every layer's attention heads, delta-rule heads and
feed-forward rows, both work on every token at once, and they add up their
partial outputs twice per layer. Both GPUs work on every token, so answers
and short prompts come about one and a half times as fast as over layers,
and long prompts about a fifth faster (Qwen3.8 27B on two Tesla P100s); in
exchange, the hidden state crosses between the GPUs twice per layer rather
than once or twice per pass, so it gains most where the GPUs have direct
access to each other.

```sh
kinetic serve model.gguf --gpus 0,1 --split-mode row
```

The partial outputs add up in another order than one GPU's products, so
answers can differ slightly from those of one GPU or of a split by layers,
as they would between two kinds of GPU. Batches, rollbacks and Warm Start
states still change no bit.

It takes exactly two GPUs, and the whole model must fit on them:
`--gpu-layers` does not apply, and `kinetic fit` does not estimate it yet.
The status line names both GPUs with every layer.

## Memory

**Weights are memory-mapped.** Kinetic does not copy the model into its own
memory; the operating system reads the file's pages as they are used and
keeps them in its file cache. On the CPU, a model needs about its file size
in RAM to run at full speed. On the GPU, the weights are copied to video
memory at load time.

**The context sets the cache size.** Each conversation gets a cache for up
to `-c` tokens, 32768 by default (less if the model was trained on less;
`-c 0` takes everything the model supports). On the GPU that cache is
reserved at load time, so a model that does not fit with the default
context may fit with `-c 8192`. Models with recurrent layers (Qwen3.5,
LFM2, LFM3, Granite 4.0 H, Falcon-H1) need much less cache per token than
pure transformers.

**A quantized cache holds about twice the context.** On CUDA,
`--cache-type-k q8_0` and `--cache-type-v q8_0` (llama.cpp's `-ctk` and
`-ctv`) store keys and values in blocks of 32 int8 values with one f16
scale, 34 bytes where f16 takes 64. Every token stays in the cache, but its
keys and values are rounded, so answers differ from the f16 cache's: close
in quality, not identical, and never what the exactness checks compare
against. It is off by default. The CPU and Vulkan paths ignore it.

**Warm Start keeps prompt states** in up to 1 GiB of RAM and 2 GiB on disk,
under `--cache-dir` (by default `kinetic/warm` in `%LOCALAPPDATA%` on
Windows and `$XDG_CACHE_HOME` or `~/.cache` on Linux and macOS). `--no-prompt-cache`
turns it off.

`kinetic bench` reports the peak memory of a run.

## Mixture-of-experts offload

A mixture-of-experts model that does not fit in GPU memory still runs on the
GPU with `--device auto` or `cuda`. Kinetic puts the attention layers, the
shared weights and the cache on the GPU, gives the remaining free memory to
a cache of experts, and leaves the other experts in the memory-mapped file.
Each layer then runs the experts it picks where that is cheapest at the
moment: from the GPU cache, on the CPU, or copied over. See
[Flywheel](how-it-is-fast.md#flywheel) for how it decides.

For this to be fast, the whole file should fit in system RAM next to your
other programs. Kinetic reads the experts into memory at load time when they
fit with 3 GiB to spare, and otherwise lets the first tokens read them as
needed.

When every expert fits on the GPU, there is no offload: routing runs on the
GPU too.

Kinetic keeps part of the GPU's memory free for the driver and other
programs: 300 MiB or a twentieth of the GPU, whichever is larger. The
expert cache gets the largest size that leaves that reserve, measured on
the state the model really builds. A looped model such as Inkling 2 runs
some layers several times a token, and those layers get a larger share of
the cache.

Gemma 4 26B A4B runs on the CPU only for now: the GPU paths have no kernels
yet for its experts, which use a GELU gate.

On Vulkan, a model whose experts do not all fit keeps every expert of as
many layers as fit on the GPU, the layers a token runs most often first: a
looped model such as Inkling 2 runs the layers of its loop once per pass.
The experts of the other layers run on the CPU where they are, in the
memory-mapped file, so nothing is copied to the GPU while the model runs.
Experts of a weight type without a Vulkan kernel run on the CPU as well.

## Coming

- **More on Vulkan**: more architectures and layer splits.
- **Several GPUs** for mixture-of-experts models, Gemma 4 E2B/E4B and
  shared passes; dense models already spread over them with `--gpus`.

See the [roadmap](roadmap.md).
