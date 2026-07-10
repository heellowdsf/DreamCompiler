# Dream

An AI learning programming languages. Write
models in a clean Python-like syntax, train them with real automatic
differentiation, and — the headline feature — **export a trained model to a
single standalone C file** that compiles anywhere with no dependencies beyond
`math.h`.

---

## Table of contents

- [What Dream can do today](#what-dream-can-do-today)
- [Install and build](#install-and-build)
- [Your first program](#your-first-program)
- [Training an MLP](#training-an-mlp)
- [The language](#the-language)
- [Built-in functions](#built-in-functions)
- [GPU acceleration](#gpu-acceleration)
- [Exporting to C](#exporting-to-c)
- [Large-model training primitives](#large-model-training-primitives)
- [Configuration](#configuration)
- [Known limitations](#known-limitations)
- [Project layout](#project-layout)
- [Contributing](#contributing)

---

## What Dream can do today

- **Train neural networks end-to-end** with reverse-mode autodiff. Verified on
  MLPs (100% accuracy on synthetic 10-class data) and a small char-level GPT.
- **Real optimizers**: SGD, Adam, AdamW, EMA, gradient accumulation — not just
  plain SGD.
- **Transformer building blocks**: attention (standard and FlashAttention),
  layer norm, softmax/log-softmax, dropout.
- **A CPU GEMM that reaches ~35 GFLOPS single-threaded** (cache-blocked,
  register-tiled, AVX2/FMA) and scales with OpenMP across cores.
- **Optional GPU acceleration** via OpenCL (and cuBLAS when available), with an
  automatic CPU/GPU dispatch threshold tuned for your hardware.
- **Export to C**: a trained model becomes one `.h` + one `.c`, compilable with
  any C99 compiler (gcc, clang, `arm-none-eabi-gcc`, `avr-gcc`) with zero
  dependencies beyond the standard library.
- **Interop with PyTorch** via safetensors — load a PyTorch model and run
  inference, or export Dream weights back out.
- **Reproducible and numerically careful**: `set_seed()` for determinism,
  `log_softmax`+`nll_loss` (not the naive `log(softmax)` that overflows), NaN
  detection, gradient clipping.
- **Cross-platform**: Windows, Linux, and macOS. (The compiler is portable C++17
  plus LLVM; the runtime is emitted C++ compiled by your local clang.)

---

## Install and build

### Prerequisites

- **LLVM + Clang** (version 15 or newer). Dream uses LLVM to build its compiler
  and invokes `clang++` at runtime to compile the generated code.
- **A C++17 compiler** (the same clang is fine) and **CMake 3.15+**.
- **Optional**: an OpenCL runtime for GPU acceleration. Nothing to install at
  build time — Dream loads OpenCL dynamically if it's present.

### Build

```bash
git clone https://github.com/heellowdsf/DreamCompiler.git Dream
cd Dream
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DLLVM_DIR=$(llvm-config --cmakedir) ..
make -j$(nproc)
```

This produces the `DreamCompiler` executable (called `dream` when installed on
your PATH). On Windows, use the LLVM you installed (e.g. `C:/LLVM`) and point
`-DLLVM_DIR` at its `lib/cmake/llvm`.

### Verify the build

```bash
./bin/DreamCompiler test tests
```

You should see `32 passed, 0 failed`. This runs the full correctness suite:
autograd gradient checks, GEMM, determinism, and the large-model training
primitives.

---

## Your first program

Every Dream program has an `app()` entry point. Save this as `hello.dream`:

```dream
fn app() {
    let x = 21.0;
    let y = x * 2.0;
    println(y);
    return 0.0;
}
```

Run it:

```bash
dream run hello.dream
```

The first run compiles the runtime (a one-time step, cached afterward), then
prints `42`.

---

## Training an MLP

Here's a complete, runnable training loop — a 2-layer network learning a
synthetic classification task:

```dream
fn app() {
    set_seed(42);

    // synthetic data: 64 samples, 8 features, 3 classes
    let X = randn(64, 8);
    let Y = randn(64, 3);

    // parameters
    let W1 = kaiming_init(8, 16);
    let b1 = zeros(1, 16);
    let W2 = kaiming_init(16, 3);
    let b2 = zeros(1, 3);

    let epoch = 0;
    while (epoch < 100) {
        // forward
        let h = relu(X @ W1 + b1);
        let logits = h @ W2 + b2;
        let loss = mse(logits, Y);

        // backward
        backward(loss);

        // update (AdamW)
        adamw_update(W1, 0.001, 0.9, 0.999, 0.00000001, 0.01); zero_grad(W1);
        adamw_update(b1, 0.001, 0.9, 0.999, 0.00000001, 0.01); zero_grad(b1);
        adamw_update(W2, 0.001, 0.9, 0.999, 0.00000001, 0.01); zero_grad(W2);
        adamw_update(b2, 0.001, 0.9, 0.999, 0.00000001, 0.01); zero_grad(b2);

        if (epoch % 20 == 0) { print("epoch "); print(epoch); print(" loss "); println(loss[0]); }
        epoch = epoch + 1;
    }
    return 0.0;
}
```

For higher-level layer abstractions (`Linear`, `Conv2d`, `ResBlock`), see
`examples/nn.dream`, which you can `import`.

---

## The language

Dream's syntax is deliberately small and Python-like.

### Values and variables

Everything numeric is a tensor; a scalar is just a 1×1 tensor. Variables are
declared with `let` and are mutable.

```dream
let a = 3.0;          // scalar
let v = zeros(1, 10); // 1x10 tensor
let m = randn(4, 4);  // 4x4 tensor
a = a + 1.0;          // reassignment
```

### Functions

```dream
fn add(a, b) {
    return a + b;
}
```

The entry point is always `fn app()`.

### Control flow

```dream
if (x > 0.0) { println(x); } else { println(0.0); }

let i = 0;
while (i < 10) { i = i + 1; }

for (let j = 0; j < 10; j = j + 1) { println(j); }
```

### Operators

Arithmetic `+ - * /`, matrix multiply `@`, comparison `< > <= >= == !=`,
logical `&& || !`. These work element-wise on tensors and broadcast where shapes
are compatible (e.g. adding a `(1, D)` bias to an `(N, D)` batch).

### Structs

```dream
struct Point { x, y }

fn app() {
    let p = Point(3.0, 4.0);
    println(p.x);
    return 0.0;
}
```

Structs are how Dream builds layer abstractions — see `examples/nn.dream`.

---

## Built-in functions

Dream ships with ~270 built-ins. The most useful, by category:

**Tensor creation**: `zeros`, `ones`, `randn`, `kaiming_init`, `arange`,
`linspace`, `eye`.

**Element-wise math**: `exp`, `log`, `sqrt`, `abs`, `sin`, `cos`, `tan`,
`pow`, `sigmoid`, `tanh`, `relu`, `min`, `max`, `clip`.

**Reductions**: `sum`, `mean`, `max_val`, `min_val`, `var`, `norm`, `argmax`,
`sum_dim`, `mean_dim`, `max_dim`, `cumsum`.

**Shape**: `transpose`, `concat`, `flatten`, `reshape_nd`, `print_shape`.

**Neural-net ops**: `softmax`, `log_softmax`, `layer_norm`, `dropout`,
`attention_causal`, `flash_attention`, `conv2d`, `maxpool2d`.

**Losses**: `mse`, `cross_entropy`, `nll_loss`, `bce_loss`.

**Autograd**: `backward`, `grad`, `zero_grad`, `clip_grad`.

**Optimizers**: `sgd_step`, `adam_step`, `adamw_update`, `ema_update`,
`grad_accum_update`.

**I/O and interop**: `save_matrix`, `load_safetensors`, `export_c_header`.

Run any `*_selftest()` built-in (e.g. `gpu_selftest()`, `flash_attention_selftest()`)
to verify a subsystem on your machine.

---

## GPU acceleration

If an OpenCL device is present, Dream automatically offloads large matmuls and
element-wise ops to it. Small operations stay on the CPU, where the kernel-launch
and data-transfer overhead would otherwise dominate.

```dream
fn app() {
    gpu_warmup();          // pre-compile GPU kernels (one-time, ~1-2s)
    let a = randn(1024, 1024);
    let b = randn(1024, 1024);
    let c = a @ b;         // runs on the GPU
    println(sum(c));
    return 0.0;
}
```

Two things worth knowing:

1. **Always call `gpu_warmup()` before timing.** The first GPU call pays a
   one-time cost to initialize OpenCL and JIT-compile the kernels. Without
   warmup, that cost is wrongly attributed to whatever op runs first.
2. **The CPU/GPU dispatch threshold is tunable** (see [Configuration](#configuration)).
   The default (512×512) was measured on Intel integrated graphics; a discrete
   GPU with faster transfer benefits from a lower threshold.

Verify your GPU computes correctly with `gpu_selftest()` — it compares GPU
results against the trusted CPU reference.

---

## Exporting to C

The reason Dream exists. After training, collect your weights into a checkpoint
and export:

```dream
fn app() {
    // ... train, producing W1, b1, W2, b2 ...

    let ckpt = ckpt_new();
    ckpt_add(ckpt, "fc1.weight", W1);
    ckpt_add(ckpt, "fc1.bias",   b1);
    ckpt_add(ckpt, "fc2.weight", W2);
    ckpt_add(ckpt, "fc2.bias",   b2);

    export_c_header(ckpt, "my_model", "mlp_forward");
    return 0.0;
}
```

This writes `my_model.h` and `my_model.c`. The C file contains the weights as
static arrays and the forward pass as plain C — no Dream runtime, no
dependencies beyond `math.h`:

```bash
arm-none-eabi-gcc -O2 -c my_model.c   # cross-compile for embedded
```

You can read the output, edit it, and embed it in someone else's C project
without them knowing Dream exists. See `examples/embedded_deploy/` and
`examples/pytorch_interop/export_to_c_demo.dream`.

---

## Large-model training primitives

Dream includes verified implementations of the core techniques used to train
large models. Each is validated to be **numerically identical** to its
single-device equivalent — the parallelism changes speed and memory, not
results. (These verify the *math*; running across a real GPU cluster additionally
requires NCCL transport, which is stubbed at the interface.)

| Technique | Self-test | What it does |
|-----------|-----------|--------------|
| Data parallelism | `dist_selftest` | Split the batch across devices; gradients sum |
| Tensor parallelism | `tensor_parallel_selftest` | Split one layer's matmul across devices |
| Pipeline parallelism | `pipeline_parallel_selftest` | Split model depth into stages |
| Micro-batch scheduling | `microbatch_selftest` | Fill the pipeline, shrink the bubble |
| ZeRO optimizer sharding | `zero_adamw_selftest` | Shard optimizer state across devices |
| Ring all-reduce | `comm_selftest` | Bandwidth-optimal gradient sync |
| FlashAttention | `flash_attention_selftest` | O(N) memory attention, fwd + bwd |
| Gradient checkpointing | `checkpoint_selftest` | Trade compute for activation memory |
| bf16 mixed precision | `bf16_selftest` | Half-precision compute, fp32 master weights |
| Mixture of Experts | `moe_selftest` | Top-k expert routing (Mixtral/Grok style) |
| Data loader | `dataloader_selftest` | Correct parallel sharding/shuffle/batching |
| Checkpoint/resume | `checkpoint_resume_selftest` | Fault-tolerant training state |

See `docs/MULTI_GPU.md` for the full design and the honest boundary between what
is verified in-process and what needs real multi-GPU hardware.

---

## Configuration

Dream reads a few environment variables:

- **`DREAM_GPU_THRESHOLD`** — matmuls with `M*N` below this many elements run on
  the CPU; at or above, they go to the GPU. Default `262144` (512×512). Lower it
  on a discrete GPU (`export DREAM_GPU_THRESHOLD=65536`), or set a huge value to
  effectively force CPU.
- **`OMP_NUM_THREADS`** — controls CPU parallelism for the GEMM and other
  OpenMP loops. Set it to your core count.

---

## Known limitations

Being honest about what's rough:

- **GPU matmul is not deeply optimized.** The OpenCL kernel is correct and
  tiled, but it's not a hand-tuned BLAS. On integrated graphics it reaches tens
  of GFLOPS — fine for prototyping, far from a discrete GPU's throughput. The
  same code on a discrete GPU is dramatically faster; the bottleneck is the
  kernel's optimization level and the hardware, not correctness.
- **`min`/`max` gradients pick a single branch at ties** (gradient flows to the
  left operand), matching common framework behavior.
- **Convolutions are less battle-tested** than the MLP/Transformer path. The
  `conv2d` API works but has seen less real training.
- **Multi-GPU is verified as math, not transport.** The parallelism primitives
  prove the numerics; wiring them to real NCCL across machines is future work.
- **Some type coercions are permissive.** For example, `"a" + 1` coerces the
  string's first byte rather than erroring. Be explicit with types.

---

## Project layout

```
src/            The compiler: lexer, parser, AST, LLVM IR generation
runtime/        The runtime, #included into one translation unit:
  core/         Memory pool, tensor structure, indexing
  ops/          Element-wise ops, GEMM, autograd
  nn/           Conv, im2col, training primitives (attention, parallelism)
  io/           Strings, structs, closures, image I/O, built-in registry
  backend/      OpenCL/CUDA loading, GPU kernels, topological sort
examples/       Runnable programs, from basic MLPs to char-GPT
tests/          The correctness suite (dream test tests)
docs/           Design docs (MULTI_GPU.md)
interop/        PyTorch/safetensors bridge
```

---

## Contributing

Dream is a research/hobby project made open. If you want to help:

- **Run `dream test tests` before and after your change** — keep it green.
- **The GPU matmul kernel has the most headroom.** Vectorized loads, better
  tiling, and `mad()` could meaningfully raise throughput on real hardware.
- **Adding a binary op** currently touches three places (the `BINOP` macro, the
  `classify_binop` dispatcher, and two `switch` statements in `gpu.inc`). This
  is a known piece of friction worth refactoring.
- **File issues with a minimal `.dream` repro.** The smaller the program that
  shows the bug, the faster it gets fixed.

---

## License

See `LICENSE`.
