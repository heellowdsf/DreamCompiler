# Runtime architecture

The runtime is organized into five layers by responsibility. All modules
are `#include`d into a single translation unit (`runtime.cpp`) in dependency
order -- the directory grouping is organizational, and does not change how
the runtime compiles or links. This keeps the build model simple (one object
file, cached and reused across compiles) while making the source navigable.

```
runtime/
  runtime.h          Public types (Tensor) and shared includes.
  runtime.cpp        Orchestrator: includes every module in order, holds the
                     entry point and the crash handler.

  core/              Tensor and memory primitives.
    memory_pool.inc    Bucketed allocator with recycling.
    tensor.inc         Tensor creation, refcounting, materialize, NaN/Inf,
                       precision (f32 mirror), console-pause-on-exit.
    indexing.inc       Element and row indexing / assignment.

  ops/               Mathematical operators.
    elementwise.inc    Binary/unary ops, matmul, the blocked GEMM, softmax,
                       reductions (incl. axis reductions), transpose, reshape.
    math_extra.inc     Additional math helpers.
    autograd.inc       Reverse-mode backward for every differentiable op.

  nn/                Neural-network layers.
    im2col.inc         im2col / col2im, shared by conv and its backward.
    conv.inc           Convolution and pooling.
    training.inc       linear, embedding, attention (+ causal), RNN/LSTM,
                       losses, AdamW, gradcheck, gemm_selftest, augment.

  io/                I/O, data, and language runtime support.
    strings.inc        String tensors.
    structs.inc        Struct construction / field access.
    closures.inc       Function references and closures.
    image_io.inc       CSV / image loading, batch iterators, accuracy.
    builtins.inc       Misc builtins: save/load, precision, augment entry
                       points, and other Dream-callable helpers.

  backend/           Hardware acceleration.
    gpu.inc            CUDA/cuBLAS and OpenCL dispatch (auto-detected at
                       runtime; falls back to the CPU kernels above).
```

## Why a single translation unit

Every module is compiled together, so a function defined earlier in the
include order is directly visible later without a forward declaration. The
compiler (`dream build` / `dream run` / `dream lib`) precompiles this unit
to one cached `.o` keyed by a fingerprint of all runtime sources, so a
warm build links in a fraction of a second. The fingerprint and the
symbol-manifest scan both descend recursively through these subdirectories.
