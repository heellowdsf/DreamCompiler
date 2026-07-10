# GPU backend

Dispatches matmul and elementwise ops to the GPU when a device is present,
falling back to the verified CPU kernels otherwise. Two acceleration paths:

- **CUDA / cuBLAS** (NVIDIA): loaded dynamically at runtime via `dlopen` /
  `LoadLibrary`; used for matmul when available, with an optional FP16
  Tensor Core path.
- **OpenCL** (cross-vendor: AMD, Intel, NVIDIA): a tiled matmul kernel
  (32x32 tiles, 4x4 register blocking, local-memory staging, boundary
  checks) plus elementwise binary/unary kernels. This is what gives
  non-NVIDIA GPUs acceleration.

Detection is automatic and cached on first use; nothing links against CUDA
or OpenCL at build time.

## Correctness

Two layers of verification:

1. **`gpu_selftest()`** (Dream builtin): on a machine with a GPU it runs the
   GPU kernels against the CPU reference and checks they agree to fp32
   tolerance (1e-3 relative). On a CPU-only host it reports SKIP. Wired into
   `dream test` as `tests/18_gpu.dream`.

2. **`tests/matmul_kernel_sim.cpp`**: re-executes the OpenCL tiled-matmul
   kernel's exact algorithm on the CPU (same tile indexing, local-memory
   loads, and boundary handling), and compares to a naive reference over
   adversarial shapes -- including non-tile-multiple and degenerate sizes.
   This validates the kernel *logic* independently of any GPU hardware:

       g++ -O2 tests/matmul_kernel_sim.cpp -o sim && ./sim
       # 7/7 PASS, worst rel err 0.00e+00

   So even without a GPU on hand, the kernel's indexing is proven correct;
   on real hardware only the runtime/driver path remains to be exercised,
   which `gpu_selftest()` does.
