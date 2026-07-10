# Multi-GPU / multi-core training

Real data-parallel training you can run and time on your own hardware.

## Files

- `data_parallel_train.dream` -- a genuine data-parallel training loop
  (batch sharded across 4 ranks, gradients all-reduce-summed, AdamW step).
  Converges to accuracy 1.0; prints wall time for 30 epochs.
- `benchmark_scaling.bat` -- Windows (cmd). Double-click or run in a terminal.
- `benchmark_scaling.ps1` -- Windows (PowerShell).
- `benchmark_scaling.sh` -- Linux / macOS.

## Measure scaling on your machine (Windows)

Command Prompt (cmd):

    benchmark_scaling.bat C:\path\to\DreamCompiler.exe

PowerShell:

    .\benchmark_scaling.ps1 C:\path\to\DreamCompiler.exe

If PowerShell blocks the script, first run in that window:

    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

Either script runs the training at 1/2/4/8 threads and prints wall time.
Dream's GEMM is OpenMP-parallel, so more CPU threads make each step faster --
you should see wall time drop as threads increase.

### If threads make no difference

That means OpenMP isn't active. Check:
- Your `dream run` compiled without errors (OpenMP is on by default via
  `-fopenmp`).
- `libomp.dll` is findable at runtime. It ships with LLVM; if you get a
  "libomp.dll not found" error, add your LLVM `bin` folder to PATH
  (e.g. `C:\LLVM\bin`).

## Manual single run with a fixed thread count

cmd:

    set OMP_NUM_THREADS=4
    DreamCompiler.exe run data_parallel_train.dream

PowerShell:

    $env:OMP_NUM_THREADS=4
    .\DreamCompiler.exe run data_parallel_train.dream

## The math behind it

`dist_selftest` proves N-way data parallelism produces the exact same
gradient as a single device with N-times the batch (bit-for-bit). The speedup
from parallelism is free -- it does not change the result, only the time.
`speedup_report(compute_ms, grad_mb, gpus)` predicts the speedup for your
model size and warns if a configuration would be communication-bound.

Real multi-GPU data parallelism scales by the same math; multi-core CPU here
is the version you can measure on a single-GPU machine.

## Single-GPU is unaffected

Distributed primitives are separate functions; single-GPU training runs with
no overhead. Verified: GEMM ~35 GFLOPS, gradcheck 48/48 unchanged.
