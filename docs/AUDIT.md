# Pre-open-source audit

A review pass before making Dream public. Records what was found, what was
fixed, and what remains as known issues.

## Fixed during this audit

### 1. CRITICAL: `#els` typo broke all non-Windows builds
`runtime/runtime.h` had `#els` instead of `#else` in the platform block. The
`#if defined(_WIN32)` branch always compiled on Windows (the only tested
platform), so the broken `#else` was never exercised. On Linux/macOS this left
`HMODULE`/`LoadLibraryA` undefined and the build failed immediately — every
non-Windows user would have hit this on first clone. One-character fix;
unblocked all 30 previously-failing tests.

### 2. GPU dispatch threshold was hard-coded
The 512×512 CPU/GPU cutoff was a literal `262144` in `elementwise.inc`. It was
tuned for Intel integrated graphics, but discrete GPUs want a different value.
Now configurable via the `DREAM_GPU_THRESHOLD` environment variable, defaulting
to the tuned value.

### 3. Missing scalar `min`/`max`
`min(a, b)` and `max(a, b)` didn't exist (only tensor-reduction `min_val`/
`max_val`). Added as element-wise binary ops with correct gradients (gradient
flows to the selected operand). Works for scalars and tensors alike.

## Known issues (not yet fixed)

### Adding a binary op touches four places
`BINOP` macro (definition) + `builtinRemap` (name) + `classify_binop`
(string→enum) + two `switch` statements. The string-matching dispatch in
`classify_binop` is fragile: a new op silently no-ops if you forget to wire up
all sites. Worth refactoring to a single table.

### Permissive type coercion
`"a" + 1` yields `98` (the string's first byte coerced to a number) rather than
a type error. Silent and surprising. A stricter type check would be safer, but
changing it risks breaking existing programs, so it's flagged rather than
changed.

### Loop variable scope leaks
A `for (let i = ...; ...)` binding remains visible after the loop. Minor, but
not what most users expect.

### Integer display
`let x = 2; println(x)` prints `2`, not `2.0`, even though everything is a
double internally. Cosmetic, arguably correct either way.

### GPU matmul optimization level
The OpenCL kernel is correct (verified bit-exact on all-ones inputs, and to
fp32 tolerance on random inputs against the CPU reference) but not deeply
tuned. On integrated graphics it reaches ~50 GFLOPS. This is a kernel-tuning
and hardware ceiling, not a correctness problem — the identical code on a
discrete GPU is far faster.

## Verified working

- Full test suite: `32 passed, 0 failed` on Linux after the `#els` fix.
- Autograd gradient check: 48/48.
- CPU GEMM: 17/17, ~35 GFLOPS single-threaded.
- All large-model training primitives (see MULTI_GPU.md): verified numerically
  identical to single-device baselines.

## Documentation fixes

The README referenced functions that didn't exist or used wrong signatures.
Caught by actually running every documented example:

- **`mse` didn't exist.** The README's training loop used it. Added `mse` (mean
  squared error) with correct backward — the most basic loss shouldn't be
  missing. Now a builtin, tested in `tests/35_mse.dream`.
- **`reshape` → `reshape_nd`**, **`bce` → `bce_loss`**: corrected names.
- **`export_c_header` signature was wrong** in the docs. It takes
  `(checkpoint_id, prefix, func_name)` and needs `ckpt_new`/`ckpt_add` first,
  not `(weights, name)`. Fixed the README to the real, runnable form.

## End-to-end verification of the headline feature

Confirmed the full export path works: trained a tiny model, exported to C,
compiled the generated `.c` with plain `gcc -lm`, linked a `main.c`, and ran
it — producing correct inference output with zero Dream dependencies. This is
Dream's core value proposition and it works as documented.
