# Dream <-> C interop

Dream stays a fully independent language; these are additional export
surfaces so C projects can use it without writing a line of Dream (route 1)
or by calling functions written in Dream (route 2).

## Route 1 -- libdream: the runtime as a C library

    dream lib                          # -> libdream.a + dream.h

C gets tensors, autograd, AdamW, attention, the blocked GEMM -- the whole
runtime -- through dream.h (79 declarations extracted verbatim from the
runtime, plus drm_* inline helpers for plain int/double ergonomics).
See c_api_demo.c: a 2-layer net trained entirely from C.

## Route 2 -- compile .dream to a C-linkable object

    dream build --lib mathlib.dream    # -> mathlib.o + mathlib.h

Every Dream function is Tensor* -> Tensor* in the ABI; the generated header
declares them with their real parameter names. See mathlib.dream +
use_mathlib.c.

## Linking

    cc app.c [model.o] libdream.a -o app -lstdc++ -lm -ldl -lgomp -Wl,--gc-sections   # Linux
    clang app.c [model.o] libdream.a -o app.exe -fopenmp -Xlinker /OPT:REF            # Windows

The runtime is compiled with -ffunction-sections, so the final link strips
every op your program does not use: measured 170KB with --gc-sections vs
678KB without, for the same program. Linking libdream does not bloat a C
executable.

## Ownership

Every function returning Tensor* returns a +1 reference the caller owns;
release with dream_release(t). Functions never steal arguments.

## Data capture (existing)

dream_capture.h lets a C program stream training data to CSV for a Dream
training script -- the collect half of the game-AI loop.
