//dream runtime   common header
#pragma once
#define NOMINMAX
#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif
#include <iostream>
#include <map>
#include <string>
#include <algorithm>
#include <vector>
#include <set>
#include <unordered_set>
#include <unordered_map>
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <cmath>
#include <random>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <atomic>
#include <functional>
#if defined(__AVX2__) || defined(__AVX__)
#include <immintrin.h>   //avx intrinsics for the f32 matmul kernel
#endif

#if defined(_WIN32)
#include <windows.h>
#include <io.h>      // _isatty/_fileno for the console keep-open check
#else
#include <dlfcn.h>
#define HMODULE void*
#define LoadLibraryA(x) dlopen(x, RTLD_LAZY)
#define GetProcAddress dlsym
#define GetModuleHandleA(x) dlopen(NULL, RTLD_LAZY)
#endif

#define CL_TARGET_OPENCL_VERSION 120
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>

struct Tensor {
    //llvm visible fields (do NOT reorder, indices 0-6 hardcoded in IR) ===
    double*     data;       // 0: raw data pointer
    double*     grad;       // 1: gradient pointer
    int         size;       // 2: total element count
    int         d0, d1, d2, d3; // 3-6: legacy shape (backward compat)

    //runtime only fields (not accessed from LLVM IR) ===
    bool        is_leaf;
    std::string grad_fn;
    Tensor      *lhs, *rhs;
    Tensor**    parents;
    int         num_parents;
    std::string bwd_name, bwd_src;
    bool        is_variable;
    int         refcount;
    int         graph_depth;    // graph chain depth; auto-detached past the soft cap
    bool        data_dirty = false;  // f32 result: fdata is live, data needs sync before read

    //new!!!: Dynamic shape + strides (zero-copy views) ---
    int         ndim;           // actual number of dimensions (1-8)
    int         dims[8];        // shape per dimension
    int         strides[8];     // elements to skip per dimension
    int         data_offset;    // offset into data buffer (for views)
    bool        owns_data;      // false = view, don't free data/grad
    int         version;        // autograd safety: incremented on in-place mutation
    Tensor*     base;           // if view, points to the tensor that owns the memory

    //data type support (fp32/fp64)
    int         dtype;         // 0 = float64 (double), 1 = float32
    float*      fdata;         // float32 storage (non-null when dtype==1)
    float*      fgrad;    // float32 gradient storage
};

//compute contiguous strides from dims
static inline void compute_strides(Tensor* t) {
    t->strides[t->ndim - 1] = 1;
    for (int i = t->ndim - 2; i >= 0; --i)
        t->strides[i] = t->strides[i + 1] * t->dims[i + 1];
}

//sync legacy d0-d3 from dims[]
static inline void sync_legacy_dims(Tensor* t) {
    t->d0 = t->ndim >= 1 ? t->dims[0] : 1;
    t->d1 = t->ndim >= 2 ? t->dims[1] : 1;
    t->d2 = t->ndim >= 3 ? t->dims[2] : 1;
    t->d3 = t->ndim >= 4 ? t->dims[3] : 1;
}

//check if tensor is contiguous (strides match a packed layout)
static inline bool is_contiguous(Tensor* t) {
    int expected = 1;
    for (int i = t->ndim - 1; i >= 0; --i) {
        if (t->strides[i] != expected) return false;
        expected *= t->dims[i];
    }
    return true;
}

//materialize: if non-contiguous, copy to a fresh contiguous buffer
static inline Tensor* materialize(Tensor* t);

