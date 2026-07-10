// Dream Runtime - Tensor Core
// =====================================================================
// PURE REFCOUNT MEMORY MODEL
//
//   alloc_tensor()        → refcount=1 (caller owns +1)
//   tensor_incref(t)      → refcount += 1
//   tensor_decref(t)      → refcount -= 1
//                           if refcount drops to 0:
//                             - decref lhs/rhs/base/parents[i]
//                               (graph edges are owning references)
//                             - release data/grad buffers
//                             - delete t
//
// GRAPH EDGE HELPERS — always go through these, NEVER assign directly:
//   tensor_set_lhs(parent, child)
//   tensor_set_rhs(parent, child)
//   tensor_set_base(parent, child)
//   tensor_set_parents(parent, par_array, n)
//   tensor_clear_graph(t)              -- used by zero_grad
//
// AST CONVENTIONS (already in AST.cpp):
//   - `let X = expr`: emit decref(old X), incref(new), store. After this,
//     the alloca holds +1 ref to the value. When let goes out of scope,
//     scope cleanup decrefs.
//   - Function return: ReturnStmt increfs rv if rv is a let-local
//     (so scope cleanup's decref doesn't free it before caller incref).
//
// NO GC. NO is_variable. NO shadow stack.
// =====================================================================

// Bit-pattern NaN/Inf checks. The runtime is built with -ffast-math, which
// assumes no NaN/Inf and both optimizes std::isnan/isinf away and warns about
// them. These detect via the IEEE754 bit pattern (exponent all ones), so they
// keep working regardless of fast-math. Used everywhere the runtime must still
// detect NaN/Inf (loss guards, print, nan_to_num, etc.).
static inline bool dream_isnan(double v) {
    uint64_t b; std::memcpy(&b, &v, sizeof(b));
    return ((b >> 52) & 0x7FF) == 0x7FF && (b & 0xFFFFFFFFFFFFFULL) != 0;
}
static inline bool dream_isinf(double v) {
    uint64_t b; std::memcpy(&b, &v, sizeof(b));
    return ((b >> 52) & 0x7FF) == 0x7FF && (b & 0xFFFFFFFFFFFFFULL) == 0;
}

// ---------------------------------------------------------------------
//  Console keep-open on Windows double-click.
//  Launched from cmd/PowerShell: the console has >= 2 attached processes,
//  we do NOTHING -- identical to how C programs and Python scripts behave
//  from a terminal. Launched by double-click: Explorer creates a fresh
//  console that vanishes the instant we exit, so we pause for Enter to let
//  the user read the output. DREAM_NO_PAUSE=1 disables; redirected stdio
//  also disables (scripts/pipes never block).
// ---------------------------------------------------------------------
#if defined(_WIN32)
static void dream_console_pause_at_exit() {
    DWORD pids[2];
    DWORD n = GetConsoleProcessList(pids, 2);
    if (n > 1) return;                              // started from a terminal
    if (std::getenv("DREAM_NO_PAUSE")) return;
    if (!_isatty(_fileno(stdin)) || !_isatty(_fileno(stdout))) return;
    std::fputs("\n[dream] finished - press Enter to close...", stdout);
    std::fflush(stdout);
    (void)std::getchar();
}
struct DreamConsolePauseReg {
    DreamConsolePauseReg() { std::atexit(dream_console_pause_at_exit); }
};
static DreamConsolePauseReg g_dream_console_pause_reg;
#endif

// ---------------------------------------------------------------------
//  High-precision reduction: Neumaier compensated summation.
//
//  Naive s += x[i] loses precision at scale: when |s| >> |x[i]|, the low bits
//  of x[i] are rounded away. A million-element sum can have ~1e-8 relative error.
//
//  Neumaier (an improved Kahan) uses a compensation variable c to recover them.
//  The extra cost is a few flops per element; reductions are memory-bandwidth bound,
//  so this arithmetic hides under the loads and exact mode is not much slower.
//  Precision: relative error drops from O(n*eps) to O(eps), independent of n.
//
//  __attribute__((optimize("no-fast-math"))): the runtime is built with -ffast-math,
//  which assumes associativity and would algebraically fold (sum-t)+x[i] to 0,
//  breaking the compensation. This attribute disables fast-math for this function only.
//  (Only this small function is affected; other ops keep their vectorization.)
// ---------------------------------------------------------------------
#if defined(__clang__)
__attribute__((optnone))
#elif defined(__GNUC__)
__attribute__((optimize("no-fast-math","no-associative-math")))
#endif
static double neumaier_sum(const double* x, int n) {
    double sum = 0.0, c = 0.0;
    for (int i = 0; i < n; ++i) {
        volatile double t = sum + x[i];   // volatile blocks cross-statement FP reordering
        if (std::fabs(sum) >= std::fabs(x[i]))
            c += (sum - t) + x[i];
        else
            c += (x[i] - t) + sum;
        sum = t;
    }
    return sum + c;
}

#if defined(__clang__)
__attribute__((optnone))
#elif defined(__GNUC__)
__attribute__((optimize("no-fast-math","no-associative-math")))
#endif
static double neumaier_sum_strided(const double* x, int n, int stride) {
    double sum = 0.0, c = 0.0;
    for (int i = 0; i < n; ++i) {
        double v = x[(size_t)i * stride];
        volatile double t = sum + v;
        if (std::fabs(sum) >= std::fabs(v))
            c += (sum - t) + v;
        else
            c += (v - t) + sum;
        sum = t;
    }
    return sum + c;
}


static std::vector<Tensor*> g_arena;   // bookkeeping: every live tensor
static int g_alloc_count = 0;          // diagnostic counter

// Forward declarations
extern "C" Tensor* tensor_decref(Tensor* t);
extern "C" Tensor* tensor_incref(Tensor* t);
Tensor* create_scalar(double v);
// Flush a dirty f32 mirror into data (double) on demand. Called before any
// read of data on a tensor that may hold a live f32 result. No-op if clean.
static inline void sync_f32_to_data(Tensor* t) {
    if (t && t->data_dirty && t->fdata) {
        for (int i = 0; i < t->size; ++i) t->data[i] = (double)t->fdata[i];
        t->data_dirty = false;
    }
}

static inline Tensor* materialize(Tensor* t);

Tensor* alloc_tensor(int d0, int d1, int d2, int d3) {
    g_alloc_count++;
    // Dimension validity: negative dims are an error (previously returned silent garbage).
    if (d0 < 0 || d1 < 0 || d2 < 0 || d3 < 0) {
        std::cerr << "RUNTIME ERROR: negative tensor dimension ("
                  << d0 << "," << d1 << "," << d2 << "," << d3 << ")\n" << std::flush;
        d0 = d1 = d2 = d3 = 1;   // degrade to 1x1 instead of crashing (stay stable)
    }
    // int64 intermediate to prevent overflow: 100000*100000 wraps in int32 to a
    // negative/small value, causing a wrong alloc size and later OOB. Use int64.
    long long sz64 = (long long)d0 * d1 * d2 * d3;
    const long long MAX_ELEMS = 500000000LL;   // 500M elems = 4GB double; beyond = misuse
    if (sz64 > MAX_ELEMS) {
        std::cerr << "RUNTIME ERROR: tensor too large (" << d0 << "x" << d1
                  << "x" << d2 << "x" << d3 << " = " << sz64
                  << " elements, limit " << MAX_ELEMS << ")\n"
                  << "  check dimensions — this usually means a shape bug\n" << std::flush;
        d0 = d1 = d2 = d3 = 1;
        sz64 = 1;
    }
    int sz = (int)sz64;
    if (sz <= 0) sz = 1;
    Tensor* t = new Tensor();
    t->data = g_pool.alloc(sz);
    t->grad = g_pool.alloc_grad(sz);
    t->size = sz;
    t->d0 = d0; t->d1 = d1;
    t->d2 = d2; t->d3 = d3;
    t->is_leaf = true;
    t->grad_fn = "";
    t->lhs = nullptr;
    t->rhs = nullptr;
    t->parents = nullptr;
    t->num_parents = 0;
    t->bwd_name = "";
    t->bwd_src = "";
    t->is_variable = false;     // legacy field; kept for ABI but unused
    t->graph_depth = 0;         // leaf depth is 0
    t->refcount = 1;            // alloc gives caller +1 ownership.
                                // Mirrors all C++ runtime functions: alloc → return.
                                // The +1 must be balanced by decref or transferred
                                // (e.g. via tensor_set_lhs which adds another +1
                                // and the original is decref'd elsewhere).
    t->ndim = (d3 > 1) ? 4 : (d2 > 1) ? 3 : (d1 > 1) ? 2 : 1;
    t->dims[0] = d0; t->dims[1] = d1; t->dims[2] = d2; t->dims[3] = d3;
    for (int i = 4; i < 8; ++i) t->dims[i] = 1;
    compute_strides(t);
    t->data_offset = 0;
    t->owns_data = true;
    t->version = 0;
    t->base = nullptr;
    t->dtype = 0;
    t->fdata = nullptr;
    t->fgrad = nullptr;
    g_arena.push_back(t);
    return t;
}

// Allocate a float32 tensor (for fast GPU transfer)
Tensor* alloc_tensor_f32(int d0, int d1, int d2, int d3) {
    Tensor* t = alloc_tensor(d0, d1, d2, d3);
    t->dtype = 1;
    t->fdata = new float[t->size]();
    t->fgrad = new float[t->size]();
    return t;
}

extern "C" Tensor* tensor_incref(Tensor* t) {
    if (!t) return nullptr;
    if (t->refcount < 0) {
        std::cerr << "[BUG] tensor_incref called on tensor with refcount=" << t->refcount
                  << " ptr=" << (void*)t
                  << " (likely use-after-free)\n" << std::flush;
        return t;
    }
    t->refcount++;
    return t;
}

// Drop one ref. If refcount hits 0, free t and cascade-decref its graph edges.
//
// Implemented ITERATIVELY with an explicit work stack to avoid stack overflow
// when the computation graph is deep.
extern "C" Tensor* tensor_decref(Tensor* t) {
    if (!t) return nullptr;

    std::vector<Tensor*> work;
    work.push_back(t);

    while (!work.empty()) {
        Tensor* x = work.back();
        work.pop_back();
        if (!x) continue;

        // DIAGNOSTIC: detect use-after-free. If we see a tensor with refcount
        // already <= 0, it means someone is decref'ing a freed tensor.
        if (x->refcount <= 0) {
            std::cerr << "[BUG] tensor_decref called on tensor with refcount=" << x->refcount
                      << " ptr=" << (void*)x
                      << " grad_fn=" << x->grad_fn
                      << " shape=" << x->d0 << "x" << x->d1 << "\n" << std::flush;
            // Don't proceed — this would double-free
            continue;
        }

        x->refcount--;
        if (x->refcount > 0) continue;

        // Struct tensors own their members (+1 each, set by dream_struct_set).
        // Extract member pointers BEFORE the data buffer is released, and
        // push them onto the work stack for cascading decref.
        if (x->data && x->grad_fn.rfind("struct:", 0) == 0) {
            for (int i = 0; i < x->size; ++i) {
                uintptr_t bits = 0;
                std::memcpy(&bits, &x->data[i], sizeof(bits));
                if (bits) work.push_back(reinterpret_cast<Tensor*>(bits));
                x->data[i] = 0.0;
            }
        }

        // Snapshot children, null out fields to break cycles defensively
        Tensor* lhs_old  = x->lhs;     x->lhs = nullptr;
        Tensor* rhs_old  = x->rhs;     x->rhs = nullptr;
        Tensor* base_old = x->base;    x->base = nullptr;
        Tensor** par_old = x->parents;
        int np = x->num_parents;
        x->parents = nullptr;
        x->num_parents = 0;

        // Remove from g_arena (swap-and-pop, O(1))
        for (size_t i = 0; i < g_arena.size(); ++i) {
            if (g_arena[i] == x) {
                g_arena[i] = g_arena.back();
                g_arena.pop_back();
                break;
            }
        }

        // Release data buffers
        if (x->owns_data) {
            if (x->data) g_pool.release(x->data, x->size);
            if (x->grad) g_pool.release_grad(x->grad, x->size);
        }
        if (x->fdata) { delete[] x->fdata; x->fdata = nullptr; }
        if (x->fgrad) { delete[] x->fgrad; x->fgrad = nullptr; }
        x->data = nullptr;
        x->grad = nullptr;
        delete x;

        // Push children onto work stack for iterative processing
        if (lhs_old)  work.push_back(lhs_old);
        if (rhs_old)  work.push_back(rhs_old);
        if (base_old) work.push_back(base_old);
        if (par_old) {
            for (int i = 0; i < np; ++i)
                if (par_old[i]) work.push_back(par_old[i]);
            delete[] par_old;
        }
    }
    return nullptr;
}

// =====================================================================
// Graph-edge ownership helpers — USE THESE EVERYWHERE INSTEAD OF DIRECT
// ASSIGNMENT TO ->lhs / ->rhs / ->base / ->parents.
// =====================================================================
extern "C" void tensor_set_lhs(Tensor* parent, Tensor* child) {
    if (!parent) return;
    if (child) {
        if (child->refcount < 0) {
            std::cerr << "[BUG] tensor_set_lhs: child has refcount=" << child->refcount
                      << " (freed?) ptr=" << (void*)child << "\n" << std::flush;
            return;
        }
        child->refcount++;
    }
    Tensor* old = parent->lhs;
    parent->lhs = child;
    if (old) tensor_decref(old);
}
extern "C" void tensor_set_rhs(Tensor* parent, Tensor* child) {
    if (!parent) return;
    if (child) child->refcount++;
    Tensor* old = parent->rhs;
    parent->rhs = child;
    if (old) tensor_decref(old);
}
extern "C" void tensor_set_base(Tensor* parent, Tensor* child) {
    if (!parent) return;
    if (child) child->refcount++;
    Tensor* old = parent->base;
    parent->base = child;
    if (old) tensor_decref(old);
}
// Adopts ownership of `par` (the array itself). Each non-null par[i]
// is incref'd. Old parents (if any) are decref'd and freed.
extern "C" void tensor_set_parents(Tensor* parent, Tensor** par, int n) {
    if (!parent) return;
    Tensor** old_par = parent->parents;
    int old_np = parent->num_parents;
    parent->parents = par;
    parent->num_parents = n;
    if (par) {
        for (int i = 0; i < n; ++i)
            if (par[i]) par[i]->refcount++;
    }
    if (old_par) {
        for (int i = 0; i < old_np; ++i)
            if (old_par[i]) tensor_decref(old_par[i]);
        delete[] old_par;
    }
}
// Clear all graph edges (used by zero_grad)
extern "C" void tensor_clear_graph(Tensor* t) {
    if (!t) return;
    tensor_set_lhs(t, nullptr);
    tensor_set_rhs(t, nullptr);
    tensor_set_base(t, nullptr);
    tensor_set_parents(t, nullptr, 0);
    t->is_leaf = true;
    t->grad_fn = "";
}

// =====================================================================
// Legacy API — kept as no-ops for backward compat with existing AST.cpp
// =====================================================================
extern "C" Tensor* tensor_pin(Tensor* t)        { return t; }
extern "C" Tensor* tensor_unpin(Tensor* t)      { return t; }
extern "C" Tensor* tensor_mark_var(Tensor* t)   { return t; }   // no-op now
extern "C" Tensor* dream_unroot(Tensor* t)      { return t; }
Tensor* dream_gc()                              { return create_scalar(0.0); }
Tensor* dream_pool_stats() {
    std::cout << "Pool: " << g_pool.pool_size() << " cached, "
              << g_pool.hits << " hits, " << g_pool.misses << " misses\n" << std::flush;
    return create_scalar((double)g_pool.hits);
}

// Diagnostic: dump distribution of currently-alive tensors by shape and grad_fn.
// Prints to stderr. Returns count.
extern "C" Tensor* dream_arena_dump() {
    std::map<std::string, int> by_shape;
    std::map<std::string, int> by_grad_fn;
    int total_refcount = 0;
    int max_refcount = 0;
    int negative_rc = 0;
    for (Tensor* t : g_arena) {
        if (!t) continue;
        std::string sh = std::to_string(t->d0) + "x" + std::to_string(t->d1);
        if (t->d2 > 1) sh += "x" + std::to_string(t->d2);
        if (t->d3 > 1) sh += "x" + std::to_string(t->d3);
        by_shape[sh]++;
        std::string gf = t->grad_fn.empty() ? "(leaf)" : t->grad_fn;
        by_grad_fn[gf]++;
        total_refcount += t->refcount;
        if (t->refcount > max_refcount) max_refcount = t->refcount;
        if (t->refcount < 0) negative_rc++;
    }
    std::cerr << "\n=== ARENA DUMP ===\n"
              << "Total alive: " << g_arena.size() << "\n"
              << "Total refcount sum: " << total_refcount << "\n"
              << "Max refcount: " << max_refcount << "\n"
              << "Negative refcount: " << negative_rc << "\n"
              << "By shape (top 20):\n";
    std::vector<std::pair<std::string, int>> sh_sorted(by_shape.begin(), by_shape.end());
    std::sort(sh_sorted.begin(), sh_sorted.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    for (size_t i = 0; i < sh_sorted.size() && i < 20; ++i)
        std::cerr << "  " << sh_sorted[i].first << ": " << sh_sorted[i].second << "\n";
    std::cerr << "By grad_fn (top 20):\n";
    std::vector<std::pair<std::string, int>> gf_sorted(by_grad_fn.begin(), by_grad_fn.end());
    std::sort(gf_sorted.begin(), gf_sorted.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    for (size_t i = 0; i < gf_sorted.size() && i < 20; ++i)
        std::cerr << "  '" << gf_sorted[i].first << "': " << gf_sorted[i].second << "\n";
    std::cerr << "==================\n" << std::flush;
    return create_scalar((double)g_arena.size());
}

// =====================================================================
// Materialize: if non-contiguous, copy to contiguous buffer
// =====================================================================
static inline Tensor* materialize(Tensor* t) {
    if (!t) return nullptr;
    sync_f32_to_data(t);   // flush any pending f32 result before reading data
    if (t->owns_data && is_contiguous(t)) return t;
    Tensor* c = alloc_tensor(t->d0, t->d1, t->d2, t->d3);
    if (t->ndim <= 2) {
        for (int i = 0; i < t->d0; ++i)
            for (int j = 0; j < t->d1; ++j) {
                int src_idx = t->data_offset + i * t->strides[0] + (t->ndim >= 2 ? j * t->strides[1] : 0);
                c->data[i * t->d1 + j] = t->data[src_idx];
            }
    } else {
        for (int i = 0; i < t->size; ++i) c->data[i] = t->data[i];
    }
    return c;
}

// =====================================================================
// Common utility ops
// =====================================================================
Tensor* create_scalar(double v) {
    Tensor* t = alloc_tensor(1, 1, 1, 1);
    t->data[0] = v;
    return t;
}
Tensor* create_temp_scalar(double v) { return create_scalar(v); }
Tensor* dream_scalar(Tensor* v) {
    return create_scalar((v && v->size > 0) ? v->data[0] : 0.0);
}

static void bump_version(Tensor* t) {
    if (t) t->version++;
}

Tensor* tensor_assign(Tensor* dest, Tensor* src) {
    if (!dest || !src) return src;
    if (!src->owns_data || !is_contiguous(src)) src = materialize(src);
    if (dest->size != src->size) {
        if (dest->owns_data) {
            g_pool.release(dest->data, dest->size);
            g_pool.release_grad(dest->grad, dest->size);
            dest->data = g_pool.alloc(src->size);
            dest->grad = g_pool.alloc_grad(src->size);
        }
        dest->size = src->size;
        dest->d0 = src->d0; dest->d1 = src->d1;
        dest->d2 = src->d2; dest->d3 = src->d3;
        for (int i = 0; i < 8; ++i) { dest->dims[i] = src->dims[i]; dest->strides[i] = src->strides[i]; }
        dest->ndim = src->ndim;
    }
    std::copy(src->data, src->data + src->size, dest->data);
    bump_version(dest);
    return dest;
}

// Note: dream_transpose itself is defined in math_ops.inc (which is
// included after tensor_core.inc). We provide _view variants here.
Tensor* dream_transpose_view(Tensor* t) {
    // Original implementation (was here before in the legacy version)
    if (!t || t->ndim < 2) return tensor_incref(t);
    t = materialize(t);
    Tensor* c = alloc_tensor(t->d1, t->d0, t->d2, t->d3);
    c->is_leaf = false;
    tensor_set_lhs(c, t);
    c->grad_fn = "transpose";
    for (int i = 0; i < t->d0; ++i)
        for (int j = 0; j < t->d1; ++j)
            c->data[j * c->d1 + i] = t->data[i * t->d1 + j];
    return c;
}

// Slice rows: returns a fresh contiguous tensor with rows [start, end)
Tensor* dream_slice_view(Tensor* t, Tensor* start_t, Tensor* end_t) {
    if (!t || !start_t || !end_t) return tensor_incref(t);
    int start = (int)start_t->data[0];
    int end = (int)end_t->data[0];
    if (end > t->d0) end = t->d0;
    if (start < 0) start = 0;
    int rows = end - start;
    if (rows <= 0) rows = 1;
    int cols = t->d1 ? t->d1 : 1;
    Tensor* c = alloc_tensor(rows, cols, t->d2, t->d3);
    c->is_leaf = false;
    tensor_set_lhs(c, t);
    c->grad_fn = "slice";
    int stride = cols * (t->d2 ? t->d2 : 1) * (t->d3 ? t->d3 : 1);
    for (int i = 0; i < rows * stride; ++i)
        c->data[i] = t->data[(start * stride) + i];
    return c;
}

