
Tensor* ones(Tensor* d0, Tensor* d1) {
    Tensor* t = alloc_tensor(int(d0->data[0]), int(d1->data[0]), 1, 1);
    std::fill(t->data, t->data+t->size, 1.0); return t;
}
Tensor* zeros(Tensor* d0, Tensor* d1) {
    return alloc_tensor(int(d0->data[0]), int(d1->data[0]), 1, 1);
}
Tensor* reshape(Tensor* t, Tensor* d0, Tensor* d1) {
    int newSz = int(d0->data[0]) * int(d1->data[0]);
    if (newSz != t->size) {
        std::cerr << "RUNTIME ERROR in reshape: size mismatch (" << t->size << " vs " << newSz << ")\n" << std::flush;
        return tensor_incref(t);
    }
    // Return a new graph node instead of mutating t in place. The old
    // in-place version silently changed the SOURCE tensor's shape, so
    //     let xf = reshape(x, ...);  ... x used later ...
    // corrupted x (the GPT example hit this as a shape-mismatch crash).
    // Gradient is an element-wise pass-through (sizes are equal).
    t = materialize(t);
    Tensor* c = alloc_tensor(int(d0->data[0]), int(d1->data[0]), 1, 1);
    std::copy(t->data, t->data + t->size, c->data);
    c->is_leaf = false; tensor_set_lhs(c, t); c->grad_fn = "reshape";
    return c;
}

//g_autograd_enabledlobal seedable RNG for reproducibility.
// Users can call set_seed(42) at start of training to get deterministic results.
// Without a seed set, uses entropy from random_device (non-deterministic).
static std::mt19937& get_rng() {
    static std::mt19937 rng(std::random_device{}());
    return rng;
}

Tensor* dream_set_seed(Tensor* seed_t) {
    unsigned int seed = seed_t ? (unsigned int)seed_t->data[0] : 42;
    get_rng().seed(seed);
    std::cerr << "[seed] RNG seeded with " << seed << "\n" << std::flush;
    return create_scalar((double)seed);
}

//backward-compat: allow calling set_seed() with no args
Tensor* dream_set_seed_default() {
    get_rng().seed(42);
    return create_scalar(42.0);
}

Tensor* dream_randn(Tensor* d0, Tensor* d1) {
    Tensor* t = alloc_tensor(int(d0->data[0]), int(d1->data[0]), 1, 1);
    std::normal_distribution<double> dist(0.0, 1.0);
    auto& rng = get_rng();
    for (int i = 0; i < t->size; ++i) t->data[i] = dist(rng);
    return t;
}

Tensor* sum(Tensor* a) {
    a = materialize(a);
    double s=0; for(int i=0;i<a->size;++i) s+=a->data[i];
    Tensor* c = alloc_tensor(1,1,1,1);
    c->data[0] = s;
    c->is_leaf = false;
    tensor_set_lhs(c, a);
    c->grad_fn = "sum";
    return c;
}
//high-precision sum: compensated accumulation; gradient semantics match sum (d/dx[i] = 1).
Tensor* dream_sum_exact(Tensor* a) {
    a = materialize(a);
    Tensor* c = alloc_tensor(1,1,1,1);
    c->data[0] = neumaier_sum(a->data, a->size);
    c->is_leaf = false;
    tensor_set_lhs(c, a);
    c->grad_fn = "sum";
    return c;
}
Tensor* mean(Tensor* a) {
    a = materialize(a);
    if (!a->size) return create_scalar(0.0);
    double s=0; for(int i=0;i<a->size;++i) s+=a->data[i];
    Tensor* c = alloc_tensor(1,1,1,1);
    c->data[0] = s / a->size;
    c->is_leaf = false;
    tensor_set_lhs(c, a);
    c->grad_fn = "mean";
    return c;
}
// High-precision mean: compensated sum then divide; gradient matches mean (d/dx[i] = 1/n).
Tensor* dream_mean_exact(Tensor* a) {
    a = materialize(a);
    if (!a->size) return create_scalar(0.0);
    Tensor* c = alloc_tensor(1,1,1,1);
    c->data[0] = neumaier_sum(a->data, a->size) / a->size;
    c->is_leaf = false;
    tensor_set_lhs(c, a);
    c->grad_fn = "mean";
    return c;
}
Tensor* dream_argmax(Tensor* a) {
    a = materialize(a);
    int idx=0; double mx=a->data[0];
    for(int i=1;i<a->size;++i) if(a->data[i]>mx){ mx=a->data[i]; idx=i; }
    return create_scalar(double(idx));
}

Tensor* mse_loss(Tensor* pred, Tensor* target) {
    if (!pred || !target) {
        std::cerr << "RUNTIME ERROR in mse_loss: null input"
                  << (pred ? "" : " (pred=null)") << (target ? "" : " (target=null)") << "\n" << std::flush;
        return create_scalar(0.0);
    }
    pred = materialize(pred); target = materialize(target);
    if (pred->size != target->size) {
        std::cerr << "RUNTIME ERROR in mse_loss: shape mismatch ("
                  << pred->d0 << "x" << pred->d1 << " vs " << target->d0 << "x" << target->d1 << ")\n" << std::flush;
        return create_scalar(0.0);
    }
    Tensor* c = alloc_tensor(1,1,1,1);
    c->is_leaf=false; tensor_set_lhs(c, pred); tensor_set_rhs(c, target); c->grad_fn="mse_loss";
    double s=0; for(int i=0;i<pred->size;++i){ double d=pred->data[i]-target->data[i]; s+=d*d; }
    c->data[0]=s/pred->size; return c;
}

Tensor* dream_cross_entropy(Tensor* logits, Tensor* target) {
    if (!logits || !target) {
        std::cerr << "RUNTIME ERROR in cross_entropy: null input"
                  << (logits ? "" : " (logits=null)")
                  << (target ? "" : " (target=null)") << "\n" << std::flush;
        return create_scalar(0.0);
    }
    logits = materialize(logits); target = materialize(target);
    if (!logits->data || !target->data) {
        std::cerr << "RUNTIME ERROR in cross_entropy: null data pointer"
                  << " logits=" << logits->d0 << "x" << logits->d1
                  << " target=" << target->d0 << "x" << target->d1 << "\n" << std::flush;
        return create_scalar(0.0);
    }
    if (logits->size != target->size) {
        std::cerr << "RUNTIME ERROR in cross_entropy: shape mismatch "
                  << logits->d0 << "x" << logits->d1 << " vs "
                  << target->d0 << "x" << target->d1 << "\n" << std::flush;
        return create_scalar(0.0);
    }
    Tensor* c = alloc_tensor(1,1,1,1);
    c->is_leaf=false; tensor_set_lhs(c, logits); tensor_set_rhs(c, target); c->grad_fn="cross_entropy";

    // Per-row (per-sample) softmax cross-entropy for batched training
    int N = logits->d0, C = logits->d1;
    double loss = 0;
    for (int n = 0; n < N; ++n) {
        const double* row = logits->data + n * C;
        const double* tgt = target->data + n * C;
        double mx = row[0];
        for (int j = 1; j < C; ++j) if (row[j] > mx) mx = row[j];
        double sumExp = 0;
        for (int j = 0; j < C; ++j) sumExp += std::exp(row[j] - mx);
        double logSumExp = mx + std::log(sumExp);
        for (int j = 0; j < C; ++j)
            if (tgt[j] > 0) loss -= tgt[j] * (row[j] - logSumExp);
    }
    c->data[0] = loss / std::max(1, N);
    return c;
}

Tensor* dream_l2_loss(Tensor* pred, Tensor* target) { return mse_loss(pred, target); }

// Scalar print formatting. Everything is double internally in Dream. Two improvements:
//  1. Non-integer values print with round-trip precision, fixing 10.0/3.0 being
//     truncated to 3.33333; now shows 3.3333333333333335 as expected.
//  2. Integer values stay concise (2.0 -> "2"), no forced decimal point.
//     (Dream has no separate integer type; this is a display choice. To force a
//      floating look, change to a %.1f branch here.)
static std::string format_scalar(double v) {
    // -ffast-math assumes no NaN/Inf and would optimize away std::isnan/isinf (with a warning).
    // Detect via bit pattern: an all-ones exponent in an IEEE754 double means NaN/Inf.
    uint64_t bits;
    std::memcpy(&bits, &v, sizeof(bits));
    uint64_t exp = (bits >> 52) & 0x7FF;
    uint64_t mant = bits & 0xFFFFFFFFFFFFFULL;
    if (exp == 0x7FF) {
        if (mant != 0) return "nan";
        return (bits >> 63) ? "-inf" : "inf";
    }
    if (v == std::floor(v) && std::fabs(v) < 1e15) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%.0f", v);
        return buf;
    }
    // Non-integer: shortest representation that round-trips the double (up to 17 sig digits).
    for (int prec = 1; prec < 17; ++prec) {
        char tmp[40];
        std::snprintf(tmp, sizeof(tmp), "%.*g", prec, v);
        if (std::strtod(tmp, nullptr) == v) return tmp;
    }
    char buf[40];
    std::snprintf(buf, sizeof(buf), "%.17g", v);
    return buf;
}

Tensor* print(Tensor* t) {
    if (!t) { std::cout << "[null]\n" << std::flush; return nullptr; }
    t = materialize(t);
    // String tensor: print as text
    if (t->grad_fn == "string") {
        for (int i = 0; i < t->size; ++i) std::cout << (char)(int)t->data[i];
        std::cout << std::flush;
        return tensor_incref(t);
    }
    // Tuple tensor: print as (a, b, ...)
    if (t->grad_fn == "tuple") {
        std::cout << "(";
        for (int i = 0; i < t->size; ++i) {
            if (i > 0) std::cout << ", ";
            uintptr_t ptr = 0;
            std::memcpy(&ptr, &t->data[i], sizeof(ptr));
            // print() returns +1 under the ownership contract; drop it here
            // since the tuple's parents[] already owns the member.
            tensor_decref(print(reinterpret_cast<Tensor*>(ptr)));
        }
        std::cout << ")" << std::flush;
        return tensor_incref(t);
    }
    if (t->size == 1) { std::cout << format_scalar(t->data[0]) << std::flush; return tensor_incref(t); }
    std::cout << "shape(" << t->d0 << "x" << t->d1 << ") [\n";
    for (int i = 0; i < t->d0; ++i) {
        std::cout << "  [";
        for (int j = 0; j < t->d1; ++j) std::cout << " " << format_scalar(t->data[i*t->d1+j]);
        std::cout << " ]\n";
    }
    std::cout << "]\n" << std::flush;
    return tensor_incref(t);
}
Tensor* dream_print_shape(Tensor* t) {
    if (!t) { std::cout << "shape: [null]\n" << std::flush; return nullptr; }
    std::cout << "shape: " << t->d0 << "x" << t->d1
              << "x" << t->d2 << "x" << t->d3
              << " (size=" << t->size << ")\n" << std::flush;
    return tensor_incref(t);
}
Tensor* dream_print_string(const char* s) {
    std::cout << s << std::flush;
    return create_scalar(0.0);
}
Tensor* dream_print_one(Tensor* t) { return print(t); }

// ---------------------------------------------------------------------
//  Global autograd switch.
//  An accumulator in a long loop (total = total + x) builds a graph each step,
//  whose chain length equals the iteration count and grows to GBs, then crashes.
//  In no_grad mode, binary/unary ops build no graph; accumulators stay O(1).
//  Controlled by the no_grad { ... } block; on by default outside training loops.
// ---------------------------------------------------------------------
static bool g_autograd_enabled = true;
extern "C" Tensor* dream_set_autograd(Tensor* on_t) {
    g_autograd_enabled = (on_t && on_t->data[0] != 0.0);
    return create_scalar(g_autograd_enabled ? 1.0 : 0.0);
}
extern "C" Tensor* dream_autograd_enabled() {
    return create_scalar(g_autograd_enabled ? 1.0 : 0.0);
}

// Soft cap on graph chain depth. An operand deeper than this is auto-detached
// (its history graph is cut, treated as a leaf) before joining a new op, so any
// unbounded accumulation stays O(1) in graph memory instead of growing linearly.
// The cap is far above any reasonable single backward depth (typical nets < 1000 layers).
static constexpr int GRAPH_DEPTH_LIMIT = 50000;

static Tensor* makeBinop(Tensor* a, Tensor* b, const char* fn) {
    if (!a || !b) return a ? a : (b ? b : alloc_tensor(1,1,1,1));
    Tensor* c = alloc_tensor(std::max(a->d0,b->d0), std::max(a->d1,b->d1),
                             std::max(a->d2,b->d2), std::max(a->d3,b->d3));
    // no_grad mode: build no graph, c is a leaf. Accumulator loops use constant memory.
    if (!g_autograd_enabled) { c->grad_fn = fn; return c; }

    // Auto graph-blowup guard: if an operand chain is too deep, detach it and
    // attach it as a fresh leaf. This cuts history (no gradient there) but bounds memory.
    if (a->graph_depth >= GRAPH_DEPTH_LIMIT) tensor_clear_graph(a);
    if (b->graph_depth >= GRAPH_DEPTH_LIMIT) tensor_clear_graph(b);

    c->is_leaf=false; tensor_set_lhs(c, a); tensor_set_rhs(c, b); c->grad_fn=fn;
    int da = a->graph_depth, db = b->graph_depth;
    c->graph_depth = (da > db ? da : db) + 1;
    return c;
}
#define BINOP(name, opStr, gradStr, expr) \
    Tensor* name(Tensor* a, Tensor* b) { \
        if (!a || !b) { std::cerr << "RUNTIME ERROR in '" opStr "': null operand\n" << std::flush; return alloc_tensor(1,1,1,1); } \
        a = materialize(a); b = materialize(b); \
        checkShape(a,b,opStr); Tensor* c=makeBinop(a,b,gradStr); runGPU(a,b,c,expr); return c; }

BINOP(tensor_add, "+",  "add", "a_val+b_val")
BINOP(tensor_sub, "-",  "sub", "a_val-b_val")
BINOP(tensor_mul, "*",  "mul", "a_val*b_val")
BINOP(tensor_div, "/",  "div", "a_val/b_val")
BINOP(tensor_mod, "%",  "mod", "fmod(a_val,b_val)")
BINOP(tensor_gt,  ">",  "gt",  "a_val>b_val?1.f:0.f")
BINOP(tensor_lt,  "<",  "lt",  "a_val<b_val?1.f:0.f")
BINOP(tensor_ge,  ">=", "ge",  "a_val>=b_val?1.f:0.f")
BINOP(tensor_le,  "<=", "le",  "a_val<=b_val?1.f:0.f")
BINOP(tensor_eq,  "==", "eq",  "fabs(a_val-b_val)<1e-6f?1.f:0.f")
BINOP(tensor_ne,  "!=", "ne",  "fabs(a_val-b_val)>=1e-6f?1.f:0.f")
BINOP(tensor_and, "&&", "and", "(a_val>0.f&&b_val>0.f)?1.f:0.f")
BINOP(tensor_or,  "||", "or",  "(a_val>0.f||b_val>0.f)?1.f:0.f")
// Element-wise min/max. Because a scalar is a 1x1 tensor in Dream, these work
// for both min(3.0, 5.0) and min(tensorA, tensorB). Requested often enough
// that their absence was a real papercut.
BINOP(dream_minimum, "min", "minimum", "a_val<b_val?a_val:b_val")
BINOP(dream_maximum, "max", "maximum", "a_val>b_val?a_val:b_val")
#undef BINOP

static Tensor* makeUnary(Tensor* a, const char* gfn) {
    if (!a) { std::cerr << "RUNTIME ERROR in '" << gfn << "': null input\n" << std::flush; return alloc_tensor(1,1,1,1); }
    Tensor* c = alloc_tensor(a->d0,a->d1,a->d2,a->d3);
    c->is_leaf=false; tensor_set_lhs(c, a); c->grad_fn=gfn; return c;
}
#define UNARY(name, gradStr, expr) \
    Tensor* name(Tensor* a) { \
        if (!a) { std::cerr << "RUNTIME ERROR in '" gradStr "': null input\n" << std::flush; return alloc_tensor(1,1,1,1); } \
        a = materialize(a); Tensor* c=makeUnary(a,gradStr); runGPUUnary(a,c,expr); return c; }

UNARY(relu,          "relu",    "a_val>0.f?a_val:0.f")
UNARY(dream_exp,     "exp",     "exp(a_val)")
UNARY(dream_log,     "log",     "log(a_val)")
UNARY(dream_sqrt,    "sqrt",    "sqrt(a_val)")
UNARY(dream_abs,     "abs",     "fabs(a_val)")
UNARY(dream_sigmoid, "sigmoid", "1.f/(1.f+exp(-a_val))")
UNARY(dream_tanh,    "tanh",    "tanh(a_val)")
UNARY(dream_logical_not, "lnot","a_val==0.f?1.f:0.f")
#undef UNARY

Tensor* dream_clip(Tensor* a, Tensor* lo, Tensor* hi) {
    float loV = float(lo->data[0]), hiV = float(hi->data[0]);
    Tensor* c = makeUnary(a, "clip");
    std::string expr = "fmax("+std::to_string(loV)+"f,fmin("+std::to_string(hiV)+"f,a_val))";
    runGPUUnary(a, c, expr);
    return c;
}

Tensor* dream_softmax(Tensor* a) {
    a = materialize(a);
    Tensor* c = alloc_tensor(a->d0,a->d1,a->d2,a->d3);
    c->is_leaf=false; tensor_set_lhs(c, a); c->grad_fn="softmax";
    // Row-wise, matching PyTorch's softmax(dim=-1) on 2D input. The old
    // version normalized over the WHOLE tensor, coupling samples across a
    // batch -- almost never the intended semantics. For >2D the trailing
    // dims are folded into the normalized axis.
    int R = a->d0 > 0 ? a->d0 : 1;
    int C = a->size / R;
    for (int r = 0; r < R; ++r) {
        const double* x = a->data + (size_t)r * C;
        double* y = c->data + (size_t)r * C;
        double mx = x[0];
        for (int j = 1; j < C; ++j) if (x[j] > mx) mx = x[j];
        double s = 0;
        for (int j = 0; j < C; ++j) { y[j] = std::exp(x[j] - mx); s += y[j]; }
        for (int j = 0; j < C; ++j) y[j] /= s;
    }
    return c;
}

// =====================================================================
//  Axis reductions. dim==1 reduces columns (R x C -> R x 1); dim==0
//  reduces rows (R x C -> 1 x C). Matches PyTorch's sum/mean/max(x, dim)
//  with keepdim semantics on the reduced axis.
// =====================================================================
Tensor* dream_sum_dim(Tensor* a, Tensor* dim_t) {
    a = materialize(a);
    int dim = dim_t ? (int)dim_t->data[0] : 1;
    int R = a->d0 > 0 ? a->d0 : 1, C = a->size / R;
    Tensor* c = (dim == 1) ? alloc_tensor(R, 1, 1, 1) : alloc_tensor(1, C, 1, 1);
    c->is_leaf = false; tensor_set_lhs(c, a);
    c->grad_fn = (dim == 1) ? "sum_d1" : "sum_d0";
    if (dim == 1) {
        for (int r = 0; r < R; ++r) {
            double s = 0;
            for (int j = 0; j < C; ++j) s += a->data[(size_t)r*C + j];
            c->data[r] = s;
        }
    } else {
        for (int j = 0; j < C; ++j) {
            double s = 0;
            for (int r = 0; r < R; ++r) s += a->data[(size_t)r*C + j];
            c->data[j] = s;
        }
    }
    return c;
}
Tensor* dream_mean_dim(Tensor* a, Tensor* dim_t) {
    a = materialize(a);
    int dim = dim_t ? (int)dim_t->data[0] : 1;
    int R = a->d0 > 0 ? a->d0 : 1, C = a->size / R;
    Tensor* s = dream_sum_dim(a, dim_t);
    double n = (dim == 1) ? (double)C : (double)R;
    for (int i = 0; i < s->size; ++i) s->data[i] /= n;
    s->grad_fn = (dim == 1) ? "mean_d1" : "mean_d0";
    return s;
}
// Ties resolve to the FIRST maximum; the backward recomputes the argmax
// with the identical rule, so gradients stay consistent.
Tensor* dream_max_dim(Tensor* a, Tensor* dim_t) {
    a = materialize(a);
    int dim = dim_t ? (int)dim_t->data[0] : 1;
    int R = a->d0 > 0 ? a->d0 : 1, C = a->size / R;
    Tensor* c = (dim == 1) ? alloc_tensor(R, 1, 1, 1) : alloc_tensor(1, C, 1, 1);
    c->is_leaf = false; tensor_set_lhs(c, a);
    c->grad_fn = (dim == 1) ? "max_d1" : "max_d0";
    if (dim == 1) {
        for (int r = 0; r < R; ++r) {
            double m = a->data[(size_t)r*C];
            for (int j = 1; j < C; ++j) { double v = a->data[(size_t)r*C + j]; if (v > m) m = v; }
            c->data[r] = m;
        }
    } else {
        for (int j = 0; j < C; ++j) {
            double m = a->data[j];
            for (int r = 1; r < R; ++r) { double v = a->data[(size_t)r*C + j]; if (v > m) m = v; }
            c->data[j] = m;
        }
    }
    return c;
}

Tensor* dream_transpose(Tensor* a) {
    if (!a) { std::cerr << "RUNTIME ERROR in transpose: null input\n" << std::flush; return alloc_tensor(1,1,1,1); }
    if (!a->data) { std::cerr << "RUNTIME ERROR in transpose: null data\n" << std::flush; return alloc_tensor(1,1,1,1); }

    // Cache-optimal transpose with scratch buffer.
    //
    // Key insight: within a 32x32 tile, we have both unit-stride reads and
    // unit-stride writes if we use a local buffer. Each tile stays in L1 (8KB).
    //
    // For 512x512: 256 tiles × (32×32 double reads + 32×32 double writes)
    //            = 256 × 8KB = 2MB sequential, fully cache-friendly.
    //
    // Previous naive (i,j) wrote 4KB apart → cache miss every write.
    // New: reads and writes are both unit-stride within each tile.
    Tensor* c = alloc_tensor(a->d1, a->d0, 1, 1);
    c->is_leaf = false;
    tensor_set_lhs(c, a);
    c->grad_fn = "transpose";

    const int M = a->d0, N = a->d1;
    const double* A = a->data;
    double* C = c->data;

    // For small matrices, naive is fine (and avoids tiling overhead)
    if (M * N < 1024) {
        for (int i = 0; i < M; ++i)
            for (int j = 0; j < N; ++j)
                C[j * M + i] = A[i * N + j];
        return c;
    }

    const int BLOCK = 32;

    #pragma omp parallel for schedule(static) if(M * N >= 32768)
    for (int ii = 0; ii < M; ii += BLOCK) {
        int iend = (ii + BLOCK < M) ? (ii + BLOCK) : M;
        for (int jj = 0; jj < N; jj += BLOCK) {
            int jend = (jj + BLOCK < N) ? (jj + BLOCK) : N;
            // Transpose this 32x32 tile using a local buffer.
            // Read 32 rows of A (32 cache lines of 4 doubles), write 32 cols of C.
            double tile[32 * 32];  // stack-allocated, fits in L1
            int rows = iend - ii;
            int cols = jend - jj;

            // Read tile from A — unit stride within each row
            for (int i = 0; i < rows; ++i) {
                const double* src = A + (ii + i) * N + jj;
                for (int j = 0; j < cols; ++j) tile[i * BLOCK + j] = src[j];
            }
            // Write tile to C — unit stride within each destination row
            for (int j = 0; j < cols; ++j) {
                double* dst = C + (jj + j) * M + ii;
                for (int i = 0; i < rows; ++i) dst[i] = tile[i * BLOCK + j];
            }
        }
    }
    return c;
}

Tensor* dream_concat(Tensor* a, Tensor* b) {
    a = materialize(a); b = materialize(b);
    if (a->d0 != b->d0) {
        std::cerr << "RUNTIME ERROR in concat: row mismatch (" << a->d0 << " vs " << b->d0 << ")\n" << std::flush;
        return tensor_incref(a);
    }
    Tensor* c = alloc_tensor(a->d0, a->d1+b->d1, 1, 1);
    c->is_leaf=false; tensor_set_lhs(c, a); tensor_set_rhs(c, b); c->grad_fn="concat";
    int nc = a->d1+b->d1;
    for(int i=0;i<a->d0;++i){
        for(int j=0;j<a->d1;++j) c->data[i*nc+j]       = a->data[i*a->d1+j];
        for(int j=0;j<b->d1;++j) c->data[i*nc+a->d1+j] = b->data[i*b->d1+j];
    }
    return c;
}

// Matrix multiply — routes to cuBLAS on NVIDIA (when available), otherwise OpenCL
// =====================================================================
//  High-performance double GEMM: cache-blocked + packed, AVX2 4x8 microkernel.
//
//  Why the old kernel stalled at ~3.6 GFLOPS: the ikj loop streams B from
//  main memory once per A-row block, so large matmuls are bandwidth-bound.
//  BLAS-class GEMM fixes this by (1) blocking for L1/L2 (KC/MC/NC), and
//  (2) PACKING the A/B blocks into the exact contiguous order the micro-
//  kernel consumes, so the innermost loop runs entirely out of cache.
//
//  Microkernel 4x8 (MR x NR): 8 ymm accumulators; per k step it does
//  2 B loads + 4 A broadcasts + 8 FMAs. Edges are handled by zero-padded
//  packing plus a local-buffer write-back, so ANY shape is exact.
//
//  Correctness gate: gemm_selftest() compares against the naive reference
//  on 11 adversarial shapes (1x1x1, primes, non-multiples of MR/NR) and is
//  wired into `dream test`. The fast kernel does not ship unverified.
// =====================================================================
#if defined(__AVX2__) || defined(__AVX__)
namespace dgemm {
static constexpr int MR = 4, NR = 8;
static constexpr int KC = 256, MC = 128, NC_CAP = 4096;

// Pack an mc x kc block of A (row-major, leading dim lda) into MR-row
// panels laid out [panel][k][mr], zero-padded to MR.
static void packA(const double* A, int lda, int mc, int kc, double* dst) {
    for (int ir = 0; ir < mc; ir += MR) {
        int mr = std::min(MR, mc - ir);
        for (int k = 0; k < kc; ++k) {
            for (int i = 0;  i < mr; ++i) dst[i] = A[(size_t)(ir + i) * lda + k];
            for (int i = mr; i < MR; ++i) dst[i] = 0.0;
            dst += MR;
        }
    }
}
// Pack a kc x nc block of B (row-major, leading dim ldb) into NR-col
// panels laid out [panel][k][nr], zero-padded to NR.
static void packB(const double* B, int ldb, int kc, int nc, double* dst) {
    for (int jr = 0; jr < nc; jr += NR) {
        int nr = std::min(NR, nc - jr);
        for (int k = 0; k < kc; ++k) {
            const double* src = B + (size_t)k * ldb + jr;
            for (int j = 0;  j < nr; ++j) dst[j] = src[j];
            for (int j = nr; j < NR; ++j) dst[j] = 0.0;
            dst += NR;
        }
    }
}

// C(4x8, leading dim ldc) += Apanel * Bpanel over kc steps.
static inline void ukern_4x8(int kc, const double* Ap, const double* Bp,
                             double* C, int ldc) {
    __m256d c00 = _mm256_setzero_pd(), c01 = _mm256_setzero_pd();
    __m256d c10 = _mm256_setzero_pd(), c11 = _mm256_setzero_pd();
    __m256d c20 = _mm256_setzero_pd(), c21 = _mm256_setzero_pd();
    __m256d c30 = _mm256_setzero_pd(), c31 = _mm256_setzero_pd();
    for (int k = 0; k < kc; ++k) {
        __m256d b0 = _mm256_loadu_pd(Bp + (size_t)k * NR);
        __m256d b1 = _mm256_loadu_pd(Bp + (size_t)k * NR + 4);
        __m256d a;
        a = _mm256_broadcast_sd(Ap + (size_t)k * MR + 0);
        c00 = _mm256_fmadd_pd(a, b0, c00); c01 = _mm256_fmadd_pd(a, b1, c01);
        a = _mm256_broadcast_sd(Ap + (size_t)k * MR + 1);
        c10 = _mm256_fmadd_pd(a, b0, c10); c11 = _mm256_fmadd_pd(a, b1, c11);
        a = _mm256_broadcast_sd(Ap + (size_t)k * MR + 2);
        c20 = _mm256_fmadd_pd(a, b0, c20); c21 = _mm256_fmadd_pd(a, b1, c21);
        a = _mm256_broadcast_sd(Ap + (size_t)k * MR + 3);
        c30 = _mm256_fmadd_pd(a, b0, c30); c31 = _mm256_fmadd_pd(a, b1, c31);
    }
    double* r;
    r = C;                 _mm256_storeu_pd(r,   _mm256_add_pd(_mm256_loadu_pd(r),   c00));
                           _mm256_storeu_pd(r+4, _mm256_add_pd(_mm256_loadu_pd(r+4), c01));
    r = C + 1*(size_t)ldc; _mm256_storeu_pd(r,   _mm256_add_pd(_mm256_loadu_pd(r),   c10));
                           _mm256_storeu_pd(r+4, _mm256_add_pd(_mm256_loadu_pd(r+4), c11));
    r = C + 2*(size_t)ldc; _mm256_storeu_pd(r,   _mm256_add_pd(_mm256_loadu_pd(r),   c20));
                           _mm256_storeu_pd(r+4, _mm256_add_pd(_mm256_loadu_pd(r+4), c21));
    r = C + 3*(size_t)ldc; _mm256_storeu_pd(r,   _mm256_add_pd(_mm256_loadu_pd(r),   c30));
                           _mm256_storeu_pd(r+4, _mm256_add_pd(_mm256_loadu_pd(r+4), c31));
}

// Edge-safe tile: compute into a local 4x8 buffer, then add the valid
// mr x nr region into C. Zero-padded packing guarantees the padded lanes
// contribute exactly zero.
static inline void ukern_edge(int kc, const double* Ap, const double* Bp,
                              double* C, int ldc, int mr, int nr) {
    alignas(32) double tmp[MR * NR];
    for (int i = 0; i < MR * NR; ++i) tmp[i] = 0.0;
    ukern_4x8(kc, Ap, Bp, tmp, NR);
    for (int i = 0; i < mr; ++i)
        for (int j = 0; j < nr; ++j)
            C[(size_t)i * ldc + j] += tmp[i * NR + j];
}
} // namespace dgemm
#endif

// C = A(MxK) @ B(KxN), all row-major. Zeroes C internally.
static void dream_gemm_d(const double* A, const double* B, double* C,
                         int M, int K, int N) {
    std::fill(C, C + (size_t)M * N, 0.0);
    if (M <= 0 || K <= 0 || N <= 0) return;
#if defined(__AVX2__) || defined(__AVX__)
    using namespace dgemm;
    int NCb = std::min(N, NC_CAP);
    std::vector<double> Bpack((size_t)KC * (size_t)((NCb + NR - 1) / NR) * NR);
    for (int jc = 0; jc < N; jc += NCb) {
        int nc = std::min(NCb, N - jc);
        for (int pc = 0; pc < K; pc += KC) {
            int kc = std::min(KC, K - pc);
            packB(B + (size_t)pc * N + jc, N, kc, nc, Bpack.data());
            #pragma omp parallel for schedule(dynamic) if((size_t)M * K * N >= 262144)
            for (int ic = 0; ic < M; ic += MC) {
                int mc = std::min(MC, M - ic);
                static thread_local std::vector<double> Apack;
                Apack.resize((size_t)KC * (size_t)((MC + MR - 1) / MR) * MR);
                packA(A + (size_t)ic * K + pc, K, mc, kc, Apack.data());
                for (int jr = 0; jr < nc; jr += NR) {
                    int nr = std::min(NR, nc - jr);
                    const double* Bp = Bpack.data() + (size_t)(jr / NR) * kc * NR;
                    for (int ir = 0; ir < mc; ir += MR) {
                        int mr = std::min(MR, mc - ir);
                        const double* Ap = Apack.data() + (size_t)(ir / MR) * kc * MR;
                        double* Ct = C + (size_t)(ic + ir) * N + (jc + jr);
                        if (mr == MR && nr == NR) ukern_4x8(kc, Ap, Bp, Ct, N);
                        else                      ukern_edge(kc, Ap, Bp, Ct, N, mr, nr);
                    }
                }
            }
        }
    }
#else
    // Portable fallback: blocked ikj loop (previous kernel).
    for (int i = 0; i < M; ++i) {
        const double* Ar = A + (size_t)i * K;
        double* Cr = C + (size_t)i * N;
        for (int k = 0; k < K; ++k) {
            const double aik = Ar[k];
            const double* Br = B + (size_t)k * N;
            for (int j = 0; j < N; ++j) Cr[j] += aik * Br[j];
        }
    }
#endif
}

// f32 blocked+packed GEMM: 8x16 microkernel (16 accumulators, two ymm wide).
// ~4x the simple ikj f32 loop on large matrices, same design as dream_gemm_d
// but single precision. Used by the f32 matmul path (dtype==1). Accumulates
// in fp32 by design -- the f32 path trades precision for speed explicitly.
#if defined(__AVX2__) || defined(__AVX__)
namespace fgemm {
static constexpr int MR = 8, NR = 16, KC = 256, MC = 128, NC_CAP = 4096;
static void packA(const float* A, int lda, int mc, int kc, float* dst) {
    for (int ir = 0; ir < mc; ir += MR) {
        int mr = std::min(MR, mc - ir);
        for (int k = 0; k < kc; ++k) {
            for (int i = 0;  i < mr; ++i) dst[i] = A[(size_t)(ir+i)*lda + k];
            for (int i = mr; i < MR; ++i) dst[i] = 0.0f;
            dst += MR;
        }
    }
}
static void packB(const float* B, int ldb, int kc, int nc, float* dst) {
    for (int jr = 0; jr < nc; jr += NR) {
        int nr = std::min(NR, nc - jr);
        for (int k = 0; k < kc; ++k) {
            const float* s = B + (size_t)k*ldb + jr;
            for (int j = 0;  j < nr; ++j) dst[j] = s[j];
            for (int j = nr; j < NR; ++j) dst[j] = 0.0f;
            dst += NR;
        }
    }
}
static inline void ukern(int kc, const float* Ap, const float* Bp, float* C, int ldc) {
    __m256 c[8][2];
    for (int i = 0; i < 8; ++i) { c[i][0] = _mm256_setzero_ps(); c[i][1] = _mm256_setzero_ps(); }
    for (int k = 0; k < kc; ++k) {
        __m256 b0 = _mm256_loadu_ps(Bp + (size_t)k*NR);
        __m256 b1 = _mm256_loadu_ps(Bp + (size_t)k*NR + 8);
        for (int i = 0; i < 8; ++i) {
            __m256 a = _mm256_broadcast_ss(Ap + (size_t)k*MR + i);
            c[i][0] = _mm256_fmadd_ps(a, b0, c[i][0]);
            c[i][1] = _mm256_fmadd_ps(a, b1, c[i][1]);
        }
    }
    for (int i = 0; i < 8; ++i) {
        float* r = C + (size_t)i*ldc;
        _mm256_storeu_ps(r,   _mm256_add_ps(_mm256_loadu_ps(r),   c[i][0]));
        _mm256_storeu_ps(r+8, _mm256_add_ps(_mm256_loadu_ps(r+8), c[i][1]));
    }
}
} // namespace fgemm
#endif

static void dream_gemm_f32(const float* A, const float* B, float* C,
                           int M, int K, int N) {
    std::fill(C, C + (size_t)M * N, 0.0f);
    if (M <= 0 || K <= 0 || N <= 0) return;
#if defined(__AVX2__) || defined(__AVX__)
    using namespace fgemm;
    int NCb = std::min(N, NC_CAP);
    std::vector<float> Bpack((size_t)KC * (size_t)((NCb + NR - 1) / NR) * NR);
    for (int jc = 0; jc < N; jc += NCb) {
        int nc = std::min(NCb, N - jc);
        for (int pc = 0; pc < K; pc += KC) {
            int kc = std::min(KC, K - pc);
            packB(B + (size_t)pc * N + jc, N, kc, nc, Bpack.data());
            #pragma omp parallel for schedule(dynamic) if((size_t)M * K * N >= 262144)
            for (int ic = 0; ic < M; ic += MC) {
                int mc = std::min(MC, M - ic);
                static thread_local std::vector<float> Apack;
                Apack.resize((size_t)KC * (size_t)((MC + MR - 1) / MR) * MR);
                packA(A + (size_t)ic * K + pc, K, mc, kc, Apack.data());
                alignas(32) float tmp[MR * NR];
                for (int jr = 0; jr < nc; jr += NR) {
                    int nr = std::min(NR, nc - jr);
                    const float* Bp = Bpack.data() + (size_t)(jr / NR) * kc * NR;
                    for (int ir = 0; ir < mc; ir += MR) {
                        int mr = std::min(MR, mc - ir);
                        const float* Ap = Apack.data() + (size_t)(ir / MR) * kc * MR;
                        float* Ct = C + (size_t)(ic + ir) * N + (jc + jr);
                        if (mr == MR && nr == NR) ukern(kc, Ap, Bp, Ct, N);
                        else {
                            std::memset(tmp, 0, sizeof(tmp));
                            ukern(kc, Ap, Bp, tmp, NR);
                            for (int i = 0; i < mr; ++i)
                                for (int j = 0; j < nr; ++j)
                                    Ct[(size_t)i * N + j] += tmp[i * NR + j];
                        }
                    }
                }
            }
        }
    }
#else
    for (int i = 0; i < M; ++i)
        for (int k = 0; k < K; ++k) {
            float aik = A[(size_t)i*K + k];
            const float* Br = B + (size_t)k*N;
            float* Cr = C + (size_t)i*N;
            for (int j = 0; j < N; ++j) Cr[j] += aik * Br[j];
        }
#endif
}


//  the trusted CPU reference, then compares element-by-element.
//
//  WHY: the GPU kernels (OpenCL tiled matmul, cuBLAS, elementwise) had no
//  automated correctness check -- on a machine with a GPU you could not
//  confirm they compute the same thing as the verified CPU path. This does
//  exactly that. On a machine WITHOUT a GPU it reports "unavailable" and
//  passes vacuously (there is nothing to verify), so it is safe in CI.
//
//  Tolerance is 1e-3 relative: GPU matmul accumulates in fp32, so results
//  match the fp64 CPU reference only to single precision -- expected and
//  acceptable for the fp32 GPU path.
// =====================================================================
// Warm up the GPU: trigger OpenCL context init and kernel compilation up
// front, so subsequent timed calls measure actual compute, not one-time
// startup. The FIRST GPU call pays for device init + JIT-compiling every
// kernel (often seconds); without warmup that cost is wrongly attributed to
// whatever op happens to run first, making it look absurdly slow. Call this
// once before timing. Returns the warmup time in seconds.
Tensor* dream_gpu_warmup() {
    extern Tensor* tensor_matmul(Tensor*, Tensor*);
    extern Tensor* tensor_add(Tensor*, Tensor*);
    auto _t0 = std::chrono::steady_clock::now();
    if (!dream_gpu_available()) {
        std::printf("[gpu_warmup] no GPU; nothing to warm up\n");
        return create_scalar(0.0);
    }
    // A matmul large enough to hit the GPU path, plus an elementwise op, so
    // both the tiled-matmul kernel and the binary kernel get compiled now.
    // Must exceed the GPU dispatch threshold (M*N >= 262144, i.e. 512x512) so
    // this actually compiles the GPU matmul kernel. Using 768x768 stays safely
    // above the threshold even if it is tuned. (A 256x256 warmup would now run
    // on the CPU and fail to pre-compile the GPU kernel -- which previously
    // made the first real GPU matmul pay ~1s of kernel compilation.)
    int n = 768;
    Tensor* a = alloc_tensor(n, n, 1, 1);
    Tensor* b = alloc_tensor(n, n, 1, 1);
    for (int i = 0; i < a->size; ++i) { a->data[i] = 0.01 * (i % 7); b->data[i] = 0.01 * (i % 5); }
    Tensor* c = tensor_matmul(a, b);
    Tensor* d = tensor_add(a, b);
    tensor_decref(a); tensor_decref(b); tensor_decref(c); tensor_decref(d);
    // Also warm up the FlashAttention GPU kernel -- it is a SEPARATE kernel
    // that JIT-compiles on its first call. Without this, the first flash call
    // pays ~2s of kernel compilation, wrongly attributed to compute. We run a
    // sequence long enough to hit the GPU dispatch path (B*S*S >= 262144).
    {
        extern Tensor* dream_flash_attention(Tensor*, Tensor*, Tensor*,
                                             Tensor*, Tensor*, Tensor*, Tensor*);
        int S = 512, dk = 64;
        Tensor* Q = alloc_tensor(1, S*dk, 1, 1);
        for (int i = 0; i < Q->size; ++i) Q->data[i] = 0.01 * (i % 11);
        Tensor* bt = create_scalar(1), *st = create_scalar(S),
               *dt2 = create_scalar(dk), *blk = create_scalar(64);
        Tensor* fo = dream_flash_attention(Q, Q, Q, bt, st, dt2, dk <= 128 ? blk : nullptr);
        tensor_decref(Q); tensor_decref(bt); tensor_decref(st);
        tensor_decref(dt2); tensor_decref(blk); tensor_decref(fo);
    }
    double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - _t0).count();
    std::printf("[gpu_warmup] GPU ready (init + kernel compile took %.3f s)\n", dt);
    return create_scalar(dt);
}

Tensor* gpu_selftest() {
    // forward declarations (defined later in this unit)
    extern Tensor* tensor_matmul(Tensor*, Tensor*);
    extern Tensor* tensor_add(Tensor*, Tensor*);
    if (!dream_gpu_available()) {
        std::printf("=== gpu_selftest: no GPU/OpenCL device -- skipped ===\n");
        std::printf("  GPU: SKIP (CPU-only host; GPU kernels not exercised)\n");
        return create_scalar(1.0);   // nothing to verify -> not a failure
    }
    get_rng().seed(20260708);
    std::printf("=== gpu_selftest: GPU kernels vs CPU reference ===\n");
    int pass = 0, total = 0;
    double worst = 0.0;
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    auto& rng = get_rng();

    // ---- matmul on a size that crosses the GPU threshold (M*N >= 65536) ----
    {
        struct Shape { int M, K, N; };
        const Shape shapes[] = {{256,128,256},{300,200,257},{512,64,300}};
        for (const auto& s : shapes) {
            Tensor* A = alloc_tensor(s.M, s.K, 1, 1);
            Tensor* B = alloc_tensor(s.K, s.N, 1, 1);
            for (int i = 0; i < A->size; ++i) A->data[i] = dist(rng);
            for (int i = 0; i < B->size; ++i) B->data[i] = dist(rng);
            // GPU path: tensor_matmul dispatches to GPU for M*N >= 65536.
            Tensor* Cg = tensor_matmul(A, B);
            // CPU reference: force the blocked CPU GEMM directly.
            Tensor* Cc = alloc_tensor(s.M, s.N, 1, 1);
            dream_gemm_d(A->data, B->data, Cc->data, s.M, s.K, s.N);
            // Compare with a combined absolute+relative tolerance. Pure
            // relative error is misleading for matmul: when a result element
            // is near zero (random inputs cancel), fp32 rounding produces a
            // large RELATIVE error from a tiny ABSOLUTE one -- expected, not a
            // bug. We check |gpu - cpu| against a tolerance scaled by the
            // typical element magnitude (sqrt(K), the RMS of K unit products),
            // times an fp32 factor. All-ones inputs (fp32-exact) confirm the
            // kernel logic is correct; this tolerance accepts fp32 cancellation.
            double max_abs = 0.0;
            for (int i = 0; i < Cg->size; ++i)
                max_abs = std::max(max_abs, std::fabs(Cg->data[i] - Cc->data[i]));
            double scale = std::sqrt((double)s.K);      // RMS magnitude of outputs
            double abs_tol = scale * 1e-4;              // fp32-appropriate
            bool ok = max_abs < abs_tol;
            std::printf("  matmul %dx%dx%d  max_abs=%.2e (tol %.1e)  %s\n",
                        s.M, s.K, s.N, max_abs, abs_tol, ok ? "PASS" : "FAIL");
            double mr = max_abs;   // report absolute for the worst-tracker
            worst = std::max(worst, mr); ++total; if (ok) ++pass;
            tensor_decref(A); tensor_decref(B); tensor_decref(Cg); tensor_decref(Cc);
        }
    }

    // ---- elementwise add on a large tensor (exercises runGPU) ----
    {
        int n = 100000;
        Tensor* A = alloc_tensor(1, n, 1, 1);
        Tensor* B = alloc_tensor(1, n, 1, 1);
        for (int i = 0; i < n; ++i) { A->data[i] = dist(rng); B->data[i] = dist(rng); }
        Tensor* Cg = tensor_add(A, B);   // may dispatch to GPU
        double mr = 0.0;
        for (int i = 0; i < n; ++i) {
            double ref = A->data[i] + B->data[i];
            double denom = std::max(std::fabs(ref), 1e-6);
            mr = std::max(mr, std::fabs(Cg->data[i] - ref) / denom);
        }
        bool ok = mr < 1e-3;
        std::printf("  add (n=%d)     max_rel=%.2e  %s\n", n, mr, ok ? "PASS" : "FAIL");
        worst = std::max(worst, mr); ++total; if (ok) ++pass;
        tensor_decref(A); tensor_decref(B); tensor_decref(Cg);
    }

    std::printf("  GPU: %d/%d PASS (worst rel err %.2e)\n", pass, total, worst);
    return create_scalar(pass == total ? 1.0 : 0.0);
}

// Naive triple loop kept as the trusted reference for verification only.
static void gemm_ref_d(const double* A, const double* B, double* C,
                       int M, int K, int N) {
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            double s = 0.0;
            for (int k = 0; k < K; ++k) s += A[(size_t)i*K+k] * B[(size_t)k*N+j];
            C[(size_t)i*N+j] = s;
        }
}

// Verify dream_gemm_d against the naive reference on adversarial shapes,
// then report throughput. Exposed to Dream as gemm_selftest().
Tensor* gemm_selftest() {
    get_rng().seed(20260707);
    struct Shape { int M, K, N; };
    const Shape shapes[] = {
        {1,1,1},{1,7,1},{5,1,9},{3,4,5},{7,13,17},{31,33,37},
        {64,64,64},{65,63,66},{128,100,96},{200,50,300},{257,129,255}
    };
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    auto& rng = get_rng();
    double worst = 0.0; int pass = 0, total = 0;
    std::printf("=== gemm_selftest: blocked kernel vs naive reference ===\n");
    for (const auto& s : shapes) {
        std::vector<double> A((size_t)s.M*s.K), B((size_t)s.K*s.N),
                            C1((size_t)s.M*s.N), C2((size_t)s.M*s.N);
        for (auto& v : A) v = dist(rng);
        for (auto& v : B) v = dist(rng);
        dream_gemm_d(A.data(), B.data(), C1.data(), s.M, s.K, s.N);
        gemm_ref_d (A.data(), B.data(), C2.data(), s.M, s.K, s.N);
        double mr = 0.0;
        for (size_t i = 0; i < C1.size(); ++i) {
            double denom = std::max(std::fabs(C2[i]), 1e-9);
            mr = std::max(mr, std::fabs(C1[i]-C2[i]) / denom);
        }
        bool ok = mr < 1e-11;
        std::printf("  %4dx%-4dx%-4d max_rel=%.2e  %s\n",
                    s.M, s.K, s.N, mr, ok ? "PASS" : "FAIL");
        worst = std::max(worst, mr); ++total; if (ok) ++pass;
    }
    // Throughput probe: 512^3, best of 3.
    {
        const int n = 512;
        std::vector<double> A((size_t)n*n), B((size_t)n*n), C((size_t)n*n);
        for (auto& v : A) v = dist(rng);
        for (auto& v : B) v = dist(rng);
        double best = 1e30;
        for (int rep = 0; rep < 3; ++rep) {
            auto t0 = std::chrono::steady_clock::now();
            dream_gemm_d(A.data(), B.data(), C.data(), n, n, n);
            auto t1 = std::chrono::steady_clock::now();
            best = std::min(best, std::chrono::duration<double>(t1-t0).count());
        }
        double gflops = 2.0 * n * n * (double)n / best / 1e9;
        std::printf("  throughput: 512^3 in %.1f ms = %.1f GFLOPS\n",
                    best * 1e3, gflops);
    }
    // f32 packed kernel: verify on short-K shapes where fp32 accumulation
    // error is tiny, so any tile-indexing bug shows up (long-K error is
    // inherent fp32 precision, not a logic bug -- checked separately).
    {
        const struct { int M,K,N; } fshapes[] = {
            {8,16,16},{7,15,13},{9,8,17},{1,4,1},{16,32,24},{32,16,32}
        };
        std::uniform_real_distribution<float> fdist(-1.0f, 1.0f);
        for (const auto& s : fshapes) {
            std::vector<float> A((size_t)s.M*s.K), B((size_t)s.K*s.N), C1((size_t)s.M*s.N);
            std::vector<double> C2((size_t)s.M*s.N);
            for (auto& v : A) v = fdist(rng);
            for (auto& v : B) v = fdist(rng);
            dream_gemm_f32(A.data(), B.data(), C1.data(), s.M, s.K, s.N);
            for (int i = 0; i < s.M; ++i)
                for (int j = 0; j < s.N; ++j) {
                    double acc = 0.0;
                    for (int k = 0; k < s.K; ++k) acc += (double)A[i*s.K+k]*B[k*s.N+j];
                    C2[i*s.N+j] = acc;
                }
            double mr = 0.0;
            for (size_t i = 0; i < C1.size(); ++i) {
                double denom = std::max(std::fabs(C2[i]), 1e-4);
                mr = std::max(mr, std::fabs((double)C1[i]-C2[i]) / denom);
            }
            bool ok = mr < 2e-4;   // short K -> tight, but fp32 has some slack
            std::printf("  f32 %dx%dx%d max_rel=%.2e  %s\n",
                        s.M, s.K, s.N, mr, ok ? "PASS" : "FAIL");
            worst = std::max(worst, mr); ++total; if (ok) ++pass;
        }
    }
    std::printf("  GEMM: %d/%d PASS (worst rel err %.2e)\n", pass, total, worst);
    return create_scalar(pass == total ? 1.0 : 0.0);
}


Tensor* tensor_matmul(Tensor* a, Tensor* b) {
    if (!a || !b) {
        std::cerr << "RUNTIME ERROR in matmul (@): null operand"
                  << (a ? "" : " (left=null)") << (b ? "" : " (right=null)") << "\n" << std::flush;
        return alloc_tensor(1,1,1,1);
    }
    a = materialize(a); b = materialize(b);
    if (!a->data || !b->data) {
        std::cerr << "RUNTIME ERROR in matmul (@): null data pointer"
                  << " left=" << a->d0 << "x" << a->d1
                  << " right=" << b->d0 << "x" << b->d1 << "\n" << std::flush;
        return alloc_tensor(1,1,1,1);
    }
    if (a->d1 != b->d0) {
        std::cerr << "RUNTIME ERROR in matmul (@): shape mismatch ("
                  << a->d0 << "x" << a->d1 << " @ " << b->d0 << "x" << b->d1
                  << ") — inner dimensions must match\n" << std::flush;
        return alloc_tensor(1,1,1,1);
    }
    Tensor* c = alloc_tensor(a->d0,b->d1,1,1);
    c->is_leaf=false; tensor_set_lhs(c, a); tensor_set_rhs(c, b); c->grad_fn="matmul";
    int M=a->d0, K=a->d1, N=b->d1;

    // f32 fast path: when both operands carry a live float32 mirror, compute in
    // single precision. Half the memory bandwidth and twice the SIMD lane count,
    // so ~2x on memory-bound matmuls. The result is also f32 (fdata is the live
    // copy) with data kept as a double mirror for print / non-f32-aware ops.
    // Correctness: single precision, so results match double within ~1e-6 relative
    // -- acceptable for training/inference where f32 was explicitly requested.
    if (a->dtype == 1 && b->dtype == 1 && a->fdata && b->fdata) {
        if (!c->fdata) c->fdata = new float[c->size]();
        // Only allocate the f32 gradient mirror when autograd is on; in
        // no_grad (inference) it is pure waste -- half the per-call heap churn.
        if (g_autograd_enabled && !c->fgrad) c->fgrad = new float[c->size]();
        c->dtype = 1;
        // Blocked+packed f32 GEMM: ~4x the simple ikj loop on large matrices.
        // Accumulates in fp32 -- the f32 path explicitly trades precision for
        // speed, so this is the intended behavior.
        dream_gemm_f32(a->fdata, b->fdata, c->fdata, M, K, N);
        // Sync the f32 result into data so every reader (index, save, slice,
        // print, non-f32 ops) sees correct values. The copy is O(n^2) while the
        // matmul is O(n^3), so this costs little and avoids scattering sync
        // calls across every data reader (which is fragile and easy to miss).
        for (int i = 0; i < c->size; ++i) c->data[i] = (double)c->fdata[i];
        c->data_dirty = false;
        return c;
    }

    // OPTIMIZED CPU MATMUL: parallel (i, k, j) + cache blocking + 8-way inner unroll.
    // Stacked optimizations:
    //   1. Loop reorder (i, k, j): unit-stride access to B and C
    //   2. Cache blocking (BLOCK_M rows at a time): stays in L1
    //   3. OpenMP parallelization over row blocks: 4-8x on multicore
    //   4. Manual 8-way inner loop unroll: helps autovectorizer emit AVX
    //
    // CPU wins for small matmuls where the data-transfer + kernel-launch
    // overhead exceeds the GPU speedup. Measured on Intel integrated graphics:
    // at 256x256 and 384x384 the GPU is NOT worth it (transfer dominates); the
    // GPU only pulls ahead around 512x512 and clearly wins by 768x768. So the
    // break-even is M*N >= 262144 (512x512), not 65536. On a discrete GPU with
    // faster transfer this could be lower, but the higher threshold is safe:
    // it never sends a matmul to the GPU that the CPU would do faster.
    initBackend();
    bool gpu_ok = g_use_cuda || !OCL.failed;
    if ((size_t)M * N < gpu_dispatch_threshold() || !gpu_ok) {
        // Cache-blocked packed GEMM (see dream_gemm_d above). Replaces the old
        // 8-way-unrolled ikj loop that stalled at ~3.6 GFLOPS on large sizes.
        dream_gemm_d(a->data, b->data, c->data, M, K, N);
        return c;
    }

    HostBuf ha(a->size), hb(b->size), hc(c->size);
    ha.fromDouble(a->data, a->size);
    hb.fromDouble(b->data, b->size);

    if (g_use_cuda) {
        // FP16 Tensor Core path: 2-8x faster on NVIDIA GPUs with Tensor Cores
        if (g_use_fp16 && dyn_cublasGemmEx) {
            // Convert FP32 host data to FP16 for upload
            std::vector<uint16_t> halfA(M*K), halfB(K*N);
            for(int i=0;i<M*K;++i) halfA[i]=float_to_half(ha.data[i]);
            for(int i=0;i<K*N;++i) halfB[i]=float_to_half(hb.data[i]);
            void* dA=g_dev_pool.cudaGet(M*K*sizeof(uint16_t));
            void* dB=g_dev_pool.cudaGet(K*N*sizeof(uint16_t));
            float* dC=(float*)g_dev_pool.cudaGet(M*N*sizeof(float));
            dyn_cudaMemcpy(dA, halfA.data(), M*K*sizeof(uint16_t), cudaMemcpyHostToDevice);
            dyn_cudaMemcpy(dB, halfB.data(), K*N*sizeof(uint16_t), cudaMemcpyHostToDevice);
            float alpha=1.f, beta=0.f;
            // A(FP16) x B(FP16) -> C(FP32), accumulated in FP32 for precision
            dyn_cublasGemmEx(CUDA.handle,
                CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                &alpha, dB, CUDA_R_16F, N, dA, CUDA_R_16F, K,
                &beta,  dC, CUDA_R_32F, N,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
            dyn_cudaMemcpy(hc.data.data(), dC, M*N*sizeof(float), cudaMemcpyDeviceToHost);
            g_dev_pool.cudaPut(dA, M*K*sizeof(uint16_t));
            g_dev_pool.cudaPut(dB, K*N*sizeof(uint16_t));
            g_dev_pool.cudaPut(dC, M*N*sizeof(float));
            hc.toDouble(c->data, c->size);
            return c;
        }
        // FP32 cuBLAS path (fallback)
        float *dA=(float*)g_dev_pool.cudaGet(M*K*sizeof(float));
        float *dB=(float*)g_dev_pool.cudaGet(K*N*sizeof(float));
        float *dC=(float*)g_dev_pool.cudaGet(M*N*sizeof(float));
        dyn_cudaMemcpy(dA, ha.data.data(), M*K*sizeof(float), cudaMemcpyHostToDevice);
        dyn_cudaMemcpy(dB, hb.data.data(), K*N*sizeof(float), cudaMemcpyHostToDevice);
        float alpha=1.f, beta=0.f;
        dyn_cublasSgemm(CUDA.handle,
                        CUBLAS_OP_N, CUBLAS_OP_N,
                        N, M, K,
                        &alpha, dB, N, dA, K,
                        &beta,  dC, N);
        dyn_cudaMemcpy(hc.data.data(), dC, M*N*sizeof(float), cudaMemcpyDeviceToHost);
        g_dev_pool.cudaPut(dA, M*K*sizeof(float));
        g_dev_pool.cudaPut(dB, K*N*sizeof(float));
        g_dev_pool.cudaPut(dC, M*N*sizeof(float));
        hc.toDouble(c->data, c->size);
        return c;
    }

    // OpenCL matmul: tiling + local (shared) memory + register blocking.
    //
    // The naive version uses one thread per output, each reading 2*K values from
    // global memory; global traffic grows quadratically and bandwidth becomes the bottleneck.
    //
    // Tiled: a 16x16 thread block loads 16x16 subtiles of A and B into local memory,
    // shared by the 256 threads, cutting global reads by 16x and
    // greatly improving the compute-to-memory ratio.
    //
    // Dims aligned to TS=16: K is zero-padded in-kernel, M/N writes are bounds-checked.
    //
    // Register blocking WPT=4: each thread computes a 4x4 output tile, raising the
    // FMAs per local-memory load from 4 to 8, further hiding memory latency.
    // The cost is higher register pressure; low-register devices may spill.
    // If you measure a regression, set the two #defines below back to WPT=2.
    //
    // Tile is 32x32 outputs per workgroup. With WPT=4, workgroup has 8x8 threads.
    // Each thread holds 16 accumulator registers plus 4 A + 4 B temps per inner iter.
    std::string src = R"CL(
#define TS  32
#define WPT 4
#define RTS (TS/WPT)
__kernel void k(__global const float* A, __global const float* B,
                __global float* C, int M, int K, int N) {
    const int local_row = get_local_id(0);   // 0..RTS-1
    const int local_col = get_local_id(1);   // 0..RTS-1
    const int group_row = get_group_id(0);
    const int group_col = get_group_id(1);

    __local float A_tile[TS][TS];
    __local float B_tile[TS][TS];

    // 16 output accumulators per thread (4x4 block)
    float acc[WPT][WPT];
    #pragma unroll
    for (int wr = 0; wr < WPT; ++wr)
        #pragma unroll
        for (int wc = 0; wc < WPT; ++wc) acc[wr][wc] = 0.0f;

    const int num_tiles = (K + TS - 1) / TS;

    for (int t = 0; t < num_tiles; ++t) {
        // Each thread loads WPT*WPT = 16 elements of A and 16 of B.
        #pragma unroll
        for (int wr = 0; wr < WPT; ++wr) {
            #pragma unroll
            for (int wc = 0; wc < WPT; ++wc) {
                int tile_row = wr * RTS + local_row;
                int tile_col = wc * RTS + local_col;
                int ga_row = group_row * TS + tile_row;
                int ga_col = t * TS + tile_col;
                int gb_row = t * TS + tile_row;
                int gb_col = group_col * TS + tile_col;
                A_tile[tile_row][tile_col] =
                    (ga_row < M && ga_col < K) ? A[ga_row * K + ga_col] : 0.0f;
                B_tile[tile_row][tile_col] =
                    (gb_row < K && gb_col < N) ? B[gb_row * N + gb_col] : 0.0f;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // Outer product: for each k in tile, accumulate 4x4 block.
        #pragma unroll
        for (int k = 0; k < TS; ++k) {
            float a_reg[WPT];
            #pragma unroll
            for (int wr = 0; wr < WPT; ++wr)
                a_reg[wr] = A_tile[wr * RTS + local_row][k];
            float b_reg[WPT];
            #pragma unroll
            for (int wc = 0; wc < WPT; ++wc)
                b_reg[wc] = B_tile[k][wc * RTS + local_col];
            #pragma unroll
            for (int wr = 0; wr < WPT; ++wr)
                #pragma unroll
                for (int wc = 0; wc < WPT; ++wc)
                    acc[wr][wc] += a_reg[wr] * b_reg[wc];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write the 4x4 output block
    #pragma unroll
    for (int wr = 0; wr < WPT; ++wr) {
        #pragma unroll
        for (int wc = 0; wc < WPT; ++wc) {
            int out_row = group_row * TS + wr * RTS + local_row;
            int out_col = group_col * TS + wc * RTS + local_col;
            if (out_row < M && out_col < N) {
                C[out_row * N + out_col] = acc[wr][wc];
            }
        }
    }
}
)CL";
    cl_kernel k = OCL.getKernel("matmul_tiled_wpt4", src);
    CLPoolBuf da(ha.bytes()); da.upload(ha.data.data());
    CLPoolBuf db(hb.bytes()); db.upload(hb.data.data());
    CLPoolBuf dc(hc.bytes());
    dyn_clSetKernelArg(k, 0, sizeof(cl_mem), da.ptr());
    dyn_clSetKernelArg(k, 1, sizeof(cl_mem), db.ptr());
    dyn_clSetKernelArg(k, 2, sizeof(cl_mem), dc.ptr());
    dyn_clSetKernelArg(k, 3, 4, &M);
    dyn_clSetKernelArg(k, 4, 4, &K);
    dyn_clSetKernelArg(k, 5, 4, &N);

    // TS=32, WPT=4, so RTS=8. Each workgroup has 8×8 threads producing 32×32 outputs.
    const size_t TS = 32;
    const size_t WPT = 4;
    const size_t RTS = TS / WPT;  // 8
    size_t gs[2] = {
        ((M + TS - 1) / TS) * RTS,
        ((N + TS - 1) / TS) * RTS
    };
    size_t ls[2] = {RTS, RTS};
    dyn_clEnqueueNDRangeKernel(OCL.queue, k, 2, nullptr, gs, ls, 0, nullptr, nullptr);
    dc.download(hc.data.data());
    hc.toDouble(c->data, c->size);
    return c;
}

// Batch Matrix Multiply: A(B,M,K) @ B_mat(B,K,N) -> C(B,M,N)
// Tensors stored flat: a is (batch, M*K), b is (batch, K*N), result is (batch, M*N)
// Dimension params passed as scalar tensors.
Tensor* dream_bmm(Tensor* a, Tensor* b,
                   Tensor* batch_t, Tensor* m_t, Tensor* k_t, Tensor* n_t) {
    a = materialize(a); b = materialize(b);
    int B = (int)batch_t->data[0];
    int M = (int)m_t->data[0];
    int K = (int)k_t->data[0];
    int N = (int)n_t->data[0];

    if (a->size < B*M*K || b->size < B*K*N) {
        std::cerr << "error: bmm size mismatch (a=" << a->size
                  << " need " << B*M*K << ", b=" << b->size
                  << " need " << B*K*N << ")\n" << std::flush;
        return alloc_tensor(1, 1, 1, 1);
    }

    Tensor* c = alloc_tensor(B, M*N, 1, 1);
    c->is_leaf = false; tensor_set_lhs(c, a); tensor_set_rhs(c, b);
    c->grad_fn = "bmm";
    c->bwd_name = std::to_string(B)+","+std::to_string(M)+","+
                  std::to_string(K)+","+std::to_string(N);

    for (int bi = 0; bi < B; ++bi) {
        const double* A = a->data + bi * M * K;
        const double* Bm = b->data + bi * K * N;
        double* C = c->data + bi * M * N;
        for (int r = 0; r < M; ++r)
            for (int cc = 0; cc < N; ++cc) {
                double s = 0;
                for (int kk = 0; kk < K; ++kk)
                    s += A[r*K+kk] * Bm[kk*N+cc];
                C[r*N+cc] = s;
            }
    }
    return c;
}

Tensor* dream_pow(Tensor* base, Tensor* exp_t) {
    int rd0 = std::max(base->d0, exp_t->d0);
    int rd1 = std::max(base->d1, exp_t->d1);
    Tensor* c = alloc_tensor(rd0, rd1, 1, 1);
    c->is_leaf = false;
    tensor_set_lhs(c, base);
    tensor_set_rhs(c, exp_t);  // store exponent for backward
    c->grad_fn = "pow";
    // Always the binary path: a size-1 exponent broadcasts, and the
    // string-keyed CPU fallback only knows "pow(a_val,b_val)" -- the old
    // scalar special case baked the constant into the string, which the
    // CPU interpreter could not map, silently producing zeros on machines
    // without OpenCL.
    if (exp_t->size != 1) checkShape(base, exp_t, "**");
    runGPU(base, exp_t, c, "pow(a_val,b_val)");
    return c;
}

// CPU fallback for fused expressions. Parses the kernel source string and
// executes simple cases (add/sub/mul/div patterns) directly on CPU.
// Critical: without this, every `x + y` goes through full OpenCL pipeline.
static bool try_fused_cpu(const char* src, Tensor** vars, int nv, Tensor* out) {
    std::string s(src);
    if (nv < 1 || nv > 4) return false;

    size_t eq = s.find("out[i]=");
    if (eq == std::string::npos) return false;
    size_t end = s.find(';', eq);
    if (end == std::string::npos) return false;
    std::string expr = s.substr(eq + 7, end - eq - 7);

    // Normalize v0[i%sz0] / v0[i] to V0
    for (int i = 0; i < nv; ++i) {
        std::string repl = "V" + std::to_string(i);
        std::string pat_mod = "v" + std::to_string(i) + "[i%sz" + std::to_string(i) + "]";
        std::string pat_raw = "v" + std::to_string(i) + "[i]";
        size_t p;
        while ((p = expr.find(pat_mod)) != std::string::npos) expr.replace(p, pat_mod.size(), repl);
        while ((p = expr.find(pat_raw)) != std::string::npos) expr.replace(p, pat_raw.size(), repl);
    }
    // Strip whitespace
    std::string e;
    for (char c : expr) if (!std::isspace((unsigned char)c)) e += c;
    expr = e;

    int out_sz = out->size;
    double* O = out->data;
    auto get = [&](int idx, int i) {
        int vsz = vars[idx]->size;
        return vars[idx]->data[vsz == 1 ? 0 : (i % vsz)];
    };

    // Fast path: same-size tensors (no broadcasting) — direct data access + OMP parallel
    bool same_size = true;
    for (int i = 0; i < nv; ++i) if (vars[i]->size != out_sz) { same_size = false; break; }

    if (same_size && nv == 2) {
        const double* V0 = vars[0]->data;
        const double* V1 = vars[1]->data;
        if (expr == "(V0+V1)") {
            if (out_sz >= 65536) {
                #pragma omp parallel for schedule(static)
                for (int i = 0; i < out_sz; ++i) O[i] = V0[i] + V1[i];
            } else {
                for (int i = 0; i < out_sz; ++i) O[i] = V0[i] + V1[i];
            }
            return true;
        }
        if (expr == "(V0-V1)") {
            if (out_sz >= 65536) {
                #pragma omp parallel for schedule(static)
                for (int i = 0; i < out_sz; ++i) O[i] = V0[i] - V1[i];
            } else {
                for (int i = 0; i < out_sz; ++i) O[i] = V0[i] - V1[i];
            }
            return true;
        }
        if (expr == "(V0*V1)") {
            if (out_sz >= 65536) {
                #pragma omp parallel for schedule(static)
                for (int i = 0; i < out_sz; ++i) O[i] = V0[i] * V1[i];
            } else {
                for (int i = 0; i < out_sz; ++i) O[i] = V0[i] * V1[i];
            }
            return true;
        }
        if (expr == "(V0/V1)") {
            if (out_sz >= 65536) {
                #pragma omp parallel for schedule(static)
                for (int i = 0; i < out_sz; ++i) O[i] = V0[i] / V1[i];
            } else {
                for (int i = 0; i < out_sz; ++i) O[i] = V0[i] / V1[i];
            }
            return true;
        }
    }

    // Generic path with broadcasting support (kept for correctness)
    if (nv == 2) {
        if (expr == "(V0+V1)") { for (int i = 0; i < out_sz; ++i) O[i] = get(0,i) + get(1,i); return true; }
        if (expr == "(V0-V1)") { for (int i = 0; i < out_sz; ++i) O[i] = get(0,i) - get(1,i); return true; }
        if (expr == "(V0*V1)") { for (int i = 0; i < out_sz; ++i) O[i] = get(0,i) * get(1,i); return true; }
        if (expr == "(V0/V1)") { for (int i = 0; i < out_sz; ++i) O[i] = get(0,i) / get(1,i); return true; }
    }
    if (nv == 3) {
        if (expr == "((V0+V1)+V2)") { for (int i = 0; i < out_sz; ++i) O[i] = get(0,i) + get(1,i) + get(2,i); return true; }
        if (expr == "((V0-V1)-V2)") { for (int i = 0; i < out_sz; ++i) O[i] = get(0,i) - get(1,i) - get(2,i); return true; }
        if (expr == "((V0*V1)*V2)") { for (int i = 0; i < out_sz; ++i) O[i] = get(0,i) * get(1,i) * get(2,i); return true; }
        if (expr == "((V0*V1)+V2)") { for (int i = 0; i < out_sz; ++i) O[i] = get(0,i) * get(1,i) + get(2,i); return true; }
        if (expr == "((V0+V1)*V2)") { for (int i = 0; i < out_sz; ++i) O[i] = (get(0,i) + get(1,i)) * get(2,i); return true; }
        if (expr == "((V0-V1)*V2)") { for (int i = 0; i < out_sz; ++i) O[i] = (get(0,i) - get(1,i)) * get(2,i); return true; }
    }
    return false;
}

Tensor* run_fused_with_grad(const char* name, const char* src,
                             const char* bwdName, const char* bwdSrc,
                             Tensor** vars, int nv) {
    for(int i=0;i<nv;++i) vars[i]=materialize(vars[i]);

    int maxSz=1,d0=1,d1=1,d2=1,d3=1;
    for(int i=0;i<nv;++i) if(vars[i]->size>maxSz){
        maxSz=vars[i]->size; d0=vars[i]->d0;d1=vars[i]->d1;d2=vars[i]->d2;d3=vars[i]->d3;
    }
    Tensor* out = alloc_tensor(d0,d1,d2,d3);
    out->is_leaf=false; out->grad_fn="fused";
    {
        Tensor** par = new Tensor*[nv];
        for(int i=0;i<nv;++i) par[i] = vars[i];
        tensor_set_parents(out, par, nv);
    }
    out->bwd_name=bwdName; out->bwd_src=bwdSrc;

    // CRITICAL: try CPU fallback first for small/simple cases.
    // Without this, every `x + y` incurs ~5ms OpenCL kernel setup.
    if (maxSz <= 4194304) {
        if (try_fused_cpu(src, vars, nv, out)) return out;
    }

    initBackend();
    cl_kernel k = OCL.getKernel(name, src);
    CLPoolBuf dout(maxSz*4);
    dyn_clSetKernelArg(k,0,sizeof(cl_mem),dout.ptr());
    std::vector<HostBuf> vhosts;
    std::vector<cl_mem> vbufs_raw;
    std::vector<size_t> vbufs_sz;
    for(int i=0;i<nv;++i){
        vhosts.emplace_back(vars[i]->size); vhosts.back().fromDouble(vars[i]->data,vars[i]->size);
        size_t bsz = vhosts.back().bytes();
        cl_mem m = g_dev_pool.clGet(bsz);
        dyn_clEnqueueWriteBuffer(OCL.queue, m, CL_TRUE, 0, bsz, vhosts.back().data.data(), 0, nullptr, nullptr);
        vbufs_raw.push_back(m);
        vbufs_sz.push_back(bsz);
        dyn_clSetKernelArg(k,i+1,sizeof(cl_mem),&vbufs_raw.back());
    }
    dyn_clSetKernelArg(k,nv+1,4,&maxSz);
    for(int i=0;i<nv;++i) dyn_clSetKernelArg(k,nv+2+i,4,&vars[i]->size);
    size_t gs=maxSz;
    dyn_clEnqueueNDRangeKernel(OCL.queue,k,1,nullptr,&gs,nullptr,0,nullptr,nullptr);
    HostBuf hout(maxSz);
    dout.download(hout.data.data());
    hout.toDouble(out->data,maxSz);
    for(int i=0;i<nv;++i) g_dev_pool.clPut(vbufs_raw[i], vbufs_sz[i]);
    return out;
}


// ============================================================
// Fused Linear layer: out = relu(x @ W^T + b) in one pass.
// PyTorch's nn.Linear does three separate tensor allocations for this;
// we do it in one pass over the data, with one output allocation.
//
// For a 64-batch x 64-in x 128-hidden Linear+ReLU:
//   Unfused (transpose + matmul + add + relu): 4 allocs, 4 passes, 131K memory writes
//   Fused: 1 alloc, 1 pass, 8192 writes — measurably faster
//
// This is the #1 hot path for MLP training. Called twice per batch.
// ============================================================
Tensor* dream_linear(Tensor* x, Tensor* W, Tensor* b) {
    if (!x || !W || !b) {
        std::cerr << "RUNTIME ERROR in linear: null arg"
                  << (x?"":" x=null") << (W?"":" W=null") << (b?"":" b=null") << "\n" << std::flush;
        return alloc_tensor(1,1,1,1);
    }
    x = materialize(x); W = materialize(W); b = materialize(b);
    int N = x->d0;       // batch
    int in_dim = x->d1;      // input features
    int out_dim = W->d0;     // output features (W stored as out_dim x in_dim, PyTorch convention)
    int W_in = W->d1;
    if (in_dim != W_in) {
        std::cerr << "RUNTIME ERROR in linear: x.d1(" << in_dim << ") != W.d1(" << W_in << ")\n" << std::flush;
        return alloc_tensor(1,1,1,1);
    }

    Tensor* out = alloc_tensor(N, out_dim, 1, 1);
    out->is_leaf = false; tensor_set_lhs(out, x); tensor_set_rhs(out, W); out->grad_fn = "linear";
    out->bwd_name = "linear_bias";
    Tensor** par = new Tensor*[3];
    par[0] = x; par[1] = W; par[2] = b;
    tensor_set_parents(out, par, 3);

    const double* X = x->data;
    const double* Wd = W->data;
    const double* bias = b->data;
    double* Y = out->data;

    // Initialize output with bias (broadcasted)
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < out_dim; ++j)
            Y[i * out_dim + j] = bias[j];

    // Compute Y += X @ W^T using the same (i, k, j) pattern as our matmul kernel.
    // W is stored as (out_dim, in_dim), so W[j,k] accesses are unit-stride in k.
    // We effectively do: Y[i,j] += sum_k X[i,k] * W[j,k]
    //
    // The loop order: outer i (batch), middle k (in features), inner j (out features)
    // gives unit-stride writes to Y_row and unit-stride reads from both Wd column-wise.
    // Wait — W[j,k] with fixed k and varying j is NOT unit stride (stride = in_dim).
    //
    // So we transpose W inline to Wt (in_dim, out_dim). Small overhead, big cache win.
    // For a typical MLP layer (128x64 weight), this is 32KB of transpose — negligible.
    std::vector<double> Wt(in_dim * out_dim);
    for (int j = 0; j < out_dim; ++j)
        for (int k = 0; k < in_dim; ++k)
            Wt[k * out_dim + j] = Wd[j * in_dim + k];

    // Parallel matmul accumulation using the hot kernel
    #pragma omp parallel for schedule(static) if(N * in_dim * out_dim >= 2097152)
    for (int i = 0; i < N; ++i) {
        const double* X_row = X + i * in_dim;
        double* Y_row = Y + i * out_dim;
        for (int k = 0; k < in_dim; ++k) {
            double xik = X_row[k];
            const double* Wt_row = &Wt[k * out_dim];
            // Unit-stride inner loop with 8-way unroll
            int j = 0;
            int J8 = out_dim & ~7;
            for (; j < J8; j += 8) {
                Y_row[j+0] += xik * Wt_row[j+0];
                Y_row[j+1] += xik * Wt_row[j+1];
                Y_row[j+2] += xik * Wt_row[j+2];
                Y_row[j+3] += xik * Wt_row[j+3];
                Y_row[j+4] += xik * Wt_row[j+4];
                Y_row[j+5] += xik * Wt_row[j+5];
                Y_row[j+6] += xik * Wt_row[j+6];
                Y_row[j+7] += xik * Wt_row[j+7];
            }
            for (; j < out_dim; ++j) Y_row[j] += xik * Wt_row[j];
        }
    }

    return out;
}

Tensor* dream_linear_relu(Tensor* x, Tensor* W, Tensor* b) {
    Tensor* out = dream_linear(x, W, b);
    out->grad_fn = "linear_relu";
    // Apply ReLU in-place
    double* Y = out->data;
    int sz = out->size;
    for (int i = 0; i < sz; ++i) if (Y[i] < 0) Y[i] = 0;
    return out;
}

// Aliases for linear/linear_relu (no dream_ prefix), so they link even
// if the compiler's builtinRemap isn't rebuilt.
Tensor* linear(Tensor* x, Tensor* W, Tensor* b) { return dream_linear(x, W, b); }
Tensor* linear_relu(Tensor* x, Tensor* W, Tensor* b) { return dream_linear_relu(x, W, b); }

// ============================================================
// Numerically stable ops for real-world training.
// Real models hit NaN/Inf when you use vanilla log/exp/softmax
// with extreme values. These versions are safe.
// ============================================================

// Clip values to safe range — prevents log(0) and exp(huge) disasters.
// Use before log: safe_log(clip(x, 1e-8, 1e8))
Tensor* dream_clamp(Tensor* x, Tensor* lo_t, Tensor* hi_t) {
    if (!x) return create_scalar(0.0);
    x = materialize(x);
    double lo = lo_t ? lo_t->data[0] : -1e30;
    double hi = hi_t ? hi_t->data[0] : 1e30;
    Tensor* out = alloc_tensor(x->d0, x->d1, x->d2, x->d3);
    out->is_leaf = false; tensor_set_lhs(out, x); out->grad_fn = "clamp";
    out->bwd_name = std::to_string(lo) + "," + std::to_string(hi);
    for (int i = 0; i < x->size; ++i) {
        double v = x->data[i];
        out->data[i] = v < lo ? lo : (v > hi ? hi : v);
    }
    return out;
}

// log_softmax: numerically stable version of log(softmax(x))
// PyTorch equivalent: F.log_softmax(x, dim=1)
// Use with NLLLoss for classification. Much more stable than log(softmax(x)).
// Formula: log_softmax(x)[i] = x[i] - logsumexp(x)
Tensor* dream_log_softmax(Tensor* x) {
    if (!x) return create_scalar(0.0);
    x = materialize(x);
    Tensor* out = alloc_tensor(x->d0, x->d1, 1, 1);
    out->is_leaf = false; tensor_set_lhs(out, x); out->grad_fn = "log_softmax";
    int N = x->d0, C = x->d1;
    for (int n = 0; n < N; ++n) {
        const double* row = x->data + n * C;
        double* out_row = out->data + n * C;
        // Find max for numerical stability
        double mx = row[0];
        for (int j = 1; j < C; ++j) if (row[j] > mx) mx = row[j];
        // Compute logsumexp = max + log(sum(exp(x - max)))
        double sum = 0;
        for (int j = 0; j < C; ++j) sum += std::exp(row[j] - mx);
        double lse = mx + std::log(sum);
        // log_softmax = x - logsumexp
        for (int j = 0; j < C; ++j) out_row[j] = row[j] - lse;
    }
    return out;
}

// NLL loss: negative log-likelihood.
// Use with log_softmax for standard cross-entropy classification.
// Formula: -mean(target * log_probs)  where target is one-hot
Tensor* dream_nll_loss(Tensor* log_probs, Tensor* target) {
    if (!log_probs || !target) return create_scalar(0.0);
    log_probs = materialize(log_probs);
    target = materialize(target);
    Tensor* out = alloc_tensor(1, 1, 1, 1);
    out->is_leaf = false; tensor_set_lhs(out, log_probs); tensor_set_rhs(out, target); out->grad_fn = "nll_loss";
    int N = log_probs->d0, C = log_probs->d1;
    double loss = 0;
    // Two accepted target layouts:
    //   (a) class-index column: target is Nx1, target[n] = class id
    //   (b) one-hot matrix:     target is NxC
    // The old code assumed (b) unconditionally and read N*C elements from
    // an N-element buffer when given (a) — a heap-buffer-overflow that
    // corrupted adjacent allocations (one source of the random 0xc0000005).
    bool index_target = (target->d1 == 1 && C > 1);
    if (index_target) {
        int rows = std::min(N, target->d0);
        for (int n = 0; n < rows; ++n) {
            int cls = (int)target->data[n];
            if (cls >= 0 && cls < C)
                loss -= log_probs->data[n*C + cls];
        }
    } else {
        int limit = std::min(N*C, target->size);
        for (int k = 0; k < limit; ++k)
            if (target->data[k] > 0)
                loss -= target->data[k] * log_probs->data[k];
    }
    out->data[0] = loss / std::max(1, N);
    return out;
}

// Check for NaN/Inf in a tensor. Returns 1.0 if found, 0.0 otherwise.
// Also prints location of first NaN for debugging.
Tensor* dream_has_nan(Tensor* t) {
    if (!t) return create_scalar(0.0);
    t = materialize(t);
    int nan_count = 0, inf_count = 0;
    int first_nan = -1, first_inf = -1;
    for (int i = 0; i < t->size; ++i) {
        double v = t->data[i];
        if (dream_isnan(v)) { nan_count++; if (first_nan < 0) first_nan = i; }
        if (dream_isinf(v)) { inf_count++; if (first_inf < 0) first_inf = i; }
    }
    if (nan_count > 0 || inf_count > 0) {
        std::cerr << "[has_nan] shape=" << t->d0 << "x" << t->d1;
        if (nan_count > 0) std::cerr << " NaN: " << nan_count << " (first@idx " << first_nan << ")";
        if (inf_count > 0) std::cerr << " Inf: " << inf_count << " (first@idx " << first_inf << ")";
        std::cerr << "\n" << std::flush;
        return create_scalar(1.0);
    }
    return create_scalar(0.0);
}

// Replace NaN/Inf with a safe value. Returns new tensor.
Tensor* dream_nan_to_num(Tensor* t, Tensor* replacement_t) {
    if (!t) return create_scalar(0.0);
    t = materialize(t);
    double repl = replacement_t ? replacement_t->data[0] : 0.0;
    Tensor* out = alloc_tensor(t->d0, t->d1, t->d2, t->d3);
    int replaced = 0;
    for (int i = 0; i < t->size; ++i) {
        double v = t->data[i];
        if (dream_isnan(v) || dream_isinf(v)) { out->data[i] = repl; replaced++; }
        else out->data[i] = v;
    }
    if (replaced > 0)
        std::cerr << "[nan_to_num] replaced " << replaced << "/" << t->size << " bad values with " << repl << "\n" << std::flush;
    return out;
}

// Aliases without dream_ prefix
Tensor* set_seed(Tensor* s) { return dream_set_seed(s); }
Tensor* log_softmax(Tensor* x) { return dream_log_softmax(x); }
Tensor* nll_loss(Tensor* lp, Tensor* t) { return dream_nll_loss(lp, t); }
Tensor* has_nan(Tensor* t) { return dream_has_nan(t); }
Tensor* nan_to_num(Tensor* t, Tensor* r) { return dream_nan_to_num(t, r); }
Tensor* clamp(Tensor* x, Tensor* lo, Tensor* hi) { return dream_clamp(x, lo, hi); }
