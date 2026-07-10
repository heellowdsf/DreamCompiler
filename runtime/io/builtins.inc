//(⊙ˍ⊙)
// assert: if condition tensor is zero (false), print message and exit
Tensor* dream_assert_fn(Tensor* cond, const char* msg) {
    if (!cond) { std::cerr << "assert: null condition\n" << std::flush; exit(1); }
    bool passed = (cond->size > 0 && cond->data[0] != 0.0);
    if (!passed) {
        std::cerr << "assertion failed: " << (msg && *msg ? msg : "(no message)") << "\n" << std::flush;
        exit(1);
    }
    return tensor_incref(cond);
}

// save tensor to a simple binary file: magic + shape + data
// Export a tensor as plain text: first line "rows cols", then rows lines
// of space-separated doubles. Trivially loadable from C (see game runner).
Tensor* dream_save_matrix(Tensor* t, Tensor* name_t) {
    if (!t || !name_t) return tensor_incref(t);
    t = materialize(t);
    std::string fn;
    if (name_t->grad_fn == "string")
        for (int i = 0; i < name_t->size; ++i) fn += (char)(int)name_t->data[i];
    else { std::cerr << "save_matrix: name must be a string\n"; return tensor_incref(t); }
    int R = t->d0 > 0 ? t->d0 : 1, C = t->size / R;
    FILE* f = std::fopen(fn.c_str(), "w");
    if (!f) { std::cerr << "save_matrix: cannot open " << fn << "\n"; return tensor_incref(t); }
    std::fprintf(f, "%d %d\n", R, C);
    for (int r = 0; r < R; ++r) {
        for (int c = 0; c < C; ++c) std::fprintf(f, "%.9g ", t->data[(size_t)r*C+c]);
        std::fprintf(f, "\n");
    }
    std::fclose(f);
    return tensor_incref(t);
}

Tensor* dream_save(Tensor* t, const char* filename) {
    if (!t) { std::cerr << "save: null tensor\n" << std::flush; return nullptr; }
    dream_mkdirs_for(filename);
    std::FILE* f = std::fopen(filename, "wb");
    if (!f) { std::cerr << "save: cannot open '" << filename << "'\n" << std::flush; return tensor_incref(t); }
    uint32_t magic = 0x44524541; // "DREA"
    std::fwrite(&magic,    4, 1, f);
    std::fwrite(&t->d0,    4, 1, f);
    std::fwrite(&t->d1,    4, 1, f);
    std::fwrite(&t->d2,    4, 1, f);
    std::fwrite(&t->d3,    4, 1, f);
    std::fwrite(&t->size,  4, 1, f);
    std::fwrite(t->data, sizeof(double), t->size, f);
    std::fclose(f);
    std::cerr << "saved '" << filename << "' shape=" << t->d0 << "x" << t->d1
              << " (" << t->size << " values)\n" << std::flush;
    return tensor_incref(t);
}

// load tensor from file saved by dream_save
Tensor* dream_load(const char* filename) {
    std::FILE* f = std::fopen(filename, "rb");
    if (!f) { std::cerr << "load: cannot open '" << filename << "'\n" << std::flush; exit(1); }
    uint32_t magic = 0;
    std::fread(&magic, 4, 1, f);
    if (magic != 0x44524541) {
        std::cerr << "load: '" << filename << "' is not a Dream tensor file\n" << std::flush;
        std::fclose(f); exit(1);
    }
    int d0,d1,d2,d3,sz;
    std::fread(&d0, 4, 1, f); std::fread(&d1, 4, 1, f);
    std::fread(&d2, 4, 1, f); std::fread(&d3, 4, 1, f);
    std::fread(&sz, 4, 1, f);
    Tensor* t = alloc_tensor(d0, d1, d2, d3);
    std::fread(t->data, sizeof(double), sz, f);
    std::fclose(f);
    std::cerr << "loaded '" << filename << "' shape=" << d0 << "x" << d1
              << " (" << sz << " values)\n" << std::flush;
    return t;
}

// flatten: reshape to (1 x size)
Tensor* dream_flatten(Tensor* t) {
    Tensor* c = alloc_tensor(1, t->size, 1, 1);
    c->is_leaf = false; tensor_set_lhs(c, t); c->grad_fn = "flatten";
    std::copy(t->data, t->data + t->size, c->data);
    return c;
}

// element-wise floor/ceil/round
Tensor* dream_floor(Tensor* t) { Tensor* c=makeUnary(t,"floor"); runGPUUnary(t,c,"floor(a_val)"); return c; }
Tensor* dream_ceil (Tensor* t) { Tensor* c=makeUnary(t,"ceil");  runGPUUnary(t,c,"ceil(a_val)");  return c; }
Tensor* dream_round(Tensor* t) { Tensor* c=makeUnary(t,"round"); runGPUUnary(t,c,"round(a_val)"); return c; }

// max / min element value
Tensor* dream_max_val(Tensor* t) {
    double v = t->data[0];
    for (int i=1;i<t->size;++i) if(t->data[i]>v) v=t->data[i];
    return create_scalar(v);
}
Tensor* dream_min_val(Tensor* t) {
    double v = t->data[0];
    for (int i=1;i<t->size;++i) if(t->data[i]<v) v=t->data[i];
    return create_scalar(v);
}

// l2 norm of the entire tensor
Tensor* dream_norm(Tensor* t) {
    double s=0; for(int i=0;i<t->size;++i) s+=t->data[i]*t->data[i];
    return create_scalar(std::sqrt(s));
}

// variance and standard deviation
Tensor* dream_var(Tensor* t) {
    if (t->size <= 1) return create_scalar(0.0);
    double s=0; for(int i=0;i<t->size;++i) s+=t->data[i];
    double mu = s/t->size;
    double v=0; for(int i=0;i<t->size;++i){ double d=t->data[i]-mu; v+=d*d; }
    return create_scalar(v/t->size);
}
Tensor* dream_std(Tensor* t) {
    Tensor* v = dream_var(t);
    double sv = std::sqrt(v->data[0]);
    return create_scalar(sv);
}

// cumulative sum (CPU)
Tensor* dream_cumsum(Tensor* t) {
    Tensor* c = alloc_tensor(t->d0,t->d1,t->d2,t->d3);
    c->is_leaf = false; tensor_set_lhs(c, t); c->grad_fn = "cumsum";
    double s=0;
    for(int i=0;i<t->size;++i){ s+=t->data[i]; c->data[i]=s; }
    return c;
}

// linspace: n evenly-spaced values from start to end (inclusive)
Tensor* dream_linspace(Tensor* start, Tensor* end, Tensor* n) {
    int count = std::max(1, (int)n->data[0]);
    Tensor* t = alloc_tensor(1, count, 1, 1);
    double lo = start->data[0], hi = end->data[0];
    for(int i=0;i<count;++i)
        t->data[i] = (count==1) ? lo : lo + (hi-lo)*i/(count-1);
    return t;
}

// arange: values from start to end with given step
Tensor* dream_arange(Tensor* start, Tensor* end, Tensor* step_t) {
    double lo=start->data[0], hi=end->data[0], st=step_t->data[0];
    if (st == 0) { std::cerr << "arange: step cannot be 0\n" << std::flush; exit(1); }
    int count = (int)std::ceil((hi-lo)/st);
    if (count < 0) count = 0;
    Tensor* t = alloc_tensor(1, count, 1, 1);
    for(int i=0;i<count;++i) t->data[i] = lo + st*i;
    return t;
}

// dropout: zero out random elements with probability (1-keep_rate) during training
Tensor* dream_dropout(Tensor* t, Tensor* keep_rate) {
    float rate = float(keep_rate->data[0]);
    Tensor* c = alloc_tensor(t->d0,t->d1,t->d2,t->d3);
    c->is_leaf = false; tensor_set_lhs(c, t); c->grad_fn = "dropout";
    { char buf[32]; std::snprintf(buf, sizeof(buf), "%.17g", (double)rate);
      c->bwd_name = buf; }   // backward needs keep to scale grads by 1/keep
    auto& rng = get_rng();
    std::uniform_real_distribution<float> dist(0.f, 1.f);
    for(int i=0;i<t->size;++i)
        c->data[i] = (dist(rng) < rate) ? t->data[i] / rate : 0.0;
    return c;
}

// Layer normalization (over all elements)
// Stores inv_std and count in bwd_name for proper backward pass
Tensor* dream_layer_norm(Tensor* t) {
    if (t->size == 0) return alloc_tensor(t->d0,t->d1,t->d2,t->d3);
    t = materialize(t);
    // Row-wise (per last-dim group), matching PyTorch's LayerNorm over the
    // feature dimension. The old version normalized over the WHOLE tensor,
    // coupling samples across a batch. The backward recomputes per-row
    // inv_std from the input, so nothing is stashed here.
    int R = t->d0 > 0 ? t->d0 : 1;
    int C = t->size / R;
    Tensor* c = alloc_tensor(t->d0,t->d1,t->d2,t->d3);
    c->is_leaf = false; tensor_set_lhs(c, t); c->grad_fn = "layer_norm";
    for (int r = 0; r < R; ++r) {
        const double* x = t->data + (size_t)r * C;
        double* y = c->data + (size_t)r * C;
        double mu = 0; for (int j = 0; j < C; ++j) mu += x[j];
        mu /= C;
        double var = 0; for (int j = 0; j < C; ++j) { double d = x[j]-mu; var += d*d; }
        var /= C;
        double inv_std = 1.0 / std::sqrt(var + 1e-5);
        for (int j = 0; j < C; ++j) y[j] = (x[j] - mu) * inv_std;
    }
    return c;
}

// Clip gradient norm: scale gradients so their L2 norm <= max_norm
Tensor* dream_clip_grad(Tensor* t, Tensor* max_norm) {
    double mn = max_norm->data[0];
    double n=0; for(int i=0;i<t->size;++i) n+=t->grad[i]*t->grad[i];
    n = std::sqrt(n);
    if (n > mn) {
        double scale = mn/n;
        for(int i=0;i<t->size;++i) t->grad[i]*=scale;
    }
    return tensor_incref(t);
}

// Current time in seconds (useful for measuring training speed)
Tensor* dream_time() {
    auto now = std::chrono::steady_clock::now();
    static auto start = now;
    double secs = std::chrono::duration<double>(now - start).count();
    return create_scalar(secs);
}

// SGD with momentum step (modifies velocity in-place, returns new param)
// Usage: v = zeros(rows, cols)
//        w = sgd_step(w, grad_w, v, lr, momentum)
Tensor* dream_sgd_step(Tensor* param, Tensor* grad, Tensor* velocity,
                        Tensor* lr, Tensor* momentum) {
    double lr_v  = lr->data[0];
    double mom_v = momentum->data[0];
    Tensor* newp = alloc_tensor(param->d0,param->d1,param->d2,param->d3);
    newp->is_leaf = false; tensor_set_lhs(newp, param); newp->grad_fn = "sgd";
    for(int i=0;i<param->size;++i) {
        double g = grad->data[i % grad->size];
        velocity->data[i % velocity->size] =
            mom_v * velocity->data[i % velocity->size] - lr_v * g;
        newp->data[i] = param->data[i] + velocity->data[i % velocity->size];
    }
    return newp;
}

// Adam optimizer step (modifies m and v in-place, returns new param)
// Usage: m = zeros(rows, cols)
//        v = zeros(rows, cols)
//        step = 1.0
//        w = adam_step(w, grad_w, m, v, lr, beta1, beta2, eps, step)
//        step += 1.0
Tensor* dream_adam_step(Tensor* param, Tensor* grad, Tensor* m, Tensor* v,
                         Tensor* lr, Tensor* beta1, Tensor* beta2, Tensor* eps, Tensor* step) {
    double lr_v   = lr->data[0];
    double b1     = beta1->data[0];
    double b2     = beta2->data[0];
    double eps_v  = eps->data[0];
    double t      = step->data[0];
    double cor1   = 1.0 - std::pow(b1, t);
    double cor2   = 1.0 - std::pow(b2, t);
    Tensor* newp  = alloc_tensor(param->d0,param->d1,param->d2,param->d3);
    newp->is_leaf = false; tensor_set_lhs(newp, param); newp->grad_fn = "adam";
    for(int i=0;i<param->size;++i) {
        double g = grad->data[i % grad->size];
        m->data[i % m->size] = b1 * m->data[i % m->size] + (1.0-b1)*g;
        v->data[i % v->size] = b2 * v->data[i % v->size] + (1.0-b2)*g*g;
        double mh = m->data[i % m->size] / cor1;
        double vh = v->data[i % v->size] / cor2;
        newp->data[i] = param->data[i] - lr_v * mh / (std::sqrt(vh) + eps_v);
    }
    return newp;
}

// Add backward pass entries for the new ops in topoSort's backward loop
// (flatten, layer_norm, dropout, sgd, adam have their gradients computed during backward)

// 2D index: t[row, col] -> data[row * d1 + col], always returns a scalar
Tensor* tensor_index_2d(Tensor* t, Tensor* row_t, Tensor* col_t) {
    if (!t || !row_t || !col_t) {
        std::cerr << "error: null in tensor_index_2d\n" << std::flush;
        return create_temp_scalar(0.0);
    }
    int row = (int)row_t->data[0];
    int col = (int)col_t->data[0];
    int rows = t->d0, cols = t->d1;
    if (row < 0) row += rows;
    if (col < 0) col += cols;
    if (row < 0 || row >= rows || col < 0 || col >= cols) {
        std::cerr << "runtime error: index [" << (int)row_t->data[0] << ", "
                  << (int)col_t->data[0] << "] out of bounds (shape "
                  << rows << "x" << cols << ")\n" << std::flush;
        return create_temp_scalar(0.0);
    }
    return create_temp_scalar(t->data[row * cols + col]);
}

// 1D index: t[i] returns a sub-tensor (row for 2D) or scalar

// no_grad in-place operations: modify tensor data directly, never touch the computation graph.
// These are essential for optimizer weight updates (SGD/Adam step) where:
//   1. The update must NOT be recorded in the autograd graph
//   2. No new tensor should be allocated (prevents OOM in training loops)
//   3. The tensor's identity (pointer) must not change (variables stay rooted)

// param.data[i] += delta.data[i % delta.size]
Tensor* dream_inplace_add_(Tensor* t, Tensor* delta) {
    if (!t || !delta) return tensor_incref(t);
    t = materialize(t);
    delta = materialize(delta);
    for (int i = 0; i < t->size; ++i)
        t->data[i] += delta->data[i % delta->size];
    bump_version(t);
    return tensor_incref(t);
}

// param.data[i] -= delta.data[i % delta.size]
Tensor* dream_inplace_sub_(Tensor* t, Tensor* delta) {
    if (!t || !delta) return tensor_incref(t);
    t = materialize(t);
    delta = materialize(delta);
    for (int i = 0; i < t->size; ++i)
        t->data[i] -= delta->data[i % delta->size];
    bump_version(t);
    return tensor_incref(t);
}

// param.data[i] *= scale.data[i % scale.size]
Tensor* dream_inplace_mul_(Tensor* t, Tensor* scale) {
    if (!t || !scale) return tensor_incref(t);
    t = materialize(t);
    scale = materialize(scale);
    for (int i = 0; i < t->size; ++i)
        t->data[i] *= scale->data[i % scale->size];
    bump_version(t);
    return tensor_incref(t);
}

// Overwrite dest data with src data, reset graph links. Dest keeps its identity.
// Usage: no_grad_assign(W, new_W)  -- W pointer unchanged, data overwritten
Tensor* dream_no_grad_assign(Tensor* dest, Tensor* src) {
    if (!dest || !src) return tensor_incref(dest);
    src = materialize(src);
    if (dest->size != src->size) {
        if (dest->owns_data) {
            g_pool.release(dest->data, dest->size);
            g_pool.release_grad(dest->grad, dest->size);
        }
        dest->data = g_pool.alloc(src->size);
        dest->grad = g_pool.alloc_grad(src->size);
        dest->size = src->size;
        dest->d0 = src->d0; dest->d1 = src->d1;
        dest->d2 = src->d2; dest->d3 = src->d3;
        dest->owns_data = true;
    }
    std::copy(src->data, src->data + src->size, dest->data);
    // Do NOT touch grad -- keep accumulated gradients from backward()
    // Do NOT set grad_fn/lhs/rhs -- this is invisible to autograd
    bump_version(dest);
    return tensor_incref(dest);
}

// Combined param update: param -= lr * grad (single pass, no allocation, no graph)
// This is the no_grad equivalent of: param = param - lr * grad(param)
Tensor* dream_param_update(Tensor* param, Tensor* grad_t, Tensor* lr_t) {
    if (!param) { std::cerr << "RUNTIME ERROR in param_update: param is null\n" << std::flush; return nullptr; }
    if (!grad_t) { std::cerr << "RUNTIME ERROR in param_update: gradient is null\n" << std::flush; return tensor_incref(param); }
    if (!lr_t) { std::cerr << "RUNTIME ERROR in param_update: lr is null\n" << std::flush; return tensor_incref(param); }
    param = materialize(param);
    grad_t = materialize(grad_t);
    if (!param->data) { std::cerr << "RUNTIME ERROR in param_update: param->data is null (shape=" << param->d0 << "x" << param->d1 << ")\n" << std::flush; return tensor_incref(param); }
    if (!grad_t->data) { std::cerr << "RUNTIME ERROR in param_update: grad->data is null (shape=" << grad_t->d0 << "x" << grad_t->d1 << ")\n" << std::flush; return tensor_incref(param); }
    if (!lr_t->data) { std::cerr << "RUNTIME ERROR in param_update: lr->data is null\n" << std::flush; return tensor_incref(param); }
    double lr = lr_t->data[0];
    // Parallelize for large parameters (>64K elements)
    // Small params just use scalar loop (OMP overhead not worth it)
    if (param->size == grad_t->size) {
        // Same-size fast path — unit stride both tensors
        if (param->size >= 65536) {
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < param->size; ++i)
                param->data[i] -= lr * grad_t->data[i];
        } else {
            for (int i = 0; i < param->size; ++i)
                param->data[i] -= lr * grad_t->data[i];
        }
    } else {
        // Broadcast path (bias update where grad might be larger)
        for (int i = 0; i < param->size; ++i)
            param->data[i] -= lr * grad_t->data[i % grad_t->size];
    }
    bump_version(param);
    return tensor_incref(param);
}

// Convert tensor to FP32 storage: allocates fdata/fgrad, copies from double with truncation.
// GPU transfer path (HostBuf::fromTensor) will use fdata directly, skipping conversion.
Tensor* dream_to_float32(Tensor* t) {
    if (!t || t->dtype == 1) return tensor_incref(t);
    t = materialize(t);
    if (!t->fdata) t->fdata = new float[t->size]();
    if (!t->fgrad) t->fgrad = new float[t->size]();
    for (int i = 0; i < t->size; ++i) {
        t->fdata[i] = (float)t->data[i];
        t->data[i]  = (double)t->fdata[i]; // truncate double to fp32 precision
    }
    for (int i = 0; i < t->size; ++i) {
        t->fgrad[i] = (float)t->grad[i];
    }
    t->dtype = 1;
    return tensor_incref(t);
}

// Convert tensor back to FP64: free fdata/fgrad, data[] already holds values
Tensor* dream_to_float64(Tensor* t) {
    if (!t || t->dtype == 0) return tensor_incref(t);
    if (t->fdata) { delete[] t->fdata; t->fdata = nullptr; }
    if (t->fgrad) { delete[] t->fgrad; t->fgrad = nullptr; }
    t->dtype = 0;
    return tensor_incref(t);
}

// Scale all gradients by a scalar (in-place, no graph).
// Typically used to divide accumulated gradients: grad_scale(W, 1.0/accum_steps)
Tensor* dream_grad_scale(Tensor* t, Tensor* scale_t) {
    if (!t || !scale_t) return tensor_incref(t);
    double s = scale_t->data[0];
    for (int i = 0; i < t->size; ++i) t->grad[i] *= s;
    if (t->dtype == 1 && t->fgrad)
        for (int i = 0; i < t->size; ++i) t->fgrad[i] *= (float)s;
    return tensor_incref(t);
}

// Combined gradient accumulation update:
// 1. Divide accumulated gradients by accum_steps
// 2. Apply SGD update: param -= lr * (grad / accum_steps)
// 3. Zero the gradients
// Usage pattern in Dream:
//   for (step in 0..total_steps) {
//       backward(loss)
//       if ((step + 1) % 4 == 0) {
//           grad_accum_update(W, 0.01, 4)  // lr=0.01, accumulated over 4 steps
//       }
//   }
Tensor* dream_grad_accum_update(Tensor* param, Tensor* lr_t, Tensor* steps_t) {
    if (!param || !lr_t || !steps_t) return tensor_incref(param);
    double lr = lr_t->data[0];
    double steps = steps_t->data[0];
    if (steps < 1.0) steps = 1.0;
    double scale = lr / steps;
    for (int i = 0; i < param->size; ++i) {
        param->data[i] -= scale * param->grad[i];
        param->grad[i] = 0.0;
    }
    if (param->dtype == 1 && param->fdata) {
        for (int i = 0; i < param->size; ++i)
            param->fdata[i] = (float)param->data[i];
    }
    if (param->dtype == 1 && param->fgrad) {
        for (int i = 0; i < param->size; ++i)
            param->fgrad[i] = 0.f;
    }
    bump_version(param);
    return tensor_incref(param);
}

// Loss scaling for FP16/FP32 mixed precision training.
// Multiplies loss by scale before backward, then divides gradients after.
// Prevents gradient underflow in low-precision training.
Tensor* dream_loss_scale(Tensor* loss, Tensor* scale_t) {
    if (!loss || !scale_t) return tensor_incref(loss);
    double s = scale_t->data[0];
    Tensor* c = alloc_tensor(loss->d0, loss->d1, loss->d2, loss->d3);
    c->is_leaf = false; tensor_set_lhs(c, loss); c->grad_fn = "mul";
    for (int i = 0; i < loss->size; ++i) c->data[i] = loss->data[i] * s;
    // Store scale as rhs for backward to divide grads correctly
    tensor_set_rhs(c, create_scalar(s));
    return c;
}
// Enable FP16 Tensor Core acceleration for matmul on NVIDIA GPUs.
// Input data is converted to FP16, accumulation stays FP32 for precision.
// Call once at program start: enable_fp16()
Tensor* dream_enable_fp16() {
    g_use_fp16 = true;
    std::cerr << "[dream] FP16 Tensor Core mode enabled\n" << std::flush;
    return create_scalar(1.0);
}

// Disable FP16 mode, revert to FP32 cuBLAS
Tensor* dream_disable_fp16() {
    g_use_fp16 = false;
    return create_scalar(1.0);
}

// Enable activation checkpointing: frees intermediate forward data during backward.
// Reduces peak memory ~50% for deep networks. Prevents double backward.
Tensor* dream_enable_checkpointing() {
    g_free_activations = true;
    std::cerr << "[dream] Activation checkpointing enabled (saves memory)\n" << std::flush;
    return create_scalar(1.0);
}

// Disable activation checkpointing
Tensor* dream_disable_checkpointing() {
    g_free_activations = false;
    return create_scalar(1.0);
}

// Residual (skip) connection: out = x + skip
// Functionally identical to tensor_add but marks grad_fn="residual" for clarity.
// The gradient flows equally to both branches (same as add backward).
Tensor* dream_residual_add(Tensor* x, Tensor* skip) {
    if (!x || !skip) return x ? x : skip;
    x = materialize(x); skip = materialize(skip);
    int rd0 = std::max(x->d0, skip->d0);
    int rd1 = std::max(x->d1, skip->d1);
    Tensor* c = alloc_tensor(rd0, rd1, x->d2, x->d3);
    c->is_leaf = false;
    tensor_set_lhs(c, x);
    tensor_set_rhs(c, skip);
    c->grad_fn = "add"; // reuse add backward (grad flows 1:1 to both)
    for (int i = 0; i < c->size; ++i)
        c->data[i] = x->data[i % x->size] + skip->data[i % skip->size];
    return c;
}

// Kaiming (He) initialization: optimal for ReLU networks.
// randn(rows, cols) * sqrt(2.0 / fan_in)
Tensor* dream_kaiming_init(Tensor* rows_t, Tensor* cols_t) {
    int rows = (int)rows_t->data[0], cols = (int)cols_t->data[0];
    Tensor* t = alloc_tensor(rows, cols, 1, 1);
    auto& rng = get_rng();
    double scale = std::sqrt(2.0 / cols); // fan_in = cols for weight matrix
    std::normal_distribution<double> dist(0.0, scale);
    for (int i = 0; i < t->size; ++i) t->data[i] = dist(rng);
    return t;
}

// Xavier (Glorot) initialization: optimal for sigmoid/tanh networks.
// randn(rows, cols) * sqrt(2.0 / (fan_in + fan_out))
Tensor* dream_xavier_init(Tensor* rows_t, Tensor* cols_t) {
    int rows = (int)rows_t->data[0], cols = (int)cols_t->data[0];
    Tensor* t = alloc_tensor(rows, cols, 1, 1);
    auto& rng = get_rng();
    double scale = std::sqrt(2.0 / (rows + cols));
    std::normal_distribution<double> dist(0.0, scale);
    for (int i = 0; i < t->size; ++i) t->data[i] = dist(rng);
    return t;
}

// Leaky ReLU: max(alpha * x, x) with autograd support
Tensor* dream_leaky_relu(Tensor* x, Tensor* alpha_t) {
    if (!x) return tensor_incref(x);
    x = materialize(x);
    double alpha = alpha_t ? alpha_t->data[0] : 0.01;
    Tensor* c = alloc_tensor(x->d0, x->d1, x->d2, x->d3);
    c->is_leaf = false; tensor_set_lhs(c, x); c->grad_fn = "leaky_relu";
    tensor_set_rhs(c, create_scalar(alpha)); // store alpha for backward
    for (int i = 0; i < x->size; ++i)
        c->data[i] = x->data[i] > 0 ? x->data[i] : alpha * x->data[i];
    return c;
}

// GELU activation (approximation): used in Transformers
// gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
Tensor* dream_gelu(Tensor* x) {
    if (!x) return tensor_incref(x);
    x = materialize(x);
    Tensor* c = alloc_tensor(x->d0, x->d1, x->d2, x->d3);
    c->is_leaf = false; tensor_set_lhs(c, x); c->grad_fn = "gelu";
    const double k = std::sqrt(2.0 / M_PI);
    for (int i = 0; i < x->size; ++i) {
        double v = x->data[i];
        double inner = k * (v + 0.044715 * v * v * v);
        c->data[i] = 0.5 * v * (1.0 + std::tanh(inner));
    }
    return c;
}

// ============================================================
// AdamW Optimizer: The industry standard for training modern AI models.
// Unlike Adam, weight decay is decoupled from the gradient update.
// This is what GPT, BERT, Stable Diffusion, and virtually every
// production model uses.
// ============================================================
struct AdamWState {
    double* m;     // first moment
    double* v;     // second moment
    int size;
    int step;
};
static std::map<Tensor*, AdamWState> g_adamw_states;

// In-place AdamW update: no allocation, no graph, modifies param directly.
// Usage: adamw_update(param, lr, beta1, beta2, eps, weight_decay)
Tensor* dream_adamw_update(Tensor* param, Tensor* lr_t, Tensor* b1_t,
                            Tensor* b2_t, Tensor* eps_t, Tensor* wd_t) {
    if (!param || !lr_t) return tensor_incref(param);
    param = materialize(param);
    double lr  = lr_t->data[0];
    double b1  = b1_t ? b1_t->data[0] : 0.9;
    double b2  = b2_t ? b2_t->data[0] : 0.999;
    double eps = eps_t ? eps_t->data[0] : 1e-8;
    double wd  = wd_t ? wd_t->data[0] : 0.01;
    int sz = param->size;

    // Lazy init state
    auto it = g_adamw_states.find(param);
    if (it == g_adamw_states.end()) {
        AdamWState st;
        st.m = new double[sz]();
        st.v = new double[sz]();
        st.size = sz;
        st.step = 0;
        g_adamw_states[param] = st;
        it = g_adamw_states.find(param);
    }
    auto& st = it->second;
    st.step++;

    double bc1 = 1.0 - std::pow(b1, st.step);
    double bc2 = 1.0 - std::pow(b2, st.step);

    for (int i = 0; i < sz; ++i) {
        double g = param->grad[i];
        // Moment updates
        st.m[i] = b1 * st.m[i] + (1.0 - b1) * g;
        st.v[i] = b2 * st.v[i] + (1.0 - b2) * g * g;
        // Bias correction
        double mh = st.m[i] / bc1;
        double vh = st.v[i] / bc2;
        // AdamW: decoupled weight decay applied directly to param
        param->data[i] -= lr * (mh / (std::sqrt(vh) + eps) + wd * param->data[i]);
        param->grad[i] = 0.0;
    }
    bump_version(param);
    return tensor_incref(param);
}

// Reset AdamW state (call when resetting training)
Tensor* dream_adamw_reset(Tensor* param) {
    auto it = g_adamw_states.find(param);
    if (it != g_adamw_states.end()) {
        delete[] it->second.m;
        delete[] it->second.v;
        g_adamw_states.erase(it);
    }
    return tensor_incref(param);
}

// ============================================================
// Learning Rate Schedulers
// All return the current learning rate as a scalar tensor.
// ============================================================

// Warmup + Cosine Annealing (the default for modern training):
// - Linear warmup for first `warmup` steps
// - Cosine decay from `lr` to `min_lr` over remaining steps
Tensor* dream_lr_cosine(Tensor* step_t, Tensor* total_t, Tensor* lr_t,
                         Tensor* min_lr_t, Tensor* warmup_t) {
    double step  = step_t->data[0];
    double total = total_t->data[0];
    double lr    = lr_t->data[0];
    double min_lr = min_lr_t ? min_lr_t->data[0] : 0.0;
    double warmup = warmup_t ? warmup_t->data[0] : 0.0;

    double current;
    if (step < warmup) {
        current = lr * (step + 1.0) / warmup; // linear warmup
    } else {
        double progress = (step - warmup) / std::max(1.0, total - warmup);
        current = min_lr + 0.5 * (lr - min_lr) * (1.0 + std::cos(M_PI * progress));
    }
    return create_scalar(current);
}

// Step decay: lr * gamma^(step / step_size)
Tensor* dream_lr_step(Tensor* step_t, Tensor* lr_t, Tensor* step_size_t, Tensor* gamma_t) {
    double step = step_t->data[0];
    double lr   = lr_t->data[0];
    double ss   = step_size_t->data[0];
    double g    = gamma_t ? gamma_t->data[0] : 0.1;
    return create_scalar(lr * std::pow(g, std::floor(step / ss)));
}

// Exponential decay: lr * gamma^step
Tensor* dream_lr_exp(Tensor* step_t, Tensor* lr_t, Tensor* gamma_t) {
    double step = step_t->data[0];
    double lr   = lr_t->data[0];
    double g    = gamma_t->data[0];
    return create_scalar(lr * std::pow(g, step));
}

// Linear decay to zero
Tensor* dream_lr_linear(Tensor* step_t, Tensor* total_t, Tensor* lr_t) {
    double step = step_t->data[0], total = total_t->data[0], lr = lr_t->data[0];
    return create_scalar(lr * std::max(0.0, 1.0 - step / total));
}

// ============================================================
// Global Gradient Clipping (by max norm)
// Clips the combined L2 norm of all given parameters.
// Usage: global_grad_clip(w1, w2, w3, ..., max_norm)
// Implemented via varargs: last arg is max_norm scalar.
// For simplicity, provide 2/3/4 param versions.
// ============================================================

static double compute_global_norm(Tensor** params, int n) {
    double total = 0;
    for (int i = 0; i < n; ++i)
        if (params[i])
            for (int j = 0; j < params[i]->size; ++j)
                total += params[i]->grad[j] * params[i]->grad[j];
    return std::sqrt(total);
}
static void scale_grads(Tensor** params, int n, double scale) {
    for (int i = 0; i < n; ++i)
        if (params[i])
            for (int j = 0; j < params[i]->size; ++j)
                params[i]->grad[j] *= scale;
}

Tensor* dream_global_clip_2(Tensor* a, Tensor* b, Tensor* max_norm) {
    Tensor* ps[2] = {a, b};
    double gn = compute_global_norm(ps, 2);
    double mn = max_norm->data[0];
    if (gn > mn) scale_grads(ps, 2, mn / gn);
    return create_scalar(gn);
}
Tensor* dream_global_clip_3(Tensor* a, Tensor* b, Tensor* c, Tensor* max_norm) {
    Tensor* ps[3] = {a, b, c};
    double gn = compute_global_norm(ps, 3);
    double mn = max_norm->data[0];
    if (gn > mn) scale_grads(ps, 3, mn / gn);
    return create_scalar(gn);
}
Tensor* dream_global_clip_4(Tensor* a, Tensor* b, Tensor* c, Tensor* d, Tensor* max_norm) {
    Tensor* ps[4] = {a, b, c, d};
    double gn = compute_global_norm(ps, 4);
    double mn = max_norm->data[0];
    if (gn > mn) scale_grads(ps, 4, mn / gn);
    return create_scalar(gn);
}

// ============================================================
// Modern Activations with autograd
// ============================================================

// SiLU (Swish): x * sigmoid(x) - used in LLaMA, Stable Diffusion
Tensor* dream_silu(Tensor* x) {
    if (!x) return tensor_incref(x);
    x = materialize(x);
    Tensor* c = alloc_tensor(x->d0, x->d1, x->d2, x->d3);
    c->is_leaf = false; tensor_set_lhs(c, x); c->grad_fn = "silu";
    for (int i = 0; i < x->size; ++i) {
        double sig = 1.0 / (1.0 + std::exp(-x->data[i]));
        c->data[i] = x->data[i] * sig;
    }
    return c;
}

// Mish: x * tanh(softplus(x)) - used in YOLOv4+
Tensor* dream_mish(Tensor* x) {
    if (!x) return tensor_incref(x);
    x = materialize(x);
    Tensor* c = alloc_tensor(x->d0, x->d1, x->d2, x->d3);
    c->is_leaf = false; tensor_set_lhs(c, x); c->grad_fn = "mish";
    for (int i = 0; i < x->size; ++i) {
        double sp = std::log(1.0 + std::exp(x->data[i])); // softplus
        c->data[i] = x->data[i] * std::tanh(sp);
    }
    return c;
}

// ELU: x if x>0, alpha*(exp(x)-1) otherwise
Tensor* dream_elu(Tensor* x, Tensor* alpha_t) {
    if (!x) return tensor_incref(x);
    x = materialize(x);
    double alpha = alpha_t ? alpha_t->data[0] : 1.0;
    Tensor* c = alloc_tensor(x->d0, x->d1, x->d2, x->d3);
    c->is_leaf = false; tensor_set_lhs(c, x); c->grad_fn = "elu";
    tensor_set_rhs(c, create_scalar(alpha));
    for (int i = 0; i < x->size; ++i)
        c->data[i] = x->data[i] > 0 ? x->data[i] : alpha * (std::exp(x->data[i]) - 1.0);
    return c;
}

// ============================================================
// Tensor Manipulation Operations
// ============================================================

// Reshape to arbitrary dimensions: reshape_nd(t, d0, d1)
// Supports -1 for one dimension (auto-compute)
Tensor* dream_reshape_nd(Tensor* t, Tensor* d0_t, Tensor* d1_t) {
    if (!t) return tensor_incref(t);
    t = materialize(t);
    int d0 = (int)d0_t->data[0], d1 = (int)d1_t->data[0];
    if (d0 == -1 && d1 > 0) d0 = t->size / d1;
    if (d1 == -1 && d0 > 0) d1 = t->size / d0;
    if (d0 * d1 != t->size) {
        std::cerr << "error: reshape size mismatch (" << t->size << " vs " << d0*d1 << ")\n" << std::flush;
        return tensor_incref(t);
    }
    Tensor* c = alloc_tensor(d0, d1, 1, 1);
    c->is_leaf = false; tensor_set_lhs(c, t); c->grad_fn = "flatten";
    std::copy(t->data, t->data + t->size, c->data);
    return c;
}

// Squeeze: remove dimensions of size 1
// squeeze(t) -> if d0==1: (d1, d2, ...), if d1==1: (d0, d2, ...)
Tensor* dream_squeeze(Tensor* t) {
    if (!t) return tensor_incref(t);
    t = materialize(t);
    int dims[4] = {t->d0, t->d1, t->d2, t->d3};
    int nd[4], cnt = 0;
    for (int i = 0; i < 4; ++i) if (dims[i] > 1) nd[cnt++] = dims[i];
    if (cnt == 0) { nd[0] = 1; cnt = 1; }
    Tensor* c = alloc_tensor(nd[0], cnt>=2?nd[1]:1, cnt>=3?nd[2]:1, cnt>=4?nd[3]:1);
    c->is_leaf = false; tensor_set_lhs(c, t); c->grad_fn = "flatten";
    std::copy(t->data, t->data + t->size, c->data);
    return c;
}

// Unsqueeze: add dimension of size 1 at given axis
Tensor* dream_unsqueeze(Tensor* t, Tensor* axis_t) {
    if (!t) return tensor_incref(t);
    t = materialize(t);
    int axis = (int)axis_t->data[0];
    int d[4] = {t->d0, t->d1, t->d2, t->d3};
    // Insert 1 at axis position, shift others right
    int nd[4] = {1, 1, 1, 1};
    int j = 0;
    for (int i = 0; i < 4; ++i) {
        if (i == axis) nd[i] = 1;
        else if (j < 4) nd[i] = d[j++];
    }
    Tensor* c = alloc_tensor(nd[0], nd[1], nd[2], nd[3]);
    c->is_leaf = false; tensor_set_lhs(c, t); c->grad_fn = "flatten";
    std::copy(t->data, t->data + t->size, c->data);
    return c;
}

// Stack: combine N tensors along a new first axis
// stack([t1, t2, t3]) -> (3, t1.size) if inputs are 1D
Tensor* dream_stack(Tensor* list) {
    if (!list || list->grad_fn != "tuple") return tensor_incref(list);
    int n = list->size;
    if (n == 0) return alloc_tensor(1, 1, 1, 1);
    // Extract first tensor to get shape
    uintptr_t p0 = 0; std::memcpy(&p0, &list->data[0], sizeof(p0));
    Tensor* first = reinterpret_cast<Tensor*>(p0);
    if (!first) return alloc_tensor(1, 1, 1, 1);
    first = materialize(first);
    int elem_size = first->size;
    Tensor* out = alloc_tensor(n, elem_size, 1, 1);
    for (int i = 0; i < n; ++i) {
        uintptr_t ptr = 0; std::memcpy(&ptr, &list->data[i], sizeof(ptr));
        Tensor* ti = reinterpret_cast<Tensor*>(ptr);
        if (ti) {
            ti = materialize(ti);
            int copy_n = std::min(ti->size, elem_size);
            std::copy(ti->data, ti->data + copy_n, out->data + i * elem_size);
        }
    }
    return out;
}

// Cat along axis 0: vertically stack two matrices
Tensor* dream_cat_0(Tensor* a, Tensor* b) {
    if (!a || !b) return a ? a : b;
    a = materialize(a); b = materialize(b);
    if (a->d1 != b->d1) {
        std::cerr << "error: cat axis=0 column mismatch (" << a->d1 << " vs " << b->d1 << ")\n" << std::flush;
        return tensor_incref(a);
    }
    Tensor* c = alloc_tensor(a->d0 + b->d0, a->d1, 1, 1);
    std::copy(a->data, a->data + a->size, c->data);
    std::copy(b->data, b->data + b->size, c->data + a->size);
    return c;
}

// Cat along axis 1: horizontally concatenate (already have dream_concat, alias it)
// dream_concat is defined in math_ops.inc

// Repeat/tile: repeat tensor contents along each dimension
Tensor* dream_repeat(Tensor* t, Tensor* r0_t, Tensor* r1_t) {
    if (!t) return tensor_incref(t);
    t = materialize(t);
    int r0 = (int)r0_t->data[0], r1 = (int)r1_t->data[0];
    if (r0 < 1) r0 = 1; if (r1 < 1) r1 = 1;
    Tensor* out = alloc_tensor(t->d0 * r0, t->d1 * r1, 1, 1);
    for (int i = 0; i < t->d0 * r0; ++i)
        for (int j = 0; j < t->d1 * r1; ++j)
            out->data[i * (t->d1 * r1) + j] = t->data[(i % t->d0) * t->d1 + (j % t->d1)];
    return out;
}

// Gather: index select along axis 0
// gather(data, indices) -> data[indices[i]] for each i
Tensor* dream_gather(Tensor* data, Tensor* indices) {
    if (!data || !indices) return tensor_incref(data);
    data = materialize(data); indices = materialize(indices);
    int n = indices->size, cols = data->d1;
    Tensor* out = alloc_tensor(n, cols, 1, 1);
    for (int i = 0; i < n; ++i) {
        int idx = (int)indices->data[i];
        if (idx >= 0 && idx < data->d0)
            std::copy(data->data + idx*cols, data->data + (idx+1)*cols, out->data + i*cols);
    }
    return out;
}

// Top-k: returns (values, indices) as tuple
Tensor* dream_topk(Tensor* t, Tensor* k_t) {
    if (!t) return create_scalar(0.0);
    t = materialize(t);
    int k = (int)k_t->data[0];
    if (k > t->size) k = t->size;
    // Sort indices by value descending
    std::vector<int> idx(t->size);
    for (int i = 0; i < t->size; ++i) idx[i] = i;
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
        [&](int a, int b) { return t->data[a] > t->data[b]; });
    Tensor* vals = alloc_tensor(1, k, 1, 1);
    Tensor* inds = alloc_tensor(1, k, 1, 1);
    for (int i = 0; i < k; ++i) {
        vals->data[i] = t->data[idx[i]];
        inds->data[i] = (double)idx[i];
    }
    Tensor** arr = new Tensor*[2]{vals, inds};
    Tensor* result = dream_pack(arr, 2);
    delete[] arr;
    return result;
}

// Clamp moved to math_ops.inc with backward support (grad-aware version).

// Masked fill: set elements where mask > 0 to value
Tensor* dream_masked_fill(Tensor* t, Tensor* mask, Tensor* val_t) {
    if (!t || !mask) return tensor_incref(t);
    t = materialize(t); mask = materialize(mask);
    double val = val_t->data[0];
    Tensor* c = alloc_tensor(t->d0, t->d1, t->d2, t->d3);
    std::copy(t->data, t->data + t->size, c->data);
    for (int i = 0; i < c->size; ++i)
        if (mask->data[i % mask->size] > 0.0) c->data[i] = val;
    return c;
}

// ============================================================
// QUANTIZATION TOOLKIT
// Dream's built-in quantization — no external tools needed.
// Supports INT8 symmetric/asymmetric, calibration, and dequant.
// ============================================================

// Compute quantization parameters: scale and zero_point
// symmetric: q = round(x / scale), scale = max(|x|) / 127
// Returns (scale, zero_point) as tuple
Tensor* dream_quantize_calibrate(Tensor* t) {
    if (!t) return create_scalar(0.0);
    t = materialize(t);
    double absmax = 0;
    for (int i = 0; i < t->size; ++i) {
        double a = std::fabs(t->data[i]);
        if (a > absmax) absmax = a;
    }
    double scale = absmax / 127.0;
    if (scale < 1e-10) scale = 1e-10;
    Tensor** arr = new Tensor*[2]{create_scalar(scale), create_scalar(0.0)};
    Tensor* result = dream_pack(arr, 2);
    delete[] arr;
    return result;
}

// Quantize tensor to INT8 range [-128, 127] (stored as double for simplicity)
// In a real deployment, these would be actual int8 buffers.
Tensor* dream_quantize(Tensor* t, Tensor* scale_t) {
    if (!t || !scale_t) return tensor_incref(t);
    t = materialize(t);
    double scale = scale_t->data[0];
    if (scale < 1e-10) scale = 1e-10;
    Tensor* q = alloc_tensor(t->d0, t->d1, t->d2, t->d3);
    for (int i = 0; i < t->size; ++i)
        q->data[i] = std::round(std::max(-128.0, std::min(127.0, t->data[i] / scale)));
    q->dtype = 2; // mark as quantized
    return q;
}

// Dequantize: recover approximate FP values from INT8
Tensor* dream_dequantize(Tensor* q, Tensor* scale_t) {
    if (!q || !scale_t) return tensor_incref(q);
    q = materialize(q);
    double scale = scale_t->data[0];
    Tensor* t = alloc_tensor(q->d0, q->d1, q->d2, q->d3);
    for (int i = 0; i < q->size; ++i)
        t->data[i] = q->data[i] * scale;
    return t;
}

// Quantization-aware training: simulate quantization noise in forward pass
// x_out = dequant(quant(x, scale), scale) — adds rounding noise
Tensor* dream_fake_quantize(Tensor* t, Tensor* scale_t) {
    if (!t || !scale_t) return tensor_incref(t);
    t = materialize(t);
    double scale = scale_t->data[0];
    if (scale < 1e-10) scale = 1e-10;
    Tensor* c = alloc_tensor(t->d0, t->d1, t->d2, t->d3);
    c->is_leaf = false; tensor_set_lhs(c, t); c->grad_fn = "fake_quant";
    for (int i = 0; i < t->size; ++i) {
        double q = std::round(std::max(-128.0, std::min(127.0, t->data[i] / scale)));
        c->data[i] = q * scale;
    }
    return c;
}

// Measure quantization error (MSE between original and quantized)
Tensor* dream_quant_error(Tensor* original, Tensor* quantized, Tensor* scale_t) {
    if (!original || !quantized || !scale_t) return create_scalar(0.0);
    original = materialize(original); quantized = materialize(quantized);
    double scale = scale_t->data[0];
    double mse = 0;
    int n = std::min(original->size, quantized->size);
    for (int i = 0; i < n; ++i) {
        double diff = original->data[i] - quantized->data[i] * scale;
        mse += diff * diff;
    }
    return create_scalar(mse / n);
}

// ============================================================
// PRUNING TOOLKIT
// Magnitude-based and structured pruning for model compression.
// ============================================================

// Magnitude pruning: zero out the smallest `ratio` fraction of weights
// Returns pruned tensor + sparsity info
Tensor* dream_prune_magnitude(Tensor* t, Tensor* ratio_t) {
    if (!t || !ratio_t) return tensor_incref(t);
    t = materialize(t);
    double ratio = ratio_t->data[0]; // fraction to prune (0.0 to 1.0)
    if (ratio <= 0) return tensor_incref(t);
    if (ratio >= 1.0) ratio = 0.99;

    // Collect absolute values and find threshold
    std::vector<double> absvals(t->size);
    for (int i = 0; i < t->size; ++i) absvals[i] = std::fabs(t->data[i]);
    std::sort(absvals.begin(), absvals.end());
    int cutoff = (int)(ratio * t->size);
    double threshold = absvals[std::min(cutoff, t->size - 1)];

    Tensor* pruned = alloc_tensor(t->d0, t->d1, t->d2, t->d3);
    int zeroed = 0;
    for (int i = 0; i < t->size; ++i) {
        if (std::fabs(t->data[i]) <= threshold) {
            pruned->data[i] = 0.0;
            zeroed++;
        } else {
            pruned->data[i] = t->data[i];
        }
    }
    double actual_sparsity = (double)zeroed / t->size;
    std::cerr << "[prune] " << t->d0 << "x" << t->d1
              << " sparsity=" << actual_sparsity * 100 << "% ("
              << zeroed << "/" << t->size << " zeros)\n" << std::flush;
    return pruned;
}

// Structured pruning: remove entire rows (neurons/filters) with smallest L2 norm
Tensor* dream_prune_structured(Tensor* t, Tensor* ratio_t) {
    if (!t || !ratio_t) return tensor_incref(t);
    t = materialize(t);
    double ratio = ratio_t->data[0];
    int rows = t->d0, cols = t->d1;
    int to_prune = (int)(ratio * rows);
    if (to_prune <= 0) return tensor_incref(t);

    // Compute L2 norm per row
    std::vector<std::pair<double, int>> norms(rows);
    for (int r = 0; r < rows; ++r) {
        double s = 0;
        for (int c = 0; c < cols; ++c) s += t->data[r * cols + c] * t->data[r * cols + c];
        norms[r] = {std::sqrt(s), r};
    }
    std::sort(norms.begin(), norms.end());

    // Zero out smallest-norm rows
    Tensor* pruned = alloc_tensor(rows, cols, t->d2, t->d3);
    std::copy(t->data, t->data + t->size, pruned->data);
    for (int i = 0; i < to_prune; ++i) {
        int r = norms[i].second;
        for (int c = 0; c < cols; ++c) pruned->data[r * cols + c] = 0.0;
    }
    std::cerr << "[prune_struct] " << rows << "x" << cols
              << " pruned " << to_prune << "/" << rows << " rows\n" << std::flush;
    return pruned;
}

// Create binary mask from pruned tensor (1 where non-zero, 0 where zero)
Tensor* dream_prune_mask(Tensor* t) {
    if (!t) return tensor_incref(t);
    t = materialize(t);
    Tensor* mask = alloc_tensor(t->d0, t->d1, t->d2, t->d3);
    for (int i = 0; i < t->size; ++i)
        mask->data[i] = (t->data[i] != 0.0) ? 1.0 : 0.0;
    return mask;
}

// Apply mask to enforce sparsity pattern (used after each SGD step)
Tensor* dream_apply_mask(Tensor* t, Tensor* mask) {
    if (!t || !mask) return tensor_incref(t);
    t = materialize(t); mask = materialize(mask);
    for (int i = 0; i < t->size && i < mask->size; ++i)
        t->data[i] *= mask->data[i];
    bump_version(t);
    return tensor_incref(t);
}

// Report sparsity statistics
Tensor* dream_sparsity(Tensor* t) {
    if (!t) return create_scalar(0.0);
    t = materialize(t);
    int zeros = 0;
    for (int i = 0; i < t->size; ++i)
        if (t->data[i] == 0.0) zeros++;
    double sp = (double)zeros / t->size;
    std::cerr << "[sparsity] " << t->d0 << "x" << t->d1
              << ": " << sp * 100 << "% sparse (" << zeros << "/" << t->size << ")\n" << std::flush;
    return create_scalar(sp);
}

// ============================================================
// Micro-benchmark utility — time any Dream callable.
// Gives users concrete numbers to see where time is spent.
// ============================================================

// Start a named timer. Returns a token for stop_timer.
Tensor* dream_tic(Tensor* name_t) {
    std::string name = "";
    if (name_t && name_t->grad_fn == "string")
        for (int i = 0; i < name_t->size; ++i) name += (char)(int)name_t->data[i];
    auto now = std::chrono::steady_clock::now();
    double us = std::chrono::duration<double, std::micro>(now.time_since_epoch()).count();
    Tensor* t = create_scalar(us);
    return t;
}

// Stop timer. Arg 1 is the tic() result, arg 2 is the label.
Tensor* dream_toc(Tensor* tic_t, Tensor* name_t) {
    if (!tic_t) return create_scalar(0.0);
    auto now = std::chrono::steady_clock::now();
    double now_us = std::chrono::duration<double, std::micro>(now.time_since_epoch()).count();
    double start_us = tic_t->data[0];
    double elapsed_us = now_us - start_us;
    std::string name = "timer";
    if (name_t && name_t->grad_fn == "string")
        for (int i = 0; i < name_t->size; ++i) name += (char)(int)name_t->data[i];

    if (elapsed_us < 1000.0)
        std::cerr << "[" << name << "] " << elapsed_us << " us\n" << std::flush;
    else if (elapsed_us < 1000000.0)
        std::cerr << "[" << name << "] " << (elapsed_us / 1000.0) << " ms\n" << std::flush;
    else
        std::cerr << "[" << name << "] " << (elapsed_us / 1000000.0) << " s\n" << std::flush;
    return create_scalar(elapsed_us);
}

// Quick op count reporter
Tensor* dream_op_stats() {
    std::cerr << "[stats] arena=" << g_arena.size()
              << " allocs_since_gc=" << g_alloc_count << "\n" << std::flush;
    return create_scalar((double)g_arena.size());
}

// Aliases without dream_ prefix so they work even if the compiler's
// builtinRemap hasn't been rebuilt with the new mappings.
Tensor* tic(Tensor* name_t) { return dream_tic(name_t); }
Tensor* toc(Tensor* tic_t, Tensor* name_t) { return dream_toc(tic_t, name_t); }
Tensor* op_stats() { return dream_op_stats(); }

// Precision annotation: set the tensor storage-precision hint (0=f64, 1=f32).
// For f32, actually materialize the float32 mirror (fdata) from data so that
// f32-aware kernels (matmul, elementwise) can compute in single precision.
// data (double) is kept as a mirror for print and non-f32-aware ops.
// Returns the same tensor (+1, ownership contract).
Tensor* dream_set_precision(Tensor* t, int code) {
    if (!t) return nullptr;
    if (code == 1) {
        if (t->dtype != 1 || !t->fdata) {
            if (!t->fdata) t->fdata = new float[t->size]();
            if (!t->fgrad) t->fgrad = new float[t->size]();
            for (int i = 0; i < t->size; ++i) t->fdata[i] = (float)t->data[i];
            t->dtype = 1;
        }
    } else {
        // Back to f64: flush the f32 mirror into data if it was the live copy.
        if (t->dtype == 1 && t->fdata)
            for (int i = 0; i < t->size; ++i) t->data[i] = (double)t->fdata[i];
        t->dtype = 0;
    }
    return tensor_incref(t);
}

// =====================================================================
//  augment(X, Y, when_below, up_to) -- SMOTE-style data augmentation.
//
//  DESIGN (why this does not emit garbage):
//   * INTERPOLATION ONLY, never extrapolation. A synthetic sample is a
//     convex combination of TWO real samples: s = a + t*(b - a), t in
//     [0,1]. It therefore always lands inside the convex hull of the real
//     data -- it cannot wander into unseen/absurd regions.
//   * Pairs are drawn from the SAME class (classification) so the label is
//     unambiguous; the label is carried through unchanged. For real-valued
//     labels the label is interpolated with the same t (relationship kept).
//   * A tiny jitter (<= 5% of each column's real std) breaks exact
//     collinearity without moving points off the data manifold.
//   * NEEDS >= 2 real rows, and >= 2 rows per class to pair within a class;
//     a single row can never seed augmentation.
//
//  Returns a NEW X with the real rows first, then the synthetic rows.
//  Y is augmented in lockstep via a parallel call keyed off the same seed.
//  Guardrails (too-small threshold, up_to accounting) are enforced by the
//  compiler-side wrapper; this runtime function trusts validated inputs.
// =====================================================================
static long long g_last_aug_synth = 0;   // rows synthesized in the last call
static long long g_last_aug_total = 0;   // total rows after the last call

// Shared RNG-paired augmentation so X and Y stay aligned. We synthesize the
// SAME (i,j,t) triples for both by re-seeding a local generator identically.
static Tensor* augment_impl(Tensor* X, Tensor* Y, int when_below, int up_to,
                            unsigned seed, bool produce_Y) {
    X = materialize(X);
    Tensor* Ym = Y ? materialize(Y) : nullptr;
    int N = X->d0 > 0 ? X->d0 : 1;
    int F = X->size / N;
    int L = (Ym ? (Ym->size / N) : 0);

    // Guard: enough data? If already at/above threshold, return as-is (+1).
    if (N >= when_below || N < 2) {
        g_last_aug_synth = 0; g_last_aug_total = N;
        return tensor_incref(produce_Y ? (Ym ? Ym : X) : X);
    }
    int target = up_to;
    if (target <= N) { g_last_aug_synth = 0; g_last_aug_total = N; 
        return tensor_incref(produce_Y ? (Ym ? Ym : X) : X); }
    int n_synth = target - N;

    // Per-column std of real X, for scaling the jitter.
    std::vector<double> col_std(F, 0.0);
    for (int c = 0; c < F; ++c) {
        double m = 0; for (int r = 0; r < N; ++r) m += X->data[(size_t)r*F + c]; m /= N;
        double v = 0; for (int r = 0; r < N; ++r){ double d = X->data[(size_t)r*F+c]-m; v += d*d; }
        col_std[c] = std::sqrt(v / N);
    }

    // Group real row indices by class label (rounded) if a label exists.
    std::map<long long, std::vector<int>> byClass;
    for (int r = 0; r < N; ++r) {
        long long key = 0;
        if (Ym) { // use first label column, rounded, as the class key
            key = (long long)std::llround(Ym->data[(size_t)r*L + 0]);
        }
        byClass[key].push_back(r);
    }

    Tensor* outX = alloc_tensor(target, F, 1, 1);
    Tensor* outY = produce_Y && Ym ? alloc_tensor(target, L, 1, 1) : nullptr;
    // Copy real rows first.
    std::copy(X->data, X->data + (size_t)N*F, outX->data);
    if (outY) std::copy(Ym->data, Ym->data + (size_t)N*L, outY->data);

    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> ut(0.0, 1.0);
    std::normal_distribution<double> jit(0.0, 1.0);

    // Class keys as a vector for random pick.
    std::vector<long long> keys;
    for (auto& kv : byClass) if (kv.second.size() >= 2) keys.push_back(kv.first);
    // Fallback: if no class has >=2 members, pair across the whole set.
    bool crossClass = keys.empty();

    for (int s = 0; s < n_synth; ++s) {
        int i, j; long long key = 0;
        if (crossClass) {
            i = (int)(ut(rng) * N); if (i >= N) i = N-1;
            j = (int)(ut(rng) * N); if (j >= N) j = N-1;
        } else {
            key = keys[(size_t)(ut(rng) * keys.size()) % keys.size()];
            auto& idx = byClass[key];
            i = idx[(size_t)(ut(rng) * idx.size()) % idx.size()];
            j = idx[(size_t)(ut(rng) * idx.size()) % idx.size()];
        }
        double t = ut(rng);
        double* dst = outX->data + (size_t)(N + s) * F;
        for (int c = 0; c < F; ++c) {
            double a = X->data[(size_t)i*F + c], b = X->data[(size_t)j*F + c];
            double val = a + t * (b - a);
            val += jit(rng) * col_std[c] * 0.05;   // <=5% std jitter
            dst[c] = val;
        }
        if (outY) {
            double* dY = outY->data + (size_t)(N + s) * L;
            if (crossClass) {
                // interpolate label too (regression-safe)
                for (int c = 0; c < L; ++c) {
                    double a = Ym->data[(size_t)i*L+c], b = Ym->data[(size_t)j*L+c];
                    dY[c] = a + t * (b - a);
                }
            } else {
                for (int c = 0; c < L; ++c) dY[c] = (double)key;   // same class
            }
        }
    }
    g_last_aug_synth = n_synth;
    g_last_aug_total = target;
    if (produce_Y) { return outY ? outY : tensor_incref(X); }
    return outX;
}

// Dream entry points. augment_x returns the augmented features; augment_y
// returns labels augmented with the SAME seed so rows line up. Both take
// (X, Y, when_below, up_to). Seed is fixed for determinism.
Tensor* dream_augment_x(Tensor* X, Tensor* Y, Tensor* below_t, Tensor* upto_t) {
    int wb = below_t ? (int)below_t->data[0] : 0;
    int ut = upto_t ? (int)upto_t->data[0] : 0;
    return augment_impl(X, Y, wb, ut, 0xA06E57u, false);
}
Tensor* dream_augment_y(Tensor* X, Tensor* Y, Tensor* below_t, Tensor* upto_t) {
    int wb = below_t ? (int)below_t->data[0] : 0;
    int ut = upto_t ? (int)upto_t->data[0] : 0;
    return augment_impl(X, Y, wb, ut, 0xA06E57u, true);
}
// Report how many rows the last augment produced (for verification/logging).
Tensor* dream_augment_count(Tensor* dummy) {
    (void)dummy;
    return create_scalar((double)g_last_aug_synth);
}
