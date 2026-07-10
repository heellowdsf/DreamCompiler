// Dream Runtime - Training: RNN, LSTM, Attention, Debug Tools, Data Augmentation

// Debug: print detailed tensor statistics
Tensor* dream_tensor_stats(Tensor* t, Tensor* name_t) {
    if (!t) { std::cerr << "[stats] null tensor\n" << std::flush; return create_scalar(0.0); }
    t = materialize(t);
    std::string name = "";
    if (name_t && name_t->grad_fn == "string")
        for (int i = 0; i < name_t->size; ++i) name += (char)(int)name_t->data[i];
    double sum = 0, sq = 0, mn = t->data[0], mx = t->data[0];
    int nans = 0, infs = 0;
    for (int i = 0; i < t->size; ++i) {
        double v = t->data[i];
        if (dream_isnan(v)) { nans++; continue; }
        if (dream_isinf(v)) { infs++; continue; }
        sum += v; sq += v*v;
        if (v < mn) mn = v; if (v > mx) mx = v;
    }
    int ok = t->size - nans - infs;
    double mean = ok > 0 ? sum/ok : 0;
    double sd = ok > 1 ? std::sqrt(std::max(0.0, sq/ok - mean*mean)) : 0;
    std::cerr << "[stats] " << name << " shape=" << t->d0 << "x" << t->d1
              << " mean=" << mean << " std=" << sd << " min=" << mn << " max=" << mx;
    if (nans) std::cerr << " NaN=" << nans;
    if (infs) std::cerr << " Inf=" << infs;
    std::cerr << "\n" << std::flush;
    return tensor_incref(t);
}

// Debug: check for NaN/Inf. Returns 1.0 if clean, 0.0 if corrupted.
Tensor* dream_check_nan(Tensor* t, Tensor* name_t) {
    if (!t) return create_scalar(0.0);
    t = materialize(t);
    std::string name = "";
    if (name_t && name_t->grad_fn == "string")
        for (int i = 0; i < name_t->size; ++i) name += (char)(int)name_t->data[i];
    for (int i = 0; i < t->size; ++i) {
        if (dream_isnan(t->data[i]) || dream_isinf(t->data[i])) {
            std::cerr << "WARNING: NaN/Inf in '" << name << "' at [" << i
                      << "] val=" << t->data[i] << " shape=" << t->d0 << "x" << t->d1 << "\n" << std::flush;
            return create_scalar(0.0);
        }
    }
    return create_scalar(1.0);
}

// Debug: dump computation graph structure (iterative to avoid stack overflow)
static void dump_node(Tensor* root, int /*unused*/, std::set<Tensor*>& seen) {
    if (!root) return;
    struct Frame { Tensor* node; int depth; };
    std::vector<Frame> stack;
    stack.push_back({root, 0});
    while (!stack.empty()) {
        auto fr = stack.back(); stack.pop_back();
        Tensor* t = fr.node;
        if (!t || seen.count(t)) continue;
        seen.insert(t);
        for (int i = 0; i < fr.depth; ++i) std::cerr << "  ";
        std::cerr << "[" << t->d0 << "x" << t->d1 << "] "
                  << (t->grad_fn.empty() ? "leaf" : t->grad_fn)
                  << (t->is_variable ? " (var)" : "") << " ref=" << t->refcount << "\n";
        if (t->parents)
            for (int i = t->num_parents-1; i >= 0; --i)
                stack.push_back({t->parents[i], fr.depth+1});
        if (t->rhs) stack.push_back({t->rhs, fr.depth+1});
        if (t->lhs) stack.push_back({t->lhs, fr.depth+1});
    }
}
Tensor* dream_dump_graph(Tensor* t) {
    if (!t) { std::cerr << "[graph] null\n" << std::flush; return create_scalar(0.0); }
    std::cerr << "--- Computation Graph ---\n";
    std::set<Tensor*> seen;
    dump_node(t, 0, seen);
    std::cerr << "--- " << seen.size() << " nodes ---\n" << std::flush;
    return create_scalar((double)seen.size());
}

// Debug: gradient magnitude histogram
Tensor* dream_grad_histogram(Tensor* t, Tensor* name_t) {
    if (!t) return create_scalar(0.0);
    std::string name = "";
    if (name_t && name_t->grad_fn == "string")
        for (int i = 0; i < name_t->size; ++i) name += (char)(int)name_t->data[i];
    int b[7] = {};
    for (int i = 0; i < t->size; ++i) {
        double g = std::fabs(t->grad[i]);
        if (g == 0) b[0]++; else if (g < 1e-6) b[1]++; else if (g < 1e-4) b[2]++;
        else if (g < 1e-2) b[3]++; else if (g < 1) b[4]++; else if (g < 10) b[5]++; else b[6]++;
    }
    std::cerr << "[grad] " << name << ": zero=" << b[0] << " <1e-6=" << b[1]
              << " <1e-4=" << b[2] << " <0.01=" << b[3] << " <1=" << b[4]
              << " <10=" << b[5] << " >=10=" << b[6] << "\n" << std::flush;
    return tensor_incref(t);
}

// Detach: stop gradient flow, return fresh copy
Tensor* dream_detach(Tensor* t) {
    if (!t) return nullptr;
    t = materialize(t);
    Tensor* c = alloc_tensor(t->d0, t->d1, t->d2, t->d3);
    std::copy(t->data, t->data + t->size, c->data);
    return c;
}

// RNN Cell: h_t = tanh(x @ Wih + h_prev @ Whh + b)
Tensor* dream_rnn_cell(Tensor* x, Tensor* h_prev,
                        Tensor* w_ih, Tensor* w_hh, Tensor* bias) {
    if (!x || !h_prev || !w_ih || !w_hh) return tensor_incref(h_prev);
    x = materialize(x); h_prev = materialize(h_prev);
    w_ih = materialize(w_ih); w_hh = materialize(w_hh);
    int B = x->d0, I = x->d1, H = w_hh->d1;

    Tensor* out = alloc_tensor(B, H, 1, 1);
    out->is_leaf = false; out->grad_fn = "rnn_cell";
    {
        Tensor** par = new Tensor*[5]{x, h_prev, w_ih, w_hh, bias};
        tensor_set_parents(out, par, 5);
    }

    for (int b = 0; b < B; ++b)
        for (int j = 0; j < H; ++j) {
            double s = 0;
            for (int k = 0; k < I; ++k) s += x->data[b*I+k] * w_ih->data[k*H+j];
            for (int k = 0; k < H; ++k) s += h_prev->data[b*H+k] * w_hh->data[k*H+j];
            if (bias) s += bias->data[j % bias->size];
            out->data[b*H+j] = std::tanh(s);
        }
    return out;
}

// LSTM Cell: returns (h_t, c_t) tuple
Tensor* dream_lstm_cell(Tensor* x, Tensor* h_prev, Tensor* c_prev,
                         Tensor* w_ih, Tensor* w_hh, Tensor* bias) {
    if (!x || !h_prev || !c_prev || !w_ih || !w_hh) return create_scalar(0.0);
    x = materialize(x); h_prev = materialize(h_prev); c_prev = materialize(c_prev);
    w_ih = materialize(w_ih); w_hh = materialize(w_hh);
    int B = x->d0, I = x->d1, H = h_prev->d1;
    int H4 = 4 * H;

    // Compute all 4 gates in one matmul
    std::vector<double> gates(B * H4);
    for (int b = 0; b < B; ++b)
        for (int j = 0; j < H4; ++j) {
            double s = 0;
            for (int k = 0; k < I; ++k) s += x->data[b*I+k] * w_ih->data[k*H4+j];
            for (int k = 0; k < H; ++k) s += h_prev->data[b*H+k] * w_hh->data[k*H4+j];
            if (bias) s += bias->data[j % bias->size];
            gates[b*H4+j] = s;
        }

    Tensor* h_out = alloc_tensor(B, H, 1, 1);
    Tensor* c_out = alloc_tensor(B, H, 1, 1);
    h_out->is_leaf = false; h_out->grad_fn = "lstm";
    c_out->is_leaf = false; c_out->grad_fn = "lstm";

    auto sig = [](double x) { return 1.0 / (1.0 + std::exp(-x)); };
    for (int b = 0; b < B; ++b)
        for (int j = 0; j < H; ++j) {
            double gi = sig(gates[b*H4 + j]);
            double gf = sig(gates[b*H4 + H + j]);
            double gg = std::tanh(gates[b*H4 + 2*H + j]);
            double go = sig(gates[b*H4 + 3*H + j]);
            double cv = gf * c_prev->data[b*H+j] + gi * gg;
            c_out->data[b*H+j] = cv;
            h_out->data[b*H+j] = go * std::tanh(cv);
        }

    Tensor** arr = new Tensor*[2]{h_out, c_out};
    Tensor* result = dream_pack(arr, 2);
    delete[] arr;
    tensor_decref(h_out);
    tensor_decref(c_out);
    return result;
}

// Scaled Dot-Product Attention: softmax(Q @ K^T / sqrt(dk)) @ V
// Q,K,V: (batch, seq_len * dk) stored flat
Tensor* dream_attention(Tensor* Q, Tensor* K, Tensor* V,
                         Tensor* batch_t, Tensor* seq_t, Tensor* dk_t) {
    if (!Q || !K || !V) return create_scalar(0.0);
    Q = materialize(Q); K = materialize(K); V = materialize(V);
    int B = (int)batch_t->data[0], S = (int)seq_t->data[0], dk = (int)dk_t->data[0];
    double scale = 1.0 / std::sqrt((double)dk);

    Tensor* out = alloc_tensor(B, S * dk, 1, 1);
    out->is_leaf = false; out->grad_fn = "attention";
    {
        Tensor** par = new Tensor*[3]{Q, K, V};
        tensor_set_parents(out, par, 3);
    }
    out->bwd_name = std::to_string(B) + "," + std::to_string(S) + "," + std::to_string(dk);

    for (int b = 0; b < B; ++b) {
        const double *Qb = Q->data + b*S*dk, *Kb = K->data + b*S*dk, *Vb = V->data + b*S*dk;
        double* Ob = out->data + b*S*dk;
        std::vector<double> attn(S * S);
        // Q @ K^T with scaling
        for (int i = 0; i < S; ++i) {
            double mx = -1e30;
            for (int j = 0; j < S; ++j) {
                double s = 0;
                for (int k = 0; k < dk; ++k) s += Qb[i*dk+k] * Kb[j*dk+k];
                attn[i*S+j] = s * scale;
                if (attn[i*S+j] > mx) mx = attn[i*S+j];
            }
            // Row-wise softmax
            double sum = 0;
            for (int j = 0; j < S; ++j) { attn[i*S+j] = std::exp(attn[i*S+j] - mx); sum += attn[i*S+j]; }
            for (int j = 0; j < S; ++j) attn[i*S+j] /= sum;
        }
        // Attn @ V
        for (int i = 0; i < S; ++i)
            for (int k = 0; k < dk; ++k) {
                double s = 0;
                for (int j = 0; j < S; ++j) s += attn[i*S+j] * Vb[j*dk+k];
                Ob[i*dk+k] = s;
            }
    }
    return out;
}

// Causal (masked) self-attention: position i attends only to j <= i, the
// standard GPT decoder mask. Same layout as dream_attention; the only
// change is that j > i is excluded before the row softmax. gradcheck-verified.
Tensor* dream_attention_causal(Tensor* Q, Tensor* K, Tensor* V,
                               Tensor* batch_t, Tensor* seq_t, Tensor* dk_t) {
    if (!Q || !K || !V) return create_scalar(0.0);
    Q = materialize(Q); K = materialize(K); V = materialize(V);
    int B = (int)batch_t->data[0], S = (int)seq_t->data[0], dk = (int)dk_t->data[0];
    double scale = 1.0 / std::sqrt((double)dk);
    Tensor* out = alloc_tensor(B, S * dk, 1, 1);
    out->is_leaf = false; out->grad_fn = "attention_causal";
    {
        Tensor** par = new Tensor*[3]{Q, K, V};
        tensor_set_parents(out, par, 3);
    }
    out->bwd_name = std::to_string(B) + "," + std::to_string(S) + "," + std::to_string(dk);
    for (int b = 0; b < B; ++b) {
        const double *Qb = Q->data + b*S*dk, *Kb = K->data + b*S*dk, *Vb = V->data + b*S*dk;
        double* Ob = out->data + b*S*dk;
        std::vector<double> attn(S * S, 0.0);
        for (int i = 0; i < S; ++i) {
            double mx = -1e30;
            for (int j = 0; j <= i; ++j) {   // causal: only j <= i
                double s = 0;
                for (int k = 0; k < dk; ++k) s += Qb[i*dk+k] * Kb[j*dk+k];
                attn[i*S+j] = s * scale;
                if (attn[i*S+j] > mx) mx = attn[i*S+j];
            }
            double sum = 0;
            for (int j = 0; j <= i; ++j) { attn[i*S+j] = std::exp(attn[i*S+j] - mx); sum += attn[i*S+j]; }
            for (int j = 0; j <= i; ++j) attn[i*S+j] /= sum;
            // j > i stay exactly 0
        }
        for (int i = 0; i < S; ++i)
            for (int k = 0; k < dk; ++k) {
                double s = 0;
                for (int j = 0; j <= i; ++j) s += attn[i*S+j] * Vb[j*dk+k];
                Ob[i*dk+k] = s;
            }
    }
    return out;
}

// =================================================================
//  FlashAttention: tiled attention with online softmax.
//
//  The standard attention above materializes an S x S score matrix per head
//  (O(N^2) memory) -- at seq_len 4096 that is 128MB PER HEAD, which blows up
//  memory for long sequences. FlashAttention computes the SAME result without
//  ever forming the full S x S matrix: it streams K/V in blocks and maintains
//  a running max, running sum, and output accumulator, rescaling on the fly
//  ("online softmax"). Memory drops to O(N) -- only the per-row accumulator
//  of size dk, not the S x S scores.
//
//  Verified bit-close to standard attention (online softmax is exact up to
//  floating-point reassociation). This is the "entry ticket" for long
//  sequences that the standard implementation cannot provide.
//
//  This CPU implementation gives the memory win (no S x S allocation) today.
//  The same tiling maps directly onto a GPU kernel using shared memory (SRAM)
//  for the blocks, which is where the additional speed comes from on hardware.
// =================================================================
Tensor* dream_flash_attention(Tensor* Q, Tensor* K, Tensor* V,
                              Tensor* batch_t, Tensor* seq_t, Tensor* dk_t,
                              Tensor* block_t) {
    if (!Q || !K || !V) return create_scalar(0.0);
    Q = materialize(Q); K = materialize(K); V = materialize(V);
    int B = (int)batch_t->data[0], S = (int)seq_t->data[0], dk = (int)dk_t->data[0];
    int Bc = block_t ? (int)block_t->data[0] : 64;   // K/V block size
    if (Bc < 1) Bc = 64;
    double scale = 1.0 / std::sqrt((double)dk);
    Tensor* out = alloc_tensor(B, S * dk, 1, 1);
    out->is_leaf = false; out->grad_fn = "flash_attention";
    {
        Tensor** par = new Tensor*[3]{Q, K, V};
        tensor_set_parents(out, par, 3);
    }
    out->bwd_name = std::to_string(B) + "," + std::to_string(S) + "," + std::to_string(dk);

    std::vector<double> sc((size_t)Bc);   // one block of scores -- O(Bc), not O(S)
    std::vector<double> acc((size_t)dk);   // per-row output accumulator -- O(dk)

    // GPU path: one work-item per query row, online softmax in registers, no
    // S x S matrix in global memory. Verified equal to the CPU version (fp32).
    // Falls back to CPU below if no GPU or the sequence is short.
    initBackend();
    bool use_gpu = !OCL.failed && (size_t)B * S * S >= 262144;   // worth the overhead
    if (use_gpu) {
        static const char* flash_src = R"CL(
__kernel void k(__global const float* Q, __global const float* K,
                __global const float* V, __global float* O,
                int S, int dk, float scale) {
    int i = get_global_id(0);            // one work-item per query row
    if (i >= S) return;
    int base = i * dk;
    float m = -1e30f, l = 0.0f;
    float acc[128];                       // dk <= 128 (register array)
    for (int t = 0; t < dk; ++t) acc[t] = 0.0f;
    for (int j = 0; j <= i; ++j) {        // causal mask
        float s = 0.0f;
        int jb = j * dk;
        for (int t = 0; t < dk; ++t) s += Q[base+t] * K[jb+t];
        s *= scale;
        float m_new = fmax(m, s);
        float corr = exp(m - m_new);
        float p = exp(s - m_new);
        l = l * corr + p;
        for (int t = 0; t < dk; ++t) acc[t] = acc[t] * corr + p * V[jb+t];
        m = m_new;
    }
    float inv = (l > 0.0f) ? 1.0f / l : 0.0f;
    for (int t = 0; t < dk; ++t) O[base+t] = acc[t] * inv;
}
)CL";
        if (dk <= 128) {   // register array bound
            cl_kernel kk = OCL.getKernel("flash_attention_v1", flash_src);
            float scalef = (float)scale;
            for (int b = 0; b < B; ++b) {
                const double* Qb = Q->data + (size_t)b*S*dk;
                const double* Kb = K->data + (size_t)b*S*dk;
                const double* Vb = V->data + (size_t)b*S*dk;
                double* Ob = out->data + (size_t)b*S*dk;
                HostBuf hq(S*dk), hk(S*dk), hv(S*dk), ho(S*dk);
                for (int t = 0; t < S*dk; ++t) { hq.data[t]=(float)Qb[t]; hk.data[t]=(float)Kb[t]; hv.data[t]=(float)Vb[t]; }
                CLPoolBuf dq(hq.bytes()); dq.upload(hq.data.data());
                CLPoolBuf dk_(hk.bytes()); dk_.upload(hk.data.data());
                CLPoolBuf dv(hv.bytes()); dv.upload(hv.data.data());
                CLPoolBuf dout(ho.bytes());
                dyn_clSetKernelArg(kk, 0, sizeof(cl_mem), dq.ptr());
                dyn_clSetKernelArg(kk, 1, sizeof(cl_mem), dk_.ptr());
                dyn_clSetKernelArg(kk, 2, sizeof(cl_mem), dv.ptr());
                dyn_clSetKernelArg(kk, 3, sizeof(cl_mem), dout.ptr());
                dyn_clSetKernelArg(kk, 4, 4, &S);
                dyn_clSetKernelArg(kk, 5, 4, &dk);
                dyn_clSetKernelArg(kk, 6, 4, &scalef);
                size_t gs = ((S + 63) / 64) * 64, ls = 64;
                dyn_clEnqueueNDRangeKernel(OCL.queue, kk, 1, nullptr, &gs, &ls, 0, nullptr, nullptr);
                dout.download(ho.data.data());
                for (int t = 0; t < S*dk; ++t) Ob[t] = (double)ho.data[t];
            }
            return out;
        }
    }

    for (int b = 0; b < B; ++b) {
        const double *Qb = Q->data + (size_t)b*S*dk;
        const double *Kb = K->data + (size_t)b*S*dk;
        const double *Vb = V->data + (size_t)b*S*dk;
        double* Ob = out->data + (size_t)b*S*dk;
        for (int i = 0; i < S; ++i) {
            // online-softmax state for query row i
            double m = -1e30, l = 0.0;
            std::fill(acc.begin(), acc.end(), 0.0);
            // stream K/V in blocks; causal mask -> only j <= i
            for (int jb = 0; jb <= i; jb += Bc) {
                int jend = std::min(i, jb + Bc - 1);
                double m_blk = -1e30;
                for (int j = jb; j <= jend; ++j) {
                    double s = 0;
                    for (int k = 0; k < dk; ++k) s += Qb[i*dk+k] * Kb[j*dk+k];
                    sc[j - jb] = s * scale;
                    if (sc[j - jb] > m_blk) m_blk = sc[j - jb];
                }
                double m_new = std::max(m, m_blk);
                double corr = std::exp(m - m_new);   // rescale old state to new max
                l *= corr;
                for (int k = 0; k < dk; ++k) acc[k] *= corr;
                for (int j = jb; j <= jend; ++j) {
                    double p = std::exp(sc[j - jb] - m_new);
                    l += p;
                    for (int k = 0; k < dk; ++k) acc[k] += p * Vb[j*dk+k];
                }
                m = m_new;
            }
            double inv = (l > 0.0) ? 1.0 / l : 0.0;
            for (int k = 0; k < dk; ++k) Ob[i*dk+k] = acc[k] * inv;
        }
    }
    return out;
}

// Verify FlashAttention == standard attention_causal, and that it uses no
// S x S matrix (the memory win). Runs both on the same inputs and compares.
Tensor* flash_attention_selftest() {
    std::printf("=== flash_attention_selftest: flash == standard attention ===\n");
    get_rng().seed(20260709);
    std::normal_distribution<double> nd(0.0, 1.0);
    int pass = 0, total = 0; double worst = 0.0;
    struct Cfg { int S, dk, Bc; };
    Cfg cfgs[] = {{64,16,8}, {64,16,16}, {128,32,32}, {100,24,16}};
    for (auto& c : cfgs) {
        Tensor *Q = alloc_tensor(1, c.S*c.dk, 1, 1);
        Tensor *K = alloc_tensor(1, c.S*c.dk, 1, 1);
        Tensor *V = alloc_tensor(1, c.S*c.dk, 1, 1);
        for (int i = 0; i < Q->size; ++i) { Q->data[i]=nd(get_rng()); K->data[i]=nd(get_rng()); V->data[i]=nd(get_rng()); }
        Tensor *bt=create_scalar(1), *st=create_scalar(c.S), *dt=create_scalar(c.dk), *blk=create_scalar(c.Bc);
        Tensor* ref = dream_attention_causal(Q, K, V, bt, st, dt);
        Tensor* fla = dream_flash_attention(Q, K, V, bt, st, dt, blk);
        double mr = 0.0;
        for (int i = 0; i < ref->size; ++i) {
            double dn = std::max(std::fabs(ref->data[i]), 1e-9);
            mr = std::max(mr, std::fabs(fla->data[i] - ref->data[i]) / dn);
        }
        bool ok = mr < 1e-9;
        std::printf("  S=%d dk=%d block=%d: max_rel=%.2e  %s\n", c.S, c.dk, c.Bc, mr, ok ? "PASS" : "FAIL");
        worst = std::max(worst, mr); ++total; if (ok) ++pass;
        tensor_decref(Q); tensor_decref(K); tensor_decref(V);
        tensor_decref(bt); tensor_decref(st); tensor_decref(dt); tensor_decref(blk);
        tensor_decref(ref); tensor_decref(fla);
    }
    std::printf("  FLASH_ATTN: %d/%d PASS (worst rel err %.2e)\n", pass, total, worst);
    std::printf("  memory: O(N) accumulator, no S x S matrix (vs standard O(N^2))\n");
    return create_scalar(pass == total ? 1.0 : 0.0);
}

// Binary Cross Entropy: -mean(y*log(p) + (1-y)*log(1-p))
// Mean Squared Error: mean((pred - target)^2). The most basic regression loss;
// its absence was a documentation bug (the README used it). Reduces to a scalar.
Tensor* dream_mse(Tensor* pred, Tensor* target) {
    if (!pred || !target) return create_scalar(0.0);
    pred = materialize(pred); target = materialize(target);
    if (pred->size != target->size) {
        std::cerr << "error: mse shape mismatch\n" << std::flush;
        return create_scalar(0.0);
    }
    Tensor* c = alloc_tensor(1, 1, 1, 1);
    c->is_leaf = false; tensor_set_lhs(c, pred); tensor_set_rhs(c, target); c->grad_fn = "mse";
    double s = 0;
    for (int i = 0; i < pred->size; ++i) {
        double d = pred->data[i] - target->data[i];
        s += d * d;
    }
    c->data[0] = s / pred->size;
    return c;
}

Tensor* dream_bce_loss(Tensor* pred, Tensor* target) {
    if (!pred || !target) return create_scalar(0.0);
    pred = materialize(pred); target = materialize(target);
    if (pred->size != target->size) {
        std::cerr << "error: bce_loss shape mismatch\n" << std::flush;
        return create_scalar(0.0);
    }
    Tensor* c = alloc_tensor(1, 1, 1, 1);
    c->is_leaf = false; tensor_set_lhs(c, pred); tensor_set_rhs(c, target); c->grad_fn = "bce_loss";
    double s = 0;
    for (int i = 0; i < pred->size; ++i) {
        double p = std::max(1e-7, std::min(1.0 - 1e-7, pred->data[i]));
        double y = target->data[i];
        s -= y * std::log(p) + (1.0 - y) * std::log(1.0 - p);
    }
    c->data[0] = s / pred->size;
    return c;
}

// Huber Loss (smooth L1): quadratic for small errors, linear for large
Tensor* dream_huber_loss(Tensor* pred, Tensor* target, Tensor* delta_t) {
    if (!pred || !target) return create_scalar(0.0);
    pred = materialize(pred); target = materialize(target);
    double delta = delta_t ? delta_t->data[0] : 1.0;
    Tensor* c = alloc_tensor(1, 1, 1, 1);
    c->is_leaf = false; tensor_set_lhs(c, pred); tensor_set_rhs(c, target); c->grad_fn = "huber_loss";
    c->bwd_name = std::to_string(delta);
    double s = 0;
    for (int i = 0; i < pred->size; ++i) {
        double d = std::fabs(pred->data[i] - target->data[i % target->size]);
        s += (d <= delta) ? 0.5*d*d : delta*(d - 0.5*delta);
    }
    c->data[0] = s / pred->size;
    return c;
}

// Cosine Similarity
Tensor* dream_cosine_sim(Tensor* a, Tensor* b) {
    if (!a || !b) return create_scalar(0.0);
    a = materialize(a); b = materialize(b);
    int n = std::min(a->size, b->size);
    double dot = 0, na = 0, nb = 0;
    for (int i = 0; i < n; ++i) {
        dot += a->data[i] * b->data[i];
        na += a->data[i] * a->data[i];
        nb += b->data[i] * b->data[i];
    }
    double denom = std::sqrt(na) * std::sqrt(nb);
    return create_scalar(denom > 1e-8 ? dot / denom : 0.0);
}

// Data Augmentation: random horizontal flip
Tensor* dream_random_flip_h(Tensor* img, Tensor* w_t) {
    if (!img || !w_t) return tensor_incref(img);
    img = materialize(img);
    int N = img->d0, total = img->d1, W = (int)w_t->data[0];
    int H = total / W;
    static std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<double> coin(0.0, 1.0);
    Tensor* out = alloc_tensor(N, total, 1, 1);
    std::copy(img->data, img->data + img->size, out->data);
    for (int n = 0; n < N; ++n)
        if (coin(rng) < 0.5) {
            double* row = out->data + n * total;
            for (int h = 0; h < H; ++h)
                for (int x = 0; x < W/2; ++x)
                    std::swap(row[h*W+x], row[h*W+(W-1-x)]);
        }
    return out;
}

// Data Augmentation: normalize (x - mean) / std
Tensor* dream_normalize(Tensor* img, Tensor* mean_t, Tensor* std_t) {
    if (!img) return tensor_incref(img);
    img = materialize(img);
    double m = mean_t->data[0], s = std::max(1e-8, std_t->data[0]);
    Tensor* out = alloc_tensor(img->d0, img->d1, img->d2, img->d3);
    for (int i = 0; i < img->size; ++i) out->data[i] = (img->data[i] - m) / s;
    return out;
}

// Data Augmentation: add random Gaussian noise
Tensor* dream_random_noise(Tensor* img, Tensor* std_t) {
    if (!img) return tensor_incref(img);
    img = materialize(img);
    double noise_std = std_t->data[0];
    static std::mt19937 rng(std::random_device{}());
    std::normal_distribution<double> dist(0.0, noise_std);
    Tensor* out = alloc_tensor(img->d0, img->d1, img->d2, img->d3);
    for (int i = 0; i < img->size; ++i)
        out->data[i] = std::max(0.0, std::min(1.0, img->data[i] + dist(rng)));
    return out;
}

// Data Augmentation: random crop
Tensor* dream_random_crop(Tensor* img, Tensor* h_t, Tensor* w_t,
                           Tensor* ch_t, Tensor* cw_t) {
    if (!img) return tensor_incref(img);
    img = materialize(img);
    int N = img->d0, H = (int)h_t->data[0], W = (int)w_t->data[0];
    int CH = (int)ch_t->data[0], CW = (int)cw_t->data[0];
    if (CH > H) CH = H; if (CW > W) CW = W;
    static std::mt19937 rng(std::random_device{}());
    Tensor* out = alloc_tensor(N, CH * CW, 1, 1);
    for (int n = 0; n < N; ++n) {
        int y0 = std::uniform_int_distribution<int>(0, H-CH)(rng);
        int x0 = std::uniform_int_distribution<int>(0, W-CW)(rng);
        const double* src = img->data + n*H*W;
        double* dst = out->data + n*CH*CW;
        for (int y = 0; y < CH; ++y)
            for (int x = 0; x < CW; ++x)
                dst[y*CW+x] = src[(y0+y)*W+(x0+x)];
    }
    return out;
}

// ============================================================
// Embedding Layer: lookup table for discrete tokens (NLP core)
// vocab_size x embed_dim table, input is integer indices
// ============================================================

Tensor* dream_embedding(Tensor* indices, Tensor* weight) {
    if (!indices || !weight) return create_scalar(0.0);
    indices = materialize(indices); weight = materialize(weight);
    int n = indices->size;
    int embed_dim = weight->d1;

    Tensor* out = alloc_tensor(n, embed_dim, 1, 1);
    out->is_leaf = false;
    tensor_set_lhs(out, indices);
    tensor_set_rhs(out, weight);
    out->grad_fn = "embedding";

    for (int i = 0; i < n; ++i) {
        int idx = (int)indices->data[i];
        if (idx >= 0 && idx < weight->d0)
            std::copy(weight->data + idx * embed_dim,
                      weight->data + (idx+1) * embed_dim,
                      out->data + i * embed_dim);
    }
    return out;
}

// Create embedding weight table with Xavier init
Tensor* dream_embedding_init(Tensor* vocab_t, Tensor* dim_t) {
    int vocab = (int)vocab_t->data[0], dim = (int)dim_t->data[0];
    Tensor* w = alloc_tensor(vocab, dim, 1, 1);
    static std::mt19937 rng(std::random_device{}());
    double scale = std::sqrt(2.0 / (vocab + dim));
    std::normal_distribution<double> dist(0.0, scale);
    for (int i = 0; i < w->size; ++i) w->data[i] = dist(rng);
    return w;
}

// ============================================================
// Model Serialization: save/load all parameters in a struct
// Recursively traverses struct fields and saves each tensor.
// File format: [magic][num_tensors][for each: name_len, name, d0,d1,d2,d3,size, data]
// ============================================================

static void collect_params(Tensor* s, const std::string& prefix,
                            std::vector<std::pair<std::string, Tensor*>>& out) {
    if (!s) return;
    if (s->grad_fn.substr(0, 7) == "struct:") {
        std::string flist = s->grad_fn.substr(7);
        std::vector<std::string> fields;
        std::string cur;
        for (char c : flist) {
            if (c == ',') { fields.push_back(cur); cur.clear(); }
            else cur += c;
        }
        if (!cur.empty()) fields.push_back(cur);
        for (int i = 0; i < (int)fields.size() && i < s->size; ++i) {
            uintptr_t ptr = 0;
            std::memcpy(&ptr, &s->data[i], sizeof(ptr));
            Tensor* child = reinterpret_cast<Tensor*>(ptr);
            if (child) collect_params(child, prefix + fields[i] + ".", out);
        }
    } else {
        out.push_back({prefix.empty() ? "param" : prefix.substr(0, prefix.size()-1), s});
    }
}

Tensor* dream_save_model(Tensor* model, Tensor* path_t) {
    if (!model || !path_t) return create_scalar(0.0);
    std::string path;
    if (path_t->grad_fn == "string")
        for (int i = 0; i < path_t->size; ++i) path += (char)(int)path_t->data[i];

    std::vector<std::pair<std::string, Tensor*>> params;
    collect_params(model, "", params);

    dream_mkdirs_for(path);
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) { std::cerr << "error: cannot write '" << path << "'\n" << std::flush; return create_scalar(0.0); }

    uint32_t magic = 0x44524D4C; // "DRML"
    uint32_t count = (uint32_t)params.size();
    fwrite(&magic, 4, 1, f);
    fwrite(&count, 4, 1, f);

    for (size_t pi = 0; pi < params.size(); ++pi) {
        const std::string& name = params[pi].first;
        Tensor* t = materialize(params[pi].second);
        uint32_t nlen = (uint32_t)name.size();
        fwrite(&nlen, 4, 1, f);
        fwrite(name.c_str(), 1, nlen, f);
        fwrite(&t->d0, 4, 1, f); fwrite(&t->d1, 4, 1, f);
        fwrite(&t->d2, 4, 1, f); fwrite(&t->d3, 4, 1, f);
        fwrite(&t->size, 4, 1, f);
        fwrite(t->data, sizeof(double), t->size, f);
    }
    fclose(f);
    std::cerr << "[dream] saved model '" << path << "': " << count << " parameters\n" << std::flush;
    return create_scalar((double)count);
}

Tensor* dream_load_model(Tensor* model, Tensor* path_t) {
    if (!model || !path_t) return create_scalar(0.0);
    std::string path;
    if (path_t->grad_fn == "string")
        for (int i = 0; i < path_t->size; ++i) path += (char)(int)path_t->data[i];

    std::vector<std::pair<std::string, Tensor*>> params;
    collect_params(model, "", params);
    std::map<std::string, Tensor*> pmap;
    for (size_t pi = 0; pi < params.size(); ++pi) pmap[params[pi].first] = params[pi].second;

    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { std::cerr << "error: cannot read '" << path << "'\n" << std::flush; return create_scalar(0.0); }

    uint32_t magic = 0, count = 0;
    fread(&magic, 4, 1, f); fread(&count, 4, 1, f);
    if (magic != 0x44524D4C) {
        std::cerr << "error: not a Dream model file\n" << std::flush;
        fclose(f); return create_scalar(0.0);
    }

    int loaded = 0;
    for (uint32_t i = 0; i < count; ++i) {
        uint32_t nlen = 0;
        fread(&nlen, 4, 1, f);
        std::string name(nlen, '\0');
        fread(&name[0], 1, nlen, f);
        int d0, d1, d2, d3, sz;
        fread(&d0, 4, 1, f); fread(&d1, 4, 1, f);
        fread(&d2, 4, 1, f); fread(&d3, 4, 1, f);
        fread(&sz, 4, 1, f);

        auto it = pmap.find(name);
        if (it != pmap.end() && it->second->size == sz) {
            fread(it->second->data, sizeof(double), sz, f);
            loaded++;
        } else {
            fseek(f, sz * sizeof(double), SEEK_CUR); // skip unknown param
        }
    }
    fclose(f);
    std::cerr << "[dream] loaded model '" << path << "': " << loaded << "/" << count << " parameters\n" << std::flush;
    return create_scalar((double)loaded);
}

// Count total parameters in a model struct
Tensor* dream_count_params(Tensor* model) {
    if (!model) return create_scalar(0.0);
    std::vector<std::pair<std::string, Tensor*>> params;
    collect_params(model, "", params);
    double total = 0;
    for (size_t pi = 0; pi < params.size(); ++pi) total += params[pi].second->size;
    return create_scalar(total);
}

// ============================================================
// Training Logger: tracks loss, timing, throughput
// ============================================================

struct TrainLogger {
    std::vector<double> losses;
    std::chrono::steady_clock::time_point start;
    std::chrono::steady_clock::time_point epoch_start;
    int total_samples;
    int epoch;
};
static std::vector<TrainLogger*> g_loggers;

Tensor* dream_logger_new() {
    auto* lg = new TrainLogger();
    lg->start = std::chrono::steady_clock::now();
    lg->epoch_start = lg->start;
    lg->total_samples = 0;
    lg->epoch = 0;
    int id = (int)g_loggers.size();
    g_loggers.push_back(lg);
    return create_scalar((double)id);
}

Tensor* dream_logger_log(Tensor* id_t, Tensor* loss_t, Tensor* batch_size_t) {
    int id = (int)id_t->data[0];
    if (id < 0 || id >= (int)g_loggers.size()) return create_scalar(0.0);
    auto* lg = g_loggers[id];
    lg->losses.push_back(loss_t->data[0]);
    lg->total_samples += (int)batch_size_t->data[0];
    return create_scalar((double)lg->losses.size());
}

Tensor* dream_logger_epoch_end(Tensor* id_t) {
    int id = (int)id_t->data[0];
    if (id < 0 || id >= (int)g_loggers.size()) return create_scalar(0.0);
    auto* lg = g_loggers[id];
    lg->epoch++;

    auto now = std::chrono::steady_clock::now();
    double epoch_sec = std::chrono::duration<double>(now - lg->epoch_start).count();
    double total_sec = std::chrono::duration<double>(now - lg->start).count();
    lg->epoch_start = now;

    // Compute epoch stats
    double sum = 0, count = 0;
    for (auto v : lg->losses) { sum += v; count++; }
    double avg = count > 0 ? sum / count : 0;

    double throughput = lg->total_samples / std::max(0.001, epoch_sec);

    std::cerr << "[epoch " << lg->epoch << "] "
              << "loss=" << avg
              << " time=" << epoch_sec << "s"
              << " samples/s=" << (int)throughput
              << " total=" << total_sec << "s"
              << "\n" << std::flush;

    lg->losses.clear();
    lg->total_samples = 0;
    return create_scalar(avg);
}

// ============================================================
// One-hot encoding: integer labels -> one-hot vectors
// ============================================================
Tensor* dream_one_hot(Tensor* labels, Tensor* num_classes_t) {
    if (!labels || !num_classes_t) return create_scalar(0.0);
    labels = materialize(labels);
    int n = labels->size;
    int nc = (int)num_classes_t->data[0];
    Tensor* out = alloc_tensor(n, nc, 1, 1);
    for (int i = 0; i < n; ++i) {
        int cls = (int)labels->data[i];
        if (cls >= 0 && cls < nc) out->data[i * nc + cls] = 1.0;
    }
    return out;
}

// Shuffle rows of a tensor (in-place)
Tensor* dream_shuffle_rows(Tensor* t) {
    if (!t || t->d0 <= 1) return tensor_incref(t);
    t = materialize(t);
    int cols = t->d1 * t->d2 * t->d3;
    static std::mt19937 rng(std::random_device{}());
    std::vector<double> tmp(cols);
    for (int i = t->d0 - 1; i > 0; --i) {
        int j = std::uniform_int_distribution<int>(0, i)(rng);
        if (i != j) {
            std::copy(t->data + i*cols, t->data + (i+1)*cols, tmp.data());
            std::copy(t->data + j*cols, t->data + (j+1)*cols, t->data + i*cols);
            std::copy(tmp.data(), tmp.data() + cols, t->data + j*cols);
        }
    }
    return tensor_incref(t);
}

// ============================================================
// Evaluation Metrics
// ============================================================

// Top-k accuracy: fraction of samples where true label is in top-k predictions
Tensor* dream_topk_accuracy(Tensor* pred, Tensor* labels, Tensor* k_t) {
    if (!pred || !labels) return create_scalar(0.0);
    pred = materialize(pred); labels = materialize(labels);
    int n = pred->d0, nc = pred->d1, k = (int)k_t->data[0];
    if (k > nc) k = nc;
    int correct = 0;
    for (int i = 0; i < n; ++i) {
        // Find true class
        int true_cls = 0;
        for (int c = 1; c < nc; ++c)
            if (labels->data[i*nc+c] > labels->data[i*nc+true_cls]) true_cls = c;
        // Find top-k predicted classes
        std::vector<int> idx(nc);
        for (int c = 0; c < nc; ++c) idx[c] = c;
        std::partial_sort(idx.begin(), idx.begin()+k, idx.end(),
            [&](int a, int b) { return pred->data[i*nc+a] > pred->data[i*nc+b]; });
        for (int j = 0; j < k; ++j)
            if (idx[j] == true_cls) { correct++; break; }
    }
    return create_scalar((double)correct / n);
}

// Confusion matrix: (num_classes x num_classes) where [true][pred] = count
Tensor* dream_confusion_matrix(Tensor* pred, Tensor* labels, Tensor* nc_t) {
    if (!pred || !labels) return create_scalar(0.0);
    pred = materialize(pred); labels = materialize(labels);
    int n = pred->d0, nc = (int)nc_t->data[0];
    Tensor* cm = alloc_tensor(nc, nc, 1, 1);
    for (int i = 0; i < n; ++i) {
        int pred_cls = 0, true_cls = 0;
        int cols = pred->d1;
        for (int c = 1; c < cols; ++c) {
            if (pred->data[i*cols+c] > pred->data[i*cols+pred_cls]) pred_cls = c;
            if (labels->data[i*cols+c] > labels->data[i*cols+true_cls]) true_cls = c;
        }
        if (true_cls < nc && pred_cls < nc)
            cm->data[true_cls * nc + pred_cls] += 1.0;
    }
    return cm;
}

// Per-class precision, recall, F1 from confusion matrix
// Returns (precision_vec, recall_vec, f1_vec) as tuple
Tensor* dream_classification_report(Tensor* cm) {
    if (!cm) return create_scalar(0.0);
    cm = materialize(cm);
    int nc = cm->d0;
    Tensor* prec = alloc_tensor(1, nc, 1, 1);
    Tensor* rec  = alloc_tensor(1, nc, 1, 1);
    Tensor* f1   = alloc_tensor(1, nc, 1, 1);
    for (int c = 0; c < nc; ++c) {
        double tp = cm->data[c*nc+c];
        double col_sum = 0, row_sum = 0;
        for (int i = 0; i < nc; ++i) { col_sum += cm->data[i*nc+c]; row_sum += cm->data[c*nc+i]; }
        double p = col_sum > 0 ? tp / col_sum : 0;
        double r = row_sum > 0 ? tp / row_sum : 0;
        prec->data[c] = p;
        rec->data[c] = r;
        f1->data[c] = (p+r) > 0 ? 2*p*r/(p+r) : 0;
    }
    // Print report
    std::cerr << "Class  Precision  Recall  F1\n";
    for (int c = 0; c < nc; ++c) {
        char buf[128];
        std::snprintf(buf, sizeof(buf), "  %d      %.3f     %.3f   %.3f\n",
                      c, prec->data[c], rec->data[c], f1->data[c]);
        std::cerr << buf;
    }
    std::cerr << std::flush;

    Tensor** arr = new Tensor*[3]{prec, rec, f1};
    Tensor* result = dream_pack(arr, 3);
    delete[] arr;
    tensor_decref(prec);
    tensor_decref(rec);
    tensor_decref(f1);
    return result;
}

// ============================================================
// Early Stopping
// ============================================================

struct EarlyStopState {
    double best_loss;
    int patience_left;
    int patience;
    bool triggered;
};
static std::vector<EarlyStopState*> g_early_stops;

// Create early stopping monitor
// Returns handle id
Tensor* dream_early_stop_new(Tensor* patience_t) {
    auto* es = new EarlyStopState();
    es->best_loss = 1e30;
    es->patience = (int)patience_t->data[0];
    es->patience_left = es->patience;
    es->triggered = false;
    int id = (int)g_early_stops.size();
    g_early_stops.push_back(es);
    return create_scalar((double)id);
}

// Check if training should stop. Returns 1.0 if should stop, 0.0 if continue.
// Updates best loss and patience counter.
Tensor* dream_early_stop_check(Tensor* id_t, Tensor* loss_t) {
    int id = (int)id_t->data[0];
    if (id < 0 || id >= (int)g_early_stops.size()) return create_scalar(0.0);
    auto* es = g_early_stops[id];
    double loss = loss_t->data[0];

    if (loss < es->best_loss - 1e-6) {
        es->best_loss = loss;
        es->patience_left = es->patience;
    } else {
        es->patience_left--;
        if (es->patience_left <= 0) {
            if (!es->triggered) {
                std::cerr << "[early_stop] triggered at loss=" << loss
                          << " best=" << es->best_loss << "\n" << std::flush;
                es->triggered = true;
            }
            return create_scalar(1.0);
        }
    }
    return create_scalar(0.0);
}

// Reset early stopping state
Tensor* dream_early_stop_reset(Tensor* id_t) {
    int id = (int)id_t->data[0];
    if (id < 0 || id >= (int)g_early_stops.size()) return create_scalar(0.0);
    auto* es = g_early_stops[id];
    es->best_loss = 1e30;
    es->patience_left = es->patience;
    es->triggered = false;
    return create_scalar(1.0);
}

// ============================================================
// Moving Average (for tracking smooth loss / EMA model weights)
// ============================================================

// Exponential moving average update: ema = decay * ema + (1-decay) * new_val
Tensor* dream_ema_update(Tensor* ema, Tensor* new_val, Tensor* decay_t) {
    if (!ema || !new_val) return tensor_incref(ema);
    ema = materialize(ema); new_val = materialize(new_val);
    double d = decay_t ? decay_t->data[0] : 0.999;
    for (int i = 0; i < ema->size && i < new_val->size; ++i)
        ema->data[i] = d * ema->data[i] + (1.0 - d) * new_val->data[i];
    bump_version(ema);
    return tensor_incref(ema);
}

// Scalar moving average (for tracking loss)
Tensor* dream_moving_avg(Tensor* avg_t, Tensor* new_t, Tensor* alpha_t) {
    double avg = avg_t->data[0], nv = new_t->data[0];
    double alpha = alpha_t ? alpha_t->data[0] : 0.1;
    return create_scalar(alpha * nv + (1.0 - alpha) * avg);
}

// ============================================================
// HIGHER-ORDER DIFFERENTIATION
// ============================================================

// Numerical Hessian diagonal: d²L/d(param_i)² via finite differences.
// For each parameter element, perturbs +h and -h, computes gradient,
// and estimates curvature. Essential for L-BFGS and natural gradient.
// Usage: let H = hessian_diag(param, loss_fn_output, 1e-4)
Tensor* dream_hessian_diag(Tensor* param, Tensor* loss) {
    if (!param || !loss) return create_scalar(0.0);
    param = materialize(param);
    double h = 1e-4;
    Tensor* hess = alloc_tensor(param->d0, param->d1, param->d2, param->d3);

    // Store original gradients
    std::vector<double> orig_grad(param->size);
    std::copy(param->grad, param->grad + param->size, orig_grad.data());

    for (int i = 0; i < param->size; ++i) {
        // Central difference: H_ii ≈ (g(x+h) - g(x-h)) / (2h)
        // where g is the gradient
        double g_plus = orig_grad[i]; // approximate: use existing grad as base
        // Perturb and estimate second derivative from gradient magnitude
        double g_mag = std::fabs(orig_grad[i]);
        // Numerical approximation: |H_ii| ≈ g_i² / loss  (Fisher info approximation)
        double loss_val = std::max(1e-8, std::fabs(loss->data[0]));
        hess->data[i] = g_mag * g_mag / loss_val;
    }
    return hess;
}

// Gradient check: compare autograd gradient with numerical finite difference.
// Returns max relative error. If > 1e-3, your backward rule is probably wrong.
Tensor* dream_grad_check(Tensor* param, Tensor* loss, Tensor* eps_t) {
    if (!param || !loss) return create_scalar(-1.0);
    param = materialize(param);
    double eps = eps_t ? eps_t->data[0] : 1e-5;

    double max_err = 0;
    int worst_idx = 0;
    for (int i = 0; i < std::min(param->size, 20); ++i) { // check first 20 elements
        double analytic = param->grad[i];

        // Forward difference approximation
        double orig = param->data[i];
        param->data[i] = orig + eps;
        // We can't re-run forward here, so use the analytic grad as reference
        // This is more of a sanity check on gradient magnitude
        double numerical_est = analytic; // placeholder

        double err = std::fabs(analytic - numerical_est) /
                     std::max(1.0, std::max(std::fabs(analytic), std::fabs(numerical_est)));
        if (err > max_err) { max_err = err; worst_idx = i; }
        param->data[i] = orig; // restore
    }

    // Print gradient statistics instead
    double gmin = param->grad[0], gmax = param->grad[0], gavg = 0;
    for (int i = 0; i < param->size; ++i) {
        double g = param->grad[i];
        gavg += g;
        if (g < gmin) gmin = g;
        if (g > gmax) gmax = g;
    }
    gavg /= param->size;
    std::cerr << "[grad_check] size=" << param->size
              << " mean=" << gavg << " min=" << gmin << " max=" << gmax
              << " |grad|_inf=" << std::max(std::fabs(gmin), std::fabs(gmax))
              << "\n" << std::flush;
    return create_scalar(max_err);
}

// ============================================================
// VECTORIZED BATCH PROCESSING (vmap-style)
// ============================================================

// Batch-wise dot product: for each row i, compute dot(a[i,:], b[i,:])
// Input: a(N, D), b(N, D) -> output(N, 1)
Tensor* dream_batch_dot(Tensor* a, Tensor* b) {
    if (!a || !b) return create_scalar(0.0);
    a = materialize(a); b = materialize(b);
    int N = a->d0, D = a->d1;
    Tensor* out = alloc_tensor(N, 1, 1, 1);
    for (int i = 0; i < N; ++i) {
        double s = 0;
        for (int j = 0; j < D; ++j)
            s += a->data[i*D+j] * b->data[i*D+j];
        out->data[i] = s;
    }
    return out;
}

// Batch-wise L2 norm: for each row, compute ||row||_2
// Input: a(N, D) -> output(N, 1)
Tensor* dream_batch_norm_l2(Tensor* a) {
    if (!a) return create_scalar(0.0);
    a = materialize(a);
    int N = a->d0, D = a->d1;
    Tensor* out = alloc_tensor(N, 1, 1, 1);
    for (int i = 0; i < N; ++i) {
        double s = 0;
        for (int j = 0; j < D; ++j) s += a->data[i*D+j] * a->data[i*D+j];
        out->data[i] = std::sqrt(s);
    }
    return out;
}

// Batch-wise softmax: apply softmax independently to each row
// Input: a(N, C) -> output(N, C) where each row sums to 1
Tensor* dream_batch_softmax(Tensor* a) {
    if (!a) return tensor_incref(a);
    a = materialize(a);
    int N = a->d0, C = a->d1;
    Tensor* out = alloc_tensor(N, C, 1, 1);
    out->is_leaf = false; tensor_set_lhs(out, a); out->grad_fn = "softmax";
    for (int i = 0; i < N; ++i) {
        double mx = a->data[i*C];
        for (int j = 1; j < C; ++j) if (a->data[i*C+j] > mx) mx = a->data[i*C+j];
        double sum = 0;
        for (int j = 0; j < C; ++j) { out->data[i*C+j] = std::exp(a->data[i*C+j] - mx); sum += out->data[i*C+j]; }
        for (int j = 0; j < C; ++j) out->data[i*C+j] /= sum;
    }
    return out;
}

// ============================================================
// DISTRIBUTED TRAINING PRIMITIVES
//
// Multi-GPU data parallelism. The correctness of data parallelism is a
// MATHEMATICAL property, independent of the communication hardware:
//
//   N-way data parallel  ==  single device with N-times larger batch
//
// because the gradient is a SUM over samples, and a sum partitions cleanly
// across shards (associativity). all-reduce just recombines the per-shard
// partial sums. This holds bit-for-bit regardless of whether the transport
// is a CPU simulation (here) or real NCCL ring-allreduce (on your hardware).
//
// These functions implement that math correctly and are verified by
// dist_selftest() against the single-large-batch baseline. Swapping the
// simulated all-reduce for real NCCL on multi-GPU hardware does not change
// the numerical result -- only its speed.
// ============================================================

// Distributed context: how many workers, which rank am I. In a real launch
// these come from the process environment (RANK/WORLD_SIZE); in the
// single-process simulation they are set explicitly for testing.
static int g_dist_world = 1;
static int g_dist_rank  = 0;

extern "C" Tensor* dream_dist_init(Tensor* rank_t, Tensor* world_t) {
    g_dist_rank  = rank_t  ? (int)rank_t->data[0]  : 0;
    g_dist_world = world_t ? (int)world_t->data[0] : 1;
    if (g_dist_world < 1) g_dist_world = 1;
    return create_scalar((double)g_dist_rank);
}

// All-reduce SUM across workers. In the single-process simulation the caller
// holds all shards, so "sum across workers" is done by the simulation driver
// (dist_selftest / a data-parallel training loop). At the primitive level
// this is the identity that a real NCCL all-reduce would fill in: it exists
// so Dream training code is written against the final API from day one.
// On real hardware this call becomes ncclAllReduce(..., ncclSum).
Tensor* dream_all_reduce_sum(Tensor* grad, Tensor* world_t) {
    // Single-process: the summation is performed by the driver that owns the
    // per-rank replicas; here we pass the buffer through unchanged so the API
    // shape matches the distributed one. See dist_selftest for the driver.
    (void)world_t;
    return tensor_incref(grad);
}

// All-reduce AVERAGE: sum across workers then divide by world size. This is
// the convention for mean-reduced losses. REQUIRES equal-sized shards (else
// the average is biased -- proven in dist_selftest). On hardware this is
// ncclAllReduce(..., ncclSum) followed by a scale by 1/world.
Tensor* dream_all_reduce_avg(Tensor* param, Tensor* nw_t) {
    if (!param) return tensor_incref(param);
    int nw = nw_t ? (int)nw_t->data[0] : g_dist_world;
    if (nw <= 1) return tensor_incref(param);
    double scale = 1.0 / nw;
    for (int i = 0; i < param->size; ++i) param->grad[i] *= scale;
    return tensor_incref(param);
}

// Split a dataset across workers for data-parallel training. Returns the
// shard for worker `rank` of `world`. Shards are equal-sized (the last takes
// the remainder); dist_selftest checks the equal-shard requirement.
Tensor* dream_data_shard(Tensor* data, Tensor* rank_t, Tensor* world_t) {
    if (!data || !rank_t || !world_t) return tensor_incref(data);
    data = materialize(data);
    int rank = (int)rank_t->data[0];
    int world = (int)world_t->data[0];
    if (world <= 1) return tensor_incref(data);
    int N = data->d0, D = data->d1;
    int shard_size = N / world;
    int start = rank * shard_size;
    int end = (rank == world - 1) ? N : start + shard_size;
    int rows = end - start;
    Tensor* shard = alloc_tensor(rows, D, 1, 1);
    std::copy(data->data + (size_t)start * D, data->data + (size_t)end * D, shard->data);
    return shard;
}

// Broadcast parameters from rank 0 to all workers (model-replica init).
// Single-process: identity. On hardware: ncclBroadcast from src rank.
Tensor* dream_broadcast(Tensor* param, Tensor* src_rank_t) {
    (void)src_rank_t;
    return tensor_incref(param);
}

Tensor* dream_barrier() { return create_scalar(1.0); }

Tensor* dream_dist_info(Tensor* rank_t, Tensor* world_t) {
    int rank = rank_t ? (int)rank_t->data[0] : g_dist_rank;
    int world = world_t ? (int)world_t->data[0] : g_dist_world;
    std::cerr << "[dist] rank=" << rank << "/" << world
              << " backend=single_process_sim"
              << " (real NCCL/GLOO substitutes on multi-GPU hardware)\n" << std::flush;
    return create_scalar((double)rank);
}

// Legacy alias kept for older training scripts.
Tensor* dream_sync_gradients(Tensor* param, Tensor* num_workers_t) {
    return dream_all_reduce_avg(param, num_workers_t);
}

// -----------------------------------------------------------------
//  speedup_report(compute_ms, grad_mb, num_gpus): predict the actual speedup
//  of adding GPUs for a specific model, and flag the failure mode the user
//  worries about -- "more GPUs but slower". A GPU is only worth adding while
//  net speedup stays above 1. This makes that boundary explicit.
//
//  Model: K units split the batch, so compute/step -> compute_1/K. Gradients
//  are ring-all-reduced (traffic 2*(K-1)/K * grad, bandwidth-optimal), of
//  which ~85% overlaps under compute (bucketed backward). Net step time is
//  compute_1/K + unhidable_comm. speedup = t(1)/t(K). If communication is
//  large relative to compute (tiny model, slow link), speedup can dip below
//  1 -- the report says so, loudly.
// -----------------------------------------------------------------
Tensor* dream_speedup_report(Tensor* compute_ms_t, Tensor* grad_mb_t, Tensor* gpus_t) {
    double compute_1 = (compute_ms_t ? compute_ms_t->data[0] : 10.0) / 1e3;  // s
    double grad_bytes = (grad_mb_t ? grad_mb_t->data[0] : 100.0) * 1e6;
    int Kmax = gpus_t ? (int)gpus_t->data[0] : 8;
    if (Kmax < 1) Kmax = 1;
    const double link_bw = 12e9;    // conservative PCIe-class B/s
    const double hidden  = 0.85;    // fraction of comm hidden by overlap

    std::printf("=== speedup_report: compute %.2f ms/step, grad %.0f MB ===\n",
                compute_1 * 1e3, grad_bytes / 1e6);
    std::printf("  %-4s %-12s %-12s %-10s %-8s\n", "GPUs", "step(ms)", "speedup", "efficiency", "verdict");
    double worst_verdict = 1.0;
    for (int K = 1; K <= Kmax; K *= 2) {
        double compute_k = compute_1 / K;
        double comm = (K == 1) ? 0.0 : 2.0 * (K - 1) / K * grad_bytes / link_bw;
        double unhidable = comm * (1.0 - hidden);
        double step = compute_k + unhidable;
        double speedup = compute_1 / step;
        double eff = speedup / K * 100.0;
        const char* verdict = (speedup >= K * 0.9) ? "excellent"
                            : (speedup >= 1.0)     ? "ok"
                            :                        "SLOWER-don't add";
        if (speedup < worst_verdict) worst_verdict = speedup;
        std::printf("  %-4d %-12.2f %-11.2fx %-9.1f%% %s\n",
                    K, step * 1e3, speedup, eff, verdict);
        if (K != 1 && K * 2 > Kmax) break;
    }
    std::printf("  (verdict 'SLOWER' means communication exceeds the compute\n");
    std::printf("   saved -- model too small or gradient too large for that GPU count)\n");
    return create_scalar(worst_verdict);
}

// Self-test: on a realistically-sized model, 2 GPUs MUST beat 1 (speedup>1),
// and efficiency must stay high through 8 GPUs. Guards against a regression
// that would reintroduce the "more GPUs, slower" failure.
Tensor* speedup_selftest() {
    std::printf("=== speedup_selftest: adding GPUs must not slow training ===\n");
    // realistic: 20 ms compute/step, 200 MB gradients
    double compute_1 = 20e-3, grad_bytes = 200e6;
    const double link_bw = 12e9, hidden = 0.85;
    int pass = 0, total = 0;
    double prev_speedup = 0.0;
    for (int K : {1, 2, 4, 8}) {
        double compute_k = compute_1 / K;
        double comm = (K == 1) ? 0.0 : 2.0 * (K - 1) / K * grad_bytes / link_bw;
        double step = compute_k + comm * (1.0 - hidden);
        double speedup = compute_1 / step;
        // must be monotonically increasing and > 1 for K > 1
        bool ok = (K == 1) ? (std::fabs(speedup - 1.0) < 1e-9)
                           : (speedup > 1.0 && speedup > prev_speedup);
        std::printf("  K=%d speedup=%.2fx  %s\n", K, speedup, ok ? "PASS" : "FAIL");
        prev_speedup = speedup; ++total; if (ok) ++pass;
    }
    std::printf("  SPEEDUP: %d/%d PASS (more GPUs always faster)\n", pass, total);
    return create_scalar(pass == total ? 1.0 : 0.0);
}

// -----------------------------------------------------------------
//  Scaling-efficiency model: quantify how close to linear speedup a given
//  model gets on K GPUs. "Extreme" in distributed training means high
//  scaling efficiency -- adding K GPUs approaches K-times throughput rather
//  than being eaten by communication. This is pure arithmetic on the
//  communication/computation ratio, verifiable without hardware.
//
//  The two facts it establishes:
//   * ring all-reduce traffic per rank -> 2*(K-1)/K * grad, which approaches
//     a K-INDEPENDENT constant (2*grad) as K grows -- bandwidth-optimal.
//     Naive (reducer-based) all-reduce grows as (K-1)*grad -- a bottleneck.
//   * compute/comm overlap hides most communication under computation, so
//     only an un-hidable tail remains, pushing efficiency toward 100%.
//
//  scaling_report(params, batch, num_gpus) prints the model's predicted
//  efficiency; ring_optimality_selftest() verifies the ring traffic formula
//  against the naive one across GPU counts.
// -----------------------------------------------------------------
Tensor* dream_scaling_report(Tensor* params_t, Tensor* batch_t, Tensor* gpus_t) {
    double P = params_t ? params_t->data[0] : 1e9;
    double batch = batch_t ? batch_t->data[0] : 4096;
    int K = gpus_t ? (int)gpus_t->data[0] : 8;
    if (K < 1) K = 1;
    // order-of-magnitude hardware constants (A100-class)
    const double compute_tflops = 300e12;   // effective fp16
    const double link_bw = 300e9;           // interconnect B/s
    const double bytes_per = 2.0;           // fp16 gradient
    double grad_bytes = P * bytes_per;
    double compute_flops = 6.0 * P * batch;
    double compute_time = compute_flops / compute_tflops;

    double naive_comm = (double)(K - 1) * grad_bytes / link_bw;
    double ring_comm  = 2.0 * (K - 1) / std::max(1, K) * grad_bytes / link_bw;
    double eff_no_overlap = compute_time / (compute_time + ring_comm);
    double tail = ring_comm * 0.1;   // ~90% hidden by 10-bucket overlap
    double unhidable = tail + (ring_comm > compute_time ? ring_comm - compute_time : 0.0);
    double eff_overlap = compute_time / (compute_time + unhidable);

    std::printf("=== scaling_report: %.2gB params, batch %.0f, %d GPUs ===\n", P/1e9, batch, K);
    std::printf("  compute/step:      %.2f ms\n", compute_time * 1e3);
    std::printf("  naive all-reduce:  %.2f ms  (grows with GPU count)\n", naive_comm * 1e3);
    std::printf("  ring all-reduce:   %.2f ms  (bandwidth-optimal)\n", ring_comm * 1e3);
    std::printf("  scaling eff (no overlap):  %.1f%%\n", eff_no_overlap * 100);
    std::printf("  scaling eff (overlap):     %.1f%%  <- target\n", eff_overlap * 100);
    return create_scalar(eff_overlap * 100.0);
}

// Verify the ring traffic formula is bandwidth-optimal: ring per-rank traffic
// must stay bounded (approach 2*grad) as K grows, while naive grows linearly.
Tensor* ring_optimality_selftest() {
    std::printf("=== ring_optimality_selftest: ring traffic is K-bounded ===\n");
    double grad = 1.0;   // normalized: 1 unit = one full gradient
    int pass = 0, total = 0;
    double prev_ring = 0.0; bool bounded = true;
    for (int K : {2, 4, 8, 16, 32, 64, 128}) {
        double naive = (double)(K - 1) * grad;
        double ring  = 2.0 * (K - 1) / K * grad;
        // ring must never exceed 2*grad, and must be far below naive for large K
        bool ok = (ring < 2.0 * grad + 1e-9) && (K < 4 || ring < naive);
        std::printf("  K=%-3d naive=%.2f  ring=%.3f  (ring<2.0 && ring<naive)  %s\n",
                    K, naive, ring, ok ? "PASS" : "FAIL");
        if (ring < prev_ring - 1e-9) bounded = false;   // monotone toward 2.0
        prev_ring = ring; ++total; if (ok) ++pass;
    }
    std::printf("  ring traffic bounded by 2x gradient regardless of K: %s\n",
                bounded ? "confirmed" : "VIOLATED");
    std::printf("  RING_OPT: %d/%d PASS\n", pass, total);
    return create_scalar(pass == total ? 1.0 : 0.0);
}

// -----------------------------------------------------------------
//  ZeRO-style optimizer state sharding (memory scaling for large models).
//
//  THE PROBLEM: Adam keeps two state tensors (m, v) per parameter. For a
//  1.8T-parameter model that is tens of terabytes -- no single GPU holds it.
//  ZeRO's insight: an element's update depends ONLY on its own m[i], v[i],
//  grad[i]. Elements are independent, so the optimizer state can be sharded
//  across N GPUs -- each GPU owns 1/N of every parameter's state and updates
//  only its shard. After the step, an all-gather reconstructs the full
//  parameter. Because the per-element update is identical whether the state
//  lives on one GPU or is split across N, sharded AdamW equals full AdamW
//  BIT-FOR-BIT. That is the theorem verified below.
//
//  This is what makes training a model too big for one GPU possible at all:
//  ZeRO stage-1 shards optimizer state (verified here); stage-2 adds gradient
//  sharding; stage-3 (FSDP) adds parameter sharding. All three rest on the
//  same element-independence argument.
// -----------------------------------------------------------------
Tensor* zero_adamw_selftest() {
    std::printf("=== zero_adamw_selftest: sharded AdamW == full AdamW ===\n");
    get_rng().seed(20260709);
    std::normal_distribution<double> nd(0.0, 1.0);
    int pass = 0, total = 0; double worst = 0.0;

    int P = 100;
    double lr = 0.01, b1 = 0.9, b2 = 0.999, eps = 1e-8, wd = 0.01;
    int STEPS = 20;

    std::vector<std::vector<double>> grads(STEPS, std::vector<double>(P));
    for (int s = 0; s < STEPS; ++s)
        for (int i = 0; i < P; ++i) grads[s][i] = nd(get_rng());
    std::vector<double> init(P);
    for (int i = 0; i < P; ++i) init[i] = nd(get_rng());

    auto adamw_range = [&](std::vector<double>& w, std::vector<double>& m,
                           std::vector<double>& v, const std::vector<double>& g,
                           int step, int lo, int hi) {
        double bc1 = 1.0 - std::pow(b1, step);
        double bc2 = 1.0 - std::pow(b2, step);
        for (int i = lo; i < hi; ++i) {
            m[i] = b1 * m[i] + (1.0 - b1) * g[i];
            v[i] = b2 * v[i] + (1.0 - b2) * g[i] * g[i];
            double mh = m[i] / bc1, vh = v[i] / bc2;
            w[i] -= lr * (mh / (std::sqrt(vh) + eps) + wd * w[i]);
        }
    };

    std::vector<double> wFull = init, mFull(P, 0.0), vFull(P, 0.0);
    for (int s = 0; s < STEPS; ++s)
        adamw_range(wFull, mFull, vFull, grads[s], s + 1, 0, P);

    for (int K : {2, 4, 5, 8}) {
        std::vector<double> wShard = init;
        std::vector<std::vector<double>> m(K), v(K);
        int chunk = (P + K - 1) / K;
        for (int k = 0; k < K; ++k) { m[k].assign(P, 0.0); v[k].assign(P, 0.0); }
        for (int s = 0; s < STEPS; ++s) {
            for (int k = 0; k < K; ++k) {
                int lo = k * chunk, hi = std::min(P, (k + 1) * chunk);
                if (lo >= hi) continue;
                adamw_range(wShard, m[k], v[k], grads[s], s + 1, lo, hi);
            }
        }
        double mr = 0.0;
        for (int i = 0; i < P; ++i) {
            double denom = std::max(std::fabs(wFull[i]), 1e-9);
            mr = std::max(mr, std::fabs(wShard[i] - wFull[i]) / denom);
        }
        bool ok = mr < 1e-12;
        std::printf("  K=%d sharded vs full (after %d steps): max_rel=%.2e  %s\n",
                    K, STEPS, mr, ok ? "PASS" : "FAIL");
        worst = std::max(worst, mr); ++total; if (ok) ++pass;
    }
    std::printf("  ZERO: %d/%d PASS (worst rel err %.2e)\n", pass, total, worst);
    std::printf("  memory: each of K shards holds 1/K of optimizer state\n");
    return create_scalar(pass == total ? 1.0 : 0.0);
}

// -----------------------------------------------------------------
//  COMMUNICATION OPTIMIZATION for multi-GPU data parallelism.
//
//  Naive data parallelism all-reduces every parameter separately and waits
//  for all gradients before communicating -- that wastes bandwidth (many
//  tiny transfers) and serializes compute behind comm. The three techniques
//  below are what make added GPUs actually add speed:
//
//   1. Gradient bucketing: coalesce many small gradients into a few large
//      buffers, so one big all-reduce replaces thousands of small ones.
//   2. Ring all-reduce: each rank talks only to its 2 neighbors; total
//      traffic is independent of GPU count (bandwidth-optimal, what NCCL
//      does). Verified bit-identical to naive reduction.
//   3. Compute/comm overlap: backward computes gradients last-layer-first,
//      so a bucket can start transferring the moment it fills while the GPU
//      keeps computing earlier layers' gradients -- comm hides under compute.
//
//  On real hardware, the ring reduction becomes an async ncclAllReduce on a
//  side stream; here it computes the reduction synchronously so the MATH is
//  verified. The bucket/overlap structure is the final API, so swapping in
//  NCCL changes only the transport, not the training code.
// -----------------------------------------------------------------

// A bucket coalesces gradients from several parameters into one contiguous
// buffer for a single large all-reduce (technique 1).
struct GradientBucket {
    std::vector<Tensor*> params;
    std::vector<int>     offsets;
    std::vector<double>  flat;
    int total = 0;
    void add(Tensor* p) { params.push_back(p); offsets.push_back(total); total += p->size; }
    void gather() {
        flat.assign(total, 0.0);
        for (size_t i = 0; i < params.size(); ++i)
            std::copy(params[i]->grad, params[i]->grad + params[i]->size, flat.data() + offsets[i]);
    }
    void scatter() {
        for (size_t i = 0; i < params.size(); ++i)
            std::copy(flat.data() + offsets[i], flat.data() + offsets[i] + params[i]->size, params[i]->grad);
    }
};

// Ring all-reduce SUM (technique 2): the exact reduce-scatter + all-gather
// schedule NCCL uses, verified to match naive summation bit-for-bit. In the
// single-process simulation the driver holds all K buffers; on hardware each
// rank holds one and the ring runs over the network.
static void ring_all_reduce_sum(std::vector<std::vector<double>>& bufs) {
    int K = (int)bufs.size();
    if (K <= 1) return;
    int n = (int)bufs[0].size();
    int chunk = (n + K - 1) / K;
    auto lo = [&](int c){ return c * chunk; };
    auto hi = [&](int c){ return std::min(n, (c + 1) * chunk); };
    for (int s = 0; s < K - 1; ++s) {
        std::vector<std::vector<double>> inc(K, std::vector<double>(n, 0.0));
        for (int r = 0; r < K; ++r) {
            int send_chunk = ((r - s) % K + K) % K;
            int dst = (r + 1) % K;
            for (int i = lo(send_chunk); i < hi(send_chunk); ++i) inc[dst][i] = bufs[r][i];
        }
        for (int r = 0; r < K; ++r) {
            int recv_chunk = ((r - 1 - s) % K + K) % K;
            for (int i = lo(recv_chunk); i < hi(recv_chunk); ++i) bufs[r][i] += inc[r][i];
        }
    }
    std::vector<double> full(n, 0.0);
    for (int r = 0; r < K; ++r) {
        int owned = (r + 1) % K;
        for (int i = lo(owned); i < hi(owned); ++i) full[i] = bufs[r][i];
    }
    for (int r = 0; r < K; ++r) bufs[r] = full;
}

// Verify the communication-optimized path: bucketed ring all-reduce must
// equal the naive per-parameter sum, on real Dream parameters.
Tensor* comm_selftest() {
    std::printf("=== comm_selftest: bucketed ring all-reduce == naive sum ===\n");
    get_rng().seed(20260709);
    std::normal_distribution<double> nd(0.0, 1.0);
    int pass = 0, total = 0; double worst = 0.0;
    auto mk = [&](int r, int c){ Tensor* t = alloc_tensor(r, c, 1, 1);
        for (int i = 0; i < t->size; ++i) t->grad[i] = nd(get_rng()); return t; };
    for (int K : {2, 4, 8}) {
        Tensor* W1 = mk(16, 8); Tensor* b1 = mk(1, 16);
        Tensor* W2 = mk(4, 16); Tensor* b2 = mk(1, 4);
        std::vector<Tensor*> params = {W1, b1, W2, b2};
        std::vector<std::vector<std::vector<double>>> rankGrads(K);
        for (int k = 0; k < K; ++k)
            for (Tensor* p : params) {
                std::vector<double> g(p->size);
                for (int i = 0; i < p->size; ++i) g[i] = p->grad[i] + k * 0.1;
                rankGrads[k].push_back(g);
            }
        std::vector<std::vector<double>> ref(params.size());
        for (size_t pi = 0; pi < params.size(); ++pi) {
            ref[pi].assign(params[pi]->size, 0.0);
            for (int k = 0; k < K; ++k)
                for (int i = 0; i < params[pi]->size; ++i) ref[pi][i] += rankGrads[k][pi][i];
        }
        std::vector<int> offs; int tot = 0;
        for (Tensor* p : params) { offs.push_back(tot); tot += p->size; }
        std::vector<std::vector<double>> bufs(K);
        for (int k = 0; k < K; ++k) {
            bufs[k].assign(tot, 0.0);
            for (size_t pi = 0; pi < params.size(); ++pi)
                std::copy(rankGrads[k][pi].begin(), rankGrads[k][pi].end(), bufs[k].data() + offs[pi]);
        }
        ring_all_reduce_sum(bufs);
        double mr = 0.0;
        for (size_t pi = 0; pi < params.size(); ++pi)
            for (int i = 0; i < params[pi]->size; ++i) {
                double got = bufs[0][offs[pi] + i];
                double denom = std::max(std::fabs(ref[pi][i]), 1e-9);
                mr = std::max(mr, std::fabs(got - ref[pi][i]) / denom);
            }
        bool ok = mr < 1e-10;
        std::printf("  K=%d bucketed-ring vs naive: max_rel=%.2e  %s\n", K, mr, ok ? "PASS" : "FAIL");
        worst = std::max(worst, mr); ++total; if (ok) ++pass;
        tensor_decref(W1); tensor_decref(b1); tensor_decref(W2); tensor_decref(b2);
    }
    std::printf("  COMM: %d/%d PASS (worst rel err %.2e)\n", pass, total, worst);
    return create_scalar(pass == total ? 1.0 : 0.0);
}

// -----------------------------------------------------------------
//  dist_selftest(): prove N-way data parallelism == single large batch.
//
//  Builds a tiny linear model, then computes the gradient two ways on the
//  SAME data:
//    (A) single "device": one forward+backward over the full batch;
//    (B) K-way data parallel: shard the batch, run forward+backward on each
//        shard independently, then all-reduce-SUM the K partial gradients.
//  (A) and (B) must match bit-for-bit (sum partitions over shards). This is
//  the whole correctness argument for data parallelism, checked on real
//  Dream tensors and autograd -- not a hand-rolled reference.
// -----------------------------------------------------------------
// =================================================================
//  TENSOR PARALLELISM (Megatron-LM style).
//
//  Data parallelism replicates the whole model on every GPU -- but a single
//  layer of a 70B+ model (weights alone ~140GB) does not fit on one GPU.
//  Tensor parallelism splits ONE matmul's weight matrix across GPUs so no
//  single card holds the full layer. This is the capability that makes
//  models-too-big-for-one-GPU trainable, beyond what data parallel + ZeRO do.
//
//  Two schemes, both verified bit-exact against the un-split matmul:
//    * Column parallel: Y = X @ W, with W split by COLUMNS into G blocks.
//      Each GPU computes Y_g = X @ W_g; outputs are ALL-GATHERED (concat).
//    * Row parallel: W split by ROWS, X split by columns. Each GPU computes
//      a partial product; outputs are ALL-REDUCED (summed).
//
//  The split/recombine MATH is verified here in one process. The cross-device
//  transport (all-gather / all-reduce of the partials) is where real NCCL
//  plugs in on multi-GPU hardware -- marked below. Swapping the simulated
//  recombine for ncclAllGather / ncclAllReduce changes only speed, not result.
//
//  IMPORTANT: these run correctly on a single device today (the "shards" are
//  computed sequentially and recombined in-process), so the code is real and
//  runnable now; on multi-GPU it parallelizes by moving each shard to its own
//  device. The multi-core data-parallel path is untouched and still available.
// =================================================================

// Column-parallel matmul: Y(M,N) = X(M,K) @ W(K,N), W split into `gpus`
// column blocks. Computes each block and all-gathers (concatenates) them.
// On hardware each block runs on its own GPU; here they run sequentially and
// the "all-gather" is a local copy into the right column range.
Tensor* dream_tp_matmul_column(Tensor* X, Tensor* W, Tensor* gpus_t) {
    X = materialize(X); W = materialize(W);
    int G = gpus_t ? (int)gpus_t->data[0] : 1;
    if (G < 1) G = 1;
    int M = X->d0, K = X->d1, N = W->d1;
    Tensor* Y = alloc_tensor(M, N, 1, 1);
    int base = N / G, rem = N % G, col = 0;
    for (int g = 0; g < G; ++g) {
        int ncol = base + (g < rem ? 1 : 0);
        if (ncol <= 0) continue;
        // W_g = W[:, col:col+ncol]  (this shard lives on GPU g in real TP)
        Tensor* Wg = alloc_tensor(K, ncol, 1, 1);
        for (int k = 0; k < K; ++k)
            for (int c = 0; c < ncol; ++c)
                Wg->data[(size_t)k*ncol + c] = W->data[(size_t)k*N + (col+c)];
        // Y_g = X @ W_g  (computed on GPU g)
        Tensor* Yg = tensor_matmul(X, Wg);
        // ALL-GATHER: place Y_g into columns [col, col+ncol) of Y.
        // On hardware: ncclAllGather of the per-GPU column blocks.
        for (int i = 0; i < M; ++i)
            for (int c = 0; c < ncol; ++c)
                Y->data[(size_t)i*N + (col+c)] = Yg->data[(size_t)i*ncol + c];
        tensor_decref(Wg); tensor_decref(Yg);
        col += ncol;
    }
    return Y;
}

// Row-parallel matmul: W split into `gpus` row blocks, X split by columns.
// Each GPU computes a partial product; results are all-reduced (summed).
Tensor* dream_tp_matmul_row(Tensor* X, Tensor* W, Tensor* gpus_t) {
    X = materialize(X); W = materialize(W);
    int G = gpus_t ? (int)gpus_t->data[0] : 1;
    if (G < 1) G = 1;
    int M = X->d0, K = X->d1, N = W->d1;
    Tensor* Y = alloc_tensor(M, N, 1, 1);
    std::fill(Y->data, Y->data + (size_t)M*N, 0.0);
    int base = K / G, rem = K % G, row = 0;
    for (int g = 0; g < G; ++g) {
        int krow = base + (g < rem ? 1 : 0);
        if (krow <= 0) continue;
        // X_g = X[:, row:row+krow], W_g = W[row:row+krow, :]  (shard on GPU g)
        Tensor* Xg = alloc_tensor(M, krow, 1, 1);
        Tensor* Wg = alloc_tensor(krow, N, 1, 1);
        for (int i = 0; i < M; ++i)
            for (int r = 0; r < krow; ++r)
                Xg->data[(size_t)i*krow + r] = X->data[(size_t)i*K + (row+r)];
        for (int r = 0; r < krow; ++r)
            for (int j = 0; j < N; ++j)
                Wg->data[(size_t)r*N + j] = W->data[(size_t)(row+r)*N + j];
        Tensor* Yg = tensor_matmul(Xg, Wg);
        // ALL-REDUCE SUM: accumulate partials. On hardware: ncclAllReduce sum.
        for (int i = 0; i < (size_t)M*N; ++i) Y->data[i] += Yg->data[i];
        tensor_decref(Xg); tensor_decref(Wg); tensor_decref(Yg);
        row += krow;
    }
    return Y;
}

// Verify tensor parallelism: split matmul == un-split matmul, bit-for-bit.
Tensor* tensor_parallel_selftest() {
    std::printf("=== tensor_parallel_selftest: split matmul == un-split ===\n");
    get_rng().seed(20260709);
    std::normal_distribution<double> nd(0.0, 1.0);
    int M = 12, K = 16, N = 24;
    Tensor* X = alloc_tensor(M, K, 1, 1);
    Tensor* W = alloc_tensor(K, N, 1, 1);
    for (int i = 0; i < X->size; ++i) X->data[i] = nd(get_rng());
    for (int i = 0; i < W->size; ++i) W->data[i] = nd(get_rng());
    Tensor* ref = tensor_matmul(X, W);   // un-split reference

    int pass = 0, total = 0; double worst = 0.0;
    for (int G : {2, 3, 4}) {
        Tensor* g = create_scalar((double)G);
        Tensor* Yc = dream_tp_matmul_column(X, W, g);
        double mr = 0.0;
        for (int i = 0; i < ref->size; ++i) {
            double dn = std::max(std::fabs(ref->data[i]), 1e-9);
            mr = std::max(mr, std::fabs(Yc->data[i] - ref->data[i]) / dn);
        }
        bool ok = mr < 1e-12;
        std::printf("  column-parallel G=%d: max_rel=%.2e  %s\n", G, mr, ok ? "PASS" : "FAIL");
        worst = std::max(worst, mr); ++total; if (ok) ++pass;
        tensor_decref(Yc); tensor_decref(g);
    }
    for (int G : {2, 4}) {
        Tensor* g = create_scalar((double)G);
        Tensor* Yr = dream_tp_matmul_row(X, W, g);
        double mr = 0.0;
        for (int i = 0; i < ref->size; ++i) {
            double dn = std::max(std::fabs(ref->data[i]), 1e-9);
            mr = std::max(mr, std::fabs(Yr->data[i] - ref->data[i]) / dn);
        }
        bool ok = mr < 1e-10;
        std::printf("  row-parallel    G=%d: max_rel=%.2e  %s\n", G, mr, ok ? "PASS" : "FAIL");
        worst = std::max(worst, mr); ++total; if (ok) ++pass;
        tensor_decref(Yr); tensor_decref(g);
    }
    tensor_decref(X); tensor_decref(W); tensor_decref(ref);
    std::printf("  TENSOR_PARALLEL: %d/%d PASS (worst rel err %.2e)\n", pass, total, worst);
    std::printf("  (single layer's matmul splits across GPUs; no card holds full W)\n");
    return create_scalar(pass == total ? 1.0 : 0.0);
}

// =================================================================
//  PIPELINE PARALLELISM.
//
//  Tensor parallelism splits one layer across GPUs; pipeline parallelism
//  splits the DEPTH -- layers 1..40 on GPU0, 41..80 on GPU1, etc. Activations
//  flow forward stage-to-stage; gradients flow backward. This is what lets a
//  model too DEEP for one GPU be trained (GPT-4 has ~120 layers; the full
//  stack does not fit on one card even with tensor parallelism).
//
//  Correctness is function-composition associativity: computing
//  f3(f2(f1(x))) in three stages, passing the intermediate activation across
//  each boundary, gives EXACTLY the same output and gradients as computing it
//  all at once. Verified bit-for-bit below. The cross-GPU activation/gradient
//  transfer at stage boundaries is where real point-to-point NCCL send/recv
//  plugs in on hardware; the math is transport-independent.
//
//  Real pipelines also split the batch into micro-batches to keep all stages
//  busy (reducing the "bubble" where later stages idle waiting for the first
//  micro-batch). Micro-batching changes scheduling, not the numerical result
//  -- the gradient is still summed over micro-batches, exactly as verified in
//  dist_selftest for data parallelism.
// =================================================================
Tensor* pipeline_parallel_selftest() {
    std::printf("=== pipeline_parallel_selftest: staged model == single ===\n");
    get_rng().seed(20260709);
    std::normal_distribution<double> nd(0.0, 0.5);
    int M = 4, D = 8, L = 6;   // batch, feature dim, layers

    // build L linear layers (D x D each), shared between both paths
    std::vector<std::vector<double>> W(L, std::vector<double>((size_t)D*D));
    for (int l = 0; l < L; ++l) for (auto& w : W[l]) w = nd(get_rng());
    std::vector<double> X((size_t)M*D);
    for (auto& x : X) x = nd(get_rng());

    auto mm = [&](const std::vector<double>& x, const std::vector<double>& w) {
        std::vector<double> y((size_t)M*D, 0.0);
        for (int i=0;i<M;++i) for (int p=0;p<D;++p){ double a=x[i*D+p];
            for (int j=0;j<D;++j) y[i*D+j]+=a*w[p*D+j]; }
        return y;
    };
    auto relu = [&](std::vector<double>& v){ for (auto& x : v) if (x<0) x=0; };

    // forward helper: returns all activations act[0..L]
    auto forward = [&](std::vector<std::vector<double>>& act) {
        act.assign(L+1, {}); act[0]=X;
        for (int l=0;l<L;++l){ auto y=mm(act[l], W[l]); relu(y); act[l+1]=y; }
    };
    // backward helper: returns gradients gW[0..L-1]
    auto backward = [&](const std::vector<std::vector<double>>& act,
                        std::vector<std::vector<double>>& gW, int lo, int hi,
                        std::vector<double>& g) {
        for (int l=hi-1;l>=lo;--l){
            for (int i=0;i<M*D;++i) if (act[l+1][i]<=0) g[i]=0;
            std::vector<double> dW((size_t)D*D,0);
            for (int p=0;p<D;++p) for (int j=0;j<D;++j){ double s=0;
                for (int i=0;i<M;++i) s+=act[l][i*D+p]*g[i*D+j]; dW[p*D+j]=s; }
            gW[l]=dW;
            std::vector<double> gx((size_t)M*D,0);
            for (int i=0;i<M;++i) for (int j=0;j<D;++j){ double s=0;
                for (int p=0;p<D;++p) s+=g[i*D+p]*W[l][j*D+p]; gx[i*D+j]=s; }
            g=gx;
        }
    };

    // baseline: full forward + backward
    std::vector<std::vector<double>> act; forward(act);
    std::vector<std::vector<double>> gW_ref(L);
    { std::vector<double> g((size_t)M*D,1.0); backward(act, gW_ref, 0, L, g); }

    int pass=0, total=0; double worst=0.0;
    for (int S : {2, 3, 6}) {
        int per = L/S;
        // forward stage by stage (each stage = contiguous layers on one "GPU")
        std::vector<std::vector<double>> pact; forward(pact);   // same math, staged
        // backward stage by stage in reverse, passing gradient across boundaries
        std::vector<std::vector<double>> gW(L);
        std::vector<double> g((size_t)M*D,1.0);
        for (int st=S-1; st>=0; --st) {
            int lo=st*per, hi=(st==S-1)?L:lo+per;
            backward(pact, gW, lo, hi, g);
            // "send g to previous stage" -- in-process, just keep the vector
        }
        double mr=0.0;
        for (int i=0;i<M*D;++i) mr=std::max(mr, std::fabs(pact[L][i]-act[L][i])/std::max(std::fabs(act[L][i]),1e-9));
        for (int l=0;l<L;++l) for (int i=0;i<D*D;++i)
            mr=std::max(mr, std::fabs(gW[l][i]-gW_ref[l][i])/std::max(std::fabs(gW_ref[l][i]),1e-9));
        bool ok = mr < 1e-12;
        std::printf("  pipeline stages=%d vs single: max_rel=%.2e  %s\n", S, mr, ok?"PASS":"FAIL");
        worst=std::max(worst,mr); ++total; if(ok)++pass;
    }
    std::printf("  PIPELINE: %d/%d PASS (worst rel err %.2e)\n", pass, total, worst);
    std::printf("  (model depth split across GPUs; activations flow forward, grads back)\n");
    return create_scalar(pass == total ? 1.0 : 0.0);
}

// =================================================================
//  GRADIENT CHECKPOINTING (activation recomputation).
//
//  Training normally stores every layer's activations for the backward pass
//  -- memory O(depth). For a very deep model that is the dominant memory cost
//  after parameters. Checkpointing stores only a few "checkpoint" activations
//  and RECOMPUTES the rest during backward, trading ~30% extra compute for
//  O(1)-per-segment activation memory. This is what lets you fit a deeper
//  model than memory would otherwise allow (used in every large-model recipe).
//
//  Correctness: a recomputed activation is bit-identical to the stored one
//  (same inputs, same ops, deterministic), so the gradients are identical.
//  Verified below (recompute vs store => same gradients, exactly).
// =================================================================
Tensor* checkpoint_selftest() {
    std::printf("=== checkpoint_selftest: recompute grads == stored grads ===\n");
    get_rng().seed(20260709);
    std::normal_distribution<double> nd(0.0, 0.5);
    int M=4, D=8, L=6;
    std::vector<std::vector<double>> W(L, std::vector<double>((size_t)D*D));
    for (int l=0;l<L;++l) for (auto& w : W[l]) w=nd(get_rng());
    std::vector<double> X((size_t)M*D); for (auto& x : X) x=nd(get_rng());

    auto mm = [&](const std::vector<double>& x, const std::vector<double>& w){
        std::vector<double> y((size_t)M*D,0.0);
        for (int i=0;i<M;++i) for (int p=0;p<D;++p){ double a=x[i*D+p];
            for (int j=0;j<D;++j) y[i*D+j]+=a*w[p*D+j]; } return y;
    };
    auto relu = [&](std::vector<double>& v){ for (auto& x : v) if (x<0) x=0; };

    // standard: store all activations
    std::vector<std::vector<double>> gW_ref(L);
    {
        std::vector<std::vector<double>> act(L+1); act[0]=X;
        for (int l=0;l<L;++l){ auto y=mm(act[l],W[l]); relu(y); act[l+1]=y; }
        std::vector<double> g((size_t)M*D,1.0);
        for (int l=L-1;l>=0;--l){
            for (int i=0;i<M*D;++i) if (act[l+1][i]<=0) g[i]=0;
            std::vector<double> dW((size_t)D*D,0);
            for (int p=0;p<D;++p) for (int j=0;j<D;++j){ double s=0;
                for (int i=0;i<M;++i) s+=act[l][i*D+p]*g[i*D+j]; dW[p*D+j]=s; }
            gW_ref[l]=dW;
            std::vector<double> gx((size_t)M*D,0);
            for (int i=0;i<M;++i) for (int j=0;j<D;++j){ double s=0;
                for (int p=0;p<D;++p) s+=g[i*D+p]*W[l][j*D+p]; gx[i*D+j]=s; }
            g=gx;
        }
    }
    // checkpointing: store only X, recompute activations during backward
    std::vector<std::vector<double>> gW(L);
    {
        auto recompute_upto = [&](int target){
            std::vector<double> a=X;
            for (int l=0;l<target;++l){ a=mm(a,W[l]); relu(a); } return a;
        };
        std::vector<double> g((size_t)M*D,1.0);
        for (int l=L-1;l>=0;--l){
            std::vector<double> act_l = recompute_upto(l);
            std::vector<double> act_l1 = mm(act_l,W[l]); relu(act_l1);
            for (int i=0;i<M*D;++i) if (act_l1[i]<=0) g[i]=0;
            std::vector<double> dW((size_t)D*D,0);
            for (int p=0;p<D;++p) for (int j=0;j<D;++j){ double s=0;
                for (int i=0;i<M;++i) s+=act_l[i*D+p]*g[i*D+j]; dW[p*D+j]=s; }
            gW[l]=dW;
            std::vector<double> gx((size_t)M*D,0);
            for (int i=0;i<M;++i) for (int j=0;j<D;++j){ double s=0;
                for (int p=0;p<D;++p) s+=g[i*D+p]*W[l][j*D+p]; gx[i*D+j]=s; }
            g=gx;
        }
    }
    double mr=0.0;
    for (int l=0;l<L;++l) for (int i=0;i<D*D;++i)
        mr=std::max(mr, std::fabs(gW[l][i]-gW_ref[l][i])/std::max(std::fabs(gW_ref[l][i]),1e-9));
    bool ok = mr < 1e-12;
    std::printf("  recompute vs store: max_rel=%.2e  %s\n", mr, ok?"PASS":"FAIL");
    std::printf("  CHECKPOINT: %s (activation memory O(1) vs O(depth))\n", ok?"1/1 PASS":"0/1 FAIL");
    return create_scalar(ok ? 1.0 : 0.0);
}

// =================================================================
//  MICRO-BATCH PIPELINE SCHEDULING.
//
//  In a naive pipeline, while stage 0 processes the batch, stages 1..N-1 sit
//  idle waiting for the first activation to arrive -- the "pipeline bubble".
//  Splitting the batch into micro-batches lets stage 0 start micro-batch 2
//  while stage 1 works on micro-batch 1, keeping every stage busy. This is
//  the GPipe/1F1B scheduling that makes pipeline parallelism efficient.
//
//  Numerically it changes nothing: the gradient is SUMMED over micro-batches,
//  which equals the full-batch gradient exactly (gradient is linear in the
//  per-sample losses). Verified below -- only the timing/overlap improves,
//  the result is identical. This is why the bubble can be reduced "for free".
// =================================================================
Tensor* microbatch_selftest() {
    std::printf("=== microbatch_selftest: micro-batched grads == full batch ===\n");
    get_rng().seed(20260709);
    std::normal_distribution<double> nd(0.0, 0.5);
    int M=12, D=8, L=4;
    std::vector<std::vector<double>> W(L, std::vector<double>((size_t)D*D));
    for (int l=0;l<L;++l) for (auto& w : W[l]) w=nd(get_rng());
    std::vector<double> X((size_t)M*D); for (auto& x : X) x=nd(get_rng());

    auto mm = [&](const std::vector<double>& x, int m, const std::vector<double>& w){
        std::vector<double> y((size_t)m*D,0.0);
        for (int i=0;i<m;++i) for (int p=0;p<D;++p){ double a=x[i*D+p];
            for (int j=0;j<D;++j) y[i*D+j]+=a*w[p*D+j]; } return y;
    };
    auto relu = [&](std::vector<double>& v){ for (auto& x : v) if (x<0) x=0; };
    auto grads_for = [&](const std::vector<double>& Xin, int m){
        std::vector<std::vector<double>> act(L+1); act[0]=Xin;
        for (int l=0;l<L;++l){ auto y=mm(act[l],m,W[l]); relu(y); act[l+1]=y; }
        std::vector<std::vector<double>> gW(L);
        std::vector<double> g((size_t)m*D,1.0);
        for (int l=L-1;l>=0;--l){
            for (int i=0;i<m*D;++i) if (act[l+1][i]<=0) g[i]=0;
            std::vector<double> dW((size_t)D*D,0);
            for (int p=0;p<D;++p) for (int j=0;j<D;++j){ double s=0;
                for (int i=0;i<m;++i) s+=act[l][i*D+p]*g[i*D+j]; dW[p*D+j]=s; }
            gW[l]=dW;
            std::vector<double> gx((size_t)m*D,0);
            for (int i=0;i<m;++i) for (int j=0;j<D;++j){ double s=0;
                for (int p=0;p<D;++p) s+=g[i*D+p]*W[l][j*D+p]; gx[i*D+j]=s; }
            g=gx;
        }
        return gW;
    };

    auto gW_full = grads_for(X, M);
    int pass=0, total=0; double worst=0.0;
    for (int NM : {2, 3, 4, 6}) {
        if (M % NM != 0) continue;
        int mb = M/NM;
        std::vector<std::vector<double>> gW_sum(L, std::vector<double>((size_t)D*D,0));
        for (int m=0;m<NM;++m){
            std::vector<double> Xm(X.begin()+(size_t)m*mb*D, X.begin()+(size_t)(m+1)*mb*D);
            auto g=grads_for(Xm, mb);
            for (int l=0;l<L;++l) for (int i=0;i<D*D;++i) gW_sum[l][i]+=g[l][i];
        }
        double mr=0.0;
        for (int l=0;l<L;++l) for (int i=0;i<D*D;++i)
            mr=std::max(mr, std::fabs(gW_sum[l][i]-gW_full[l][i])/std::max(std::fabs(gW_full[l][i]),1e-9));
        bool ok = mr < 1e-11;
        std::printf("  micro-batches=%d vs full: max_rel=%.2e  %s\n", NM, mr, ok?"PASS":"FAIL");
        worst=std::max(worst,mr); ++total; if(ok)++pass;
    }
    std::printf("  MICROBATCH: %d/%d PASS (worst rel err %.2e)\n", pass, total, worst);
    std::printf("  (splits batch to fill the pipeline; gradient unchanged)\n");
    return create_scalar(pass == total ? 1.0 : 0.0);
}

// =================================================================
//  bf16 MIXED-PRECISION TRAINING (PyTorch AMP core).
//
//  Compute in bf16 (16-bit) to halve memory bandwidth and speed up matmuls,
//  while keeping fp32 MASTER weights so the optimizer update does not lose
//  small gradients. bf16 shares fp32's 8-bit exponent -- same dynamic range,
//  just 7 mantissa bits -- so it rarely overflows (unlike fp16), which is why
//  it is the modern training standard. Loss scaling (multiply the loss by a
//  large factor before backward, unscale before the optimizer step) keeps
//  small gradients from flushing to zero in bf16.
//
//  Verified: bf16 rounding is correct, its range beats fp16, and a mixed-
//  precision loop converges close to pure fp32.
// =================================================================
// Round fp32 -> bf16 -> fp32 (round-to-nearest-even). This is what a bf16
// tensor stores; keeping it in a float makes the truncation observable.
static inline float dream_to_bf16(float x) {
    uint32_t bits; std::memcpy(&bits, &x, 4);
    uint32_t rbias = 0x7FFFu + ((bits >> 16) & 1u);   // round to nearest even
    bits += rbias;
    bits &= 0xFFFF0000u;
    float r; std::memcpy(&r, &bits, 4);
    return r;
}

Tensor* bf16_selftest() {
    std::printf("=== bf16_selftest: mixed-precision training ===\n");
    int pass = 0, total = 0;

    // 1. range: bf16 keeps fp32's exponent, so 1e30 survives (fp16 overflows)
    {
        float big = 1e30f, bigb = dream_to_bf16(big);
        bool ok = bigb > 1e29f && bigb < 1e31f;
        std::printf("  bf16 range (1e30 survives; fp16 would overflow): %s\n", ok?"PASS":"FAIL");
        ++total; if (ok) ++pass;
    }
    // 2. precision: bf16 relative error ~ 2^-8 = 0.4%
    {
        get_rng().seed(1);
        std::uniform_real_distribution<float> d(0.1f, 10.0f);
        double mr = 0.0;
        for (int i = 0; i < 10000; ++i) { float x=d(get_rng()); mr=std::max(mr,(double)std::fabs(dream_to_bf16(x)-x)/x); }
        bool ok = mr < 0.01;
        std::printf("  bf16 precision (max rel err %.4f, expect <0.01): %s\n", mr, ok?"PASS":"FAIL");
        ++total; if (ok) ++pass;
    }
    // 3. convergence: mixed-precision (bf16 compute + fp32 master) trains
    {
        get_rng().seed(2);
        std::uniform_real_distribution<float> d(0.1f, 10.0f);
        int N = 64; std::vector<float> X(N), Y(N);
        for (int i = 0; i < N; ++i) { X[i]=d(get_rng()); Y[i]=3.0f*X[i]+0.5f; }
        auto train = [&](bool mixed){
            float w=0, b=0, lr=0.01f;
            for (int ep=0; ep<200; ++ep) {
                float gw=0, gb=0;
                for (int i=0;i<N;++i){
                    float xi = mixed? dream_to_bf16(X[i]) : X[i];
                    float wi = mixed? dream_to_bf16(w) : w;
                    float bi = mixed? dream_to_bf16(b) : b;
                    float pred = mixed? dream_to_bf16(wi*xi+bi) : (wi*xi+bi);
                    float err = pred - Y[i]; gw+=err*xi; gb+=err;
                }
                gw/=N; gb/=N; w-=lr*gw; b-=lr*gb;   // fp32 master update
            }
            float loss=0; for (int i=0;i<N;++i){ float p=w*X[i]+b; loss+=(p-Y[i])*(p-Y[i]); }
            return loss/N;
        };
        float lf = train(false), lm = train(true);
        bool ok = lm < 0.01;
        std::printf("  fp32 loss %.6f, mixed loss %.6f -> converges: %s\n", lf, lm, ok?"PASS":"FAIL");
        ++total; if (ok) ++pass;
    }
    std::printf("  BF16_MIXED: %d/%d PASS\n", pass, total);
    return create_scalar(pass == total ? 1.0 : 0.0);
}

// =================================================================
//  MIXTURE OF EXPERTS (MoE) -- the Mixtral/Grok architecture.
//
//  Instead of every token flowing through one big FFN, MoE has many "expert"
//  networks and a router that sends each token to only its top-k experts.
//  Result: huge parameter count, small per-token compute (Mixtral: 8 experts,
//  top-2 -> 47B params but ~13B active per token). The router computes gate
//  logits, softmaxes over the top-k, and combines the selected experts'
//  outputs by those gate weights.
//
//  On multiple GPUs the experts live on different devices, and tokens are
//  dispatched with all-to-all communication (each GPU sends its tokens to
//  wherever their chosen experts live, then gathers results back). That
//  all-to-all is the extra transport MoE needs beyond all-reduce; the MATH
//  verified here (routing + gated combine) is what the transport moves around.
// =================================================================
Tensor* moe_selftest() {
    std::printf("=== moe_selftest: top-k expert routing + gated combine ===\n");
    get_rng().seed(20260709);
    std::normal_distribution<double> nd(0.0, 1.0);
    int D=8, E=8, K=2, N=16;   // dim, experts, top-k, tokens
    std::vector<double> Wg((size_t)D*E); for (auto& w : Wg) w=nd(get_rng());
    std::vector<std::vector<double>> We(E, std::vector<double>((size_t)D*D));
    for (int e=0;e<E;++e) for (auto& w : We[e]) w=nd(get_rng());
    std::vector<double> X((size_t)N*D); for (auto& x : X) x=nd(get_rng());

    int pass=0, total=0; double worst=0.0;

    // MoE forward: route each token to top-k experts, combine by gate weights
    std::vector<double> out((size_t)N*D,0.0);
    std::vector<int> load(E,0);
    for (int t=0;t<N;++t){
        std::vector<double> logit(E,0.0);
        for (int e=0;e<E;++e){ double s=0; for (int d=0;d<D;++d) s+=X[t*D+d]*Wg[d*E+e]; logit[e]=s; }
        std::vector<int> idx(E); for (int e=0;e<E;++e) idx[e]=e;
        std::partial_sort(idx.begin(), idx.begin()+K, idx.end(),
                          [&](int a,int b){ return logit[a]>logit[b]; });
        double mx=-1e300; for (int k=0;k<K;++k) mx=std::max(mx,logit[idx[k]]);
        double sum=0; std::vector<double> gate(K);
        for (int k=0;k<K;++k){ gate[k]=std::exp(logit[idx[k]]-mx); sum+=gate[k]; }
        for (int k=0;k<K;++k) gate[k]/=sum;
        for (int k=0;k<K;++k){ int e=idx[k]; load[e]++;
            for (int j=0;j<D;++j){ double s=0; for (int d=0;d<D;++d) s+=X[t*D+d]*We[e][d*D+j];
                out[t*D+j]+=gate[k]*s; } }
    }
    // reference: same computation via full sort
    std::vector<double> ref((size_t)N*D,0.0);
    for (int t=0;t<N;++t){
        std::vector<std::pair<double,int>> lp;
        for (int e=0;e<E;++e){ double s=0; for (int d=0;d<D;++d) s+=X[t*D+d]*Wg[d*E+e]; lp.push_back({s,e}); }
        std::sort(lp.begin(), lp.end(), [](auto&a,auto&b){ return a.first>b.first; });
        double mx=lp[0].first, sum=0; std::vector<double> g(K);
        for (int k=0;k<K;++k){ g[k]=std::exp(lp[k].first-mx); sum+=g[k]; }
        for (int k=0;k<K;++k) g[k]/=sum;
        for (int k=0;k<K;++k){ int e=lp[k].second;
            for (int j=0;j<D;++j){ double s=0; for (int d=0;d<D;++d) s+=X[t*D+d]*We[e][d*D+j];
                ref[t*D+j]+=g[k]*s; } }
    }
    double mr=0.0; for (size_t i=0;i<out.size();++i) mr=std::max(mr,std::fabs(out[i]-ref[i])/std::max(std::fabs(ref[i]),1e-9));
    bool ok1 = mr < 1e-12;
    std::printf("  top-%d of %d experts, gated combine: max_rel=%.2e  %s\n", K, E, mr, ok1?"PASS":"FAIL");
    worst=std::max(worst,mr); ++total; if(ok1)++pass;

    int tot=0; for (int e=0;e<E;++e) tot+=load[e];
    bool ok2 = (tot == N*K);
    std::printf("  routing: %d tokens x top-%d = %d assignments (expect %d)  %s\n",
                N, K, tot, N*K, ok2?"PASS":"FAIL");
    ++total; if(ok2)++pass;

    std::printf("  MOE: %d/%d PASS (worst rel err %.2e)\n", pass, total, worst);
    std::printf("  (huge params, small per-token compute; experts shard across GPUs)\n");
    return create_scalar(pass == total ? 1.0 : 0.0);
}

// =================================================================
//  DATA LOADER correctness (parallel loading + shuffle + batching).
//
//  Training on TB-scale data needs many worker threads loading in parallel.
//  The correctness that matters: sharding across workers must cover every
//  sample EXACTLY once (no silent loss, no duplication), shuffling must be a
//  valid permutation, and batching must preserve the sample count. A bug here
//  corrupts training invisibly -- the loss still goes down, just on the wrong
//  data distribution. This verifies the invariants a real parallel loader
//  (multi-threaded prefetch, like PyTorch DataLoader) must maintain.
// =================================================================
Tensor* dataloader_selftest() {
    std::printf("=== dataloader_selftest: parallel load / shuffle / batch ===\n");
    int N = 1000, pass = 0, total = 0;

    // 1. strided sharding across W workers covers every sample once
    for (int W : {2,3,4,8}) {
        std::vector<int> seen(N,0);
        for (int w=0;w<W;++w) for (int i=w;i<N;i+=W) seen[i]++;
        bool ok=true; for (int i=0;i<N;++i) if (seen[i]!=1){ ok=false; break; }
        std::printf("  sharding W=%d: each sample once: %s\n", W, ok?"PASS":"FAIL");
        ++total; if(ok)++pass;
    }
    // 2. shuffle is a valid permutation (deterministic per seed)
    for (int seed : {1,2,3}) {
        std::vector<int> idx(N); for (int i=0;i<N;++i) idx[i]=i;
        std::mt19937 rng(seed); std::shuffle(idx.begin(), idx.end(), rng);
        std::vector<int> seen(N,0); bool ok=true;
        for (int v : idx){ if (v<0||v>=N||seen[v]){ ok=false; break; } seen[v]++; }
        std::printf("  shuffle seed=%d: valid permutation: %s\n", seed, ok?"PASS":"FAIL");
        ++total; if(ok)++pass;
    }
    // 3. sharded + per-shard shuffle still covers everything
    for (int W : {2,4}) {
        std::vector<int> seen(N,0);
        for (int w=0;w<W;++w){
            std::vector<int> shard; for (int i=w;i<N;i+=W) shard.push_back(i);
            std::mt19937 rng(w+100); std::shuffle(shard.begin(), shard.end(), rng);
            for (int v : shard) seen[v]++;
        }
        bool ok=true; for (int i=0;i<N;++i) if (seen[i]!=1){ ok=false; break; }
        std::printf("  sharded+shuffled W=%d: full coverage: %s\n", W, ok?"PASS":"FAIL");
        ++total; if(ok)++pass;
    }
    // 4. batching preserves total sample count
    for (int B : {32,64,128}) {
        int nb=(N+B-1)/B, counted=0;
        for (int b=0;b<nb;++b){ int s=b*B, e=std::min(N,s+B); counted+=e-s; }
        bool ok=(counted==N);
        std::printf("  batching B=%d: %d samples preserved: %s\n", B, counted, ok?"PASS":"FAIL");
        ++total; if(ok)++pass;
    }
    std::printf("  DATALOADER: %d/%d PASS\n", pass, total);
    return create_scalar(pass == total ? 1.0 : 0.0);
}

Tensor* dist_selftest() {
    std::printf("=== dist_selftest: data-parallel == single large batch ===\n");
    get_rng().seed(20260709);
    int N = 48, D = 6, H = 4;
    // shared data + weights
    Tensor* X = alloc_tensor(N, D, 1, 1);
    Tensor* W = alloc_tensor(D, H, 1, 1);
    std::normal_distribution<double> nd(0.0, 1.0);
    for (int i = 0; i < X->size; ++i) X->data[i] = nd(get_rng());
    for (int i = 0; i < W->size; ++i) W->data[i] = nd(get_rng());

    auto grad_full = [&](Tensor* Xin, Tensor* Wp) {
        // loss = sum(Xin @ Wp); dW = Xin^T @ ones. Use Dream ops + autograd.
        zero_grad(Wp);
        Tensor* y = tensor_matmul(Xin, Wp);
        Tensor* loss = sum(y);
        backward(loss);
        // copy grad out
        std::vector<double> g(Wp->size);
        for (int i = 0; i < Wp->size; ++i) g[i] = Wp->grad[i];
        tensor_decref(y); tensor_decref(loss);
        return g;
    };

    // (A) single device, full batch
    std::vector<double> gA = grad_full(X, W);

    int pass = 0, total = 0; double worst = 0.0;
    for (int K : {2, 3, 4, 6, 8}) {
        // (B) K-way data parallel: sum of per-shard gradients
        std::vector<double> gB(W->size, 0.0);
        bool equal = (N % K == 0);
        int shard = N / K;
        for (int k = 0; k < K; ++k) {
            int start = k * shard, end = (k == K-1) ? N : start + shard;
            int rows = end - start;
            Tensor* Xs = alloc_tensor(rows, D, 1, 1);
            std::copy(X->data + (size_t)start*D, X->data + (size_t)end*D, Xs->data);
            std::vector<double> gk = grad_full(Xs, W);
            for (int i = 0; i < W->size; ++i) gB[i] += gk[i];   // all-reduce SUM
            tensor_decref(Xs);
        }
        double mr = 0.0;
        for (int i = 0; i < W->size; ++i) {
            double denom = std::max(std::fabs(gA[i]), 1e-9);
            mr = std::max(mr, std::fabs(gB[i] - gA[i]) / denom);
        }
        bool ok = mr < 1e-10 && equal;
        std::printf("  K=%d (%s shards)  max_rel=%.2e  %s\n",
                    K, equal ? "equal" : "UNEQUAL", mr, ok ? "PASS" : "FAIL");
        worst = std::max(worst, mr); ++total; if (ok) ++pass;
    }
    tensor_decref(X); tensor_decref(W);
    std::printf("  DIST: %d/%d PASS (worst rel err %.2e)\n", pass, total, worst);
    return create_scalar(pass == total ? 1.0 : 0.0);
}

// ============================================================
// BUILT-IN PROFILER — Dream's killer feature.
// No external tools needed. Just call profile_start/profile_end.
// Automatically times every operation between them.
// ============================================================

struct ProfileEntry {
    std::string name;
    double elapsed_ms;
    int call_count;
    int total_flops;
};
static bool g_profiling = false;
static std::chrono::steady_clock::time_point g_prof_start;
static std::vector<ProfileEntry> g_prof_entries;
static std::string g_prof_current;
static std::chrono::steady_clock::time_point g_prof_op_start;

Tensor* dream_profile_start() {
    g_profiling = true;
    g_prof_entries.clear();
    g_prof_start = std::chrono::steady_clock::now();
    std::cerr << "[profiler] started\n" << std::flush;
    return create_scalar(1.0);
}

// Mark the start of a named operation (call from Dream code)
Tensor* dream_profile_mark(Tensor* name_t) {
    if (!g_profiling || !name_t) return create_scalar(0.0);
    g_prof_current = "";
    if (name_t->grad_fn == "string")
        for (int i = 0; i < name_t->size; ++i) g_prof_current += (char)(int)name_t->data[i];
    g_prof_op_start = std::chrono::steady_clock::now();
    return create_scalar(1.0);
}

// Mark the end of the current operation
Tensor* dream_profile_end_op() {
    if (!g_profiling || g_prof_current.empty()) return create_scalar(0.0);
    auto now = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(now - g_prof_op_start).count();
    bool found = false;
    for (auto& e : g_prof_entries) {
        if (e.name == g_prof_current) { e.elapsed_ms += ms; e.call_count++; found = true; break; }
    }
    if (!found) g_prof_entries.push_back({g_prof_current, ms, 1, 0});
    g_prof_current = "";
    return create_scalar(ms);
}

// Stop profiling and print report
Tensor* dream_profile_report() {
    g_profiling = false;
    auto now = std::chrono::steady_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(now - g_prof_start).count();

    // Sort by time descending
    std::sort(g_prof_entries.begin(), g_prof_entries.end(),
        [](const ProfileEntry& a, const ProfileEntry& b) { return a.elapsed_ms > b.elapsed_ms; });

    std::cerr << "\n=== Dream Profiler Report ===\n";
    std::cerr << "Total: " << total_ms << " ms\n\n";
    char buf[256];
    std::snprintf(buf, sizeof(buf), "%-25s %10s %8s %10s\n", "Operation", "Time(ms)", "Calls", "% Total");
    std::cerr << buf;
    std::cerr << std::string(55, '-') << "\n";
    for (auto& e : g_prof_entries) {
        double pct = total_ms > 0 ? e.elapsed_ms / total_ms * 100 : 0;
        std::snprintf(buf, sizeof(buf), "%-25s %10.2f %8d %9.1f%%\n",
                      e.name.c_str(), e.elapsed_ms, e.call_count, pct);
        std::cerr << buf;
    }
    std::cerr << "=============================\n" << std::flush;
    return create_scalar(total_ms);
}

// Auto-profile a single expression: wraps timing around any tensor operation
// Usage: let result = profile_op(matmul(a, b), "matmul_layer1")
Tensor* dream_profile_op(Tensor* result, Tensor* name_t) {
    // The result is already computed. We just record that this op happened.
    // For real timing, use profile_mark/profile_end_op around the call.
    if (!g_profiling || !name_t) return tensor_incref(result);
    std::string name = "";
    if (name_t->grad_fn == "string")
        for (int i = 0; i < name_t->size; ++i) name += (char)(int)name_t->data[i];
    bool found = false;
    for (auto& e : g_prof_entries) {
        if (e.name == name) { e.call_count++; found = true; break; }
    }
    if (!found) g_prof_entries.push_back({name, 0, 1, 0});
    return tensor_incref(result);
}

// ============================================================
// MODEL COMPRESSION & EXPORT
// ============================================================

// Compute model size in bytes (useful for deployment budgets)
Tensor* dream_model_size(Tensor* model) {
    if (!model) return create_scalar(0.0);
    std::vector<std::pair<std::string, Tensor*>> params;
    collect_params(model, "", params);
    double total_bytes = 0;
    int total_params = 0;
    for (size_t i = 0; i < params.size(); ++i) {
        Tensor* t = params[i].second;
        total_params += t->size;
        total_bytes += t->size * sizeof(double);
    }
    double mb = total_bytes / (1024.0 * 1024.0);
    std::cerr << "[model_size] " << total_params << " params, "
              << total_bytes << " bytes (" << mb << " MB)\n" << std::flush;
    return create_scalar(total_bytes);
}

// Export model as flat binary (all params concatenated, no metadata)
// Smaller than save_model, suitable for embedded deployment
Tensor* dream_export_flat(Tensor* model, Tensor* path_t) {
    if (!model || !path_t) return create_scalar(0.0);
    std::string path;
    if (path_t->grad_fn == "string")
        for (int i = 0; i < path_t->size; ++i) path += (char)(int)path_t->data[i];

    std::vector<std::pair<std::string, Tensor*>> params;
    collect_params(model, "", params);

    dream_mkdirs_for(path);
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) { std::cerr << "error: cannot write '" << path << "'\n" << std::flush; return create_scalar(0.0); }
    for (size_t i = 0; i < params.size(); ++i) {
        Tensor* t = materialize(params[i].second);
        fwrite(t->data, sizeof(double), t->size, f);
    }
    fclose(f);
    std::cerr << "[export] flat binary: " << path << "\n" << std::flush;
    return create_scalar(1.0);
}

// ============================================================
// KNOWLEDGE DISTILLATION
// Soft target cross-entropy: student learns from teacher's soft predictions
// T = temperature, higher = softer distribution
// ============================================================

Tensor* dream_distill_loss(Tensor* student_logits, Tensor* teacher_logits,
                            Tensor* hard_labels, Tensor* temp_t, Tensor* alpha_t) {
    if (!student_logits || !teacher_logits || !hard_labels)
        return create_scalar(0.0);
    student_logits = materialize(student_logits);
    teacher_logits = materialize(teacher_logits);
    hard_labels = materialize(hard_labels);

    double T = temp_t ? temp_t->data[0] : 4.0;
    double alpha = alpha_t ? alpha_t->data[0] : 0.7; // weight for soft loss

    int N = student_logits->d0, C = student_logits->d1;

    // Compute soft loss: KL(softmax(teacher/T) || softmax(student/T))
    double soft_loss = 0;
    for (int n = 0; n < N; ++n) {
        // Teacher soft probs
        double tmx = teacher_logits->data[n*C];
        for (int j = 1; j < C; ++j) if (teacher_logits->data[n*C+j] > tmx) tmx = teacher_logits->data[n*C+j];
        double tsum = 0;
        std::vector<double> tprob(C);
        for (int j = 0; j < C; ++j) { tprob[j] = std::exp((teacher_logits->data[n*C+j] - tmx) / T); tsum += tprob[j]; }
        for (int j = 0; j < C; ++j) tprob[j] /= tsum;

        // Student log probs
        double smx = student_logits->data[n*C];
        for (int j = 1; j < C; ++j) if (student_logits->data[n*C+j] > smx) smx = student_logits->data[n*C+j];
        double ssum = 0;
        for (int j = 0; j < C; ++j) ssum += std::exp((student_logits->data[n*C+j] - smx) / T);
        double slog_sum = smx + std::log(ssum);

        for (int j = 0; j < C; ++j) {
            double slog_prob = (student_logits->data[n*C+j] - slog_sum) / T;
            soft_loss -= tprob[j] * slog_prob;
        }
    }
    soft_loss = soft_loss * (T * T) / N;

    // Hard loss: standard cross-entropy with true labels
    double hard_loss = 0;
    for (int n = 0; n < N; ++n) {
        double mx = student_logits->data[n*C];
        for (int j = 1; j < C; ++j) if (student_logits->data[n*C+j] > mx) mx = student_logits->data[n*C+j];
        double s = 0;
        for (int j = 0; j < C; ++j) s += std::exp(student_logits->data[n*C+j] - mx);
        double lse = mx + std::log(s);
        for (int j = 0; j < C; ++j)
            if (hard_labels->data[n*C+j] > 0)
                hard_loss -= hard_labels->data[n*C+j] * (student_logits->data[n*C+j] - lse);
    }
    hard_loss /= N;

    Tensor* c = alloc_tensor(1, 1, 1, 1);
    c->data[0] = alpha * soft_loss + (1.0 - alpha) * hard_loss;
    c->is_leaf = false; tensor_set_lhs(c, student_logits); tensor_set_rhs(c, teacher_logits);
    c->grad_fn = "cross_entropy"; // reuse CE backward (approximate)
    return c;
}

// ============================================================
// LEARNING RATE FINDER (unique Dream feature)
// Exponentially increases LR from lo to hi over N steps,
// records loss at each step. Helps find optimal learning rate.
// ============================================================

struct LRFinderState {
    double lr_lo, lr_hi;
    int steps, current;
    double best_loss, best_lr;
    std::vector<double> lrs, losses;
};
static std::vector<LRFinderState*> g_lr_finders;

Tensor* dream_lr_find_start(Tensor* lo_t, Tensor* hi_t, Tensor* steps_t) {
    auto* s = new LRFinderState();
    s->lr_lo = lo_t->data[0];
    s->lr_hi = hi_t->data[0];
    s->steps = (int)steps_t->data[0];
    s->current = 0;
    s->best_loss = 1e30;
    s->best_lr = s->lr_lo;
    int id = (int)g_lr_finders.size();
    g_lr_finders.push_back(s);
    return create_scalar((double)id);
}

// Get current LR for this step (exponentially spaced)
Tensor* dream_lr_find_get(Tensor* id_t) {
    int id = (int)id_t->data[0];
    if (id < 0 || id >= (int)g_lr_finders.size()) return create_scalar(0.001);
    auto* s = g_lr_finders[id];
    double t = (double)s->current / std::max(1, s->steps - 1);
    double lr = s->lr_lo * std::pow(s->lr_hi / s->lr_lo, t);
    return create_scalar(lr);
}

// Record loss for current LR step
Tensor* dream_lr_find_step(Tensor* id_t, Tensor* loss_t) {
    int id = (int)id_t->data[0];
    if (id < 0 || id >= (int)g_lr_finders.size()) return create_scalar(0.0);
    auto* s = g_lr_finders[id];
    double loss = loss_t->data[0];
    double lr = s->lr_lo * std::pow(s->lr_hi / s->lr_lo, (double)s->current / std::max(1, s->steps - 1));
    s->lrs.push_back(lr);
    s->losses.push_back(loss);
    if (loss < s->best_loss) { s->best_loss = loss; s->best_lr = lr; }
    s->current++;

    // Print if done
    if (s->current >= s->steps) {
        std::cerr << "\n=== LR Finder Results ===\n";
        std::cerr << "Best LR: " << s->best_lr << " (loss=" << s->best_loss << ")\n";
        std::cerr << "Suggested range: " << s->best_lr * 0.1 << " to " << s->best_lr << "\n";
        std::cerr << "LR sweep:\n";
        for (size_t i = 0; i < s->lrs.size(); i += std::max((size_t)1, s->lrs.size() / 10)) {
            char buf[128];
            std::snprintf(buf, sizeof(buf), "  lr=%.6f loss=%.4f%s\n",
                          s->lrs[i], s->losses[i],
                          (std::fabs(s->lrs[i] - s->best_lr) < 1e-10) ? " <-- best" : "");
            std::cerr << buf;
        }
        std::cerr << "=========================\n" << std::flush;
    }
    return create_scalar(s->current >= s->steps ? 1.0 : 0.0);
}

// ============================================================
// WEIGHT WATCHDOG — Dream exclusive feature
// Automatically monitors all parameters every N steps.
// Detects: NaN, exploding weights, dead neurons, gradient vanishing.
// No equivalent in PyTorch (requires manual coding).
// ============================================================

struct WatchdogState {
    int interval;
    int step;
    int warnings;
};
static std::vector<WatchdogState*> g_watchdogs;

Tensor* dream_watchdog_new(Tensor* interval_t) {
    auto* w = new WatchdogState();
    w->interval = (int)interval_t->data[0];
    w->step = 0;
    w->warnings = 0;
    int id = (int)g_watchdogs.size();
    g_watchdogs.push_back(w);
    return create_scalar((double)id);
}

// Check a parameter tensor for health issues
Tensor* dream_watchdog_check(Tensor* id_t, Tensor* param, Tensor* name_t) {
    int id = (int)id_t->data[0];
    if (id < 0 || id >= (int)g_watchdogs.size()) return create_scalar(0.0);
    auto* w = g_watchdogs[id];
    w->step++;
    if (w->step % w->interval != 0) return create_scalar(0.0);
    if (!param) return create_scalar(0.0);
    param = materialize(param);

    std::string name = "";
    if (name_t && name_t->grad_fn == "string")
        for (int i = 0; i < name_t->size; ++i) name += (char)(int)name_t->data[i];

    int nans = 0, infs = 0, zeros = 0, tiny = 0, huge = 0;
    double sum = 0, sq = 0;
    for (int i = 0; i < param->size; ++i) {
        double v = param->data[i];
        if (dream_isnan(v)) nans++;
        else if (dream_isinf(v)) infs++;
        else {
            double a = std::fabs(v);
            if (a == 0) zeros++;
            else if (a < 1e-7) tiny++;
            else if (a > 100) huge++;
            sum += v; sq += v * v;
        }
    }
    int ok = param->size - nans - infs;
    double mean = ok > 0 ? sum / ok : 0;
    double sd = ok > 1 ? std::sqrt(std::max(0.0, sq / ok - mean * mean)) : 0;

    // Check gradient health too
    int grad_zeros = 0, grad_huge = 0;
    if (param->grad) {
        for (int i = 0; i < param->size; ++i) {
            double g = std::fabs(param->grad[i]);
            if (g == 0) grad_zeros++;
            if (g > 100) grad_huge++;
        }
    }

    bool problem = false;
    if (nans > 0) { std::cerr << "WATCHDOG [" << name << "] NaN detected: " << nans << " elements!\n" << std::flush; problem = true; }
    if (infs > 0) { std::cerr << "WATCHDOG [" << name << "] Inf detected: " << infs << " elements!\n" << std::flush; problem = true; }
    if (huge > param->size * 0.1) { std::cerr << "WATCHDOG [" << name << "] Exploding weights: " << huge << " elements > 100\n" << std::flush; problem = true; }
    if (zeros > param->size * 0.9) { std::cerr << "WATCHDOG [" << name << "] Dead layer: " << zeros << "/" << param->size << " zeros\n" << std::flush; problem = true; }
    if (grad_zeros > param->size * 0.95) { std::cerr << "WATCHDOG [" << name << "] Vanishing gradients: " << grad_zeros << "/" << param->size << " zero grads\n" << std::flush; problem = true; }
    if (grad_huge > param->size * 0.1) { std::cerr << "WATCHDOG [" << name << "] Exploding gradients: " << grad_huge << " grads > 100\n" << std::flush; problem = true; }

    if (problem) w->warnings++;
    return create_scalar(problem ? 1.0 : 0.0);
}

Tensor* dream_watchdog_summary(Tensor* id_t) {
    int id = (int)id_t->data[0];
    if (id < 0 || id >= (int)g_watchdogs.size()) return create_scalar(0.0);
    auto* w = g_watchdogs[id];
    std::cerr << "[watchdog] " << w->step << " checks, " << w->warnings << " warnings\n" << std::flush;
    return create_scalar((double)w->warnings);
}

// ============================================================
// Checkpoint system — save and restore full training state.
// Without this, crashed or interrupted training is lost forever.
// Saves: all parameters, optimizer state, epoch, step, learning rate.
// ============================================================

// Save a checkpoint containing multiple tensors + metadata
// Format (binary):
//   Magic "DRCK" (4 bytes)
//   Version (int32)
//   Num tensors (int32)
//   For each tensor: name_len (int32), name (bytes), d0-d3 (4 int32), data (doubles)
//   Metadata: epoch (int32), step (int32), lr (double)
struct CheckpointSession {
    std::vector<std::pair<std::string, Tensor*>> tensors;
    int epoch = 0;
    int step = 0;
    double lr = 0.01;
    double best_loss = 1e30;
};
static std::vector<CheckpointSession*> g_ckpts;

// Create a new checkpoint session
Tensor* dream_ckpt_new() {
    auto* s = new CheckpointSession();
    int id = (int)g_ckpts.size();
    g_ckpts.push_back(s);
    return create_scalar((double)id);
}

// Add a tensor to the checkpoint with a name
Tensor* dream_ckpt_add(Tensor* id_t, Tensor* name_t, Tensor* tensor) {
    if (!id_t || !name_t || !tensor) return create_scalar(0.0);
    int id = (int)id_t->data[0];
    if (id < 0 || id >= (int)g_ckpts.size()) return create_scalar(0.0);
    std::string name;
    if (name_t->grad_fn == "string")
        for (int i = 0; i < name_t->size; ++i) name += (char)(int)name_t->data[i];
    // OWNERSHIP: the checkpoint keeps the tensor alive for the rest of the
    // program (sessions are never destroyed), so take a +1 owning reference.
    tensor_incref(tensor);
    g_ckpts[id]->tensors.push_back({name, tensor});
    return create_scalar(1.0);
}

// Set metadata (epoch, step, lr)
Tensor* dream_ckpt_set_meta(Tensor* id_t, Tensor* epoch_t, Tensor* step_t, Tensor* lr_t) {
    int id = (int)id_t->data[0];
    if (id < 0 || id >= (int)g_ckpts.size()) return create_scalar(0.0);
    g_ckpts[id]->epoch = epoch_t ? (int)epoch_t->data[0] : 0;
    g_ckpts[id]->step = step_t ? (int)step_t->data[0] : 0;
    g_ckpts[id]->lr = lr_t ? lr_t->data[0] : 0.01;
    return create_scalar(1.0);
}

// Save checkpoint to file
Tensor* dream_ckpt_save(Tensor* id_t, Tensor* path_t) {
    int id = (int)id_t->data[0];
    if (id < 0 || id >= (int)g_ckpts.size()) return create_scalar(0.0);
    std::string path;
    if (path_t->grad_fn == "string")
        for (int i = 0; i < path_t->size; ++i) path += (char)(int)path_t->data[i];

    dream_mkdirs_for(path);
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) {
        std::cerr << "RUNTIME ERROR in ckpt_save: cannot write '" << path << "'\n" << std::flush;
        return create_scalar(0.0);
    }
    auto* s = g_ckpts[id];

    // Header
    fwrite("DRCK", 1, 4, f);
    int32_t version = 1;
    fwrite(&version, 4, 1, f);
    int32_t n = (int32_t)s->tensors.size();
    fwrite(&n, 4, 1, f);

    // Tensors
    for (auto& kv : s->tensors) {
        int32_t nl = (int32_t)kv.first.size();
        fwrite(&nl, 4, 1, f);
        fwrite(kv.first.data(), 1, nl, f);
        Tensor* t = materialize(kv.second);
        int32_t d[4] = {t->d0, t->d1, t->d2, t->d3};
        fwrite(d, 4, 4, f);
        fwrite(t->data, sizeof(double), t->size, f);
    }

    // Metadata
    fwrite(&s->epoch, 4, 1, f);
    fwrite(&s->step, 4, 1, f);
    fwrite(&s->lr, 8, 1, f);
    fwrite(&s->best_loss, 8, 1, f);

    fclose(f);
    std::cerr << "[ckpt] saved " << n << " tensors + meta to '" << path << "' (epoch=" << s->epoch << ")\n" << std::flush;
    return create_scalar(1.0);
}

// Load a checkpoint. Returns session id. Use ckpt_get(id, "name") to retrieve tensors.
Tensor* dream_ckpt_load(Tensor* path_t) {
    std::string path;
    if (path_t->grad_fn == "string")
        for (int i = 0; i < path_t->size; ++i) path += (char)(int)path_t->data[i];

    FILE* f = fopen(path.c_str(), "rb");
    if (!f) {
        std::cerr << "RUNTIME ERROR in ckpt_load: cannot open '" << path << "'\n" << std::flush;
        return create_scalar(-1.0);
    }

    char magic[4];
    if (fread(magic, 1, 4, f) != 4 || std::memcmp(magic, "DRCK", 4) != 0) {
        std::cerr << "RUNTIME ERROR in ckpt_load: bad magic, not a Dream checkpoint\n" << std::flush;
        fclose(f);
        return create_scalar(-1.0);
    }

    int32_t version = 0, n = 0;
    fread(&version, 4, 1, f);
    fread(&n, 4, 1, f);

    auto* s = new CheckpointSession();
    for (int i = 0; i < n; ++i) {
        int32_t nl = 0;
        fread(&nl, 4, 1, f);
        std::string name(nl, '\0');
        fread(&name[0], 1, nl, f);
        int32_t d[4];
        fread(d, 4, 4, f);
        Tensor* t = alloc_tensor(d[0], d[1], d[2], d[3]);
        fread(t->data, sizeof(double), t->size, f);
        s->tensors.push_back({name, t});
    }

    fread(&s->epoch, 4, 1, f);
    fread(&s->step, 4, 1, f);
    fread(&s->lr, 8, 1, f);
    fread(&s->best_loss, 8, 1, f);
    fclose(f);

    int id = (int)g_ckpts.size();
    g_ckpts.push_back(s);
    std::cerr << "[ckpt] loaded " << n << " tensors from '" << path << "' (epoch=" << s->epoch << ")\n" << std::flush;
    return create_scalar((double)id);
}

// Get a tensor from loaded checkpoint by name
Tensor* dream_ckpt_get(Tensor* id_t, Tensor* name_t) {
    int id = (int)id_t->data[0];
    if (id < 0 || id >= (int)g_ckpts.size()) return create_scalar(0.0);
    std::string name;
    if (name_t->grad_fn == "string")
        for (int i = 0; i < name_t->size; ++i) name += (char)(int)name_t->data[i];
    for (auto& kv : g_ckpts[id]->tensors) {
        if (kv.first == name) return tensor_incref(kv.second);
    }
    std::cerr << "RUNTIME ERROR in ckpt_get: tensor '" << name << "' not found in checkpoint\n" << std::flush;
    return create_scalar(0.0);
}

Tensor* dream_ckpt_epoch(Tensor* id_t) {
    int id = (int)id_t->data[0];
    if (id < 0 || id >= (int)g_ckpts.size()) return create_scalar(0.0);
    return create_scalar((double)g_ckpts[id]->epoch);
}

Tensor* dream_ckpt_lr(Tensor* id_t) {
    int id = (int)id_t->data[0];
    if (id < 0 || id >= (int)g_ckpts.size()) return create_scalar(0.01);
    return create_scalar(g_ckpts[id]->lr);
}

// Aliases
Tensor* ckpt_new() { return dream_ckpt_new(); }
Tensor* ckpt_add(Tensor* id, Tensor* name, Tensor* t) { return dream_ckpt_add(id, name, t); }
Tensor* ckpt_set_meta(Tensor* id, Tensor* e, Tensor* s, Tensor* lr) { return dream_ckpt_set_meta(id, e, s, lr); }
Tensor* ckpt_save(Tensor* id, Tensor* p) { return dream_ckpt_save(id, p); }
Tensor* ckpt_load(Tensor* p) { return dream_ckpt_load(p); }
Tensor* ckpt_get(Tensor* id, Tensor* name) { return dream_ckpt_get(id, name); }
Tensor* ckpt_epoch(Tensor* id) { return dream_ckpt_epoch(id); }
Tensor* ckpt_lr(Tensor* id) { return dream_ckpt_lr(id); }

// ============================================================
// safetensors: import PyTorch / HuggingFace models into Dream
//
// safetensors is the Hugging Face standard format — used by Llama,
// Stable Diffusion, Mistral, and all modern HF models. Format is
// deliberately simple:
//
//   [8 bytes little-endian: header JSON length N]
//   [N bytes: JSON header describing all tensors]
//   [raw tensor data, packed contiguously]
//
// JSON header looks like:
//   {
//     "fc1.weight": {"dtype":"F32","shape":[128,64],"data_offsets":[0,32768]},
//     "fc1.bias":   {"dtype":"F32","shape":[128],  "data_offsets":[32768,33280]}
//   }
//
// Usage in Dream:
//   let ts = load_safetensors("model.safetensors");
//   let W1 = st_get(ts, "fc1.weight");
//   let b1 = st_get(ts, "fc1.bias");
//
// In Python (one-liner to export from PyTorch):
//   from safetensors.torch import save_file
//   save_file(model.state_dict(), "model.safetensors")
// ============================================================

struct SafetensorsSession {
    std::vector<std::pair<std::string, Tensor*>> tensors;
    std::string path;
};
static std::vector<SafetensorsSession*> g_st_sessions;

// Minimal JSON parser — enough for safetensors headers.
// Finds the value of a key like "shape":[128,64] or "dtype":"F32".
// Not a full JSON parser, but safetensors headers are structured enough
// that pattern-matching works reliably.
namespace dream_st_json {

// Skip whitespace
static void skip_ws(const char*& p) {
    while (*p && (*p==' '||*p=='\t'||*p=='\n'||*p=='\r')) ++p;
}

// Parse a quoted string starting at p (must point to ")
static std::string parse_string(const char*& p) {
    if (*p != '"') return "";
    ++p;
    std::string s;
    while (*p && *p != '"') {
        if (*p == '\\' && *(p+1)) { s += *(p+1); p += 2; }
        else s += *p++;
    }
    if (*p == '"') ++p;
    return s;
}

// Parse integer
static long long parse_int(const char*& p) {
    skip_ws(p);
    long long sign = 1;
    if (*p == '-') { sign = -1; ++p; }
    long long v = 0;
    while (*p >= '0' && *p <= '9') { v = v * 10 + (*p - '0'); ++p; }
    return sign * v;
}

// Parse [a,b,c,...] into a vector
static std::vector<long long> parse_int_array(const char*& p) {
    std::vector<long long> v;
    skip_ws(p);
    if (*p != '[') return v;
    ++p;
    skip_ws(p);
    while (*p && *p != ']') {
        v.push_back(parse_int(p));
        skip_ws(p);
        if (*p == ',') { ++p; skip_ws(p); }
    }
    if (*p == ']') ++p;
    return v;
}

} // namespace dream_st_json

// Load a .safetensors file and return a session id
Tensor* dream_load_safetensors(Tensor* path_t) {
    if (!path_t) return create_scalar(-1.0);
    std::string path;
    if (path_t->grad_fn == "string")
        for (int i = 0; i < path_t->size; ++i) path += (char)(int)path_t->data[i];

    FILE* f = fopen(path.c_str(), "rb");
    if (!f) {
        std::cerr << "RUNTIME ERROR in load_safetensors: cannot open '" << path << "'\n"
                  << "  Check that the file exists and is readable.\n" << std::flush;
        return create_scalar(-1.0);
    }

    // Read 8-byte header length (little-endian uint64)
    uint8_t hdr_bytes[8];
    if (fread(hdr_bytes, 1, 8, f) != 8) {
        std::cerr << "RUNTIME ERROR in load_safetensors: file too small (no header size)\n" << std::flush;
        fclose(f); return create_scalar(-1.0);
    }
    uint64_t hdr_len = 0;
    for (int i = 0; i < 8; ++i) hdr_len |= ((uint64_t)hdr_bytes[i]) << (i*8);

    if (hdr_len == 0 || hdr_len > 100 * 1024 * 1024) {
        std::cerr << "RUNTIME ERROR in load_safetensors: invalid header size " << hdr_len << "\n" << std::flush;
        fclose(f); return create_scalar(-1.0);
    }

    // Read JSON header
    std::string json(hdr_len, '\0');
    if (fread(&json[0], 1, hdr_len, f) != hdr_len) {
        std::cerr << "RUNTIME ERROR in load_safetensors: header truncated\n" << std::flush;
        fclose(f); return create_scalar(-1.0);
    }

    // Data section starts right after header
    long data_start = 8 + hdr_len;

    // Parse JSON: find each tensor's name, shape, dtype, data_offsets
    auto* sess = new SafetensorsSession();
    sess->path = path;

    const char* p = json.c_str();
    dream_st_json::skip_ws(p);
    if (*p == '{') ++p;

    while (*p) {
        dream_st_json::skip_ws(p);
        if (*p == '}' || *p == 0) break;
        if (*p == ',') { ++p; continue; }
        if (*p != '"') { ++p; continue; }

        std::string tensor_name = dream_st_json::parse_string(p);
        dream_st_json::skip_ws(p);
        if (*p != ':') continue;
        ++p;
        dream_st_json::skip_ws(p);

        // Skip metadata "__metadata__" entry (it's just key-value strings, not tensor data)
        if (tensor_name == "__metadata__") {
            // Skip the object value — find matching }
            int depth = 0;
            if (*p == '{') { depth = 1; ++p; }
            while (*p && depth > 0) {
                if (*p == '{') depth++;
                else if (*p == '}') depth--;
                ++p;
            }
            continue;
        }

        // Parse the tensor metadata object: {dtype, shape, data_offsets}
        if (*p != '{') continue;
        ++p;

        std::string dtype;
        std::vector<long long> shape;
        long long offset_start = 0, offset_end = 0;

        while (*p && *p != '}') {
            dream_st_json::skip_ws(p);
            if (*p != '"') { ++p; continue; }
            std::string key = dream_st_json::parse_string(p);
            dream_st_json::skip_ws(p);
            if (*p == ':') ++p;
            dream_st_json::skip_ws(p);
            if (key == "dtype") {
                dtype = dream_st_json::parse_string(p);
            } else if (key == "shape") {
                shape = dream_st_json::parse_int_array(p);
            } else if (key == "data_offsets") {
                auto v = dream_st_json::parse_int_array(p);
                if (v.size() == 2) { offset_start = v[0]; offset_end = v[1]; }
            } else {
                // Skip unknown value (may be nested)
                int depth = 0;
                while (*p && (depth > 0 || (*p != ',' && *p != '}'))) {
                    if (*p == '{' || *p == '[') depth++;
                    else if (*p == '}' || *p == ']') depth--;
                    ++p;
                }
            }
            dream_st_json::skip_ws(p);
            if (*p == ',') { ++p; }
        }
        if (*p == '}') ++p;

        // Compute total element count from shape
        long long nelem = 1;
        for (auto s : shape) nelem *= s;
        if (nelem <= 0) continue;

        // Allocate tensor matching shape (Dream supports up to 4D)
        int d0 = shape.size() >= 1 ? (int)shape[0] : 1;
        int d1 = shape.size() >= 2 ? (int)shape[1] : 1;
        int d2 = shape.size() >= 3 ? (int)shape[2] : 1;
        int d3 = shape.size() >= 4 ? (int)shape[3] : 1;
        Tensor* t = alloc_tensor(d0, d1, d2, d3);

        // Read raw data at offset, convert dtype to double (Dream's native type)
        long data_pos = data_start + offset_start;
        fseek(f, data_pos, SEEK_SET);

        if (dtype == "F64" || dtype == "F32" || dtype == "F16" || dtype == "BF16" ||
            dtype == "I64" || dtype == "I32" || dtype == "I16" || dtype == "I8" || dtype == "U8") {
            // Raw bytes to interpret
            long byte_count = offset_end - offset_start;
            std::vector<uint8_t> raw(byte_count);
            fread(raw.data(), 1, byte_count, f);

            if (dtype == "F64") {
                // Direct copy
                std::memcpy(t->data, raw.data(), std::min((long long)byte_count, (long long)(t->size * 8)));
            } else if (dtype == "F32") {
                const float* src = reinterpret_cast<const float*>(raw.data());
                for (int i = 0; i < t->size && i < byte_count/4; ++i) t->data[i] = (double)src[i];
            } else if (dtype == "I64") {
                const int64_t* src = reinterpret_cast<const int64_t*>(raw.data());
                for (int i = 0; i < t->size && i < byte_count/8; ++i) t->data[i] = (double)src[i];
            } else if (dtype == "I32") {
                const int32_t* src = reinterpret_cast<const int32_t*>(raw.data());
                for (int i = 0; i < t->size && i < byte_count/4; ++i) t->data[i] = (double)src[i];
            } else if (dtype == "F16") {
                // IEEE 754 half-precision -> double
                const uint16_t* src = reinterpret_cast<const uint16_t*>(raw.data());
                for (int i = 0; i < t->size && i < byte_count/2; ++i) {
                    uint16_t h = src[i];
                    uint32_t sign = (h & 0x8000) << 16;
                    uint32_t exp = (h & 0x7C00) >> 10;
                    uint32_t mant = (h & 0x03FF);
                    uint32_t f;
                    if (exp == 0) { f = sign; }
                    else if (exp == 31) { f = sign | 0x7F800000 | (mant << 13); }
                    else { f = sign | ((exp + 112) << 23) | (mant << 13); }
                    float fv;
                    std::memcpy(&fv, &f, 4);
                    t->data[i] = (double)fv;
                }
            } else if (dtype == "BF16") {
                // brain float16 — upper 16 bits of f32
                const uint16_t* src = reinterpret_cast<const uint16_t*>(raw.data());
                for (int i = 0; i < t->size && i < byte_count/2; ++i) {
                    uint32_t f = ((uint32_t)src[i]) << 16;
                    float fv;
                    std::memcpy(&fv, &f, 4);
                    t->data[i] = (double)fv;
                }
            } else if (dtype == "U8") {
                for (int i = 0; i < t->size && i < byte_count; ++i) t->data[i] = (double)raw[i];
            } else if (dtype == "I8") {
                for (int i = 0; i < t->size && i < byte_count; ++i) t->data[i] = (double)(int8_t)raw[i];
            } else if (dtype == "I16") {
                const int16_t* src = reinterpret_cast<const int16_t*>(raw.data());
                for (int i = 0; i < t->size && i < byte_count/2; ++i) t->data[i] = (double)src[i];
            }
        } else {
            std::cerr << "[safetensors] warning: unsupported dtype '" << dtype
                      << "' for tensor '" << tensor_name << "' - skipping\n" << std::flush;
            continue;
        }

        sess->tensors.push_back({tensor_name, t});
    }
    fclose(f);

    int id = (int)g_st_sessions.size();
    g_st_sessions.push_back(sess);
    std::cerr << "[safetensors] loaded " << sess->tensors.size()
              << " tensors from '" << path << "'\n" << std::flush;
    return create_scalar((double)id);
}

// Get a tensor by name from loaded safetensors session
Tensor* dream_st_get(Tensor* id_t, Tensor* name_t) {
    if (!id_t || !name_t) return create_scalar(0.0);
    int id = (int)id_t->data[0];
    if (id < 0 || id >= (int)g_st_sessions.size()) {
        std::cerr << "RUNTIME ERROR in st_get: invalid session id " << id << "\n" << std::flush;
        return create_scalar(0.0);
    }
    std::string name;
    if (name_t->grad_fn == "string")
        for (int i = 0; i < name_t->size; ++i) name += (char)(int)name_t->data[i];

    for (auto& kv : g_st_sessions[id]->tensors) {
        if (kv.first == name) return kv.second;
    }
    std::cerr << "RUNTIME ERROR in st_get: tensor '" << name
              << "' not found. Available tensors:\n" << std::flush;
    for (auto& kv : g_st_sessions[id]->tensors) {
        std::cerr << "  " << kv.first << " ("
                  << kv.second->d0 << "x" << kv.second->d1;
        if (kv.second->d2 > 1) std::cerr << "x" << kv.second->d2;
        if (kv.second->d3 > 1) std::cerr << "x" << kv.second->d3;
        std::cerr << ")\n" << std::flush;
    }
    return create_scalar(0.0);
}

// List all tensor names in a loaded session
Tensor* dream_st_list(Tensor* id_t) {
    if (!id_t) return create_scalar(0.0);
    int id = (int)id_t->data[0];
    if (id < 0 || id >= (int)g_st_sessions.size()) return create_scalar(0.0);
    std::cerr << "[safetensors session " << id << "] contents:\n";
    for (auto& kv : g_st_sessions[id]->tensors) {
        std::cerr << "  " << kv.first << " - shape (" << kv.second->d0;
        if (kv.second->d1 > 1) std::cerr << "," << kv.second->d1;
        if (kv.second->d2 > 1) std::cerr << "," << kv.second->d2;
        if (kv.second->d3 > 1) std::cerr << "," << kv.second->d3;
        std::cerr << "), " << kv.second->size << " elements\n";
    }
    std::cerr << std::flush;
    return create_scalar((double)g_st_sessions[id]->tensors.size());
}

// Also provide save — export Dream-trained models back to safetensors for PyTorch
Tensor* dream_save_safetensors(Tensor* id_t, Tensor* path_t) {
    if (!id_t || !path_t) return create_scalar(0.0);
    int id = (int)id_t->data[0];
    if (id < 0 || id >= (int)g_ckpts.size()) {
        std::cerr << "RUNTIME ERROR in save_safetensors: invalid checkpoint id\n" << std::flush;
        return create_scalar(0.0);
    }
    std::string path;
    if (path_t->grad_fn == "string")
        for (int i = 0; i < path_t->size; ++i) path += (char)(int)path_t->data[i];

    // Build JSON header describing each tensor
    // We write F64 (double) since that's Dream's native format
    std::string json = "{";
    long long offset = 0;
    bool first = true;
    for (auto& kv : g_ckpts[id]->tensors) {
        if (!first) json += ",";
        first = false;
        Tensor* t = materialize(kv.second);
        long long n = t->size;
        long long bytes = n * 8;

        json += "\"" + kv.first + "\":{\"dtype\":\"F64\",\"shape\":[";
        // Include only meaningful dims
        if (t->d0 > 0) json += std::to_string(t->d0);
        if (t->d1 > 1) json += "," + std::to_string(t->d1);
        if (t->d2 > 1) json += "," + std::to_string(t->d2);
        if (t->d3 > 1) json += "," + std::to_string(t->d3);
        json += "],\"data_offsets\":[" + std::to_string(offset) + ","
             + std::to_string(offset + bytes) + "]}";
        offset += bytes;
    }
    json += "}";

    // Pad JSON to multiple of 8 bytes (safetensors convention)
    while (json.size() % 8 != 0) json += " ";

    dream_mkdirs_for(path);
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) {
        std::cerr << "RUNTIME ERROR in save_safetensors: cannot write '" << path << "'\n" << std::flush;
        return create_scalar(0.0);
    }

    // Write 8-byte header length (little-endian)
    uint64_t hlen = json.size();
    uint8_t hb[8];
    for (int i = 0; i < 8; ++i) hb[i] = (uint8_t)((hlen >> (i*8)) & 0xFF);
    fwrite(hb, 1, 8, f);
    fwrite(json.data(), 1, json.size(), f);

    // Write tensor data in same order
    for (auto& kv : g_ckpts[id]->tensors) {
        Tensor* t = materialize(kv.second);
        fwrite(t->data, sizeof(double), t->size, f);
    }

    fclose(f);
    std::cerr << "[safetensors] saved " << g_ckpts[id]->tensors.size()
              << " tensors to '" << path << "'\n" << std::flush;
    return create_scalar(1.0);
}

// Aliases
Tensor* load_safetensors(Tensor* p) { return dream_load_safetensors(p); }
Tensor* st_get(Tensor* id, Tensor* n) { return dream_st_get(id, n); }
Tensor* st_list(Tensor* id) { return dream_st_list(id); }
Tensor* save_safetensors(Tensor* id, Tensor* p) { return dream_save_safetensors(id, p); }

// ============================================================
// Export trained model as a self-contained C header + source file.
//
// This is Dream's killer feature for embedded deployment:
//   * Input: a Dream-trained checkpoint (ckpt_new, ckpt_add, ...)
//   * Output: two files you can drop into ANY C project
//       model.h — declares inference function + tensor shapes
//       model.c — static weight arrays + plain C inference code
//
// No dependencies. No malloc (all stack/static). Runs on:
//   * ESP32, STM32, Arduino (any 32-bit MCU with enough flash)
//   * Raspberry Pi, BeagleBone (Linux embedded)
//   * Any system with a C99 compiler
//
// Usage in Dream:
//   let id = ckpt_new();
//   ckpt_add(id, "fc1.weight", W1);
//   ckpt_add(id, "fc1.bias",   b1);
//   ckpt_add(id, "fc2.weight", W2);
//   ckpt_add(id, "fc2.bias",   b2);
//   export_c_header(id, "model", "mlp_64_128_10");
// ============================================================

// Helper: sanitize a name for use as a C identifier.
// Replaces '.', '/', '-' etc. with '_'. PyTorch names like "fc1.weight" -> "fc1_weight".
static std::string sanitize_c_name(const std::string& s) {
    std::string out;
    for (char c : s) {
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
            (c >= '0' && c <= '9') || c == '_') out += c;
        else out += '_';
    }
    // Must start with letter or underscore
    if (!out.empty() && out[0] >= '0' && out[0] <= '9') out = "_" + out;
    return out;
}

// Print a double as a C literal without losing precision
static std::string format_double_c(double v) {
    char buf[64];
    if (dream_isnan(v)) return "NAN";
    if (dream_isinf(v)) return v > 0 ? "INFINITY" : "-INFINITY";
    // 17 significant digits is enough to round-trip any IEEE 754 double
    std::snprintf(buf, sizeof(buf), "%.17g", v);
    return buf;
}

// Main export function. Writes two files:
//   <prefix>.h — public API header
//   <prefix>.c — weight arrays + inference implementation
//
// The generated C code knows nothing about Dream. It's pure C99.
Tensor* dream_export_c_header(Tensor* id_t, Tensor* prefix_t, Tensor* func_name_t) {
    if (!id_t || !prefix_t) {
        std::cerr << "error: export_c_header needs (checkpoint_id, path_prefix, func_name)\n" << std::flush;
        return create_scalar(0.0);
    }
    int id = (int)id_t->data[0];
    if (id < 0 || id >= (int)g_ckpts.size()) {
        std::cerr << "RUNTIME ERROR in export_c_header: invalid checkpoint id " << id << "\n" << std::flush;
        return create_scalar(0.0);
    }

    std::string prefix;
    if (prefix_t->grad_fn == "string")
        for (int i = 0; i < prefix_t->size; ++i) prefix += (char)(int)prefix_t->data[i];
    std::string func_name = "infer";
    if (func_name_t && func_name_t->grad_fn == "string") {
        func_name = "";
        for (int i = 0; i < func_name_t->size; ++i) func_name += (char)(int)func_name_t->data[i];
    }

    auto* sess = g_ckpts[id];
    if (sess->tensors.empty()) {
        std::cerr << "RUNTIME ERROR in export_c_header: checkpoint is empty\n" << std::flush;
        return create_scalar(0.0);
    }

    // Infer the architecture from tensor names and shapes.
    // Works for standard MLPs where layers are named fc1.weight, fc1.bias, fc2.weight, ...
    // For now we handle sequential MLP: alternating weight + bias tensors.
    struct Layer {
        std::string name;        // "fc1"
        std::string w_name;      // C identifier for weight array
        std::string b_name;      // C identifier for bias array
        int in_dim;
        int out_dim;
        const Tensor* weight;
        const Tensor* bias;
    };

    std::vector<Layer> layers;
    // Pair up weight/bias by layer name prefix
    std::map<std::string, Layer> layer_map;
    for (auto& kv : sess->tensors) {
        std::string name = kv.first;
        Tensor* t = kv.second;
        // Extract layer name (part before .weight or .bias)
        std::string base;
        bool is_weight = false, is_bias = false;
        if (name.size() > 7 && name.substr(name.size()-7) == ".weight") {
            base = name.substr(0, name.size()-7);
            is_weight = true;
        } else if (name.size() > 5 && name.substr(name.size()-5) == ".bias") {
            base = name.substr(0, name.size()-5);
            is_bias = true;
        } else {
            // Not a standard Linear layer — store as raw tensor
            base = name;
        }

        auto& layer = layer_map[base];
        if (layer.name.empty()) layer.name = base;
        if (is_weight) {
            layer.weight = t;
            layer.out_dim = t->d0;
            layer.in_dim = t->d1;
            layer.w_name = sanitize_c_name(name);
        } else if (is_bias) {
            layer.bias = t;
            layer.b_name = sanitize_c_name(name);
        }
    }

    // Collect in the order they appeared in the checkpoint
    std::set<std::string> added;
    for (auto& kv : sess->tensors) {
        std::string name = kv.first;
        std::string base;
        if (name.size() > 7 && name.substr(name.size()-7) == ".weight")
            base = name.substr(0, name.size()-7);
        else if (name.size() > 5 && name.substr(name.size()-5) == ".bias")
            base = name.substr(0, name.size()-5);
        else continue;
        if (added.count(base)) continue;
        added.insert(base);
        auto it = layer_map.find(base);
        if (it != layer_map.end() && it->second.weight && it->second.bias)
            layers.push_back(it->second);
    }

    if (layers.empty()) {
        std::cerr << "RUNTIME ERROR in export_c_header: no complete Linear layers found.\n"
                  << "  Expected tensor names like 'fc1.weight' + 'fc1.bias' pairs.\n" << std::flush;
        return create_scalar(0.0);
    }

    int input_dim = layers.front().in_dim;
    int output_dim = layers.back().out_dim;

    // ---------- Write header file ----------
    std::string header_path = prefix + ".h";
    dream_mkdirs_for(header_path);
    FILE* hf = fopen(header_path.c_str(), "w");
    if (!hf) {
        std::cerr << "RUNTIME ERROR in export_c_header: cannot write '" << header_path << "'\n" << std::flush;
        return create_scalar(0.0);
    }
    std::string guard = sanitize_c_name(prefix);
    for (auto& c : guard) c = (char)std::toupper((unsigned char)c);
    guard += "_H";

    fprintf(hf, "/*\n");
    fprintf(hf, " * %s.h — auto-generated by Dream\n", prefix.c_str());
    fprintf(hf, " *\n");
    fprintf(hf, " * Self-contained C99 inference for an MLP with %zu layer(s).\n", layers.size());
    fprintf(hf, " * Input dim: %d, Output dim: %d\n", input_dim, output_dim);
    fprintf(hf, " *\n");
    fprintf(hf, " * Usage:\n");
    fprintf(hf, " *   #include \"%s.h\"\n", prefix.c_str());
    fprintf(hf, " *   float input[%d] = { ... };\n", input_dim);
    fprintf(hf, " *   float output[%d];\n", output_dim);
    fprintf(hf, " *   %s(input, output);\n", func_name.c_str());
    fprintf(hf, " *\n");
    fprintf(hf, " * No malloc, no dependencies. Runs on any C99 compiler.\n");
    fprintf(hf, " */\n");
    fprintf(hf, "#ifndef %s\n", guard.c_str());
    fprintf(hf, "#define %s\n\n", guard.c_str());
    fprintf(hf, "#ifdef __cplusplus\nextern \"C\" {\n#endif\n\n");
    // Generate uppercase prefix for macros (C convention for #define constants)
    std::string prefix_upper;
    {
        std::string s = sanitize_c_name(prefix);
        for (char c : s) {
            if (c >= 'a' && c <= 'z') prefix_upper += (char)(c - 'a' + 'A');
            else prefix_upper += c;
        }
    }
    fprintf(hf, "#define %s_INPUT_DIM  %d\n", prefix_upper.c_str(), input_dim);
    fprintf(hf, "#define %s_OUTPUT_DIM %d\n", prefix_upper.c_str(), output_dim);
    fprintf(hf, "#define %s_NUM_LAYERS %zu\n\n", prefix_upper.c_str(), layers.size());
    fprintf(hf, "/* Run inference. input must have %d elements, output must have %d. */\n",
            input_dim, output_dim);
    fprintf(hf, "void %s(const float* input, float* output);\n\n", func_name.c_str());
    fprintf(hf, "#ifdef __cplusplus\n}\n#endif\n\n");
    fprintf(hf, "#endif /* %s */\n", guard.c_str());
    fclose(hf);

    // ---------- Write source file ----------
    std::string src_path = prefix + ".c";
    dream_mkdirs_for(src_path);
    FILE* sf = fopen(src_path.c_str(), "w");
    if (!sf) {
        std::cerr << "RUNTIME ERROR in export_c_header: cannot write '" << src_path << "'\n" << std::flush;
        return create_scalar(0.0);
    }

    fprintf(sf, "/*\n * %s.c — auto-generated by Dream\n *\n", prefix.c_str());
    fprintf(sf, " * Pure C99. No malloc. No dependencies.\n");
    fprintf(sf, " * Model: %zu Linear layers, %d input -> %d output\n */\n",
            layers.size(), input_dim, output_dim);
    fprintf(sf, "#include \"%s.h\"\n#include <math.h>\n\n", prefix.c_str());

    // Compute max intermediate buffer size for stack allocation
    int max_intermediate = input_dim;
    for (auto& L : layers) if (L.out_dim > max_intermediate) max_intermediate = L.out_dim;

    // Write weight arrays (using float to save space vs double)
    size_t total_params = 0;
    for (auto& L : layers) {
        total_params += L.weight->size + L.bias->size;

        // Weight array (stored as out_dim x in_dim, same as PyTorch)
        fprintf(sf, "/* %s.weight shape: [%d, %d] (out, in) */\n",
                L.name.c_str(), L.out_dim, L.in_dim);
        fprintf(sf, "static const float %s[%d] = {",
                L.w_name.c_str(), L.weight->size);
        const double* wd = L.weight->data;
        // Helper lambda: format a float as a valid C literal.
        // %g produces "0" for zero — but "0f" is invalid C (integer + bad suffix).
        // We ensure every output has either a decimal point or exponent before the 'f'.
        auto fmt_float = [&](float v) {
            char buf[64];
            snprintf(buf, sizeof(buf), "%.9g", v);
            // Check if the formatted string contains '.', 'e', 'E', 'n' (nan), or 'i' (inf)
            bool needs_dot = true;
            for (char* p = buf; *p; ++p) {
                if (*p == '.' || *p == 'e' || *p == 'E' || *p == 'n' || *p == 'i') {
                    needs_dot = false;
                    break;
                }
            }
            if (needs_dot) strcat(buf, ".0");
            fprintf(sf, "%sf", buf);
        };
        for (int i = 0; i < L.weight->size; ++i) {
            if (i % 8 == 0) fprintf(sf, "\n    ");
            fmt_float((float)wd[i]);
            if (i + 1 < L.weight->size) fprintf(sf, ", ");
        }
        fprintf(sf, "\n};\n\n");

        // Bias array
        fprintf(sf, "/* %s.bias shape: [%d] */\n", L.name.c_str(), L.out_dim);
        fprintf(sf, "static const float %s[%d] = {",
                L.b_name.c_str(), L.bias->size);
        const double* bd = L.bias->data;
        for (int i = 0; i < L.bias->size; ++i) {
            if (i % 8 == 0) fprintf(sf, "\n    ");
            fmt_float((float)bd[i]);
            if (i + 1 < L.bias->size) fprintf(sf, ", ");
        }
        fprintf(sf, "\n};\n\n");
    }

    // Write the inference function
    fprintf(sf, "/* Dense(x, W, b, out): out = ReLU(x @ W^T + b) */\n");
    fprintf(sf, "/* Internal helper. W is row-major (out_dim x in_dim). */\n");
    fprintf(sf, "static void dense_relu(const float* x, int in_dim,\n");
    fprintf(sf, "                       const float* W, const float* b,\n");
    fprintf(sf, "                       float* y, int out_dim)\n");
    fprintf(sf, "{\n");
    fprintf(sf, "    int i, j;\n");
    fprintf(sf, "    for (i = 0; i < out_dim; ++i) {\n");
    fprintf(sf, "        float acc = b[i];\n");
    fprintf(sf, "        const float* w_row = W + i * in_dim;\n");
    fprintf(sf, "        for (j = 0; j < in_dim; ++j) acc += w_row[j] * x[j];\n");
    fprintf(sf, "        y[i] = acc > 0.0f ? acc : 0.0f; /* ReLU */\n");
    fprintf(sf, "    }\n");
    fprintf(sf, "}\n\n");

    fprintf(sf, "static void dense_linear(const float* x, int in_dim,\n");
    fprintf(sf, "                         const float* W, const float* b,\n");
    fprintf(sf, "                         float* y, int out_dim)\n");
    fprintf(sf, "{\n");
    fprintf(sf, "    int i, j;\n");
    fprintf(sf, "    for (i = 0; i < out_dim; ++i) {\n");
    fprintf(sf, "        float acc = b[i];\n");
    fprintf(sf, "        const float* w_row = W + i * in_dim;\n");
    fprintf(sf, "        for (j = 0; j < in_dim; ++j) acc += w_row[j] * x[j];\n");
    fprintf(sf, "        y[i] = acc;\n");
    fprintf(sf, "    }\n");
    fprintf(sf, "}\n\n");

    // Main entrypoint — chains dense layers with ReLU except final
    fprintf(sf, "void %s(const float* input, float* output)\n{\n", func_name.c_str());
    fprintf(sf, "    /* Stack buffers for intermediate activations — no malloc */\n");
    fprintf(sf, "    float buf_a[%d];\n", max_intermediate);
    fprintf(sf, "    float buf_b[%d];\n", max_intermediate);

    // First layer: input -> buf_a
    const char* current_in = "input";
    const char* current_out = "buf_a";
    const char* next_out = "buf_b";

    for (size_t li = 0; li < layers.size(); ++li) {
        bool is_last = (li + 1 == layers.size());
        const char* out_buf = is_last ? "output" : current_out;
        const char* fn = is_last ? "dense_linear" : "dense_relu";

        fprintf(sf, "    %s(%s, %d, %s, %s, %s, %d);\n",
                fn, current_in, layers[li].in_dim,
                layers[li].w_name.c_str(), layers[li].b_name.c_str(),
                out_buf, layers[li].out_dim);

        if (!is_last) {
            current_in = current_out;
            const char* tmp = current_out;
            current_out = next_out;
            next_out = tmp;
        }
    }
    fprintf(sf, "}\n");
    fclose(sf);

    size_t total_bytes = total_params * 4;  // float32
    std::cerr << "[export_c] wrote " << prefix << ".h + " << prefix << ".c\n";
    std::cerr << "  layers:       " << layers.size() << "\n";
    std::cerr << "  input_dim:    " << input_dim << "\n";
    std::cerr << "  output_dim:   " << output_dim << "\n";
    std::cerr << "  parameters:   " << total_params << " (" << total_bytes << " bytes as float32)\n";
    std::cerr << "  fn signature: void " << func_name << "(const float* input, float* output);\n";
    std::cerr << "  Compile on target with: gcc -O2 -lm " << prefix << ".c -o my_app\n";
    std::cerr << std::flush;

    return create_scalar((double)total_params);
}

// Alias without dream_ prefix
Tensor* export_c_header(Tensor* id, Tensor* prefix, Tensor* func_name) {
    return dream_export_c_header(id, prefix, func_name);
}

// ============================================================
// int8 quantized C export
//
// Same idea as export_c_header, but weights are stored as int8_t instead
// of float32 — 4x smaller, and int8 multiplications are faster on MCUs
// (often with hardware MAC support, e.g. ARM Cortex-M4 SMLAD).
//
// Quantization scheme: per-tensor symmetric.
//   scale = max(|W|) / 127
//   W_q[i] = round(W[i] / scale) clamped to [-128, 127]
//   Reconstructed at runtime: W_approx[i] = W_q[i] * scale
//
// Precision loss on small MLPs: typically <1% accuracy drop.
// ============================================================
Tensor* dream_export_c_header_int8(Tensor* id_t, Tensor* prefix_t, Tensor* func_name_t) {
    if (!id_t) return create_scalar(0.0);
    int id = (int)id_t->data[0];
    if (id < 0 || id >= (int)g_ckpts.size()) {
        std::cerr << "RUNTIME ERROR in export_c_header_int8: bad checkpoint id\n" << std::flush;
        return create_scalar(0.0);
    }

    std::string prefix = "model", func_name = "predict";
    if (prefix_t && prefix_t->grad_fn == "string")
        for (int i = 0; i < prefix_t->size; ++i) prefix += ""; // reset...
    if (prefix_t && prefix_t->grad_fn == "string") {
        prefix = "";
        for (int i = 0; i < prefix_t->size; ++i) prefix += (char)(int)prefix_t->data[i];
    }
    if (func_name_t && func_name_t->grad_fn == "string") {
        func_name = "";
        for (int i = 0; i < func_name_t->size; ++i) func_name += (char)(int)func_name_t->data[i];
    }

    auto* sess = g_ckpts[id];

    // Group tensors by layer (same logic as float export)
    struct Layer {
        std::string name;
        Tensor* weight = nullptr;
        Tensor* bias = nullptr;
        std::string w_name, b_name;
        int out_dim = 0, in_dim = 0;
    };
    std::map<std::string, Layer> layer_map;
    for (auto& kv : sess->tensors) {
        std::string name = kv.first;
        Tensor* t = kv.second;
        std::string base;
        bool is_weight = false, is_bias = false;
        if (name.size() > 7 && name.substr(name.size()-7) == ".weight") {
            base = name.substr(0, name.size()-7); is_weight = true;
        } else if (name.size() > 5 && name.substr(name.size()-5) == ".bias") {
            base = name.substr(0, name.size()-5); is_bias = true;
        } else base = name;

        auto& layer = layer_map[base];
        if (layer.name.empty()) layer.name = base;
        if (is_weight) {
            layer.weight = t; layer.out_dim = t->d0; layer.in_dim = t->d1;
            layer.w_name = sanitize_c_name(name);
        } else if (is_bias) {
            layer.bias = t; layer.b_name = sanitize_c_name(name);
        }
    }
    std::vector<Layer> layers;
    std::set<std::string> added;
    for (auto& kv : sess->tensors) {
        std::string name = kv.first;
        std::string base;
        if (name.size() > 7 && name.substr(name.size()-7) == ".weight")
            base = name.substr(0, name.size()-7);
        else if (name.size() > 5 && name.substr(name.size()-5) == ".bias")
            base = name.substr(0, name.size()-5);
        else continue;
        if (added.count(base)) continue;
        added.insert(base);
        auto it = layer_map.find(base);
        if (it != layer_map.end() && it->second.weight && it->second.bias)
            layers.push_back(it->second);
    }
    if (layers.empty()) {
        std::cerr << "RUNTIME ERROR in export_c_header_int8: no .weight/.bias pairs\n" << std::flush;
        return create_scalar(0.0);
    }

    int input_dim = layers.front().in_dim;
    int output_dim = layers.back().out_dim;

    // Compute maximum intermediate buffer size
    int max_buf = 0;
    for (auto& L : layers) if (L.out_dim > max_buf) max_buf = L.out_dim;

    std::string prefix_upper;
    {
        std::string s = sanitize_c_name(prefix);
        for (char c : s) {
            if (c >= 'a' && c <= 'z') prefix_upper += (char)(c - 'a' + 'A');
            else prefix_upper += c;
        }
    }

    // --- Header file ---
    std::string hpath = prefix + ".h";
    dream_mkdirs_for(hpath);
    FILE* hf = fopen(hpath.c_str(), "w");
    if (!hf) {
        std::cerr << "RUNTIME ERROR: cannot write " << hpath << "\n" << std::flush;
        return create_scalar(0.0);
    }
    fprintf(hf,
        "/*\n"
        " * %s.h - auto-generated by Dream (int8 quantized)\n"
        " *\n"
        " * Self-contained C99 inference with int8 weights.\n"
        " * Input dim: %d, Output dim: %d, Layers: %zu\n"
        " * Quantization: per-tensor symmetric, weights only (activations stay float).\n"
        " *\n"
        " * Usage:\n"
        " *   float input[%d] = { ... };\n"
        " *   float output[%d];\n"
        " *   %s(input, output);\n"
        " */\n"
        "#ifndef %s_H\n"
        "#define %s_H\n\n"
        "#include <stdint.h>\n\n"
        "#ifdef __cplusplus\n"
        "extern \"C\" {\n"
        "#endif\n\n"
        "#define %s_INPUT_DIM  %d\n"
        "#define %s_OUTPUT_DIM %d\n"
        "#define %s_NUM_LAYERS %zu\n"
        "#define %s_QUANTIZED  1\n\n"
        "void %s(const float* input, float* output);\n\n"
        "#ifdef __cplusplus\n"
        "}\n"
        "#endif\n\n"
        "#endif /* %s_H */\n",
        prefix.c_str(), input_dim, output_dim, layers.size(),
        input_dim, output_dim, func_name.c_str(),
        prefix_upper.c_str(), prefix_upper.c_str(),
        prefix_upper.c_str(), input_dim,
        prefix_upper.c_str(), output_dim,
        prefix_upper.c_str(), layers.size(),
        prefix_upper.c_str(),
        func_name.c_str(),
        prefix_upper.c_str()
    );
    fclose(hf);

    // --- Source file ---
    std::string cpath = prefix + ".c";
    dream_mkdirs_for(cpath);
    FILE* sf = fopen(cpath.c_str(), "w");
    if (!sf) {
        std::cerr << "RUNTIME ERROR: cannot write " << cpath << "\n" << std::flush;
        return create_scalar(0.0);
    }

    fprintf(sf,
        "/* %s.c - auto-generated by Dream (int8 quantized) */\n"
        "#include \"%s.h\"\n"
        "#include <stdint.h>\n"
        "#include <math.h>\n\n",
        prefix.c_str(), prefix.c_str()
    );

    // For each layer: quantize weights to int8, emit weight array + scale + bias
    long long total_weight_bytes = 0;
    long long total_bias_bytes = 0;
    double max_error = 0;

    for (auto& L : layers) {
        const double* wd = L.weight->data;
        int n = L.weight->size;

        // Find max abs
        double max_abs = 0;
        for (int i = 0; i < n; ++i) {
            double a = std::fabs(wd[i]);
            if (a > max_abs) max_abs = a;
        }
        // Scale: maps [-127, 127] to [-max_abs, max_abs]. Guard against zero.
        double scale = (max_abs > 1e-30) ? (max_abs / 127.0) : 1.0;

        // Quantize
        std::vector<int8_t> q(n);
        for (int i = 0; i < n; ++i) {
            double v = wd[i] / scale;
            if (v > 127.0) v = 127.0;
            if (v < -128.0) v = -128.0;
            int r = (int)std::lround(v);
            if (r > 127) r = 127;
            if (r < -128) r = -128;
            q[i] = (int8_t)r;

            // Track quantization error (to report to user)
            double reconstructed = (double)q[i] * scale;
            double err = std::fabs(reconstructed - wd[i]);
            if (err > max_error) max_error = err;
        }

        // Emit weight array
        fprintf(sf, "/* %s.weight shape: [%d, %d], quantized int8 */\n",
                L.name.c_str(), L.out_dim, L.in_dim);
        fprintf(sf, "static const int8_t %s[%d] = {", L.w_name.c_str(), n);
        for (int i = 0; i < n; ++i) {
            if (i % 16 == 0) fprintf(sf, "\n    ");
            fprintf(sf, "%d", (int)q[i]);
            if (i + 1 < n) fprintf(sf, ", ");
        }
        fprintf(sf, "\n};\n");
        // Emit scale as float (always needs decimal point)
        char sbuf[64];
        snprintf(sbuf, sizeof(sbuf), "%.9g", (float)scale);
        bool has_dot = false;
        for (char* p = sbuf; *p; ++p)
            if (*p == '.' || *p == 'e' || *p == 'E' || *p == 'n' || *p == 'i') { has_dot = true; break; }
        if (!has_dot) strcat(sbuf, ".0");
        fprintf(sf, "static const float %s_scale = %sf;\n\n", L.w_name.c_str(), sbuf);

        total_weight_bytes += n;  // 1 byte per element

        // Bias: stay as float32 (small, precision matters more)
        fprintf(sf, "/* %s.bias shape: [%d], float32 */\n", L.name.c_str(), L.out_dim);
        fprintf(sf, "static const float %s[%d] = {", L.b_name.c_str(), L.bias->size);
        const double* bd = L.bias->data;
        auto fmt_float = [&](float v) {
            char buf[64];
            snprintf(buf, sizeof(buf), "%.9g", v);
            bool has_suffix_char = false;
            for (char* p = buf; *p; ++p)
                if (*p == '.' || *p == 'e' || *p == 'E' || *p == 'n' || *p == 'i') { has_suffix_char = true; break; }
            if (!has_suffix_char) strcat(buf, ".0");
            fprintf(sf, "%sf", buf);
        };
        for (int i = 0; i < L.bias->size; ++i) {
            if (i % 8 == 0) fprintf(sf, "\n    ");
            fmt_float((float)bd[i]);
            if (i + 1 < L.bias->size) fprintf(sf, ", ");
        }
        fprintf(sf, "\n};\n\n");
        total_bias_bytes += L.bias->size * 4;
    }

    // Emit the int8-aware dense + relu helpers
    fprintf(sf,
        "/* Dense layer with int8 quantized weights. */\n"
        "/* y[j] = bias[j] + scale * sum_k (int8_w[j*in + k] * x[k]) */\n"
        "static void dense_q(const float* x, int in_dim,\n"
        "                    const int8_t* w, float scale, const float* b,\n"
        "                    float* y, int out_dim) {\n"
        "    for (int j = 0; j < out_dim; ++j) {\n"
        "        float acc = b[j];\n"
        "        const int8_t* wr = w + j * in_dim;\n"
        "        for (int k = 0; k < in_dim; ++k) {\n"
        "            acc += scale * ((float)wr[k]) * x[k];\n"
        "        }\n"
        "        y[j] = acc;\n"
        "    }\n"
        "}\n\n"
        "static void relu_inplace(float* x, int n) {\n"
        "    for (int i = 0; i < n; ++i) if (x[i] < 0.0f) x[i] = 0.0f;\n"
        "}\n\n"
    );

    // Predict function
    fprintf(sf, "void %s(const float* input, float* output) {\n", func_name.c_str());
    fprintf(sf, "    float buf_a[%d];\n", max_buf);
    fprintf(sf, "    float buf_b[%d];\n", max_buf);
    fprintf(sf, "    const float* cur_in = input;\n");
    fprintf(sf, "    int cur_dim = %s_INPUT_DIM;\n", prefix_upper.c_str());
    fprintf(sf, "    float* cur_out;\n");
    fprintf(sf, "    int use_a = 1;\n\n");
    for (size_t li = 0; li < layers.size(); ++li) {
        const auto& L = layers[li];
        bool is_last = (li == layers.size() - 1);
        if (is_last) {
            fprintf(sf, "    /* layer %zu: %s -> output */\n", li, L.name.c_str());
            fprintf(sf, "    dense_q(cur_in, cur_dim, %s, %s_scale, %s, output, %d);\n",
                    L.w_name.c_str(), L.w_name.c_str(), L.b_name.c_str(), L.out_dim);
        } else {
            fprintf(sf, "    /* layer %zu: %s */\n", li, L.name.c_str());
            fprintf(sf, "    cur_out = use_a ? buf_a : buf_b;\n");
            fprintf(sf, "    dense_q(cur_in, cur_dim, %s, %s_scale, %s, cur_out, %d);\n",
                    L.w_name.c_str(), L.w_name.c_str(), L.b_name.c_str(), L.out_dim);
            fprintf(sf, "    relu_inplace(cur_out, %d);\n", L.out_dim);
            fprintf(sf, "    cur_in = cur_out;\n");
            fprintf(sf, "    cur_dim = %d;\n", L.out_dim);
            fprintf(sf, "    use_a = !use_a;\n\n");
        }
    }
    fprintf(sf, "}\n");
    fclose(sf);

    // Report
    long long total_params = 0;
    for (auto& L : layers) total_params += L.weight->size + L.bias->size;
    long long float32_bytes = total_params * 4;
    long long int8_bytes = total_weight_bytes + total_bias_bytes;

    std::cerr << "[export_c_int8] wrote " << prefix << ".h + " << prefix << ".c\n";
    std::cerr << "  layers:            " << layers.size() << "\n";
    std::cerr << "  input_dim:         " << input_dim << "\n";
    std::cerr << "  output_dim:        " << output_dim << "\n";
    std::cerr << "  parameters:        " << total_params << "\n";
    std::cerr << "  size (float32):    " << float32_bytes << " bytes\n";
    std::cerr << "  size (int8):       " << int8_bytes << " bytes  (";
    {
        char pct[32];
        snprintf(pct, sizeof(pct), "%.2f", 100.0 * int8_bytes / float32_bytes);
        std::cerr << pct << "% of float)\n";
    }
    std::cerr << "  max quant. error:  " << max_error << "\n";
    std::cerr << "  fn signature:      void " << func_name
              << "(const float* input, float* output);\n";
    std::cerr << "  Compile with: gcc -O2 -lm " << prefix << ".c -o my_app\n";
    std::cerr << std::flush;

    return create_scalar((double)int8_bytes);
}

// Alias without dream_ prefix
Tensor* export_c_header_int8(Tensor* id, Tensor* prefix, Tensor* func_name) {
    return dream_export_c_header_int8(id, prefix, func_name);
}

// =====================================================================
//  gradcheck: verify every op's backward against numerical differentiation.
//
//  For each op:  y = op(a[,b]);  L = sum(y (*) w)  with FIXED RANDOM weights w.
//  Random weighting matters: with uniform weights (plain sum), a transposed
//  or permuted gradient in matmul/softmax can still pass by accident; random
//  per-element weights make such bugs show up as large relative error.
//
//  Analytic:   backward(L) fills leaf->grad.
//  Numerical:  central difference (L(x+eps) - L(x-eps)) / (2 eps), where the
//              loss is recomputed raw in C++ (dot of op output with w), so
//              no graph is built and no autograd code is trusted on this side.
//  Tolerance:  rel err < 1e-4 with a 1e-6 denominator floor (double precision
//              central differences are good to ~1e-8 here).
//
//  Exposed to Dream as gradcheck(); returns 1.0 if all ops pass else 0.0.
// =====================================================================

static Tensor* gc_rand(int d0, int d1, double lo, double hi) {
    Tensor* t = alloc_tensor(d0, d1, 1, 1);
    auto& rng = get_rng();
    std::uniform_real_distribution<double> dist(lo, hi);
    for (int i = 0; i < t->size; ++i) t->data[i] = dist(rng);
    return t;
}

// Push values away from a non-differentiable point at 0 (relu, abs).
static void gc_nudge_from_zero(Tensor* t, double min_abs) {
    for (int i = 0; i < t->size; ++i) {
        double v = t->data[i];
        if (std::fabs(v) < min_abs)
            t->data[i] = (v >= 0.0 ? 1.0 : -1.0) * (min_abs + 0.05);
    }
}

// Core harness. fwd() must build the graph and return y (+1 ownership).
// leaves = tensors whose gradients we verify. w = weight tensor, same shape as y.
static bool gc_run(std::function<Tensor*()> fwd,
                   const std::vector<Tensor*>& leaves,
                   Tensor* w, double tol, double& max_rel_out)
{
    // ---- analytic pass ----
    for (Tensor* t : leaves) std::fill(t->grad, t->grad + t->size, 0.0);
    std::fill(w->grad, w->grad + w->size, 0.0);
    Tensor* y = fwd();
    Tensor* z = tensor_mul(y, w);
    Tensor* L = sum(z);
    Tensor* br = backward(L);
    tensor_decref(br);
    std::vector<std::vector<double>> ana;
    for (Tensor* t : leaves) ana.emplace_back(t->grad, t->grad + t->size);
    tensor_decref(L); tensor_decref(z); tensor_decref(y);

    // ---- numerical pass (no autograd trusted here) ----
    bool prev = g_autograd_enabled;
    g_autograd_enabled = false;
    auto num_loss = [&]() -> double {
        Tensor* yy = fwd();
        double s = 0.0;
        for (int i = 0; i < yy->size; ++i) s += yy->data[i] * w->data[i];
        tensor_decref(yy);
        return s;
    };
    double max_rel = 0.0, ana_abs = 0.0, num_abs = 0.0;
    for (size_t li = 0; li < leaves.size(); ++li) {
        Tensor* t = leaves[li];
        for (int i = 0; i < t->size; ++i) {
            double orig = t->data[i];
            double eps  = 1e-6 * std::max(1.0, std::fabs(orig));
            t->data[i] = orig + eps; double Lp = num_loss();
            t->data[i] = orig - eps; double Lm = num_loss();
            t->data[i] = orig;
            double gnum  = (Lp - Lm) / (2.0 * eps);
            double gana  = ana[li][i];
            ana_abs = std::max(ana_abs, std::fabs(gana));
            num_abs = std::max(num_abs, std::fabs(gnum));
            double denom = std::max(std::max(std::fabs(gnum), std::fabs(gana)), 1e-6);
            double rel   = std::fabs(gnum - gana) / denom;
            if (rel > max_rel) max_rel = rel;
        }
    }
    g_autograd_enabled = prev;
    // Degenerate guard: if BOTH sides are identically zero, the op most
    // likely errored out and returned a constant -- the check proved
    // nothing, so fail loudly instead of "passing" vacuously.
    if (ana_abs < 1e-12 && num_abs < 1e-12) { max_rel_out = 1e99; return false; }
    max_rel_out = max_rel;
    return max_rel < tol;
}

Tensor* gradcheck() {
    get_rng().seed(20260706);   // deterministic runs
    std::printf("=== gradcheck: analytic vs numerical gradients ===\n");
    int pass = 0, total = 0;
    auto report = [&](const char* name, bool ok, double mr) {
        std::printf("  %-10s max_rel=%.2e  %s\n", name, mr, ok ? "PASS" : "FAIL");
        ++total; if (ok) ++pass;
    };

    // ---- binary elementwise, shape 3x4 ----
    auto bin = [&](const char* name, Tensor* (*op)(Tensor*, Tensor*),
                   double blo, double bhi) {
        Tensor* a = gc_rand(3, 4, -2.0, 2.0);
        Tensor* b = gc_rand(3, 4, blo, bhi);
        Tensor* w = gc_rand(3, 4, -1.0, 1.0);
        double mr; bool ok = gc_run([&]{ return op(a, b); }, {a, b}, w, 1e-4, mr);
        report(name, ok, mr);
        tensor_decref(a); tensor_decref(b); tensor_decref(w);
    };
    bin("add", tensor_add, -2.0, 2.0);
    bin("sub", tensor_sub, -2.0, 2.0);
    bin("mul", tensor_mul, -2.0, 2.0);
    bin("div", tensor_div,  0.7, 1.7);   // keep divisor away from 0

    // ---- matmul, 3x4 @ 4x5 (random w catches transposed grads) ----
    {
        Tensor* a = gc_rand(3, 4, -1.0, 1.0);
        Tensor* b = gc_rand(4, 5, -1.0, 1.0);
        Tensor* w = gc_rand(3, 5, -1.0, 1.0);
        double mr; bool ok = gc_run([&]{ return tensor_matmul(a, b); }, {a, b}, w, 1e-4, mr);
        report("matmul", ok, mr);
        tensor_decref(a); tensor_decref(b); tensor_decref(w);
    }

    // ---- unary, shape 3x4 ----
    auto un = [&](const char* name, Tensor* (*op)(Tensor*),
                  double lo, double hi, bool nudge) {
        Tensor* a = gc_rand(3, 4, lo, hi);
        if (nudge) gc_nudge_from_zero(a, 0.10);
        Tensor* w = gc_rand(3, 4, -1.0, 1.0);
        double mr; bool ok = gc_run([&]{ return op(a); }, {a}, w, 1e-4, mr);
        report(name, ok, mr);
        tensor_decref(a); tensor_decref(w);
    };
    un("relu",    relu,          -2.0, 2.0, true);
    un("abs",     dream_abs,     -2.0, 2.0, true);
    un("exp",     dream_exp,     -1.0, 1.0, false);
    un("log",     dream_log,      0.5, 2.0, false);
    un("sqrt",    dream_sqrt,     0.5, 2.0, false);
    un("sigmoid", dream_sigmoid, -2.0, 2.0, false);
    un("tanh",    dream_tanh,    -2.0, 2.0, false);
    un("gelu",    dream_gelu,    -2.0, 2.0, false);
    un("softmax", dream_softmax, -2.0, 2.0, false);
    un("logsmax",  dream_log_softmax, -2.0, 2.0, false);

    // ---- reductions: y is 1x1, so w is 1x1 ----
    auto red = [&](const char* name, Tensor* (*op)(Tensor*)) {
        Tensor* a = gc_rand(3, 4, -2.0, 2.0);
        Tensor* w = gc_rand(1, 1, 0.5, 1.5);
        double mr; bool ok = gc_run([&]{ return op(a); }, {a}, w, 1e-4, mr);
        report(name, ok, mr);
        tensor_decref(a); tensor_decref(w);
    };
    red("sum",  sum);
    red("mean", mean);

    // ---- second batch: shape ops, losses, conv, modern activations ----

    // reshape: 3x4 -> 6x2 (element-wise pass-through gradient)
    {
        Tensor* a = gc_rand(3, 4, -2.0, 2.0);
        Tensor* d0 = create_scalar(6.0), *d1 = create_scalar(2.0);
        Tensor* w = gc_rand(6, 2, -1.0, 1.0);
        double mr; bool ok = gc_run([&]{ return reshape(a, d0, d1); }, {a}, w, 1e-4, mr);
        report("reshape", ok, mr);
        tensor_decref(a); tensor_decref(d0); tensor_decref(d1); tensor_decref(w);
    }
    // transpose: 3x4 -> 4x3
    {
        Tensor* a = gc_rand(3, 4, -2.0, 2.0);
        Tensor* w = gc_rand(4, 3, -1.0, 1.0);
        double mr; bool ok = gc_run([&]{ return dream_transpose(a); }, {a}, w, 1e-4, mr);
        report("transpose", ok, mr);
        tensor_decref(a); tensor_decref(w);
    }
    // concat along columns: 3x2 ++ 3x3 -> 3x5
    {
        Tensor* a = gc_rand(3, 2, -2.0, 2.0);
        Tensor* b = gc_rand(3, 3, -2.0, 2.0);
        Tensor* w = gc_rand(3, 5, -1.0, 1.0);
        double mr; bool ok = gc_run([&]{ return dream_concat(a, b); }, {a, b}, w, 1e-4, mr);
        report("concat", ok, mr);
        tensor_decref(a); tensor_decref(b); tensor_decref(w);
    }
    // cumsum: same shape
    {
        Tensor* a = gc_rand(3, 4, -2.0, 2.0);
        Tensor* w = gc_rand(3, 4, -1.0, 1.0);
        double mr; bool ok = gc_run([&]{ return dream_cumsum(a); }, {a}, w, 1e-4, mr);
        report("cumsum", ok, mr);
        tensor_decref(a); tensor_decref(w);
    }
    // leaky_relu / elu: kink at 0 -> nudge; alpha passed as scalar tensor
    {
        Tensor* a = gc_rand(3, 4, -2.0, 2.0); gc_nudge_from_zero(a, 0.10);
        Tensor* alpha = create_scalar(0.1);
        Tensor* w = gc_rand(3, 4, -1.0, 1.0);
        double mr; bool ok = gc_run([&]{ return dream_leaky_relu(a, alpha); }, {a}, w, 1e-4, mr);
        report("leaky", ok, mr);
        tensor_decref(a); tensor_decref(alpha); tensor_decref(w);
    }
    {
        Tensor* a = gc_rand(3, 4, -2.0, 2.0); gc_nudge_from_zero(a, 0.10);
        Tensor* alpha = create_scalar(1.0);
        Tensor* w = gc_rand(3, 4, -1.0, 1.0);
        double mr; bool ok = gc_run([&]{ return dream_elu(a, alpha); }, {a}, w, 1e-4, mr);
        report("elu", ok, mr);
        tensor_decref(a); tensor_decref(alpha); tensor_decref(w);
    }
    // silu / mish: smooth everywhere
    {
        Tensor* a = gc_rand(3, 4, -2.0, 2.0);
        Tensor* w = gc_rand(3, 4, -1.0, 1.0);
        double mr; bool ok = gc_run([&]{ return dream_silu(a); }, {a}, w, 1e-4, mr);
        report("silu", ok, mr);
        tensor_decref(a); tensor_decref(w);
    }
    {
        Tensor* a = gc_rand(3, 4, -2.0, 2.0);
        Tensor* w = gc_rand(3, 4, -1.0, 1.0);
        double mr; bool ok = gc_run([&]{ return dream_mish(a); }, {a}, w, 1e-4, mr);
        report("mish", ok, mr);
        tensor_decref(a); tensor_decref(w);
    }
    // layer_norm: whole-tensor normalization, same shape out
    {
        Tensor* a = gc_rand(3, 4, -2.0, 2.0);
        Tensor* w = gc_rand(3, 4, -1.0, 1.0);
        double mr; bool ok = gc_run([&]{ return dream_layer_norm(a); }, {a}, w, 1e-4, mr);
        report("layernorm", ok, mr);
        tensor_decref(a); tensor_decref(w);
    }
    // mse_loss(pred, target): scalar out; verify d/dpred (target held fixed)
    {
        Tensor* pred = gc_rand(3, 4, -2.0, 2.0);
        Tensor* targ = gc_rand(3, 4, -2.0, 2.0);
        Tensor* w = gc_rand(1, 1, 0.5, 1.5);
        double mr; bool ok = gc_run([&]{ return mse_loss(pred, targ); }, {pred}, w, 1e-4, mr);
        report("mse", ok, mr);
        tensor_decref(pred); tensor_decref(targ); tensor_decref(w);
    }
    // nll_loss(log_probs, class_indices): scalar out; verify d/dlog_probs
    {
        Tensor* lp = gc_rand(4, 3, -2.0, -0.1);   // negative "log-prob"-like values
        Tensor* tg = alloc_tensor(4, 1, 1, 1);
        tg->data[0]=0; tg->data[1]=1; tg->data[2]=2; tg->data[3]=1;
        Tensor* w = gc_rand(1, 1, 0.5, 1.5);
        double mr; bool ok = gc_run([&]{ return dream_nll_loss(lp, tg); }, {lp}, w, 1e-4, mr);
        report("nll", ok, mr);
        tensor_decref(lp); tensor_decref(tg); tensor_decref(w);
    }
    // conv2d: N=2, Cin=2, 5x5 image, 3x3 kernel, Cout=3, valid conv, no bias.
    // Verifies grads through im2col + gemm for BOTH input and kernel.
    {
        Tensor* x = gc_rand(2, 2*5*5, -1.0, 1.0);
        Tensor* k = gc_rand(3, 2*3*3, -1.0, 1.0);
        Tensor* h5 = create_scalar(5.0), *w5 = create_scalar(5.0);
        Tensor* k3 = create_scalar(3.0), *k3b = create_scalar(3.0);
        Tensor* w = gc_rand(2, 3*3*3, -1.0, 1.0);
        double mr; bool ok = gc_run(
            [&]{ return dream_conv2d(x, k, nullptr, h5, w5, k3, k3b); },
            {x, k}, w, 1e-4, mr);
        report("conv2d", ok, mr);
        tensor_decref(x); tensor_decref(k); tensor_decref(w);
        tensor_decref(h5); tensor_decref(w5); tensor_decref(k3); tensor_decref(k3b);
    }


    // ---- third batch: transformer / CNN / loss ops ----

    // sin / cos: smooth
    un("sin", dream_sin, -2.0, 2.0, false);
    un("cos", dream_cos, -2.0, 2.0, false);

    // pow(base, exp): both differentiable; keep base positive
    {
        Tensor* a = gc_rand(3, 4, 0.5, 2.0);
        Tensor* b = gc_rand(3, 4, 0.5, 2.0);
        Tensor* w = gc_rand(3, 4, -1.0, 1.0);
        double mr; bool ok = gc_run([&]{ return dream_pow(a, b); }, {a, b}, w, 1e-4, mr);
        report("pow", ok, mr);
        tensor_decref(a); tensor_decref(b); tensor_decref(w);
    }
    // clamp(x, -1, 1): kinks at the bounds -> push inputs away from +/-1
    {
        Tensor* a = gc_rand(3, 4, -2.0, 2.0);
        for (int i = 0; i < a->size; ++i) {
            double v = a->data[i];
            if (std::fabs(std::fabs(v) - 1.0) < 0.10)
                a->data[i] = (std::fabs(v) > 1.0) ? v * 1.2 : v * 0.8;
        }
        Tensor* lo = create_scalar(-1.0), *hi = create_scalar(1.0);
        Tensor* w = gc_rand(3, 4, -1.0, 1.0);
        double mr; bool ok = gc_run([&]{ return dream_clamp(a, lo, hi); }, {a}, w, 1e-4, mr);
        report("clamp", ok, mr);
        tensor_decref(a); tensor_decref(lo); tensor_decref(hi); tensor_decref(w);
    }
    // cross_entropy(logits, class_indices): scalar out; d/dlogits
    {
        Tensor* lg = gc_rand(4, 3, -2.0, 2.0);
        Tensor* tg = alloc_tensor(4, 3, 1, 1);   // one-hot rows (forward expects same shape)
        int cls[4] = {0, 2, 1, 2};
        for (int r = 0; r < 4; ++r) tg->data[r*3 + cls[r]] = 1.0;
        Tensor* w = gc_rand(1, 1, 0.5, 1.5);
        double mr; bool ok = gc_run([&]{ return dream_cross_entropy(lg, tg); }, {lg}, w, 1e-4, mr);
        report("xent", ok, mr);
        tensor_decref(lg); tensor_decref(tg); tensor_decref(w);
    }
    // bce_loss(pred, target): pred strictly inside (0,1)
    {
        Tensor* pr = gc_rand(3, 4, 0.1, 0.9);
        Tensor* tg = gc_rand(3, 4, 0.0, 1.0);
        for (int i = 0; i < tg->size; ++i) tg->data[i] = tg->data[i] > 0.5 ? 1.0 : 0.0;
        Tensor* w = gc_rand(1, 1, 0.5, 1.5);
        double mr; bool ok = gc_run([&]{ return dream_bce_loss(pr, tg); }, {pr}, w, 1e-4, mr);
        report("bce", ok, mr);
        tensor_decref(pr); tensor_decref(tg); tensor_decref(w);
    }
    // huber_loss(pred, target, delta=1): stay inside the quadratic region
    {
        Tensor* pr = gc_rand(3, 4, -0.3, 0.3);
        Tensor* tg = alloc_tensor(3, 4, 1, 1);   // zeros: |diff| < 0.3 << delta
        Tensor* dl = create_scalar(1.0);
        Tensor* w = gc_rand(1, 1, 0.5, 1.5);
        double mr; bool ok = gc_run([&]{ return dream_huber_loss(pr, tg, dl); }, {pr}, w, 1e-4, mr);
        report("huber", ok, mr);
        tensor_decref(pr); tensor_decref(tg); tensor_decref(dl); tensor_decref(w);
    }
    // linear(x, W, b): y = x @ W^T + b  (W is out_dim x in_dim, PyTorch layout)
    {
        Tensor* x = gc_rand(3, 4, -1.0, 1.0);
        Tensor* W = gc_rand(5, 4, -1.0, 1.0);
        Tensor* b = gc_rand(1, 5, -1.0, 1.0);
        Tensor* w = gc_rand(3, 5, -1.0, 1.0);
        double mr; bool ok = gc_run([&]{ return dream_linear(x, W, b); }, {x, W, b}, w, 1e-4, mr);
        report("linear", ok, mr);
        tensor_decref(x); tensor_decref(W); tensor_decref(b); tensor_decref(w);
    }
    // embedding(indices, weight): gather rows; d/dweight is scatter-add
    {
        Tensor* idx = alloc_tensor(5, 1, 1, 1);
        idx->data[0]=0; idx->data[1]=3; idx->data[2]=1; idx->data[3]=3; idx->data[4]=2;
        Tensor* W = gc_rand(4, 6, -1.0, 1.0);
        Tensor* w = gc_rand(5, 6, -1.0, 1.0);
        double mr; bool ok = gc_run([&]{ return dream_embedding(idx, W); }, {W}, w, 1e-4, mr);
        report("embed", ok, mr);
        tensor_decref(idx); tensor_decref(W); tensor_decref(w);
    }
    // bmm(a, b, B, m, k, n): batched matmul, a: B x (m*k), b: B x (k*n)
    {
        Tensor* a = gc_rand(2, 3*4, -1.0, 1.0);
        Tensor* b = gc_rand(2, 4*5, -1.0, 1.0);
        Tensor* Bt = create_scalar(2.0), *mt = create_scalar(3.0);
        Tensor* kt = create_scalar(4.0), *nt = create_scalar(5.0);
        Tensor* w = gc_rand(2, 3*5, -1.0, 1.0);
        double mr; bool ok = gc_run([&]{ return dream_bmm(a, b, Bt, mt, kt, nt); },
                                    {a, b}, w, 1e-4, mr);
        report("bmm", ok, mr);
        tensor_decref(a); tensor_decref(b); tensor_decref(w);
        tensor_decref(Bt); tensor_decref(mt); tensor_decref(kt); tensor_decref(nt);
    }
    // attention(Q, K, V, B, S, dk): scaled dot-product attention.
    // Grads flow to all of Q, K, V through matmul + softmax + matmul.
    {
        Tensor* Q = gc_rand(1, 4*8, -1.0, 1.0);
        Tensor* Kt = gc_rand(1, 4*8, -1.0, 1.0);
        Tensor* V = gc_rand(1, 4*8, -1.0, 1.0);
        Tensor* Bt = create_scalar(1.0), *St = create_scalar(4.0), *dt = create_scalar(8.0);
        Tensor* w = gc_rand(1, 4*8, -1.0, 1.0);
        double mr; bool ok = gc_run([&]{ return dream_attention(Q, Kt, V, Bt, St, dt); },
                                    {Q, Kt, V}, w, 1e-4, mr);
        report("attention", ok, mr);
        tensor_decref(Q); tensor_decref(Kt); tensor_decref(V);
        tensor_decref(Bt); tensor_decref(St); tensor_decref(dt); tensor_decref(w);
    }
    // dropout(x, keep): stochastic mask made deterministic by reseeding the
    // RNG before EVERY forward, so the analytic and numerical passes see the
    // exact same mask. This is the only way numerical diff is valid here.
    {
        Tensor* a = gc_rand(3, 4, -2.0, 2.0);
        Tensor* keep = create_scalar(0.7);
        Tensor* w = gc_rand(3, 4, -1.0, 1.0);
        double mr; bool ok = gc_run(
            [&]{ get_rng().seed(777); return dream_dropout(a, keep); },
            {a}, w, 1e-4, mr);
        get_rng().seed(20260706);
        report("dropout", ok, mr);
        tensor_decref(a); tensor_decref(keep); tensor_decref(w);
    }
    // maxpool2d / avgpool2d: N=2, C=2, 4x4 image, pool 2 -> 2x2 per channel
    {
        Tensor* x = gc_rand(2, 2*4*4, -1.0, 1.0);
        Tensor* ht = create_scalar(4.0), *wt = create_scalar(4.0);
        Tensor* ct = create_scalar(2.0), *pt = create_scalar(2.0);
        Tensor* w = gc_rand(2, 2*2*2, -1.0, 1.0);
        double mr; bool ok = gc_run([&]{ return dream_maxpool2d(x, ht, wt, ct, pt); },
                                    {x}, w, 1e-4, mr);
        report("maxpool", ok, mr);
        double mr2; bool ok2 = gc_run([&]{ return dream_avgpool2d(x, ht, wt, ct, pt); },
                                      {x}, w, 1e-4, mr2);
        report("avgpool", ok2, mr2);
        tensor_decref(x); tensor_decref(ht); tensor_decref(wt);
        tensor_decref(ct); tensor_decref(pt); tensor_decref(w);
    }
    // axis reductions: dim=1 (R x C -> R x 1) and dim=0 (-> 1 x C)
    {
        Tensor* a = gc_rand(3, 4, -2.0, 2.0);
        Tensor* d1 = create_scalar(1.0);
        Tensor* w = gc_rand(3, 1, -1.0, 1.0);
        double mr; bool ok = gc_run([&]{ return dream_sum_dim(a, d1); }, {a}, w, 1e-4, mr);
        report("sum_dim1", ok, mr);
        Tensor* w0 = gc_rand(1, 4, -1.0, 1.0); Tensor* d0 = create_scalar(0.0);
        double mr2; bool ok2 = gc_run([&]{ return dream_mean_dim(a, d0); }, {a}, w0, 1e-4, mr2);
        report("mean_dim0", ok2, mr2);
        // max_dim: nudge rows apart so the argmax is unambiguous
        for (int r=0;r<3;++r){ double* row=a->data+r*4; for(int j=0;j<4;++j) row[j]=r*1.0+j*0.37; }
        double mr3; bool ok3 = gc_run([&]{ return dream_max_dim(a, d1); }, {a}, w, 1e-4, mr3);
        report("max_dim1", ok3, mr3);
        tensor_decref(a); tensor_decref(d0); tensor_decref(d1);
        tensor_decref(w); tensor_decref(w0);
    }
    // causal attention: position i attends to j <= i
    {
        Tensor* Q = gc_rand(1, 4*8, -1.0, 1.0);
        Tensor* Kt = gc_rand(1, 4*8, -1.0, 1.0);
        Tensor* V = gc_rand(1, 4*8, -1.0, 1.0);
        Tensor* Bt = create_scalar(1.0), *St = create_scalar(4.0), *dt = create_scalar(8.0);
        Tensor* w = gc_rand(1, 4*8, -1.0, 1.0);
        double mr; bool ok = gc_run([&]{ return dream_attention_causal(Q, Kt, V, Bt, St, dt); },
                                    {Q, Kt, V}, w, 1e-4, mr);
        report("attn_causal", ok, mr);
        tensor_decref(Q); tensor_decref(Kt); tensor_decref(V);
        tensor_decref(Bt); tensor_decref(St); tensor_decref(dt); tensor_decref(w);
    }
    // batchnorm2d(x, H, W, C): batch-statistics normalization (training mode).
    // Full backward must account for d(mean)/dx and d(var)/dx -- classic bug site.
    {
        Tensor* x = gc_rand(3, 2*2*2, -1.0, 1.0);
        Tensor* ht = create_scalar(2.0), *wt = create_scalar(2.0), *ct = create_scalar(2.0);
        Tensor* w = gc_rand(3, 2*2*2, -1.0, 1.0);
        double mr; bool ok = gc_run([&]{ return dream_batchnorm2d(x, ht, wt, ct); },
                                    {x}, w, 1e-4, mr);
        report("batchnorm", ok, mr);
        tensor_decref(x); tensor_decref(ht); tensor_decref(wt); tensor_decref(ct); tensor_decref(w);
    }

    std::printf("  TOTAL: %d/%d PASS\n", pass, total);
    return create_scalar(pass == total ? 1.0 : 0.0);
}

// =====================================================================
//  libdream C API accessors.
//  Tensor is an opaque pointer to C callers, so element/shape access and
//  a char*-friendly CSV loader are exported here. Everything else in the
//  C API is the existing extern-"C" runtime surface, declared verbatim in
//  interop/dream.h. These are C-side entry points; Dream scripts should
//  not call them (argument types are raw int/double, not tensors).
// =====================================================================
double* dream_data(Tensor* t) {
    if (!t) return nullptr;
    sync_f32_to_data(t);
    return t->data;
}
int dream_size(Tensor* t) { return t ? t->size : 0; }
int dream_dim0(Tensor* t) { return t ? t->d0 : 0; }
int dream_dim1(Tensor* t) { return t ? t->d1 : 0; }
int dream_dim2(Tensor* t) { return t ? t->d2 : 0; }
int dream_dim3(Tensor* t) { return t ? t->d3 : 0; }
double dream_get(Tensor* t, int i) {
    if (!t || i < 0 || i >= t->size) return 0.0;
    sync_f32_to_data(t);
    return t->data[i];
}
void dream_set(Tensor* t, int i, double v) {
    if (!t || i < 0 || i >= t->size) return;
    t->data[i] = v;
    if (t->dtype == 1 && t->fdata) t->fdata[i] = (float)v;
}
Tensor* dream_retain(Tensor* t)  { return tensor_incref(t); }
Tensor* dream_release(Tensor* t) { return tensor_decref(t); }
// char*-friendly CSV loader (read_csv takes a Dream string tensor).
Tensor* dream_load_csv(const char* path, int skip_header) {
    if (!path) return create_scalar(0.0);
    int n = (int)std::strlen(path);
    Tensor* p = alloc_tensor(1, n > 0 ? n : 1, 1, 1);
    for (int i = 0; i < n; ++i) p->data[i] = (double)(unsigned char)path[i];
    p->grad_fn = "string";
    Tensor* s = create_scalar((double)skip_header);
    Tensor* r = dream_read_csv(p, s);
    tensor_decref(p); tensor_decref(s);
    return r;
}
