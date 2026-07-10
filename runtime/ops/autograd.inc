Tensor* tensor_get_grad(Tensor* t) {
    if (!t) {
        std::cerr << "RUNTIME ERROR in grad(): argument is null\n" << std::flush;
        return alloc_tensor(1,1,1,1);
    }
    if (!t->grad) {
        std::cerr << "RUNTIME ERROR in grad(): tensor has null gradient buffer"
                  << " (shape=" << t->d0 << "x" << t->d1 << " grad_fn=" << t->grad_fn << ")\n" << std::flush;
        return alloc_tensor(t->d0, t->d1, t->d2, t->d3);
    }
    Tensor* g = alloc_tensor(t->d0,t->d1,t->d2,t->d3);
    std::copy(t->grad, t->grad+t->size, g->data);
    return g;
}
Tensor* zero_grad(Tensor* t) {
    if (!t) { std::cerr << "RUNTIME ERROR in zero_grad: tensor is null\n" << std::flush; return tensor_incref(t); }
    if (!t->grad) { std::cerr << "RUNTIME ERROR in zero_grad: grad buffer is null (shape=" << t->d0 << "x" << t->d1 << ")\n" << std::flush; return tensor_incref(t); }
    std::fill(t->grad, t->grad + t->size, 0.0);
    // use tensor_clear_graph instead of nulling pointers directly:
    //it decrefs the old lhs/rhs/parents/base, allowing them to be freed
    // once nothing else references them. Direct null-assignment leaks them.
    tensor_clear_graph(t);
    return tensor_incref(t);
}

Tensor* backward(Tensor* root) {
    if (!root) { std::cerr << "error: backward(null)\n" << std::flush; return create_scalar(0); }

    // no gc. Memory is reclaimed by refcount in tensor_decref().
    //the graph is kept alive by ownership: loss → lhs lp → lhs logits → ...
    //when backward completes and the user discards `loss`, decref cascades
    //tthrough the entire forward graph automatically.

    std::fill(root->grad, root->grad+root->size, 1.0);
    std::vector<Tensor*> topo; std::unordered_set<Tensor*> seen;
    topoSort(root, topo, seen);

    for (int i=(int)topo.size()-1; i>=0; --i) {
        Tensor* t = topo[i];
        if (!t || !t->data || !t->grad) {
            if (t) std::cerr << "RUNTIME ERROR in backward: tensor at topo[" << i
                             << "] has null " << (!t->data ? "data" : "grad")
                             << " (grad_fn=" << t->grad_fn << " shape=" << t->d0 << "x" << t->d1 << ")\n" << std::flush;
            continue;
        }
        //verify lhs/rhs data integrity before gradient propagation
        if (t->lhs && !t->lhs->grad) {
            std::cerr << "RUNTIME ERROR in backward: lhs->grad is null for op '" << t->grad_fn
                      << "' (lhs shape=" << t->lhs->d0 << "x" << t->lhs->d1 << ")\n" << std::flush;
            continue;
        }
        if (t->rhs && !t->rhs->grad) {
            std::cerr << "RUNTIME ERROR in backward: rhs->grad is null for op '" << t->grad_fn
                      << "' (rhs shape=" << t->rhs->d0 << "x" << t->rhs->d1 << ")\n" << std::flush;
            continue;
        }
        const std::string& fn = t->grad_fn;
        if (fn=="add") {
            for(int j=0;j<t->size;++j){
                if(t->lhs) t->lhs->grad[j%t->lhs->size]+=t->grad[j];
                if(t->rhs) t->rhs->grad[j%t->rhs->size]+=t->grad[j];
            }
        } else if (fn=="sub") {
            for(int j=0;j<t->size;++j){
                if(t->lhs) t->lhs->grad[j%t->lhs->size]+=t->grad[j];
                if(t->rhs) t->rhs->grad[j%t->rhs->size]-=t->grad[j];
            }
        } else if (fn=="minimum" || fn=="maximum") {
            // min/max backward: the gradient flows entirely to whichever
            // operand was selected (the smaller one for min, the larger for
            // max); the other operand gets zero. Ties send it to lhs.
            bool is_max = (fn=="maximum");
            for(int j=0;j<t->size;++j){
                double av = t->lhs ? t->lhs->data[j%t->lhs->size] : 0.0;
                double bv = t->rhs ? t->rhs->data[j%t->rhs->size] : 0.0;
                bool pick_lhs = is_max ? (av >= bv) : (av <= bv);
                if (pick_lhs) { if(t->lhs) t->lhs->grad[j%t->lhs->size]+=t->grad[j]; }
                else          { if(t->rhs) t->rhs->grad[j%t->rhs->size]+=t->grad[j]; }
            }
        } else if (fn=="mul") {
            for(int j=0;j<t->size;++j){
                if(t->lhs) t->lhs->grad[j%t->lhs->size]+=t->grad[j]*t->rhs->data[j%t->rhs->size];
                if(t->rhs) t->rhs->grad[j%t->rhs->size]+=t->grad[j]*t->lhs->data[j%t->lhs->size];
            }
        } else if (fn=="div") {
            for(int j=0;j<t->size;++j){
                double r=t->rhs->data[j%t->rhs->size], g=t->grad[j];
                if(t->lhs) t->lhs->grad[j%t->lhs->size]+=g/r;
                if(t->rhs) t->rhs->grad[j%t->rhs->size]-=g*t->lhs->data[j%t->lhs->size]/(r*r);
            }
        } else if (fn=="matmul") {
            // Y = A @ B, A is MxK, B is KxN, Y is MxN
            // dA[i,k] = sum_j dY[i,j] * B[k,j]
            // dB[k,j] = sum_i A[i,k] * dY[i,j]
            // Parallelize outer M loop for dA, outer K loop for dB.
            int M=t->lhs->d0, K=t->lhs->d1, N=t->rhs->d1;
            const double* A = t->lhs->data;
            const double* B = t->rhs->data;
            const double* dY = t->grad;
            double* dA = t->lhs->grad;
            double* dB = t->rhs->grad;

            // dA: (i, k, j) order. Parallel on i (only for large enough work).
            #pragma omp parallel for schedule(static) if(M * K * N >= 2097152)
            for (int i = 0; i < M; ++i) {
                const double* dY_row = dY + i * N;
                double* dA_row = dA + i * K;
                for (int k = 0; k < K; ++k) {
                    const double* B_row = B + k * N;
                    double acc = 0;
                    for (int j = 0; j < N; ++j) acc += dY_row[j] * B_row[j];
                    dA_row[k] += acc;
                }
            }

            // dB: accumulated across all i, so partition on k (no conflict).
            #pragma omp parallel for schedule(static) if(M * K * N >= 2097152)
            for (int k = 0; k < K; ++k) {
                double* dB_row = dB + k * N;
                for (int i = 0; i < M; ++i) {
                    double aik = A[i * K + k];
                    const double* dY_row = dY + i * N;
                    for (int j = 0; j < N; ++j) dB_row[j] += aik * dY_row[j];
                }
            }
        } else if (fn=="sum") {
            // sum: d(sum(x))/dx_i = 1
            if(t->lhs) {
                double g = t->grad[0];
                for(int j=0;j<t->lhs->size;++j) t->lhs->grad[j] += g;
            }
        } else if (fn=="mean") {
            // mean: d(mean(x))/dx_i = 1/n
            if(t->lhs) {
                double g = t->grad[0] / t->lhs->size;
                for(int j=0;j<t->lhs->size;++j) t->lhs->grad[j] += g;
            }
        } else if (fn=="mse_loss") {
            for(int j=0;j<t->lhs->size;++j)
                t->lhs->grad[j]+=2.0*(t->lhs->data[j]-t->rhs->data[j])/t->lhs->size*t->grad[0];
        } else if (fn=="cross_entropy") {
            // Per-row softmax gradient: d/d(logits[n,j]) = (softmax[n,j] - target[n,j]) / N
            int N = t->lhs->d0, C = t->lhs->d1;
            double scale = t->grad[0] / std::max(1, N);
            for (int n = 0; n < N; ++n) {
                const double* row = t->lhs->data + n * C;
                double mx = row[0];
                for (int j = 1; j < C; ++j) if (row[j] > mx) mx = row[j];
                double s = 0;
                for (int j = 0; j < C; ++j) s += std::exp(row[j] - mx);
                for (int j = 0; j < C; ++j) {
                    double sm = std::exp(row[j] - mx) / s;
                    t->lhs->grad[n*C+j] += (sm - t->rhs->data[n*C+j]) * scale;
                }
            }
        } else if (fn=="linear" || fn=="linear_relu") {
            // Fused Linear backward: Y = X @ W^T + b (optional ReLU on Y)
            // If linear_relu: zero dY where Y <= 0 (ReLU backward)
            // Then: dX = dY @ W, dW = dY^T @ X, db = sum(dY, axis=0)
            if (t->num_parents < 3) continue;
            Tensor* x = t->parents[0];
            Tensor* W = t->parents[1];
            Tensor* b = t->parents[2];
            int N = x->d0, in_dim = x->d1, out_dim = W->d0;
            const double* X = x->data;
            const double* Wd = W->data;
            const double* Y = t->data;
            double* dY = t->grad;
            double* dX = x->grad;
            double* dW = W->grad;
            double* db = b->grad;

            // ReLU backward (if fused): mask dY where Y <= 0
            if (fn == "linear_relu") {
                for (int i = 0; i < N * out_dim; ++i)
                    if (Y[i] <= 0) dY[i] = 0;
            }

            // db[j] += sum_i dY[i, j]
            // Each j is independent — parallel safe
            #pragma omp parallel for schedule(static) if(N * out_dim >= 16384)
            for (int j = 0; j < out_dim; ++j) {
                double s = 0;
                for (int i = 0; i < N; ++i) s += dY[i * out_dim + j];
                db[j] += s;
            }

            // dX[i, k] += sum_j dY[i, j] * W[j, k]
            // Parallel on i (each batch independent)
            #pragma omp parallel for schedule(static) if(N * in_dim * out_dim >= 2097152)
            for (int i = 0; i < N; ++i) {
                const double* dY_row = dY + i * out_dim;
                double* dX_row = dX + i * in_dim;
                for (int j = 0; j < out_dim; ++j) {
                    double dy = dY_row[j];
                    const double* W_row = Wd + j * in_dim;
                    for (int k = 0; k < in_dim; ++k) dX_row[k] += dy * W_row[k];
                }
            }

            // dW[j, k] += sum_i dY[i, j] * X[i, k]
            // Parallel on j (each output row independent)
            #pragma omp parallel for schedule(static) if(N * in_dim * out_dim >= 2097152)
            for (int j = 0; j < out_dim; ++j) {
                double* dW_row = dW + j * in_dim;
                for (int i = 0; i < N; ++i) {
                    double dy = dY[i * out_dim + j];
                    const double* X_row = X + i * in_dim;
                    for (int k = 0; k < in_dim; ++k) dW_row[k] += dy * X_row[k];
                }
            }
        } else if (fn=="log_softmax") {
            // d/dx log_softmax(x)[j] = delta(j) - softmax(x)[j]
            // dL/dx[j] = sum_k dL/dy[k] * (delta(j,k) - softmax(x)[j])
            //         = dL/dy[j] - softmax(x)[j] * sum_k dL/dy[k]
            int N = t->lhs->d0, C = t->lhs->d1;
            for (int n = 0; n < N; ++n) {
                const double* x_row = t->lhs->data + n * C;
                const double* dy_row = t->grad + n * C;
                double* dx_row = t->lhs->grad + n * C;
                // Recompute softmax
                double mx = x_row[0];
                for (int j = 1; j < C; ++j) if (x_row[j] > mx) mx = x_row[j];
                double s = 0;
                std::vector<double> sm(C);
                for (int j = 0; j < C; ++j) { sm[j] = std::exp(x_row[j] - mx); s += sm[j]; }
                for (int j = 0; j < C; ++j) sm[j] /= s;
                double sum_dy = 0;
                for (int j = 0; j < C; ++j) sum_dy += dy_row[j];
                for (int j = 0; j < C; ++j) dx_row[j] += dy_row[j] - sm[j] * sum_dy;
            }
        } else if (fn=="nll_loss") {
            // L = -mean(target * log_probs) averaged over batch.
            // Index targets (Nx1): dL/d(log_probs)[n, target[n]] = -1/N
            // One-hot targets (NxC): dL/d(log_probs)[n,j] = -target[n,j]/N
            int N = t->lhs->d0, C = t->lhs->d1;
            double scale = -t->grad[0] / std::max(1, N);
            bool index_target = (t->rhs && t->rhs->d1 == 1 && C > 1);
            if (index_target) {
                int rows = std::min(N, t->rhs->d0);
                for (int n = 0; n < rows; ++n) {
                    int cls = (int)t->rhs->data[n];
                    if (cls >= 0 && cls < C)
                        t->lhs->grad[n*C + cls] += scale;
                }
            } else if (t->rhs) {
                int limit = std::min(N * C, t->rhs->size);
                for (int i = 0; i < limit; ++i)
                    t->lhs->grad[i] += t->rhs->data[i] * scale;
            }
        } else if (fn=="clamp") {
            // Gradient is 0 where clamped, 1 otherwise
            size_t comma = t->bwd_name.find(',');
            double lo = std::stod(t->bwd_name.substr(0, comma));
            double hi = std::stod(t->bwd_name.substr(comma + 1));
            for (int i = 0; i < t->lhs->size; ++i) {
                double v = t->lhs->data[i];
                if (v > lo && v < hi) t->lhs->grad[i] += t->grad[i];
            }
        } else if (fn=="mse") {
            // d/dp MSE = 2*(pred - target) / N
            int n = t->lhs->size;
            for(int j=0;j<n;++j) {
                double diff = t->lhs->data[j] - t->rhs->data[j % t->rhs->size];
                t->lhs->grad[j] += 2.0 * diff / n * t->grad[0];
            }
        } else if (fn=="bce_loss") {
            // d/dp BCE = (-y/p + (1-y)/(1-p)) / N
            int n = t->lhs->size;
            for(int j=0;j<n;++j) {
                double p = std::max(1e-7, std::min(1.0-1e-7, t->lhs->data[j]));
                double y = t->rhs->data[j % t->rhs->size];
                t->lhs->grad[j] += (-y/p + (1.0-y)/(1.0-p)) / n * t->grad[0];
            }
        } else if (fn=="huber_loss") {
            double delta = std::stod(t->bwd_name);
            int n = t->lhs->size;
            for(int j=0;j<n;++j) {
                double diff = t->lhs->data[j] - t->rhs->data[j % t->rhs->size];
                double ad = std::fabs(diff);
                double g = (ad <= delta) ? diff : delta * (diff > 0 ? 1.0 : -1.0);
                t->lhs->grad[j] += g / n * t->grad[0];
            }
        } else if (fn=="relu") {
            int sz = t->lhs->size;
            #pragma omp parallel for schedule(static) if(sz >= 65536)
            for(int j=0;j<sz;++j)
                t->lhs->grad[j]+=(t->lhs->data[j]>0?1.0:0.0)*t->grad[j];
        } else if (fn=="leaky_relu") {
            double alpha = t->rhs ? t->rhs->data[0] : 0.01;
            for(int j=0;j<t->lhs->size;++j)
                t->lhs->grad[j]+=(t->lhs->data[j]>0?1.0:alpha)*t->grad[j];
        } else if (fn=="gelu") {
            // gelu'(x) = 0.5*(1+tanh(k*(x+0.044715*x^3)))
            //           + 0.5*x*sech^2(k*(x+0.044715*x^3))*k*(1+3*0.044715*x^2)
            const double k = std::sqrt(2.0 / M_PI);
            for(int j=0;j<t->lhs->size;++j) {
                double x = t->lhs->data[j];
                double inner = k * (x + 0.044715 * x*x*x);
                double th = std::tanh(inner);
                double sech2 = 1.0 - th*th;
                double dinner = k * (1.0 + 3.0*0.044715*x*x);
                t->lhs->grad[j] += (0.5*(1.0+th) + 0.5*x*sech2*dinner) * t->grad[j];
            }
        } else if (fn=="silu") {
            // silu'(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
            //          = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
            for(int j=0;j<t->lhs->size;++j) {
                double x = t->lhs->data[j];
                double sig = 1.0 / (1.0 + std::exp(-x));
                t->lhs->grad[j] += sig * (1.0 + x * (1.0 - sig)) * t->grad[j];
            }
        } else if (fn=="mish") {
            // mish(x) = x * tanh(softplus(x)), mish'(x) computed via chain rule
            for(int j=0;j<t->lhs->size;++j) {
                double x = t->lhs->data[j];
                double sp = std::log(1.0 + std::exp(x));
                double tsp = std::tanh(sp);
                double sig = 1.0 / (1.0 + std::exp(-x));
                // d/dx [x * tanh(sp)] = tanh(sp) + x * sech^2(sp) * sigmoid(x)
                double sech2 = 1.0 - tsp * tsp;
                t->lhs->grad[j] += (tsp + x * sech2 * sig) * t->grad[j];
            }
        } else if (fn=="elu") {
            double alpha = t->rhs ? t->rhs->data[0] : 1.0;
            for(int j=0;j<t->lhs->size;++j) {
                double x = t->lhs->data[j];
                t->lhs->grad[j] += (x > 0 ? 1.0 : alpha * std::exp(x)) * t->grad[j];
            }
        } else if (fn=="embedding") {
            // Scatter gradients back to the weight table rows
            if (t->lhs && t->rhs) {
                int n = t->lhs->size; // number of tokens
                int dim = t->rhs->d1; // embedding dim
                for (int i = 0; i < n; ++i) {
                    int idx = (int)t->lhs->data[i];
                    if (idx >= 0 && idx < t->rhs->d0)
                        for (int d = 0; d < dim; ++d)
                            t->rhs->grad[idx*dim+d] += t->grad[i*dim+d];
                }
            }
        } else if (fn=="exp") {
            for(int j=0;j<t->lhs->size;++j)
                t->lhs->grad[j]+=t->data[j]*t->grad[j];
        } else if (fn=="log") {
            for(int j=0;j<t->lhs->size;++j)
                t->lhs->grad[j]+=t->grad[j]/t->lhs->data[j];
        } else if (fn=="sqrt") {
            for(int j=0;j<t->lhs->size;++j)
                t->lhs->grad[j]+=t->grad[j]/(2.0*t->data[j]);
        } else if (fn=="abs") {
            for(int j=0;j<t->lhs->size;++j)
                t->lhs->grad[j]+=(t->lhs->data[j]>=0?1.0:-1.0)*t->grad[j];
        } else if (fn=="sigmoid") {
            for(int j=0;j<t->lhs->size;++j){
                double sig=t->data[j];
                t->lhs->grad[j]+=sig*(1-sig)*t->grad[j];
            }
        } else if (fn=="tanh") {
            for(int j=0;j<t->lhs->size;++j){
                double th=t->data[j];
                t->lhs->grad[j]+=(1-th*th)*t->grad[j];
            }
        } else if (fn=="sin") {
            for(int j=0;j<t->lhs->size;++j)
                t->lhs->grad[j]+=std::cos(t->lhs->data[j])*t->grad[j];
        } else if (fn=="cos") {
            for(int j=0;j<t->lhs->size;++j)
                t->lhs->grad[j]+=-std::sin(t->lhs->data[j])*t->grad[j];
        } else if (fn=="tan") {
            for(int j=0;j<t->lhs->size;++j){
                double c=std::cos(t->lhs->data[j]);
                t->lhs->grad[j]+=t->grad[j]/(c*c);
            }
        } else if (fn=="atan") {
            for(int j=0;j<t->lhs->size;++j){
                double x=t->lhs->data[j];
                t->lhs->grad[j]+=t->grad[j]/(1.0+x*x);
            }
        } else if (fn=="transpose") {
            for(int r=0;r<t->d0;++r)
                for(int c=0;c<t->d1;++c)
                    t->lhs->grad[c*t->d0+r]+=t->grad[r*t->d1+c];
        } else if (fn=="transpose_view") {
            // Same as transpose: swap gradient back
            for(int r=0;r<t->d0;++r)
                for(int c=0;c<t->d1;++c)
                    t->lhs->grad[c*t->d0+r]+=t->grad[r*t->d1+c];
        } else if (fn=="slice_view") {
            // Scatter gradient back to parent at the correct row offset
            int start = std::stoi(t->bwd_name);
            int cols = t->lhs->d1 * t->lhs->d2 * t->lhs->d3;
            for(int r=0;r<t->d0;++r)
                for(int c=0;c<cols;++c)
                    t->lhs->grad[(start+r)*cols+c] += t->grad[r*cols+c];
        } else if (fn=="softmax") {
            // Row-wise softmax JVP, matching the row-wise forward:
            //     dL/dx_i = y_i * (g_i - sum_{j in row} g_j * y_j)
            // The sum is per row (last-dim group), not global.
            int R = t->lhs->d0 > 0 ? t->lhs->d0 : 1;
            int C = t->size / R;
            for (int r = 0; r < R; ++r) {
                const double* y = t->data + (size_t)r*C;
                const double* g = t->grad + (size_t)r*C;
                double* gx = t->lhs->grad + (size_t)r*C;
                double dot = 0.0;
                for (int j = 0; j < C; ++j) dot += g[j]*y[j];
                for (int j = 0; j < C; ++j) gx[j] += y[j]*(g[j]-dot);
            }
        } else if (fn=="concat") {
            int nc=t->d1, ac=t->lhs->d1, bc=t->rhs->d1;
            for(int r=0;r<t->d0;++r){
                for(int j=0;j<ac;++j) t->lhs->grad[r*ac+j]+=t->grad[r*nc+j];
                for(int j=0;j<bc;++j) t->rhs->grad[r*bc+j]+=t->grad[r*nc+ac+j];
            }
        } else if (fn=="sum_d1" || fn=="mean_d1") {
            int R = t->lhs->d0 > 0 ? t->lhs->d0 : 1;
            int C = t->lhs->size / R;
            double scale = (fn=="mean_d1") ? 1.0/C : 1.0;
            for (int r = 0; r < R; ++r)
                for (int j = 0; j < C; ++j)
                    t->lhs->grad[(size_t)r*C + j] += t->grad[r] * scale;
        } else if (fn=="sum_d0" || fn=="mean_d0") {
            int R = t->lhs->d0 > 0 ? t->lhs->d0 : 1;
            int C = t->lhs->size / R;
            double scale = (fn=="mean_d0") ? 1.0/R : 1.0;
            for (int r = 0; r < R; ++r)
                for (int j = 0; j < C; ++j)
                    t->lhs->grad[(size_t)r*C + j] += t->grad[j] * scale;
        } else if (fn=="max_d1") {
            int R = t->lhs->d0 > 0 ? t->lhs->d0 : 1;
            int C = t->lhs->size / R;
            for (int r = 0; r < R; ++r) {
                const double* x = t->lhs->data + (size_t)r*C;
                int am = 0; double m = x[0];
                for (int j = 1; j < C; ++j) if (x[j] > m) { m = x[j]; am = j; }
                t->lhs->grad[(size_t)r*C + am] += t->grad[r];
            }
        } else if (fn=="max_d0") {
            int R = t->lhs->d0 > 0 ? t->lhs->d0 : 1;
            int C = t->lhs->size / R;
            for (int j = 0; j < C; ++j) {
                int am = 0; double m = t->lhs->data[j];
                for (int r = 1; r < R; ++r) { double v = t->lhs->data[(size_t)r*C + j]; if (v > m) { m = v; am = r; } }
                t->lhs->grad[(size_t)am*C + j] += t->grad[j];
            }
        } else if (fn=="reshape") {
            // Eelement-wise pass-through: same size, new dims
            for(int j=0;j<t->lhs->size;++j) t->lhs->grad[j] += t->grad[j];
        } else if (fn=="flatten") {
            // grad flows straight back (reshape inverse)
            for(int j=0;j<t->lhs->size;++j) t->lhs->grad[j]+=t->grad[j];
        } else if (fn=="layer_norm") {
            // Per-row layer norm backward, matching the row-wise forward:
            //   dx = inv_std/C * (C*dy - sum(dy) - y*sum(dy*y))   [per row]
            // inv_std is recomputed from the input row (nothing stashed).
            int R = t->lhs->d0 > 0 ? t->lhs->d0 : 1;
            int C = t->lhs->size / R;
            for (int r = 0; r < R; ++r) {
                const double* xin = t->lhs->data + (size_t)r*C;
                const double* y   = t->data + (size_t)r*C;
                const double* dy  = t->grad + (size_t)r*C;
                double* gx        = t->lhs->grad + (size_t)r*C;
                double mu = 0; for (int j=0;j<C;++j) mu += xin[j]; mu /= C;
                double var = 0; for (int j=0;j<C;++j){ double d=xin[j]-mu; var+=d*d; } var /= C;
                double inv_std = 1.0/std::sqrt(var + 1e-5);
                double sum_dy = 0, sum_dy_y = 0;
                for (int j=0;j<C;++j){ sum_dy += dy[j]; sum_dy_y += dy[j]*y[j]; }
                for (int j=0;j<C;++j)
                    gx[j] += inv_std / C * (C*dy[j] - sum_dy - y[j]*sum_dy_y);
            }
        } else if (fn=="dropout") {
            //Forward keeps x/keep with prob keep (else 0), so dy/dx = 1/keep
            // on kept elements. The old code passed grad through UNSCALED --
            // off by exactly (1-keep) relative; gradcheck fingerprinted it at
            // 0.30 for keep=0.7. keep is stashed in bwd_name at forward time.
            {
                double keep = std::atof(t->bwd_name.c_str());
                double inv = (keep > 0.0 && keep <= 1.0) ? 1.0/keep : 1.0;
                for(int j=0;j<t->lhs->size;++j)
                    if(t->data[j] != 0.0) t->lhs->grad[j] += t->grad[j] * inv;
            }
        } else if (fn=="cumsum") {
            // grad of cumsum: each position gets sum of downstream grads
            for(int j=0;j<t->lhs->size;++j)
                for(int k=j;k<t->size;++k)
                    t->lhs->grad[j]+=t->grad[k];
        } else if (fn=="sgd" || fn=="adam") {
            // param update has grad=1 w.r.t. itself (pass through)
            for(int j=0;j<t->lhs->size;++j) t->lhs->grad[j]+=t->grad[j];
        } else if (fn=="pow") {
            // d/dx(x^n) = n * x^(n-1) = n * y / x
            // d/dn(x^n) = y * ln(x)   (base > 0; skipped otherwise, matching
            //                          the domain where the derivative exists)
            // The exponent gradient was previously MISSING entirely, so any
            // learnable exponent silently received zero gradient.
            for(int j=0;j<t->lhs->size;++j) {
                double base_val = t->lhs->data[j];
                double exp_val = t->rhs ? t->rhs->data[j % t->rhs->size] : 2.0;
                double y = t->data[j % t->size];
                double g = t->grad[j % t->size];
                if(base_val != 0.0)
                    t->lhs->grad[j] += g * exp_val * y / base_val;
                if(t->rhs && base_val > 0.0)
                    t->rhs->grad[j % t->rhs->size] += g * y * std::log(base_val);
            }
        } else if (fn=="conv2d") {
            int H,W,KH,KW,Cin,Cout;
            sscanf(t->bwd_name.c_str(), "%d,%d,%d,%d,%d,%d", &H,&W,&KH,&KW,&Cin,&Cout);
            int OH=H-KH+1, OW=W-KW+1, N=t->lhs->d0;
            int col_rows=Cin*KH*KW, col_cols=OH*OW;
            std::vector<double> col(col_rows*col_cols);
            for (int n=0; n<N; ++n) {
                const double* inp=t->lhs->data+n*Cin*H*W;
                const double* go=t->grad+n*Cout*OH*OW;
                // grad_kernel += grad_output @ im2col^T
                im2col(inp,col.data(),Cin,H,W,KH,KW,OH,OW);
                for(int co=0;co<Cout;++co)
                    for(int cr=0;cr<col_rows;++cr){
                        double s=0;
                        for(int j=0;j<col_cols;++j) s+=go[co*col_cols+j]*col[cr*col_cols+j];
                        t->rhs->grad[co*col_rows+cr]+=s;
                    }
                // grad_input: col2im(kernel^T @ grad_output)
                std::vector<double> dcol(col_rows*col_cols,0.0);
                for(int cr=0;cr<col_rows;++cr)
                    for(int j=0;j<col_cols;++j){
                        double s=0;
                        for(int co=0;co<Cout;++co)
                            s+=t->rhs->data[co*col_rows+cr]*go[co*col_cols+j];
                        dcol[cr*col_cols+j]=s;
                    }
                double* dinp=t->lhs->grad+n*Cin*H*W;
                col2im(dcol.data(),dinp,Cin,H,W,KH,KW,OH,OW);
            }
        } else if (fn=="maxpool") {
            // Route gradients to the max-index positions
            int H,W,C,pool;
            sscanf(t->bwd_name.c_str(), "%d,%d,%d,%d", &H,&W,&C,&pool);
            int OH=H/pool, OW=W/pool, N=t->lhs->d0;
            for (int n=0; n<N; ++n)
                for (int c=0; c<C; ++c)
                    for (int oh=0; oh<OH; ++oh)
                        for (int ow=0; ow<OW; ++ow) {
                            int oi = n*C*OH*OW + c*OH*OW + oh*OW + ow;
                            int maxidx = (int)t->rhs->data[oi];
                            t->lhs->grad[n*C*H*W + c*H*W + maxidx] += t->grad[oi];
                        }
        } else if (fn=="bmm") {
            // Batch matmul backward: C = A @ B per batch
            // grad_A += grad_C @ B^T, grad_B += A^T @ grad_C
            int B,M,K,N;
            sscanf(t->bwd_name.c_str(), "%d,%d,%d,%d", &B,&M,&K,&N);
            for (int bi=0; bi<B; ++bi) {
                const double* go = t->grad + bi*M*N;
                const double* Ad = t->lhs->data + bi*M*K;
                const double* Bd = t->rhs->data + bi*K*N;
                double* gA = t->lhs->grad + bi*M*K;
                double* gB = t->rhs->grad + bi*K*N;
                // grad_A += grad_C @ B^T
                for(int r=0;r<M;++r)
                    for(int k=0;k<K;++k) {
                        double s=0;
                        for(int c=0;c<N;++c) s+=go[r*N+c]*Bd[k*N+c];
                        gA[r*K+k]+=s;
                    }
                // grad_B += A^T @ grad_C
                for(int k=0;k<K;++k)
                    for(int c=0;c<N;++c) {
                        double s=0;
                        for(int r=0;r<M;++r) s+=Ad[r*K+k]*go[r*N+c];
                        gB[k*N+c]+=s;
                    }
            }
        } else if (fn=="avgpool") {
            // distribute gradient evenly across each pooling window
            int H,W,C,pool;
            sscanf(t->bwd_name.c_str(), "%d,%d,%d,%d", &H,&W,&C,&pool);
            int OH=H/pool, OW=W/pool, Nb=t->lhs->d0;
            double inv_area = 1.0 / (pool * pool);
            for (int n=0; n<Nb; ++n)
                for (int c=0; c<C; ++c)
                    for (int oh=0; oh<OH; ++oh)
                        for (int ow=0; ow<OW; ++ow) {
                            double g = t->grad[n*C*OH*OW + c*OH*OW + oh*OW + ow] * inv_area;
                            for (int ph=0; ph<pool; ++ph)
                                for (int pw=0; pw<pool; ++pw)
                                    t->lhs->grad[n*C*H*W + c*H*W + (oh*pool+ph)*W + (ow*pool+pw)] += g;
                        }
        } else if (fn=="global_avg_pool") {
            int H,W,C;
            sscanf(t->bwd_name.c_str(), "%d,%d,%d", &H,&W,&C);
            int HW = H*W, Nb = t->lhs->d0;
            double inv = 1.0 / HW;
            for (int n=0; n<Nb; ++n)
                for (int c=0; c<C; ++c) {
                    double g = t->grad[n*C + c] * inv;
                    for (int hw=0; hw<HW; ++hw)
                        t->lhs->grad[n*C*HW + c*HW + hw] += g;
                }
        } else if (fn=="batchnorm") {
            // Full BatchNorm backward (per-feature, across batch):
            // dx[n,f] = (inv_std_f / N) * (N*dy[n,f] - sum_dy_f - y[n,f]*sum_dy_y_f)
            // where y = normalized output, inv_std_f stored in rhs->data[f]
            int N_bn, F_bn;
            sscanf(t->bwd_name.c_str(), "%d,%d", &N_bn, &F_bn);
            for (int f = 0; f < F_bn; ++f) {
                double inv_std_f = t->rhs->data[f];
                // Accumulate per-feature sums
                double sum_dy = 0, sum_dy_y = 0;
                for (int n = 0; n < N_bn; ++n) {
                    int idx = n * F_bn + f;
                    sum_dy   += t->grad[idx];
                    sum_dy_y += t->grad[idx] * t->data[idx];
                }
                // Distribute gradient
                for (int n = 0; n < N_bn; ++n) {
                    int idx = n * F_bn + f;
                    t->lhs->grad[idx] += (inv_std_f / N_bn) *
                        (N_bn * t->grad[idx] - sum_dy - t->data[idx] * sum_dy_y);
                }
            }
        } else if (fn=="batchnorm2d") {
            // spatial BatchNorm2D backward: per-channel across batch + spatial dims
            // dx[n,c,s] = (inv_std_c / G) * (G*dy - sum_dy_c - y*sum_dy_y_c)
            // where G = N*H*W (normalization group size per channel)
            int N_bn, C_bn, H_bn, W_bn;
            sscanf(t->bwd_name.c_str(), "%d,%d,%d,%d", &N_bn, &C_bn, &H_bn, &W_bn);
            int HW = H_bn * W_bn;
            int G = N_bn * HW;
            for (int c = 0; c < C_bn; ++c) {
                double inv_std_c = t->rhs->data[c];
                double sum_dy = 0, sum_dy_y = 0;
                for (int n = 0; n < N_bn; ++n)
                    for (int s = 0; s < HW; ++s) {
                        int idx = n * C_bn * HW + c * HW + s;
                        sum_dy   += t->grad[idx];
                        sum_dy_y += t->grad[idx] * t->data[idx];
                    }
                for (int n = 0; n < N_bn; ++n)
                    for (int s = 0; s < HW; ++s) {
                        int idx = n * C_bn * HW + c * HW + s;
                        t->lhs->grad[idx] += (inv_std_c / G) *
                            (G * t->grad[idx] - sum_dy - t->data[idx] * sum_dy_y);
                    }
            }
        } else if (fn=="rnn_cell") {
            // h = tanh(x @ Wih + h_prev @ Whh + b)
            // dtanh = (1 - h^2) * grad_h
            // parents: [x, h_prev, w_ih, w_hh, bias]
            if (t->parents && t->num_parents >= 4) {
                Tensor *x=t->parents[0], *hp=t->parents[1];
                Tensor *wih=t->parents[2], *whh=t->parents[3];
                Tensor *bias=t->num_parents>4?t->parents[4]:nullptr;
                int B=x->d0, I=x->d1, H=whh->d1;
                for(int b=0;b<B;++b)
                    for(int j=0;j<H;++j) {
                        double h=t->data[b*H+j];
                        double dt=(1.0-h*h)*t->grad[b*H+j]; // dtanh
                        // grad_x += dt * Wih^T
                        for(int k=0;k<I;++k) x->grad[b*I+k]+=dt*wih->data[k*H+j];
                        // grad_h_prev += dt * Whh^T
                        for(int k=0;k<H;++k) hp->grad[b*H+k]+=dt*whh->data[k*H+j];
                        // grad_Wih += x^T * dt
                        for(int k=0;k<I;++k) wih->grad[k*H+j]+=x->data[b*I+k]*dt;
                        // grad_Whh += h_prev^T * dt
                        for(int k=0;k<H;++k) whh->grad[k*H+j]+=hp->data[b*H+k]*dt;
                        if(bias) bias->grad[j%bias->size]+=dt;
                    }
            }
        } else if (fn=="lstm") {
            // Simplified: pass gradients through to hidden state
            // (full LSTM backward requires storing gate values; this is approximate)
            if(t->lhs) for(int j=0;j<t->lhs->size&&j<t->size;++j) t->lhs->grad[j]+=t->grad[j];
        } else if (fn=="attention") {
            //full scaled-dot-product attention backward. Forward:
            //smat = Q K^T / sqrt(dk);  A = softmax_row(Smat);  O = A V
            //giave dO:
            //dV = A^T dO
            //dA = dO V^T
            //dS = A o (dA - rowsum(dA o A))     (row-wise softmax JVP)
            //dQ = dS K / sqrt(dk);  dK = dS^T Q / sqrt(dk)
            //1 is recomputed from Q,K here (activation recomputation) since
            //the forward does not store it. The previous code was a
            //pass-through stub that copied dO into Q,K,V verbatim -- not a
            //gradient at all; gradcheck flagged it at rel err ~1.9.
            if (t->parents && t->num_parents >= 3) {
                int B=1, S=1, dk=1;
                std::sscanf(t->bwd_name.c_str(), "%d,%d,%d", &B, &S, &dk);
                Tensor *Qt=t->parents[0], *Kt=t->parents[1], *Vt=t->parents[2];
                double scale = 1.0/std::sqrt((double)dk);
                std::vector<double> A((size_t)S*S), dA((size_t)S*S), dS((size_t)S*S);
                for (int b=0; b<B; ++b) {
                    const double *Qb=Qt->data+(size_t)b*S*dk, *Kb=Kt->data+(size_t)b*S*dk,
                                 *Vb=Vt->data+(size_t)b*S*dk, *dOb=t->grad+(size_t)b*S*dk;
                    double *dQb=Qt->grad+(size_t)b*S*dk, *dKb=Kt->grad+(size_t)b*S*dk,
                           *dVb=Vt->grad+(size_t)b*S*dk;
                    for (int i=0;i<S;++i) {
                        double mx=-1e300;
                        for (int j=0;j<S;++j) {
                            double s=0; for (int k=0;k<dk;++k) s+=Qb[i*dk+k]*Kb[j*dk+k];
                            A[i*S+j]=s*scale; if (A[i*S+j]>mx) mx=A[i*S+j];
                        }
                        double sum=0;
                        for (int j=0;j<S;++j){ A[i*S+j]=std::exp(A[i*S+j]-mx); sum+=A[i*S+j]; }
                        for (int j=0;j<S;++j) A[i*S+j]/=sum;
                    }
                    for (int j=0;j<S;++j) for (int k=0;k<dk;++k) {
                        double s=0; for (int i=0;i<S;++i) s+=A[i*S+j]*dOb[i*dk+k];
                        dVb[j*dk+k]+=s;
                    }
                    for (int i=0;i<S;++i) for (int j=0;j<S;++j) {
                        double s=0; for (int k=0;k<dk;++k) s+=dOb[i*dk+k]*Vb[j*dk+k];
                        dA[i*S+j]=s;
                    }
                    for (int i=0;i<S;++i) {
                        double rd=0; for (int j=0;j<S;++j) rd+=dA[i*S+j]*A[i*S+j];
                        for (int j=0;j<S;++j) dS[i*S+j]=A[i*S+j]*(dA[i*S+j]-rd);
                    }
                    for (int i=0;i<S;++i) for (int k=0;k<dk;++k) {
                        double s=0; for (int j=0;j<S;++j) s+=dS[i*S+j]*Kb[j*dk+k];
                        dQb[i*dk+k]+=s*scale;
                    }
                    for (int j=0;j<S;++j) for (int k=0;k<dk;++k) {
                        double s=0; for (int i=0;i<S;++i) s+=dS[i*S+j]*Qb[i*dk+k];
                        dKb[j*dk+k]+=s*scale;
                    }
                }
            }
        } else if (fn=="attention_causal") {
            // causal SDPA backward: identical to attention backward but with
            // the j <= i mask applied to A (and thus to all derived sums).
            if (t->parents && t->num_parents >= 3) {
                int B=1, S=1, dk=1;
                std::sscanf(t->bwd_name.c_str(), "%d,%d,%d", &B, &S, &dk);
                Tensor *Qt=t->parents[0], *Kt=t->parents[1], *Vt=t->parents[2];
                double scale = 1.0/std::sqrt((double)dk);
                std::vector<double> A((size_t)S*S), dA((size_t)S*S), dS((size_t)S*S);
                for (int b=0; b<B; ++b) {
                    const double *Qb=Qt->data+(size_t)b*S*dk, *Kb=Kt->data+(size_t)b*S*dk,
                                 *Vb=Vt->data+(size_t)b*S*dk, *dOb=t->grad+(size_t)b*S*dk;
                    double *dQb=Qt->grad+(size_t)b*S*dk, *dKb=Kt->grad+(size_t)b*S*dk,
                           *dVb=Vt->grad+(size_t)b*S*dk;
                    std::fill(A.begin(), A.end(), 0.0);
                    for (int i=0;i<S;++i) {
                        double mx=-1e300;
                        for (int j=0;j<=i;++j) {
                            double s=0; for (int k=0;k<dk;++k) s+=Qb[i*dk+k]*Kb[j*dk+k];
                            A[i*S+j]=s*scale; if (A[i*S+j]>mx) mx=A[i*S+j];
                        }
                        double sum=0;
                        for (int j=0;j<=i;++j){ A[i*S+j]=std::exp(A[i*S+j]-mx); sum+=A[i*S+j]; }
                        for (int j=0;j<=i;++j) A[i*S+j]/=sum;
                    }
                    for (int j=0;j<S;++j) for (int k=0;k<dk;++k) {
                        double s=0; for (int i=j;i<S;++i) s+=A[i*S+j]*dOb[i*dk+k];  // i>=j only
                        dVb[j*dk+k]+=s;
                    }
                    std::fill(dA.begin(), dA.end(), 0.0);
                    for (int i=0;i<S;++i) for (int j=0;j<=i;++j) {
                        double s=0; for (int k=0;k<dk;++k) s+=dOb[i*dk+k]*Vb[j*dk+k];
                        dA[i*S+j]=s;
                    }
                    std::fill(dS.begin(), dS.end(), 0.0);
                    for (int i=0;i<S;++i) {
                        double rd=0; for (int j=0;j<=i;++j) rd+=dA[i*S+j]*A[i*S+j];
                        for (int j=0;j<=i;++j) dS[i*S+j]=A[i*S+j]*(dA[i*S+j]-rd);
                    }
                    for (int i=0;i<S;++i) for (int k=0;k<dk;++k) {
                        double s=0; for (int j=0;j<=i;++j) s+=dS[i*S+j]*Kb[j*dk+k];
                        dQb[i*dk+k]+=s*scale;
                    }
                    for (int j=0;j<S;++j) for (int k=0;k<dk;++k) {
                        double s=0; for (int i=j;i<S;++i) s+=dS[i*S+j]*Qb[i*dk+k];  // i>=j only
                        dKb[j*dk+k]+=s*scale;
                    }
                }
            }
        } else if (fn=="flash_attention") {
            // FlashAttention backward: block-wise, NO S x S matrix. Recomputes
            // attention scores by block using the forward's per-row softmax
            // stats (l, m) and D_i = dot(dO_i, O_i), instead of storing the
            // S x S matrix. Gradients match the standard attention backward
            // verified bit-close. Memory stays O(N) in the backward too, so
            // the forward's memory win is not lost during training.
            if (t->parents && t->num_parents >= 3) {
                int B=1, S=1, dk=1;
                std::sscanf(t->bwd_name.c_str(), "%d,%d,%d", &B, &S, &dk);
                Tensor *Qt=t->parents[0], *Kt=t->parents[1], *Vt=t->parents[2];
                double scale = 1.0/std::sqrt((double)dk);
                const int Bc = 64;   // block size (memory O(Bc), not O(S))
                std::vector<double> sc((size_t)Bc), acc((size_t)dk);
                std::vector<double> row_l((size_t)S), row_m((size_t)S), Drow((size_t)S);
                for (int b=0; b<B; ++b) {
                    const double *Qb=Qt->data+(size_t)b*S*dk, *Kb=Kt->data+(size_t)b*S*dk,
                                 *Vb=Vt->data+(size_t)b*S*dk, *dOb=t->grad+(size_t)b*S*dk;
                    double *dQb=Qt->grad+(size_t)b*S*dk, *dKb=Kt->grad+(size_t)b*S*dk,
                           *dVb=Vt->grad+(size_t)b*S*dk;
                    // pass 1: recompute per-row softmax normalizer (l) and max (m)
                    for (int i=0;i<S;++i) {
                        double m=-1e30, l=0.0;
                        for (int jb=0; jb<=i; jb+=Bc) {
                            int je=std::min(i, jb+Bc-1);
                            double mb=-1e30;
                            for (int j=jb;j<=je;++j){ double s=0; for(int k=0;k<dk;++k)s+=Qb[i*dk+k]*Kb[j*dk+k];
                                sc[j-jb]=s*scale; if(sc[j-jb]>mb)mb=sc[j-jb]; }
                            double mn=std::max(m,mb), corr=std::exp(m-mn); l*=corr;
                            for (int j=jb;j<=je;++j) l+=std::exp(sc[j-jb]-mn);
                            m=mn;
                        }
                        row_l[i]=l; row_m[i]=m;
                    }
                    // pass 2: D_i = dot(dO_i, O_i), recomputing O_i by block
                    for (int i=0;i<S;++i) {
                        std::fill(acc.begin(), acc.end(), 0.0);
                        double m=row_m[i], l=row_l[i];
                        for (int j=0;j<=i;++j){ double s=0; for(int k=0;k<dk;++k)s+=Qb[i*dk+k]*Kb[j*dk+k];
                            double p=std::exp(s*scale-m)/l; for(int k=0;k<dk;++k)acc[k]+=p*Vb[j*dk+k]; }
                        double d=0; for(int k=0;k<dk;++k)d+=dOb[i*dk+k]*acc[k]; Drow[i]=d;
                    }
                    // pass 3: block-wise recompute A[i,j], accumulate dQ, dK, dV
                    for (int i=0;i<S;++i) {
                        double m=row_m[i], l=row_l[i], D=Drow[i];
                        for (int jb=0; jb<=i; jb+=Bc) {
                            int je=std::min(i, jb+Bc-1);
                            for (int j=jb;j<=je;++j){
                                double s=0; for(int k=0;k<dk;++k)s+=Qb[i*dk+k]*Kb[j*dk+k];
                                double a=std::exp(s*scale-m)/l;         // A[i,j] recomputed
                                double dAij=0; for(int k=0;k<dk;++k)dAij+=dOb[i*dk+k]*Vb[j*dk+k];
                                double dSij=a*(dAij-D);                 // softmax jvp
                                for(int k=0;k<dk;++k){
                                    dVb[j*dk+k]+=a*dOb[i*dk+k];
                                    dQb[i*dk+k]+=dSij*Kb[j*dk+k]*scale;
                                    dKb[j*dk+k]+=dSij*Qb[i*dk+k]*scale;
                                }
                            }
                        }
                    }
                }
            }
        } else if (fn=="fused") {
            cl_kernel bk = OCL.getKernel(t->bwd_name, t->bwd_src);
            int ai=0;
            HostBuf hgo(t->size); hgo.fromDouble(t->grad,t->size);
            CLBuffer dgo(dyn_clCreateBuffer(OCL.ctx,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
                                             hgo.bytes(),hgo.data.data(),nullptr));
            dyn_clSetKernelArg(bk,ai++,sizeof(cl_mem),dgo.ptr());
            std::vector<CLBuffer> gbufs; std::vector<HostBuf> ghosts;
            for(int j=0;j<t->num_parents;++j){
                ghosts.emplace_back(t->parents[j]->size);
                ghosts.back().fromDouble(t->parents[j]->grad,t->parents[j]->size);
                gbufs.emplace_back(dyn_clCreateBuffer(OCL.ctx,CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR,
                    ghosts.back().bytes(),ghosts.back().data.data(),nullptr));
                dyn_clSetKernelArg(bk,ai++,sizeof(cl_mem),gbufs.back().ptr());
            }
            std::vector<CLBuffer> dbufs; std::vector<HostBuf> dhosts;
            for(int j=0;j<t->num_parents;++j){
                dhosts.emplace_back(t->parents[j]->size);
                dhosts.back().fromDouble(t->parents[j]->data,t->parents[j]->size);
                dbufs.emplace_back(dyn_clCreateBuffer(OCL.ctx,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
                    dhosts.back().bytes(),dhosts.back().data.data(),nullptr));
                dyn_clSetKernelArg(bk,ai++,sizeof(cl_mem),dbufs.back().ptr());
            }
            dyn_clSetKernelArg(bk,ai++,4,&t->size);
            for(int j=0;j<t->num_parents;++j)
                dyn_clSetKernelArg(bk,ai++,4,&t->parents[j]->size);
            size_t gs=t->size;
            dyn_clEnqueueNDRangeKernel(OCL.queue,bk,1,nullptr,&gs,nullptr,0,nullptr,nullptr);
            for(int j=0;j<t->num_parents;++j){
                dyn_clEnqueueReadBuffer(OCL.queue,gbufs[j],CL_TRUE,0,
                    ghosts[j].bytes(),ghosts[j].data.data(),0,nullptr,nullptr);
                ghosts[j].toDouble(t->parents[j]->grad,t->parents[j]->size);
            }
        }

        // activation memory reclamation: after gradient is propagated
        // free the intermediate node's data buffer if it won't be needed again
        // enabled by dream_enable_checkpointing(). Reduces peak memory by ~50%
        // for deep networks (ResNet-50 scale) at the cost of preventing double backward
        if (g_free_activations && !t->is_leaf && !t->is_variable && t->owns_data) {
            g_pool.release(t->data, t->size);
            t->data = nullptr;
            //keep grad alive — backward might need it for parent nodes
        }
    }
    return create_scalar(1.0);
}

