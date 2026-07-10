// Dream Runtime - Convolution and Pooling Operations
// Conv2D uses im2col + GEMM (helpers defined in im2col.inc)

// Helper: add bias to one sample's output (Cout channels, each OH*OW elements)
static void conv_add_bias(double* outp, const Tensor* bias, int Cout, int spatial) {
    if (!bias) return;
    for (int co = 0; co < Cout && co < bias->size; ++co)
        for (int j = 0; j < spatial; ++j)
            outp[co * spatial + j] += bias->data[co];
}

Tensor* dream_conv2d(Tensor* input, Tensor* kernel, Tensor* bias,
                     Tensor* h_t, Tensor* w_t, Tensor* kh_t, Tensor* kw_t) {
    input = materialize(input); kernel = materialize(kernel);
    if (bias) bias = materialize(bias);

    int H  = (int)h_t->data[0],  W  = (int)w_t->data[0];
    int KH = (int)kh_t->data[0], KW = (int)kw_t->data[0];
    int OH = H - KH + 1, OW = W - KW + 1;
    if (OH <= 0 || OW <= 0) {
        std::cerr << "error: conv2d kernel larger than input\n" << std::flush;
        return alloc_tensor(1, 1, 1, 1);
    }

    int N    = input->d0;
    int Cin  = input->d1 / (H * W); if (Cin < 1) Cin = 1;
    int Cout = kernel->d0;
    int col_rows = Cin * KH * KW;
    int col_cols = OH * OW;
    int out_per_sample = Cout * OH * OW;

    Tensor* out = alloc_tensor(N, out_per_sample, 1, 1);
    out->is_leaf = false; tensor_set_lhs(out, input); tensor_set_rhs(out, kernel);
    out->grad_fn = "conv2d";
    out->bwd_name = std::to_string(H)+","+std::to_string(W)+","+
                    std::to_string(KH)+","+std::to_string(KW)+","+
                    std::to_string(Cin)+","+std::to_string(Cout);

    // GEMM dimensions: kernel(M x K) @ col(K x Nc) = result(M x Nc)
    int M = Cout, K = col_rows, Nc = col_cols;
    std::vector<double> col(K * Nc);

    // CPU path: small matrices or no GPU
    if (M * Nc < GPU_THRESHOLD || (initBackend(), OCL.failed)) {
        for (int n = 0; n < N; ++n) {
            im2col(input->data + n*Cin*H*W, col.data(), Cin, H, W, KH, KW, OH, OW);
            cpu_gemm(kernel->data, col.data(), out->data + n*out_per_sample, M, K, Nc);
            conv_add_bias(out->data + n*out_per_sample, bias, Cout, OH*OW);
        }
        return out;
    }

    // GPU path: allocate all buffers ONCE, upload kernel ONCE
    HostBuf hA(M*K), hB(K*Nc), hC(M*Nc);
    for (int i = 0; i < M*K; ++i) hA.data[i] = float(kernel->data[i]);

    if (g_use_cuda) {
        float *dA=(float*)g_dev_pool.cudaGet(M*K*sizeof(float));
        float *dB=(float*)g_dev_pool.cudaGet(K*Nc*sizeof(float));
        float *dC=(float*)g_dev_pool.cudaGet(M*Nc*sizeof(float));
        dyn_cudaMemcpy(dA, hA.data.data(), M*K*sizeof(float), cudaMemcpyHostToDevice);
        float alpha = 1.f, beta = 0.f;

        for (int n = 0; n < N; ++n) {
            im2col(input->data + n*Cin*H*W, col.data(), Cin, H, W, KH, KW, OH, OW);
            for (int i = 0; i < K*Nc; ++i) hB.data[i] = float(col.data()[i]);
            dyn_cudaMemcpy(dB, hB.data.data(), K*Nc*sizeof(float), cudaMemcpyHostToDevice);
            dyn_cublasSgemm(CUDA.handle, CUBLAS_OP_N, CUBLAS_OP_N,
                            Nc, M, K, &alpha, dB, Nc, dA, K, &beta, dC, Nc);
            dyn_cudaMemcpy(hC.data.data(), dC, M*Nc*sizeof(float), cudaMemcpyDeviceToHost);
            double* outp = out->data + n * out_per_sample;
            for (int i = 0; i < M*Nc; ++i) outp[i] = double(hC.data[i]);
            conv_add_bias(outp, bias, Cout, OH*OW);
        }
        g_dev_pool.cudaPut(dA, M*K*sizeof(float));
        g_dev_pool.cudaPut(dB, K*Nc*sizeof(float));
        g_dev_pool.cudaPut(dC, M*Nc*sizeof(float));
    } else {
        std::string src =
            "__kernel void k(__global const float* A, __global const float* B,"
            " __global float* C, int M,int K,int N){"
            " int r=get_global_id(0),cc=get_global_id(1);"
            " if(r<M&&cc<N){float s=0; for(int i=0;i<K;i++) s+=A[r*K+i]*B[i*N+cc]; C[r*N+cc]=s;}}";
        cl_kernel k = OCL.getKernel("gemm_conv", src);
        CLPoolBuf dA(hA.bytes()); dA.upload(hA.data.data());
        CLPoolBuf dB(K*Nc*sizeof(float));
        CLPoolBuf dC(M*Nc*sizeof(float));
        int Marg = M, Karg = K, Narg = Nc;
        size_t gs[2] = {(size_t)M, (size_t)Nc};

        for (int n = 0; n < N; ++n) {
            im2col(input->data + n*Cin*H*W, col.data(), Cin, H, W, KH, KW, OH, OW);
            for (int i = 0; i < K*Nc; ++i) hB.data[i] = float(col.data()[i]);
            dB.upload(hB.data.data());
            dyn_clSetKernelArg(k, 0, sizeof(cl_mem), dA.ptr());
            dyn_clSetKernelArg(k, 1, sizeof(cl_mem), dB.ptr());
            dyn_clSetKernelArg(k, 2, sizeof(cl_mem), dC.ptr());
            dyn_clSetKernelArg(k, 3, 4, &Marg);
            dyn_clSetKernelArg(k, 4, 4, &Karg);
            dyn_clSetKernelArg(k, 5, 4, &Narg);
            dyn_clEnqueueNDRangeKernel(OCL.queue, k, 2, nullptr, gs, nullptr, 0, nullptr, nullptr);
            dC.download(hC.data.data());
            double* outp = out->data + n * out_per_sample;
            for (int i = 0; i < M*Nc; ++i) outp[i] = double(hC.data[i]);
            conv_add_bias(outp, bias, Cout, OH*OW);
        }
    }
    return out;
}

// MaxPool2D: input(N, C*H*W), pool_size
// Returns (N, C*OH*OW) where OH=H/pool, OW=W/pool
// Conv2D with padding and stride support
// padding: number of zeros to add on each side
// stride: step size for the sliding window
Tensor* dream_conv2d_padded(Tensor* input, Tensor* kernel, Tensor* bias,
                             Tensor* h_t, Tensor* w_t, Tensor* kh_t, Tensor* kw_t,
                             Tensor* pad_t, Tensor* stride_t) {
    input = materialize(input); kernel = materialize(kernel);
    if (bias) bias = materialize(bias);

    int H = (int)h_t->data[0], W = (int)w_t->data[0];
    int KH = (int)kh_t->data[0], KW = (int)kw_t->data[0];
    int pad = pad_t ? (int)pad_t->data[0] : 0;
    int stride = stride_t ? (int)stride_t->data[0] : 1;
    if (stride < 1) stride = 1;

    int PH = H + 2 * pad, PW = W + 2 * pad;
    int OH = (PH - KH) / stride + 1, OW = (PW - KW) / stride + 1;
    if (OH <= 0 || OW <= 0) {
        std::cerr << "error: conv2d output size <= 0\n" << std::flush;
        return alloc_tensor(1, 1, 1, 1);
    }

    int N = input->d0;
    int Cin = input->d1 / (H * W); if (Cin < 1) Cin = 1;
    int Cout = kernel->d0;
    int col_rows = Cin * KH * KW;
    int col_cols = OH * OW;
    int out_per_sample = Cout * OH * OW;

    Tensor* out = alloc_tensor(N, out_per_sample, 1, 1);
    out->is_leaf = false; tensor_set_lhs(out, input); tensor_set_rhs(out, kernel);
    out->grad_fn = "conv2d";
    out->bwd_name = std::to_string(H)+","+std::to_string(W)+","+
                    std::to_string(KH)+","+std::to_string(KW)+","+
                    std::to_string(Cin)+","+std::to_string(Cout);

    // Zero-pad input, then im2col with stride
    std::vector<double> padded(Cin * PH * PW, 0.0);
    std::vector<double> col(col_rows * col_cols);

    for (int n = 0; n < N; ++n) {
        // Copy input to padded buffer
        const double* inp = input->data + n * Cin * H * W;
        std::fill(padded.begin(), padded.end(), 0.0);
        for (int c = 0; c < Cin; ++c)
            for (int h = 0; h < H; ++h)
                for (int w = 0; w < W; ++w)
                    padded[c*PH*PW + (h+pad)*PW + (w+pad)] = inp[c*H*W + h*W + w];

        // im2col with stride
        int r = 0;
        for (int ci = 0; ci < Cin; ++ci)
            for (int kh = 0; kh < KH; ++kh)
                for (int kw = 0; kw < KW; ++kw) {
                    for (int oh = 0; oh < OH; ++oh)
                        for (int ow = 0; ow < OW; ++ow)
                            col[r*col_cols + oh*OW + ow] =
                                padded[ci*PH*PW + (oh*stride+kh)*PW + (ow*stride+kw)];
                    r++;
                }

        double* outp = out->data + n * out_per_sample;
        cpu_gemm(kernel->data, col.data(), outp, Cout, col_rows, col_cols);
        conv_add_bias(outp, bias, Cout, OH*OW);
    }
    return out;
}

// Global Average Pooling: (N, C*H*W) -> (N, C)
// Averages over all spatial positions per channel. Used in ResNet/EfficientNet.
Tensor* dream_global_avg_pool(Tensor* input, Tensor* h_t, Tensor* w_t, Tensor* c_t) {
    if (!input) return tensor_incref(input);
    input = materialize(input);
    int N = input->d0;
    int H = (int)h_t->data[0], W = (int)w_t->data[0], C = (int)c_t->data[0];
    int HW = H * W;

    Tensor* out = alloc_tensor(N, C, 1, 1);
    out->is_leaf = false; tensor_set_lhs(out, input); out->grad_fn = "global_avg_pool";
    out->bwd_name = std::to_string(H)+","+std::to_string(W)+","+std::to_string(C);

    double inv = 1.0 / HW;
    for (int n = 0; n < N; ++n)
        for (int c = 0; c < C; ++c) {
            double s = 0;
            for (int hw = 0; hw < HW; ++hw)
                s += input->data[n*C*HW + c*HW + hw];
            out->data[n*C + c] = s * inv;
        }
    return out;
}

// Zero-pad a 2D feature map: adds `pad` zeros on all sides per channel
Tensor* dream_pad2d(Tensor* input, Tensor* h_t, Tensor* w_t, Tensor* c_t, Tensor* pad_t) {
    if (!input) return tensor_incref(input);
    input = materialize(input);
    int N = input->d0, H = (int)h_t->data[0], W = (int)w_t->data[0];
    int C = (int)c_t->data[0], pad = (int)pad_t->data[0];
    int PH = H + 2*pad, PW = W + 2*pad;

    Tensor* out = alloc_tensor(N, C*PH*PW, 1, 1);
    for (int n = 0; n < N; ++n)
        for (int c = 0; c < C; ++c)
            for (int h = 0; h < H; ++h)
                for (int w = 0; w < W; ++w)
                    out->data[n*C*PH*PW + c*PH*PW + (h+pad)*PW + (w+pad)] =
                        input->data[n*C*H*W + c*H*W + h*W + w];
    return out;
}

Tensor* dream_maxpool2d(Tensor* input, Tensor* h_t, Tensor* w_t,
                        Tensor* channels_t, Tensor* pool_t) {
    input = materialize(input);
    int H = (int)h_t->data[0], W = (int)w_t->data[0];
    int C = (int)channels_t->data[0];
    int pool = (int)pool_t->data[0];
    int OH = H / pool, OW = W / pool;
    int N = input->d0;

    if (OH <= 0 || OW <= 0) {
        std::cerr << "error: maxpool output size zero\n" << std::flush;
        return alloc_tensor(1, 1, 1, 1);
    }

    Tensor* out = alloc_tensor(N, C * OH * OW, 1, 1);
    out->is_leaf = false;
    tensor_set_lhs(out, input);
    out->grad_fn = "maxpool";
    out->bwd_name = std::to_string(H) + "," + std::to_string(W) + "," +
                    std::to_string(C) + "," + std::to_string(pool);

    // Store max indices for backward pass
    Tensor* indices = alloc_tensor(N, C * OH * OW, 1, 1);
    tensor_set_rhs(out, indices);

    for (int n = 0; n < N; ++n) {
        const double* inp = input->data + n * C * H * W;
        double* outp = out->data + n * C * OH * OW;
        double* idxp = indices->data + n * C * OH * OW;
        for (int c = 0; c < C; ++c) {
            for (int oh = 0; oh < OH; ++oh) {
                for (int ow = 0; ow < OW; ++ow) {
                    double maxv = -1e30;
                    int maxidx = 0;
                    for (int ph = 0; ph < pool; ++ph) {
                        for (int pw = 0; pw < pool; ++pw) {
                            int ih = oh * pool + ph;
                            int iw = ow * pool + pw;
                            double v = inp[c*H*W + ih*W + iw];
                            if (v > maxv) { maxv = v; maxidx = ih*W + iw; }
                        }
                    }
                    outp[c*OH*OW + oh*OW + ow] = maxv;
                    idxp[c*OH*OW + oh*OW + ow] = (double)maxidx;
                }
            }
        }
    }
    return out;
}

// AvgPool2D: input(N, C*H*W), pool_size
// Returns (N, C*OH*OW) where OH=H/pool, OW=W/pool
Tensor* dream_avgpool2d(Tensor* input, Tensor* h_t, Tensor* w_t,
                         Tensor* channels_t, Tensor* pool_t) {
    input = materialize(input);
    int H = (int)h_t->data[0], W = (int)w_t->data[0];
    int C = (int)channels_t->data[0];
    int pool = (int)pool_t->data[0];
    int OH = H / pool, OW = W / pool;
    int N = input->d0;

    if (OH <= 0 || OW <= 0) {
        std::cerr << "error: avgpool output size zero\n" << std::flush;
        return alloc_tensor(1, 1, 1, 1);
    }

    Tensor* out = alloc_tensor(N, C * OH * OW, 1, 1);
    out->is_leaf = false;
    tensor_set_lhs(out, input);
    out->grad_fn = "avgpool";
    out->bwd_name = std::to_string(H) + "," + std::to_string(W) + "," +
                    std::to_string(C) + "," + std::to_string(pool);

    double inv_area = 1.0 / (pool * pool);
    for (int n = 0; n < N; ++n) {
        const double* inp = input->data + n * C * H * W;
        double* outp = out->data + n * C * OH * OW;
        for (int c = 0; c < C; ++c) {
            for (int oh = 0; oh < OH; ++oh) {
                for (int ow = 0; ow < OW; ++ow) {
                    double s = 0;
                    for (int ph = 0; ph < pool; ++ph)
                        for (int pw = 0; pw < pool; ++pw)
                            s += inp[c*H*W + (oh*pool+ph)*W + (ow*pool+pw)];
                    outp[c*OH*OW + oh*OW + ow] = s * inv_area;
                }
            }
        }
    }
    return out;
}

// Resize image tensor using bilinear interpolation
// input: (H, W) grayscale, returns (newH, newW)
Tensor* dream_resize(Tensor* img, Tensor* nh_t, Tensor* nw_t) {
    img = materialize(img);
    int srcH = img->d0, srcW = img->d1;
    int dstH = (int)nh_t->data[0], dstW = (int)nw_t->data[0];
    Tensor* out = alloc_tensor(dstH, dstW, 1, 1);

    for (int y = 0; y < dstH; ++y) {
        double sy = (double)y * (srcH - 1) / std::max(1, dstH - 1);
        int y0 = (int)sy, y1 = std::min(y0 + 1, srcH - 1);
        double fy = sy - y0;
        for (int x = 0; x < dstW; ++x) {
            double sx = (double)x * (srcW - 1) / std::max(1, dstW - 1);
            int x0 = (int)sx, x1 = std::min(x0 + 1, srcW - 1);
            double fx = sx - x0;
            double v = img->data[y0*srcW+x0] * (1-fy)*(1-fx)
                     + img->data[y0*srcW+x1] * (1-fy)*fx
                     + img->data[y1*srcW+x0] * fy*(1-fx)
                     + img->data[y1*srcW+x1] * fy*fx;
            out->data[y * dstW + x] = v;
        }
    }
    return out;
}

// Flatten: reshape (N, C*H*W) to keep as-is (already flat per sample)
// For transitioning from conv output to dense layer
Tensor* dream_flatten_2d(Tensor* input) {
    // Already stored as (N, features), just return
    return tensor_incref(input);
}

// Batch normalize (per-feature, across batch)
// Stores per-feature inv_std in rhs tensor and N,F in bwd_name for backward
Tensor* dream_batch_norm(Tensor* input) {
    if (!input || input->size == 0) return tensor_incref(input);
    input = materialize(input);
    int N = input->d0, F = input->d1;
    Tensor* out = alloc_tensor(N, F, 1, 1);
    out->is_leaf = false;
    tensor_set_lhs(out, input);
    out->grad_fn = "batchnorm";
    out->bwd_name = std::to_string(N) + "," + std::to_string(F);

    // Store per-feature inv_std for backward
    Tensor* stats = alloc_tensor(1, F, 1, 1);
    tensor_set_rhs(out, stats);

    for (int f = 0; f < F; ++f) {
        double mean = 0, var = 0;
        for (int n = 0; n < N; ++n) mean += input->data[n*F+f];
        mean /= N;
        for (int n = 0; n < N; ++n) {
            double d = input->data[n*F+f] - mean;
            var += d * d;
        }
        var /= N;
        double inv_std = 1.0 / std::sqrt(var + 1e-5);
        stats->data[f] = inv_std;
        for (int n = 0; n < N; ++n)
            out->data[n*F+f] = (input->data[n*F+f] - mean) * inv_std;
    }
    return out;
}

// BatchNorm2D with spatial dimensions: normalizes per-channel across batch AND spatial.
// Input: (N, C*H*W), each row is C feature maps of size H*W.
// For each channel c, computes mean/var over all N*H*W values (not just N).
// This is the correct normalization for convolutional networks.
Tensor* dream_batchnorm2d(Tensor* input, Tensor* h_t, Tensor* w_t, Tensor* c_t) {
    if (!input || input->size == 0) return tensor_incref(input);
    input = materialize(input);
    int N = input->d0;
    int H = (int)h_t->data[0], W = (int)w_t->data[0], C = (int)c_t->data[0];
    int HW = H * W;
    int group = N * HW; // normalization group size per channel

    Tensor* out = alloc_tensor(N, C * HW, 1, 1);
    out->is_leaf = false;
    tensor_set_lhs(out, input);
    out->grad_fn = "batchnorm2d";
    out->bwd_name = std::to_string(N) + "," + std::to_string(C) + "," +
                    std::to_string(H) + "," + std::to_string(W);

    // Store per-channel inv_std for backward
    Tensor* stats = alloc_tensor(1, C, 1, 1);
    tensor_set_rhs(out, stats);

    for (int c = 0; c < C; ++c) {
        // Compute mean across batch + spatial
        double mean = 0;
        for (int n = 0; n < N; ++n)
            for (int s = 0; s < HW; ++s)
                mean += input->data[n * C * HW + c * HW + s];
        mean /= group;

        // Compute variance
        double var = 0;
        for (int n = 0; n < N; ++n)
            for (int s = 0; s < HW; ++s) {
                double d = input->data[n * C * HW + c * HW + s] - mean;
                var += d * d;
            }
        var /= group;
        double inv_std = 1.0 / std::sqrt(var + 1e-5);
        stats->data[c] = inv_std;

        // Normalize
        for (int n = 0; n < N; ++n)
            for (int s = 0; s < HW; ++s) {
                int idx = n * C * HW + c * HW + s;
                out->data[idx] = (input->data[idx] - mean) * inv_std;
            }
    }
    return out;
}

// Generate synthetic 16x16 shape patterns for a more challenging task
// Shapes: circle, square, triangle, cross, diamond (5 classes)
static void draw_circle_16(double* p) {
    for (int y = 0; y < 16; ++y)
        for (int x = 0; x < 16; ++x) {
            double dx = x-7.5, dy = y-7.5;
            p[y*16+x] = (dx*dx+dy*dy <= 36) ? 1.0 : 0.0;
        }
}
static void draw_square_16(double* p) {
    for (int y = 0; y < 16; ++y)
        for (int x = 0; x < 16; ++x)
            p[y*16+x] = (x>=3&&x<=12&&y>=3&&y<=12) ? 1.0 : 0.0;
}
static void draw_triangle_16(double* p) {
    for (int y = 0; y < 16; ++y)
        for (int x = 0; x < 16; ++x) {
            int row = y - 2;
            int half = row / 2;
            p[y*16+x] = (row>=0 && row<12 && x>=8-half && x<=7+half) ? 1.0 : 0.0;
        }
}
static void draw_cross_16(double* p) {
    for (int y = 0; y < 16; ++y)
        for (int x = 0; x < 16; ++x)
            p[y*16+x] = ((x>=6&&x<=9)||(y>=6&&y<=9)) ? 1.0 : 0.0;
}
static void draw_diamond_16(double* p) {
    for (int y = 0; y < 16; ++y)
        for (int x = 0; x < 16; ++x) {
            int dist = std::abs(x-8) + std::abs(y-8);
            p[y*16+x] = (dist <= 6) ? 1.0 : 0.0;
        }
}

// Generate shape dataset: 16x16 images, 5 classes
// Returns tuple: (data[n*5, 256], labels[n*5, 5])
Tensor* dream_gen_shapes(Tensor* n_per_t) {
    int n_per = (int)n_per_t->data[0];
    if (n_per < 1) n_per = 1;
    int total = n_per * 5;

    Tensor* data   = alloc_tensor(total, 256, 1, 1);
    Tensor* labels = alloc_tensor(total, 5, 1, 1);
    std::fill(labels->data, labels->data + total * 5, 0.0);

    typedef void (*DrawFn)(double*);
    DrawFn drawers[5] = {draw_circle_16, draw_square_16, draw_triangle_16,
                         draw_cross_16, draw_diamond_16};

    static std::mt19937 rng(123);
    std::normal_distribution<double> noise(0.0, 0.2);
    std::uniform_real_distribution<double> shift(-2.0, 2.0);

    for (int cls = 0; cls < 5; ++cls) {
        double clean[256];
        drawers[cls](clean);
        for (int s = 0; s < n_per; ++s) {
            int idx = cls * n_per + s;
            // Add noise + slight random intensity variation
            double brightness = 0.7 + 0.6 * (rng() % 1000) / 1000.0;
            for (int p = 0; p < 256; ++p) {
                double v = clean[p] * brightness + noise(rng);
                data->data[idx * 256 + p] = std::max(0.0, std::min(1.0, v));
            }
            labels->data[idx * 5 + cls] = 1.0;
        }
    }

    Tensor** arr = new Tensor*[2]{data, labels};
    Tensor* result = dream_pack(arr, 2);
    delete[] arr;
    tensor_decref(data);
    tensor_decref(labels);
    return result;
}

// Print 16x16 image to console
Tensor* dream_print_image_16(Tensor* img) {
    if (!img || img->size < 256) {
        // Try 8x8
        if (img && img->size >= 64) {
            for (int y = 0; y < 8; ++y) {
                for (int x = 0; x < 8; ++x)
                    std::cout << (img->data[y*8+x] > 0.4 ? "##" : "  ");
                std::cout << "\n";
            }
            std::cout << std::flush;
            return create_scalar(1.0);
        }
        return create_scalar(0.0);
    }
    for (int y = 0; y < 16; ++y) {
        for (int x = 0; x < 16; ++x) {
            double v = img->data[y * 16 + x];
            if (v > 0.6)      std::cout << "@@";
            else if (v > 0.3) std::cout << "..";
            else               std::cout << "  ";
        }
        std::cout << "\n";
    }
    std::cout << std::flush;
    return create_scalar(1.0);
}
