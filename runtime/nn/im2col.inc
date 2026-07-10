// im2col / col2im helpers (shared between conv_ops.inc and autograd.inc)
static void im2col(const double* inp, double* col,
                   int Cin, int H, int W, int KH, int KW, int OH, int OW) {
    int r=0;
    for(int ci=0;ci<Cin;++ci)
        for(int kh=0;kh<KH;++kh)
            for(int kw=0;kw<KW;++kw){
                for(int oh=0;oh<OH;++oh)
                    for(int ow=0;ow<OW;++ow)
                        col[r*(OH*OW)+oh*OW+ow]=inp[ci*H*W+(oh+kh)*W+(ow+kw)];
                r++;
            }
}
static void col2im(const double* col, double* inp,
                   int Cin, int H, int W, int KH, int KW, int OH, int OW) {
    int r=0;
    for(int ci=0;ci<Cin;++ci)
        for(int kh=0;kh<KH;++kh)
            for(int kw=0;kw<KW;++kw){
                for(int oh=0;oh<OH;++oh)
                    for(int ow=0;ow<OW;++ow)
                        inp[ci*H*W+(oh+kh)*W+(ow+kw)]+=col[r*(OH*OW)+oh*OW+ow];
                r++;
            }
}
static void cpu_gemm(const double* A, const double* B, double* C, int M, int K, int N) {
    // Routed to the cache-blocked packed kernel (math_ops.inc); conv2d's
    // im2col GEMM gets the same speedup as matmul for free.
    dream_gemm_d(A, B, C, M, K, N);
}

// GPU-accelerated GEMM on raw double* arrays.
// Falls back to cpu_gemm for small matrices or when no GPU is available.
static void gpu_gemm(const double* A, const double* B, double* C, int M, int K, int N) {
    if (M * N < GPU_THRESHOLD) { cpu_gemm(A, B, C, M, K, N); return; }

    initBackend();
    if (OCL.failed) { cpu_gemm(A, B, C, M, K, N); return; }

    HostBuf ha(M*K), hb(K*N), hc(M*N);
    for(int i=0;i<M*K;++i) ha.data[i]=float(A[i]);
    for(int i=0;i<K*N;++i) hb.data[i]=float(B[i]);

    if (g_use_cuda) {
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
        for(int i=0;i<M*N;++i) C[i]=double(hc.data[i]);
        return;
    }

    // OpenCL path
    std::string src =
        "__kernel void k(__global const float* A, __global const float* B,"
        " __global float* C, int M,int K,int N){"
        " int r=get_global_id(0),cc=get_global_id(1);"
        " if(r<M&&cc<N){float s=0; for(int i=0;i<K;i++) s+=A[r*K+i]*B[i*N+cc]; C[r*N+cc]=s;}}";
    cl_kernel k = OCL.getKernel("gemm_conv", src);
    CLPoolBuf da(ha.bytes()); da.upload(ha.data.data());
    CLPoolBuf db(hb.bytes()); db.upload(hb.data.data());
    CLPoolBuf dc(hc.bytes());
    dyn_clSetKernelArg(k,0,sizeof(cl_mem),da.ptr());
    dyn_clSetKernelArg(k,1,sizeof(cl_mem),db.ptr());
    dyn_clSetKernelArg(k,2,sizeof(cl_mem),dc.ptr());
    dyn_clSetKernelArg(k,3,4,&M);
    dyn_clSetKernelArg(k,4,4,&K);
    dyn_clSetKernelArg(k,5,4,&N);
    size_t gs[2]={(size_t)M,(size_t)N};
    dyn_clEnqueueNDRangeKernel(OCL.queue,k,2,nullptr,gs,nullptr,0,nullptr,nullptr);
    dc.download(hc.data.data());
    for(int i=0;i<M*N;++i) C[i]=double(hc.data[i]);
}
