// ============================================================
// Manually declared CUDA/cuBLAS types
// No CUDA Toolkit headers needed — everything is loaded at runtime.
// ============================================================
typedef int            CUresult;
typedef void*          CUdevice_ptr;
typedef void*          cudaStream_t;
typedef int            cudaError_t;

// cuBLAS opaque handle
typedef struct cublasContext* cublasHandle_t;
typedef int cublasStatus_t;
typedef int cublasOperation_t;
#define CUBLAS_OP_N       0
#define CUBLAS_STATUS_SUCCESS 0

// CUDA runtime function pointer types
typedef cudaError_t (*PFN_cudaSetDevice)   (int);
typedef cudaError_t (*PFN_cudaMalloc)      (void**, size_t);
typedef cudaError_t (*PFN_cudaFree)        (void*);
typedef cudaError_t (*PFN_cudaMemcpy)      (void*, const void*, size_t, int);
#define cudaMemcpyHostToDevice 1
#define cudaMemcpyDeviceToHost 2

// cuBLAS function pointer types
typedef cublasStatus_t (*PFN_cublasCreate) (cublasHandle_t*);
typedef cublasStatus_t (*PFN_cublasDestroy)(cublasHandle_t);
typedef cublasStatus_t (*PFN_cublasSgemm)  (cublasHandle_t,
                                             cublasOperation_t, cublasOperation_t,
                                             int, int, int,
                                             const float*,
                                             const float*, int,
                                             const float*, int,
                                             const float*,
                                             float*, int);

// FP16 Tensor Core support via cublasGemmEx
typedef int cudaDataType_t;
typedef int cublasComputeType_t;
typedef int cublasGemmAlgo_t;
#define CUDA_R_16F            2
#define CUDA_R_32F            0
#define CUBLAS_COMPUTE_32F   68
#define CUBLAS_GEMM_DEFAULT  -1

typedef cublasStatus_t (*PFN_cublasGemmEx)(cublasHandle_t,
    cublasOperation_t, cublasOperation_t,
    int, int, int,
    const void*, const void*, cudaDataType_t, int,
    const void*, cudaDataType_t, int,
    const void*, void*, cudaDataType_t, int,
    cublasComputeType_t, cublasGemmAlgo_t);

// Half-precision (FP16) conversion using IEEE 754 bit manipulation
static inline uint16_t float_to_half(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    uint16_t sign = (x >> 16) & 0x8000;
    int exp = ((x >> 23) & 0xFF) - 127 + 15;
    uint16_t frac = (x >> 13) & 0x03FF;
    if (exp <= 0)       return sign;              // underflow to zero
    if (exp >= 0x1F)    return sign | 0x7C00;     // overflow to inf
    return sign | (uint16_t)(exp << 10) | frac;
}
static inline float half_to_float(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp  = (h & 0x7C00) >> 10;
    uint32_t frac = (h & 0x03FF) << 13;
    if (exp == 0)       { return 0.f; }           // zero/denormal
    if (exp == 0x1F)    { exp = 0xFF; }           // inf/nan
    else                { exp = exp - 15 + 127; }
    uint32_t bits = sign | (exp << 23) | frac;
    float result;
    std::memcpy(&result, &bits, 4);
    return result;
}

// ============================================================
// Dynamic OpenCL loading
// ============================================================
#define DECLARE_CL(name) decltype(&name) dyn_##name = nullptr;
DECLARE_CL(clGetPlatformIDs)   DECLARE_CL(clGetDeviceIDs)
DECLARE_CL(clGetDeviceInfo)
DECLARE_CL(clCreateContext)    DECLARE_CL(clCreateCommandQueue)
DECLARE_CL(clCreateProgramWithSource) DECLARE_CL(clBuildProgram)
DECLARE_CL(clGetProgramBuildInfo)     DECLARE_CL(clCreateKernel)
DECLARE_CL(clCreateBuffer)     DECLARE_CL(clSetKernelArg)
DECLARE_CL(clEnqueueNDRangeKernel)    DECLARE_CL(clEnqueueReadBuffer)
DECLARE_CL(clEnqueueWriteBuffer)
DECLARE_CL(clFinish)           DECLARE_CL(clReleaseMemObject)
DECLARE_CL(clReleaseKernel)    DECLARE_CL(clReleaseProgram)
#undef DECLARE_CL

static bool loadOpenCL() {
    HMODULE h = LoadLibraryA("OpenCL.dll");
    if (!h) return false;
#define LOAD(n) dyn_##n = reinterpret_cast<decltype(&n)>(GetProcAddress(h, #n));
    LOAD(clGetPlatformIDs)   LOAD(clGetDeviceIDs)   LOAD(clGetDeviceInfo)
    LOAD(clCreateContext)    LOAD(clCreateCommandQueue)
    LOAD(clCreateProgramWithSource) LOAD(clBuildProgram)
    LOAD(clGetProgramBuildInfo) LOAD(clCreateKernel) LOAD(clCreateBuffer)
    LOAD(clSetKernelArg)     LOAD(clEnqueueNDRangeKernel) LOAD(clEnqueueReadBuffer)
    LOAD(clEnqueueWriteBuffer)
    LOAD(clFinish)           LOAD(clReleaseMemObject)
    LOAD(clReleaseKernel)    LOAD(clReleaseProgram)
#undef LOAD
    return true;
}

// ============================================================
// Dynamic CUDA/cuBLAS loading (no Toolkit needed at compile time)
// ============================================================
static PFN_cudaSetDevice  dyn_cudaSetDevice  = nullptr;
static PFN_cudaMalloc     dyn_cudaMalloc     = nullptr;
static PFN_cudaFree       dyn_cudaFree       = nullptr;
static PFN_cudaMemcpy     dyn_cudaMemcpy     = nullptr;
static PFN_cublasCreate   dyn_cublasCreate   = nullptr;
static PFN_cublasDestroy  dyn_cublasDestroy  = nullptr;
static PFN_cublasSgemm    dyn_cublasSgemm    = nullptr;
static PFN_cublasGemmEx   dyn_cublasGemmEx   = nullptr;

// Try to load CUDA runtime. Returns true if loaded.
static bool loadCUDARuntime() {
    // Try common DLL names across CUDA versions
    const char* names[] = {
        "cudart64_12.dll", "cudart64_11.dll", "cudart64_120.dll",
        "cudart64_110.dll", "cudart64_102.dll", "cudart64.dll", nullptr
    };
    HMODULE h = nullptr;
    for (int i = 0; names[i]; ++i) {
        h = LoadLibraryA(names[i]);
        if (h) break;
    }
    if (!h) return false;
    dyn_cudaSetDevice = (PFN_cudaSetDevice) GetProcAddress(h, "cudaSetDevice");
    dyn_cudaMalloc    = (PFN_cudaMalloc)    GetProcAddress(h, "cudaMalloc");
    dyn_cudaFree      = (PFN_cudaFree)      GetProcAddress(h, "cudaFree");
    dyn_cudaMemcpy    = (PFN_cudaMemcpy)    GetProcAddress(h, "cudaMemcpy");
    return dyn_cudaSetDevice && dyn_cudaMalloc && dyn_cudaFree && dyn_cudaMemcpy;
}

// Try to load cuBLAS. Returns true if loaded.
static bool loadCuBLAS() {
    const char* names[] = {
        "cublas64_12.dll", "cublas64_11.dll", "cublas64_120.dll",
        "cublas64_110.dll", "cublas64_102.dll", "cublas64.dll", nullptr
    };
    HMODULE h = nullptr;
    for (int i = 0; names[i]; ++i) {
        h = LoadLibraryA(names[i]);
        if (h) break;
    }
    if (!h) return false;
    dyn_cublasCreate  = (PFN_cublasCreate)  GetProcAddress(h, "cublasCreate_v2");
    dyn_cublasDestroy = (PFN_cublasDestroy) GetProcAddress(h, "cublasDestroy_v2");
    dyn_cublasSgemm   = (PFN_cublasSgemm)   GetProcAddress(h, "cublasSgemm_v2");
    // GemmEx is optional: enables FP16 Tensor Core path when available
    dyn_cublasGemmEx  = (PFN_cublasGemmEx)  GetProcAddress(h, "cublasGemmEx");
    return dyn_cublasCreate && dyn_cublasDestroy && dyn_cublasSgemm;
}

// ============================================================
// RAII helpers
// ============================================================
struct CLBuffer {
    cl_mem mem = nullptr;
    CLBuffer() = default;
    explicit CLBuffer(cl_mem m) : mem(m) {}
    ~CLBuffer() { if (mem) dyn_clReleaseMemObject(mem); }
    CLBuffer(const CLBuffer&)            = delete;
    CLBuffer& operator=(const CLBuffer&) = delete;
    CLBuffer(CLBuffer&& o) noexcept : mem(o.mem) { o.mem = nullptr; }
    operator cl_mem() const { return mem; }
    cl_mem* ptr() { return &mem; }
};

struct HostBuf {
    std::vector<float> data;
    explicit HostBuf(int n) : data(n, 0.f) {}
    void fromDouble(const double* src, int n) { for (int i=0;i<n;++i) data[i]=float(src[i]); }
    void toDouble(double* dst, int n) const   { for (int i=0;i<n;++i) dst[i]=double(data[i]); }
    // FP32 fast path: direct memcpy, zero conversion overhead
    void fromFloat(const float* src, int n) { std::copy(src, src+n, data.data()); }
    void toFloat(float* dst, int n) const   { std::copy(data.data(), data.data()+n, dst); }
    // Load from Tensor: auto-selects fast path for fp32 tensors
    void fromTensor(const Tensor* t) {
        if (t->dtype == 1 && t->fdata) fromFloat(t->fdata, t->size);
        else fromDouble(t->data, t->size);
    }
    // Write back to Tensor: sync both double and float storage
    void toTensor(Tensor* t) const {
        toDouble(t->data, t->size);
        if (t->dtype == 1 && t->fdata) toFloat(t->fdata, t->size);
    }
    size_t bytes() const { return data.size() * sizeof(float); }
};

// ============================================================
// OpenCL state
// ============================================================
struct OpenCLState {
    cl_context       ctx    = nullptr;
    cl_command_queue queue  = nullptr;
    cl_device_id     device = nullptr;
    bool             ready  = false;
    bool             failed = false;
    std::string      deviceName;
    std::map<std::string, cl_program> programs;
    std::map<std::string, cl_kernel>  kernels;

    bool setup() {
        if (ready)  return true;
        if (failed) return false;
        if (!loadOpenCL()) {
            std::cerr << "error: OpenCL.dll not found\n" << std::flush;
            failed = true; return false;
        }
        cl_uint nPlat = 0;
        if (dyn_clGetPlatformIDs(0, nullptr, &nPlat) != CL_SUCCESS || nPlat == 0) {
            std::cerr << "error: no OpenCL platform found\n" << std::flush;
            failed = true; return false;
        }
        cl_platform_id plat;
        dyn_clGetPlatformIDs(1, &plat, nullptr);
        cl_uint nDev = 0;
        if (dyn_clGetDeviceIDs(plat, CL_DEVICE_TYPE_ALL, 0, nullptr, &nDev) != CL_SUCCESS || nDev == 0) {
            std::cerr << "error: no OpenCL device found\n" << std::flush;
            failed = true; return false;
        }
        cl_int err;
        dyn_clGetDeviceIDs(plat, CL_DEVICE_TYPE_ALL, 1, &device, nullptr);
        char name[256] = {};
        dyn_clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(name), name, nullptr);
        deviceName = name;
        ctx   = dyn_clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
        queue = dyn_clCreateCommandQueue(ctx, device, 0, &err);
        ready = true;
        return true;
    }

    cl_kernel getKernel(const std::string& name, const std::string& src) {
        if (!setup()) {
            std::cerr << "fatal: no OpenCL device for kernel '" << name << "'\n" << std::flush;
            exit(1);
        }
        auto it = kernels.find(name);
        if (it != kernels.end()) return it->second;
        const char* s = src.c_str();
        cl_int err;
        cl_program p = dyn_clCreateProgramWithSource(ctx, 1, &s, nullptr, &err);
        err = dyn_clBuildProgram(p, 1, &device, nullptr, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            char log[4096] = {};
            dyn_clGetProgramBuildInfo(p, device, CL_PROGRAM_BUILD_LOG, sizeof(log), log, nullptr);
            std::cerr << "error: kernel compile failed:\n" << log << "\n" << std::flush;
            exit(1);
        }
        cl_kernel k = dyn_clCreateKernel(p, "k", &err);
        if (err != CL_SUCCESS || !k) {
            std::cerr << "error: clCreateKernel failed (err " << err << ")\n" << std::flush;
            exit(1);
        }
        kernels[name] = k;
        programs[name] = p;
        return k;
    }
} OCL;

// Queryable from generated code: lets FusedExpr pick GPU or CPU at runtime
// instead of hard-exiting on machines without OpenCL.
extern "C" int dream_gpu_available() {
    return OCL.setup() ? 1 : 0;
}

// ============================================================
// CUDA state
// ============================================================
struct CUDAState {
    cublasHandle_t handle = nullptr;
    bool           ready  = false;
} CUDA;

// Whether matmul should use cuBLAS
static bool g_use_cuda = false;

// GPU dispatch threshold: matmuls with M*N below this run on the CPU (the
// transfer + kernel-launch overhead isn't worth it for small matrices). The
// default (262144 = 512x512) was tuned on Intel integrated graphics; discrete
// GPUs with faster host<->device transfer benefit from a LOWER threshold, and
// some setups may want it higher. Override with the DREAM_GPU_THRESHOLD
// environment variable (a value in elements, i.e. M*N). Set to 0 to always
// prefer the GPU, or a huge number to effectively force CPU.
static size_t gpu_dispatch_threshold() {
    static size_t cached = 0;
    static bool init = false;
    if (!init) {
        init = true;
        cached = 262144;   // 512x512 default
        const char* env = std::getenv("DREAM_GPU_THRESHOLD");
        if (env && *env) {
            char* end = nullptr;
            unsigned long long v = std::strtoull(env, &end, 10);
            if (end != env) cached = (size_t)v;
        }
    }
    return cached;
}
static bool g_use_fp16 = false;         // FP16 Tensor Core mode (set by dream_enable_fp16)
static bool g_free_activations = false; // Free intermediate activations during backward

// Detect GPU and pick the best available backend.
// Called automatically the first time any GPU op runs.
static void initBackend() {
    static bool done = false;
    if (done) return;
    done = true;

    if (!OCL.setup()) {
        std::cerr << "warning: no OpenCL device — GPU ops unavailable\n" << std::flush;
        return;
    }

    bool isNvidia = OCL.deviceName.find("NVIDIA") != std::string::npos;

    if (!isNvidia) {
        std::cerr << "[dream] backend: OpenCL  (device: " << OCL.deviceName << ")\n" << std::flush;
        return;
    }

    // NVIDIA detected — try to load CUDA driver and cuBLAS
    if (!loadCUDARuntime()) {
        std::cerr << "[dream] NVIDIA GPU detected but CUDA driver not found — using OpenCL\n"
                  << "        Install NVIDIA GPU drivers from https://www.nvidia.com/drivers\n"
                  << std::flush;
        std::cerr << "[dream] backend: OpenCL  (device: " << OCL.deviceName << ")\n" << std::flush;
        return;
    }

    if (!loadCuBLAS()) {
        std::cerr << "[dream] NVIDIA GPU detected, CUDA driver found, but cuBLAS not found — using OpenCL\n"
                  << "        Install CUDA Toolkit from https://developer.nvidia.com/cuda-downloads\n"
                  << std::flush;
        std::cerr << "[dream] backend: OpenCL  (device: " << OCL.deviceName << ")\n" << std::flush;
        return;
    }

    // All good — initialize cuBLAS handle
    if (dyn_cudaSetDevice(0) != 0) {
        std::cerr << "[dream] cudaSetDevice failed — using OpenCL\n" << std::flush;
        return;
    }
    if (dyn_cublasCreate(&CUDA.handle) != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "[dream] cublasCreate failed — using OpenCL\n" << std::flush;
        return;
    }

    CUDA.ready  = true;
    g_use_cuda  = true;
    std::cerr << "[dream] backend: CUDA  (device: " << OCL.deviceName << ")"
              << (dyn_cublasGemmEx ? " [FP16 Tensor Core available]" : "") << "\n" << std::flush;
}

// ============================================================
// GPU dispatch helpers
// ============================================================
// Threshold: tensors smaller than this use CPU directly (avoids PCI-e round-trip)
static const int GPU_THRESHOLD = 4096;

// Device memory pool: caches freed GPU buffers by size for reuse.
// Eliminates the per-op cudaMalloc/cudaFree and clCreateBuffer/clReleaseMemObject overhead
// that would otherwise dominate multi-layer networks (each @ or + triggers 3 alloc/free cycles).
struct DeviceMemPool {
    std::map<size_t, std::vector<void*>> cuda_bufs;
    std::map<size_t, std::vector<cl_mem>> cl_bufs;

    void* cudaGet(size_t bytes) {
        auto it = cuda_bufs.find(bytes);
        if (it != cuda_bufs.end() && !it->second.empty()) {
            void* p = it->second.back();
            it->second.pop_back();
            return p;
        }
        void* p = nullptr;
        dyn_cudaMalloc(&p, bytes);
        return p;
    }
    void cudaPut(void* p, size_t bytes) {
        if (!p) return;
        auto& v = cuda_bufs[bytes];
        if (v.size() < 64) v.push_back(p);
        else dyn_cudaFree(p);
    }

    cl_mem clGet(size_t bytes) {
        auto it = cl_bufs.find(bytes);
        if (it != cl_bufs.end() && !it->second.empty()) {
            cl_mem m = it->second.back();
            it->second.pop_back();
            return m;
        }
        return dyn_clCreateBuffer(OCL.ctx, CL_MEM_READ_WRITE, bytes, nullptr, nullptr);
    }
    void clPut(cl_mem m, size_t bytes) {
        if (!m) return;
        auto& v = cl_bufs[bytes];
        if (v.size() < 64) v.push_back(m);
        else dyn_clReleaseMemObject(m);
    }

    void clear() {
        for (auto& kv : cuda_bufs)
            for (auto* p : kv.second) if (dyn_cudaFree) dyn_cudaFree(p);
        for (auto& kv : cl_bufs)
            for (auto m : kv.second) if (dyn_clReleaseMemObject) dyn_clReleaseMemObject(m);
        cuda_bufs.clear();
        cl_bufs.clear();
    }
    ~DeviceMemPool() { clear(); }
};
static DeviceMemPool g_dev_pool;

// RAII wrapper for pooled OpenCL buffer (returns to pool on destruction instead of freeing)
struct CLPoolBuf {
    cl_mem mem = nullptr;
    size_t sz = 0;
    CLPoolBuf() = default;
    explicit CLPoolBuf(size_t bytes) : sz(bytes) { mem = g_dev_pool.clGet(bytes); }
    ~CLPoolBuf() { if (mem) g_dev_pool.clPut(mem, sz); }
    CLPoolBuf(const CLPoolBuf&) = delete;
    CLPoolBuf& operator=(const CLPoolBuf&) = delete;
    CLPoolBuf(CLPoolBuf&& o) noexcept : mem(o.mem), sz(o.sz) { o.mem = nullptr; }
    operator cl_mem() const { return mem; }
    cl_mem* ptr() { return &mem; }
    void upload(const void* data) {
        dyn_clEnqueueWriteBuffer(OCL.queue, mem, CL_TRUE, 0, sz, data, 0, nullptr, nullptr);
    }
    void download(void* data) {
        dyn_clEnqueueReadBuffer(OCL.queue, mem, CL_TRUE, 0, sz, data, 0, nullptr, nullptr);
    }
};

// Compute broadcast strides: stride=0 when dim==1 (broadcast), else contiguous stride.
// This replaces the old modulo hack with proper zero-stride semantics.
static void bcastStrides(const Tensor* t, int out[4]) {
    out[0] = (t->d0 > 1) ? t->d1 * t->d2 * t->d3 : 0;
    out[1] = (t->d1 > 1) ? t->d2 * t->d3           : 0;
    out[2] = (t->d2 > 1) ? t->d3                     : 0;
    out[3] = (t->d3 > 1) ? 1                          : 0;
}

// Opcode-based dispatch: string compared ONCE outside the loop, not per-element.
// The old version did 14 strcmps per element — for a 128x128 add, that's 229K
// string compares just to add two matrices. Now the branch is hoisted outside
// and each inner loop is a simple vectorizable arithmetic op.
enum BinopCode {
    BOP_UNKNOWN = -1,
    BOP_ADD = 0, BOP_SUB, BOP_MUL, BOP_DIV, BOP_MOD,
    BOP_GT, BOP_LT, BOP_GE, BOP_LE, BOP_EQ, BOP_NE,
    BOP_AND, BOP_OR, BOP_ATAN2, BOP_POW, BOP_MIN, BOP_MAX
};
static BinopCode classify_binop(const std::string& op) {
    if (op == "a_val+b_val") return BOP_ADD;
    if (op == "a_val-b_val") return BOP_SUB;
    if (op == "a_val*b_val") return BOP_MUL;
    if (op == "a_val/b_val") return BOP_DIV;
    if (op == "fmod(a_val,b_val)") return BOP_MOD;
    if (op == "a_val>b_val?1.f:0.f")  return BOP_GT;
    if (op == "a_val<b_val?1.f:0.f")  return BOP_LT;
    if (op == "a_val>=b_val?1.f:0.f") return BOP_GE;
    if (op == "a_val<=b_val?1.f:0.f") return BOP_LE;
    if (op == "fabs(a_val-b_val)<1e-6f?1.f:0.f")  return BOP_EQ;
    if (op == "fabs(a_val-b_val)>=1e-6f?1.f:0.f") return BOP_NE;
    if (op == "(a_val>0.f&&b_val>0.f)?1.f:0.f") return BOP_AND;
    if (op == "(a_val>0.f||b_val>0.f)?1.f:0.f") return BOP_OR;
    if (op == "atan2(a_val,b_val)") return BOP_ATAN2;
    if (op == "pow(a_val,b_val)")   return BOP_POW;
    if (op == "a_val<b_val?a_val:b_val") return BOP_MIN;
    if (op == "a_val>b_val?a_val:b_val") return BOP_MAX;
    return BOP_UNKNOWN;
}

// Fast path: no broadcasting needed, same shape, contiguous.
// OpenMP parallel for large tensors (>=65536 elements = 256x256).
// Small tensors use single-thread (OMP overhead > compute).
#define KERNEL_SAME_SHAPE(OP_EXPR) \
    if (sz >= 65536) { \
        _Pragma("omp parallel for schedule(static)") \
        for (size_t i = 0; i < (size_t)sz; ++i) c->data[i] = OP_EXPR; \
    } else { \
        for (size_t i = 0; i < (size_t)sz; ++i) c->data[i] = OP_EXPR; \
    } \
    return true;

static bool runCPUBinary(Tensor* a, Tensor* b, Tensor* c, const std::string& op) {
    int sz = c->size;
    BinopCode code = classify_binop(op);
    if (code == BOP_UNKNOWN) return false;

    // SIMD-FRIENDLY FAST PATH: same shape, no broadcasting.
    // Most binary ops in training hit this path (z1 + b1, a * b, etc.)
    // Simple linear loops that the compiler auto-vectorizes.
    if (a->size == sz && b->size == sz) {
        const double* A = a->data;
        const double* B = b->data;
        double* C = c->data;
        switch (code) {
            case BOP_ADD: KERNEL_SAME_SHAPE(A[i] + B[i])
            case BOP_SUB: KERNEL_SAME_SHAPE(A[i] - B[i])
            case BOP_MUL: KERNEL_SAME_SHAPE(A[i] * B[i])
            case BOP_DIV: KERNEL_SAME_SHAPE(A[i] / B[i])
            case BOP_MOD: KERNEL_SAME_SHAPE(std::fmod(A[i], B[i]))
            case BOP_GT:  KERNEL_SAME_SHAPE(A[i] >  B[i] ? 1.0 : 0.0)
            case BOP_LT:  KERNEL_SAME_SHAPE(A[i] <  B[i] ? 1.0 : 0.0)
            case BOP_GE:  KERNEL_SAME_SHAPE(A[i] >= B[i] ? 1.0 : 0.0)
            case BOP_LE:  KERNEL_SAME_SHAPE(A[i] <= B[i] ? 1.0 : 0.0)
            case BOP_EQ:  KERNEL_SAME_SHAPE(std::fabs(A[i]-B[i]) < 1e-6 ? 1.0 : 0.0)
            case BOP_NE:  KERNEL_SAME_SHAPE(std::fabs(A[i]-B[i]) >= 1e-6 ? 1.0 : 0.0)
            case BOP_AND: KERNEL_SAME_SHAPE((A[i] > 0 && B[i] > 0) ? 1.0 : 0.0)
            case BOP_OR:  KERNEL_SAME_SHAPE((A[i] > 0 || B[i] > 0) ? 1.0 : 0.0)
            case BOP_ATAN2: KERNEL_SAME_SHAPE(std::atan2(A[i], B[i]))
            case BOP_POW:   KERNEL_SAME_SHAPE(std::pow(A[i], B[i]))
            case BOP_MIN:   KERNEL_SAME_SHAPE(A[i] < B[i] ? A[i] : B[i])
            case BOP_MAX:   KERNEL_SAME_SHAPE(A[i] > B[i] ? A[i] : B[i])
            default: return false;
        }
    }

    // BROADCASTING PATH: slower but handles row/column broadcasting.
    // Common case: bias add (N, D) + (1, D) — broadcast bias across batch.
    int as[4], bs[4];
    bcastStrides(a, as);
    bcastStrides(b, bs);
    int cd3=c->d3, cd2=c->d2, cd1=c->d1;

    // Dispatch once outside the loop (not per-element)
    #define BCAST_LOOP(OP_EXPR) \
        for (int i = 0; i < sz; ++i) { \
            int i3=i%cd3, i2=(i/cd3)%cd2, i1=(i/(cd3*cd2))%cd1, i0=i/(cd3*cd2*cd1); \
            int ai = i0*as[0] + i1*as[1] + i2*as[2] + i3*as[3]; \
            int bi = i0*bs[0] + i1*bs[1] + i2*bs[2] + i3*bs[3]; \
            double av = a->data[ai], bv = b->data[bi]; \
            c->data[i] = (OP_EXPR); \
        } \
        return true;

    switch (code) {
        case BOP_ADD: BCAST_LOOP(av + bv)
        case BOP_SUB: BCAST_LOOP(av - bv)
        case BOP_MUL: BCAST_LOOP(av * bv)
        case BOP_DIV: BCAST_LOOP(av / bv)
        case BOP_MOD: BCAST_LOOP(std::fmod(av, bv))
        case BOP_GT:  BCAST_LOOP(av >  bv ? 1.0 : 0.0)
        case BOP_LT:  BCAST_LOOP(av <  bv ? 1.0 : 0.0)
        case BOP_GE:  BCAST_LOOP(av >= bv ? 1.0 : 0.0)
        case BOP_LE:  BCAST_LOOP(av <= bv ? 1.0 : 0.0)
        case BOP_EQ:  BCAST_LOOP(std::fabs(av-bv) < 1e-6 ? 1.0 : 0.0)
        case BOP_NE:  BCAST_LOOP(std::fabs(av-bv) >= 1e-6 ? 1.0 : 0.0)
        case BOP_AND: BCAST_LOOP((av > 0 && bv > 0) ? 1.0 : 0.0)
        case BOP_OR:  BCAST_LOOP((av > 0 || bv > 0) ? 1.0 : 0.0)
        case BOP_ATAN2: BCAST_LOOP(std::atan2(av, bv))
        case BOP_POW:   BCAST_LOOP(std::pow(av, bv))
        case BOP_MIN:   BCAST_LOOP(av < bv ? av : bv)
        case BOP_MAX:   BCAST_LOOP(av > bv ? av : bv)
        default: return false;
    }
    #undef BCAST_LOOP
}
#undef KERNEL_SAME_SHAPE

// Unary op: dispatch once, vectorizable inner loop.
// Same pattern as binary — hoist the op identity out of the hot loop.
enum UnopCode {
    UOP_UNKNOWN = -1,
    UOP_RELU = 0, UOP_EXP, UOP_LOG, UOP_SQRT, UOP_ABS,
    UOP_SIGMOID, UOP_TANH, UOP_LNOT,
    UOP_SIN, UOP_COS, UOP_TAN, UOP_ATAN,
    UOP_FLOOR, UOP_CEIL, UOP_ROUND
};
static UnopCode classify_unop(const std::string& op) {
    if (op == "a_val>0.f?a_val:0.f") return UOP_RELU;
    if (op == "exp(a_val)")   return UOP_EXP;
    if (op == "log(a_val)")   return UOP_LOG;
    if (op == "sqrt(a_val)")  return UOP_SQRT;
    if (op == "fabs(a_val)")  return UOP_ABS;
    if (op == "1.f/(1.f+exp(-a_val))") return UOP_SIGMOID;
    if (op == "tanh(a_val)")  return UOP_TANH;
    if (op == "a_val==0.f?1.f:0.f") return UOP_LNOT;
    if (op == "sin(a_val)")   return UOP_SIN;
    if (op == "cos(a_val)")   return UOP_COS;
    if (op == "tan(a_val)")   return UOP_TAN;
    if (op == "atan(a_val)")  return UOP_ATAN;
    if (op == "floor(a_val)") return UOP_FLOOR;
    if (op == "ceil(a_val)")  return UOP_CEIL;
    if (op == "round(a_val)") return UOP_ROUND;
    return UOP_UNKNOWN;
}

static bool runCPUUnary(Tensor* a, Tensor* c, const std::string& op) {
    UnopCode code = classify_unop(op);
    if (code == UOP_UNKNOWN) return false;
    int sz = a->size;
    const double* A = a->data;
    double* C = c->data;

    #define UOP_LOOP(EXPR) \
        if (sz >= 65536) { \
            _Pragma("omp parallel for schedule(static)") \
            for (int i = 0; i < sz; ++i) C[i] = (EXPR); \
        } else { \
            for (int i = 0; i < sz; ++i) C[i] = (EXPR); \
        } \
        return true;

    switch (code) {
        // ReLU as branchless max(0, x) — compiler vectorizes this to single SSE instruction.
        // ~3x faster than ternary operator on modern CPUs.
        case UOP_RELU:    UOP_LOOP(A[i] > 0.0 ? A[i] : 0.0)
        case UOP_EXP:     UOP_LOOP(std::exp(A[i]))
        case UOP_LOG:     UOP_LOOP(std::log(A[i]))
        case UOP_SQRT:    UOP_LOOP(std::sqrt(A[i]))
        case UOP_ABS:     UOP_LOOP(std::fabs(A[i]))
        case UOP_SIGMOID: UOP_LOOP(1.0 / (1.0 + std::exp(-A[i])))
        case UOP_TANH:    UOP_LOOP(std::tanh(A[i]))
        case UOP_LNOT:    UOP_LOOP(A[i] == 0.0 ? 1.0 : 0.0)
        case UOP_SIN:     UOP_LOOP(std::sin(A[i]))
        case UOP_COS:     UOP_LOOP(std::cos(A[i]))
        case UOP_TAN:     UOP_LOOP(std::tan(A[i]))
        case UOP_ATAN:    UOP_LOOP(std::atan(A[i]))
        case UOP_FLOOR:   UOP_LOOP(std::floor(A[i]))
        case UOP_CEIL:    UOP_LOOP(std::ceil(A[i]))
        case UOP_ROUND:   UOP_LOOP(std::round(A[i]))
        default: return false;
    }
    #undef UOP_LOOP
}

static void runGPU(Tensor* a, Tensor* b, Tensor* c, const std::string& op) {
    a = materialize(a);
    b = materialize(b);
    if (c->size <= 4194304) {
        if (runCPUBinary(a, b, c, op)) return;
    }
    initBackend();
    // Elementwise kernels are OpenCL-only. Without a device, force the CPU
    // interpreter for any size instead of exit(1)-ing inside getKernel.
    if (OCL.failed) {
        if (runCPUBinary(a, b, c, op)) return;
        std::cerr << "RUNTIME ERROR: binary op '" << op
                  << "' has no CPU fallback and no OpenCL device is available\n" << std::flush;
        return;
    }
    int as[4], bs[4];
    bcastStrides(a, as);
    bcastStrides(b, bs);
    std::string src =
        "__kernel void k(__global const float* A, __global const float* B,"
        " __global float* C, int sz,"
        " int cd0,int cd1,int cd2,int cd3,"
        " int as0,int as1,int as2,int as3,"
        " int bs0,int bs1,int bs2,int bs3) {"
        " int i=get_global_id(0); if(i<sz){"
        " int i3=i%cd3; int i2=(i/cd3)%cd2; int i1=(i/(cd3*cd2))%cd1; int i0=i/(cd3*cd2*cd1);"
        " float a_val=A[i0*as0+i1*as1+i2*as2+i3*as3];"
        " float b_val=B[i0*bs0+i1*bs1+i2*bs2+i3*bs3];"
        " C[i]=" + op + ";}}";
    cl_kernel k = OCL.getKernel("kb_"+op, src);
    HostBuf ha(a->size), hb(b->size), hc(c->size);
    ha.fromTensor(a); hb.fromTensor(b);
    CLPoolBuf da(ha.bytes()); da.upload(ha.data.data());
    CLPoolBuf db(hb.bytes()); db.upload(hb.data.data());
    CLPoolBuf dc(hc.bytes());
    int ai=0;
    dyn_clSetKernelArg(k, ai++, sizeof(cl_mem), da.ptr());
    dyn_clSetKernelArg(k, ai++, sizeof(cl_mem), db.ptr());
    dyn_clSetKernelArg(k, ai++, sizeof(cl_mem), dc.ptr());
    dyn_clSetKernelArg(k, ai++, 4, &c->size);
    dyn_clSetKernelArg(k, ai++, 4, &c->d0); dyn_clSetKernelArg(k, ai++, 4, &c->d1);
    dyn_clSetKernelArg(k, ai++, 4, &c->d2); dyn_clSetKernelArg(k, ai++, 4, &c->d3);
    dyn_clSetKernelArg(k, ai++, 4, &as[0]); dyn_clSetKernelArg(k, ai++, 4, &as[1]);
    dyn_clSetKernelArg(k, ai++, 4, &as[2]); dyn_clSetKernelArg(k, ai++, 4, &as[3]);
    dyn_clSetKernelArg(k, ai++, 4, &bs[0]); dyn_clSetKernelArg(k, ai++, 4, &bs[1]);
    dyn_clSetKernelArg(k, ai++, 4, &bs[2]); dyn_clSetKernelArg(k, ai++, 4, &bs[3]);
    size_t gs = c->size;
    dyn_clEnqueueNDRangeKernel(OCL.queue, k, 1, nullptr, &gs, nullptr, 0, nullptr, nullptr);
    dc.download(hc.data.data());
    hc.toTensor(c);
}

static void runGPUUnary(Tensor* a, Tensor* c, const std::string& op) {
    a = materialize(a);
    // Same reasoning as binary: CPU is faster than GPU overhead for most sizes.
    if (a->size <= 4194304) {
        if (runCPUUnary(a, c, op)) return;
    }
    initBackend();
    if (OCL.failed) {
        if (runCPUUnary(a, c, op)) return;
        std::cerr << "RUNTIME ERROR: unary op '" << op
                  << "' has no CPU fallback and no OpenCL device is available\n" << std::flush;
        return;
    }
    std::string src =
        "__kernel void k(__global const float* A, __global float* C, int sz) {"
        " int i=get_global_id(0); if(i<sz){ float a_val=A[i]; C[i]=" + op + ";}}";
    cl_kernel k = OCL.getKernel("ku_"+op, src);
    HostBuf ha(a->size), hc_buf(c->size);
    ha.fromTensor(a);
    CLPoolBuf da(ha.bytes()); da.upload(ha.data.data());
    CLPoolBuf dc(hc_buf.bytes());
    dyn_clSetKernelArg(k, 0, sizeof(cl_mem), da.ptr());
    dyn_clSetKernelArg(k, 1, sizeof(cl_mem), dc.ptr());
    dyn_clSetKernelArg(k, 2, 4, &a->size);
    size_t gs = a->size;
    dyn_clEnqueueNDRangeKernel(OCL.queue, k, 1, nullptr, &gs, nullptr, 0, nullptr, nullptr);
    dc.download(hc_buf.data.data());
    hc_buf.toTensor(c);
}

static void checkShape(Tensor* a, Tensor* b, const char* op) {
    if (!a || !b) {
        std::cerr << "RUNTIME ERROR in '" << op << "': null operand"
                  << (a ? "" : " (left=null)") << (b ? "" : " (right=null)") << "\n" << std::flush;
        return;
    }
    if (!a->data || !b->data) {
        std::cerr << "RUNTIME ERROR in '" << op << "': null data pointer"
                  << " left=" << a->d0 << "x" << a->d1 << (a->data ? "" : "(data=NULL)")
                  << " right=" << b->d0 << "x" << b->d1 << (b->data ? "" : "(data=NULL)") << "\n" << std::flush;
        return;
    }
    auto ok = [](int x, int y) { return x == y || x == 1 || y == 1; };
    if (!ok(a->d0,b->d0) || !ok(a->d1,b->d1) || !ok(a->d2,b->d2) || !ok(a->d3,b->d3)) {
        std::cerr << "warning: shape mismatch in '" << op << "' ("
                  << a->d0 << "x" << a->d1 << " vs " << b->d0 << "x" << b->d1
                  << ") — broadcasting\n" << std::flush;
    }
}

// Iterative topological sort (post-order DFS).
// Recursive version overflows the stack on deep computation graphs (>500 ops)
// which commonly arise in training loops. Windows default stack is only 1MB.
static void topoSort(Tensor* root, std::vector<Tensor*>& out, std::unordered_set<Tensor*>& seen) {
    if (!root) return;
    struct Frame { Tensor* node; bool expanded; };
    std::vector<Frame> stack;
    stack.reserve(256);
    stack.push_back({root, false});
    while (!stack.empty()) {
        Tensor* cur = stack.back().node;
        bool exp = stack.back().expanded;
        if (!cur) { stack.pop_back(); continue; }
        // CRITICAL: check expanded FIRST, before seen.
        // When we revisit a node after its children are done,
        // it's already in 'seen' — but we must still emit it.
        if (exp) {
            out.push_back(cur);
            stack.pop_back();
            continue;
        }
        // Only skip if we haven't started processing this node yet
        if (seen.count(cur)) { stack.pop_back(); continue; }
        stack.back().expanded = true;
        seen.insert(cur);
        // Push children (they'll be processed before this node)
        // Nodes with an explicit parents[] array (fused kernels, attention,
        // flash_attention, tensor-parallel matmuls) enumerate their inputs
        // through parents -- traverse them so gradients propagate PAST these
        // ops to whatever produced their inputs (e.g. the QKV projection
        // before attention). Previously only "fused" did this, so gradients
        // died at attention's inputs and never reached the projection weights.
        if (cur->parents && cur->num_parents > 0) {
            for (int i = cur->num_parents - 1; i >= 0; --i)
                stack.push_back({cur->parents[i], false});
        } else {
            if (cur->rhs) stack.push_back({cur->rhs, false});
            if (cur->lhs) stack.push_back({cur->lhs, false});
        }
    }
}

