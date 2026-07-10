// ---------------------------------------------------------------------
// Create the parent directories of a file path on demand (cross-platform).
// So ckpt_save / save_model auto-create dirs like "models/" when writing,
// and the compiler no longer pre-creates any project directories.
// ---------------------------------------------------------------------
#if defined(_WIN32)
#include <direct.h>
#define DREAM_MKDIR(p) _mkdir(p)
#else
#include <sys/stat.h>
#include <sys/types.h>
#define DREAM_MKDIR(p) mkdir(p, 0755)
#endif
static void dream_mkdirs_for(const std::string& filepath) {
    for (size_t i = 1; i < filepath.size(); ++i) {
        char c = filepath[i];
        if (c == '/' || c == '\\') {
            std::string dir = filepath.substr(0, i);
            if (!dir.empty() && dir != "." && dir != "..")
                DREAM_MKDIR(dir.c_str());   // silently failing if it exists is fine
        }
    }
}

// Dream Runtime - Memory Pool
// Caches freed double[] buffers by size bucket for reuse.
// Eliminates the cost of repeated malloc/free in training loops.

struct MemoryPool {
    // Buckets: size -> list of free buffers
    std::map<int, std::vector<double*>> free_bufs;
    std::map<int, std::vector<double*>> free_grad_bufs;
    std::map<int, std::vector<float*>>  free_f32_bufs;   // f32 mirror (fdata)
    int hits = 0, misses = 0;

    // DIAGNOSTIC: Set to true to disable buffer recycling.
    // If crashes disappear with recycling off, the bug is in stale pointer
    // reuse (Tensor freed but its data pointer reused for a new tensor while
    // old graph nodes still reference it).
    static constexpr bool DISABLE_RECYCLE = false;

    double* alloc(int n) {
        if (!DISABLE_RECYCLE) {
            auto it = free_bufs.find(n);
            if (it != free_bufs.end() && !it->second.empty()) {
                double* p = it->second.back();
                it->second.pop_back();
                std::fill(p, p + n, 0.0);
                hits++;
                return p;
            }
        }
        misses++;
        return new double[n]();
    }

    double* alloc_grad(int n) {
        if (!DISABLE_RECYCLE) {
            auto it = free_grad_bufs.find(n);
            if (it != free_grad_bufs.end() && !it->second.empty()) {
                double* p = it->second.back();
                it->second.pop_back();
                std::fill(p, p + n, 0.0);
                return p;
            }
        }
        return new double[n]();
    }

    // A buffer larger than this element count (~8MB) is not pooled; return it to the OS.
    // Otherwise a large temporary would sit in the pool and long training wastes GBs.
    static constexpr int POOL_MAX_ELEMS = 1 << 20;   // 1M doubles = 8MB

    // Per-size-bucket cache cap (small buffers are reused often, so keep more).
    static int bucketCap(int n) {
        if (n <= 4096)    return 512;   // small tensors: training hot path, cache more
        if (n <= 65536)   return 128;
        if (n <= 262144)  return 32;
        return 8;                        // large tensors: cache few, save memory
    }

    void release(double* p, int n) {
        if (!p) return;
        if (DISABLE_RECYCLE || n > POOL_MAX_ELEMS) {
            delete[] p;
            return;
        }
        auto& bucket = free_bufs[n];
        if ((int)bucket.size() < bucketCap(n))
            bucket.push_back(p);
        else
            delete[] p;
    }

    void release_grad(double* p, int n) {
        if (!p) return;
        if (DISABLE_RECYCLE || n > POOL_MAX_ELEMS) {
            delete[] p;
            return;
        }
        auto& bucket = free_grad_bufs[n];
        if ((int)bucket.size() < bucketCap(n))
            bucket.push_back(p);
        else
            delete[] p;
    }

    // f32 mirror buffers (fdata). Same recycling as the double path so the
    // f32 matmul path stops calling `new float[]` on every op -- that heap
    // churn, not the compute, was what made f32 slower than f64 end-to-end.
    float* alloc_f32(int n) {
        if (!DISABLE_RECYCLE) {
            auto it = free_f32_bufs.find(n);
            if (it != free_f32_bufs.end() && !it->second.empty()) {
                float* p = it->second.back();
                it->second.pop_back();
                std::fill(p, p + n, 0.0f);
                return p;
            }
        }
        return new float[n]();
    }

    void release_f32(float* p, int n) {
        if (!p) return;
        if (DISABLE_RECYCLE || n > POOL_MAX_ELEMS) {
            delete[] p;
            return;
        }
        auto& bucket = free_f32_bufs[n];
        if ((int)bucket.size() < bucketCap(n))
            bucket.push_back(p);
        else
            delete[] p;
    }

    void clear() {
        for (auto& kv : free_bufs)
            for (auto* p : kv.second) delete[] p;
        for (auto& kv : free_grad_bufs)
            for (auto* p : kv.second) delete[] p;
        for (auto& kv : free_f32_bufs)
            for (auto* p : kv.second) delete[] p;
        free_bufs.clear();
        free_grad_bufs.clear();
        free_f32_bufs.clear();
    }

    int pool_size() const {
        int n = 0;
        for (auto& kv : free_bufs) n += (int)kv.second.size();
        for (auto& kv : free_grad_bufs) n += (int)kv.second.size();
        return n;
    }

    ~MemoryPool() { clear(); }
};

static MemoryPool g_pool;
