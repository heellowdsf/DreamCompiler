// Simulate the OpenCL tiled matmul kernel's EXACT algorithm on CPU, to verify
// its tile indexing / boundary handling is correct (independent of hardware).
// Mirrors the kernel in ops/elementwise.inc line-for-line.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
using namespace std;

#define TS  32
#define WPT 4
#define RTS (TS/WPT)

// Faithful CPU re-execution of the kernel: loop over workgroups and local ids.
void kernel_sim(const float* A, const float* B, float* C, int M, int K, int N) {
    int gr_max = (M + TS - 1) / TS;
    int gc_max = (N + TS - 1) / TS;
    fill(C, C + (size_t)M*N, 0.0f);
    for (int group_row = 0; group_row < gr_max; ++group_row)
    for (int group_col = 0; group_col < gc_max; ++group_col) {
        // local memory tiles
        vector<vector<float>> A_tile(TS, vector<float>(TS));
        vector<vector<float>> B_tile(TS, vector<float>(TS));
        // per-thread accumulators: [local_row][local_col][WPT][WPT]
        static float acc[RTS][RTS][WPT][WPT];
        for (int lr=0;lr<RTS;++lr) for(int lc=0;lc<RTS;++lc)
            for(int wr=0;wr<WPT;++wr) for(int wc=0;wc<WPT;++wc) acc[lr][lc][wr][wc]=0.0f;

        int num_tiles = (K + TS - 1) / TS;
        for (int t = 0; t < num_tiles; ++t) {
            // load phase: every thread loads WPT*WPT elems of A and B
            for (int lr=0;lr<RTS;++lr) for(int lc=0;lc<RTS;++lc)
                for (int wr=0;wr<WPT;++wr) for(int wc=0;wc<WPT;++wc) {
                    int tile_row = wr*RTS + lr, tile_col = wc*RTS + lc;
                    int ga_row = group_row*TS + tile_row, ga_col = t*TS + tile_col;
                    int gb_row = t*TS + tile_row,        gb_col = group_col*TS + tile_col;
                    A_tile[tile_row][tile_col] = (ga_row<M && ga_col<K) ? A[ga_row*K+ga_col] : 0.0f;
                    B_tile[tile_row][tile_col] = (gb_row<K && gb_col<N) ? B[gb_row*N+gb_col] : 0.0f;
                }
            // compute phase
            for (int lr=0;lr<RTS;++lr) for(int lc=0;lc<RTS;++lc)
                for (int k=0;k<TS;++k) {
                    float a_reg[WPT], b_reg[WPT];
                    for (int wr=0;wr<WPT;++wr) a_reg[wr]=A_tile[wr*RTS+lr][k];
                    for (int wc=0;wc<WPT;++wc) b_reg[wc]=B_tile[k][wc*RTS+lc];
                    for (int wr=0;wr<WPT;++wr) for(int wc=0;wc<WPT;++wc)
                        acc[lr][lc][wr][wc] += a_reg[wr]*b_reg[wc];
                }
        }
        // write phase
        for (int lr=0;lr<RTS;++lr) for(int lc=0;lc<RTS;++lc)
            for (int wr=0;wr<WPT;++wr) for(int wc=0;wc<WPT;++wc) {
                int out_row = group_row*TS + wr*RTS + lr;
                int out_col = group_col*TS + wc*RTS + lc;
                if (out_row<M && out_col<N) C[out_row*N+out_col] = acc[lr][lc][wr][wc];
            }
    }
}
void ref(const float* A, const float* B, float* C, int M, int K, int N) {
    for (int i=0;i<M;++i) for(int j=0;j<N;++j){ float s=0; for(int k=0;k<K;++k) s+=A[i*K+k]*B[k*N+j]; C[i*N+j]=s; }
}
int main() {
    mt19937 rng(123); uniform_real_distribution<float> d(-1,1);
    struct S{int M,K,N;}; S sh[]={{32,32,32},{33,31,35},{256,128,256},{300,200,257},{100,100,100},{1,64,1},{65,1,63}};
    double worst=0; int pass=0,total=0;
    for (auto& s: sh) {
        vector<float> A(s.M*s.K),B(s.K*s.N),C1(s.M*s.N),C2(s.M*s.N);
        for(auto&v:A)v=d(rng); for(auto&v:B)v=d(rng);
        kernel_sim(A.data(),B.data(),C1.data(),s.M,s.K,s.N);
        ref(A.data(),B.data(),C2.data(),s.M,s.K,s.N);
        double mr=0; for(size_t i=0;i<C1.size();++i){ double dn=max((double)fabs(C2[i]),1e-6); mr=max(mr,(double)fabs(C1[i]-C2[i])/dn); }
        bool ok=mr<1e-4; printf("  %dx%dx%d  max_rel=%.2e  %s\n",s.M,s.K,s.N,mr,ok?"PASS":"FAIL");
        worst=max(worst,mr); ++total; if(ok)++pass;
    }
    printf("KERNEL_SIM: %d/%d PASS (worst %.2e)\n",pass,total,worst);
    return pass==total?0:1;
}
