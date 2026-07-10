/* =====================================================================
 *  c_api_demo.c -- train a neural net from pure C via libdream.
 *
 *  Zero lines of Dream code. The full runtime -- tensors, autograd,
 *  AdamW, the blocked GEMM -- is driven through dream.h.
 *
 *  Task: learn y = 2*x0 - x1 + 0.5 with a 2-8-1 relu net.
 *
 *  Build:
 *    dream lib
 *    cc c_api_demo.c libdream.a -o demo -lstdc++ -lm -ldl -lgomp -Wl,--gc-sections
 * ===================================================================== */
#include <stdio.h>
#include "dream.h"

int main(void) {
    drm_seed(42);

    /* ---- data: y = 2*x0 - x1 + 0.5 ---- */
    int N = 64;
    Tensor* X = drm_randn(N, 2);
    Tensor* Y = drm_zeros(N, 1);
    for (int i = 0; i < N; ++i) {
        double x0 = dream_get(X, i * 2), x1 = dream_get(X, i * 2 + 1);
        dream_set(Y, i, 2.0 * x0 - x1 + 0.5);
    }

    /* ---- params ---- */
    Tensor* W1 = drm_kaiming(8, 2);  Tensor* b1 = drm_zeros(1, 8);
    Tensor* W2 = drm_kaiming(1, 8);  Tensor* b2 = drm_zeros(1, 1);

    /* ---- training loop: forward, backward, AdamW, release ---- */
    double final_loss = 1e9;
    for (int step = 0; step < 1000; ++step) {
        Tensor* h  = dream_linear(X, W1, b1);
        Tensor* hr = relu(h);
        Tensor* p  = dream_linear(hr, W2, b2);
        Tensor* L  = mse_loss(p, Y);
        Tensor* br = backward(L);
        final_loss = dream_get(L, 0);
        drm_adamw(W1, 0.01, 0.9, 0.999, 1e-8, 0.0);
        drm_adamw(b1, 0.01, 0.9, 0.999, 1e-8, 0.0);
        drm_adamw(W2, 0.01, 0.9, 0.999, 1e-8, 0.0);
        drm_adamw(b2, 0.01, 0.9, 0.999, 1e-8, 0.0);
        dream_release(br); dream_release(L); dream_release(p);
        dream_release(hr); dream_release(h);
        if (step % 100 == 0) printf("step %3d  loss %.6f\n", step, final_loss);
    }
    printf("final loss %.6f\n", final_loss);

    /* ---- predict f(1,1); ground truth = 2 - 1 + 0.5 = 1.5 ---- */
    Tensor* q = drm_zeros(1, 2);
    dream_set(q, 0, 1.0); dream_set(q, 1, 1.0);
    Tensor* h  = dream_linear(q, W1, b1);
    Tensor* hr = relu(h);
    Tensor* p  = dream_linear(hr, W2, b2);
    double pred = dream_get(p, 0);
    printf("f(1,1) = %.4f  (expected 1.5)\n", pred);

    int ok = (final_loss < 1e-3) && (pred > 1.4 && pred < 1.6);
    printf(ok ? "C_API_DEMO: PASS\n" : "C_API_DEMO: FAIL\n");

    dream_release(p); dream_release(hr); dream_release(h); dream_release(q);
    dream_release(W1); dream_release(b1); dream_release(W2); dream_release(b2);
    dream_release(X); dream_release(Y);
    return ok ? 0 : 1;
}
