/* C calls functions WRITTEN IN DREAM, compiled by `dream build --lib`. */
#include <stdio.h>
#include "dream.h"
#include "mathlib.h"

int main(void) {
    Tensor* x = create_scalar(2.0);
    Tensor* y = poly(x);                     /* 3*4 + 2 + 1 = 15 */
    printf("poly(2)   = %.1f  (expected 15)\n", dream_get(y, 0));

    Tensor* v = drm_ones(1, 4);
    Tensor* W = drm_ones(4, 3);
    Tensor* s = net_score(v, W);             /* sum(relu(1x4 @ 4x3 of ones)) = 12 */
    printf("net_score = %.1f  (expected 12)\n", dream_get(s, 0));

    int ok = dream_get(y, 0) == 15.0 && dream_get(s, 0) == 12.0;
    printf(ok ? "ROUTE2: PASS\n" : "ROUTE2: FAIL\n");
    dream_release(y); dream_release(x);
    dream_release(s); dream_release(W); dream_release(v);
    return !ok;
}
