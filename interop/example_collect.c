/* =====================================================================
 *  example_collect.c - demo of dream_capture.h
 *
 *  Simulates a "C project" (here a toy binary-classification problem)
 *  collecting training data. In practice this could be a game AI, a
 *  sensor program, a simulator -- any C code that produces
 *  "features + label". The graphics/IO parts run as usual; you only
 *  add these few capture lines.
 *
 *  Build and run:
 *    gcc -O2 example_collect.c -o collect -lm
 *    ./collect
 *  Produces demo_X.csv / demo_Y.csv for example_train.dream.
 * =================================================================== */
#define DREAM_CAPTURE_IMPL
#include "dream_capture.h"
#include <math.h>
#include <stdlib.h>

/* Ground truth: a point inside the unit circle is 1, outside is 0
 * (the rule Dream must learn). */
static int label_of(float x, float y) {
    return (x * x + y * y < 1.0f) ? 1 : 0;
}

int main(void) {
    DreamCapture* cap = dream_capture_open("demo", 2);
    if (!cap) { fprintf(stderr, "could not open data channel\n"); return 1; }

    srand(1234);
    for (int i = 0; i < 20000; ++i) {
        float x = (float)rand() / RAND_MAX * 4.0f - 2.0f;   /* [-2, 2] */
        float y = (float)rand() / RAND_MAX * 4.0f - 2.0f;
        float feat[2] = { x, y };
        int   label   = label_of(x, y);
        dream_capture_row(cap, feat, 2, (double)label);
    }

    printf("collected %ld samples -> demo_X.csv / demo_Y.csv\n",
           dream_capture_count(cap));
    dream_capture_close(cap);
    return 0;
}
