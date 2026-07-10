/*
 * test_both_versions.c - Verify that float and int8 models give similar results.
 *
 * Compile:
 *   gcc -O2 -I. mlp_float.c mlp_int8.c test_both_versions.c -lm -o test_both
 * Run:
 *   ./test_both
 *
 * Expected: the two versions should pick the same class most of the time.
 * Int8 may disagree on borderline cases but shouldn't be wildly different.
 */
#include <stdio.h>
#include <stdlib.h>
#include "mlp_float.h"
#include "mlp_int8.h"

int argmax(const float* arr, int n) {
    int best = 0;
    for (int i = 1; i < n; ++i) if (arr[i] > arr[best]) best = i;
    return best;
}

int main(void) {
    printf("=== Comparing float32 vs int8 inference ===\n\n");

    /* 20 random inputs, compare predictions */
    srand(42);
    int agree = 0, disagree = 0;
    float total_abs_diff = 0.0f;
    int total_values = 0;

    for (int trial = 0; trial < 20; ++trial) {
        float input[MLP_FLOAT_INPUT_DIM];
        for (int i = 0; i < MLP_FLOAT_INPUT_DIM; ++i) {
            input[i] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
        }

        float out_f[MLP_FLOAT_OUTPUT_DIM];
        float out_q[MLP_INT8_OUTPUT_DIM];
        mlp_predict_float(input, out_f);
        mlp_predict_int8(input, out_q);

        int cls_f = argmax(out_f, MLP_FLOAT_OUTPUT_DIM);
        int cls_q = argmax(out_q, MLP_INT8_OUTPUT_DIM);
        if (cls_f == cls_q) agree++; else disagree++;

        /* Measure logit-level difference */
        for (int j = 0; j < MLP_FLOAT_OUTPUT_DIM; ++j) {
            float d = out_f[j] - out_q[j];
            if (d < 0) d = -d;
            total_abs_diff += d;
            total_values++;
        }

        if (trial < 5) {
            printf("Trial %2d: float->%d   int8->%d   %s\n",
                   trial, cls_f, cls_q, cls_f == cls_q ? "ok" : "DIFFER");
        }
    }

    float avg_diff = total_abs_diff / total_values;
    printf("\nAgreement: %d/%d predictions match\n", agree, agree + disagree);
    printf("Average per-logit error: %.5f\n", avg_diff);
    printf("\nIf agreement >= 80%% and avg error < 0.1, quantization is good.\n");
    return 0;
}
