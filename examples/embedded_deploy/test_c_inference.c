/*
 * test_c_inference.c — Test harness for Dream's exported C model.
 *
 * Compile:
 *   gcc -O2 mlp_model.c test_c_inference.c -lm -o test_inference
 * Run:
 *   ./test_inference
 *
 * This proves that Dream's trained model runs as pure C — no Dream runtime,
 * no PyTorch, no Python, no dependencies other than libc/libm.
 */

#include <stdio.h>
#include <stdlib.h>
#include "mlp_model.h"

int main(void) {
    printf("=== Running Dream-trained model in pure C ===\n");
    printf("Model: %d-dim input -> %d-dim output (%d layers)\n",
           MLP_MODEL_INPUT_DIM, MLP_MODEL_OUTPUT_DIM, MLP_MODEL_NUM_LAYERS);

    /* Random test input — in real use this would be sensor data, image pixels, etc. */
    float input[MLP_MODEL_INPUT_DIM];
    srand(42);
    for (int i = 0; i < MLP_MODEL_INPUT_DIM; ++i) {
        input[i] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
    }

    /* Run inference */
    float output[MLP_MODEL_OUTPUT_DIM];
    mlp_predict(input, output);

    /* Find the predicted class (argmax of output) */
    int best = 0;
    float best_score = output[0];
    for (int i = 1; i < MLP_MODEL_OUTPUT_DIM; ++i) {
        if (output[i] > best_score) {
            best_score = output[i];
            best = i;
        }
    }

    printf("\nOutput logits:\n");
    for (int i = 0; i < MLP_MODEL_OUTPUT_DIM; ++i) {
        printf("  class %d: %8.4f%s\n", i, output[i], i == best ? "  <-- argmax" : "");
    }
    printf("\nPredicted class: %d\n", best);
    printf("\n=== C inference complete (no dependencies used) ===\n");

    return 0;
}
