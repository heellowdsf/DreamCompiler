/* =====================================================================
 *  dream_capture.h - collect Dream training data from a C/C++ project
 *                    (header-only)
 *
 *  Lets any C project (game, simulator, sensor program, ...) stream
 *  "features -> label" samples to disk with one or two lines of code.
 *  Dream then reads them back with read_csv for training. Skips all
 *  graphics/rendering, captures only the data. Pure C99, no deps.
 *
 *  --- Quick start -------------------------------------------------
 *    #define DREAM_CAPTURE_IMPL      // define in exactly one .c/.cpp
 *    #include "dream_capture.h"
 *
 *    // Open a channel: 24 features, written under "tank_data"
 *    DreamCapture* cap = dream_capture_open("tank_data", 24);
 *
 *    // Record one sample per frame (feat is float[24], label is a class id)
 *    dream_capture_row(cap, feat, 24, label);
 *
 *    // Before the program exits
 *    dream_capture_close(cap);
 *
 *  Produces two files (compatible with Dream's read_csv):
 *    tank_data_X.csv   24 features per line
 *    tank_data_Y.csv   1 label per line
 *
 *  Dream side:
 *    let X = read_csv("tank_data_X.csv", 0);
 *    let Y = read_csv("tank_data_Y.csv", 0);
 *
 *  --- Sampling control (avoid flooding with duplicate frames) -----
 *    dream_capture_set_stride(cap, 3);   // record 1 of every 3 calls
 *    dream_capture_row_if(cap, cond, feat, n, label);  // record only if cond
 * =================================================================== */
#ifndef DREAM_CAPTURE_H
#define DREAM_CAPTURE_H

#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct DreamCapture DreamCapture;

/* Open a data channel. prefix sets the output file names
 * (prefix_X.csv / prefix_Y.csv). n_features is only used to validate
 * each row length; pass 0 to skip validation. Returns NULL on failure. */
DreamCapture* dream_capture_open(const char* prefix, int n_features);

/* Record one sample: features[n] floats plus a label. Returns whether a
 * row was actually written (stride sampling may skip it). label is a
 * double so both classification and regression work. */
int dream_capture_row(DreamCapture* c, const float* features, int n, double label);

/* Same, but only considered for writing when cond is nonzero (then stride applies). */
int dream_capture_row_if(DreamCapture* c, int cond, const float* features, int n, double label);

/* With stride=k, write 1 of every k "eligible" calls, downsampling duplicate frames. Default 1. */
void dream_capture_set_stride(DreamCapture* c, int stride);

/* Total number of samples written so far. */
long dream_capture_count(const DreamCapture* c);

/* Flush and close. Always call before exit or the tail buffer may be lost. */
void dream_capture_close(DreamCapture* c);

#ifdef __cplusplus
}
#endif

/* ===================================================================
 *  Implementation (define DREAM_CAPTURE_IMPL in exactly one TU)
 * =================================================================== */
#ifdef DREAM_CAPTURE_IMPL

#include <stdlib.h>
#include <string.h>

struct DreamCapture {
    FILE* fx;
    FILE* fy;
    int   n_features;
    int   stride;
    int   stride_ctr;
    long  count;
};

DreamCapture* dream_capture_open(const char* prefix, int n_features) {
    if (!prefix) return NULL;
    DreamCapture* c = (DreamCapture*)calloc(1, sizeof(DreamCapture));
    if (!c) return NULL;

    char pathx[1024], pathy[1024];
    snprintf(pathx, sizeof(pathx), "%s_X.csv", prefix);
    snprintf(pathy, sizeof(pathy), "%s_Y.csv", prefix);
    c->fx = fopen(pathx, "w");
    c->fy = fopen(pathy, "w");
    if (!c->fx || !c->fy) {
        if (c->fx) fclose(c->fx);
        if (c->fy) fclose(c->fy);
        free(c);
        return NULL;
    }
    c->n_features = n_features;
    c->stride     = 1;
    c->stride_ctr = 0;
    c->count      = 0;
    return c;
}

int dream_capture_row(DreamCapture* c, const float* features, int n, double label) {
    if (!c || !features || n <= 0) return 0;
    if (c->n_features > 0 && n != c->n_features) return 0;  /* length mismatch, skip safely */

    /* stride downsampling */
    if (c->stride > 1) {
        c->stride_ctr++;
        if (c->stride_ctr < c->stride) return 0;
        c->stride_ctr = 0;
    }

    for (int i = 0; i < n; ++i)
        fprintf(c->fx, "%s%.6g", i ? "," : "", (double)features[i]);
    fputc('\n', c->fx);
    fprintf(c->fy, "%.6g\n", label);
    c->count++;
    return 1;
}

int dream_capture_row_if(DreamCapture* c, int cond, const float* features, int n, double label) {
    if (!cond) return 0;
    return dream_capture_row(c, features, n, label);
}

void dream_capture_set_stride(DreamCapture* c, int stride) {
    if (c && stride >= 1) { c->stride = stride; c->stride_ctr = 0; }
}

long dream_capture_count(const DreamCapture* c) {
    return c ? c->count : 0;
}

void dream_capture_close(DreamCapture* c) {
    if (!c) return;
    if (c->fx) fclose(c->fx);
    if (c->fy) fclose(c->fy);
    free(c);
}

#endif /* DREAM_CAPTURE_IMPL */
#endif /* DREAM_CAPTURE_H */
