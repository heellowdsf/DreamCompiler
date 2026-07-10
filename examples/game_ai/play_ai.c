/* =====================================================================
 *  play_ai.c -- the trained policy plays the dodge game, in pure C.
 *
 *  Loads the weight matrices exported by train_policy.dream and runs the
 *  5->32->32->3 forward pass by hand (no libdream needed -- this shows the
 *  model is just numbers you can deploy anywhere). Compares the learned
 *  policy against the expert over many episodes, and renders one game.
 *
 *  Build:  cc play_ai.c -o play_ai -lm
 * ===================================================================== */
#include "game.h"
#include <stdio.h>
#include <string.h>
#include <math.h>

typedef struct { int R, C; double* v; } Mat;

static Mat load_mat(const char* fn) {
    FILE* f = fopen(fn, "r");
    if (!f) { fprintf(stderr, "missing %s (run train_policy.dream first)\n", fn); exit(1); }
    Mat m; if (fscanf(f, "%d %d", &m.R, &m.C) != 2) { exit(1); }
    m.v = (double*)malloc(sizeof(double) * m.R * m.C);
    for (int i = 0; i < m.R * m.C; ++i) if (fscanf(f, "%lf", &m.v[i]) != 1) { exit(1); }
    fclose(f); return m;
}

/* y = x @ W^T + b, W is (out x in) matching Dream's linear layout. */
static void linear(const double* x, const Mat* W, const Mat* b, double* y) {
    for (int o = 0; o < W->R; ++o) {
        double s = b->v[o];
        for (int i = 0; i < W->C; ++i) s += x[i] * W->v[o * W->C + i];
        y[o] = s;
    }
}
static void relu(double* v, int n){ for (int i=0;i<n;++i) if (v[i]<0) v[i]=0; }
static int argmax(const double* v, int n){ int m=0; for(int i=1;i<n;++i) if(v[i]>v[m])m=i; return m; }

static Mat W1,b1,W2,b2,W3,b3;
static int policy(const Game* g) {
    float f[N_FEAT]; game_features(g, f);
    double x[N_FEAT]; for (int i=0;i<N_FEAT;++i) x[i]=f[i];
    double h1[32], h2[32], logits[N_ACT];
    linear(x, &W1, &b1, h1); relu(h1, 32);
    linear(h1, &W2, &b2, h2); relu(h2, 32);
    linear(h2, &W3, &b3, logits);
    return argmax(logits, N_ACT);
}

int main(int argc, char** argv) {
    W1=load_mat("policy_W1.txt"); b1=load_mat("policy_b1.txt");
    W2=load_mat("policy_W2.txt"); b2=load_mat("policy_b2.txt");
    W3=load_mat("policy_W3.txt"); b3=load_mat("policy_b3.txt");

    /* ---- benchmark: AI vs expert over 200 episodes ---- */
    int N = 200; long ai_score = 0, ex_score = 0;
    for (int ep = 0; ep < N; ++ep) {
        Game g; game_reset(&g, 9000u + ep*13u);
        int steps=0; while (g.alive && steps<3000){ game_step(&g, policy(&g)); steps++; }
        ai_score += g.score;
        Game e; game_reset(&e, 9000u + ep*13u);
        steps=0; while (e.alive && steps<3000){ game_step(&e, game_expert(&e)); steps++; }
        ex_score += e.score;
    }
    double ai_avg = (double)ai_score/N, ex_avg = (double)ex_score/N;
    printf("=== dodge game: learned policy vs expert (%d episodes) ===\n", N);
    printf("  AI     avg score: %.1f\n", ai_avg);
    printf("  expert avg score: %.1f\n", ex_avg);
    printf("  AI reached %.0f%% of expert\n", 100.0*ai_avg/ex_avg);

    /* ---- show one AI-played game if asked ---- */
    if (argc > 1 && strcmp(argv[1], "--render") == 0) {
        Game g; game_reset(&g, 42u);
        char buf[(GRID_W+1)*GRID_H + 1];
        int steps=0;
        while (g.alive && steps<80) {
            game_render(&g, buf);
            printf("\033[H\033[2J%s\nAI playing  score=%d\n", buf, g.score);
            game_step(&g, policy(&g));
            steps++;
        }
        printf("game over, score=%d\n", g.score);
    }
    int ok = ai_avg >= 0.8 * ex_avg;
    printf(ok ? "GAME_AI: PASS (AI learned to play)\n" : "GAME_AI: FAIL\n");
    return !ok;
}
