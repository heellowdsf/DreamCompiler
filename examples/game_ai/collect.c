/* =====================================================================
 *  collect.c -- generate demonstration data for the dodge game.
 *
 *  The expert policy plays many episodes; every (state, action) pair is
 *  streamed to game_X.csv / game_Y.csv via dream_capture.h. In a live
 *  build this loop would instead read arrow keys from a human -- the
 *  capture API is identical; only the action source changes.
 *
 *  Build:  cc collect.c -o collect -lm
 * ===================================================================== */
#define DREAM_CAPTURE_IMPL
#include "dream_capture.h"
#include "game.h"
#include <stdio.h>

int main(int argc, char** argv) {
    int episodes = (argc > 1) ? atoi(argv[1]) : 300;
    DreamCapture* cap = dream_capture_open("game", N_FEAT);
    if (!cap) { fprintf(stderr, "cannot open capture\n"); return 1; }

    long rows = 0; int total_score = 0;
    for (int ep = 0; ep < episodes; ++ep) {
        Game g; game_reset(&g, 1234u + ep*7u);
        int steps = 0;
        while (g.alive && steps < 2000) {
            float f[N_FEAT]; game_features(&g, f);
            int a = game_expert(&g);
            dream_capture_row(cap, f, N_FEAT, (double)a);   /* label = action */
            game_step(&g, a);
            rows++; steps++;
        }
        total_score += g.score;
    }
    dream_capture_close(cap);
    printf("collected %ld rows from %d episodes (avg score %.1f)\n",
           rows, episodes, (double)total_score / episodes);
    return 0;
}
