/* =====================================================================
 *  game.h -- a tiny terminal dodge game, shared by the human-play data
 *  collector and the AI-plays runner. Pure C, no dependencies.
 *
 *  World: player moves in a fixed column, obstacles scroll left. Each
 *  obstacle is a vertical bar with a gap; steer the player into the gap.
 *  State handed to the policy (5 features, all normalized to ~[-1,1]):
 *     player_y, dx_to_wall, gap_center_y, player_vy, gap_half_height
 *  Action (3 classes): 0 = up, 1 = stay, 2 = down.
 * ===================================================================== */
#ifndef GAME_H
#define GAME_H
#include <stdlib.h>

#define GRID_H 15
#define GRID_W 40
#define N_FEAT 5
#define N_ACT  3

typedef struct {
    double player_y, player_vy;
    double wall_x, gap_y, gap_half;
    int    score, alive, ticks;
    unsigned rng;
} Game;

static unsigned g_rand(Game* g){ g->rng = g->rng*1103515245u+12345u; return (g->rng>>16)&0x7fff; }
static double   g_frand(Game* g){ return g_rand(g)/32768.0; }

static void game_reset(Game* g, unsigned seed) {
    g->rng = seed ? seed : 1u;
    g->player_y = GRID_H/2.0; g->player_vy = 0;
    g->wall_x = GRID_W - 1;
    g->gap_half = 2.0 + g_frand(g)*1.5;
    g->gap_y = g->gap_half + g_frand(g)*(GRID_H - 2*g->gap_half);
    g->score = 0; g->alive = 1; g->ticks = 0;
}

/* Fill a normalized feature vector for the current state. */
static void game_features(const Game* g, float* f) {
    f[0] = (float)((g->player_y - GRID_H/2.0) / (GRID_H/2.0));
    f[1] = (float)((g->wall_x) / (double)GRID_W);
    f[2] = (float)((g->gap_y - GRID_H/2.0) / (GRID_H/2.0));
    f[3] = (float)(g->player_vy / 2.0);
    f[4] = (float)(g->gap_half / (GRID_H/2.0));
}

/* Advance one tick given an action; returns 1 if still alive. */
static int game_step(Game* g, int action) {
    if (action == 0) g->player_vy = -1.0;
    else if (action == 2) g->player_vy = 1.0;
    else g->player_vy *= 0.5;
    g->player_y += g->player_vy;
    if (g->player_y < 0) { g->player_y = 0; g->player_vy = 0; }
    if (g->player_y > GRID_H-1) { g->player_y = GRID_H-1; g->player_vy = 0; }

    g->wall_x -= 1.0;
    g->ticks++;
    if (g->wall_x <= 1.0) {
        /* did the player hit the gap? */
        double dy = g->player_y - g->gap_y;
        if (dy < -g->gap_half || dy > g->gap_half) { g->alive = 0; return 0; }
        g->score++;
        g->wall_x = GRID_W - 1;
        g->gap_half = 2.0 + g_frand(g)*1.5;
        g->gap_y = g->gap_half + g_frand(g)*(GRID_H - 2*g->gap_half);
    }
    return g->alive;
}

/* A hand-coded expert policy: steer toward the gap center. Used to
 * auto-generate demonstration data (so the demo needs no live keyboard),
 * and as the baseline the learned net is measured against. */
static int game_expert(const Game* g) {
    double target = g->gap_y;
    if (g->player_y < target - 0.5) return 2;   /* need to go down */
    if (g->player_y > target + 0.5) return 0;   /* need to go up */
    return 1;
}

/* ASCII render into a caller buffer (GRID_H lines of GRID_W chars + NUL). */
static void game_render(const Game* g, char* out) {
    int wx = (int)(g->wall_x + 0.5);
    int py = (int)(g->player_y + 0.5);
    char* p = out;
    for (int y = 0; y < GRID_H; ++y) {
        for (int x = 0; x < GRID_W; ++x) {
            char c = ' ';
            if (x == wx) {
                double dy = y - g->gap_y;
                c = (dy < -g->gap_half || dy > g->gap_half) ? '#' : ' ';
            }
            if (x == 3 && y == py) c = '>';
            *p++ = c;
        }
        *p++ = '\n';
    }
    *p = '\0';
}
#endif
