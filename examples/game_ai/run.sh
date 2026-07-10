#!/bin/sh
# One command: collect demos -> train in Dream -> AI plays. The whole
# game-AI loop. Change game.h (rules/features), rerun, new AI comes out.
set -e
DREAM="${DREAM:-../../bin/DreamCompiler}"
echo "[1/3] collecting demonstration data (C) ..."
cc collect.c -o collect -lm && ./collect 40 >/dev/null
python3 - <<'PY'
lx=open('game_X.csv').readlines(); ly=open('game_Y.csv').readlines()
n=(len(lx)//64)*64
open('game_X.csv','w').writelines(lx[:n]); open('game_Y.csv','w').writelines(ly[:n])
print(f"      {n} (state, action) rows")
PY
echo "[2/3] training policy (Dream) ..."
"$DREAM" run train_policy.dream 2>/dev/null | grep -E "epoch 11|exported"
echo "[3/3] AI plays (C) ..."
cc play_ai.c -o play_ai -lm && ./play_ai
echo
echo "watch it play:  ./play_ai --render"
