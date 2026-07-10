# Game AI in one loop: C game -> Dream training -> C plays

The pitch: **edit C, run one command, an AI comes out and plays your game.**
No Python, no framework install, no GPU required. The trained model is just
six small text matrices you can drop into any C program.

## Run it

    ./run.sh

which does three things:

1. **collect.c** (pure C) plays the dodge game and streams every
   `(state, action)` pair to `game_X.csv` / `game_Y.csv` via
   `dream_capture.h`. (Here an expert policy generates the demos; swapping
   in live arrow-key input is the same capture call.)
2. **train_policy.dream** learns a 5->32->32->3 policy with AdamW and
   exports the weights as plain text. Every op is gradcheck-verified.
3. **play_ai.c** (pure C) loads those weights, runs the forward pass by
   hand, and plays -- benchmarked against the expert.

Result: the learned policy matches the expert (~100% of its score).

    ./play_ai --render      # watch the AI play in the terminal

## The point

Change the game in `game.h` -- new obstacles, new features, new controls --
then `./run.sh` again. A new AI trained on the new game pops out. That edit
-> train -> play loop is the whole demo: the model is portable numbers, the
training is one Dream command, and nothing outside standard C is needed to
ship it.

## Files

    game.h           game logic + feature encoding + expert baseline
    collect.c        C data collector (dream_capture.h)
    train_policy.dream   Dream training + weight export
    play_ai.c        C runner: loads weights, plays, benchmarks
    run.sh           the one command
