#!/bin/sh
# Measure real multi-core scaling of Dream training on YOUR machine.
# Dream's GEMM is OpenMP-parallel, so more threads = faster steps. This runs
# the same training at increasing thread counts and prints the speedup.
#
# Usage:  ./benchmark_scaling.sh [path-to-DreamCompiler]
DREAM="${1:-../../bin/DreamCompiler}"
SCRIPT="data_parallel_train.dream"

echo "=== Dream multi-core scaling benchmark ==="
echo "machine has $(nproc) CPU cores"
echo ""
printf "%-8s %-14s %-10s\n" "threads" "wall_time(s)" "speedup"

base=""
for T in 1 2 4 8; do
    # skip thread counts above core count
    if [ "$T" -gt "$(nproc)" ]; then continue; fi
    # run, extract the last numeric line (wall time)
    wt=$(OMP_NUM_THREADS=$T "$DREAM" run "$SCRIPT" 2>/dev/null | tail -1)
    if [ -z "$base" ]; then base="$wt"; fi
    sp=$(echo "scale=2; $base / $wt" | bc 2>/dev/null)
    printf "%-8s %-14s %-10s\n" "$T" "$wt" "${sp}x"
done
echo ""
echo "speedup > 1 at higher thread counts = multi-core is faster (as expected)."
echo "Note: this measures CPU-core parallelism of the GEMM. Real multi-GPU"
echo "data parallelism scales the same way (see dist_selftest for the math)."
