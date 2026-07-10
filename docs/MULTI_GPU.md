# Multi-GPU training in Dream

Dream targets large-model training, which requires scaling past one GPU.
This document is the engineering blueprint: what is implemented and verified,
what is interface-only pending real hardware, and why the theory is sound.

## The honest boundary

Distributed-training correctness is ~90% mathematics (how gradients combine,
how matrices split and recombine) and ~10% communication engineering (the
actual NCCL transport). The math is verified here in a single process; the
transport is an interface that a real multi-GPU launch fills in with NCCL.

- **Verified in-process (no GPU needed):** the gradient math, the ring
  all-reduce schedule, gradient bucketing, and that the parallel result
  equals the single-device result bit-for-bit.
- **Interface-only, needs your hardware:** real async NCCL transfers, the
  actual speedup, cross-machine networking, fault recovery.

Because the verified math is independent of the transport, swapping the
simulated all-reduce for real NCCL changes only speed, not the numbers.

## Three parallelism strategies

### 1. Data parallelism (implemented + verified)
Every GPU holds the full model and processes a different data shard. After
backward, gradients are all-reduced (summed/averaged) so every replica takes
the same step. Solves throughput: N GPUs ~= N x batch.

Core guarantee (`dist_selftest`, tests/19): N-way data parallel produces the
exact same gradient as a single device with N x the batch -- because the
gradient is a sum over samples, which partitions cleanly across shards.
Verified bit-identical for K = 2,3,4,6,8.

### 2. Tensor parallelism (design + interface)
One large matmul (e.g. attention QKV projection) is split column-wise across
GPUs; partial outputs are all-gathered. Solves "one layer too big for one
GPU". The matmul-split math is standard; the interface mirrors data_shard.

### 3. Pipeline parallelism (design + interface)
Layer ranges live on different GPUs (layers 1-40 on GPU0, 41-80 on GPU1...).
Micro-batches flow through the pipeline. Solves "model too deep for one GPU".

## Communication optimization (implemented + verified)

Added GPUs only add speed if communication does not dominate. Three techniques,
all verified numerically transparent (`comm_selftest`, tests/20):

1. **Gradient bucketing** -- coalesce many small gradients into a few large
   buffers; one big all-reduce replaces thousands of tiny ones.
2. **Ring all-reduce** -- each rank talks only to its two neighbors; total
   traffic is independent of GPU count (bandwidth-optimal, NCCL's algorithm).
   Verified bit-identical to naive reduction for K = 2,4,8.
3. **Compute/comm overlap** -- backward produces gradients last-layer-first,
   so a bucket starts transferring the moment it fills while the GPU keeps
   computing earlier gradients. Communication hides under computation. On
   hardware this is an async ncclAllReduce on a side stream.

## Single-GPU efficiency is unchanged

The distributed primitives are separate functions invoked only when a training
script opts into data parallelism. They do not appear in the single-GPU
forward/backward hot path, so single-GPU throughput (GEMM ~35 GFLOPS, gradcheck
48/48) is identical with or without the multi-GPU code present.

## API

    dist_init(rank, world)              set up this worker's identity
    data_shard(data, rank, world)       this worker's slice of the batch
    all_reduce_sum(grad, world)         sum gradients across workers (NCCL sum)
    all_reduce_avg(param, world)        average gradients (mean-loss convention)
    broadcast(param, src_rank)          replicate rank-0 params to all
    barrier()                           synchronization point

    dist_selftest()   verify data-parallel == single large batch
    comm_selftest()   verify bucketed ring all-reduce == naive sum

## Path to real hardware

1. Compile with CUDA + NCCL available.
2. Replace the simulated reduction in ring_all_reduce_sum with ncclAllReduce
   on a side stream (the bucket/schedule structure stays).
3. Launch one process per GPU (RANK/WORLD_SIZE from the environment).
The training script does not change -- it already calls the final API.

## Scaling efficiency -- the measure of "extreme"

Adding K GPUs is only worthwhile if throughput approaches K-times. That ratio
is scaling efficiency, and it is set by communication cost vs computation cost
-- pure arithmetic, quantifiable without hardware.

`ring_optimality_selftest` (tests/22) proves ring all-reduce is
bandwidth-optimal: per-rank traffic stays bounded by 2x the gradient for any
GPU count (1.0 -> 1.984 as K goes 2 -> 128), while naive reducer-based
all-reduce grows linearly (up to 127x at K=128). This K-independence is what
makes large clusters viable.

`scaling_report(params, batch, gpus)` predicts a model's efficiency. For a
175B-parameter model on 64 GPUs:
- naive all-reduce: 73.5 s communication vs 14.3 s compute (adding GPUs loses)
- ring all-reduce: 2.3 s communication (bandwidth-optimal)
- with compute/comm overlap: 98.4% scaling efficiency -- PyTorch DDP class

The overlap number assumes ~90% of communication hides under computation via
bucketed backward (a bucket transfers the moment it fills while earlier
gradients still compute). On real hardware the achieved number depends on the
interconnect, but the ring schedule and overlap structure -- the parts that
make 98% reachable -- are verified here.
