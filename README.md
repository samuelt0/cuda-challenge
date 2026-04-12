# CUDA Challenge: INT4 Quantize + GEMM

Optimize a naive INT4 quantization and GEMM (matrix multiply) CUDA kernel for maximum throughput.

## Overview

You are given two naive CUDA kernels:

- **INT4 Quantization** -- Converts FP16 tensors to packed INT4 format with per-group symmetric scaling
- **INT4 GEMM** -- Computes `C = A @ B^T` where A and B are packed INT4 with per-group scales

Both kernels live in a single file: `your_solution/kernel.cu`. Your job is to make them as fast as possible while keeping the output correct.

## Quick Start

```bash
# One-time setup (creates conda env + installs PyTorch, ninja, numpy)
./setup.sh

# Run the benchmark (builds automatically on first run)
./benchmark.sh
```

The benchmark script will:
1. Compile the reference and your solution
2. Verify your kernel produces correct results
3. Report your performance in GB/s with speedup vs the reference

## What to Optimize

Edit **one file**: `your_solution/kernel.cu`

This file contains both kernels:
- `quantize_int4_kernel` + `quantize_int4_custom` -- the quantization kernel and its C++ wrapper
- `gemm_int4_kernel` + `gemm_int4_custom` -- the GEMM kernel and its C++ wrapper

The reference (naive) implementations are in `reference/` for you to study.

### The naive GEMM kernel

The starting GEMM kernel assigns **one thread per output element** with a 16x16 thread block. Each thread:
1. Loops over all groups in the K dimension
2. Unpacks INT4 values from packed bytes with sign extension
3. Computes dot product in INT32, scales by per-group FP16 scales
4. Accumulates across groups in FP32

Example optimization targets: tiling, shared memory, memory coalescing, vectorized loads, tensor cores, etc.

### The naive quantization kernel

The starting quantization kernel assigns **one thread per (row, group) pair**. Each thread:
1. Finds the max absolute value in the group (serial loop)
2. Computes the scale
3. Quantizes and packs element pairs into bytes (serial loop)

## Rules

- **Only edit** `your_solution/kernel.cu` -- do not modify `reference/`, `benchmark.py`, or `benchmark.sh`
- **External libraries are allowed** -- cuBLAS, CUTLASS, Thrust, CUB, etc. are all fair game
- **Must pass correctness** -- your end-to-end output (quantize + GEMM) must achieve cosine similarity > 0.98 vs FP16 torch matmul
- **Function signatures are fixed** -- do not change the C++ wrapper function signatures (`quantize_int4_custom` and `gemm_int4_custom`). You can freely change the CUDA kernel signatures and launch configurations.

## Scoring

Your score is the **GEMM GB/s on the Large size (4096x4096x4096)**. Higher is better.

The benchmark reports:
- Quantize GB/s (effective memory bandwidth for the quantization kernel)
- GEMM GB/s (effective memory bandwidth for the GEMM kernel)
- Reference GEMM GB/s (the naive baseline)
- Speedup vs reference

## Benchmark Sizes

| Label  | M    | N    | K    |
|--------|------|------|------|
| Small  | 256  | 256  | 1024 |
| Medium | 1024 | 1024 | 4096 |
| Large  | 4096 | 4096 | 4096 |

## Repository Structure

```
cuda-challenge/
  reference/
    quantize_int4.cu    # Reference quantization kernel (read-only)
    gemm_int4.cu        # Reference naive GEMM kernel (read-only)
    pybind.cpp          # Python bindings for reference
  your_solution/
    kernel.cu           # YOUR CODE HERE
    pybind.cpp          # Python bindings for your solution
  benchmark.py          # Correctness + performance measurement
  benchmark.sh          # Entry point
  README.md             # This file
```
