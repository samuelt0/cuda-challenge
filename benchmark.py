"""CUDA Challenge: INT4 Quantize + GEMM Benchmark

Builds reference and solution kernels, validates correctness,
then measures performance in GB/s.
"""

import os
import sys
import torch
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GROUP_SIZE = 64

def get_cuda_flags():
    """Build CUDA compiler flags with auto-detected GPU architecture."""
    cap = torch.cuda.get_device_capability()
    arch = f"compute_{cap[0]}{cap[1]}"
    code = f"sm_{cap[0]}{cap[1]}"
    return [
        "-O2",
        f"-gencode=arch={arch},code={code}",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
    ]


def build_modules():
    """JIT-compile reference and solution CUDA modules."""
    from torch.utils.cpp_extension import load

    cuda_flags = get_cuda_flags()
    cpp_flags = ["-O2"]

    print("Building reference kernels...")
    ref = load(
        name="ref_kernels",
        sources=[
            os.path.join(SCRIPT_DIR, "reference", "pybind.cpp"),
            os.path.join(SCRIPT_DIR, "reference", "quantize_int4.cu"),
            os.path.join(SCRIPT_DIR, "reference", "gemm_int4.cu"),
        ],
        extra_cflags=cpp_flags,
        extra_cuda_cflags=cuda_flags,
        verbose=False,
    )

    print("Building solution kernels...")
    try:
        sol = load(
            name="sol_kernels",
            sources=[
                os.path.join(SCRIPT_DIR, "your_solution", "pybind.cpp"),
                os.path.join(SCRIPT_DIR, "your_solution", "kernel.cu"),
            ],
            extra_cflags=cpp_flags,
            extra_cuda_cflags=cuda_flags,
            verbose=False,
        )
    except Exception as e:
        print(f"\nERROR: Your solution failed to compile:\n{e}")
        sys.exit(1)

    return ref, sol

def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a_flat = a.float().flatten()
    b_flat = b.float().flatten()
    return (torch.dot(a_flat, b_flat) / (a_flat.norm() * b_flat.norm())).item()

# Correctness

def check_correctness(ref, sol, M, N, K, group_size=GROUP_SIZE):
    """End-to-end correctness: solution quantize+GEMM vs torch FP16 matmul.

    Returns (passed: bool, cosine: float).
    """
    torch.manual_seed(42)
    A = torch.randn(M, K, dtype=torch.float16, device="cuda")
    B = torch.randn(N, K, dtype=torch.float16, device="cuda")

    # Reference: pure torch FP16 matmul
    C_ref = (A.float() @ B.float().T).half()

    # Sanity check: reference kernels themselves should be correct
    ref_Ap, ref_As = ref.quantize_int4_naive(A.contiguous(), group_size)
    ref_Bp, ref_Bs = ref.quantize_int4_naive(B.contiguous(), group_size)
    C_ref_kernel = ref.gemm_int4_naive(ref_Ap, ref_Bp, ref_As, ref_Bs, group_size)
    ref_cos = cosine_similarity(C_ref_kernel, C_ref)
    if ref_cos < 0.97:
        print(f"  WARNING: Reference kernel itself has low similarity ({ref_cos:.4f})")
        print(f"  This may indicate a CUDA setup issue.")

    # Solution: quantize + GEMM
    sol_Ap, sol_As = sol.quantize_int4(A.contiguous(), group_size)
    sol_Bp, sol_Bs = sol.quantize_int4(B.contiguous(), group_size)
    C_sol = sol.gemm_int4(sol_Ap, sol_Bp, sol_As, sol_Bs, group_size)

    cos = cosine_similarity(C_sol, C_ref)
    passed = cos > 0.98
    return passed, cos


# Benchmark

def benchmark_kernel(fn, args, warmup=5, iters=20):
    """Time a CUDA kernel using CUDA events. Returns median time in seconds."""
    # Warmup
    for _ in range(warmup):
        fn(*args)

    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn(*args)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) / 1000.0)  # ms -> s

    times.sort()
    return times[len(times) // 2]  # median


def compute_quantize_bytes(M, K, group_size):
    """Total bytes moved for quantization: input + packed output + scales."""
    input_bytes = M * K * 2              # FP16 input
    packed_bytes = M * (K // 2)          # uint8 packed output
    scale_bytes = M * (K // group_size) * 2  # FP16 scales
    return input_bytes + packed_bytes + scale_bytes


def compute_gemm_bytes(M, N, K, group_size):
    """Total bytes moved for GEMM: packed A + packed B + scales + output."""
    a_bytes = M * (K // 2)                   # packed A
    b_bytes = N * (K // 2)                   # packed B
    sa_bytes = M * (K // group_size) * 2     # scales A
    sb_bytes = N * (K // group_size) * 2     # scales B
    c_bytes = M * N * 2                      # FP16 output
    return a_bytes + b_bytes + sa_bytes + sb_bytes + c_bytes


def run_benchmark(mod, label, M, N, K, group_size=GROUP_SIZE, warmup=5, iters=20):
    """Benchmark quantize and GEMM separately. Returns (quant_gbs, gemm_gbs)."""
    A = torch.randn(M, K, dtype=torch.float16, device="cuda")
    B = torch.randn(N, K, dtype=torch.float16, device="cuda")

    # Which function names to call
    quant_fn = getattr(mod, "quantize_int4", None) or mod.quantize_int4_naive
    gemm_fn = getattr(mod, "gemm_int4", None) or mod.gemm_int4_naive

    # Benchmark quantize
    quant_time = benchmark_kernel(
        lambda a, b: quant_fn(a, b),
        [A.contiguous(), group_size],
        warmup=warmup,
        iters=iters,
    )
    quant_bytes = compute_quantize_bytes(M, K, group_size)
    quant_gbs = quant_bytes / quant_time / 1e9

    # Prepare quantized inputs for GEMM benchmark
    Ap, As = quant_fn(A.contiguous(), group_size)
    Bp, Bs = quant_fn(B.contiguous(), group_size)

    # Benchmark GEMM
    gemm_time = benchmark_kernel(
        lambda a, b, sa, sb, gs: gemm_fn(a, b, sa, sb, gs),
        [Ap, Bp, As, Bs, group_size],
        warmup=warmup,
        iters=iters,
    )
    gemm_bytes = compute_gemm_bytes(M, N, K, group_size)
    gemm_gbs = gemm_bytes / gemm_time / 1e9

    return quant_gbs, gemm_gbs

SIZES = [
    ("Small",  256,  256,  1024),
    ("Medium", 1024, 1024, 4096),
    ("Large",  4096, 4096, 4096),
]

def main():
    parser = argparse.ArgumentParser(description="INT4 Quantize + GEMM Benchmark")
    parser.add_argument("--group-size", type=int, default=GROUP_SIZE)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    gs = args.group_size

    ref, sol = build_modules()

    gpu_name = torch.cuda.get_device_name(0)
    print(f"\nGPU: {gpu_name}")
    print(f"Group size: {gs}")

    # ---- Correctness ---- #
    print("\n" + "=" * 65)
    print("CORRECTNESS CHECK")
    print("=" * 65)

    all_passed = True
    for label, M, N, K in SIZES:
        passed, cos = check_correctness(ref, sol, M, N, K, gs)
        status = "PASS" if passed else "FAIL"
        print(f"  {label:8s} ({M}x{N}x{K}):  cosine={cos:.6f}  [{status}]")
        if not passed:
            all_passed = False

    if not all_passed:
        print("\nFAILED: Your solution did not pass correctness checks.")
        print("Fix your kernel and try again. No benchmark will run.")
        sys.exit(1)

    print("\nAll correctness checks passed.")

    # ---- Benchmark ---- #
    print("\n" + "=" * 65)
    print("PERFORMANCE BENCHMARK")
    print("=" * 65)

    header = f"  {'Size':8s} {'Dims':>18s}  {'Quant GB/s':>12s} {'GEMM GB/s':>12s}  {'Ref GEMM':>12s}  {'Speedup':>8s}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for label, M, N, K in SIZES:
        sol_q_gbs, sol_g_gbs = run_benchmark(
            sol, "solution", M, N, K, gs,
            warmup=args.warmup, iters=args.iters,
        )
        _, ref_g_gbs = run_benchmark(
            ref, "reference", M, N, K, gs,
            warmup=args.warmup, iters=args.iters,
        )
        speedup = sol_g_gbs / ref_g_gbs if ref_g_gbs > 0 else 0

        dims = f"{M}x{N}x{K}"
        print(f"  {label:8s} {dims:>18s}  {sol_q_gbs:>10.2f}   {sol_g_gbs:>10.2f}    {ref_g_gbs:>10.2f}    {speedup:>6.2f}x")

    print("\n" + "=" * 65)
    print("Done.")


if __name__ == "__main__":
    main()
