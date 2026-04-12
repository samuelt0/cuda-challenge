#include <cuda_fp16.h>
#include <cstdint>
#include <array>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

// Optimized W4A4 INT4 GEMM Kernel

// --- Configuration (must match quantize_int4_optimized.cu) ---
static constexpr int BLOCK_M     = 256;
static constexpr int BLOCK_N     = 128;
static constexpr int WARP_SZ     = 32;
static constexpr int NUM_WARPS   = 8;
static constexpr int INSN_M      = 16;
static constexpr int INSN_N      = 16;
static constexpr int INSN_K      = 64;
static constexpr int WARP_M      = BLOCK_M / NUM_WARPS;  // 32
static constexpr int WARP_N      = BLOCK_N;               // 128
static constexpr int WARP_K      = INSN_K;                // 64
static constexpr int WARP_M_TILES = WARP_M / INSN_M;     // 2
static constexpr int WARP_N_TILES = WARP_N / INSN_N;     // 8

static constexpr int ASCALES_PACK_SIZE   = 2;
static constexpr int ASCALES_NUM_PACKS   = 1;
static constexpr int ASCALES_VALID_LANES = 16;

static constexpr int WSCALES_PACK_SIZE   = 4;
static constexpr int WSCALES_NUM_PACKS   = 1;
static constexpr int WSCALES_VALID_LANES = 32;

// --- Packed types ---
struct alignas(16) packed_fpsum_t {
    half2 data[4];  // 4 x half2 = 8 halfs representing a 16x16 output tile fragment
};

struct packed_psum_t {
    int data[8];    // 8 x int32 from MMA
};

struct packed_ascale_t {
    half2 data[ASCALES_PACK_SIZE / 2];
};

struct packed_wscale_t {
    half2 data[WSCALES_PACK_SIZE / 2];
};

// --- Predicated load from global memory ---
__device__ __forceinline__ static uint4 load_pred_uint4(const uint4 *addr, bool pred) {
    uint4 data = {0, 0, 0, 0};
    if (pred) data = __ldg(addr);
    return data;
}

template<typename T>
__device__ __forceinline__ static T load_pred_t(const T *addr, bool pred) {
    T data{};
    if (pred) data = *addr;
    return data;
}

// --- Predicated store to global memory ---
template<typename T>
__device__ __forceinline__ static void store_pred_t(T *addr, const T &val, bool pred) {
    if (pred) *addr = val;
}

// --- MMA instruction: m16n8k64 INT4xINT4 -> INT32 ---
// Requires SM >= 80 for native m16n8k64, SM75 uses decomposed m8n8k32
__device__ __forceinline__ static uint4 mma_m16n8k64_s4s4s32(uint4 a, uint2 b, uint4 c) {
    uint4 d;
#if __CUDA_ARCH__ >= 800
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 "
        "{%0, %1, %2, %3},"
        "{%4, %5, %6, %7},"
        "{%8, %9},"
        "{%10, %11, %12, %13};\n"
        : "=r"(d.x), "=r"(d.y), "=r"(d.z), "=r"(d.w)
        : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w),
          "r"(b.x), "r"(b.y),
          "r"(c.x), "r"(c.y), "r"(c.z), "r"(c.w));
#else
    // SM75 fallback: decompose m16n8k64 into 4x m8n8k32
    asm volatile("{"
        ".reg .b32 t0, t1, t2, t3;"
        "mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32 "
        "{t0, t1}, {%4}, {%8}, {%10, %11};\n"
        "mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32 "
        "{t2, t3}, {%5}, {%8}, {%12, %13};\n"
        "mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32 "
        "{%0, %1}, {%6}, {%9}, {t0, t1};\n"
        "mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32 "
        "{%2, %3}, {%7}, {%9}, {t2, t3};\n"
        "}\n"
        : "=r"(d.x), "=r"(d.y), "=r"(d.z), "=r"(d.w)
        : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w),
          "r"(b.x), "r"(b.y),
          "r"(c.x), "r"(c.y), "r"(c.z), "r"(c.w));
#endif
    return d;
}

// --- INT32 to half2 conversion ---
__device__ __forceinline__ static half2 int2half2(int x, int y) {
    return half2(__int2float_rn(x), __int2float_rn(y));
}

// --- Scale broadcasting ---
// Get {k}-th and {k+1}-th wscale from block (k must be even)
__device__ __forceinline__ static half2 broadcast_wscale(
    const packed_wscale_t (&block)[WSCALES_NUM_PACKS], int k, int laneId) {
    const int packIdx    = k / (WSCALES_PACK_SIZE * WARP_SZ);
    const int srcLane    = 4 * (k / WSCALES_PACK_SIZE) + laneId % 4;
    const int elementIdx = k % WSCALES_PACK_SIZE / 2;
    return __shfl_sync(~0u, block[packIdx].data[elementIdx], srcLane);
}

// Get {k}-th and {k+1}-th ascale from block (k must be even)
__device__ __forceinline__ static half2 broadcast_ascale(
    const packed_ascale_t (&block)[ASCALES_NUM_PACKS], int k, int laneId) {
    const int packIdx    = k / (ASCALES_PACK_SIZE * WARP_SZ);
    const int srcLane    = 8 * (k / ASCALES_PACK_SIZE) + laneId / 4;
    const int elementIdx = k % ASCALES_PACK_SIZE / 2;
    return __shfl_sync(~0u, block[packIdx].data[elementIdx], srcLane);
}

// --- Load functions ---
__device__ __forceinline__ static void load_act(
    const uint4 *act, int k, int K,
    uint4 (&out)[WARP_M_TILES], bool pred) {
    int laneId = threadIdx.x % WARP_SZ;
    int warpId = threadIdx.x / WARP_SZ;
    #pragma unroll
    for (int i = 0; i < WARP_M_TILES; i++) {
        out[i] = load_pred_uint4(
            &act[((k * NUM_WARPS + warpId) * WARP_M_TILES + i) * WARP_SZ + laneId], pred);
    }
}

__device__ __forceinline__ static void load_wgt(
    const uint4 *wgt, int k, int K,
    uint4 (&out)[WARP_N_TILES], bool pred) {
    int laneId = threadIdx.x % WARP_SZ;
    const uint4 *ptr = &wgt[(0 + k * WARP_N_TILES) * WARP_SZ + laneId];
    #pragma unroll
    for (int i = 0; i < WARP_N_TILES; i++) {
        out[i] = load_pred_uint4(&ptr[i * WARP_SZ], pred);
    }
}

__device__ __forceinline__ static void load_ascale(
    const packed_ascale_t *ascales, int group, int M,
    packed_ascale_t (&out)[ASCALES_NUM_PACKS], bool pred) {
    int laneId = threadIdx.x % WARP_SZ;
    int warpId = threadIdx.x / WARP_SZ;
    #pragma unroll
    for (int i = 0; i < ASCALES_NUM_PACKS; i++) {
        out[i] = load_pred_t(
            &ascales[(group * NUM_WARPS + warpId) * ASCALES_NUM_PACKS * ASCALES_VALID_LANES +
                     i * ASCALES_VALID_LANES + laneId],
            pred && laneId < ASCALES_VALID_LANES);
    }
}

__device__ __forceinline__ static void load_wscale(
    const packed_wscale_t *wscales, int group, int N,
    packed_wscale_t (&out)[WSCALES_NUM_PACKS], bool pred) {
    int laneId = threadIdx.x % WARP_SZ;
    #pragma unroll
    for (int i = 0; i < WSCALES_NUM_PACKS; i++) {
        out[i] = load_pred_t(
            &wscales[(group * WSCALES_NUM_PACKS + i) * WSCALES_VALID_LANES + laneId],
            pred && laneId < WSCALES_VALID_LANES);
    }
}

// --- Compute: MMA + scale application ---
__device__ __forceinline__ static void compute_mma(
    uint4 (&A)[WARP_M_TILES],
    uint4 (&W)[WARP_N_TILES],
    packed_ascale_t (&ascale)[ASCALES_NUM_PACKS],
    packed_wscale_t (&wscale)[WSCALES_NUM_PACKS],
    packed_fpsum_t (&fpsum)[WARP_M_TILES * WARP_N_TILES]) {
    const int laneId = threadIdx.x % WARP_SZ;

    // Broadcast activation scales
    half2 asx[WARP_M_TILES], asy[WARP_M_TILES];
    for (int i = 0; i < WARP_M_TILES; i++) {
        half2 as = broadcast_ascale(ascale, i * 2, laneId);
        asx[i] = half2(as.x, as.x);
        asy[i] = half2(as.y, as.y);
    }

    for (int j = 0; j < WARP_N_TILES; j++) {
        half2 ws1 = broadcast_wscale(wscale, j * 4, laneId);
        half2 ws2 = broadcast_wscale(wscale, j * 4 + 2, laneId);

        for (int i = 0; i < WARP_M_TILES; i++) {
            auto &fsum = fpsum[i * WARP_N_TILES + j];

            // Two MMA calls per tile pair: first half and second half of N
            uint4 out1 = mma_m16n8k64_s4s4s32(A[i], uint2{W[j].x, W[j].y}, uint4{0,0,0,0});
            uint4 out2 = mma_m16n8k64_s4s4s32(A[i], uint2{W[j].z, W[j].w}, uint4{0,0,0,0});

            packed_psum_t psum;
            psum.data[0] = out1.x; psum.data[1] = out1.y;
            psum.data[2] = out1.z; psum.data[3] = out1.w;
            psum.data[4] = out2.x; psum.data[5] = out2.y;
            psum.data[6] = out2.z; psum.data[7] = out2.w;

            // Apply scales: fpsum += int2half(psum) * ascale * wscale
            fsum.data[0] = __hfma2(int2half2(psum.data[0], psum.data[1]), __hmul2(asx[i], ws1), fsum.data[0]);
            fsum.data[1] = __hfma2(int2half2(psum.data[2], psum.data[3]), __hmul2(asy[i], ws1), fsum.data[1]);
            fsum.data[2] = __hfma2(int2half2(psum.data[4], psum.data[5]), __hmul2(asx[i], ws2), fsum.data[2]);
            fsum.data[3] = __hfma2(int2half2(psum.data[6], psum.data[7]), __hmul2(asy[i], ws2), fsum.data[3]);
        }
    }
}

// --- Epilogue: unpack fpsum registers to global memory ---
__device__ __forceinline__ static void epilogue_default(
    packed_fpsum_t (&fpsum)[WARP_M_TILES * WARP_N_TILES],
    half *output, int stride, int maxRows, int maxCols) {
    const int laneId = threadIdx.x % WARP_SZ;
    constexpr int PACK_SZ = WARP_N / WARP_SZ;  // 4

    // Shared memory for register-to-memory transposition (+8 padding to avoid bank conflicts)
    __shared__ alignas(128) half shmem[NUM_WARPS][8][WARP_N + 8];
    const int warpId = threadIdx.x / WARP_SZ;
    auto &mat = shmem[warpId];

    #pragma unroll
    for (int i = 0; i < WARP_M_TILES; i++) {
        // Write first half (rows 0-7) from registers to shared memory
        #pragma unroll
        for (int j = 0; j < WARP_N_TILES; j++) {
            auto &fsum = fpsum[i * WARP_N_TILES + j];
            int row = laneId / 4;
            int col = laneId % 4 * 2 + j * INSN_N;
            *reinterpret_cast<half2 *>(&mat[row][col + 0]) = fsum.data[0];
            *reinterpret_cast<half2 *>(&mat[row][col + 8]) = fsum.data[2];
        }
        __syncwarp();

        // Read from shared memory and store to global memory
        #pragma unroll
        for (int row = 0; row < 8; row++) {
            bool pred = (i * INSN_M + row < maxRows) && (laneId * PACK_SZ < maxCols);
            if (pred) {
                *reinterpret_cast<uint2 *>(&output[(i * INSN_M + row) * stride + laneId * PACK_SZ]) =
                    *reinterpret_cast<uint2 *>(&mat[row][laneId * PACK_SZ]);
            }
        }
        __syncwarp();

        // Write second half (rows 8-15)
        #pragma unroll
        for (int j = 0; j < WARP_N_TILES; j++) {
            auto &fsum = fpsum[i * WARP_N_TILES + j];
            int row = laneId / 4;
            int col = laneId % 4 * 2 + j * INSN_N;
            *reinterpret_cast<half2 *>(&mat[row][col + 0]) = fsum.data[1];
            *reinterpret_cast<half2 *>(&mat[row][col + 8]) = fsum.data[3];
        }
        __syncwarp();

        #pragma unroll
        for (int row = 0; row < 8; row++) {
            bool pred = (i * INSN_M + 8 + row < maxRows) && (laneId * PACK_SZ < maxCols);
            if (pred) {
                *reinterpret_cast<uint2 *>(&output[(i * INSN_M + 8 + row) * stride + laneId * PACK_SZ]) =
                    *reinterpret_cast<uint2 *>(&mat[row][laneId * PACK_SZ]);
            }
        }
        __syncwarp();
    }
}

// --- Main GEMM kernel ---
__global__ void gemm_w4a4_optimized_kernel(
    const uint4* __restrict__ act,
    const uint4* __restrict__ wgt,
    const packed_ascale_t* __restrict__ ascales,
    const packed_wscale_t* __restrict__ wscales,
    half* __restrict__ output,
    int M, int N, int K,
    int actualM, int actualN
) {
    int bm = blockIdx.x;
    int bn = blockIdx.y;

    constexpr int NUM_STAGES = 2;

    // Per-block offsets into the packed tensors
    int sizeActPerBlock = (K / WARP_K) * NUM_WARPS * WARP_M_TILES * WARP_SZ;
    int sizeWgtPerBlock = (K / WARP_K) * WARP_N_TILES * WARP_SZ;
    int sizeAscalesPerBlock = (K / WARP_K) * NUM_WARPS * ASCALES_NUM_PACKS * ASCALES_VALID_LANES;
    int sizeWscalesPerBlock = (K / WARP_K) * WSCALES_NUM_PACKS * WSCALES_VALID_LANES;

    const uint4 *block_act = act + bm * sizeActPerBlock;
    const uint4 *block_wgt = wgt + bn * sizeWgtPerBlock;
    const packed_ascale_t *block_ascales = ascales + bm * sizeAscalesPerBlock;
    const packed_wscale_t *block_wscales = wscales + bn * sizeWscalesPerBlock;

    // Register storage
    uint4 A[WARP_M_TILES];
    uint4 W[WARP_N_TILES];
    packed_ascale_t as_reg[ASCALES_NUM_PACKS];
    packed_wscale_t ws_reg[WSCALES_NUM_PACKS];

    // FP16 accumulators
    packed_fpsum_t fpsum[WARP_M_TILES * WARP_N_TILES];
    #pragma unroll
    for (int i = 0; i < WARP_M_TILES * WARP_N_TILES; i++) {
        fpsum[i].data[0] = __float2half2_rn(0.0f);
        fpsum[i].data[1] = __float2half2_rn(0.0f);
        fpsum[i].data[2] = __float2half2_rn(0.0f);
        fpsum[i].data[3] = __float2half2_rn(0.0f);
    }

    int numKTiles = K / WARP_K;

    // Simple k-loop (no pipelining for correctness first)
    for (int k = 0; k < numKTiles; k++) {
        load_act(block_act, k, K, A, true);
        load_wgt(block_wgt, k, K, W, true);
        load_ascale(block_ascales, k, M, as_reg, true);
        load_wscale(block_wscales, k, N, ws_reg, true);

        compute_mma(A, W, as_reg, ws_reg, fpsum);
    }


    // Epilogue: store results to global memory
    int warpId = threadIdx.x / WARP_SZ;
    int warpIdM = warpId;  // All warps in M dimension (NUM_WARPS_N = 1)
    int m_offset = bm * BLOCK_M + warpIdM * WARP_M;
    int n_offset = bn * BLOCK_N;

    epilogue_default(fpsum, output + m_offset * actualN + n_offset,
                     actualN, actualM - m_offset, actualN - n_offset);
}

// --- Host wrapper ---
torch::Tensor gemm_w4a4_optimized(
    torch::Tensor act_packed,
    torch::Tensor wgt_packed,
    torch::Tensor ascales,
    torch::Tensor wscales,
    torch::Tensor act_meta,
    torch::Tensor wgt_meta
) {
    TORCH_CHECK(act_packed.is_cuda(), "act must be CUDA");
    TORCH_CHECK(wgt_packed.is_cuda(), "wgt must be CUDA");

    // Extract dimensions from metadata tensors
    auto act_meta_cpu = act_meta.cpu();
    auto wgt_meta_cpu = wgt_meta.cpu();
    int actualM = act_meta_cpu[0].item<int>();
    int K       = act_meta_cpu[1].item<int>();
    int M_pad   = act_meta_cpu[2].item<int>();
    int K_pad   = act_meta_cpu[3].item<int>();
    int actualN = wgt_meta_cpu[0].item<int>();
    int N_pad   = wgt_meta_cpu[2].item<int>();

    TORCH_CHECK(act_meta_cpu[1].item<int>() == wgt_meta_cpu[1].item<int>(), "K dimensions must match");
    TORCH_CHECK(act_meta_cpu[3].item<int>() == wgt_meta_cpu[3].item<int>(), "K_pad must match");

    auto output = torch::zeros({actualM, actualN}, torch::TensorOptions().dtype(torch::kHalf).device(act_packed.device()));

    dim3 grid(M_pad / BLOCK_M, N_pad / BLOCK_N);
    dim3 block(WARP_SZ * NUM_WARPS);

    gemm_w4a4_optimized_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const uint4*>(act_packed.data_ptr<int32_t>()),
        reinterpret_cast<const uint4*>(wgt_packed.data_ptr<int32_t>()),
        reinterpret_cast<const packed_ascale_t*>(ascales.data_ptr<at::Half>()),
        reinterpret_cast<const packed_wscale_t*>(wscales.data_ptr<at::Half>()),
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        M_pad, N_pad, K_pad,
        actualM, actualN
    );

    return output;
}
