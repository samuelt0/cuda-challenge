#include <cuda_fp16.h>
#include <cstdint>
#include <array>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

// Optimized INT4 Quantization Kernel

// --- Configuration constants ---
static constexpr int BLOCK_M     = 256;
static constexpr int BLOCK_N     = 128;
static constexpr int WARP_SIZE   = 32;
static constexpr int NUM_WARPS   = 8;
static constexpr int INSN_M      = 16;
static constexpr int INSN_N      = 16;
static constexpr int INSN_K      = 64;
static constexpr int WARP_M      = BLOCK_M / NUM_WARPS;  // 32
static constexpr int WARP_N      = BLOCK_N;               // 128
static constexpr int WARP_K      = INSN_K;                // 64
static constexpr int WARP_M_TILES = WARP_M / INSN_M;     // 2
static constexpr int WARP_N_TILES = WARP_N / INSN_N;     // 8

// Scale packing constants
static constexpr int ASCALES_PACK_SIZE   = 2;
static constexpr int ASCALES_NUM_PACKS   = 1;
static constexpr int ASCALES_VALID_LANES = 16;

static constexpr int WSCALES_PACK_SIZE   = 4;
static constexpr int WSCALES_NUM_PACKS   = 1;
static constexpr int WSCALES_VALID_LANES = 32;

// --- Packed data types ---
struct packed_ascale_t {
    half2 data[ASCALES_PACK_SIZE / 2];  // 1 half2 = 2 scales
};

struct packed_wscale_t {
    half2 data[WSCALES_PACK_SIZE / 2];  // 2 half2 = 4 scales
};

// --- Utility functions ---
template<typename T>
__device__ __forceinline__ static T vec_load(const T *addr) {
    if constexpr (sizeof(T) == 16) {
        uint4 data = __ldg(reinterpret_cast<const uint4 *>(addr));
        return *reinterpret_cast<T *>(&data);
    }
    return *addr;
}

template<typename T>
__device__ __forceinline__ static void vec_store(T *addr, const T &val) {
    *addr = val;
}

// PTX INT4 packing: quantize two floats to packed signed INT4 (2 values in low 8 bits)
__device__ __forceinline__ uint32_t quantize_float2_s4(float2 value) {
    int v1, v2;
    uint32_t result;
    asm volatile("cvt.rni.s32.f32 %0, %1;" : "=r"(v1) : "f"(value.x));
    asm volatile("cvt.rni.s32.f32 %0, %1;" : "=r"(v2) : "f"(value.y));
    asm volatile("cvt.pack.sat.s4.s32.b32 %0, %1, %2, 0;" : "=r"(result) : "r"(v2), "r"(v1));
    return result;
}

// Load 8x8 matrix from shared memory in IMMA format
__device__ __forceinline__ static void ldmatrix_x4(const void *ptr, uint4 &out) {
    asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(out.x), "=r"(out.y), "=r"(out.z), "=r"(out.w)
                 : "l"(__cvta_generic_to_shared(ptr)));
}

// --- Core quantize function: one warp quantizes a 16x64 tile ---
// Input: 16 rows x 64 cols of half in global memory (row-major with stride)
// Output: packed_act_t (uint4) per thread in register, scales in shared memory
__device__ __forceinline__ static void
quantize_warp_16x64(const half *input, int stride, uint4 &output, half *output_scale, void *shmem) {
    const int laneId = threadIdx.x % WARP_SIZE;

    constexpr int QVALUE_MAX       = 7;
    constexpr int PACK_SIZE        = INSN_K / 8;             // 8 elements per pack
    constexpr int PACKS_PER_ROW    = INSN_K / PACK_SIZE;     // 8
    constexpr int ROWS_PER_ITER    = PACK_SIZE * WARP_SIZE / INSN_K;  // 4
    constexpr int NUM_ITERS        = INSN_M / ROWS_PER_ITER; // 4

    using pack_t = std::array<half, PACK_SIZE>;
    pack_t packs[NUM_ITERS];

    // Step 1: Load data — each thread loads 8 half values per iteration
    #pragma unroll
    for (int i = 0; i < NUM_ITERS; i++) {
        int rowId = i * ROWS_PER_ITER + laneId / PACKS_PER_ROW;
        int colId = laneId % PACKS_PER_ROW * PACK_SIZE;
        packs[i] = vec_load(reinterpret_cast<const pack_t *>(input + rowId * stride + colId));
    }

    // Step 2: Find max absolute value per row group
    half maxval[NUM_ITERS];
    #pragma unroll
    for (int i = 0; i < NUM_ITERS; i++) {
        maxval[i] = __habs(packs[i][0]);
        #pragma unroll
        for (int j = 1; j < PACK_SIZE; j++) {
            maxval[i] = __hmax(maxval[i], __habs(packs[i][j]));
        }
    }

    // Step 3: Warp-level max reduction across 8 lanes per row
    #pragma unroll
    for (int mask = PACKS_PER_ROW / 2; mask > 0; mask /= 2) {
        #pragma unroll
        for (int i = 0; i < NUM_ITERS; i++) {
            maxval[i] = __hmax(maxval[i], __shfl_xor_sync(~0u, maxval[i], mask));
        }
    }

    // Step 4: Broadcast max to all lanes in the same row group
    #pragma unroll
    for (int i = 0; i < NUM_ITERS; i++) {
        maxval[i] = __shfl_sync(~0u, maxval[i], laneId / PACKS_PER_ROW * PACKS_PER_ROW);
    }

    // Step 5: Quantize to INT4 and write to shared memory
    using matrix_t = uint32_t[INSN_M][PACKS_PER_ROW];
    matrix_t &mat = *reinterpret_cast<matrix_t *>(shmem);

    #pragma unroll
    for (int i = 0; i < NUM_ITERS; i++) {
        half scale  = __hdiv(maxval[i], __float2half((float)QVALUE_MAX));
        half rscale = __hdiv(__float2half((float)QVALUE_MAX), maxval[i]);

        // Store scale (one per row, only first lane in each row writes)
        if (laneId % PACKS_PER_ROW == 0) {
            output_scale[i * ROWS_PER_ITER + laneId / PACKS_PER_ROW] = scale;
        }

        // Quantize: multiply by rscale, round, clamp, pack
        uint32_t qpack = 0;
        #pragma unroll
        for (int j = 0; j < PACK_SIZE; j += 2) {
            half2 hval = __hmul2(half2(rscale, rscale), half2(packs[i][j], packs[i][j + 1]));
            float2 fval = __half22float2(hval);
            qpack |= quantize_float2_s4(fval) << (j * 4);
        }
        mat[i * ROWS_PER_ITER + laneId / PACKS_PER_ROW][laneId % PACKS_PER_ROW] = qpack;
    }
    __syncwarp();

    // Step 6: Transform to IMMA format via ldmatrix
    int row = laneId % 16;
    int col = laneId / 16 * 4;
    ldmatrix_x4(&mat[row][col], output);
    __syncwarp();
}

// --- Pack activation scales from shared memory to packed format ---
__device__ __forceinline__ static void pack_ascales_fn(const half *input, packed_ascale_t *output) {
    const int laneId = threadIdx.x % WARP_SIZE;
    #pragma unroll
    for (int j = 0; j < ASCALES_NUM_PACKS; j++) {
        if (laneId < ASCALES_VALID_LANES) {
            packed_ascale_t tmp;
            #pragma unroll
            for (int i = 0; i < ASCALES_PACK_SIZE; i += 2) {
                tmp.data[i / 2].x = input[j * ASCALES_PACK_SIZE * WARP_SIZE + laneId / 8 * 8 * ASCALES_PACK_SIZE +
                                          laneId % 8 + i * 8];
                tmp.data[i / 2].y = input[j * ASCALES_PACK_SIZE * WARP_SIZE + laneId / 8 * 8 * ASCALES_PACK_SIZE +
                                          laneId % 8 + (i + 1) * 8];
            }
            output[j * ASCALES_VALID_LANES + laneId] = tmp;
        }
    }
}

// --- Pack weight scales from shared memory to packed format ---
__device__ __forceinline__ static void pack_wscales_fn(const half *input, packed_wscale_t *output) {
    const int laneId = threadIdx.x % WARP_SIZE;
    #pragma unroll
    for (int j = 0; j < WSCALES_NUM_PACKS; j++) {
        if (laneId < WSCALES_VALID_LANES) {
            packed_wscale_t tmp;
            #pragma unroll
            for (int i = 0; i < WSCALES_PACK_SIZE; i += 2) {
                tmp.data[i / 2] = *reinterpret_cast<const half2 *>(
                    &input[j * WSCALES_PACK_SIZE * WARP_SIZE + laneId / 4 * 4 * WSCALES_PACK_SIZE + laneId % 4 * 2 +
                           i * 4]);
            }
            output[j * WSCALES_VALID_LANES + laneId] = tmp;
        }
    }
}

// --- Activation quantize kernel ---
// Each block = 1 warp (32 threads), processes WARP_M x WARP_K = 32 x 64 tile
__global__ void quantize_act_optimized_kernel(
    const half* __restrict__ input,        // [M, K]
    uint4* __restrict__ output,            // [M/BLOCK_M, K/64, NUM_WARPS, WARP_M_TILES, WARP_SIZE]
    packed_ascale_t* __restrict__ oscales,  // [M/BLOCK_M, K/64, NUM_WARPS, ASCALES_NUM_PACKS, ASCALES_VALID_LANES]
    int M, int K
) {
    const int laneId = threadIdx.x % WARP_SIZE;

    // blockIdx.x encodes both the M-block and warp-within-block
    const int bm     = blockIdx.x / (BLOCK_M / WARP_M);  // M-block index
    const int warpId = blockIdx.x % (BLOCK_M / WARP_M);  // warp within block [0, NUM_WARPS)
    const int bk     = blockIdx.y;                         // K-block index

    const int row = blockIdx.x * WARP_M;
    const int col = blockIdx.y * WARP_K;

    __shared__ alignas(128) half oscale_shmem[WARP_M];          // 32 scales
    __shared__ alignas(128) uint8_t tmp_shmem[INSN_M * INSN_K / 2];  // 512 bytes

    #pragma unroll
    for (int tileId = 0; tileId < WARP_M_TILES; tileId++) {
        uint4 tmpout;

        quantize_warp_16x64(
            input + (row + tileId * INSN_M) * K + col,
            K,
            tmpout,
            oscale_shmem + tileId * INSN_M,
            tmp_shmem
        );

        vec_store(&output[(((bm * K / WARP_K + bk) * NUM_WARPS + warpId) * WARP_M_TILES + tileId) * WARP_SIZE + laneId],
                  tmpout);
    }

    pack_ascales_fn(
        oscale_shmem,
        &oscales[((bm * K / WARP_K + bk) * NUM_WARPS + warpId) * ASCALES_NUM_PACKS * ASCALES_VALID_LANES]);
}

// --- Weight quantize kernel ---
// Each block = 1 warp (32 threads), processes WARP_N x WARP_K = 128 x 64 tile
// Note: weights are stored as [N, K] (N rows, K columns) — each row is one output channel
__global__ void quantize_wgt_optimized_kernel(
    const half* __restrict__ input,        // [N, K]
    uint4* __restrict__ output,            // [N/BLOCK_N, K/64, WARP_N_TILES, WARP_SIZE]
    packed_wscale_t* __restrict__ oscales,  // [N/BLOCK_N, K/64, WSCALES_NUM_PACKS, WSCALES_VALID_LANES]
    int N, int K
) {
    const int laneId = threadIdx.x % WARP_SIZE;

    const int bn = blockIdx.x / (BLOCK_N / WARP_N);  // = blockIdx.x (since BLOCK_N == WARP_N)
    const int bk = blockIdx.y;

    const int col = blockIdx.x * WARP_N;  // Starting output channel
    const int row = blockIdx.y * WARP_K;  // Starting K position

    __shared__ alignas(128) half oscale_shmem[WARP_N];
    __shared__ alignas(128) uint8_t tmp_shmem[INSN_M * INSN_K / 2];

    #pragma unroll
    for (int tileId = 0; tileId < WARP_N_TILES; tileId++) {
        uint4 tmpout;

        // Weights: quantize along K dimension for each output channel
        // Input is [N, K], so (col + tileId*INSN_N) is the row, row is the column
        quantize_warp_16x64(
            input + (col + tileId * INSN_N) * K + row,
            K,
            tmpout,
            oscale_shmem + tileId * INSN_N,
            tmp_shmem
        );

        // Swap y and z for weight transpose (column-major for GEMM)
        uint32_t tmp = tmpout.y;
        tmpout.y = tmpout.z;
        tmpout.z = tmp;

        vec_store(&output[((bn * K / WARP_K + bk) * WARP_N_TILES + tileId) * WARP_SIZE + laneId], tmpout);
    }

    pack_wscales_fn(oscale_shmem, &oscales[(bn * K / WARP_K + bk) * WSCALES_NUM_PACKS * WSCALES_VALID_LANES]);
}

// --- Host wrapper: quantize activations ---
std::vector<torch::Tensor> quantize_act_optimized(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "input must be CUDA");
    TORCH_CHECK(input.dtype() == torch::kHalf, "input must be float16");
    TORCH_CHECK(input.dim() == 2, "input must be 2D [M, K]");

    int M = input.size(0);
    int K = input.size(1);

    // Pad M to BLOCK_M and K to WARP_K
    int M_pad = ((M + BLOCK_M - 1) / BLOCK_M) * BLOCK_M;
    int K_pad = ((K + WARP_K - 1) / WARP_K) * WARP_K;

    // Pad input if needed
    torch::Tensor input_padded = input;
    if (M_pad != M || K_pad != K) {
        input_padded = torch::zeros({M_pad, K_pad}, input.options());
        input_padded.narrow(0, 0, M).narrow(1, 0, K).copy_(input);
    }

    int num_act_elements = (M_pad / BLOCK_M) * (K_pad / WARP_K) * NUM_WARPS * WARP_M_TILES * WARP_SIZE;
    auto output = torch::empty({num_act_elements * 4}, torch::TensorOptions().dtype(torch::kInt32).device(input.device()));

    int num_scale_elements = (M_pad / BLOCK_M) * (K_pad / WARP_K) * NUM_WARPS * ASCALES_NUM_PACKS * ASCALES_VALID_LANES;
    auto scales = torch::empty({num_scale_elements * 2}, torch::TensorOptions().dtype(torch::kHalf).device(input.device()));

    dim3 grid(M_pad / WARP_M, K_pad / WARP_K);
    dim3 block(WARP_SIZE);

    quantize_act_optimized_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const half*>(input_padded.data_ptr<at::Half>()),
        reinterpret_cast<uint4*>(output.data_ptr<int32_t>()),
        reinterpret_cast<packed_ascale_t*>(scales.data_ptr<at::Half>()),
        M_pad, K_pad
    );

    return {output, scales, torch::tensor({M, K, M_pad, K_pad}, torch::kInt32)};
}

// --- Host wrapper: quantize weights ---
std::vector<torch::Tensor> quantize_wgt_optimized(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "input must be CUDA");
    TORCH_CHECK(input.dtype() == torch::kHalf, "input must be float16");
    TORCH_CHECK(input.dim() == 2, "input must be 2D [N, K]");

    int N = input.size(0);
    int K = input.size(1);

    int N_pad = ((N + BLOCK_N - 1) / BLOCK_N) * BLOCK_N;
    int K_pad = ((K + WARP_K - 1) / WARP_K) * WARP_K;

    torch::Tensor input_padded = input;
    if (N_pad != N || K_pad != K) {
        input_padded = torch::zeros({N_pad, K_pad}, input.options());
        input_padded.narrow(0, 0, N).narrow(1, 0, K).copy_(input);
    }

    int num_wgt_elements = (N_pad / BLOCK_N) * (K_pad / WARP_K) * WARP_N_TILES * WARP_SIZE;
    auto output = torch::empty({num_wgt_elements * 4}, torch::TensorOptions().dtype(torch::kInt32).device(input.device()));

    int num_scale_elements = (N_pad / BLOCK_N) * (K_pad / WARP_K) * WSCALES_NUM_PACKS * WSCALES_VALID_LANES;
    auto scales = torch::empty({num_scale_elements * 4}, torch::TensorOptions().dtype(torch::kHalf).device(input.device()));

    dim3 grid(N_pad / WARP_N, K_pad / WARP_K);
    dim3 block(WARP_SIZE);

    quantize_wgt_optimized_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const half*>(input_padded.data_ptr<at::Half>()),
        reinterpret_cast<uint4*>(output.data_ptr<int32_t>()),
        reinterpret_cast<packed_wscale_t*>(scales.data_ptr<at::Half>()),
        N_pad, K_pad
    );

    return {output, scales, torch::tensor({N, K, N_pad, K_pad}, torch::kInt32)};
}
