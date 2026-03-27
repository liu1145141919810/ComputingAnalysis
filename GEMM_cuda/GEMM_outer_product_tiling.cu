#include <crtdbg.h>
#include "cuda_runtime.h"
#include "GEMM_cuda.h"
#include "../GEMM_common/GEMM_base.h"

using namespace GEMM_cuda;

constexpr size_t VECTOR_WIDTH = 4;  // 使用 float4 向量化加载和存储，这个值不能修改

static_assert(TILE_DIM_N % 4 == 0, "TILE_DIM_N must be multiple of 4");
static_assert((BLOCK_DIM * BLOCK_DIM * 4) == (K_TILE_SIZE * TILE_DIM_N), "threads_per_block * 4 must equal K_TILE_SIZE * TILE_DIM_N");
static_assert(BLOCK_DIM * ELEMENT_PER_THREAD_M == TILE_DIM_M, "BLOCK_DIM * ELEMENT_PER_THREAD_M == TILE_DIM_M");
static_assert(BLOCK_DIM * ELEMENT_PER_THREAD_N == TILE_DIM_N, "BLOCK_DIM * ELEMENT_PER_THREAD_N == TILE_DIM_N");

__device__ __forceinline__ void loadKTileA(const float* vMatrixA, float* vKTileOfA,
    int vMatrixRowStart, int vMatrixColStart, int vK, size_t vThreadLinearID)
{
    int RowA = vThreadLinearID / 2;
    int ColA = (vThreadLinearID & 1) * VECTOR_WIDTH;

    float4 RegA = *reinterpret_cast<const float4*>(&vMatrixA[MATRIX_OFFSET(vMatrixRowStart + RowA, vMatrixColStart + ColA, vK)]);
// 将 A 的四个分量分别写入共享内存（按列主序排列以便后续按向量加载）
    vKTileOfA[RowA + ColA * TILE_DIM_M] = RegA.x;
    vKTileOfA[RowA + (ColA + 1) * TILE_DIM_M] = RegA.y;
    vKTileOfA[RowA + (ColA + 2) * TILE_DIM_M] = RegA.z;
    vKTileOfA[RowA + (ColA + 3) * TILE_DIM_M] = RegA.w;
}

__device__ __forceinline__ void loadKTileB(const float* vMatrixB, float* vKTileOfB,
    int vMatrixRowStart, int vMatrixColStart, int vN, size_t vThreadLinearID)
{
    int RowB = vThreadLinearID / (TILE_DIM_N / VECTOR_WIDTH);
    int ColB = (vThreadLinearID * VECTOR_WIDTH) % TILE_DIM_N;

    float4 RegB = *reinterpret_cast<const float4*>(&vMatrixB[MATRIX_OFFSET(vMatrixRowStart + RowB, vMatrixColStart + ColB, vN)]);
    *reinterpret_cast<float4*>(&vKTileOfB[vThreadLinearID * 4]) = RegB;
}

__global__ void kernelOuterProductTiling(const float* vMatrixA, const float* vMatrixB, float* vMatrixC,
    int vM, int vN, int vK, float vAlpha, float vBeta)
{
    const int MatrixColStart4ThreadBlock = blockIdx.x * TILE_DIM_N;
    const int MatrixRowStart4ThreadBlock = blockIdx.y * TILE_DIM_M;

    const size_t threadLinearID = threadIdx.y * blockDim.x + threadIdx.x;

    float c[ELEMENT_PER_THREAD_M * ELEMENT_PER_THREAD_N] = {};
    float resC[ELEMENT_PER_THREAD_M * ELEMENT_PER_THREAD_N] = {};

    __shared__ float KTileA[TILE_DIM_M * K_TILE_SIZE];
    __shared__ float KTileB[K_TILE_SIZE * TILE_DIM_N];

    float4 RegA[ELEMENT_PER_THREAD_M / VECTOR_WIDTH];
    float4 RegB[ELEMENT_PER_THREAD_N / VECTOR_WIDTH];

    for (int k = 0; k < vK; k += K_TILE_SIZE)
    {// handle each K-tile
		loadKTileA(vMatrixA, KTileA, MatrixRowStart4ThreadBlock, k, vK, threadLinearID);
		loadKTileB(vMatrixB, KTileB, k, MatrixColStart4ThreadBlock, vN, threadLinearID);
        __syncthreads();

#pragma unroll
        for (int ii = 0; ii < K_TILE_SIZE; ii++)
        {// conduct GEMM for K-tile from A and B based on outer products
            for (int ra = 0; ra < ELEMENT_PER_THREAD_M / VECTOR_WIDTH; ++ra)
                RegA[ra] = *reinterpret_cast<float4*>(&KTileA[ii * TILE_DIM_M + threadIdx.x * ELEMENT_PER_THREAD_M + ra * VECTOR_WIDTH]);
            for (int rb = 0; rb < ELEMENT_PER_THREAD_N / VECTOR_WIDTH; ++rb)
                RegB[rb] = *reinterpret_cast<float4*>(&KTileB[ii * TILE_DIM_N + threadIdx.y * ELEMENT_PER_THREAD_N + rb * VECTOR_WIDTH]);

#pragma unroll
            for (int cpi = 0; cpi < ELEMENT_PER_THREAD_M / VECTOR_WIDTH; cpi++)
            {
#pragma unroll
                for (int cpj = 0; cpj < ELEMENT_PER_THREAD_N / VECTOR_WIDTH; cpj++)
                {
                    c[cpi * 4 * ELEMENT_PER_THREAD_M + cpj * 4] += RegA[cpi].x * RegB[cpj].x;
                    c[cpi * 4 * ELEMENT_PER_THREAD_M + cpj * 4 + 1] += RegA[cpi].x * RegB[cpj].y;
                    c[cpi * 4 * ELEMENT_PER_THREAD_M + cpj * 4 + 2] += RegA[cpi].x * RegB[cpj].z;
                    c[cpi * 4 * ELEMENT_PER_THREAD_M + cpj * 4 + 3] += RegA[cpi].x * RegB[cpj].w;

                    c[(cpi * 4 + 1) * ELEMENT_PER_THREAD_M + cpj * 4] += RegA[cpi].y * RegB[cpj].x;
                    c[(cpi * 4 + 1) * ELEMENT_PER_THREAD_M + cpj * 4 + 1] += RegA[cpi].y * RegB[cpj].y;
                    c[(cpi * 4 + 1) * ELEMENT_PER_THREAD_M + cpj * 4 + 2] += RegA[cpi].y * RegB[cpj].z;
                    c[(cpi * 4 + 1) * ELEMENT_PER_THREAD_M + cpj * 4 + 3] += RegA[cpi].y * RegB[cpj].w;

                    c[(cpi * 4 + 2) * ELEMENT_PER_THREAD_M + cpj * 4] += RegA[cpi].z * RegB[cpj].x;
                    c[(cpi * 4 + 2) * ELEMENT_PER_THREAD_M + cpj * 4 + 1] += RegA[cpi].z * RegB[cpj].y;
                    c[(cpi * 4 + 2) * ELEMENT_PER_THREAD_M + cpj * 4 + 2] += RegA[cpi].z * RegB[cpj].z;
                    c[(cpi * 4 + 2) * ELEMENT_PER_THREAD_M + cpj * 4 + 3] += RegA[cpi].z * RegB[cpj].w;

                    c[(cpi * 4 + 3) * ELEMENT_PER_THREAD_M + cpj * 4] += RegA[cpi].w * RegB[cpj].x;
                    c[(cpi * 4 + 3) * ELEMENT_PER_THREAD_M + cpj * 4 + 1] += RegA[cpi].w * RegB[cpj].y;
                    c[(cpi * 4 + 3) * ELEMENT_PER_THREAD_M + cpj * 4 + 2] += RegA[cpi].w * RegB[cpj].z;
                    c[(cpi * 4 + 3) * ELEMENT_PER_THREAD_M + cpj * 4 + 3] += RegA[cpi].w * RegB[cpj].w;
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < ELEMENT_PER_THREAD_M; i++)
#pragma unroll
        for (int j = 0; j < ELEMENT_PER_THREAD_N; j += 4)
            *reinterpret_cast<float4*>(&resC[i * ELEMENT_PER_THREAD_M + j]) =
            *reinterpret_cast<const float4*>(&vMatrixC[MATRIX_OFFSET(MatrixRowStart4ThreadBlock + threadIdx.x * ELEMENT_PER_THREAD_M + i, MatrixColStart4ThreadBlock + threadIdx.y * ELEMENT_PER_THREAD_N + j, vN)]);

#pragma unroll
    for (int i = 0; i < ELEMENT_PER_THREAD_M; i++)
#pragma unroll
        for (int j = 0; j < ELEMENT_PER_THREAD_N; j++)
            resC[i * ELEMENT_PER_THREAD_M + j] = resC[i * ELEMENT_PER_THREAD_M + j] * vBeta + vAlpha * c[i * ELEMENT_PER_THREAD_M + j];

#pragma unroll
    for (int i = 0; i < ELEMENT_PER_THREAD_M; i++)
#pragma unroll
        for (int j = 0; j < ELEMENT_PER_THREAD_N; j += 4)
            *reinterpret_cast<float4*>(&vMatrixC[MATRIX_OFFSET(MatrixRowStart4ThreadBlock + threadIdx.x * ELEMENT_PER_THREAD_M + i, MatrixColStart4ThreadBlock + threadIdx.y * ELEMENT_PER_THREAD_N + j, vN)]) =
            *reinterpret_cast<float4*>(&resC[i * ELEMENT_PER_THREAD_M + j]);
}

void GEMM_cuda::outer_product_tiling_gpu(const float* vMatrixA, const float* vMatrixB, float* vMatrixC, size_t vM, size_t vN, size_t vK, float vAlpha, float vBeta, const std::any& vCustomParam)
{
    dim3 BlockDim(BLOCK_DIM, BLOCK_DIM);
    dim3 GridDim((vN + TILE_DIM_N - 1) / TILE_DIM_N, (vM + TILE_DIM_M - 1) / TILE_DIM_M);

    kernelOuterProductTiling << <GridDim, BlockDim >> > (vMatrixA, vMatrixB, vMatrixC, vM, vN, vK, vAlpha, vBeta);
    _ASSERTE(cudaGetLastError() == cudaSuccess);
}