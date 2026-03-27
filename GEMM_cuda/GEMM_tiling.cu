#include <crtdbg.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../GEMM_common/GEMM_base.h"
#include "GEMM_cuda.h"

using namespace GEMM_cuda;

__device__ __forceinline__ void loadKTileA(const float* vMatrixA, float* vKTileA,
    int vGlobalRowStart, int vGlobalColStart, int vNumCol, int vLocalRowStart, int vLocalColStart)
{
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int nthreads = blockDim.x * blockDim.y;
    int total = K_TILE_SIZE * TILE_DIM_M; 

    for (int idx = tid; idx < total; idx += nthreads)
    {
        int kk = idx / TILE_DIM_M;   
        int row = idx % TILE_DIM_M;   
        vKTileA[MATRIX_OFFSET(kk, row, TILE_DIM_M)] =
            vMatrixA[MATRIX_OFFSET(vGlobalRowStart + row, vGlobalColStart + kk, vNumCol)];
    }
}

__device__ __forceinline__ void loadKTileB(const float* vMatrixB, float* vKTileB,
    int vGlobalRowStart, int vGlobalColStart, int vNumCol, int vLocalRowStart, int vLocalColStart)
{
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int nthreads = blockDim.x * blockDim.y;
    int total = K_TILE_SIZE * TILE_DIM_N; 

    for (int idx = tid; idx < total; idx += nthreads)
    {
        int kk = idx / TILE_DIM_N;  
        int col = idx % TILE_DIM_N;   
        vKTileB[MATRIX_OFFSET(kk, col, TILE_DIM_N)] =
            vMatrixB[MATRIX_OFFSET(vGlobalRowStart + kk, vGlobalColStart + col, vNumCol)];
    }
}

__global__ void kernelTiling(const float* vMatrixA, const float* vMatrixB, float* vMatrixC,
    int vM, int vN, int vK, float vAlpha, float vBeta)
{
    int TileGlobalRowStart = blockIdx.y * TILE_DIM_M;
    int TileGlobalColStart = blockIdx.x * TILE_DIM_N;

    float ThreadAccBuf[ELEMENT_PER_THREAD_M * ELEMENT_PER_THREAD_N] = {};

    __shared__ float KTileA[TILE_DIM_M * K_TILE_SIZE];
    __shared__ float KTileB[K_TILE_SIZE * TILE_DIM_N];

    int TileLocalRowStart = threadIdx.y * ELEMENT_PER_THREAD_M;
    int TileLocalColStart = threadIdx.x * ELEMENT_PER_THREAD_N;

    for (int KTileBase = 0; KTileBase < vK; KTileBase += K_TILE_SIZE)
    {
        loadKTileA(vMatrixA, KTileA, TileGlobalRowStart, KTileBase, vK, TileLocalRowStart, TileLocalColStart);
        loadKTileB(vMatrixB, KTileB, KTileBase, TileGlobalColStart, vN, TileLocalRowStart, TileLocalColStart);

        __syncthreads();

        for (int kk = 0; kk < K_TILE_SIZE; ++kk)
        {
            for (int m = 0; m < ELEMENT_PER_THREAD_M; ++m)
            {
                float a = KTileA[MATRIX_OFFSET(kk, TileLocalRowStart + m, TILE_DIM_M)];
                for (int n = 0; n < ELEMENT_PER_THREAD_N; ++n)
                {
                    ThreadAccBuf[MATRIX_OFFSET(m, n, ELEMENT_PER_THREAD_N)] += a * KTileB[MATRIX_OFFSET(kk, TileLocalColStart + n, TILE_DIM_N)];
                }
            }
        }

        __syncthreads();
    }

    int MatrixRowStart4Me = (blockIdx.y * blockDim.y + threadIdx.y) * ELEMENT_PER_THREAD_M;
    int MatrixColStart4Me = (blockIdx.x * blockDim.x + threadIdx.x) * ELEMENT_PER_THREAD_N;

    for (int m = 0; m < ELEMENT_PER_THREAD_M; ++m)
    {
        for (int n = 0; n < ELEMENT_PER_THREAD_N; ++n)
        {
            int Offset = MATRIX_OFFSET(MatrixRowStart4Me + m, MatrixColStart4Me + n, vN);
            vMatrixC[Offset] = vBeta * vMatrixC[Offset] + vAlpha * ThreadAccBuf[MATRIX_OFFSET(m, n, ELEMENT_PER_THREAD_N)];
        }
    }
}

void GEMM_cuda::tiling_gpu(const float* vMatrixA, const float* vMatrixB, float* vMatrixC, size_t vM, size_t vN, size_t vK, float vAlpha, float vBeta, const std::any& vCustomParam)
{
    dim3 BlockDim(BLOCK_DIM, BLOCK_DIM);
    dim3 GridDim((vN + TILE_DIM_N - 1) / TILE_DIM_N, (vM + TILE_DIM_M - 1) / TILE_DIM_M);

    kernelTiling << <GridDim, BlockDim >> > (vMatrixA, vMatrixB, vMatrixC, static_cast<int>(vM), static_cast<int>(vN), static_cast<int>(vK), vAlpha, vBeta);
    _ASSERTE(cudaGetLastError() == cudaSuccess);
}