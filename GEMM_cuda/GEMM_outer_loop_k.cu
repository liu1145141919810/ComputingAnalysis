#include <crtdbg.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../GEMM_common/GEMM_base.h"
#include "GEMM_cuda.h"

using namespace GEMM_cuda;

__global__ void kernelOuterLoopK(const float* vMatrixA, const float* vMatrixB, float* vMatrixC,
    int vM, int vN, int vK, float vAlpha, float vBeta)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < vM && col < vN) {
        float sum = 0.0f;
        for (int k = 0; k < vK; ++k) {
            sum += vMatrixA[row * vK + k] * vMatrixB[k * vN + col];
        }
        vMatrixC[row * vN + col] = vAlpha * sum + vBeta * vMatrixC[row * vN + col];
    }
}

void GEMM_cuda::outer_loop_k_gpu(const float* vMatrixA, const float* vMatrixB, float* vMatrixC, size_t vM, 
    size_t vN, size_t vK, float vAlpha, float vBeta, const std::any& vCustomParam)
{
    dim3 BlockDim(BLOCK_DIM, BLOCK_DIM);
    dim3 GridDim((vN + TILE_DIM_N - 1) / TILE_DIM_N, (vM + TILE_DIM_M - 1) / TILE_DIM_M);

    kernelOuterLoopK<<<GridDim, BlockDim >>>(vMatrixA, vMatrixB, vMatrixC, vM, vN, vK, vAlpha, vBeta);
    _ASSERTE(cudaGetLastError() == cudaSuccess);
}