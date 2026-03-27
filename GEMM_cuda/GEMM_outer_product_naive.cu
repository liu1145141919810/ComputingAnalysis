#include <crtdbg.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "GEMM_cuda.h"
#include "../GEMM_common/GEMM_base.h"

using namespace GEMM_cuda;

__global__ void kernelOuterProductNaive(const float* vA, const float* vB, float* vC,
    int vM, int vN, int vK, float vAlpha, float vBeta)
{
    int MatrixColStart4Me = (blockIdx.x * blockDim.x + threadIdx.x) * ELEMENT_PER_THREAD_N;
    int MatrixRowStart4Me = (blockIdx.y * blockDim.y + threadIdx.y) * ELEMENT_PER_THREAD_M;

    float ThreadAccBuf[ELEMENT_PER_THREAD_M * ELEMENT_PER_THREAD_N] = {};

    for (int k = 0; k < vK; ++k)
    {
        float SegA[ELEMENT_PER_THREAD_M];
        for (int m = 0; m < ELEMENT_PER_THREAD_M; ++m)
        {
            SegA[m] = vA[MATRIX_OFFSET(MatrixRowStart4Me + m, k, vK)];
        }

        float SegB[ELEMENT_PER_THREAD_N];
        for (int n = 0; n < ELEMENT_PER_THREAD_N; ++n)
        {
            SegB[n] = vB[MATRIX_OFFSET(k, MatrixColStart4Me + n, vN)];
        }

        for (int m = 0; m < ELEMENT_PER_THREAD_M; ++m)
        {
            float am = SegA[m];
            for (int n = 0; n < ELEMENT_PER_THREAD_N; ++n)
            {
                ThreadAccBuf[m * ELEMENT_PER_THREAD_N + n] += am * SegB[n];
            }
        }
    }

    for (int m = 0; m < ELEMENT_PER_THREAD_M; ++m)
    {
        for (int n = 0; n < ELEMENT_PER_THREAD_N; ++n)
        {
            int row = MatrixRowStart4Me + m;
            int col = MatrixColStart4Me + n;
            int idx = MATRIX_OFFSET(row, col, vN);
            float oldc = vC[idx];
            vC[idx] = vBeta * oldc + vAlpha * ThreadAccBuf[m * ELEMENT_PER_THREAD_N + n];
        }
    }
}

void GEMM_cuda::outer_product_naive_gpu(const float* vA, const float* vB, float* vC,
    size_t vM, size_t vN, size_t vK, float vAlpha, float vBeta, const std::any& vCustomParam)
{
    dim3 BlockDim(BLOCK_DIM, BLOCK_DIM);
    dim3 GridDim((vN + TILE_DIM_N - 1) / TILE_DIM_N, (vM + TILE_DIM_M - 1) / TILE_DIM_M);

    kernelOuterProductNaive << <GridDim, BlockDim >> > (vA, vB, vC, static_cast<int>(vM), static_cast<int>(vN), static_cast<int>(vK), vAlpha, vBeta);
    _ASSERTE(cudaGetLastError() == cudaSuccess);
}