#include <crtdbg.h>
#include "cuda_runtime.h"
#include "GEMM_cuda.h"

__global__ void kernelSplitThread(const float* vMatrixA, const float* vMatrixB, float* vMatrixC,
	int M, int vN, int vK, float vAlpha, float vBeta)
{
	const size_t baseX = blockIdx.x * blockDim.x * ELEMENT_PER_THREAD_M;
	const size_t baseY = blockIdx.y * blockDim.y * ELEMENT_PER_THREAD_N;

	const int moveNum = SHARED_MEM_ELEMENT / (BLOCK_DIM * BLOCK_DIM) / 2;
	const size_t threadLinearID = threadIdx.y * blockDim.x + threadIdx.x;

	constexpr size_t threadsNum = BLOCK_DIM * BLOCK_DIM;

	float c[ELEMENT_PER_THREAD_M * ELEMENT_PER_THREAD_N] = {};
	float resC[ELEMENT_PER_THREAD_M * ELEMENT_PER_THREAD_N] = {};

	__shared__ float subA[TILE_DIM_M * TILE_DIM_K];
	__shared__ float subB[TILE_DIM_N * TILE_DIM_K];

	float4 regB[ELEMENT_PER_THREAD_M / 4]; // hopefully, these should reside in register.
	float4 regA[ELEMENT_PER_THREAD_M / 4];

	const float* baseA = vMatrixA + baseY * vK;
	const float* baseB = vMatrixB + baseX;

	int rowA = threadLinearID >> 1, rowB = threadLinearID >> 5, colA = (threadLinearID & 1) << 2, colB = (threadLinearID << 2) & 127;
	int warpId = threadLinearID >> 5, warpBaseId = threadLinearID & 31;
	int rowC = ((warpId >> 1 << 3) + (warpBaseId & 3)) << 2, colC = (((warpId & 1) << 4) + (warpBaseId >> 2)) << 2;
	float* baseC = vMatrixC + (baseY + rowC) * vN + baseX + colC;

	for (int i = 0; i < vK; i += TILE_DIM_K)
	{
		regB[0] = *reinterpret_cast<const float4*>(baseB + i * vN + rowB * vN + colB);
		regA[0] = *reinterpret_cast<const float4*>(baseA + i + rowA * vK + colA);
		*reinterpret_cast<float4*>(&subB[threadLinearID * 4]) = regB[0];
		subA[rowA + colA * TILE_DIM_M] = regA[0].x;
		subA[rowA + (colA + 1) * TILE_DIM_M] = regA[0].y;
		subA[rowA + (colA + 2) * TILE_DIM_M] = regA[0].z;
		subA[rowA + (colA + 3) * TILE_DIM_M] = regA[0].w;

		__syncthreads();

#pragma unroll
		for (int ii = 0; ii < TILE_DIM_K; ii++)
		{
			regB[0] = *reinterpret_cast<float4*>(&subB[colC + TILE_DIM_N * ii]);
			regB[1] = *reinterpret_cast<float4*>(&subB[colC + 32 + TILE_DIM_N * ii]);

			regA[0] = *reinterpret_cast<float4*>(&subA[rowC + ii * TILE_DIM_M]);
			regA[1] = *reinterpret_cast<float4*>(&subA[(rowC + 16) + ii * TILE_DIM_M]);

#pragma unroll
			for (int cpi = 0; cpi < ELEMENT_PER_THREAD_M / 4; cpi++)
			{
#pragma unroll
				for (int cpj = 0; cpj < ELEMENT_PER_THREAD_N / 4; cpj++)
				{
					c[cpi * 4 * ELEMENT_PER_THREAD_M + cpj * 4] += regA[cpi].x * regB[cpj].x;
					c[cpi * 4 * ELEMENT_PER_THREAD_M + cpj * 4 + 1] += regA[cpi].x * regB[cpj].y;
					c[cpi * 4 * ELEMENT_PER_THREAD_M + cpj * 4 + 2] += regA[cpi].x * regB[cpj].z;
					c[cpi * 4 * ELEMENT_PER_THREAD_M + cpj * 4 + 3] += regA[cpi].x * regB[cpj].w;

					c[(cpi * 4 + 1) * ELEMENT_PER_THREAD_M + cpj * 4] += regA[cpi].y * regB[cpj].x;
					c[(cpi * 4 + 1) * ELEMENT_PER_THREAD_M + cpj * 4 + 1] += regA[cpi].y * regB[cpj].y;
					c[(cpi * 4 + 1) * ELEMENT_PER_THREAD_M + cpj * 4 + 2] += regA[cpi].y * regB[cpj].z;
					c[(cpi * 4 + 1) * ELEMENT_PER_THREAD_M + cpj * 4 + 3] += regA[cpi].y * regB[cpj].w;

					c[(cpi * 4 + 2) * ELEMENT_PER_THREAD_M + cpj * 4] += regA[cpi].z * regB[cpj].x;
					c[(cpi * 4 + 2) * ELEMENT_PER_THREAD_M + cpj * 4 + 1] += regA[cpi].z * regB[cpj].y;
					c[(cpi * 4 + 2) * ELEMENT_PER_THREAD_M + cpj * 4 + 2] += regA[cpi].z * regB[cpj].z;
					c[(cpi * 4 + 2) * ELEMENT_PER_THREAD_M + cpj * 4 + 3] += regA[cpi].z * regB[cpj].w;

					c[(cpi * 4 + 3) * ELEMENT_PER_THREAD_M + cpj * 4] += regA[cpi].w * regB[cpj].x;
					c[(cpi * 4 + 3) * ELEMENT_PER_THREAD_M + cpj * 4 + 1] += regA[cpi].w * regB[cpj].y;
					c[(cpi * 4 + 3) * ELEMENT_PER_THREAD_M + cpj * 4 + 2] += regA[cpi].w * regB[cpj].z;
					c[(cpi * 4 + 3) * ELEMENT_PER_THREAD_M + cpj * 4 + 3] += regA[cpi].w * regB[cpj].w;
				}
			}
		}
		__syncthreads();
	}

#pragma unroll
	for (int i = 0; i < 4; i++)
	{
		*reinterpret_cast<float4*>(&regA[0]) = *reinterpret_cast<float4*>(&baseC[i * vN]);
		regA[0].x = regA[0].x * vBeta + vAlpha * c[i * ELEMENT_PER_THREAD_N];
		regA[0].y = regA[0].y * vBeta + vAlpha * c[1 + i * ELEMENT_PER_THREAD_N];
		regA[0].z = regA[0].z * vBeta + vAlpha * c[2 + i * ELEMENT_PER_THREAD_N];
		regA[0].w = regA[0].w * vBeta + vAlpha * c[3 + i * ELEMENT_PER_THREAD_N];
		*reinterpret_cast<float4*>(&baseC[i * vN]) = *reinterpret_cast<float4*>(&regA[0]);

		*reinterpret_cast<float4*>(&regA[0]) = *reinterpret_cast<float4*>(&baseC[i * vN + 32]);
		regA[0].x = regA[0].x * vBeta + vAlpha * c[4 + i * ELEMENT_PER_THREAD_N];
		regA[0].y = regA[0].y * vBeta + vAlpha * c[5 + i * ELEMENT_PER_THREAD_N];
		regA[0].z = regA[0].z * vBeta + vAlpha * c[6 + i * ELEMENT_PER_THREAD_N];
		regA[0].w = regA[0].w * vBeta + vAlpha * c[7 + i * ELEMENT_PER_THREAD_N];
		*reinterpret_cast<float4*>(&baseC[i * vN + 32]) = *reinterpret_cast<float4*>(&regA[0]);

		*reinterpret_cast<float4*>(&regA[0]) = *reinterpret_cast<float4*>(&baseC[(i + 16) * vN]);
		regA[0].x = regA[0].x * vBeta + vAlpha * c[32 + i * ELEMENT_PER_THREAD_N];
		regA[0].y = regA[0].y * vBeta + vAlpha * c[33 + i * ELEMENT_PER_THREAD_N];
		regA[0].z = regA[0].z * vBeta + vAlpha * c[34 + i * ELEMENT_PER_THREAD_N];
		regA[0].w = regA[0].w * vBeta + vAlpha * c[35 + i * ELEMENT_PER_THREAD_N];
		*reinterpret_cast<float4*>(&baseC[(i + 16) * vN]) = *reinterpret_cast<float4*>(&regA[0]);

		*reinterpret_cast<float4*>(&regA[0]) = *reinterpret_cast<float4*>(&baseC[(i + 16) * vN + 32]);
		regA[0].x = regA[0].x * vBeta + vAlpha * c[36 + i * ELEMENT_PER_THREAD_N];
		regA[0].y = regA[0].y * vBeta + vAlpha * c[37 + i * ELEMENT_PER_THREAD_N];
		regA[0].z = regA[0].z * vBeta + vAlpha * c[38 + i * ELEMENT_PER_THREAD_N];
		regA[0].w = regA[0].w * vBeta + vAlpha * c[39 + i * ELEMENT_PER_THREAD_N];
		*reinterpret_cast<float4*>(&baseC[(i + 16) * vN + 32]) = *reinterpret_cast<float4*>(&regA[0]);
	}
}

void split_thread_gpu(const float* vMatrixA, const float* vMatrixB, float* vMatrixC, uint32_t vM, uint32_t vN, uint32_t vK, float vAlpha, float vBeta)
{
	dim3 BlockDim(BLOCK_DIM, BLOCK_DIM);
	dim3 GridDim((vM + TILE_DIM_M - 1) / TILE_DIM_M, (vN + TILE_DIM_N - 1) / TILE_DIM_N);

	kernelSplitThread<<<GridDim, BlockDim >>> (vMatrixA, vMatrixB, vMatrixC, vM, vN, vK, vAlpha, vBeta);
	_ASSERTE(cudaGetLastError() == cudaSuccess);
}