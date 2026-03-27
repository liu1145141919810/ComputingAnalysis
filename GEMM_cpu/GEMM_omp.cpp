#include <omp.h>
#include <crtdbg.h>
#include "GEMM_omp.h"
#include "../GEMM_common/GEMM_base.h"

using namespace GEMM_openmp;

void GEMM_openmp::naive(const float* vMatrixA, const float* vMatrixB, float* vMatrixC, size_t vM, size_t vN, size_t vK, 
	float vAlpha, float vBeta, const std::any& vCustomParam)
{
	_ASSERTE(vMatrixA && vMatrixB && vMatrixC);

//todo: add your code here
}

void GEMM_openmp::collapse_manual(const float* vMatrixA, const float* vMatrixB, float* vMatrixC, size_t vM, size_t vN, size_t vK, 
	float vAlpha, float vBeta, const std::any& vCustomParam)
{
	_ASSERTE(vMatrixA && vMatrixB && vMatrixC);

//todo: add your code here
}

void GEMM_openmp::collapse(const float* vMatrixA, const float* vMatrixB, float* vMatrixC, size_t vM, size_t vN, size_t vK, 
	float vAlpha, float vBeta, const std::any& vCustomParam)
{
	_ASSERTE(vMatrixA && vMatrixB && vMatrixC);

//todo: add your code here
}

void GEMM_openmp::collapse_tiled_outer_k(const float* vMatrixA, const float* vMatrixB, float* vMatrixC, size_t vM, size_t vN, size_t vK, 
	float vAlpha, float vBeta, const std::any& vCustomParam)
{
	_ASSERTE(vMatrixA && vMatrixB && vMatrixC);

	constexpr int BS = 16;

#pragma omp parallel for collapse(2) schedule(static)
	for (size_t bm = 0; bm < vM; bm += BS)
	{
		for (size_t bn = 0; bn < vN; bn += BS)
		{
			float localTileC[BS * BS];
			for (int i = 0; i < BS * BS; ++i) localTileC[i] = 0.0f;

			for (size_t k = 0; k < vK; ++k)
			{
				for (int i = 0; i < BS; ++i)
				{
					float a = vMatrixA[MATRIX_OFFSET(bm + i, k, vK)];
					for (int j = 0; j < BS; ++j)
					{
						float b = vMatrixB[MATRIX_OFFSET(k, bn + j, vN)];
						localTileC[i * BS + j] += a * b;
					}
				}
			}

			for (int i = 0; i < BS; ++i)
			{
				for (int j = 0; j < BS; ++j)
				{
					int row = static_cast<int>(bm + i);
					int col = static_cast<int>(bn + j);
					int idx = MATRIX_OFFSET(row, col, vN);
					vMatrixC[idx] = vBeta * vMatrixC[idx] + vAlpha * localTileC[i * BS + j];
				}
			}
		}
	}
}