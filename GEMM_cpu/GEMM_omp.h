#pragma once
#include <any>

namespace GEMM_openmp
{
	void naive(const float* vMatrixA, const float* vMatrixB, float* vMatrixC, size_t vM, size_t vN, size_t vK, float vAlpha, float vBeta, const std::any& vCustomParam);
	void collapse_manual(const float* vMatrixA, const float* vMatrixB, float* vMatrixC, size_t vM, size_t vN, size_t vK, float vAlpha, float vBeta, const std::any& vCustomParam);
	void collapse(const float* vMatrixA, const float* vMatrixB, float* vMatrixC, size_t vM, size_t vN, size_t vK, float vAlpha, float vBeta, const std::any& vCustomParam);
	void collapse_tiled_outer_k(const float* vMatrixA, const float* vMatrixB, float* vMatrixC, size_t vM, size_t vN, size_t vK, float vAlpha, float vBeta, const std::any& vCustomParam);
}