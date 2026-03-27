#pragma once
#include <any>

namespace GEMM_serial
{ 
	void change_loop_order(const float* vMatrixA, const float* vMatrixB, float* vMatrixC, size_t vM, size_t vN, size_t vK, float vAlpha, float vBeta, const std::any& vCustomParam);
}