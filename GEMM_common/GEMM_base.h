#pragma once
#include <cstdint>
#include <string>
#include <any>
#include "CommonExport.h"

#define MATRIX_OFFSET(row, col, numCols) ((row) * (numCols) + (col))

namespace GEMM_common
{
	using GEMMCoreFunc = void(*)(const float* vMatrixA, const float* vMatrixB, float* vMatrixC, size_t vM, size_t vN, size_t vK, float vAlpha, float vBeta, const std::any& vCustomParam);
	
	enum class EMemoryType : uint8_t
	{
		CPU,
		GPU,
		Unkonwn
	};

	class CGEMMContext;
	class CMatrix;
	class IAllocator;
	class ITimer;

	void COMMON_DECLSPEC GEMM_naive(CGEMMContext& vGEMMContext, size_t vNumBenchmarkRuns);
	void COMMON_DECLSPEC GEMM_wrapper(GEMMCoreFunc vGEMMFunc, CGEMMContext& vGEMMContext, const std::string& vMethodName, const GEMM_common::CMatrix* vGroundTruth, float vBaselinePerf, size_t vNumBenchmarkRuns, const std::any& vCustomParam = std::any{});
	void COMMON_DECLSPEC GEMM(const CGEMMContext& vBaselineContext, GEMMCoreFunc vGEMMFunc, const std::string& vMethodName, IAllocator* vAllocator, ITimer* vTimer, size_t vNumBenchmarkRuns, const std::any& vCustomParam);

	void naive(const float* vA, const float* vB, float* vC, size_t vM, size_t vN, size_t vK, float vAlpha, float vBeta, const std::any& vCustomParam);
}
