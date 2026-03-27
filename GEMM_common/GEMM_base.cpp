#include "pch.h"
#include "GEMM_base.h"
#include "GEMMContext.h"
#include "Timer.h"

using namespace GEMM_common;

const double g_RelateiveErrorThreshold = 1e-6;

void GEMM_common::naive(const float* vMatrixA, const float* vMatrixB, float* vMatrixC, size_t vM, size_t vN, size_t vK, float vAlpha, float vBeta, const std::any& vCustomParam)
{
	_ASSERTE(vMatrixA && vMatrixB && vMatrixC);

	for (auto m = 0; m < vM; m++)
	{
		for (auto n = 0; n < vN; n++)
		{
			float Sum = 0;
			for (auto k = 0; k < vK; k++)
			{
				Sum += vMatrixA[m * vK + k] * vMatrixB[k * vN + n];
			}
			vMatrixC[m * vN + n] = vAlpha * Sum + vBeta * vMatrixC[m * vN + n];
		}
	}
}

void GEMM_common::GEMM_wrapper(GEMMCoreFunc vGEMMFunc, CGEMMContext& vGEMMContext, const std::string& vMethodName, const GEMM_common::CMatrix* vGroundTruth, float vBaselinePerf, size_t vNumBenchmarkRuns, const std::any& vCustomParam)
{
	_ASSERTE(vNumBenchmarkRuns > 0);
	std::vector<float> ElapsedTimeSet;
	double AvgError = 0.0;

	for (size_t i = 0; i < vNumBenchmarkRuns; i++)
	{
		vGEMMContext.initOutputMatrix();

		{
			CScopedTimer Timer(vGEMMContext.getTimer());
			vGEMMFunc(vGEMMContext.getMatrixA(), vGEMMContext.getMatrixB(), vGEMMContext.fetchMatrixC(), vGEMMContext.getM(), vGEMMContext.getN(), vGEMMContext.getK(), vGEMMContext.getAlpha(), vGEMMContext.getBeta(), vCustomParam);
		}
		ElapsedTimeSet.push_back(vGEMMContext.getElapsedTime());

		if (vGroundTruth) AvgError += vGEMMContext.computeError(vGroundTruth);
	}
	AvgError /= vNumBenchmarkRuns;

	std::sort(ElapsedTimeSet.begin(), ElapsedTimeSet.end());
	float Perf = (vNumBenchmarkRuns % 2 == 0) ? (ElapsedTimeSet[vNumBenchmarkRuns / 2 - 1] + ElapsedTimeSet[vNumBenchmarkRuns / 2]) / 2 : ElapsedTimeSet[vNumBenchmarkRuns / 2];
	vGEMMContext.recordMedianElapsedTime(Perf);
	std::cout << std::format("\nGEMM [{}] is done in {}ms.\n", vMethodName, Perf);
	if (vBaselinePerf > 0) std::cout << std::format("  Speedup ratio over [naive]: {:.2f}x.\n", vBaselinePerf / Perf);

	if (vGroundTruth)
	{
		std::cout << std::format("  The error coefficient of [{}] is [{:.4f}].\n", vMethodName, AvgError / g_RelateiveErrorThreshold);
	}
}

void GEMM_common::GEMM_naive(CGEMMContext& vGEMMContext, size_t vNumBenchmarkRuns)
{
	GEMM_common::GEMM_wrapper(GEMM_common::naive, vGEMMContext, "Serial_Naive", nullptr, 0, vNumBenchmarkRuns, std::nullopt);
}

void GEMM_common::GEMM(const CGEMMContext& vBaselineContext, GEMMCoreFunc vGEMMFunc, const std::string& vMethodName, IAllocator *vAllocator, ITimer* vTimer, size_t vNumBenchmarkRuns, const std::any& vCustomParam)
{
	_ASSERTE(vAllocator && vTimer);
	GEMM_common::CGEMMContext Context;
	Context.construct(&vBaselineContext, vAllocator, vTimer);
	const GEMM_common::CMatrix* pGroundTruth = vBaselineContext.getOutputMatrix();
	float BaselinePerf = vBaselineContext.getMedianElapsedTime();
	_ASSERTE(pGroundTruth);
	GEMM_common::GEMM_wrapper(vGEMMFunc, Context, vMethodName, pGroundTruth, BaselinePerf, vNumBenchmarkRuns, vCustomParam);
}
