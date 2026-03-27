#include "../GEMM_common/GEMM_base.h"
#include "../GEMM_common/GEMMContext.h"
#include "../GEMM_common/Allocator.h"
#include "../GEMM_common/CPUTimer.h"
#include "GEMM_serial.h"
#include "GEMM_omp.h"

GEMM_common::CCPUAllocator g_CPUAllocator;
GEMM_common::CCPUTimer g_CPUTimer;
GEMM_common::CGEMMContext g_Context_naive;
size_t g_NumBenchmarkRuns = 5;

void GEMM_cpu(GEMM_common::GEMMCoreFunc vGEMMFunc, const std::string& vMethodName, const std::any& vCustomParam)
{
	GEMM_common::GEMM(g_Context_naive, vGEMMFunc, vMethodName, &g_CPUAllocator, &g_CPUTimer, g_NumBenchmarkRuns, vCustomParam);
}

int main()
{
	g_Context_naive.construct(1024, 1024, 1024, &g_CPUAllocator, &g_CPUTimer);
	GEMM_common::GEMM_naive(g_Context_naive, g_NumBenchmarkRuns);

	GEMM_cpu(GEMM_serial::change_loop_order, "Serial_Change_Loop_Order", std::nullopt);
	GEMM_cpu(GEMM_openmp::naive, "OMP_Naive", std::nullopt);
	GEMM_cpu(GEMM_openmp::collapse_manual, "OMP_Collapse_Manual", std::nullopt);
	GEMM_cpu(GEMM_openmp::collapse, "OMP_Collapse", std::nullopt);
	GEMM_cpu(GEMM_openmp::collapse_tiled_outer_k, "OMP_Tiling_Collapse_Outer_K", 4);

	return 0;
}