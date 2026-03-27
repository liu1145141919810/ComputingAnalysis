#include <iostream>
#include <format>
#include <crtdbg.h>
#include <optional>
#include "cuda_runtime.h"
#include "../GEMM_common/Allocator.h"
#include "../GEMM_common/CPUTimer.h"
#include "../GEMM_common/GEMMContext.h"
#include "GEMM_cuda.h"
#include "GPUTimer.h"
#include "GPUAllocator.h"

GEMM_common::CGEMMContext g_Context_naive;
GEMM_common::CCPUAllocator g_CPUAllocator;
GEMM_common::CCPUTimer g_CPUTimer;
GEMM_cuda::CGPUAllocator g_GPUAllocator;
GEMM_cuda::CGPUTimer g_GPUTimer;
size_t g_NumBaeslineRuns = 3;
size_t g_NumBenchmarkRuns = 10;

void GEMM_gpu(GEMM_common::GEMMCoreFunc vGEMMFunc, const std::string& vMethodName, const std::any& vCustomParam)
{
	GEMM_common::GEMM(g_Context_naive, vGEMMFunc, vMethodName, &g_GPUAllocator, &g_GPUTimer, g_NumBenchmarkRuns, vCustomParam);
}

int main()
{
	CUDA_CALL(cudaSetDevice(0));

	g_Context_naive.construct(1024, 1024, 1024, &g_CPUAllocator, &g_CPUTimer);
	GEMM_common::GEMM_naive(g_Context_naive, g_NumBaeslineRuns);

	GEMM_gpu(GEMM_cuda::naive_gpu, "GPU_Naive", std::nullopt);
	GEMM_gpu(GEMM_cuda::outer_loop_k_gpu, "GPU_Outer_K", std::nullopt);
	GEMM_gpu(GEMM_cuda::tiling_gpu, "GPU_Tiling", std::nullopt);
	GEMM_gpu(GEMM_cuda::outer_product_naive_gpu, "GPU_Outer_Product_Naive", std::nullopt);
	GEMM_gpu(GEMM_cuda::outer_product_tiling_gpu, "GPU_Outer_Product_Tiling", std::nullopt);
	return 0;
}