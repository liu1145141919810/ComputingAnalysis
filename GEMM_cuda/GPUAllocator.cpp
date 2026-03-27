#include "GPUAllocator.h"
#include "cuda_runtime.h"
#include "GEMM_cuda.h"

using namespace GEMM_cuda;

float* CGPUAllocator::allocate(size_t vNumElment)
{
	float* p;
	CUDA_CALL(cudaMalloc((void**)&p, vNumElment * sizeof(float)));
	return p;
}

void CGPUAllocator::free(float* vPtr)
{
	CUDA_CALL(cudaFree(vPtr));
}

void CGPUAllocator::copy(float* vDst, const float* vSrc, size_t vNumElement) const
{
	CUDA_CALL(cudaMemcpy(vDst, vSrc, vNumElement * sizeof(float), cudaMemcpyDefault));
	_ASSERTE(cudaGetLastError() == cudaSuccess);
}