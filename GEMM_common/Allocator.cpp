#include "pch.h"
#include "Allocator.h"

using namespace GEMM_common;

float* CCPUAllocator::allocate(size_t vNumElement)
{
	return new float[vNumElement];
}

void CCPUAllocator::free(float* vPtr)
{
	delete[] vPtr;
}

void CCPUAllocator::copy(float* vDst, const float* vSrc, size_t vNumElement) const
{
	memcpy(vDst, vSrc, vNumElement * sizeof(float));
}