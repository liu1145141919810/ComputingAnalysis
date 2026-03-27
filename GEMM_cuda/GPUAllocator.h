#pragma once
#include "../GEMM_common/Allocator.h"

namespace GEMM_cuda
{
	class CGPUAllocator : public GEMM_common::IAllocator
	{
	public:
		CGPUAllocator() { m_MemoryType = GEMM_common::EMemoryType::GPU; }
		virtual ~CGPUAllocator() = default;

		float* allocate(size_t vNumElment) override;

		void free(float* vPtr) override;
		void copy(float* vDst, const float* vSrc, size_t vNumElement) const override;
	};
}