#pragma once
#include "CommonExport.h"
#include "GEMM_base.h"

namespace GEMM_common
{
	class COMMON_DECLSPEC IAllocator
	{
	public:
		IAllocator() = default;
		virtual ~IAllocator() = default;

		virtual float* allocate(size_t vNumElement) = 0;

		virtual void free(float* vPtr) = 0;
		virtual void copy(float* vDst, const float* vSrc, size_t vNumElement) const = 0;

		EMemoryType getMemoryType() const { return m_MemoryType; }

	protected:
		EMemoryType m_MemoryType = EMemoryType::Unkonwn;
	};

	class COMMON_DECLSPEC CCPUAllocator : public IAllocator
	{
	public:
		CCPUAllocator() { m_MemoryType = EMemoryType::CPU; }

		float* allocate(size_t vNumElement) override;

		void free(float* vPtr) override;
		void copy(float* vDst, const float* vSrc, size_t vNumElement) const override;
	};
}