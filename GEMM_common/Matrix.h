#pragma once
#include <cstddef> 
#include <utility>
#include "GEMM_base.h"

namespace GEMM_common
{
	class IAllocator;

	class CMatrix 
	{
	public:
		CMatrix() = default;
		~CMatrix();

		void constructRandomMatrix(size_t vNumRows, size_t vNumCols, IAllocator* vAllocator, float vMinValue = -10.0f, float vMaxValue=10.0f);
		void copyFrom(const CMatrix* vSrcMatrix, IAllocator* vAllocator);

		CMatrix(const CMatrix&) = delete;
		CMatrix& operator=(const CMatrix&) = delete;
		CMatrix(CMatrix&&) = delete;
		CMatrix& operator=(CMatrix&&) = delete;

		double computeRelativeError(const CMatrix* vTargetMatrix) const;

		bool isSameDimension(const CMatrix* vOther) const;

		EMemoryType getMemoryType() const;

		size_t getRows() const { return m_NumRows; }
		size_t getCols() const { return m_NumCols; }

		float* fetchData() { return m_pData; }
		const float* getData() const { return m_pData; }

	private:
		float *m_pData=nullptr;
		size_t m_NumRows=0;
		size_t m_NumCols=0;
		IAllocator *m_pAllocator = nullptr; 

		float* __generateRandomMatrix(size_t vNumRows, size_t vNumCols, float vMinValue, float vMaxValue);
	};
}
