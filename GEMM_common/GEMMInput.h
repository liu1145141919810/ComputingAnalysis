#pragma once

namespace GEMM_common
{
	class CMatrix;
	class IAllocator;

	class CGEMMInput
	{
	public:
		CGEMMInput()=delete;
		CGEMMInput(IAllocator* vAllocator);
		~CGEMMInput();

		void construct(size_t vM, size_t vN, size_t vK, float vMin, float vMax);
		void construct(const CGEMMInput *vSource);

		CMatrix* getMatrixA() const { return m_pMatrixA; }
		CMatrix* getMatrixB() const { return m_pMatrixB; }
		CMatrix* getMatrixC() const { return m_pMatrixC; }

		float getAlpha() const { return m_Alpha; }
		float getBeta() const { return m_Beta; }

	private:
		CMatrix* m_pMatrixA = nullptr;
		CMatrix* m_pMatrixB = nullptr;
		CMatrix* m_pMatrixC = nullptr;
		IAllocator* m_pAllocator = nullptr;
		float m_Alpha = 0;
		float m_Beta = 0;

		float __generateRandomFloat(float vMin, float vMax);
	};
}

