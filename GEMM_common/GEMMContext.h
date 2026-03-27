#pragma once
#include "CommonExport.h"
#include "GEMMInput.h"
#include "GEMMOutput.h"
#include "Matrix.h"

namespace GEMM_common
{
	class ITimer;
	class IAllocator;

	class COMMON_DECLSPEC CGEMMContext
	{
	public:
		CGEMMContext() = default;
		~CGEMMContext() = default;

		void construct(size_t vM, size_t vN, size_t vK, IAllocator* vAllocator, ITimer* vTimer, float vMin = -5.0f, float vMax = 10.0f);
		void construct(const CGEMMContext* vSource, IAllocator* vAllocator, ITimer* vTimer);
		void initOutputMatrix();
		void recordMedianElapsedTime(float vElapsedTime) { m_pOutput->recordMedianElapsedTime(vElapsedTime); }

		ITimer* getTimer() const { return m_pTimer; }

		const float* getMatrixA() const { return m_pInput->getMatrixA()->getData(); }
		const float* getMatrixB() const { return m_pInput->getMatrixB()->getData(); }
		
		double computeError(const CMatrix* vGroundTruth) const;

		float* fetchMatrixC() { return m_pOutput->fetchMatrix()->fetchData(); }

		float getAlpha() const { return m_pInput->getAlpha(); }
		float getBeta() const { return m_pInput->getBeta(); }
		float getElapsedTime() const;
		float getMedianElapsedTime() const { return m_pOutput->getMedianElapsedTime(); }

		size_t getM() const { return m_pInput->getMatrixA()->getRows(); }
		size_t getN() const { return m_pInput->getMatrixA()->getCols(); }
		size_t getK() const { return m_pInput->getMatrixB()->getCols(); }

		const CMatrix* getOutputMatrix() const { return m_pOutput->getMatrix(); }

	private:
		CGEMMInput* m_pInput = nullptr;
		CGEMMOutput* m_pOutput = nullptr;
		ITimer* m_pTimer = nullptr;
	};
}