#pragma once

namespace GEMM_common
{
	class CMatrix;

	class CGEMMOutput
	{
	public:
		CGEMMOutput() = default;
		~CGEMMOutput();		

		void init(const CMatrix* vMatrix);
		void recordMedianElapsedTime(float vElapsedTime) { m_MedianElapsedTime = vElapsedTime; }

		float getMedianElapsedTime() const { return m_MedianElapsedTime; }

		CMatrix* fetchMatrix() const { return m_pResult; }	
		const CMatrix* getMatrix() const { return m_pResult; }

	private:
		CMatrix* m_pResult = nullptr;
		float m_MedianElapsedTime = 0;
	};
}
