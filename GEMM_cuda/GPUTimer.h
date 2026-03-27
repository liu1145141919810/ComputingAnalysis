#pragma once
#include "../GEMM_common/Timer.h"
#include "cuda_runtime.h"

namespace GEMM_cuda
{
	class CGPUTimer : public GEMM_common::ITimer
	{
	public:
		CGPUTimer();
		~CGPUTimer();

		void start() override;
		void stop() override;

	private:
		cudaEvent_t m_Start;
		cudaEvent_t m_Stop;
	};
}
