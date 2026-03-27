#pragma once
#include "Timer.h"
#include <chrono>
#include "CommonExport.h"

namespace GEMM_common
{
	class COMMON_DECLSPEC CCPUTimer : public ITimer
	{
	public:
		CCPUTimer() = default;
		~CCPUTimer() = default;

		void start() override;
		void stop() override;

	private:
		std::chrono::steady_clock::time_point m_StartPoint;
	};
}
