#pragma once
#include <crtdbg.h>
#include "CommonExport.h"

namespace GEMM_common
{
	class COMMON_DECLSPEC ITimer
	{
	public:
		ITimer() = default;
		virtual ~ITimer() = default;

		virtual void start() = 0;
		virtual void stop() = 0;

		float getElapsedTime() const { return m_ElapsedTime; }

	protected:
		float m_ElapsedTime = 0;
	};

	class CScopedTimer
	{
	public:
		CScopedTimer() = delete;
		CScopedTimer(ITimer* vTimer);
		~CScopedTimer();

	private:
		ITimer* m_pTimer = nullptr;
	};
}