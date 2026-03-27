#include "pch.h"
#include "CPUTimer.h"

using namespace GEMM_common;

void CCPUTimer::start()
{
	m_StartPoint = std::chrono::steady_clock::now();
}

void CCPUTimer::stop()
{
	std::chrono::steady_clock::time_point End = std::chrono::steady_clock::now();
	m_ElapsedTime = std::chrono::duration_cast<std::chrono::duration<float, std::milli>>(End - m_StartPoint).count();
}