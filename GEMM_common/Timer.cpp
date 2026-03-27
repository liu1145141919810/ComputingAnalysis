#include "pch.h"
#include "Timer.h"

using namespace GEMM_common;

CScopedTimer::CScopedTimer(ITimer* vTimer)
{
	_ASSERTE(vTimer);
	m_pTimer = vTimer;
	m_pTimer->start();
}

CScopedTimer::~CScopedTimer()
{
	m_pTimer->stop();
}