#include "pch.h"
#include "GEMMContext.h"
#include "CPUTimer.h"
#include "GEMMInput.h"
#include "GEMMOutput.h"
#include "Matrix.h"

using namespace GEMM_common;

void CGEMMContext::construct(size_t vM, size_t vN, size_t vK, IAllocator* vAllocator, ITimer* vTimer, float vMin, float vMax)
{
	_ASSERTE(!m_pInput && !m_pOutput && vAllocator && vTimer);
	m_pInput = new CGEMMInput(vAllocator);
	m_pInput->construct(vM, vN, vK, vMin, vMax);
	m_pOutput = new CGEMMOutput();
	m_pTimer = vTimer;
}

void CGEMMContext::construct(const CGEMMContext* vSource, IAllocator* vAllocator, ITimer* vTimer)
{
	_ASSERTE(!m_pInput && !m_pOutput && vSource && vAllocator && vTimer);
	m_pInput = new CGEMMInput(vAllocator);
	m_pInput->construct(vSource->m_pInput);
	m_pOutput = new CGEMMOutput();
	m_pTimer = vTimer;
}

void CGEMMContext::initOutputMatrix()
{
	_ASSERTE(m_pOutput && m_pInput);
	m_pOutput->init(m_pInput->getMatrixC());
}

double CGEMMContext::computeError(const CMatrix* vGroundTruth) const
{
	_ASSERTE(vGroundTruth && m_pOutput && m_pOutput->getMatrix());
	return m_pOutput->getMatrix()->computeRelativeError(vGroundTruth);
}

float CGEMMContext::getElapsedTime() const
{
	_ASSERTE(m_pTimer);
	return m_pTimer->getElapsedTime();
}