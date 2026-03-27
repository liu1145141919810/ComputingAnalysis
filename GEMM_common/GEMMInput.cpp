#include "pch.h"
#include "GEMMInput.h"
#include "Matrix.h"

using namespace GEMM_common;

CGEMMInput::CGEMMInput(IAllocator* vAllocator)
{
	_ASSERTE(vAllocator);
	m_pAllocator = vAllocator;
}

CGEMMInput::~CGEMMInput()
{
	if (m_pMatrixA) delete m_pMatrixA;
	if (m_pMatrixB) delete m_pMatrixB;
	if (m_pMatrixC) delete m_pMatrixC;
}

void CGEMMInput::construct(size_t vM, size_t vN, size_t vK, float vMin, float vMax)
{
	_ASSERTE(!m_pMatrixA && !m_pMatrixB && !m_pMatrixC && m_pAllocator);

	m_pMatrixA = new CMatrix();	
	m_pMatrixA->constructRandomMatrix(vM, vK, m_pAllocator, vMin, vMax);
	m_pMatrixB = new CMatrix();
	m_pMatrixB->constructRandomMatrix(vK, vN, m_pAllocator, vMin, vMax);
	m_pMatrixC = new CMatrix();
	m_pMatrixC->constructRandomMatrix(vM, vN, m_pAllocator, vMin, vMax);
	m_Alpha = __generateRandomFloat(vMin, vMax);
	m_Beta = __generateRandomFloat(vMin, vMax);
}

void CGEMMInput::construct(const CGEMMInput* vSource)
{
	_ASSERTE(vSource && !m_pMatrixA && !m_pMatrixB && !m_pMatrixC && m_pAllocator);
	m_pMatrixA = new CMatrix();
	m_pMatrixA->copyFrom(vSource->m_pMatrixA, m_pAllocator);
	m_pMatrixB = new CMatrix();
	m_pMatrixB->copyFrom(vSource->m_pMatrixB, m_pAllocator);
	m_pMatrixC = new CMatrix();
	m_pMatrixC->copyFrom(vSource->m_pMatrixC, m_pAllocator);
	m_Alpha = vSource->m_Alpha;
	m_Beta = vSource->m_Beta;
}

float CGEMMInput::__generateRandomFloat(float vMin, float vMax)
{
	std::random_device rd;
	std::mt19937 eng(rd());
	std::uniform_real_distribution<float> distr(vMin, vMax);
	return distr(eng);
}