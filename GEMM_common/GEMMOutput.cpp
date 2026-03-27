#include "pch.h"
#include "GEMMOutput.h"
#include "Matrix.h"

using namespace GEMM_common;

CGEMMOutput::~CGEMMOutput()
{
	if (m_pResult) delete m_pResult;
}

void CGEMMOutput::init(const CMatrix *vMatrix)
{
	_ASSERTE(vMatrix);
	if (m_pResult)
		_ASSERTE((m_pResult->getRows() == vMatrix->getRows()) && (m_pResult->getCols() == vMatrix->getCols()));
	else 
		m_pResult = new CMatrix();
	m_pResult->copyFrom(vMatrix, nullptr);
}