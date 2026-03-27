#include "pch.h"
#include "Matrix.h"
#include "Allocator.h"

using namespace GEMM_common;

CMatrix::~CMatrix() 
{
	if (m_pData && m_pAllocator) m_pAllocator->free(m_pData);
}

float* CMatrix::__generateRandomMatrix(size_t vNumRows, size_t vNumCols, float vMinValue, float vMaxValue)
{
	_ASSERTE((vNumCols > 0) && (vNumRows > 0) && (vMaxValue > vMinValue));
	float* pData = new float[vNumCols * vNumRows];
	std::random_device rd;
	std::mt19937 eng(rd());
	std::uniform_real_distribution<float> distr(vMinValue, vMaxValue);

	for (int i = 0; i < vNumRows; i++)
	{
		for (int k = 0; k < vNumCols; k++)
		{
			pData[i * vNumCols + k] = distr(eng);
		}
	}
	return pData;
}

void CMatrix::constructRandomMatrix(size_t vNumRows, size_t vNumCols, IAllocator* vAllocator, float vMinValue /* = -10.0f */, float vMaxValue/* =10.0f */)
{
	_ASSERTE(vAllocator && (vNumRows > 0) && (vNumCols > 0));
	
	float* pData = __generateRandomMatrix(vNumRows, vNumCols, vMinValue, vMaxValue);

	m_NumRows = vNumRows;
	m_NumCols = vNumCols;
	m_pAllocator = vAllocator;
	m_pData = m_pAllocator->allocate(vNumCols * vNumRows);
	m_pAllocator->copy(m_pData, pData, vNumCols * vNumRows);

	delete[] pData;
}

void CMatrix::copyFrom(const CMatrix* vSrcMatrix, IAllocator *vAllocator)
{
	_ASSERTE(vSrcMatrix);
	m_NumRows = vSrcMatrix->m_NumRows;
	m_NumCols = vSrcMatrix->m_NumCols;
	m_pAllocator = vAllocator ? vAllocator : vSrcMatrix->m_pAllocator;
	m_pData = m_pAllocator->allocate(m_NumRows * m_NumCols);

	switch (getMemoryType())
	{
	case EMemoryType::GPU:
		m_pAllocator->copy(m_pData, vSrcMatrix->m_pData, m_NumRows * m_NumCols);
		break;
	case EMemoryType::CPU:
		vSrcMatrix->m_pAllocator->copy(m_pData, vSrcMatrix->m_pData, m_NumRows * m_NumCols);
		break;
	default:
		_ASSERTE(false);
		break;
	}
}

double CMatrix::computeRelativeError(const CMatrix* vTargetMatrix) const
{
	_ASSERTE(m_pAllocator && vTargetMatrix && isSameDimension(vTargetMatrix) && (vTargetMatrix->getMemoryType() == EMemoryType::CPU));
	
	float* pData = nullptr;
	if (getMemoryType() == EMemoryType::GPU)
	{
		pData = new float[m_NumRows * m_NumCols];
		m_pAllocator->copy(pData, m_pData, m_NumRows * m_NumCols);
	}
	else
		pData = m_pData;

	double t1 = 0, t2 = 0;
	for (auto i=0; i<m_NumRows*m_NumCols; i++)
	{
		float e = vTargetMatrix->m_pData[i];
		t1 += ((pData[i] - e) * (pData[i] - e));
		t2 += e * e;
	}
	return std::sqrt(t1 / t2);
}

bool CMatrix::isSameDimension(const CMatrix* vOther) const
{
	_ASSERTE(vOther);
	return (m_NumRows == vOther->m_NumRows) && (m_NumCols == vOther->m_NumCols);
}

EMemoryType CMatrix::getMemoryType() const
{
	return m_pAllocator ? m_pAllocator->getMemoryType() : EMemoryType::Unkonwn;
}
