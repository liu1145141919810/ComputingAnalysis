#include "GPUTimer.h"
#include <crtdbg.h>

using namespace GEMM_cuda;

CGPUTimer::CGPUTimer()
{
	cudaEventCreate(&m_Start);
	cudaEventCreate(&m_Stop);
	_ASSERTE(cudaGetLastError() == cudaSuccess);

	cudaEventRecord(m_Start);
	_ASSERTE(cudaGetLastError() == cudaSuccess);
}

CGPUTimer::~CGPUTimer()
{
	cudaEventDestroy(m_Start);
	cudaEventDestroy(m_Stop);
	_ASSERTE(cudaGetLastError() == cudaSuccess);
}

void CGPUTimer::start()
{
	cudaEventRecord(m_Start);
	_ASSERTE(cudaGetLastError() == cudaSuccess);
}

void CGPUTimer::stop()
{
	cudaEventRecord(m_Stop);
	cudaEventSynchronize(m_Stop);
	cudaEventElapsedTime(&m_ElapsedTime, m_Start, m_Stop);
	_ASSERTE(cudaGetLastError() == cudaSuccess);
}