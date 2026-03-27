#pragma once
#include <cstdint>
#include <any>

constexpr size_t BLOCK_DIM = 16;  //一个线程块包含16x16个线程
constexpr size_t TILE_DIM_M = 128; //一个线程块在矩阵C上负责处理128x128个元素的生成
constexpr size_t TILE_DIM_N = 128;
constexpr size_t K_TILE_SIZE = 8;

constexpr size_t ELEMENT_PER_THREAD_M = TILE_DIM_M / BLOCK_DIM;
constexpr size_t ELEMENT_PER_THREAD_N = TILE_DIM_N / BLOCK_DIM;

#define CUDA_CALL(call) \
    do { \
        cudaError_t _err = (call); \
        if (_err != cudaSuccess) { \
            fprintf(stderr, \
                "[CUDA WARNING]\n \
                Call: %s\n \
                Function: %s\n \
                File: %s:%d\n \
                Error: %s (Code: %d)\n", \
                #call, __func__, __FILE__, __LINE__, cudaGetErrorString(_err), _err); \
        } \
    } while (0)

namespace GEMM_cuda
{
	void naive_gpu(const float* vA, const float* vB, float* vC, size_t vM, size_t vN, size_t vK, float vAlpha, float vBeta, const std::any& vCustomParam);
	void outer_loop_k_gpu(const float* vA, const float* vB, float* vC, size_t vM, size_t vN, size_t vK, float vAlpha, float vBeta, const std::any& vCustomParam);
    void tiling_gpu(const float* vMatrixA, const float* vMatrixB, float* vMatrixC, size_t vM, size_t vN, size_t vK, float vAlpha, float vBeta, const std::any& vCustomParam);
	void outer_product_naive_gpu(const float* vA, const float* vB, float* vC, size_t vM, size_t vN, size_t vK, float vAlpha, float vBeta, const std::any& vCustomParam);
	void outer_product_tiling_gpu(const float* vA, const float* vB, float* vC, size_t vM, size_t vN, size_t vK, float vAlpha, float vBeta, const std::any& vCustomParam);
}

