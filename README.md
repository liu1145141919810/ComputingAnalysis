# 基于OpenMP的GEMM实现

## 1目標

GEMM（General Matrix to Matrix Multiplication，通用矩阵乘法）在线性代数、机器学习、统计学和许多其他领域中扮演关键角色。本文通过实现朴素GEMM、调整循环次序、基于OpenMP的优化等多个方法，来加深对GEMM各种优化算法的理解。

具体，给定三个矩 A(形状MxK维度），B（形状KxN)和C(形状MxN).同时提供了两个幅度变化标量 α 和 β，通过如下的公式计算结果矩阵C以实现类似神经网络中一层处理的线性操作：C = α×A×B + β×C。A和B按照标准矩阵乘法操作，即A的第i行和B的第j列对应的元素相乘求和，得到C的（i,j)坐标处的元素。计算完毕后使用α放缩，再加上C放缩β倍的这个偏置量。

## 2算法实现

### 2.1 调整k-循环位置

调整K-循环后的GEMM
```
void GEMM_serial::change_loop_order(const float* vMatrixA, const float* vMatrixB, float* vMatrixC, size_t vM, size_t vN, size_t vK, float vAlpha, float vBeta, const std::any& vCustomParam)
{
//todo: add your code here
    for (size_t i = 0; i < vM; ++i) {
        for (size_t j = 0; j < vN; ++j) {
            vMatrixC[i * vN + j] *= vBeta;
        }
    }
    for (size_t k = 0; k < vK; ++k) {
        for (size_t i = 0; i < vM; ++i) {
            const float a_ik = vMatrixA[i * vK + k];
            for (size_t j = 0; j < vN; ++j) {
                vMatrixC[i * vN + j] += vAlpha * a_ik * vMatrixB[k * vN + j];
            }
        }
    }
}
```

问题：为什么原先的M-N-K计算顺序效率很低？

这种朴素的实现存在命中率低的问题。
矩阵A的访问为行优先连续访问，但B的访问为列优先访问。对B计算C[i][j]时访问J列上的所有k个元素在内存里是不连续的，会导致频繁的缓存缺失。
对C，每次迭代只对C[i][j]更新，无法利用cache的其他空间，从而将C迭代空间局部性浪费
事实上，主存访问延迟比缓存访问的延迟存在一个数量级以上的差距，频繁的缺失显著增加了主存访问的次数，导致了GEMM的性能很差。

问题：为什么把K放在最外层循环，就可以提高GEMM性能

这种方法每次不是直接计算出一个C[i][j],而是计算出其中的一部分（每个k），最终k个最外迭代的累计加和就是结果。此时A矩阵有良好时间局部性（同一个元素在相邻计算被多次访问）但空间局部由于变成列访问优先并不理想。而B访问时，内部的循环是在N维度上的，就可以按照行优先的方法访问B。同时这里每次最内部的迭代是对结果的一行计算出部分数值，对C的更新不局限在某个C[i][j]上，因此这个方案能在保证正确性的前提下既提高C的空间局部性使用（可以进行cache line连续写等优化），又使得B按照行优先被访问，同时A的访问又具有良好的时间局部性。


### 2.2 基于OpenMp的朴素GEMM

```
void GEMM_serial::change_loop_order(const float* vMatrixA, const float* vMatrixB, float* vMatrixC, size_t vM, size_t vN, size_t vK, float vAlpha, float vBeta, const std::any& vCustomParam)
{
//todo: add your code here
    for (size_t i = 0; i < vM; ++i) {
        for (size_t j = 0; j < vN; ++j) {
            vMatrixC[i * vN + j] *= vBeta;
        }
    }
    for (size_t k = 0; k < vK; ++k) {
        for (size_t i = 0; i < vM; ++i) {
            const float a_ik = vMatrixA[i * vK + k];
            for (size_t j = 0; j < vN; ++j) {
                vMatrixC[i * vN + j] += vAlpha * a_ik * vMatrixB[k * vN + j];
            }
        }
    }
}
```
在朴素实现的基础上，我们通过并行化多个矩阵行的计算，可以有效利用多核CPU的计算能力，实现显著的并行计算加速（通过#pragma omp parallel处理最外层的M迭代）。其并没有改变M-N-K的低效访问方式，只是使用OpenMP计算独立板块，发挥CPU的并行能力。具体的，我们将不同的矩阵行分配到多个线程执行，矩阵不同位置的计算不存在依赖，所以这种并行化是安全的，并能显著降低计算耗时

### 2.3 基于collapse子句的GEMM

```
void GEMM_openmp::collapse(const float* vMatrixA, const float* vMatrixB, float* vMatrixC, size_t vM, size_t vN, size_t vK, float vAlpha, float vBeta, const std::any& vCustomParam)
{
_ASSERTE(vMatrixA && vMatrixB && vMatrixC);
//todo: add your code here
int threadCount = 1;
    if(vCustomParam.has_value()) {
        threadCount = std::any_cast<int>(vCustomParam);
    }
    omp_set_num_threads(threadCount);
    #pragma omp parallel
    {
        #pragma omp for collapse(2)
        for (size_t i = 0; i < vM; ++i) {
            for (size_t j = 0; j < vN; ++j) {
                vMatrixC[i * vN + j] *= vBeta;
            }
        }
        #pragma omp for collapse(2)
        for (size_t i = 0; i < vM; ++i) {
            for (size_t j = 0; j < vN; ++j) {
                float sum = 0.0f;
                for (size_t k = 0; k < vK; ++k) {
                    sum += vMatrixA[i * vK + k] * vMatrixB[k * vN + j];
                }
                vMatrixC[i * vN + j] += vAlpha * sum;
            }
        }
    }
}
```
OpenMP 的 collapse 子句用于合并相邻层级的循环，将多层紧凑嵌套循环的迭代空间展平成一个更大的并行迭代空间。通过增加可并行的迭代数量，提高并行度，从而有利于 OpenMP 进行负载均衡分配，避免出现部分核心空闲而部分核心繁忙的情况。
使用说明：其基本格式为 #pragma omp for collapse(x)，要求后面的 x 层循环必须是紧凑嵌套的，且不同迭代之间不存在数据依赖关系。语义上，该指令会将原本的多维循环迭代空间映射为一维的并行迭代空间，例如：
```
#pragma omp for collapse(2)
for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
        // logic    }
}
```
转换成
```
int total_iters = M * N; 
#pragma omp parallel for
for (int idx = 0; idx < total_iters; ++idx) {
    int i = idx / N; 
int j = idx % N
//logic
;}
```
的等价形式

核心代码已经在表 3 中。这里我们对 GEMM 的三层循环（M–N–K）使用 #pragma omp for collapse(2) 对最外层的 M 和 N 两层循环进行合并（要求循环是紧凑且相邻的）。由于 M 和 N 对应的循环迭代之间不存在数据依赖，因此可以安全地进行并行化。
同时，为了减少内存访问次数并提高计算效率，在每个线程内部使用局部变量 sum 对最内层 k 循环的乘加结果进行累加，最后再一次性写回到 C[i][j] 中，从而避免对同一元素的频繁读写。

数据竞争：内层K循环的累加依赖于同一个C[i][j]的K次计算加入结果，因此不能三个维度一起colllapse，否则同一个C[i][j]会被多个线程设法同时写入，造成竞争降低效率。
避免线程间同步开销：
K循环作为核心的串行可避免线程间同步的花费，充分利用寄存器缓存A[i][k]和B[k][j]的局部性
过细导致问题：
若将三层循环全部 collapse，则每个 (i,j,k) 迭代成为并行单元，粒度过细，线程调度和同步开销可能超过并行加速收益

### 2.4 基于collapse手动实现的GEMM

```
void GEMM_openmp::collapse_manual(const float* vMatrixA, const float* vMatrixB, float* vMatrixC, size_t vM, size_t vN, size_t vK, float vAlpha, float vBeta, const std::any& vCustomParam)
{
_ASSERTE(vMatrixA && vMatrixB && vMatrixC);
//todo: add your code here
int threadCount = 1;
    if(vCustomParam.has_value()) {
        threadCount = std::any_cast<int>(vCustomParam);
    }
    omp_set_num_threads(threadCount);
#pragma omp parallel
    {
        const size_t total_iter = vM * vN;
        const size_t thread_num = omp_get_num_threads();
        const size_t thread_id = omp_get_thread_num();
        const size_t iter_per_thread = total_iter / thread_num;
        const size_t start_iter = thread_id * iter_per_thread;
        const size_t end_iter = (thread_id == thread_num - 1) ? total_iter : (thread_id + 1) * iter_per_thread;
        for (size_t iter = start_iter; iter < end_iter; ++iter) {
            const size_t i = iter / vN;
            const size_t j = iter % vN;
            vMatrixC[i * vN + j] *= vBeta;
        }
        for (size_t iter = start_iter; iter < end_iter; ++iter) {
            const size_t i = iter / vN;
            const size_t j = iter % vN;
            float sum = 0.0f;
            for (size_t k = 0; k < vK; ++k) {
                sum += vMatrixA[i * vK + k] * vMatrixB[k * vN + j];
            }
            vMatrixC[i * vN + j] += vAlpha * sum;
        }
    }
}
```

Step1:写上并行声明#paragma omp parallel, 初始化所有相关参数，包括M，N合并后的迭代数目total_iter,总线程数thread_num和thread_id

Step2:计算各个线程负责的迭代空间，包括每个线程处理的迭代数目 iter_per_thread, 处理的起始迭代编号start_iter和处理的最终迭代编号end_iter.

Step3.在C=beta x C 部分从每个迭代中还原出i,j,呈上变化系数beta

Step4.在等价于M x N 的一次外迭代里复原出处理的坐标i,j. 定义sum存储累加结果减少重复内存访问

Step5. 执行内部核心的k次迭代。将计算的结果加入sum中，而后对其施加变化系数alpha后加入到最终的C[i][j]中

### 2.5 基于分块+collapse+调整循环的GEMM

## 3实现细节

### 3.1 clang编译器

### 3.2 程序面向对象设计

### 3.3 程序正确性验证

### 3.4 工程化落地评估

## 4实验结果

### 4.1实验环境

### 4.2基准测试

### 4.3不同矩阵大小下的OpenMP方法性能对比

### 4.4不同线程数下的OpenMP方法性能对比

## 5 总结
