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

### 2.3 基于collapse子句的GEMM

### 2.4 基于collapse手动实现的GEMM

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
