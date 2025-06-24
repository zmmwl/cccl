#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <memory>
#include <vector>
#include <type_traits>
#include <iostream>

namespace zmm2 {

// 前向声明
template<int N>
class ColumnComputeFramework;

/**
 * CUDA内核函数，用于并行计算每一行
 * 使用模板避免虚函数调用
 */
template<int N, typename ExpressionType>
__global__ void compute_kernel(
    const float* const* input_columns,  // N个输入列的指针数组
    float* output_column,               // 输出列
    size_t num_elements,               // 每列的元素数量
    ExpressionType expression          // 算术表达式对象（值传递）
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    // 网格化步进，处理所有元素
    for (size_t i = idx; i < num_elements; i += stride) {
        float values[N];
        
        // 从N个输入列中读取第i行的值
        #pragma unroll
        for (int col = 0; col < N; ++col) {
            values[col] = input_columns[col][i];
        }
        
        // 执行算术表达式计算（直接调用函数对象）
        output_column[i] = expression(values);
    }
}

/**
 * 主要的计算框架类
 */
template<int N>
class ColumnComputeFramework {
private:
    static_assert(N > 0, "列数必须大于0");
    
    size_t num_elements_;                    // 每列的元素数量
    std::vector<float*> d_input_columns_;   // GPU上的输入列数据
    float* d_output_column_;                // GPU上的输出列数据
    float** d_column_pointers_;             // GPU上的列指针数组
    
    // CUDA流
    cudaStream_t stream_;
    
    // 内存管理
    bool memory_allocated_;
    
public:
    /**
     * 构造函数
     * @param num_elements 每列的元素数量
     */
    explicit ColumnComputeFramework(size_t num_elements) 
        : num_elements_(num_elements), d_output_column_(nullptr), 
          d_column_pointers_(nullptr), memory_allocated_(false) {
        
        // 创建CUDA流
        cudaStreamCreate(&stream_);
        
        // 分配GPU内存
        allocateMemory();
    }
    
    /**
     * 析构函数
     */
    ~ColumnComputeFramework() {
        deallocateMemory();
        cudaStreamDestroy(stream_);
    }
    
    // 禁止拷贝构造和赋值
    ColumnComputeFramework(const ColumnComputeFramework&) = delete;
    ColumnComputeFramework& operator=(const ColumnComputeFramework&) = delete;
    
    /**
     * 设置输入数据
     * @param host_columns 主机端的N个列数据指针
     */
    void setInputData(const std::vector<const float*>& host_columns) {
        if (host_columns.size() != N) {
            throw std::invalid_argument("输入列数量不匹配");
        }
        
        // 异步拷贝数据到GPU
        for (int i = 0; i < N; ++i) {
            cudaMemcpyAsync(d_input_columns_[i], host_columns[i], 
                          num_elements_ * sizeof(float), 
                          cudaMemcpyHostToDevice, stream_);
        }
    }
    
    /**
     * 执行计算
     * @param expression 算术表达式函数对象
     */
    template<typename ExpressionType>
    void compute(const ExpressionType& expression) {
        // 计算网格和块的大小
        int block_size = 256;
        int num_blocks = std::min(65535, (int)((num_elements_ + block_size - 1) / block_size));
        
        // 启动内核（直接传递表达式对象）
        compute_kernel<N><<<num_blocks, block_size, 0, stream_>>>(
            d_column_pointers_, d_output_column_, num_elements_, expression
        );
        
        // 同步等待计算完成
        cudaStreamSynchronize(stream_);
        
        // 检查错误
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            throw std::runtime_error("CUDA内核执行失败: " + std::string(cudaGetErrorString(error)));
        }
    }
    
    /**
     * 获取计算结果
     * @param host_output 主机端输出缓冲区
     */
    void getResult(float* host_output) {
        cudaMemcpyAsync(host_output, d_output_column_, 
                       num_elements_ * sizeof(float), 
                       cudaMemcpyDeviceToHost, stream_);
        cudaStreamSynchronize(stream_);
    }
    
    /**
     * 获取CUDA流
     */
    cudaStream_t getStream() const {
        return stream_;
    }
    
    /**
     * 获取每列元素数量
     */
    size_t getNumElements() const {
        return num_elements_;
    }
    
private:
    /**
     * 分配GPU内存
     */
    void allocateMemory() {
        if (memory_allocated_) return;
        
        // 分配输入列内存
        d_input_columns_.resize(N);
        for (int i = 0; i < N; ++i) {
            cudaMalloc(&d_input_columns_[i], num_elements_ * sizeof(float));
        }
        
        // 分配输出列内存
        cudaMalloc(&d_output_column_, num_elements_ * sizeof(float));
        
        // 分配列指针数组内存并拷贝指针
        cudaMalloc(&d_column_pointers_, N * sizeof(float*));
        cudaMemcpy(d_column_pointers_, d_input_columns_.data(), 
                  N * sizeof(float*), cudaMemcpyHostToDevice);
        
        memory_allocated_ = true;
    }
    
    /**
     * 释放GPU内存
     */
    void deallocateMemory() {
        if (!memory_allocated_) return;
        
        for (auto ptr : d_input_columns_) {
            if (ptr) cudaFree(ptr);
        }
        d_input_columns_.clear();
        
        if (d_output_column_) {
            cudaFree(d_output_column_);
            d_output_column_ = nullptr;
        }
        
        if (d_column_pointers_) {
            cudaFree(d_column_pointers_);
            d_column_pointers_ = nullptr;
        }
        
        memory_allocated_ = false;
    }
};

} // namespace zmm2 