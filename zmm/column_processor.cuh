#pragma once

#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <vector>
#include <memory>
#include <functional>
#include <iostream>

namespace zmm {

// CUDA错误检查宏
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(error) << std::endl; \
            exit(1); \
        } \
    } while(0)

// 设备内存管理类
template<typename T>
class DeviceVector {
private:
    T* d_ptr;
    size_t size_;
    
public:
    DeviceVector(size_t size) : size_(size) {
        CUDA_CHECK(cudaMalloc(&d_ptr, size * sizeof(T)));
    }
    
    ~DeviceVector() {
        if (d_ptr) {
            cudaFree(d_ptr);
        }
    }
    
    // 禁用拷贝构造和赋值
    DeviceVector(const DeviceVector&) = delete;
    DeviceVector& operator=(const DeviceVector&) = delete;
    
    // 移动构造和赋值
    DeviceVector(DeviceVector&& other) noexcept : d_ptr(other.d_ptr), size_(other.size_) {
        other.d_ptr = nullptr;
        other.size_ = 0;
    }
    
    DeviceVector& operator=(DeviceVector&& other) noexcept {
        if (this != &other) {
            if (d_ptr) cudaFree(d_ptr);
            d_ptr = other.d_ptr;
            size_ = other.size_;
            other.d_ptr = nullptr;
            other.size_ = 0;
        }
        return *this;
    }
    
    T* data() { return d_ptr; }
    const T* data() const { return d_ptr; }
    size_t size() const { return size_; }
    
    // 从主机复制数据到设备
    void copyFromHost(const T* host_data) {
        CUDA_CHECK(cudaMemcpy(d_ptr, host_data, size_ * sizeof(T), cudaMemcpyHostToDevice));
    }
    
    // 从设备复制数据到主机
    void copyToHost(T* host_data) const {
        CUDA_CHECK(cudaMemcpy(host_data, d_ptr, size_ * sizeof(T), cudaMemcpyDeviceToHost));
    }
};

// 核心CUDA内核函数模板
template<int N, typename Operation>
__global__ void column_compute_kernel(
    float* const* input_columns,  // N个输入列的指针数组
    float* output_column,         // 输出列
    size_t num_elements,          // 元素数量M
    Operation op                  // 业务逻辑操作符
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    // 网格跨步循环，处理大量数据
    for (size_t i = idx; i < num_elements; i += stride) {
        // 读取当前行的N列数据
        float values[N];
        for (int col = 0; col < N; ++col) {
            values[col] = input_columns[col][i];
        }
        
        // 应用业务逻辑
        output_column[i] = op(values);
    }
}

// 主要的列处理器类
template<int N>
class ColumnProcessor {
private:
    size_t num_elements_;
    std::vector<DeviceVector<float>> input_columns_;
    DeviceVector<float> output_column_;
    
    // 设备上的列指针数组
    DeviceVector<float*> d_column_ptrs_;
    
    // CUDA流用于异步操作
    cudaStream_t stream_;
    
public:
    ColumnProcessor(size_t num_elements) 
        : num_elements_(num_elements)
        , output_column_(num_elements)
        , d_column_ptrs_(N) {
        
        // 为每一列分配内存
        input_columns_.reserve(N);
        for (int i = 0; i < N; ++i) {
            input_columns_.emplace_back(num_elements);
        }
        
        // 创建CUDA流
        CUDA_CHECK(cudaStreamCreate(&stream_));
        
        // 设置设备列指针数组
        updateColumnPointers();
    }
    
    ~ColumnProcessor() {
        if (stream_) {
            cudaStreamDestroy(stream_);
        }
    }
    
    // 更新设备上的列指针数组
    void updateColumnPointers() {
        std::vector<float*> host_ptrs(N);
        for (int i = 0; i < N; ++i) {
            host_ptrs[i] = input_columns_[i].data();
        }
        d_column_ptrs_.copyFromHost(host_ptrs.data());
    }
    
    // 设置输入数据
    void setInputColumn(int column_index, const float* host_data) {
        if (column_index >= 0 && column_index < N) {
            input_columns_[column_index].copyFromHost(host_data);
        }
    }
    
    // 执行计算
    template<typename Operation>
    void compute(Operation op) {
        // 计算网格和块大小
        int blockSize = 256;
        int numBlocks = std::min(65535, static_cast<int>((num_elements_ + blockSize - 1) / blockSize));
        
        // 启动内核
        column_compute_kernel<N><<<numBlocks, blockSize, 0, stream_>>>(
            d_column_ptrs_.data(),
            output_column_.data(),
            num_elements_,
            op
        );
        
        // 检查内核启动错误
        CUDA_CHECK(cudaGetLastError());
    }
    
    // 获取结果
    void getResult(float* host_output) {
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        output_column_.copyToHost(host_output);
    }
    
    // 异步获取结果
    void getResultAsync(float* host_output, cudaStream_t user_stream = nullptr) {
        cudaStream_t target_stream = user_stream ? user_stream : stream_;
        CUDA_CHECK(cudaMemcpyAsync(host_output, output_column_.data(), 
                                 num_elements_ * sizeof(float), 
                                 cudaMemcpyDeviceToHost, target_stream));
    }
    
    // 获取元素数量
    size_t getNumElements() const { return num_elements_; }
    
    // 同步流
    void synchronize() {
        CUDA_CHECK(cudaStreamSynchronize(stream_));
    }
};

// 预定义的常用操作符
struct AddOperation {
    template<int N>
    __device__ float operator()(const float (&values)[N]) const {
        float result = 0.0f;
        #pragma unroll
        for (int i = 0; i < N; ++i) {
            result += values[i];
        }
        return result;
    }
};

struct MultiplyOperation {
    template<int N>
    __device__ float operator()(const float (&values)[N]) const {
        float result = 1.0f;
        #pragma unroll
        for (int i = 0; i < N; ++i) {
            result *= values[i];
        }
        return result;
    }
};

struct WeightedSumOperation {
    float weights[16]; // 支持最多16列
    int actual_cols;
    
    WeightedSumOperation(const std::vector<float>& w) : actual_cols(w.size()) {
        for (int i = 0; i < w.size() && i < 16; ++i) {
            weights[i] = w[i];
        }
    }
    
    template<int N>
    __device__ float operator()(const float (&values)[N]) const {
        float result = 0.0f;
        #pragma unroll
        for (int i = 0; i < N && i < actual_cols; ++i) {
            result += values[i] * weights[i];
        }
        return result;
    }
};

// 便捷的工厂函数
template<int N>
std::unique_ptr<ColumnProcessor<N>> createProcessor(size_t num_elements) {
    return std::make_unique<ColumnProcessor<N>>(num_elements);
}

} // namespace zmm 