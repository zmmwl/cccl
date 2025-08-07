#pragma once

#include "mixed_types.h"
#include "mixed_kernels.cuh"
#include "mixed_operations.cuh"
#include <vector>
#include <memory>
#include <unordered_map>
#include <iostream>

namespace zmm {

// 混合列处理器主类
class MixedColumnProcessor {
private:
    size_t num_elements_;
    std::vector<std::unique_ptr<IColumn>> columns_;
    std::vector<ColumnDataType> column_types_;
    
    // 设备端的元数据
    void** d_column_ptrs_;              // 设备上各列数据指针数组
    ColumnDataType* d_column_types_;    // 设备上各列类型数组
    float* d_output_;                   // 设备上的输出数组
    
    // CUDA流
    cudaStream_t stream_;
    
    // 性能统计
    mutable float last_compute_time_ms_;
    mutable size_t total_operations_;
    
public:
    explicit MixedColumnProcessor(size_t num_elements);
    ~MixedColumnProcessor();
    
    // 禁用拷贝构造和赋值
    MixedColumnProcessor(const MixedColumnProcessor&) = delete;
    MixedColumnProcessor& operator=(const MixedColumnProcessor&) = delete;
    
    // 移动构造和赋值
    MixedColumnProcessor(MixedColumnProcessor&& other) noexcept;
    MixedColumnProcessor& operator=(MixedColumnProcessor&& other) noexcept;
    
    // === 列管理方法 ===
    
    // 添加浮点列
    int addFloatColumn(const std::vector<float>& data);
    int addFloatColumn(const float* data, size_t size);
    
    // 添加字符串列
    int addStringColumn(const std::vector<std::string>& data);
    
    // 添加固定长度字符串列
    template<int MAX_LEN = 256>
    int addFixedStringColumn(const std::vector<std::string>& data);
    
    // 通用添加列方法
    int addColumn(ColumnDataType type, const void* data, size_t size);
    
    // 获取列信息
    size_t getNumColumns() const { return columns_.size(); }
    size_t getNumElements() const { return num_elements_; }
    ColumnDataType getColumnType(int column_index) const;
    
    // 更新列数据
    bool updateFloatColumn(int column_index, const std::vector<float>& data);
    bool updateStringColumn(int column_index, const std::vector<std::string>& data);
    
    // === 计算执行方法 ===
    
    // 使用预定义操作执行计算
    bool compute(MixedOperationFactory::OperationType operation_type, float param = 1.0f);
    
    // 使用自定义操作对象执行计算
    bool compute(const IMixedOperation& operation);
    
    // 使用函数对象执行计算
    template<typename FunctorType>
    bool computeWithFunctor(FunctorType functor);
    
    // 使用操作ID执行计算（条件分支方式）
    bool computeConditional(int operation_id);
    
    // 异步计算版本
    template<typename FunctorType>
    bool computeAsync(FunctorType functor);
    
    // === 结果获取方法 ===
    
    // 同步获取结果
    std::vector<float> getResult() const;
    void getResult(float* output) const;
    void getResult(std::vector<float>& output) const;
    
    // 异步获取结果
    void getResultAsync(float* output, cudaStream_t user_stream = nullptr) const;
    
    // === 流控制方法 ===
    
    void synchronize() const;
    cudaStream_t getStream() const { return stream_; }
    
    // === 实用方法 ===
    
    // 数据验证
    bool validateData() const;
    
    // 性能统计
    float getLastComputeTimeMs() const { return last_compute_time_ms_; }
    size_t getTotalOperations() const { return total_operations_; }
    void resetPerformanceCounters();
    
    // 调试信息
    void printColumnInfo() const;
    void printSampleData(int num_samples = 5) const;
    
    // 内存使用情况
    size_t getDeviceMemoryUsage() const;
    
private:
    // 内部辅助方法
    void updateDeviceMetadata();
    void cleanupDeviceMemory();
    bool validateColumnIndex(int column_index) const;
    void recordComputeTime(float time_ms) const;
    
    // GPU内存管理
    bool allocateDeviceMetadata();
    void freeDeviceMetadata();
};

// 模板方法实现

template<int MAX_LEN>
int MixedColumnProcessor::addFixedStringColumn(const std::vector<std::string>& data) {
    if (data.size() != num_elements_) {
        std::cerr << "MixedColumnProcessor: Data size mismatch, expected " 
                  << num_elements_ << " but got " << data.size() << std::endl;
        return -1;
    }
    
    try {
        auto column = std::make_unique<FixedStringColumn<MAX_LEN>>(num_elements_);
        column->setData(data);
        
        int column_index = static_cast<int>(columns_.size());
        columns_.push_back(std::move(column));
        column_types_.push_back(ColumnDataType::STRING);
        
        updateDeviceMetadata();
        return column_index;
    } catch (const std::exception& e) {
        std::cerr << "MixedColumnProcessor: Failed to add fixed string column: " << e.what() << std::endl;
        return -1;
    }
}

template<typename FunctorType>
bool MixedColumnProcessor::computeWithFunctor(FunctorType functor) {
    if (columns_.empty()) {
        std::cerr << "MixedColumnProcessor: No columns to process" << std::endl;
        return false;
    }
    
    cudaError_t result = kernel_launcher::launch_mixed_compute_kernel(
        d_column_ptrs_,
        d_column_types_,
        static_cast<int>(columns_.size()),
        num_elements_,
        d_output_,
        functor,
        stream_
    );
    
    if (result != cudaSuccess) {
        std::cerr << "MixedColumnProcessor: Kernel launch failed: " 
                  << cudaGetErrorString(result) << std::endl;
        return false;
    }
    
    // 记录性能（简化版本）
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    total_operations_++;
    return true;
}

template<typename FunctorType>
bool MixedColumnProcessor::computeAsync(FunctorType functor) {
    if (columns_.empty()) {
        std::cerr << "MixedColumnProcessor: No columns to process" << std::endl;
        return false;
    }
    
    cudaError_t result = kernel_launcher::launch_mixed_compute_kernel(
        d_column_ptrs_,
        d_column_types_,
        static_cast<int>(columns_.size()),
        num_elements_,
        d_output_,
        functor,
        stream_
    );
    
    if (result != cudaSuccess) {
        std::cerr << "MixedColumnProcessor: Async kernel launch failed: " 
                  << cudaGetErrorString(result) << std::endl;
        return false;
    }
    
    total_operations_++;
    return true;
}

// 便捷的工厂函数
std::unique_ptr<MixedColumnProcessor> createMixedProcessor(size_t num_elements);

// 构建器模式的处理器创建器
class MixedProcessorBuilder {
private:
    size_t num_elements_;
    std::vector<std::pair<ColumnDataType, const void*>> column_data_;
    std::vector<size_t> data_sizes_;
    
public:
    explicit MixedProcessorBuilder(size_t num_elements) : num_elements_(num_elements) {}
    
    MixedProcessorBuilder& addFloatColumn(const std::vector<float>& data) {
        column_data_.emplace_back(ColumnDataType::FLOAT, data.data());
        data_sizes_.push_back(data.size());
        return *this;
    }
    
    MixedProcessorBuilder& addStringColumn(const std::vector<std::string>& data) {
        column_data_.emplace_back(ColumnDataType::STRING, &data);
        data_sizes_.push_back(data.size());
        return *this;
    }
    
    std::unique_ptr<MixedColumnProcessor> build() {
        auto processor = std::make_unique<MixedColumnProcessor>(num_elements_);
        
        for (size_t i = 0; i < column_data_.size(); ++i) {
            ColumnDataType type = column_data_[i].first;
            const void* data = column_data_[i].second;
            processor->addColumn(type, data, data_sizes_[i]);
        }
        
        return processor;
    }
};

// 批量处理器（处理多个数据集）
class BatchMixedProcessor {
private:
    std::vector<std::unique_ptr<MixedColumnProcessor>> processors_;
    size_t batch_size_;
    
public:
    explicit BatchMixedProcessor(size_t batch_size) : batch_size_(batch_size) {}
    
    void addProcessor(std::unique_ptr<MixedColumnProcessor> processor) {
        processors_.push_back(std::move(processor));
    }
    
    // 批量执行相同的操作
    bool batchCompute(MixedOperationFactory::OperationType operation_type) {
        bool all_success = true;
        for (auto& processor : processors_) {
            if (!processor->compute(operation_type)) {
                all_success = false;
            }
        }
        return all_success;
    }
    
    // 获取所有结果
    std::vector<std::vector<float>> getAllResults() const {
        std::vector<std::vector<float>> results;
        results.reserve(processors_.size());
        
        for (const auto& processor : processors_) {
            results.push_back(processor->getResult());
        }
        
        return results;
    }
    
    size_t getBatchSize() const { return processors_.size(); }
};

// 性能基准测试类
class MixedProcessorBenchmark {
public:
    struct BenchmarkResult {
        float avg_compute_time_ms;
        float min_compute_time_ms;
        float max_compute_time_ms;
        float throughput_ops_per_sec;
        size_t total_operations;
        size_t memory_usage_bytes;
    };
    
    static BenchmarkResult benchmark(
        MixedColumnProcessor& processor,
        MixedOperationFactory::OperationType operation_type,
        int num_iterations = 100
    );
    
    static void compareBenchmarks(
        const std::vector<BenchmarkResult>& results,
        const std::vector<std::string>& test_names
    );
};

} // namespace zmm
