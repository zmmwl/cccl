#pragma once

#include "column_compute_framework.cuh"
#include <vector>
#include <memory>
#include <algorithm>
#include <functional>

namespace zmm2 {

/**
 * 批处理器类，用于处理超大数据量的分块计算
 * 当数据量超过GPU内存容量时，自动分批处理
 */
template<int N>
class BatchProcessor {
private:
    size_t batch_size_;                     // 每批处理的元素数量
    size_t total_elements_;                 // 总元素数量
    std::unique_ptr<ColumnComputeFramework<N>> framework_;

public:
    /**
     * 构造函数
     * @param total_elements 总元素数量
     * @param max_gpu_memory_mb 最大GPU内存使用量（MB）
     */
    BatchProcessor(size_t total_elements, size_t max_gpu_memory_mb = 4096) 
        : total_elements_(total_elements) {
        
        // 计算每批的大小（考虑内存限制）
        calculateBatchSize(max_gpu_memory_mb);
        
        // 创建计算框架
        framework_ = std::make_unique<ColumnComputeFramework<N>>(batch_size_);
    }
    
    /**
     * 执行批量计算
     * @param host_input_columns 主机端输入列数据
     * @param host_output_column 主机端输出列数据
     * @param expression 算术表达式函数对象
     * @param progress_callback 进度回调函数（可选）
     */
    template<typename ExpressionType>
    void processBatches(
        const std::vector<const float*>& host_input_columns,
        float* host_output_column,
        const ExpressionType& expression,
        std::function<void(size_t, size_t)> progress_callback = nullptr
    ) {
        if (host_input_columns.size() != N) {
            throw std::invalid_argument("输入列数量不匹配");
        }
        
        size_t processed_elements = 0;
        size_t batch_count = (total_elements_ + batch_size_ - 1) / batch_size_;
        
        for (size_t batch_idx = 0; batch_idx < batch_count; ++batch_idx) {
            size_t current_batch_size = std::min(batch_size_, total_elements_ - processed_elements);
            size_t offset = processed_elements;
            
            // 准备当前批次的输入数据
            std::vector<const float*> batch_input_columns(N);
            for (int col = 0; col < N; ++col) {
                batch_input_columns[col] = host_input_columns[col] + offset;
            }
            
            // 如果当前批次大小与框架设置不同，需要重新创建框架
            if (current_batch_size != batch_size_ && batch_idx == batch_count - 1) {
                framework_ = std::make_unique<ColumnComputeFramework<N>>(current_batch_size);
            }
            
            // 处理当前批次
            framework_->setInputData(batch_input_columns);
            framework_->compute(expression);
            framework_->getResult(host_output_column + offset);
            
            processed_elements += current_batch_size;
            
            // 调用进度回调
            if (progress_callback) {
                progress_callback(processed_elements, total_elements_);
            }
        }
    }
    
    /**
     * 获取批处理大小
     */
    size_t getBatchSize() const {
        return batch_size_;
    }
    
    /**
     * 获取批次数量
     */
    size_t getBatchCount() const {
        return (total_elements_ + batch_size_ - 1) / batch_size_;
    }
    
    /**
     * 获取总元素数量
     */
    size_t getTotalElements() const {
        return total_elements_;
    }

private:
    /**
     * 计算批处理大小
     * @param max_gpu_memory_mb 最大GPU内存使用量（MB）
     */
    void calculateBatchSize(size_t max_gpu_memory_mb) {
        // 每个元素需要的内存：N个输入列 + 1个输出列
        size_t memory_per_element = (N + 1) * sizeof(float);
        
        // 转换为字节
        size_t max_gpu_memory_bytes = max_gpu_memory_mb * 1024 * 1024;
        
        // 预留一些内存用于其他用途（如表达式对象、临时变量等）
        size_t available_memory = max_gpu_memory_bytes * 0.8; // 使用80%的内存
        
        // 计算最大可处理的元素数量
        size_t max_elements_per_batch = available_memory / memory_per_element;
        
        // 确保批处理大小不超过总元素数量
        batch_size_ = std::min(max_elements_per_batch, total_elements_);
        
        // 确保批处理大小至少为1
        batch_size_ = std::max(batch_size_, size_t(1));
        
        // 对于非常大的批处理，限制最大值以避免内核启动开销
        batch_size_ = std::min(batch_size_, size_t(100000000)); // 1亿元素
    }
};

/**
 * 便利函数：自动选择最优的处理方式
 * 对于小数据量使用单次处理，对于大数据量使用批处理
 */
template<int N, typename ExpressionType>
void processColumns(
    const std::vector<const float*>& host_input_columns,
    float* host_output_column,
    size_t total_elements,
    const ExpressionType& expression,
    size_t max_gpu_memory_mb = 4096,
    std::function<void(size_t, size_t)> progress_callback = nullptr
) {
    // 估算内存需求
    size_t memory_needed_mb = (total_elements * (N + 1) * sizeof(float)) / (1024 * 1024);
    
    if (memory_needed_mb <= max_gpu_memory_mb * 0.8) {
        // 内存足够，使用单次处理
        ColumnComputeFramework<N> framework(total_elements);
        framework.setInputData(host_input_columns);
        framework.compute(expression);
        framework.getResult(host_output_column);
        
        if (progress_callback) {
            progress_callback(total_elements, total_elements);
        }
    } else {
        // 内存不足，使用批处理
        BatchProcessor<N> processor(total_elements, max_gpu_memory_mb);
        processor.processBatches(host_input_columns, host_output_column, 
                               expression, progress_callback);
    }
}

/**
 * 内存使用情况查询器
 */
class MemoryInfo {
public:
    /**
     * 查询GPU内存使用情况
     */
    static void queryGPUMemory(size_t& free_memory, size_t& total_memory) {
        cudaMemGetInfo(&free_memory, &total_memory);
    }
    
    /**
     * 估算处理指定数据量所需的内存
     */
    template<int N>
    static size_t estimateMemoryUsage(size_t num_elements) {
        // 输入列 + 输出列 + 一些额外开销
        return num_elements * (N + 1) * sizeof(float) * 1.2; // 20%的额外开销
    }
    
    /**
     * 推荐的批处理大小
     */
    template<int N>
    static size_t recommendBatchSize(size_t total_elements) {
        size_t free_memory, total_memory;
        queryGPUMemory(free_memory, total_memory);
        
        // 使用70%的可用内存
        size_t usable_memory = free_memory * 0.7;
        size_t memory_per_element = (N + 1) * sizeof(float);
        
        size_t recommended_batch = usable_memory / memory_per_element;
        return std::min(recommended_batch, total_elements);
    }
};

} // namespace zmm2 