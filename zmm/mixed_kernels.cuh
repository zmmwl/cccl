#pragma once

#include "mixed_types.h"
#include <cuda_runtime.h>

namespace zmm {

// 混合类型数据处理的核心内核函数
__global__ void mixed_compute_kernel(
    void** column_ptrs,           // 各列数据的设备指针数组
    ColumnDataType* column_types, // 各列的数据类型数组
    uint32_t* column_name_hashes, // 各列名称哈希（可为null）
    int num_columns,             // 列数
    size_t num_elements,         // 元素数量
    float* output,               // 输出数组
    void* operation_params       // 操作参数（可选）
);

// 专用的混合计算内核（带操作符指针）
template<typename OperationFunc>
__global__ void mixed_compute_kernel_with_op(
    void** column_ptrs,           // 各列数据的设备指针数组
    ColumnDataType* column_types, // 各列的数据类型数组
    uint32_t* column_name_hashes, // 各列名称哈希（可为null）
    int num_columns,             // 列数
    size_t num_elements,         // 元素数量
    float* output,               // 输出数组
    OperationFunc operation      // 操作函数对象
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    for (size_t i = idx; i < num_elements; i += stride) {
        // 构造当前行的混合数据
        MixedRowData row(column_ptrs, column_types, column_name_hashes, num_columns, i);
        
        // 应用操作并存储结果
        output[i] = operation(row);
    }
}

// 条件分支处理的内核（用于运行时确定的操作）
__global__ void mixed_conditional_kernel(
    void** column_ptrs,
    ColumnDataType* column_types,
    int num_columns,
    size_t num_elements,
    float* output,
    int operation_type  // 操作类型ID
);

// 字符串处理专用内核
__global__ void string_processing_kernel(
    GPUString* string_columns,
    int num_string_columns,
    float* float_columns,
    int num_float_columns,
    size_t num_elements,
    float* output,
    int operation_mode
);

// 固定长度字符串处理内核
template<int MAX_STRING_LENGTH>
__global__ void fixed_string_processing_kernel(
    FixedString<MAX_STRING_LENGTH>* string_columns,
    int num_string_columns,
    float* float_columns,
    int num_float_columns,
    size_t num_elements,
    float* output,
    int operation_mode
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    for (size_t i = idx; i < num_elements; i += stride) {
        float result = 0.0f;
        
        // 处理字符串列
        for (int col = 0; col < num_string_columns; ++col) {
            const FixedString<MAX_STRING_LENGTH>& str = string_columns[col * num_elements + i];
            
            switch (operation_mode) {
                case 0: // 字符串长度
                    result += static_cast<float>(str.length());
                    break;
                case 1: // 字符串hash（简单实现）
                    {
                        uint32_t hash = 0;
                        for (uint32_t j = 0; j < str.actual_length; ++j) {
                            hash = hash * 31 + static_cast<uint32_t>(str.data[j]);
                        }
                        result += static_cast<float>(hash % 1000); // 模1000避免过大
                    }
                    break;
                case 2: // 字符串数值转换
                    {
                        float val = 0.0f;
                        bool negative = false;
                        uint32_t start = 0;
                        
                        if (str.actual_length > 0 && str.data[0] == '-') {
                            negative = true;
                            start = 1;
                        }
                        
                        for (uint32_t j = start; j < str.actual_length; ++j) {
                            if (str.data[j] >= '0' && str.data[j] <= '9') {
                                val = val * 10.0f + (str.data[j] - '0');
                            } else if (str.data[j] == '.') {
                                // 简单的小数处理
                                float decimal = 0.1f;
                                for (uint32_t k = j + 1; k < str.actual_length && str.data[k] >= '0' && str.data[k] <= '9'; ++k) {
                                    val += (str.data[k] - '0') * decimal;
                                    decimal *= 0.1f;
                                }
                                break;
                            } else {
                                break; // 遇到非数字字符停止
                            }
                        }
                        
                        result += negative ? -val : val;
                    }
                    break;
            }
        }
        
        // 处理浮点列
        for (int col = 0; col < num_float_columns; ++col) {
            result += float_columns[col * num_elements + i];
        }
        
        output[i] = result;
    }
}

// 高级字符串操作内核
__global__ void advanced_string_ops_kernel(
    GPUString* strings,
    size_t num_elements,
    float* output,
    const char* pattern,        // 匹配模式
    int pattern_length,
    int operation_type         // 0=contains, 1=starts_with, 2=ends_with
);

// 字符串分类内核（根据字符串内容分类）
template<int MAX_CATEGORIES = 10>
__global__ void string_classification_kernel(
    GPUString* strings,
    size_t num_elements,
    const char** category_patterns,    // 分类模式数组
    int* pattern_lengths,             // 各模式长度
    int num_categories,
    float* category_weights,          // 各分类权重
    float* output
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    for (size_t i = idx; i < num_elements; i += stride) {
        const GPUString& str = strings[i];
        float result = 0.0f;
        
        // 检查每个分类模式
        for (int cat = 0; cat < num_categories && cat < MAX_CATEGORIES; ++cat) {
            bool matches = false;
            const char* pattern = category_patterns[cat];
            int pattern_len = pattern_lengths[cat];
            
            if (pattern_len <= str.length) {
                matches = true;
                for (int j = 0; j < pattern_len; ++j) {
                    if (str.data[j] != pattern[j]) {
                        matches = false;
                        break;
                    }
                }
            }
            
            if (matches) {
                result += category_weights[cat];
            }
        }
        
        output[i] = result;
    }
}

// 混合聚合内核（支持不同类型的聚合操作）
__global__ void mixed_aggregation_kernel(
    void** column_ptrs,
    ColumnDataType* column_types,
    int num_columns,
    size_t num_elements,
    float* partial_results,    // 每个块的部分结果
    int aggregation_type       // 0=sum, 1=avg, 2=count_non_null, 3=mixed
);

// 混合排序辅助内核
__global__ void mixed_sort_key_generation_kernel(
    void** column_ptrs,
    ColumnDataType* column_types,
    int* sort_columns,         // 排序列索引
    int num_sort_columns,
    size_t num_elements,
    float* sort_keys          // 生成的排序键
);

// 数据类型转换内核
__global__ void mixed_type_conversion_kernel(
    void** input_ptrs,
    ColumnDataType* input_types,
    void** output_ptrs,
    ColumnDataType* output_types,
    int num_columns,
    size_t num_elements
);

// 多输出混合计算内核（支持多个operator）
template<typename... OperationFuncs>
__global__ void mixed_compute_kernel_multi_op(
    void** column_ptrs,
    ColumnDataType* column_types,
    uint32_t* column_name_hashes,
    int num_columns,
    size_t num_elements,
    float** outputs,  // 多个输出数组
    int num_outputs,
    OperationFuncs... operations
);

// 辅助函数用于执行多个operations
template<int OpIdx, typename... Ops>
struct MultiOpExecutor;

// 递归情况：至少有一个operator
template<int OpIdx, typename FirstOp, typename... RestOps>
struct MultiOpExecutor<OpIdx, FirstOp, RestOps...> {
    __device__ static void execute(const MixedRowData& row, float** outputs, size_t i, 
                                    FirstOp first, RestOps... rest) {
        outputs[OpIdx][i] = first(row);
        MultiOpExecutor<OpIdx + 1, RestOps...>::execute(row, outputs, i, rest...);
    }
};

// 递归终止：没有更多operators
template<int OpIdx>
struct MultiOpExecutor<OpIdx> {
    __device__ static void execute(const MixedRowData& row, float** outputs, size_t i) {
        // 终止递归
    }
};

// 多operator内核实现
template<typename... OperationFuncs>
__global__ void mixed_compute_kernel_multi_op(
    void** column_ptrs,
    ColumnDataType* column_types,
    uint32_t* column_name_hashes,
    int num_columns,
    size_t num_elements,
    float** outputs,
    int num_outputs,
    OperationFuncs... operations
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    for (size_t i = idx; i < num_elements; i += stride) {
        MixedRowData row(column_ptrs, column_types, column_name_hashes, num_columns, i);
        MultiOpExecutor<0, OperationFuncs...>::execute(row, outputs, i, operations...);
    }
}

// 内核启动辅助函数
namespace kernel_launcher {

// 计算合适的网格和块大小
inline void calculate_launch_config(size_t num_elements, int& num_blocks, int& block_size) {
    block_size = 256;
    num_blocks = std::min(65535, static_cast<int>((num_elements + block_size - 1) / block_size));
}

// 启动混合计算内核
template<typename OperationFunc>
cudaError_t launch_mixed_compute_kernel(
    void** column_ptrs,
    ColumnDataType* column_types,
    uint32_t* column_name_hashes,
    int num_columns,
    size_t num_elements,
    float* output,
    OperationFunc operation,
    cudaStream_t stream = 0
) {
    int num_blocks, block_size;
    calculate_launch_config(num_elements, num_blocks, block_size);

    printf("launch_mixed_compute_kernel: num_blocks=%d, block_size=%d\n", num_blocks, block_size);
    
    mixed_compute_kernel_with_op<<<num_blocks, block_size, 0, stream>>>(
        column_ptrs, column_types, column_name_hashes, num_columns, num_elements, output, operation
    );
    
    return cudaGetLastError();
}

// 启动多operator混合计算内核
template<typename... OperationFuncs>
cudaError_t launch_mixed_compute_kernel_multi(
    void** column_ptrs,
    ColumnDataType* column_types,
    uint32_t* column_name_hashes,
    int num_columns,
    size_t num_elements,
    float** outputs,
    int num_outputs,
    cudaStream_t stream,
    OperationFuncs... operations
) {
    int num_blocks, block_size;
    calculate_launch_config(num_elements, num_blocks, block_size);

    printf("launch_mixed_compute_kernel_multi: num_blocks=%d, block_size=%d, num_outputs=%d\n", 
           num_blocks, block_size, num_outputs);
    
    mixed_compute_kernel_multi_op<<<num_blocks, block_size, 0, stream>>>(
        column_ptrs, column_types, column_name_hashes, num_columns, num_elements, 
        outputs, num_outputs, operations...
    );
    
    return cudaGetLastError();
}

// 启动字符串处理内核
cudaError_t launch_string_processing_kernel(
    GPUString* string_columns,
    int num_string_columns,
    float* float_columns,
    int num_float_columns,
    size_t num_elements,
    float* output,
    int operation_mode,
    cudaStream_t stream = 0
);

// 启动固定长度字符串处理内核
template<int MAX_STRING_LENGTH>
cudaError_t launch_fixed_string_processing_kernel(
    FixedString<MAX_STRING_LENGTH>* string_columns,
    int num_string_columns,
    float* float_columns,
    int num_float_columns,
    size_t num_elements,
    float* output,
    int operation_mode,
    cudaStream_t stream = 0
) {
    int num_blocks, block_size;
    calculate_launch_config(num_elements, num_blocks, block_size);

    fixed_string_processing_kernel<<<num_blocks, block_size, 0, stream>>>(
        string_columns, num_string_columns, float_columns, num_float_columns,
        num_elements, output, operation_mode
    );
    
    return cudaGetLastError();
}

} // namespace kernel_launcher

} // namespace zmm
