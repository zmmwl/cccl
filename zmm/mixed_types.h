#pragma once

#include <cuda_runtime.h>
#include <memory>
#include <vector>
#include <string>
#include <cstring>

namespace zmm {

// 支持的混合数据类型枚举
enum class ColumnDataType {
    FLOAT,
    STRING,
    INT,
    DOUBLE
};

// GPU字符串表示
struct GPUString {
    char* data;         // 指向字符数据的设备指针
    uint32_t length;    // 字符串长度
    
    __device__ __host__ GPUString() : data(nullptr), length(0) {}
    __device__ __host__ GPUString(char* d, uint32_t len) : data(d), length(len) {}
};

// 固定长度字符串（用于简单场景）
template<int MAX_LEN = 256>
struct FixedString {
    char data[MAX_LEN];
    uint32_t actual_length;
    
    __device__ __host__ FixedString() : actual_length(0) {
        data[0] = '\0';
    }
    
    __device__ __host__ void set(const char* str) {
        actual_length = 0;
        while (str[actual_length] != '\0' && actual_length < MAX_LEN - 1) {
            data[actual_length] = str[actual_length];
            actual_length++;
        }
        data[actual_length] = '\0';
    }
    
    __device__ __host__ const char* c_str() const {
        return data;
    }
    
    __device__ __host__ uint32_t length() const {
        return actual_length;
    }
};

// 类型擦除的列接口基类
class IColumn {
public:
    virtual ~IColumn() = default;
    virtual ColumnDataType getType() const = 0;
    virtual void* getDevicePointer() = 0;
    virtual const void* getDevicePointer() const = 0;
    virtual size_t getElementSize() const = 0;
    virtual size_t getNumElements() const = 0;
    virtual void copyFromHost(const void* host_data, size_t num_elements) = 0;
    virtual void copyToHost(void* host_data) const = 0;
};

// 浮点列实现
class FloatColumn : public IColumn {
private:
    float* d_data_;
    size_t num_elements_;
    
public:
    FloatColumn(size_t num_elements);
    ~FloatColumn();
    
    ColumnDataType getType() const override { return ColumnDataType::FLOAT; }
    void* getDevicePointer() override { return d_data_; }
    const void* getDevicePointer() const override { return d_data_; }
    size_t getElementSize() const override { return sizeof(float); }
    size_t getNumElements() const override { return num_elements_; }
    
    void copyFromHost(const void* host_data, size_t num_elements) override;
    void copyToHost(void* host_data) const override;
    
    void setData(const std::vector<float>& data);
    std::vector<float> getData() const;
};

// 字符串列实现
class StringColumn : public IColumn {
private:
    GPUString* d_strings_;      // 设备上的字符串引用数组
    char* d_string_pool_;       // 设备上的字符串池
    size_t num_elements_;
    size_t pool_size_;
    
public:
    StringColumn(size_t num_elements, size_t estimated_total_chars = 1024 * 1024);
    ~StringColumn();
    
    ColumnDataType getType() const override { return ColumnDataType::STRING; }
    void* getDevicePointer() override { return d_strings_; }
    const void* getDevicePointer() const override { return d_strings_; }
    size_t getElementSize() const override { return sizeof(GPUString); }
    size_t getNumElements() const override { return num_elements_; }
    
    void copyFromHost(const void* host_data, size_t num_elements) override;
    void copyToHost(void* host_data) const override;
    
    void setData(const std::vector<std::string>& data);
    std::vector<std::string> getData() const;
};

// 固定长度字符串列实现（更简单，性能更好）
template<int MAX_STRING_LENGTH = 256>
class FixedStringColumn : public IColumn {
private:
    FixedString<MAX_STRING_LENGTH>* d_data_;
    size_t num_elements_;
    
public:
    FixedStringColumn(size_t num_elements);
    ~FixedStringColumn();
    
    ColumnDataType getType() const override { return ColumnDataType::STRING; }
    void* getDevicePointer() override { return d_data_; }
    const void* getDevicePointer() const override { return d_data_; }
    size_t getElementSize() const override { return sizeof(FixedString<MAX_STRING_LENGTH>); }
    size_t getNumElements() const override { return num_elements_; }
    
    void copyFromHost(const void* host_data, size_t num_elements) override;
    void copyToHost(void* host_data) const override;
    
    void setData(const std::vector<std::string>& data);
    std::vector<std::string> getData() const;
};

// 混合行数据结构（用于传递给操作符）
struct MixedRowData {
    void** column_ptrs;         // 指向各列数据的指针数组
    ColumnDataType* types;      // 各列的数据类型
    int num_columns;           // 列数
    size_t row_index;          // 当前行索引
    
    __device__ __host__ MixedRowData() 
        : column_ptrs(nullptr), types(nullptr), num_columns(0), row_index(0) {}
    
    __device__ __host__ MixedRowData(void** ptrs, ColumnDataType* t, int n, size_t idx)
        : column_ptrs(ptrs), types(t), num_columns(n), row_index(idx) {}
    
    // 获取指定列的浮点值
    __device__ __host__ float getFloat(int column_index) const {
        if (column_index >= num_columns || types[column_index] != ColumnDataType::FLOAT) {
            return 0.0f;
        }
        return ((float*)column_ptrs[column_index])[row_index];
    }
    
    // 获取指定列的字符串
    __device__ __host__ GPUString getString(int column_index) const {
        if (column_index >= num_columns || types[column_index] != ColumnDataType::STRING) {
            return GPUString();
        }
        return ((GPUString*)column_ptrs[column_index])[row_index];
    }
    
    // 获取指定列的固定字符串
    template<int MAX_LEN>
    __device__ __host__ FixedString<MAX_LEN> getFixedString(int column_index) const {
        if (column_index >= num_columns || types[column_index] != ColumnDataType::STRING) {
            return FixedString<MAX_LEN>(); // 返回默认构造的对象
        }
        return ((FixedString<MAX_LEN>*)column_ptrs[column_index])[row_index];
    }
};

// 混合操作接口
class IMixedOperation {
public:
    virtual ~IMixedOperation() = default;
    virtual __device__ __host__ float execute(const MixedRowData& row) const = 0;
    virtual const char* getName() const = 0;
};

// CUDA错误检查宏（复用现有的）
#ifndef CUDA_CHECK
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(error) << std::endl; \
            exit(1); \
        } \
    } while(0)
#endif

// GPU字符串工具函数
__device__ __host__ inline bool gpu_string_equals(const GPUString& a, const char* b) {
    if (a.data == nullptr || b == nullptr) return false;
    
    uint32_t b_len = 0;
    while (b[b_len] != '\0') b_len++;
    
    if (a.length != b_len) return false;
    
    for (uint32_t i = 0; i < a.length; ++i) {
        if (a.data[i] != b[i]) return false;
    }
    return true;
}

__device__ __host__ inline bool gpu_string_equals(const GPUString& a, const GPUString& b) {
    if (a.data == nullptr || b.data == nullptr) return false;
    if (a.length != b.length) return false;
    
    for (uint32_t i = 0; i < a.length; ++i) {
        if (a.data[i] != b.data[i]) return false;
    }
    return true;
}

template<int MAX_LEN>
__device__ __host__ inline bool gpu_string_equals(const FixedString<MAX_LEN>& a, const char* b) {
    if (b == nullptr) return false;
    
    uint32_t b_len = 0;
    while (b[b_len] != '\0') b_len++;
    
    if (a.actual_length != b_len) return false;
    
    for (uint32_t i = 0; i < a.actual_length; ++i) {
        if (a.data[i] != b[i]) return false;
    }
    return true;
}

__device__ __host__ inline float gpu_string_to_float(const GPUString& str) {
    if (str.data == nullptr || str.length == 0) return 0.0f;
    
    float result = 0.0f;
    bool negative = false;
    uint32_t i = 0;
    
    // 处理符号
    if (str.data[0] == '-') {
        negative = true;
        i = 1;
    } else if (str.data[0] == '+') {
        i = 1;
    }
    
    // 处理整数部分
    for (; i < str.length && str.data[i] != '.' && str.data[i] >= '0' && str.data[i] <= '9'; ++i) {
        result = result * 10.0f + (str.data[i] - '0');
    }
    
    // 处理小数部分
    if (i < str.length && str.data[i] == '.') {
        i++;
        float decimal = 0.1f;
        for (; i < str.length && str.data[i] >= '0' && str.data[i] <= '9'; ++i) {
            result += (str.data[i] - '0') * decimal;
            decimal *= 0.1f;
        }
    }
    
    return negative ? -result : result;
}

} // namespace zmm
