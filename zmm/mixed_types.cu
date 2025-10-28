#include "mixed_types.h"
#include <iostream>
#include <algorithm>

namespace zmm {

// FloatColumn 实现
FloatColumn::FloatColumn(size_t num_elements) : num_elements_(num_elements) {
    CUDA_CHECK(cudaMalloc(&d_data_, num_elements * sizeof(float)));
}

FloatColumn::~FloatColumn() {
    if (d_data_) {
        cudaFree(d_data_);
    }
}

void FloatColumn::copyFromHost(const void* host_data, size_t num_elements) {
    if (num_elements != num_elements_) {
        std::cerr << "FloatColumn: Element count mismatch" << std::endl;
        return;
    }
    CUDA_CHECK(cudaMemcpy(d_data_, host_data, num_elements * sizeof(float), cudaMemcpyHostToDevice));
}

void FloatColumn::copyToHost(void* host_data) const {
    CUDA_CHECK(cudaMemcpy(host_data, d_data_, num_elements_ * sizeof(float), cudaMemcpyDeviceToHost));
}

void FloatColumn::setData(const std::vector<float>& data) {
    if (data.size() != num_elements_) {
        std::cerr << "FloatColumn: Data size mismatch" << std::endl;
        return;
    }
    copyFromHost(data.data(), data.size());
}

std::vector<float> FloatColumn::getData() const {
    std::vector<float> result(num_elements_);
    copyToHost(result.data());
    return result;
}

// IntColumn 实现
IntColumn::IntColumn(size_t num_elements) : num_elements_(num_elements) {
    CUDA_CHECK(cudaMalloc(&d_data_, num_elements * sizeof(int)));
}

IntColumn::~IntColumn() {
    if (d_data_) {
        cudaFree(d_data_);
    }
}

void IntColumn::copyFromHost(const void* host_data, size_t num_elements) {
    if (num_elements != num_elements_) {
        std::cerr << "IntColumn: Element count mismatch" << std::endl;
        return;
    }
    CUDA_CHECK(cudaMemcpy(d_data_, host_data, num_elements * sizeof(int), cudaMemcpyHostToDevice));
}

void IntColumn::copyToHost(void* host_data) const {
    CUDA_CHECK(cudaMemcpy(host_data, d_data_, num_elements_ * sizeof(int), cudaMemcpyDeviceToHost));
}

void IntColumn::setData(const std::vector<int>& data) {
    if (data.size() != num_elements_) {
        std::cerr << "IntColumn: Data size mismatch" << std::endl;
        return;
    }
    copyFromHost(data.data(), data.size());
}

std::vector<int> IntColumn::getData() const {
    std::vector<int> result(num_elements_);
    copyToHost(result.data());
    return result;
}

// StringColumn 实现
StringColumn::StringColumn(size_t num_elements, size_t estimated_total_chars) 
    : num_elements_(num_elements), pool_size_(estimated_total_chars) {
    CUDA_CHECK(cudaMalloc(&d_strings_, num_elements * sizeof(GPUString)));
    CUDA_CHECK(cudaMalloc(&d_string_pool_, pool_size_ * sizeof(char)));
}

StringColumn::~StringColumn() {
    if (d_strings_) {
        cudaFree(d_strings_);
    }
    if (d_string_pool_) {
        cudaFree(d_string_pool_);
    }
}

void StringColumn::copyFromHost(const void* host_data, size_t num_elements) {
    // 这个方法期望host_data是std::vector<std::string>*
    const std::vector<std::string>* strings = static_cast<const std::vector<std::string>*>(host_data);
    setData(*strings);
}

void StringColumn::copyToHost(void* host_data) const {
    std::vector<std::string>* result = static_cast<std::vector<std::string>*>(host_data);
    *result = getData();
}

void StringColumn::setData(const std::vector<std::string>& data) {
    if (data.size() != num_elements_) {
        std::cerr << "StringColumn: Data size mismatch" << std::endl;
        return;
    }
    
    // 计算总字符数
    size_t total_chars = 0;
    for (const auto& str : data) {
        total_chars += str.length() + 1; // +1 for null terminator
    }
    
    if (total_chars > pool_size_) {
        std::cerr << "StringColumn: String pool too small, need " << total_chars 
                  << " but have " << pool_size_ << std::endl;
        return;
    }
    
    // 准备主机端的数据
    std::vector<char> host_pool(total_chars);
    std::vector<GPUString> host_strings(num_elements_);
    
    size_t pool_offset = 0;
    for (size_t i = 0; i < data.size(); ++i) {
        const std::string& str = data[i];
        
        // 复制字符串到池中
        std::memcpy(host_pool.data() + pool_offset, str.c_str(), str.length() + 1);
        
        // 设置字符串引用（注意：这里的指针是设备指针）
        host_strings[i].data = d_string_pool_ + pool_offset;
        host_strings[i].length = static_cast<uint32_t>(str.length());
        
        pool_offset += str.length() + 1;
    }
    
    // 复制到设备
    CUDA_CHECK(cudaMemcpy(d_string_pool_, host_pool.data(), total_chars, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_strings_, host_strings.data(), num_elements_ * sizeof(GPUString), cudaMemcpyHostToDevice));
}

std::vector<std::string> StringColumn::getData() const {
    // 先获取字符串引用
    std::vector<GPUString> host_strings(num_elements_);
    CUDA_CHECK(cudaMemcpy(host_strings.data(), d_strings_, num_elements_ * sizeof(GPUString), cudaMemcpyDeviceToHost));
    
    // 获取字符串池
    std::vector<char> host_pool(pool_size_);
    CUDA_CHECK(cudaMemcpy(host_pool.data(), d_string_pool_, pool_size_, cudaMemcpyDeviceToHost));
    
    // 重构字符串
    std::vector<std::string> result;
    result.reserve(num_elements_);
    
    for (size_t i = 0; i < num_elements_; ++i) {
        const GPUString& gpu_str = host_strings[i];
        
        // 计算在host_pool中的偏移
        size_t offset = gpu_str.data - d_string_pool_;
        if (offset < pool_size_ && gpu_str.length > 0) {
            result.emplace_back(host_pool.data() + offset, gpu_str.length);
        } else {
            result.emplace_back("");
        }
    }
    
    return result;
}

// FixedStringColumn 模板特化实现
template<int MAX_STRING_LENGTH>
FixedStringColumn<MAX_STRING_LENGTH>::FixedStringColumn(size_t num_elements) : num_elements_(num_elements) {
    CUDA_CHECK(cudaMalloc(&d_data_, num_elements * sizeof(FixedString<MAX_STRING_LENGTH>)));
}

template<int MAX_STRING_LENGTH>
FixedStringColumn<MAX_STRING_LENGTH>::~FixedStringColumn() {
    if (d_data_) {
        cudaFree(d_data_);
    }
}

template<int MAX_STRING_LENGTH>
void FixedStringColumn<MAX_STRING_LENGTH>::copyFromHost(const void* host_data, size_t num_elements) {
    const std::vector<std::string>* strings = static_cast<const std::vector<std::string>*>(host_data);
    setData(*strings);
}

template<int MAX_STRING_LENGTH>
void FixedStringColumn<MAX_STRING_LENGTH>::copyToHost(void* host_data) const {
    std::vector<std::string>* result = static_cast<std::vector<std::string>*>(host_data);
    *result = getData();
}

template<int MAX_STRING_LENGTH>
void FixedStringColumn<MAX_STRING_LENGTH>::setData(const std::vector<std::string>& data) {
    if (data.size() != num_elements_) {
        std::cerr << "FixedStringColumn: Data size mismatch" << std::endl;
        return;
    }
    
    std::vector<FixedString<MAX_STRING_LENGTH>> host_data(num_elements_);
    
    for (size_t i = 0; i < data.size(); ++i) {
        if (data[i].length() >= MAX_STRING_LENGTH) {
            std::cerr << "Warning: String too long, will be truncated: " << data[i] << std::endl;
        }
        host_data[i].set(data[i].c_str());
    }
    
    CUDA_CHECK(cudaMemcpy(d_data_, host_data.data(), 
                         num_elements_ * sizeof(FixedString<MAX_STRING_LENGTH>), 
                         cudaMemcpyHostToDevice));
}

template<int MAX_STRING_LENGTH>
std::vector<std::string> FixedStringColumn<MAX_STRING_LENGTH>::getData() const {
    std::vector<FixedString<MAX_STRING_LENGTH>> host_data(num_elements_);
    CUDA_CHECK(cudaMemcpy(host_data.data(), d_data_, 
                         num_elements_ * sizeof(FixedString<MAX_STRING_LENGTH>), 
                         cudaMemcpyDeviceToHost));
    
    std::vector<std::string> result;
    result.reserve(num_elements_);
    
    for (const auto& fixed_str : host_data) {
        result.emplace_back(fixed_str.c_str());
    }
    
    return result;
}

// 显式实例化常用的模板
template class FixedStringColumn<64>;
template class FixedStringColumn<128>;
template class FixedStringColumn<256>;
template class FixedStringColumn<512>;

} // namespace zmm
