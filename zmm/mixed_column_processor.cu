#include "mixed_column_processor.cuh"
#include <cub/cub.cuh>
#include <thrust/device_ptr.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include <numeric>

namespace zmm {

// 构造函数
MixedColumnProcessor::MixedColumnProcessor(size_t num_elements) 
    : num_elements_(num_elements)
    , d_column_ptrs_(nullptr)
    , d_column_types_(nullptr)
    , d_column_name_hashes_(nullptr)
    , d_output_(nullptr)
    , num_outputs_(0)
    , stream_(nullptr)
    , last_compute_time_ms_(0.0f)
    , total_operations_(0) {
    
    // 创建CUDA流
    CUDA_CHECK(cudaStreamCreate(&stream_));
    
    // 分配输出内存
    CUDA_CHECK(cudaMalloc(&d_output_, num_elements * sizeof(float)));
    
    // 初始化设备元数据
    allocateDeviceMetadata();
}

// 析构函数
MixedColumnProcessor::~MixedColumnProcessor() {
    cleanupDeviceMemory();
    
    if (stream_) {
        cudaStreamDestroy(stream_);
    }
}

// 移动构造函数
MixedColumnProcessor::MixedColumnProcessor(MixedColumnProcessor&& other) noexcept
    : num_elements_(other.num_elements_)
    , columns_(std::move(other.columns_))
    , column_types_(std::move(other.column_types_))
    , d_column_ptrs_(other.d_column_ptrs_)
    , d_column_types_(other.d_column_types_)
    , d_column_name_hashes_(other.d_column_name_hashes_)
    , d_output_(other.d_output_)
    , stream_(other.stream_)
    , last_compute_time_ms_(other.last_compute_time_ms_)
    , total_operations_(other.total_operations_) {
    
    // 清空源对象
    other.d_column_ptrs_ = nullptr;
    other.d_column_types_ = nullptr;
    other.d_column_name_hashes_ = nullptr;
    other.d_output_ = nullptr;
    other.stream_ = nullptr;
}

// 移动赋值运算符
MixedColumnProcessor& MixedColumnProcessor::operator=(MixedColumnProcessor&& other) noexcept {
    if (this != &other) {
        // 清理当前资源
        cleanupDeviceMemory();
        if (stream_) {
            cudaStreamDestroy(stream_);
        }
        
        // 移动数据
        num_elements_ = other.num_elements_;
        columns_ = std::move(other.columns_);
        column_types_ = std::move(other.column_types_);
        d_column_ptrs_ = other.d_column_ptrs_;
        d_column_types_ = other.d_column_types_;
        d_column_name_hashes_ = other.d_column_name_hashes_;
        d_output_ = other.d_output_;
        stream_ = other.stream_;
        last_compute_time_ms_ = other.last_compute_time_ms_;
        total_operations_ = other.total_operations_;
        
        // 清空源对象
        other.d_column_ptrs_ = nullptr;
        other.d_column_types_ = nullptr;
        other.d_column_name_hashes_ = nullptr;
        other.d_output_ = nullptr;
        other.stream_ = nullptr;
    }
    return *this;
}

// 添加浮点列
int MixedColumnProcessor::addFloatColumn(const std::vector<float>& data) {
    if (data.size() != num_elements_) {
        std::cerr << "MixedColumnProcessor: Data size mismatch, expected " 
                  << num_elements_ << " but got " << data.size() << std::endl;
        return -1;
    }
    
    try {
        auto column = std::make_unique<FloatColumn>(num_elements_);
        column->setData(data);
        
        int column_index = static_cast<int>(columns_.size());
        columns_.push_back(std::move(column));
        column_types_.push_back(ColumnDataType::FLOAT);
        column_name_hashes_.push_back(0u);
        
        updateDeviceMetadata();
        return column_index;
    } catch (const std::exception& e) {
        std::cerr << "MixedColumnProcessor: Failed to add float column: " << e.what() << std::endl;
        return -1;
    }
}

int MixedColumnProcessor::addNamedFloatColumn(const std::string& name, const std::vector<float>& data) {
    int idx = addFloatColumn(data);
    if (idx >= 0) {
        name_to_index_[name] = idx;
        if (column_name_hashes_.size() < columns_.size()) column_name_hashes_.resize(columns_.size());
        column_name_hashes_[static_cast<size_t>(idx)] = MixedRowData::hashColumnName(name.c_str());
        updateDeviceMetadata();
    }
    return idx;
}

int MixedColumnProcessor::addNamedFloatColumn(const std::string& name, const float* data, size_t size) {
    if (size != num_elements_) {
        std::cerr << "MixedColumnProcessor: Data size mismatch" << std::endl;
        return -1;
    }
    std::vector<float> vec_data(data, data + size);
    return addNamedFloatColumn(name, vec_data);
}

int MixedColumnProcessor::addFloatColumn(const float* data, size_t size) {
    if (size != num_elements_) {
        std::cerr << "MixedColumnProcessor: Data size mismatch" << std::endl;
        return -1;
    }
    
    std::vector<float> vec_data(data, data + size);
    return addFloatColumn(vec_data);
}

// 添加整数列
int MixedColumnProcessor::addIntColumn(const std::vector<int>& data) {
    if (data.size() != num_elements_) {
        std::cerr << "MixedColumnProcessor: Data size mismatch, expected " 
                  << num_elements_ << " but got " << data.size() << std::endl;
        return -1;
    }
    
    try {
        auto column = std::make_unique<IntColumn>(num_elements_);
        column->setData(data);
        
        int column_index = static_cast<int>(columns_.size());
        columns_.push_back(std::move(column));
        column_types_.push_back(ColumnDataType::INT);
        column_name_hashes_.push_back(0u);
        
        updateDeviceMetadata();
        return column_index;
    } catch (const std::exception& e) {
        std::cerr << "MixedColumnProcessor: Failed to add int column: " << e.what() << std::endl;
        return -1;
    }
}

int MixedColumnProcessor::addNamedIntColumn(const std::string& name, const std::vector<int>& data) {
    int idx = addIntColumn(data);
    if (idx >= 0) {
        name_to_index_[name] = idx;
        if (column_name_hashes_.size() < columns_.size()) column_name_hashes_.resize(columns_.size());
        column_name_hashes_[static_cast<size_t>(idx)] = MixedRowData::hashColumnName(name.c_str());
        updateDeviceMetadata();
    }
    return idx;
}

int MixedColumnProcessor::addNamedIntColumn(const std::string& name, const int* data, size_t size) {
    if (size != num_elements_) {
        std::cerr << "MixedColumnProcessor: Data size mismatch" << std::endl;
        return -1;
    }
    std::vector<int> vec_data(data, data + size);
    return addNamedIntColumn(name, vec_data);
}

int MixedColumnProcessor::addIntColumn(const int* data, size_t size) {
    if (size != num_elements_) {
        std::cerr << "MixedColumnProcessor: Data size mismatch" << std::endl;
        return -1;
    }
    
    std::vector<int> vec_data(data, data + size);
    return addIntColumn(vec_data);
}

// 添加字符串列
int MixedColumnProcessor::addStringColumn(const std::vector<std::string>& data) {
    if (data.size() != num_elements_) {
        std::cerr << "MixedColumnProcessor: Data size mismatch" << std::endl;
        return -1;
    }
    
    try {
        // 估算总字符数
        size_t total_chars = 0;
        for (const auto& str : data) {
            total_chars += str.length() + 1;
        }
        total_chars = std::max(total_chars, static_cast<size_t>(1024 * 1024)); // 至少1MB
        
        auto column = std::make_unique<StringColumn>(num_elements_, total_chars);
        column->setData(data);
        
        int column_index = static_cast<int>(columns_.size());
        columns_.push_back(std::move(column));
        column_types_.push_back(ColumnDataType::STRING);
        column_name_hashes_.push_back(0u);
        
        updateDeviceMetadata();
        return column_index;
    } catch (const std::exception& e) {
        std::cerr << "MixedColumnProcessor: Failed to add string column: " << e.what() << std::endl;
        return -1;
    }
}

int MixedColumnProcessor::addNamedStringColumn(const std::string& name, const std::vector<std::string>& data) {
    int idx = addStringColumn(data);
    if (idx >= 0) {
        name_to_index_[name] = idx;
        if (column_name_hashes_.size() < columns_.size()) column_name_hashes_.resize(columns_.size());
        column_name_hashes_[static_cast<size_t>(idx)] = MixedRowData::hashColumnName(name.c_str());
        updateDeviceMetadata();
    }
    return idx;
}

// 通用添加列方法
int MixedColumnProcessor::addColumn(ColumnDataType type, const void* data, size_t size) {
    if (size != num_elements_) {
        std::cerr << "MixedColumnProcessor: Data size mismatch" << std::endl;
        return -1;
    }
    
    switch (type) {
        case ColumnDataType::FLOAT:
            return addFloatColumn(static_cast<const float*>(data), size);
            
        case ColumnDataType::STRING:
            {
                const auto* string_data = static_cast<const std::vector<std::string>*>(data);
                return addStringColumn(*string_data);
            }
            
        default:
            std::cerr << "MixedColumnProcessor: Unsupported column type" << std::endl;
            return -1;
    }
}

int MixedColumnProcessor::addNamedColumn(const std::string& name, ColumnDataType type, const void* data, size_t size) {
    int idx = addColumn(type, data, size);
    if (idx >= 0) {
        name_to_index_[name] = idx;
        if (column_name_hashes_.size() < columns_.size()) column_name_hashes_.resize(columns_.size());
        column_name_hashes_[static_cast<size_t>(idx)] = MixedRowData::hashColumnName(name.c_str());
        updateDeviceMetadata();
    }
    return idx;
}

// 获取列类型
ColumnDataType MixedColumnProcessor::getColumnType(int column_index) const {
    if (!validateColumnIndex(column_index)) {
        return ColumnDataType::FLOAT; // 默认返回
    }
    return column_types_[column_index];
}

int MixedColumnProcessor::getColumnIndexByName(const std::string& name) const {
    auto it = name_to_index_.find(name);
    if (it == name_to_index_.end()) return -1;
    return it->second;
}

// 更新浮点列数据
bool MixedColumnProcessor::updateFloatColumn(int column_index, const std::vector<float>& data) {
    if (!validateColumnIndex(column_index) || 
        column_types_[column_index] != ColumnDataType::FLOAT ||
        data.size() != num_elements_) {
        return false;
    }
    
    try {
        auto* float_column = static_cast<FloatColumn*>(columns_[column_index].get());
        float_column->setData(data);
        return true;
    } catch (const std::exception& e) {
        std::cerr << "MixedColumnProcessor: Failed to update float column: " << e.what() << std::endl;
        return false;
    }
}

// 更新字符串列数据
bool MixedColumnProcessor::updateStringColumn(int column_index, const std::vector<std::string>& data) {
    if (!validateColumnIndex(column_index) || 
        column_types_[column_index] != ColumnDataType::STRING ||
        data.size() != num_elements_) {
        return false;
    }
    
    try {
        auto* string_column = static_cast<StringColumn*>(columns_[column_index].get());
        string_column->setData(data);
        return true;
    } catch (const std::exception& e) {
        std::cerr << "MixedColumnProcessor: Failed to update string column: " << e.what() << std::endl;
        return false;
    }
}

// 使用预定义操作执行计算
bool MixedColumnProcessor::compute(MixedOperationFactory::OperationType operation_type, float param) {
    auto operation = MixedOperationFactory::createOperation(operation_type, param);
    return compute(*operation);
}

// 使用自定义操作对象执行计算
bool MixedColumnProcessor::compute(const IMixedOperation& operation) {
    if (columns_.empty()) {
        std::cerr << "MixedColumnProcessor: No columns to process" << std::endl;
        return false;
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // 根据操作类型选择合适的函数对象
    bool success = false;
    const char* op_name = operation.getName();
    
    if (strcmp(op_name, "MixedAddOperation") == 0) {
        success = computeWithFunctor(MixedAddFunctor{});
    } else if (strcmp(op_name, "StringLengthSumOperation") == 0) {
        success = computeWithFunctor(StringLengthSumFunctor{});
    } else if (strcmp(op_name, "ConditionalMixedOperation") == 0) {
        success = computeWithFunctor(ConditionalMixedFunctor{});
    } else if (strcmp(op_name, "EcommerceScoreOperation") == 0) {
        success = computeWithFunctor(EcommerceScoreFunctor{});
    } else if (strcmp(op_name, "NamedPriceRatingSumOperation") == 0) {
        success = computeWithFunctor(NamedPriceRatingSumFunctor{});
    } else if (strcmp(op_name, "NamedEcommerceScoreOperation") == 0) {
        success = computeWithFunctor(NamedEcommerceScoreFunctor{});
    } else {
        // 默认使用条件分支内核
        success = computeConditional(0);
    }
    
    if (success) {
        auto end_time = std::chrono::high_resolution_clock::now();
        float compute_time = std::chrono::duration<float, std::milli>(end_time - start_time).count();
        recordComputeTime(compute_time);
        total_operations_++;
    }
    
    return success;
}

// 使用操作ID执行计算（条件分支方式）
bool MixedColumnProcessor::computeConditional(int operation_id) {
    if (columns_.empty()) {
        std::cerr << "MixedColumnProcessor: No columns to process" << std::endl;
        return false;
    }
    
    int num_blocks, block_size;
    kernel_launcher::calculate_launch_config(num_elements_, num_blocks, block_size);
    
    mixed_conditional_kernel<<<num_blocks, block_size, 0, stream_>>>(
        d_column_ptrs_,
        d_column_types_,
        static_cast<int>(columns_.size()),
        num_elements_,
        d_output_,
        operation_id
    );
    
    cudaError_t result = cudaGetLastError();
    if (result != cudaSuccess) {
        std::cerr << "MixedColumnProcessor: Conditional kernel launch failed: " 
                  << cudaGetErrorString(result) << std::endl;
        return false;
    }
    
    total_operations_++;
    return true;
}

// 同步获取结果
std::vector<float> MixedColumnProcessor::getResult() const {
    std::vector<float> result(num_elements_);
    getResult(result.data());
    return result;
}

void MixedColumnProcessor::getResult(float* output) const {
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    CUDA_CHECK(cudaMemcpy(output, d_output_, num_elements_ * sizeof(float), cudaMemcpyDeviceToHost));
}

void MixedColumnProcessor::getResult(std::vector<float>& output) const {
    if (output.size() != num_elements_) {
        output.resize(num_elements_);
    }
    getResult(output.data());
}

// 异步获取结果
void MixedColumnProcessor::getResultAsync(float* output, cudaStream_t user_stream) const {
    cudaStream_t target_stream = user_stream ? user_stream : stream_;
    CUDA_CHECK(cudaMemcpyAsync(output, d_output_, num_elements_ * sizeof(float), 
                              cudaMemcpyDeviceToHost, target_stream));
}

// 同步获取多列结果
std::vector<std::vector<float>> MixedColumnProcessor::getResults() const {
    std::vector<std::vector<float>> results;
    results.reserve(num_outputs_);
    
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    
    for (size_t i = 0; i < num_outputs_; ++i) {
        std::vector<float> result(num_elements_);
        CUDA_CHECK(cudaMemcpy(result.data(), d_outputs_[i], 
                             num_elements_ * sizeof(float), cudaMemcpyDeviceToHost));
        results.push_back(std::move(result));
    }
    
    return results;
}

void MixedColumnProcessor::getResults(std::vector<float*> outputs) const {
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    
    for (size_t i = 0; i < num_outputs_ && i < outputs.size(); ++i) {
        CUDA_CHECK(cudaMemcpy(outputs[i], d_outputs_[i], 
                             num_elements_ * sizeof(float), cudaMemcpyDeviceToHost));
    }
}

// 流同步
void MixedColumnProcessor::synchronize() const {
    CUDA_CHECK(cudaStreamSynchronize(stream_));
}

// 数据验证
bool MixedColumnProcessor::validateData() const {
    for (size_t i = 0; i < columns_.size(); ++i) {
        if (!columns_[i]) {
            std::cerr << "MixedColumnProcessor: Column " << i << " is null" << std::endl;
            return false;
        }
        
        if (columns_[i]->getNumElements() != num_elements_) {
            std::cerr << "MixedColumnProcessor: Column " << i << " size mismatch" << std::endl;
            return false;
        }
    }
    return true;
}

// 重置性能计数器
void MixedColumnProcessor::resetPerformanceCounters() {
    last_compute_time_ms_ = 0.0f;
    total_operations_ = 0;
}

// 打印列信息
void MixedColumnProcessor::printColumnInfo() const {
    std::cout << "=== MixedColumnProcessor Info ===" << std::endl;
    std::cout << "Number of elements: " << num_elements_ << std::endl;
    std::cout << "Number of columns: " << columns_.size() << std::endl;
    
    for (size_t i = 0; i < columns_.size(); ++i) {
        std::cout << "Column " << i << ": ";
        switch (column_types_[i]) {
            case ColumnDataType::FLOAT:
                std::cout << "FLOAT";
                break;
            case ColumnDataType::STRING:
                std::cout << "STRING";
                break;
            case ColumnDataType::INT:
                std::cout << "INT";
                break;
            case ColumnDataType::DOUBLE:
                std::cout << "DOUBLE";
                break;
        }
        std::cout << " (" << columns_[i]->getElementSize() << " bytes per element)" << std::endl;
    }
    
    std::cout << "Total operations: " << total_operations_ << std::endl;
    std::cout << "Last compute time: " << last_compute_time_ms_ << " ms" << std::endl;
    std::cout << "Device memory usage: " << getDeviceMemoryUsage() / (1024*1024) << " MB" << std::endl;
}

// 打印示例数据
void MixedColumnProcessor::printSampleData(int num_samples) const {
    num_samples = std::min(num_samples, static_cast<int>(num_elements_));
    
    std::cout << "=== Sample Data (first " << num_samples << " rows) ===" << std::endl;
    
    for (int row = 0; row < num_samples; ++row) {
        std::cout << "Row " << row << ": ";
        
        for (size_t col = 0; col < columns_.size(); ++col) {
            if (col > 0) std::cout << ", ";
            
            switch (column_types_[col]) {
                case ColumnDataType::FLOAT:
                    {
                        auto data = static_cast<FloatColumn*>(columns_[col].get())->getData();
                        std::cout << std::fixed << std::setprecision(2) << data[row];
                    }
                    break;
                    
                case ColumnDataType::STRING:
                    {
                        auto data = static_cast<StringColumn*>(columns_[col].get())->getData();
                        std::cout << "\"" << data[row] << "\"";
                    }
                    break;
                    
                default:
                    std::cout << "?";
                    break;
            }
        }
        std::cout << std::endl;
    }
}

// 获取设备内存使用情况
size_t MixedColumnProcessor::getDeviceMemoryUsage() const {
    size_t total_usage = 0;
    
    // 输出内存
    total_usage += num_elements_ * sizeof(float);
    
    // 各列数据内存
    for (const auto& column : columns_) {
        total_usage += column->getNumElements() * column->getElementSize();
    }
    
    // 元数据内存
    total_usage += columns_.size() * sizeof(void*); // 列指针数组
    total_usage += columns_.size() * sizeof(ColumnDataType); // 类型数组
    
    return total_usage;
}

// === 私有方法实现 ===

void MixedColumnProcessor::updateDeviceMetadata() {
    freeDeviceMetadata();
    allocateDeviceMetadata();
    
    if (columns_.empty()) return;
    
    // 准备主机端数据
    std::vector<void*> host_column_ptrs(columns_.size());
    std::vector<ColumnDataType> host_column_types(columns_.size());
    std::vector<uint32_t> host_name_hashes(columns_.size());
    
    for (size_t i = 0; i < columns_.size(); ++i) {
        host_column_ptrs[i] = columns_[i]->getDevicePointer();
        host_column_types[i] = column_types_[i];
        uint32_t h = 0u;
        if (i < column_name_hashes_.size()) h = column_name_hashes_[i];
        host_name_hashes[i] = h;
    }
    
    // 复制到设备
    CUDA_CHECK(cudaMemcpy(d_column_ptrs_, host_column_ptrs.data(), 
                         columns_.size() * sizeof(void*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_column_types_, host_column_types.data(), 
                         columns_.size() * sizeof(ColumnDataType), cudaMemcpyHostToDevice));
    if (d_column_name_hashes_ == nullptr) {
        CUDA_CHECK(cudaMalloc(&d_column_name_hashes_, columns_.size() * sizeof(uint32_t)));
    }
    CUDA_CHECK(cudaMemcpy(d_column_name_hashes_, host_name_hashes.data(),
                         columns_.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
}

void MixedColumnProcessor::cleanupDeviceMemory() {
    freeDeviceMetadata();
    
    if (d_output_) {
        cudaFree(d_output_);
        d_output_ = nullptr;
    }
    
    // 清理多输出缓冲区
    for (auto& d_out : d_outputs_) {
        if (d_out) {
            cudaFree(d_out);
            d_out = nullptr;
        }
    }
    d_outputs_.clear();
    num_outputs_ = 0;
    
    if (d_column_name_hashes_) {
        cudaFree(d_column_name_hashes_);
        d_column_name_hashes_ = nullptr;
    }
}

bool MixedColumnProcessor::validateColumnIndex(int column_index) const {
    if (column_index < 0 || static_cast<size_t>(column_index) >= columns_.size()) {
        std::cerr << "MixedColumnProcessor: Invalid column index " << column_index << std::endl;
        return false;
    }
    return true;
}

void MixedColumnProcessor::recordComputeTime(float time_ms) const {
    last_compute_time_ms_ = time_ms;
}

bool MixedColumnProcessor::allocateDeviceMetadata() {
    if (d_column_ptrs_ == nullptr && !columns_.empty()) {
        CUDA_CHECK(cudaMalloc(&d_column_ptrs_, columns_.size() * sizeof(void*)));
    }
    if (d_column_types_ == nullptr && !columns_.empty()) {
        CUDA_CHECK(cudaMalloc(&d_column_types_, columns_.size() * sizeof(ColumnDataType)));
    }
    if (d_column_name_hashes_ == nullptr && !columns_.empty()) {
        CUDA_CHECK(cudaMalloc(&d_column_name_hashes_, columns_.size() * sizeof(uint32_t)));
    }
    return true;
}

void MixedColumnProcessor::freeDeviceMetadata() {
    if (d_column_ptrs_) {
        cudaFree(d_column_ptrs_);
        d_column_ptrs_ = nullptr;
    }
    if (d_column_types_) {
        cudaFree(d_column_types_);
        d_column_types_ = nullptr;
    }
    if (d_column_name_hashes_) {
        cudaFree(d_column_name_hashes_);
        d_column_name_hashes_ = nullptr;
    }
}

// === 全局函数实现 ===

std::unique_ptr<MixedColumnProcessor> createMixedProcessor(size_t num_elements) {
    return std::make_unique<MixedColumnProcessor>(num_elements);
}

// === 基准测试实现 ===

MixedProcessorBenchmark::BenchmarkResult MixedProcessorBenchmark::benchmark(
    MixedColumnProcessor& processor,
    MixedOperationFactory::OperationType operation_type,
    int num_iterations
) {
    BenchmarkResult result = {};
    
    std::vector<float> compute_times;
    compute_times.reserve(num_iterations);
    
    // 预热
    for (int i = 0; i < 3; ++i) {
        processor.compute(operation_type);
        processor.synchronize();
    }
    processor.resetPerformanceCounters();
    
    // 基准测试
    auto total_start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_iterations; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        
        if (!processor.compute(operation_type)) {
            std::cerr << "Benchmark: Compute failed at iteration " << i << std::endl;
            break;
        }
        
        processor.synchronize();
        auto end = std::chrono::high_resolution_clock::now();
        
        float time_ms = std::chrono::duration<float, std::milli>(end - start).count();
        compute_times.push_back(time_ms);
    }
    
    auto total_end = std::chrono::high_resolution_clock::now();
    
    if (!compute_times.empty()) {
        result.avg_compute_time_ms = std::accumulate(compute_times.begin(), compute_times.end(), 0.0f) / compute_times.size();
        result.min_compute_time_ms = *std::min_element(compute_times.begin(), compute_times.end());
        result.max_compute_time_ms = *std::max_element(compute_times.begin(), compute_times.end());
        
        float total_time_sec = std::chrono::duration<float>(total_end - total_start).count();
        result.throughput_ops_per_sec = static_cast<float>(num_iterations) / total_time_sec;
        
        result.total_operations = processor.getTotalOperations();
        result.memory_usage_bytes = processor.getDeviceMemoryUsage();
    }
    
    return result;
}

void MixedProcessorBenchmark::compareBenchmarks(
    const std::vector<BenchmarkResult>& results,
    const std::vector<std::string>& test_names
) {
    std::cout << "\n=== Benchmark Comparison ===" << std::endl;
    std::cout << std::setw(20) << "Test Name"
              << std::setw(15) << "Avg Time (ms)"
              << std::setw(15) << "Min Time (ms)"
              << std::setw(15) << "Max Time (ms)"
              << std::setw(15) << "Throughput (ops/s)"
              << std::setw(15) << "Memory (MB)" << std::endl;
    std::cout << std::string(95, '-') << std::endl;
    
    for (size_t i = 0; i < results.size() && i < test_names.size(); ++i) {
        const auto& result = results[i];
        std::cout << std::setw(20) << test_names[i]
                  << std::setw(15) << std::fixed << std::setprecision(3) << result.avg_compute_time_ms
                  << std::setw(15) << std::fixed << std::setprecision(3) << result.min_compute_time_ms
                  << std::setw(15) << std::fixed << std::setprecision(3) << result.max_compute_time_ms
                  << std::setw(15) << std::fixed << std::setprecision(1) << result.throughput_ops_per_sec
                  << std::setw(15) << std::fixed << std::setprecision(2) << (result.memory_usage_bytes / (1024.0f * 1024.0f))
                  << std::endl;
    }
}

// === 分组聚合实现 ===

MixedColumnProcessor::GroupByResult MixedColumnProcessor::groupBySum(int key_column_index) {
    return groupByAggregate(key_column_index, AggregationType::SUM);
}

MixedColumnProcessor::GroupByResult MixedColumnProcessor::groupByAggregate(
    int key_column_index, 
    AggregationType agg_type,
    int value_column_index,
    DataSource value_source
) {
    GroupByResult result;
    result.num_groups = 0;
    result.is_multi_key = false;
    
    // 验证列索引和类型
    if (!validateColumnIndex(key_column_index)) {
        std::cerr << "MixedColumnProcessor: Invalid key column index" << std::endl;
        return result;
    }
    
    if (column_types_[key_column_index] != ColumnDataType::INT) {
        std::cerr << "MixedColumnProcessor: Key column must be INT type" << std::endl;
        return result;
    }
    
    // 获取key列指针
    int* d_keys_in = static_cast<int*>(columns_[key_column_index]->getDevicePointer());
    
    // 确定待聚合的值数据源
    float* d_values_in = nullptr;
    
    if (value_column_index == -1) {
        // 使用默认的d_output_
        if (!d_output_) {
            std::cerr << "MixedColumnProcessor: No compute results available" << std::endl;
            return result;
        }
        d_values_in = d_output_;
    } else {
        // 使用指定的列
        if (value_source == DataSource::INPUT_COLUMN) {
            // 从输入列中选择
            if (!validateColumnIndex(value_column_index)) {
                std::cerr << "MixedColumnProcessor: Invalid value column index" << std::endl;
                return result;
            }
            
            // 检查列类型是否为FLOAT或INT
            if (column_types_[value_column_index] == ColumnDataType::FLOAT) {
                d_values_in = static_cast<float*>(columns_[value_column_index]->getDevicePointer());
            } else if (column_types_[value_column_index] == ColumnDataType::INT) {
                // 需要将INT转换为FLOAT
                int* d_int_values = static_cast<int*>(columns_[value_column_index]->getDevicePointer());
                CUDA_CHECK(cudaMalloc(&d_values_in, num_elements_ * sizeof(float)));
                
                // 使用简单的转换kernel
                int num_blocks, block_size;
                kernel_launcher::calculate_launch_config(num_elements_, num_blocks, block_size);
                
                // Lambda转换kernel
                auto convert_kernel = [] __device__ (int val) -> float { return static_cast<float>(val); };
                
                // 简单的类型转换（在GPU上）
                thrust::device_ptr<int> d_int_ptr(d_int_values);
                thrust::device_ptr<float> d_float_ptr(d_values_in);
                thrust::transform(d_int_ptr, d_int_ptr + num_elements_, d_float_ptr, 
                                thrust::identity<int>());
            } else {
                std::cerr << "MixedColumnProcessor: Value column must be FLOAT or INT type" << std::endl;
                return result;
            }
        } else {
            // 从operator结果中选择
            if (value_column_index < 0 || static_cast<size_t>(value_column_index) >= num_outputs_) {
                std::cerr << "MixedColumnProcessor: Invalid operator result index" << std::endl;
                return result;
            }
            d_values_in = d_outputs_[value_column_index];
        }
    }
    
    // 分配临时内存用于排序后的keys和values
    int* d_keys_out = nullptr;
    float* d_values_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_keys_out, num_elements_ * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_values_out, num_elements_ * sizeof(float)));
    
    // 第一步：根据keys对keys和values进行排序
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    
    // 确定临时存储大小
    cub::DeviceRadixSort::SortPairs(
        d_temp_storage, temp_storage_bytes,
        d_keys_in, d_keys_out,
        d_values_in, d_values_out,
        num_elements_,
        0, sizeof(int) * 8,
        stream_
    );
    
    // 分配临时存储
    CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    
    // 执行排序
    cub::DeviceRadixSort::SortPairs(
        d_temp_storage, temp_storage_bytes,
        d_keys_in, d_keys_out,
        d_values_in, d_values_out,
        num_elements_,
        0, sizeof(int) * 8,
        stream_
    );
    
    // 释放排序临时存储
    CUDA_CHECK(cudaFree(d_temp_storage));
    d_temp_storage = nullptr;
    
    // 第二步：使用ReduceByKey进行分组聚合
    int* d_unique_keys = nullptr;
    float* d_aggregated_values = nullptr;
    int* d_num_runs = nullptr;
    
    CUDA_CHECK(cudaMalloc(&d_unique_keys, num_elements_ * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_aggregated_values, num_elements_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_num_runs, sizeof(int)));
    
    // 确定ReduceByKey临时存储大小
    temp_storage_bytes = 0;
    
    if (agg_type == AggregationType::SUM) {
        cub::DeviceReduce::ReduceByKey(
            d_temp_storage, temp_storage_bytes,
            d_keys_out, d_unique_keys,
            d_values_out, d_aggregated_values,
            d_num_runs,
            cub::Sum(),
            num_elements_,
            stream_
        );
    } else if (agg_type == AggregationType::MAX) {
        cub::DeviceReduce::ReduceByKey(
            d_temp_storage, temp_storage_bytes,
            d_keys_out, d_unique_keys,
            d_values_out, d_aggregated_values,
            d_num_runs,
            cub::Max(),
            num_elements_,
            stream_
        );
    } else if (agg_type == AggregationType::MIN) {
        cub::DeviceReduce::ReduceByKey(
            d_temp_storage, temp_storage_bytes,
            d_keys_out, d_unique_keys,
            d_values_out, d_aggregated_values,
            d_num_runs,
            cub::Min(),
            num_elements_,
            stream_
        );
    } else {
        // 默认使用SUM
        cub::DeviceReduce::ReduceByKey(
            d_temp_storage, temp_storage_bytes,
            d_keys_out, d_unique_keys,
            d_values_out, d_aggregated_values,
            d_num_runs,
            cub::Sum(),
            num_elements_,
            stream_
        );
    }
    
    // 分配临时存储
    CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    
    // 执行ReduceByKey
    if (agg_type == AggregationType::SUM) {
        cub::DeviceReduce::ReduceByKey(
            d_temp_storage, temp_storage_bytes,
            d_keys_out, d_unique_keys,
            d_values_out, d_aggregated_values,
            d_num_runs,
            cub::Sum(),
            num_elements_,
            stream_
        );
    } else if (agg_type == AggregationType::MAX) {
        cub::DeviceReduce::ReduceByKey(
            d_temp_storage, temp_storage_bytes,
            d_keys_out, d_unique_keys,
            d_values_out, d_aggregated_values,
            d_num_runs,
            cub::Max(),
            num_elements_,
            stream_
        );
    } else if (agg_type == AggregationType::MIN) {
        cub::DeviceReduce::ReduceByKey(
            d_temp_storage, temp_storage_bytes,
            d_keys_out, d_unique_keys,
            d_values_out, d_aggregated_values,
            d_num_runs,
            cub::Min(),
            num_elements_,
            stream_
        );
    } else {
        cub::DeviceReduce::ReduceByKey(
            d_temp_storage, temp_storage_bytes,
            d_keys_out, d_unique_keys,
            d_values_out, d_aggregated_values,
            d_num_runs,
            cub::Sum(),
            num_elements_,
            stream_
        );
    }
    
    // 同步
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    
    // 获取分组数量
    int num_groups_host;
    CUDA_CHECK(cudaMemcpy(&num_groups_host, d_num_runs, sizeof(int), cudaMemcpyDeviceToHost));
    result.num_groups = static_cast<size_t>(num_groups_host);
    
    // 复制结果到主机
    result.unique_keys.resize(result.num_groups);
    result.aggregated_values.resize(result.num_groups);
    
    CUDA_CHECK(cudaMemcpy(result.unique_keys.data(), d_unique_keys, 
                         result.num_groups * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(result.aggregated_values.data(), d_aggregated_values, 
                         result.num_groups * sizeof(float), cudaMemcpyDeviceToHost));
    
    // 清理临时内存
    CUDA_CHECK(cudaFree(d_temp_storage));
    CUDA_CHECK(cudaFree(d_keys_out));
    CUDA_CHECK(cudaFree(d_values_out));
    CUDA_CHECK(cudaFree(d_unique_keys));
    CUDA_CHECK(cudaFree(d_aggregated_values));
    CUDA_CHECK(cudaFree(d_num_runs));
    
    // 如果从INT列转换，需要释放临时分配的内存
    if (value_column_index != -1 && value_source == DataSource::INPUT_COLUMN &&
        column_types_[value_column_index] == ColumnDataType::INT) {
        CUDA_CHECK(cudaFree(d_values_in));
    }
    
    return result;
}

// 多键分组聚合
MixedColumnProcessor::GroupByResult MixedColumnProcessor::groupByAggregateMultiKey(
    const std::vector<int>& key_column_indices,
    AggregationType agg_type,
    int value_column_index,
    DataSource value_source
) {
    GroupByResult result;
    result.num_groups = 0;
    result.is_multi_key = true;
    
    if (key_column_indices.empty()) {
        std::cerr << "MixedColumnProcessor: No key columns specified" << std::endl;
        return result;
    }
    
    // 验证所有key列
    for (int key_idx : key_column_indices) {
        if (!validateColumnIndex(key_idx)) {
            std::cerr << "MixedColumnProcessor: Invalid key column index: " << key_idx << std::endl;
            return result;
        }
        if (column_types_[key_idx] != ColumnDataType::INT) {
            std::cerr << "MixedColumnProcessor: All key columns must be INT type" << std::endl;
            return result;
        }
    }
    
    // 确定待聚合的值数据源（与单键版本相同的逻辑）
    float* d_values_in = nullptr;
    bool need_free_values = false;
    
    if (value_column_index == -1) {
        if (!d_output_) {
            std::cerr << "MixedColumnProcessor: No compute results available" << std::endl;
            return result;
        }
        d_values_in = d_output_;
    } else {
        if (value_source == DataSource::INPUT_COLUMN) {
            if (!validateColumnIndex(value_column_index)) {
                std::cerr << "MixedColumnProcessor: Invalid value column index" << std::endl;
                return result;
            }
            
            if (column_types_[value_column_index] == ColumnDataType::FLOAT) {
                d_values_in = static_cast<float*>(columns_[value_column_index]->getDevicePointer());
            } else if (column_types_[value_column_index] == ColumnDataType::INT) {
                int* d_int_values = static_cast<int*>(columns_[value_column_index]->getDevicePointer());
                CUDA_CHECK(cudaMalloc(&d_values_in, num_elements_ * sizeof(float)));
                need_free_values = true;
                
                thrust::device_ptr<int> d_int_ptr(d_int_values);
                thrust::device_ptr<float> d_float_ptr(d_values_in);
                thrust::transform(d_int_ptr, d_int_ptr + num_elements_, d_float_ptr, 
                                thrust::identity<int>());
            } else {
                std::cerr << "MixedColumnProcessor: Value column must be FLOAT or INT type" << std::endl;
                return result;
            }
        } else {
            if (value_column_index < 0 || static_cast<size_t>(value_column_index) >= num_outputs_) {
                std::cerr << "MixedColumnProcessor: Invalid operator result index" << std::endl;
                return result;
            }
            d_values_in = d_outputs_[value_column_index];
        }
    }
    
    // 创建组合键：使用简单的哈希方法组合多个键
    // 对于每一行，计算 hash = key[0] + key[1] * 10000 + key[2] * 10000^2 + ...
    // 这种方法假设每个键的范围不太大（< 10000）
    long long* d_combined_keys = nullptr;
    CUDA_CHECK(cudaMalloc(&d_combined_keys, num_elements_ * sizeof(long long)));
    
    // 启动kernel组合多个键
    int num_blocks, block_size;
    kernel_launcher::calculate_launch_config(num_elements_, num_blocks, block_size);
    
    // 准备设备端的键列指针数组
    int** d_key_ptrs = nullptr;
    std::vector<int*> host_key_ptrs(key_column_indices.size());
    for (size_t i = 0; i < key_column_indices.size(); ++i) {
        host_key_ptrs[i] = static_cast<int*>(columns_[key_column_indices[i]]->getDevicePointer());
    }
    CUDA_CHECK(cudaMalloc(&d_key_ptrs, key_column_indices.size() * sizeof(int*)));
    CUDA_CHECK(cudaMemcpy(d_key_ptrs, host_key_ptrs.data(), 
                         key_column_indices.size() * sizeof(int*), cudaMemcpyHostToDevice));
    
    // Lambda kernel来组合键
    auto combine_keys_kernel = [=] __device__ (size_t idx, int** key_ptrs, int num_keys, 
                                               long long* combined_keys, size_t num_elements) {
        for (size_t i = idx; i < num_elements; i += blockDim.x * gridDim.x) {
            long long combined = 0;
            long long multiplier = 1;
            for (int k = 0; k < num_keys; ++k) {
                combined += static_cast<long long>(key_ptrs[k][i]) * multiplier;
                multiplier *= 100000LL;  // 假设每个键的范围 < 100000
            }
            combined_keys[i] = combined;
        }
    };
    
    // 使用简单的kernel来组合键
    // 创建一个临时kernel
    auto kernel = [d_key_ptrs, num_keys = static_cast<int>(key_column_indices.size()), 
                   d_combined_keys, num_elements = num_elements_] 
        __device__ (size_t idx) {
        long long combined = 0;
        long long multiplier = 1;
        for (int k = 0; k < num_keys; ++k) {
            combined += static_cast<long long>(d_key_ptrs[k][idx]) * multiplier;
            multiplier *= 100000LL;
        }
        return combined;
    };
    
    // 使用thrust来组合键
    thrust::device_ptr<long long> d_combined_ptr(d_combined_keys);
    thrust::counting_iterator<size_t> first(0);
    thrust::counting_iterator<size_t> last(num_elements_);
    
    // 手动实现组合逻辑
    // 简化：直接在CPU上创建临时host数据，然后复制到GPU
    std::vector<long long> host_combined_keys(num_elements_);
    std::vector<std::vector<int>> host_keys(key_column_indices.size());
    
    for (size_t i = 0; i < key_column_indices.size(); ++i) {
        host_keys[i].resize(num_elements_);
        auto* int_col = static_cast<IntColumn*>(columns_[key_column_indices[i]].get());
        host_keys[i] = int_col->getData();
    }
    
    for (size_t i = 0; i < num_elements_; ++i) {
        long long combined = 0;
        long long multiplier = 1;
        for (size_t k = 0; k < key_column_indices.size(); ++k) {
            combined += static_cast<long long>(host_keys[k][i]) * multiplier;
            multiplier *= 100000LL;
        }
        host_combined_keys[i] = combined;
    }
    
    CUDA_CHECK(cudaMemcpy(d_combined_keys, host_combined_keys.data(), 
                         num_elements_ * sizeof(long long), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaFree(d_key_ptrs));
    
    // 分配临时内存用于排序后的keys和values
    long long* d_combined_keys_out = nullptr;
    float* d_values_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_combined_keys_out, num_elements_ * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_values_out, num_elements_ * sizeof(float)));
    
    // 根据组合键排序
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    
    cub::DeviceRadixSort::SortPairs(
        d_temp_storage, temp_storage_bytes,
        d_combined_keys, d_combined_keys_out,
        d_values_in, d_values_out,
        num_elements_,
        0, sizeof(long long) * 8,
        stream_
    );
    
    CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    
    cub::DeviceRadixSort::SortPairs(
        d_temp_storage, temp_storage_bytes,
        d_combined_keys, d_combined_keys_out,
        d_values_in, d_values_out,
        num_elements_,
        0, sizeof(long long) * 8,
        stream_
    );
    
    CUDA_CHECK(cudaFree(d_temp_storage));
    d_temp_storage = nullptr;
    
    // ReduceByKey
    long long* d_unique_combined_keys = nullptr;
    float* d_aggregated_values = nullptr;
    int* d_num_runs = nullptr;
    
    CUDA_CHECK(cudaMalloc(&d_unique_combined_keys, num_elements_ * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_aggregated_values, num_elements_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_num_runs, sizeof(int)));
    
    temp_storage_bytes = 0;
    
    if (agg_type == AggregationType::SUM) {
        cub::DeviceReduce::ReduceByKey(
            d_temp_storage, temp_storage_bytes,
            d_combined_keys_out, d_unique_combined_keys,
            d_values_out, d_aggregated_values,
            d_num_runs,
            cub::Sum(),
            num_elements_,
            stream_
        );
    } else if (agg_type == AggregationType::MAX) {
        cub::DeviceReduce::ReduceByKey(
            d_temp_storage, temp_storage_bytes,
            d_combined_keys_out, d_unique_combined_keys,
            d_values_out, d_aggregated_values,
            d_num_runs,
            cub::Max(),
            num_elements_,
            stream_
        );
    } else if (agg_type == AggregationType::MIN) {
        cub::DeviceReduce::ReduceByKey(
            d_temp_storage, temp_storage_bytes,
            d_combined_keys_out, d_unique_combined_keys,
            d_values_out, d_aggregated_values,
            d_num_runs,
            cub::Min(),
            num_elements_,
            stream_
        );
    } else {
        cub::DeviceReduce::ReduceByKey(
            d_temp_storage, temp_storage_bytes,
            d_combined_keys_out, d_unique_combined_keys,
            d_values_out, d_aggregated_values,
            d_num_runs,
            cub::Sum(),
            num_elements_,
            stream_
        );
    }
    
    CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    
    if (agg_type == AggregationType::SUM) {
        cub::DeviceReduce::ReduceByKey(
            d_temp_storage, temp_storage_bytes,
            d_combined_keys_out, d_unique_combined_keys,
            d_values_out, d_aggregated_values,
            d_num_runs,
            cub::Sum(),
            num_elements_,
            stream_
        );
    } else if (agg_type == AggregationType::MAX) {
        cub::DeviceReduce::ReduceByKey(
            d_temp_storage, temp_storage_bytes,
            d_combined_keys_out, d_unique_combined_keys,
            d_values_out, d_aggregated_values,
            d_num_runs,
            cub::Max(),
            num_elements_,
            stream_
        );
    } else if (agg_type == AggregationType::MIN) {
        cub::DeviceReduce::ReduceByKey(
            d_temp_storage, temp_storage_bytes,
            d_combined_keys_out, d_unique_combined_keys,
            d_values_out, d_aggregated_values,
            d_num_runs,
            cub::Min(),
            num_elements_,
            stream_
        );
    } else {
        cub::DeviceReduce::ReduceByKey(
            d_temp_storage, temp_storage_bytes,
            d_combined_keys_out, d_unique_combined_keys,
            d_values_out, d_aggregated_values,
            d_num_runs,
            cub::Sum(),
            num_elements_,
            stream_
        );
    }
    
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    
    // 获取分组数量
    int num_groups_host;
    CUDA_CHECK(cudaMemcpy(&num_groups_host, d_num_runs, sizeof(int), cudaMemcpyDeviceToHost));
    result.num_groups = static_cast<size_t>(num_groups_host);
    
    // 复制结果到主机
    std::vector<long long> unique_combined_keys(result.num_groups);
    result.aggregated_values.resize(result.num_groups);
    
    CUDA_CHECK(cudaMemcpy(unique_combined_keys.data(), d_unique_combined_keys, 
                         result.num_groups * sizeof(long long), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(result.aggregated_values.data(), d_aggregated_values, 
                         result.num_groups * sizeof(float), cudaMemcpyDeviceToHost));
    
    // 将组合键解码回多个键
    result.unique_multi_keys.resize(result.num_groups);
    for (size_t i = 0; i < result.num_groups; ++i) {
        long long combined = unique_combined_keys[i];
        std::vector<int> keys(key_column_indices.size());
        for (size_t k = 0; k < key_column_indices.size(); ++k) {
            keys[k] = static_cast<int>(combined % 100000LL);
            combined /= 100000LL;
        }
        result.unique_multi_keys[i] = keys;
    }
    
    // 清理临时内存
    CUDA_CHECK(cudaFree(d_temp_storage));
    CUDA_CHECK(cudaFree(d_combined_keys));
    CUDA_CHECK(cudaFree(d_combined_keys_out));
    CUDA_CHECK(cudaFree(d_values_out));
    CUDA_CHECK(cudaFree(d_unique_combined_keys));
    CUDA_CHECK(cudaFree(d_aggregated_values));
    CUDA_CHECK(cudaFree(d_num_runs));
    
    if (need_free_values) {
        CUDA_CHECK(cudaFree(d_values_in));
    }
    
    return result;
}

} // namespace zmm
