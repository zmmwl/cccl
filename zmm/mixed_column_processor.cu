#include "mixed_column_processor.cuh"
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

} // namespace zmm
