#include "column_processor.cuh"
#include <iostream>
#include <vector>
#include <random>
#include <chrono>

using namespace zmm;

// 自定义业务逻辑：计算三列数据的复合运算 (a + b) * c
struct CustomOperation3 {
    __device__ float operator()(const float (&values)[3]) const {
        return (values[0] + values[1]) * values[2];
    }
};

// 自定义业务逻辑：计算五列数据的加权平均
struct CustomOperation5 {
    __device__ float operator()(const float (&values)[5]) const {
        // 权重为 [0.3, 0.2, 0.2, 0.2, 0.1]
        return values[0] * 0.3f + values[1] * 0.2f + values[2] * 0.2f + 
               values[3] * 0.2f + values[4] * 0.1f;
    }
};

// 生成测试数据
std::vector<float> generateTestData(size_t size, float min_val = 0.0f, float max_val = 100.0f) {
    std::vector<float> data(size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min_val, max_val);
    
    for (size_t i = 0; i < size; ++i) {
        data[i] = dis(gen);
    }
    return data;
}

// 测试性能
template<int N>
void benchmarkProcessor(size_t num_elements) {
    std::cout << "=== 测试 " << N << " 列数据，" << num_elements << " 个元素 ===" << std::endl;
    
    // 创建处理器
    auto processor = createProcessor<N>(num_elements);
    
    // 生成测试数据
    std::vector<std::vector<float>> input_data(N);
    for (int i = 0; i < N; ++i) {
        input_data[i] = generateTestData(num_elements);
    }
    
    // 设置输入数据
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; ++i) {
        processor->setInputColumn(i, input_data[i].data());
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto data_transfer_time = std::chrono::duration<double, std::milli>(end - start).count();
    
    // 执行计算
    start = std::chrono::high_resolution_clock::now();
    processor->compute(AddOperation{});
    processor->synchronize();
    end = std::chrono::high_resolution_clock::now();
    auto compute_time = std::chrono::duration<double, std::milli>(end - start).count();
    
    // 获取结果
    std::vector<float> result(num_elements);
    start = std::chrono::high_resolution_clock::now();
    processor->getResult(result.data());
    end = std::chrono::high_resolution_clock::now();
    auto result_transfer_time = std::chrono::duration<double, std::milli>(end - start).count();
    
    // 验证结果（检查前几个元素）
    std::cout << "验证结果（前5个元素）:" << std::endl;
    for (int i = 0; i < std::min(5, static_cast<int>(num_elements)); ++i) {
        float expected = 0.0f;
        for (int j = 0; j < N; ++j) {
            expected += input_data[j][i];
        }
        std::cout << "  [" << i << "] GPU结果: " << result[i] 
                  << ", 期望值: " << expected 
                  << ", 差异: " << std::abs(result[i] - expected) << std::endl;
    }
    
    // 性能统计
    double total_time = data_transfer_time + compute_time + result_transfer_time;
    double throughput = (static_cast<double>(num_elements) * N) / (compute_time / 1000.0) / 1e9; // GFLOPS
    
    std::cout << "性能统计:" << std::endl;
    std::cout << "  数据传输时间: " << data_transfer_time << " ms" << std::endl;
    std::cout << "  计算时间: " << compute_time << " ms" << std::endl;
    std::cout << "  结果传输时间: " << result_transfer_time << " ms" << std::endl;
    std::cout << "  总时间: " << total_time << " ms" << std::endl;
    std::cout << "  计算吞吐量: " << throughput << " GFLOPS" << std::endl;
    std::cout << std::endl;
}

int main() {
    std::cout << "ZMM CUDA列数据处理框架示例程序" << std::endl;
    std::cout << "================================" << std::endl;
    
    // 检查CUDA设备
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        std::cerr << "没有找到CUDA设备！" << std::endl;
        return -1;
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "使用GPU: " << prop.name << std::endl;
    std::cout << "全局内存: " << prop.totalGlobalMem / (1024*1024*1024) << " GB" << std::endl;
    std::cout << std::endl;
    
    // 测试不同的数据规模
    std::vector<size_t> test_sizes = {1000000, 10000000, 100000000}; // 100万, 1千万, 1亿
    
    for (size_t size : test_sizes) {
        benchmarkProcessor<3>(size);   // 3列测试
        benchmarkProcessor<5>(size);   // 5列测试
        benchmarkProcessor<10>(size);  // 10列测试
    }
    
    // 演示自定义业务逻辑
    std::cout << "=== 自定义业务逻辑演示 ===" << std::endl;
    
    // 3列自定义运算示例
    {
        const size_t num_elements = 1000000;
        auto processor = createProcessor<3>(num_elements);
        
        // 创建测试数据
        auto data1 = generateTestData(num_elements, 1.0f, 10.0f);
        auto data2 = generateTestData(num_elements, 1.0f, 10.0f);
        auto data3 = generateTestData(num_elements, 1.0f, 5.0f);
        
        processor->setInputColumn(0, data1.data());
        processor->setInputColumn(1, data2.data());
        processor->setInputColumn(2, data3.data());
        
        // 使用自定义运算: (a + b) * c
        processor->compute(CustomOperation3{});
        
        std::vector<float> result(num_elements);
        processor->getResult(result.data());
        
        std::cout << "3列自定义运算 (a + b) * c 验证:" << std::endl;
        for (int i = 0; i < 3; ++i) {
            float expected = (data1[i] + data2[i]) * data3[i];
            std::cout << "  [" << i << "] GPU: " << result[i] 
                      << ", 期望: " << expected 
                      << ", 差异: " << std::abs(result[i] - expected) << std::endl;
        }
    }
    
    // 5列加权平均示例
    {
        const size_t num_elements = 1000000;
        auto processor = createProcessor<5>(num_elements);
        
        // 创建测试数据
        std::vector<std::vector<float>> data(5);
        for (int i = 0; i < 5; ++i) {
            data[i] = generateTestData(num_elements, 0.0f, 100.0f);
            processor->setInputColumn(i, data[i].data());
        }
        
        // 使用加权平均运算
        processor->compute(CustomOperation5{});
        
        std::vector<float> result(num_elements);
        processor->getResult(result.data());
        
        std::cout << "5列加权平均验证:" << std::endl;
        for (int i = 0; i < 3; ++i) {
            float expected = data[0][i] * 0.3f + data[1][i] * 0.2f + data[2][i] * 0.2f + 
                           data[3][i] * 0.2f + data[4][i] * 0.1f;
            std::cout << "  [" << i << "] GPU: " << result[i] 
                      << ", 期望: " << expected 
                      << ", 差异: " << std::abs(result[i] - expected) << std::endl;
        }
    }
    
    // 使用预定义的乘法操作
    {
        const size_t num_elements = 1000000;
        auto processor = createProcessor<4>(num_elements);
        
        // 创建测试数据
        std::vector<std::vector<float>> data(4);
        for (int i = 0; i < 4; ++i) {
            data[i] = generateTestData(num_elements, 1.0f, 3.0f);
            processor->setInputColumn(i, data[i].data());
        }
        
        // 使用乘法运算
        processor->compute(MultiplyOperation{});
        
        std::vector<float> result(num_elements);
        processor->getResult(result.data());
        
        std::cout << "4列乘法运算验证:" << std::endl;
        for (int i = 0; i < 3; ++i) {
            float expected = data[0][i] * data[1][i] * data[2][i] * data[3][i];
            std::cout << "  [" << i << "] GPU: " << result[i] 
                      << ", 期望: " << expected 
                      << ", 差异: " << std::abs(result[i] - expected) << std::endl;
        }
    }
    
    std::cout << "测试完成！" << std::endl;
    return 0;
} 