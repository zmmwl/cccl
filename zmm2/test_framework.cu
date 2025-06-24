#include "column_compute_framework.cuh"
#include "business_expressions.cuh"
#include "batch_processor.cuh"
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cassert>

using namespace zmm2;

/**
 * 生成测试数据
 */
template<int N>
void generateTestData(std::vector<std::vector<float>>& columns, size_t num_elements) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-10.0f, 10.0f);
    
    columns.resize(N);
    for (int i = 0; i < N; ++i) {
        columns[i].resize(num_elements);
        for (size_t j = 0; j < num_elements; ++j) {
            columns[i][j] = dis(gen);
        }
    }
}

/**
 * 验证计算结果
 */
template<int N, typename ExpressionType>
bool verifyResults(
    const std::vector<std::vector<float>>& input_columns,
    const std::vector<float>& gpu_result,
    const ExpressionType& expression,
    size_t sample_size = 1000
) {
    size_t num_elements = input_columns[0].size();
    size_t stride = std::max(size_t(1), num_elements / sample_size);
    
    float tolerance = 1e-5f;
    
    for (size_t i = 0; i < num_elements; i += stride) {
        float values[N];
        for (int col = 0; col < N; ++col) {
            values[col] = input_columns[col][i];
        }
        
        // 在CPU上计算期望结果（需要模拟device函数）
        float expected = 0.0f;
        if constexpr (std::is_same_v<ExpressionType, ThreeColumnSum>) {
            expected = values[0] + values[1] + values[2];
        } else if constexpr (std::is_same_v<ExpressionType, TwoColumnMax>) {
            expected = std::max(values[0], values[1]);
        }
        // 可以添加更多表达式的验证
        
        float actual = gpu_result[i];
        if (std::abs(expected - actual) > tolerance) {
            std::cout << "验证失败: 位置 " << i << ", 期望 " << expected 
                     << ", 实际 " << actual << std::endl;
            return false;
        }
    }
    
    return true;
}

/**
 * 性能测试函数
 */
template<int N, typename ExpressionType>
void performanceTest(const std::string& test_name, size_t num_elements, 
                    const ExpressionType& expression) {
    std::cout << "\n=== " << test_name << " ===" << std::endl;
    std::cout << "列数: " << N << ", 元素数量: " << num_elements << std::endl;
    
    // 生成测试数据
    std::vector<std::vector<float>> input_columns;
    generateTestData<N>(input_columns, num_elements);
    
    // 准备输入指针
    std::vector<const float*> input_ptrs(N);
    for (int i = 0; i < N; ++i) {
        input_ptrs[i] = input_columns[i].data();
    }
    
    // 分配输出缓冲区
    std::vector<float> output_column(num_elements);
    
    // 测试计算性能
    auto start_time = std::chrono::high_resolution_clock::now();
    
    try {
        // 根据数据大小选择处理方式
        size_t memory_needed_mb = (num_elements * (N + 1) * sizeof(float)) / (1024 * 1024);
        
        if (memory_needed_mb > 2048) { // 超过2GB使用批处理
            std::cout << "使用批处理模式..." << std::endl;
            BatchProcessor<N> processor(num_elements, 4096);
            std::cout << "批处理大小: " << processor.getBatchSize() << std::endl;
            std::cout << "批次数量: " << processor.getBatchCount() << std::endl;
            
            processor.processBatches(input_ptrs, output_column.data(), expression,
                [](size_t processed, size_t total) {
                    if (processed % (total / 10) == 0 || processed == total) {
                        std::cout << "进度: " << (processed * 100 / total) << "%" << std::endl;
                    }
                }
            );
        } else {
            std::cout << "使用单次处理模式..." << std::endl;
            ColumnComputeFramework<N> framework(num_elements);
            framework.setInputData(input_ptrs);
            framework.compute(expression);
            framework.getResult(output_column.data());
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        std::cout << "计算完成!" << std::endl;
        std::cout << "用时: " << duration.count() << " 毫秒" << std::endl;
        
        // 计算吞吐量
        double throughput = (double)num_elements / duration.count() * 1000.0;
        std::cout << "吞吐量: " << throughput << " 元素/秒" << std::endl;
        
        // 验证部分结果
        std::cout << "前5个结果: ";
        for (int i = 0; i < std::min(5, (int)num_elements); ++i) {
            std::cout << output_column[i] << " ";
        }
        std::cout << std::endl;
        
        // 查询GPU内存使用情况
        size_t free_mem, total_mem;
        MemoryInfo::queryGPUMemory(free_mem, total_mem);
        std::cout << "GPU内存: " << (total_mem - free_mem) / (1024*1024) << "MB / " 
                 << total_mem / (1024*1024) << "MB" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "计算失败: " << e.what() << std::endl;
    }
}

/**
 * 主测试函数
 */
int main() {
    std::cout << "ZMM2 列数据计算框架测试程序" << std::endl;
    std::cout << "=============================" << std::endl;
    
    // 检查CUDA设备
    int device_count;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        std::cout << "错误: 未找到CUDA设备!" << std::endl;
        return -1;
    }
    
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, 0);
    std::cout << "CUDA设备: " << device_prop.name << std::endl;
    std::cout << "全局内存: " << device_prop.totalGlobalMem / (1024*1024) << " MB" << std::endl;
    std::cout << "计算能力: " << device_prop.major << "." << device_prop.minor << std::endl;
    
    // 测试1: 小数据量，三列相加
    {
        ThreeColumnSum expression;
        performanceTest<3>("小数据量 - 三列相加", 1000000, expression);
    }
    
    // 测试2: 中等数据量，两列最大值
    {
        TwoColumnMax expression;
        performanceTest<2>("中等数据量 - 两列最大值", 50000000, expression);
    }
    
    // 测试3: 大数据量，三列加权和
    {
        ThreeColumnWeightedSum expression(0.3f, 0.5f, 0.2f);
        performanceTest<3>("大数据量 - 三列加权和", 200000000, expression);
    }
    
    // 测试4: 超大数据量，五列平均值
    {
        FiveColumnAverage expression;
        performanceTest<5>("超大数据量 - 五列平均值", 500000000, expression);
    }
    
    // 测试5: 复杂数学表达式
    {
        ComplexMathExpression expression(1e-6f);
        performanceTest<3>("复杂数学表达式", 100000000, expression);
    }
    
    // 测试6: 条件表达式  
    {
        ConditionalExpression expression(0.0f, 2.0f);
        performanceTest<3>("条件表达式", 80000000, expression);
    }
    
    // 测试7: 统计分析表达式
    {
        StatisticalExpression expression(0.5f);
        performanceTest<4>("统计分析表达式", 60000000, expression);
    }
    
    // 测试8: 极限情况 - 10亿元素（如果内存足够）
    size_t free_mem, total_mem;
    MemoryInfo::queryGPUMemory(free_mem, total_mem);
    size_t max_elements = free_mem / (3 * sizeof(float)) / 4; // 保守估计
    
    if (max_elements > 1000000000) { // 如果可以处理10亿+元素
        ThreeColumnSum expression;
        performanceTest<3>("极限测试 - 10亿+元素", 1000000000, expression);
    } else {
        std::cout << "\n跳过极限测试 - GPU内存不足以处理10亿元素" << std::endl;
        std::cout << "当前可处理的最大元素数: " << max_elements << std::endl;
    }
    
    std::cout << "\n所有测试完成!" << std::endl;
    return 0;
} 