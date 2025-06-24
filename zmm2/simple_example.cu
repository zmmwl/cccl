#include "column_compute_framework.cuh"
#include "business_expressions.cuh"
#include "batch_processor.cuh"
#include <iostream>
#include <vector>
#include <chrono>

using namespace zmm2;

int main() {
    std::cout << "ZMM2 框架简单使用示例" << std::endl;
    std::cout << "====================" << std::endl;
    
    // 示例1: 三列数据相加
    {
        std::cout << "\n示例1: 三列数据相加" << std::endl;
        
        // 准备测试数据
        const size_t num_elements = 10;
        std::vector<float> col1 = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f};
        std::vector<float> col2 = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f};
        std::vector<float> col3 = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f, 90.0f, 100.0f};
        
        // 准备输入指针
        std::vector<const float*> input_columns = {col1.data(), col2.data(), col3.data()};
        
        // 分配输出缓冲区
        std::vector<float> output(num_elements);
        
        // 创建计算框架和表达式
        ColumnComputeFramework<3> framework(num_elements);
        ThreeColumnSum expression;
        
        // 执行计算
        framework.setInputData(input_columns);
        framework.compute(expression);
        framework.getResult(output.data());
        
        // 输出结果
        std::cout << "输入数据:" << std::endl;
        for (size_t i = 0; i < num_elements; ++i) {
            std::cout << "行 " << i << ": " << col1[i] << " + " << col2[i] << " + " << col3[i] 
                     << " = " << output[i] << std::endl;
        }
    }
    
    // 示例2: 两列数据求最大值
    {
        std::cout << "\n示例2: 两列数据求最大值" << std::endl;
        
        // 准备测试数据
        const size_t num_elements = 8;
        std::vector<float> col1 = {1.5f, 3.2f, 2.1f, 5.8f, 4.3f, 6.7f, 3.9f, 7.2f};
        std::vector<float> col2 = {2.1f, 2.9f, 3.5f, 4.2f, 5.1f, 5.8f, 6.3f, 6.9f};
        
        // 准备输入指针
        std::vector<const float*> input_columns = {col1.data(), col2.data()};
        
        // 分配输出缓冲区
        std::vector<float> output(num_elements);
        
        // 创建计算框架和表达式
        ColumnComputeFramework<2> framework(num_elements);
        TwoColumnMax expression;
        
        // 执行计算
        framework.setInputData(input_columns);
        framework.compute(expression);
        framework.getResult(output.data());
        
        // 输出结果
        std::cout << "输入数据:" << std::endl;
        for (size_t i = 0; i < num_elements; ++i) {
            std::cout << "行 " << i << ": max(" << col1[i] << ", " << col2[i] 
                     << ") = " << output[i] << std::endl;
        }
    }
    
    // 示例3: 三列加权和
    {
        std::cout << "\n示例3: 三列加权和" << std::endl;
        
        // 准备测试数据
        const size_t num_elements = 6;
        std::vector<float> col1 = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        std::vector<float> col2 = {0.5f, 1.0f, 1.5f, 2.0f, 2.5f, 3.0f};
        std::vector<float> col3 = {2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f};
        
        // 准备输入指针
        std::vector<const float*> input_columns = {col1.data(), col2.data(), col3.data()};
        
        // 分配输出缓冲区
        std::vector<float> output(num_elements);
        
        // 创建计算框架和表达式（权重为0.5, 0.3, 0.2）
        ColumnComputeFramework<3> framework(num_elements);
        ThreeColumnWeightedSum expression(0.5f, 0.3f, 0.2f);
        
        // 执行计算
        framework.setInputData(input_columns);
        framework.compute(expression);
        framework.getResult(output.data());
        
        // 输出结果
        std::cout << "输入数据 (权重: 0.5, 0.3, 0.2):" << std::endl;
        for (size_t i = 0; i < num_elements; ++i) {
            std::cout << "行 " << i << ": 0.5*" << col1[i] << " + 0.3*" << col2[i] 
                     << " + 0.2*" << col3[i] << " = " << output[i] << std::endl;
        }
    }
    
    // 示例4: 使用便利函数处理中等规模数据
    {
        std::cout << "\n示例4: 使用便利函数处理中等规模数据" << std::endl;
        
        // 准备较大的测试数据
        const size_t num_elements = 1000000; // 100万元素
        std::vector<float> col1(num_elements);
        std::vector<float> col2(num_elements);
        
        // 初始化数据
        for (size_t i = 0; i < num_elements; ++i) {
            col1[i] = static_cast<float>(i % 100) / 10.0f;
            col2[i] = static_cast<float>((i + 50) % 100) / 10.0f;
        }
        
        // 准备输入指针
        std::vector<const float*> input_columns = {col1.data(), col2.data()};
        
        // 分配输出缓冲区
        std::vector<float> output(num_elements);
        
        // 使用便利函数自动处理
        TwoColumnMax expression;
        
        auto start = std::chrono::high_resolution_clock::now();
        
        processColumns<2>(input_columns, output.data(), num_elements, expression, 
                         4096, // 最大GPU内存4GB
                         [](size_t processed, size_t total) {
                             if (processed == total) {
                                 std::cout << "处理完成: " << processed << "/" << total << std::endl;
                             }
                         });
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        std::cout << "处理了 " << num_elements << " 个元素" << std::endl;
        std::cout << "用时: " << duration.count() << " 微秒" << std::endl;
        std::cout << "吞吐量: " << (num_elements * 1000000.0 / duration.count()) << " 元素/秒" << std::endl;
        
        // 显示部分结果
        std::cout << "前10个结果: ";
        for (int i = 0; i < 10; ++i) {
            std::cout << output[i] << " ";
        }
        std::cout << std::endl;
    }
    
    std::cout << "\n所有示例完成!" << std::endl;
    return 0;
} 