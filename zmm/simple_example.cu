#include "column_processor.cuh"
#include <iostream>
#include <vector>

using namespace zmm;

// 简单的自定义业务逻辑：计算 a * b + c
struct SimpleCustomOperation {
    __device__ float operator()(const float (&values)[3]) const {
        return values[0] * values[1] + values[2];
    }
};

int main() {
    std::cout << "简单使用示例" << std::endl;
    
    // 定义数据大小
    const size_t num_elements = 1000;
    
    // 创建3列数据的处理器
    auto processor = createProcessor<3>(num_elements);
    
    // 准备输入数据
    std::vector<float> column1(num_elements);
    std::vector<float> column2(num_elements);
    std::vector<float> column3(num_elements);
    
    // 填充测试数据
    for (size_t i = 0; i < num_elements; ++i) {
        column1[i] = static_cast<float>(i + 1);      // 1, 2, 3, ...
        column2[i] = static_cast<float>((i + 1) * 2); // 2, 4, 6, ...
        column3[i] = static_cast<float>((i + 1) * 3); // 3, 6, 9, ...
    }
    
    // 设置输入数据
    processor->setInputColumn(0, column1.data());
    processor->setInputColumn(1, column2.data());
    processor->setInputColumn(2, column3.data());
    
    // 方法1：使用预定义的加法操作
    std::cout << "\n=== 使用预定义加法操作 ===" << std::endl;
    processor->compute(AddOperation{});
    
    std::vector<float> result_add(num_elements);
    processor->getResult(result_add.data());
    
    // 验证前5个结果
    std::cout << "加法结果（前5个）：" << std::endl;
    for (int i = 0; i < 5; ++i) {
        float expected = column1[i] + column2[i] + column3[i];
        std::cout << "  [" << i << "] " << column1[i] << " + " << column2[i] 
                  << " + " << column3[i] << " = " << result_add[i] 
                  << " (期望: " << expected << ")" << std::endl;
    }
    
    // 方法2：使用自定义业务逻辑 a * b + c
    std::cout << "\n=== 使用自定义业务逻辑 (a * b + c) ===" << std::endl;
    processor->compute(SimpleCustomOperation{});
    
    std::vector<float> result_custom(num_elements);
    processor->getResult(result_custom.data());
    
    // 验证前5个结果
    std::cout << "自定义运算结果（前5个）：" << std::endl;
    for (int i = 0; i < 5; ++i) {
        float expected = column1[i] * column2[i] + column3[i];
        std::cout << "  [" << i << "] " << column1[i] << " * " << column2[i] 
                  << " + " << column3[i] << " = " << result_custom[i] 
                  << " (期望: " << expected << ")" << std::endl;
    }
    
    // 方法3：使用Lambda表达式（需要C++14或更高版本）
    std::cout << "\n=== 使用Lambda表达式 (平均值) ===" << std::endl;
    auto average_op = [] __device__ (const float (&values)[3]) -> float {
        return (values[0] + values[1] + values[2]) / 3.0f;
    };
    
    processor->compute(average_op);
    
    std::vector<float> result_avg(num_elements);
    processor->getResult(result_avg.data());
    
    // 验证前5个结果
    std::cout << "平均值结果（前5个）：" << std::endl;
    for (int i = 0; i < 5; ++i) {
        float expected = (column1[i] + column2[i] + column3[i]) / 3.0f;
        std::cout << "  [" << i << "] (" << column1[i] << " + " << column2[i] 
                  << " + " << column3[i] << ") / 3 = " << result_avg[i] 
                  << " (期望: " << expected << ")" << std::endl;
    }
    
    std::cout << "\n示例完成！" << std::endl;
    return 0;
} 