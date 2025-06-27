#include "dynamic_processor.h"
#include <iostream>
#include <vector>

using namespace zmm;

int main() {
    std::cout << "ZMM 动态操作加载示例" << std::endl;
    std::cout << "====================" << std::endl;
    
    // 定义数据大小
    const size_t num_elements = 1000;
    
    // 创建动态处理器（3列）
    auto processor = createDynamicProcessor<3>(num_elements);
    
    // 准备测试数据
    std::vector<float> column1(num_elements);
    std::vector<float> column2(num_elements);
    std::vector<float> column3(num_elements);
    
    for (size_t i = 0; i < num_elements; ++i) {
        column1[i] = static_cast<float>(i + 1);
        column2[i] = static_cast<float>((i + 1) * 2);
        column3[i] = static_cast<float>((i + 1) * 3);
    }
    
    // 设置输入数据
    processor->setInputColumn(0, column1.data());
    processor->setInputColumn(1, column2.data());
    processor->setInputColumn(2, column3.data());
    
    // 尝试加载不同的操作动态库
    std::vector<std::pair<std::string, std::string>> operations = {
        {"./lib/liboperations_add_3.so", "AddOperation"},
        {"./lib/liboperations_multiply_3.so", "MultiplyOperation"}
    };
    
    for (const auto& op : operations) {
        std::cout << "\n=== 尝试加载 " << op.second << " ===" << std::endl;
        
        if (processor->loadOperation(op.first, op.second)) {
            std::cout << "执行 " << op.second << "..." << std::endl;
            
            if (processor->executeOperation(op.second)) {
                std::vector<float> result(num_elements);
                processor->getResult(result.data());
                
                // 显示前5个结果
                std::cout << "结果（前5个）：" << std::endl;
                for (int i = 0; i < 5; ++i) {
                    float expected = 0.0f;
                    if (op.second == "AddOperation") {
                        expected = column1[i] + column2[i] + column3[i];
                    } else if (op.second == "MultiplyOperation") {
                        expected = column1[i] * column2[i] * column3[i];
                    }
                    
                    std::cout << "  [" << i << "] GPU结果: " << result[i] 
                              << ", 期望值: " << expected 
                              << ", 差异: " << std::abs(result[i] - expected) << std::endl;
                }
            } else {
                std::cout << "执行操作失败！" << std::endl;
            }
        } else {
            std::cout << "加载操作失败！" << std::endl;
            std::cout << "提示：请确保动态库文件存在并且路径正确" << std::endl;
        }
    }
    
    // 显示已加载的操作
    auto loaded_ops = processor->getLoadedOperations();
    std::cout << "\n已加载的操作：" << std::endl;
    for (const auto& op : loaded_ops) {
        std::cout << "  - " << op << std::endl;
    }
    
    std::cout << "\n示例完成！" << std::endl;
    return 0;
} 