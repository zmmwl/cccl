#include "mixed_column_processor.cuh"
#include <iostream>
#include <vector>
#include <map>
#include <cmath>

using namespace zmm;

// PassThroughFunctor 必须在全局作用域定义
struct PassThroughFunctor {
    __device__ float operator()(const MixedRowData& row) const {
        // 返回第二列（索引1）的浮点值
        if (row.num_columns > 2 && row.types[1] == ColumnDataType::FLOAT && row.types[2] == ColumnDataType::FLOAT) {
            return row.getFloat(1)+row.getFloat(2);
        }
        return 0.0f;
    }
};

int main() {
    std::cout << "=== GroupBy 功能简单测试 ===" << std::endl;
    
    // 创建简单测试数据
    const size_t num_elements = 20;
    
    // ID列：有重复的ID
    std::vector<int> ids = {
        1, 2, 1, 3, 2,
        1, 3, 2, 1, 3,
        2, 1, 3, 2, 1,
        3, 2, 1, 3, 2
    };
    
    // Value列
    std::vector<float> values = {
        10.0f, 20.0f, 15.0f, 30.0f, 25.0f,
        12.0f, 35.0f, 22.0f, 13.0f, 32.0f,
        21.0f, 14.0f, 33.0f, 23.0f, 16.0f,
        34.0f, 24.0f, 17.0f, 31.0f, 26.0f
    };

    
    // Value列
    std::vector<float> values2 = {
        10.0f, 20.0f, 15.0f, 30.0f, 25.0f,
        12.0f, 35.0f, 22.0f, 13.0f, 32.0f,
        21.0f, 14.0f, 33.0f, 23.0f, 16.0f,
        34.0f, 24.0f, 17.0f, 31.0f, 26.0f
    };
    
    
    std::cout << "\n原始数据:" << std::endl;
    std::cout << "ID   Value" << std::endl;
    std::cout << "----------" << std::endl;
    for (size_t i = 0; i < num_elements; ++i) {
        std::cout << ids[i] << "    " << values[i] << std::endl;
    }
    
    // 创建处理器
    auto processor = createMixedProcessor(num_elements);
    int id_col = processor->addIntColumn(ids);
    processor->addFloatColumn(values);
    processor->addFloatColumn(values2);
    
    std::cout << "\n添加列: ID列索引=" << id_col << std::endl;
    
    // 执行计算（使用全局作用域的 PassThroughFunctor）
    std::cout << "\n执行计算..." << std::endl;
    if (!processor->computeWithFunctor(PassThroughFunctor{})) {
        std::cerr << "计算失败!" << std::endl;
        return 1;
    }
    processor->synchronize();
    std::cout << "计算完成" << std::endl;
    
    // 获取计算结果
    auto compute_results = processor->getResult();
    std::cout << "\n计算结果（前10个）:" << std::endl;
    for (int i = 0; i < 10; ++i) {
        std::cout << "  [" << i << "] = " << compute_results[i] << std::endl;
    }
    
    // 执行分组求和
    std::cout << "\n执行分组求和..." << std::endl;
    auto groupby_result = processor->groupBySum(id_col);
    
    std::cout << "分组求和完成!" << std::endl;
    std::cout << "分组数量: " << groupby_result.num_groups << std::endl;
    
    // 显示分组结果
    std::cout << "\nGPU 分组求和结果:" << std::endl;
    std::cout << "ID   Sum" << std::endl;
    std::cout << "---------" << std::endl;
    for (size_t i = 0; i < groupby_result.num_groups; ++i) {
        std::cout << groupby_result.unique_keys[i] << "    " 
                  << groupby_result.aggregated_values[i] << std::endl;
    }
    
    // CPU验证
    std::cout << "\nCPU 验证:" << std::endl;
    std::map<int, float> cpu_sums;
    for (size_t i = 0; i < num_elements; ++i) {
        cpu_sums[ids[i]] += (values[i]+values2[i]);
    }
    
    std::cout << "ID   Sum (CPU)" << std::endl;
    std::cout << "--------------" << std::endl;
    for (const auto& pair : cpu_sums) {
        std::cout << pair.first << "    " << pair.second << std::endl;
    }
    
    // 比对结果
    std::cout << "\n结果对比:" << std::endl;
    bool all_match = true;
    for (size_t i = 0; i < groupby_result.num_groups; ++i) {
        int key = groupby_result.unique_keys[i];
        float gpu_sum = groupby_result.aggregated_values[i];
        float cpu_sum = cpu_sums[key];
        float diff = std::abs(gpu_sum - cpu_sum);
        
        std::cout << "ID " << key << ": GPU=" << gpu_sum << ", CPU=" << cpu_sum 
                  << ", 差值=" << diff;
        
        if (diff < 1e-3f) {
            std::cout << " ✓" << std::endl;
        } else {
            std::cout << " ✗" << std::endl;
            all_match = false;
        }
    }
    
    if (all_match) {
        std::cout << "\n✓ 测试通过！所有结果匹配！" << std::endl;
        return 0;
    } else {
        std::cout << "\n✗ 测试失败！存在不匹配的结果！" << std::endl;
        return 1;
    }
}

