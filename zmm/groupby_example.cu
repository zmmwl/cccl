#include "mixed_column_processor.cuh"
#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <map>
#include <algorithm>
#include <chrono>

using namespace zmm;

// 全局 Functor 定义（必须在全局作用域，不能在函数内部）
struct AddTwoColumnsFunctor {
    __device__ float operator()(const MixedRowData& row) const {
        float v1 = (row.num_columns > 1 && row.types[1] == ColumnDataType::FLOAT) ? row.getFloat(1) : 0.0f;
        float v2 = (row.num_columns > 2 && row.types[2] == ColumnDataType::FLOAT) ? row.getFloat(2) : 0.0f;
        return v1 + v2;
    }
};

struct PassThroughFunctor {
    __device__ float operator()(const MixedRowData& row) const {
        return (row.num_columns > 1 && row.types[1] == ColumnDataType::FLOAT) ? row.getFloat(1) : 0.0f;
    }
};

struct AddFunctor {
    __device__ float operator()(const MixedRowData& row) const {
        float v1 = (row.num_columns > 1 && row.types[1] == ColumnDataType::FLOAT) ? row.getFloat(1) : 0.0f;
        float v2 = (row.num_columns > 2 && row.types[2] == ColumnDataType::FLOAT) ? row.getFloat(2) : 0.0f;
        return v1 + v2;
    }
};

struct SalesAmountFunctor {
    __device__ float operator()(const MixedRowData& row) const {
        float price = (row.num_columns > 1 && row.types[1] == ColumnDataType::FLOAT) ? row.getFloat(1) : 0.0f;
        float qty = (row.num_columns > 2 && row.types[2] == ColumnDataType::FLOAT) ? row.getFloat(2) : 0.0f;
        return price * qty;
    }
};

// 生成随机测试数据
std::vector<int> generateGroupIds(size_t size, int num_groups) {
    std::vector<int> ids(size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(0, num_groups - 1);
    
    for (size_t i = 0; i < size; ++i) {
        ids[i] = dis(gen);
    }
    return ids;
}

std::vector<float> generateValues(size_t size, float min_val = 1.0f, float max_val = 100.0f) {
    std::vector<float> values(size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min_val, max_val);
    
    for (size_t i = 0; i < size; ++i) {
        values[i] = dis(gen);
    }
    return values;
}

// CPU端验证函数
std::map<int, float> cpuGroupBySum(const std::vector<int>& ids, const std::vector<float>& values) {
    std::map<int, float> result;
    for (size_t i = 0; i < ids.size(); ++i) {
        result[ids[i]] += values[i];
    }
    return result;
}

void demonstrateBasicGroupBy() {
    std::cout << "\n=== 基础分组求和演示 ===" << std::endl;
    
    const size_t num_elements = 1000;
    const int num_groups = 10;
    
    // 生成测试数据
    auto ids = generateGroupIds(num_elements, num_groups);
    auto values1 = generateValues(num_elements, 1.0f, 50.0f);
    auto values2 = generateValues(num_elements, 1.0f, 30.0f);
    
    std::cout << "数据大小: " << num_elements << " 行, " << num_groups << " 个分组" << std::endl;
    
    // 显示前10行数据
    std::cout << "\n前10行原始数据:" << std::endl;
    std::cout << std::setw(8) << "ID" << std::setw(12) << "Value1" << std::setw(12) << "Value2" << std::endl;
    std::cout << std::string(32, '-') << std::endl;
    for (int i = 0; i < 10; ++i) {
        std::cout << std::setw(8) << ids[i] 
                  << std::setw(12) << std::fixed << std::setprecision(2) << values1[i]
                  << std::setw(12) << std::fixed << std::setprecision(2) << values2[i]
                  << std::endl;
    }
    
    // 创建处理器并添加列
    auto processor = createMixedProcessor(num_elements);
    int id_col = processor->addIntColumn(ids);
    int val1_col = processor->addFloatColumn(values1);
    int val2_col = processor->addFloatColumn(values2);
    
    std::cout << "\n添加的列: ID(" << id_col << "), Value1(" << val1_col << "), Value2(" << val2_col << ")" << std::endl;
    
    // 执行计算（使用全局的 AddTwoColumnsFunctor）
    std::cout << "\n执行计算: Value1 + Value2..." << std::endl;
    auto compute_start = std::chrono::high_resolution_clock::now();
    processor->computeWithFunctor(AddTwoColumnsFunctor{});
    processor->synchronize();
    auto compute_end = std::chrono::high_resolution_clock::now();
    auto compute_time = std::chrono::duration<float, std::milli>(compute_end - compute_start).count();
    std::cout << "计算完成，用时: " << compute_time << " ms" << std::endl;
    
    // 获取未分组的结果
    auto raw_results = processor->getResult();
    std::cout << "\n前10个计算结果:" << std::endl;
    for (int i = 0; i < 10; ++i) {
        std::cout << "  Row " << i << ": " << std::fixed << std::setprecision(2) 
                  << raw_results[i] << " = " << values1[i] << " + " << values2[i] << std::endl;
    }
    
    // 执行分组求和
    std::cout << "\n执行分组求和（按ID分组）..." << std::endl;
    auto groupby_start = std::chrono::high_resolution_clock::now();
    auto groupby_result = processor->groupBySum(id_col);
    auto groupby_end = std::chrono::high_resolution_clock::now();
    auto groupby_time = std::chrono::duration<float, std::milli>(groupby_end - groupby_start).count();
    
    std::cout << "分组求和完成，用时: " << groupby_time << " ms" << std::endl;
    std::cout << "分组数量: " << groupby_result.num_groups << std::endl;
    
    // 显示分组结果
    std::cout << "\n分组求和结果:" << std::endl;
    std::cout << std::setw(8) << "ID" << std::setw(15) << "Sum" << std::endl;
    std::cout << std::string(23, '-') << std::endl;
    for (size_t i = 0; i < groupby_result.num_groups; ++i) {
        std::cout << std::setw(8) << groupby_result.unique_keys[i]
                  << std::setw(15) << std::fixed << std::setprecision(2) << groupby_result.aggregated_values[i]
                  << std::endl;
    }
    
    // CPU验证
    std::cout << "\nCPU端验证..." << std::endl;
    auto cpu_result = cpuGroupBySum(ids, raw_results);
    
    bool all_match = true;
    float max_error = 0.0f;
    for (size_t i = 0; i < groupby_result.num_groups; ++i) {
        int key = groupby_result.unique_keys[i];
        float gpu_sum = groupby_result.aggregated_values[i];
        float cpu_sum = cpu_result[key];
        float error = std::abs(gpu_sum - cpu_sum);
        
        if (error > 1e-3f) {
            all_match = false;
            max_error = std::max(max_error, error);
            std::cout << "  不匹配: ID=" << key << ", GPU=" << gpu_sum << ", CPU=" << cpu_sum 
                      << ", 误差=" << error << std::endl;
        }
    }
    
    if (all_match) {
        std::cout << "✓ GPU结果与CPU验证完全匹配！" << std::endl;
    } else {
        std::cout << "✗ 存在误差，最大误差: " << max_error << std::endl;
    }
}

void demonstrateAggregationTypes() {
    std::cout << "\n=== 不同聚合类型演示 ===" << std::endl;
    
    const size_t num_elements = 500;
    const int num_groups = 5;
    
    auto ids = generateGroupIds(num_elements, num_groups);
    auto values = generateValues(num_elements, 1.0f, 100.0f);
    
    auto processor = createMixedProcessor(num_elements);
    int id_col = processor->addIntColumn(ids);
    processor->addFloatColumn(values);
    
    // 使用全局的 PassThroughFunctor
    processor->computeWithFunctor(PassThroughFunctor{});
    processor->synchronize();
    
    // 测试不同的聚合类型
    std::vector<std::pair<std::string, MixedColumnProcessor::AggregationType>> agg_types = {
        {"SUM", MixedColumnProcessor::AggregationType::SUM},
        {"MAX", MixedColumnProcessor::AggregationType::MAX},
        {"MIN", MixedColumnProcessor::AggregationType::MIN}
    };
    
    for (const auto& pair : agg_types) {
        const std::string& name = pair.first;
        MixedColumnProcessor::AggregationType type = pair.second;
        std::cout << "\n--- " << name << " 聚合 ---" << std::endl;
        auto result = processor->groupByAggregate(id_col, type);
        
        std::cout << std::setw(8) << "ID" << std::setw(15) << name << std::endl;
        std::cout << std::string(23, '-') << std::endl;
        for (size_t i = 0; i < result.num_groups; ++i) {
            std::cout << std::setw(8) << result.unique_keys[i]
                      << std::setw(15) << std::fixed << std::setprecision(2) << result.aggregated_values[i]
                      << std::endl;
        }
    }
}

void demonstrateLargeDataset() {
    std::cout << "\n=== 大数据集性能测试 ===" << std::endl;
    
    const size_t num_elements = 10000000; // 1000万行
    const int num_groups = 1000;
    
    std::cout << "生成测试数据: " << num_elements << " 行, " << num_groups << " 个分组..." << std::endl;
    
    auto ids = generateGroupIds(num_elements, num_groups);
    auto values1 = generateValues(num_elements, 1.0f, 100.0f);
    auto values2 = generateValues(num_elements, 1.0f, 100.0f);
    
    auto processor = createMixedProcessor(num_elements);
    int id_col = processor->addIntColumn(ids);
    processor->addFloatColumn(values1);
    processor->addFloatColumn(values2);
    
    std::cout << "数据加载完成" << std::endl;
    
    // 执行计算（使用全局的 AddFunctor）
    std::cout << "\n执行计算..." << std::endl;
    auto compute_start = std::chrono::high_resolution_clock::now();
    processor->computeWithFunctor(AddFunctor{});
    processor->synchronize();
    auto compute_end = std::chrono::high_resolution_clock::now();
    auto compute_time = std::chrono::duration<float, std::milli>(compute_end - compute_start).count();
    
    std::cout << "计算完成，用时: " << compute_time << " ms" << std::endl;
    std::cout << "吞吐量: " << (num_elements / compute_time / 1000.0) << " M 元素/秒" << std::endl;
    
    // 分组求和
    std::cout << "\n执行分组求和..." << std::endl;
    auto groupby_start = std::chrono::high_resolution_clock::now();
    auto result = processor->groupBySum(id_col);
    auto groupby_end = std::chrono::high_resolution_clock::now();
    auto groupby_time = std::chrono::duration<float, std::milli>(groupby_end - groupby_start).count();
    
    std::cout << "分组求和完成，用时: " << groupby_time << " ms" << std::endl;
    std::cout << "分组数量: " << result.num_groups << std::endl;
    std::cout << "吞吐量: " << (num_elements / groupby_time / 1000.0) << " M 元素/秒" << std::endl;
    
    // 显示部分结果
    std::cout << "\n前10个分组的结果:" << std::endl;
    std::cout << std::setw(8) << "ID" << std::setw(18) << "Sum" << std::setw(15) << "Avg (估算)" << std::endl;
    std::cout << std::string(41, '-') << std::endl;
    for (size_t i = 0; i < std::min(size_t(10), result.num_groups); ++i) {
        float estimated_count = static_cast<float>(num_elements) / num_groups;
        float estimated_avg = result.aggregated_values[i] / estimated_count;
        std::cout << std::setw(8) << result.unique_keys[i]
                  << std::setw(18) << std::fixed << std::setprecision(2) << result.aggregated_values[i]
                  << std::setw(15) << std::fixed << std::setprecision(2) << estimated_avg
                  << std::endl;
    }
    
    std::cout << "\n总体性能:" << std::endl;
    std::cout << "  总时间: " << (compute_time + groupby_time) << " ms" << std::endl;
    std::cout << "  计算阶段占比: " << (compute_time / (compute_time + groupby_time) * 100) << "%" << std::endl;
    std::cout << "  分组阶段占比: " << (groupby_time / (compute_time + groupby_time) * 100) << "%" << std::endl;
}

void demonstrateBusinessScenario() {
    std::cout << "\n=== 业务场景: 销售数据分析 ===" << std::endl;
    
    const size_t num_transactions = 100000;
    const int num_products = 50;
    
    std::cout << "模拟场景: " << num_transactions << " 笔交易, " << num_products << " 种商品" << std::endl;
    
    // 生成模拟数据
    auto product_ids = generateGroupIds(num_transactions, num_products);
    auto prices = generateValues(num_transactions, 10.0f, 500.0f);
    auto quantities = generateValues(num_transactions, 1.0f, 10.0f);
    
    auto processor = createMixedProcessor(num_transactions);
    int product_col = processor->addNamedIntColumn("product_id", product_ids);
    processor->addNamedFloatColumn("price", prices);
    processor->addNamedFloatColumn("quantity", quantities);
    
    // 计算总销售额 = price * quantity（使用全局的 SalesAmountFunctor）
    std::cout << "\n计算每笔交易的销售额..." << std::endl;
    processor->computeWithFunctor(SalesAmountFunctor{});
    processor->synchronize();
    
    // 按商品分组求和
    std::cout << "按商品ID分组统计销售额..." << std::endl;
    auto sales_by_product = processor->groupBySum(product_col);
    
    std::cout << "\n商品销售统计 (前20个商品):" << std::endl;
    std::cout << std::setw(12) << "商品ID" << std::setw(18) << "总销售额" << std::endl;
    std::cout << std::string(30, '-') << std::endl;
    
    // 按销售额排序
    std::vector<std::pair<float, int>> sorted_sales;
    for (size_t i = 0; i < sales_by_product.num_groups; ++i) {
        sorted_sales.emplace_back(sales_by_product.aggregated_values[i], sales_by_product.unique_keys[i]);
    }
    std::sort(sorted_sales.rbegin(), sorted_sales.rend());
    
    float total_sales = 0.0f;
    for (size_t i = 0; i < std::min(size_t(20), sorted_sales.size()); ++i) {
        total_sales += sorted_sales[i].first;
        std::cout << std::setw(12) << sorted_sales[i].second
                  << std::setw(18) << std::fixed << std::setprecision(2) << sorted_sales[i].first
                  << std::endl;
    }
    
    // 计算总销售额（剩余的商品）
    for (size_t i = 20; i < sorted_sales.size(); ++i) {
        total_sales += sorted_sales[i].first;
    }
    
    std::cout << "\n总销售额: " << std::fixed << std::setprecision(2) << total_sales << std::endl;
    std::cout << "平均每商品销售额: " << (total_sales / sales_by_product.num_groups) << std::endl;
}

int main() {
    std::cout << "ZMM 混合类型框架 - GroupBy 分组求和功能演示" << std::endl;
    std::cout << "================================================" << std::endl;
    
    // 检查CUDA设备
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        std::cerr << "错误: 未找到CUDA设备！" << std::endl;
        return -1;
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "使用GPU: " << prop.name << std::endl;
    std::cout << "全局内存: " << prop.totalGlobalMem / (1024*1024*1024) << " GB\n" << std::endl;
    
    try {
        // 运行演示
        demonstrateBasicGroupBy();
        demonstrateAggregationTypes();
        demonstrateBusinessScenario();
        demonstrateLargeDataset();
        
        std::cout << "\n=== 所有演示完成 ===" << std::endl;
        std::cout << "GroupBy分组求和功能测试成功！" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "演示过程中发生错误: " << e.what() << std::endl;
        return -1;
    } catch (...) {
        std::cerr << "演示过程中发生未知错误！" << std::endl;
        return -1;
    }
    
    return 0;
}

