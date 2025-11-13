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
        // return v1 + v2;
        return (float)((double)v1 + (double)v2);
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

// 多operator演示用的全局functors
struct SumOperator {
    __device__ float operator()(const MixedRowData& row) const {
        float v1 = (row.num_columns > 1 && row.types[1] == ColumnDataType::FLOAT) ? row.getFloat(1) : 0.0f;
        float v2 = (row.num_columns > 2 && row.types[2] == ColumnDataType::FLOAT) ? row.getFloat(2) : 0.0f;
        return v1 + v2;
    }
};

struct ProductOperator {
    __device__ float operator()(const MixedRowData& row) const {
        float v1 = (row.num_columns > 1 && row.types[1] == ColumnDataType::FLOAT) ? row.getFloat(1) : 0.0f;
        float v2 = (row.num_columns > 2 && row.types[2] == ColumnDataType::FLOAT) ? row.getFloat(2) : 0.0f;
        return v1 * v2;  // price * quantity = total
    }
};

struct AvgOperator {
    __device__ float operator()(const MixedRowData& row) const {
        float v1 = (row.num_columns > 1 && row.types[1] == ColumnDataType::FLOAT) ? row.getFloat(1) : 0.0f;
        float v2 = (row.num_columns > 2 && row.types[2] == ColumnDataType::FLOAT) ? row.getFloat(2) : 0.0f;
        return (v1 + v2) / 2.0f;
    }
};

// 收入计算functor
struct RevenueOperator {
    __device__ float operator()(const MixedRowData& row) const {
        float price = (row.num_columns > 1 && row.types[1] == ColumnDataType::FLOAT) ? row.getFloat(1) : 0.0f;
        float sales = (row.num_columns > 2 && row.types[2] == ColumnDataType::FLOAT) ? row.getFloat(2) : 0.0f;
        return price * sales;
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
    
    // const size_t num_elements = 1000;
    // const int num_groups = 10;
    
    const size_t num_elements = 10000000;
    const int num_groups = 1000;
    
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
    std::cout << "\n前10个分组求和结果:" << std::endl;
    std::cout << std::setw(8) << "ID" << std::setw(15) << "Sum" << std::endl;
    std::cout << std::string(23, '-') << std::endl;
    for (size_t i = 0; i < 10; ++i) {
        std::cout << std::setw(8) << groupby_result.unique_keys[i]
                  << std::setw(15) << std::fixed << std::setprecision(2) << groupby_result.aggregated_values[i]
                  << std::endl;
    }
    
    // CPU验证
    bool doCPUVerify = true;
    if (doCPUVerify){
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
                break;
            }
        }
        
        if (all_match) {
            std::cout << "✓ GPU结果与CPU验证完全匹配！" << std::endl;
        } else {
            std::cout << "✗ 存在误差，最大误差: " << max_error << std::endl;
        }
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

// 演示多operator功能
void demonstrateMultiOperators() {
    std::cout << "\n=== 多Operator演示 ===" << std::endl;
    
    const size_t num_elements = 1000;
    const int num_groups = 5;
    
    auto ids = generateGroupIds(num_elements, num_groups);
    auto prices = generateValues(num_elements, 10.0f, 100.0f);
    auto quantities = generateValues(num_elements, 1.0f, 10.0f);
    
    auto processor = createMixedProcessor(num_elements);
    int id_col = processor->addIntColumn(ids);
    processor->addFloatColumn(prices);
    processor->addFloatColumn(quantities);
    
    std::cout << "数据: " << num_elements << " 行, " << num_groups << " 个分组" << std::endl;
    
    // 执行多operator计算（使用全局定义的operators）
    std::cout << "\n执行3个operators: Sum, Product, Average..." << std::endl;
    processor->computeWithFunctors(SumOperator{}, ProductOperator{}, AvgOperator{});
    
    // 获取多列结果
    auto results = processor->getResults();
    std::cout << "生成了 " << results.size() << " 列结果" << std::endl;
    
    // 显示前5行的结果
    std::cout << "\n前5行的多operator结果:" << std::endl;
    std::cout << std::setw(8) << "Row" << std::setw(12) << "Sum" << std::setw(12) << "Product" 
              << std::setw(12) << "Average" << std::endl;
    std::cout << std::string(44, '-') << std::endl;
    for (int i = 0; i < 5; ++i) {
        std::cout << std::setw(8) << i;
        for (size_t j = 0; j < results.size(); ++j) {
            std::cout << std::setw(12) << std::fixed << std::setprecision(2) << results[j][i];
        }
        std::cout << std::endl;
    }
    
    // 对每个结果列进行分组求和
    std::cout << "\n对Product结果（列1）按ID分组求和..." << std::endl;
    auto groupby_result = processor->groupByAggregate(
        id_col, 
        MixedColumnProcessor::AggregationType::SUM,
        1,  // 使用第1个operator结果（Product）
        MixedColumnProcessor::DataSource::OPERATOR_RESULT
    );
    
    std::cout << "分组数量: " << groupby_result.num_groups << std::endl;
    std::cout << "\n前5个分组的Product总和:" << std::endl;
    std::cout << std::setw(8) << "ID" << std::setw(18) << "Total Product" << std::endl;
    std::cout << std::string(26, '-') << std::endl;
    for (size_t i = 0; i < std::min(size_t(5), groupby_result.num_groups); ++i) {
        std::cout << std::setw(8) << groupby_result.unique_keys[i]
                  << std::setw(18) << std::fixed << std::setprecision(2) 
                  << groupby_result.aggregated_values[i] << std::endl;
    }
}

// 演示指定聚合字段功能
void demonstrateSpecificAggregation() {
    std::cout << "\n=== 指定聚合字段演示 ===" << std::endl;
    
    const size_t num_elements = 800;
    const int num_groups = 4;
    
    auto category_ids = generateGroupIds(num_elements, num_groups);
    auto prices = generateValues(num_elements, 50.0f, 200.0f);
    auto sales_counts = generateValues(num_elements, 1.0f, 20.0f);
    auto ratings = generateValues(num_elements, 1.0f, 5.0f);
    
    auto processor = createMixedProcessor(num_elements);
    int category_col = processor->addIntColumn(category_ids);
    int price_col = processor->addFloatColumn(prices);
    int sales_col = processor->addFloatColumn(sales_counts);
    int rating_col = processor->addFloatColumn(ratings);
    
    std::cout << "数据: " << num_elements << " 行, " << num_groups << " 个分类" << std::endl;
    std::cout << "列: category_id, price, sales_count, rating" << std::endl;
    
    // 测试1: 从输入列聚合（聚合price列）
    std::cout << "\n测试1: 按category聚合price列（从输入列）..." << std::endl;
    auto result1 = processor->groupByAggregate(
        category_col,
        MixedColumnProcessor::AggregationType::SUM,
        price_col,
        MixedColumnProcessor::DataSource::INPUT_COLUMN
    );
    
    std::cout << "各分类的price总和:" << std::endl;
    std::cout << std::setw(12) << "Category" << std::setw(18) << "Total Price" << std::endl;
    std::cout << std::string(30, '-') << std::endl;
    for (size_t i = 0; i < result1.num_groups; ++i) {
        std::cout << std::setw(12) << result1.unique_keys[i]
                  << std::setw(18) << std::fixed << std::setprecision(2) 
                  << result1.aggregated_values[i] << std::endl;
    }
    
    // 测试2: 从输入列聚合（聚合rating列，使用MAX）
    std::cout << "\n测试2: 按category找最大rating（从输入列）..." << std::endl;
    auto result2 = processor->groupByAggregate(
        category_col,
        MixedColumnProcessor::AggregationType::MAX,
        rating_col,
        MixedColumnProcessor::DataSource::INPUT_COLUMN
    );
    
    std::cout << "各分类的最大rating:" << std::endl;
    std::cout << std::setw(12) << "Category" << std::setw(15) << "Max Rating" << std::endl;
    std::cout << std::string(27, '-') << std::endl;
    for (size_t i = 0; i < result2.num_groups; ++i) {
        std::cout << std::setw(12) << result2.unique_keys[i]
                  << std::setw(15) << std::fixed << std::setprecision(2) 
                  << result2.aggregated_values[i] << std::endl;
    }
    
    // 测试3: 先计算operator结果，然后从operator结果聚合
    std::cout << "\n测试3: 计算price*sales_count，然后按category聚合..." << std::endl;
    
    // 使用全局定义的RevenueOperator
    processor->computeWithFunctor(RevenueOperator{});
    
    auto result3 = processor->groupByAggregate(
        category_col,
        MixedColumnProcessor::AggregationType::SUM,
        -1,  // 使用默认的d_output_
        MixedColumnProcessor::DataSource::OPERATOR_RESULT
    );
    
    std::cout << "各分类的总收入 (price * sales_count):" << std::endl;
    std::cout << std::setw(12) << "Category" << std::setw(20) << "Total Revenue" << std::endl;
    std::cout << std::string(32, '-') << std::endl;
    for (size_t i = 0; i < result3.num_groups; ++i) {
        std::cout << std::setw(12) << result3.unique_keys[i]
                  << std::setw(20) << std::fixed << std::setprecision(2) 
                  << result3.aggregated_values[i] << std::endl;
    }
}

// 演示多键分组功能
void demonstrateMultiKeyGroupBy() {
    std::cout << "\n=== 多键分组演示 ===" << std::endl;
    
    const size_t num_elements = 1000;
    const int num_categories = 3;
    const int num_regions = 4;
    
    auto category_ids = generateGroupIds(num_elements, num_categories);
    auto region_ids = generateGroupIds(num_elements, num_regions);
    auto sales = generateValues(num_elements, 100.0f, 1000.0f);
    
    auto processor = createMixedProcessor(num_elements);
    int category_col = processor->addIntColumn(category_ids);
    int region_col = processor->addIntColumn(region_ids);
    int sales_col = processor->addFloatColumn(sales);
    
    std::cout << "数据: " << num_elements << " 行" << std::endl;
    std::cout << "分组键: category (" << num_categories << " 个) x region (" 
              << num_regions << " 个)" << std::endl;
    std::cout << "理论分组数: " << (num_categories * num_regions) << std::endl;
    
    // 显示前10行原始数据
    std::cout << "\n前10行原始数据:" << std::endl;
    std::cout << std::setw(10) << "Category" << std::setw(10) << "Region" 
              << std::setw(12) << "Sales" << std::endl;
    std::cout << std::string(32, '-') << std::endl;
    for (int i = 0; i < 10; ++i) {
        std::cout << std::setw(10) << category_ids[i] 
                  << std::setw(10) << region_ids[i]
                  << std::setw(12) << std::fixed << std::setprecision(2) << sales[i]
                  << std::endl;
    }
    
    // 按多键分组（category, region）聚合sales
    std::cout << "\n按 (category, region) 多键分组求和sales..." << std::endl;
    std::vector<int> key_columns = {category_col, region_col};
    auto result = processor->groupByAggregateMultiKey(
        key_columns,
        MixedColumnProcessor::AggregationType::SUM,
        sales_col,
        MixedColumnProcessor::DataSource::INPUT_COLUMN
    );
    
    std::cout << "实际分组数: " << result.num_groups << std::endl;
    std::cout << "是否多键分组: " << (result.is_multi_key ? "是" : "否") << std::endl;
    
    // 显示所有分组结果
    std::cout << "\n所有分组的sales总和:" << std::endl;
    std::cout << std::setw(12) << "Category" << std::setw(10) << "Region" 
              << std::setw(18) << "Total Sales" << std::endl;
    std::cout << std::string(40, '-') << std::endl;
    
    for (size_t i = 0; i < result.num_groups; ++i) {
        const auto& keys = result.unique_multi_keys[i];
        std::cout << std::setw(12) << keys[0]  // category
                  << std::setw(10) << keys[1]  // region
                  << std::setw(18) << std::fixed << std::setprecision(2) 
                  << result.aggregated_values[i] << std::endl;
    }
    
    // 找出销售额最高的category-region组合
    size_t max_idx = 0;
    float max_sales = result.aggregated_values[0];
    for (size_t i = 1; i < result.num_groups; ++i) {
        if (result.aggregated_values[i] > max_sales) {
            max_sales = result.aggregated_values[i];
            max_idx = i;
        }
    }
    
    std::cout << "\n销售额最高的组合:" << std::endl;
    std::cout << "  Category: " << result.unique_multi_keys[max_idx][0] << std::endl;
    std::cout << "  Region: " << result.unique_multi_keys[max_idx][1] << std::endl;
    std::cout << "  Total Sales: " << std::fixed << std::setprecision(2) << max_sales << std::endl;
}

// 综合演示：三键分组
void demonstrateThreeKeyGroupBy() {
    std::cout << "\n=== 三键分组演示 ===" << std::endl;
    
    const size_t num_elements = 2000;
    const int num_products = 3;
    const int num_stores = 4;
    const int num_months = 3;
    
    auto product_ids = generateGroupIds(num_elements, num_products);
    auto store_ids = generateGroupIds(num_elements, num_stores);
    auto month_ids = generateGroupIds(num_elements, num_months);
    auto revenues = generateValues(num_elements, 500.0f, 5000.0f);
    
    auto processor = createMixedProcessor(num_elements);
    int product_col = processor->addIntColumn(product_ids);
    int store_col = processor->addIntColumn(store_ids);
    int month_col = processor->addIntColumn(month_ids);
    int revenue_col = processor->addFloatColumn(revenues);
    
    std::cout << "数据: " << num_elements << " 笔交易" << std::endl;
    std::cout << "维度: " << num_products << " 个产品 x " << num_stores 
              << " 个门店 x " << num_months << " 个月份" << std::endl;
    std::cout << "理论分组数: " << (num_products * num_stores * num_months) << std::endl;
    
    // 三键分组
    std::cout << "\n按 (product, store, month) 三键分组求和revenue..." << std::endl;
    std::vector<int> key_columns = {product_col, store_col, month_col};
    auto result = processor->groupByAggregateMultiKey(
        key_columns,
        MixedColumnProcessor::AggregationType::SUM,
        revenue_col,
        MixedColumnProcessor::DataSource::INPUT_COLUMN
    );
    
    std::cout << "实际分组数: " << result.num_groups << std::endl;
    
    // 显示前15个分组结果
    std::cout << "\n前15个分组的revenue总和:" << std::endl;
    std::cout << std::setw(10) << "Product" << std::setw(8) << "Store" 
              << std::setw(8) << "Month" << std::setw(18) << "Total Revenue" << std::endl;
    std::cout << std::string(44, '-') << std::endl;
    
    for (size_t i = 0; i < std::min(size_t(15), result.num_groups); ++i) {
        const auto& keys = result.unique_multi_keys[i];
        std::cout << std::setw(10) << keys[0]  // product
                  << std::setw(8) << keys[1]  // store
                  << std::setw(8) << keys[2]  // month
                  << std::setw(18) << std::fixed << std::setprecision(2) 
                  << result.aggregated_values[i] << std::endl;
    }
    
    // 找出表现最好的组合
    size_t max_idx = std::distance(result.aggregated_values.begin(),
                                    std::max_element(result.aggregated_values.begin(), 
                                                    result.aggregated_values.end()));
    
    std::cout << "\n业绩最好的组合:" << std::endl;
    std::cout << "  Product ID: " << result.unique_multi_keys[max_idx][0] << std::endl;
    std::cout << "  Store ID: " << result.unique_multi_keys[max_idx][1] << std::endl;
    std::cout << "  Month: " << result.unique_multi_keys[max_idx][2] << std::endl;
    std::cout << "  Total Revenue: " << std::fixed << std::setprecision(2) 
              << result.aggregated_values[max_idx] << std::endl;
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
        // 运行原有演示
        demonstrateBasicGroupBy();
        demonstrateAggregationTypes();
        demonstrateBusinessScenario();
        demonstrateLargeDataset();
        
        // 运行新功能演示
        std::cout << "\n\n" << std::string(60, '=') << std::endl;
        std::cout << "新功能演示" << std::endl;
        std::cout << std::string(60, '=') << std::endl;
        
        demonstrateMultiOperators();
        demonstrateSpecificAggregation();
        demonstrateMultiKeyGroupBy();
        demonstrateThreeKeyGroupBy();
        
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

