#include "mixed_column_processor.cuh"
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include <algorithm>

using namespace zmm;

// 生成测试数据的辅助函数
std::vector<float> generateRandomFloats(size_t size, float min_val = 0.0f, float max_val = 100.0f) {
    std::vector<float> data(size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min_val, max_val);
    
    for (size_t i = 0; i < size; ++i) {
        data[i] = dis(gen);
    }
    return data;
}

std::vector<std::string> generateRandomStrings(size_t size) {
    std::vector<std::string> data(size);
    std::vector<std::string> categories = {"high", "medium", "low", "premium", "discount"};
    std::vector<std::string> brands = {"premium", "popular", "generic"};
    std::vector<std::string> types = {"electronics", "books", "clothing", "luxury"};
    
    std::random_device rd;
    std::mt19937 gen(rd());
    
    for (size_t i = 0; i < size; ++i) {
        if (i % 3 == 0) {
            data[i] = categories[gen() % categories.size()];
        } else if (i % 3 == 1) {
            data[i] = brands[gen() % brands.size()];
        } else {
            data[i] = types[gen() % types.size()];
        }
    }
    return data;
}

void demonstrateBasicUsage() {
    std::cout << "\n=== 基础使用演示 ===" << std::endl;
    
    const size_t num_elements = 10000;
    
    // 创建混合处理器
    auto processor = createMixedProcessor(num_elements);
    
    // 准备测试数据
    auto prices = generateRandomFloats(num_elements, 10.0f, 1000.0f);
    auto ratings = generateRandomFloats(num_elements, 1.0f, 5.0f);
    auto categories = generateRandomStrings(num_elements);
    auto brands = generateRandomStrings(num_elements);
    
    // 添加列数据
    int price_col = processor->addFloatColumn(prices);
    int rating_col = processor->addFloatColumn(ratings);
    int category_col = processor->addStringColumn(categories);
    int brand_col = processor->addStringColumn(brands);
    
    std::cout << "已添加列: 价格(" << price_col << "), 评分(" << rating_col 
              << "), 类别(" << category_col << "), 品牌(" << brand_col << ")" << std::endl;
    
    // 显示处理器信息
    processor->printColumnInfo();
    
    // 显示示例数据
    processor->printSampleData(5);
    
    std::cout << "基础使用演示完成。" << std::endl;
}

void demonstrateNamedAccess() {
    std::cout << "\n=== 按字段名访问演示 ===" << std::endl;
    const size_t num_elements = 8192;

    auto processor = createMixedProcessor(num_elements);
    
    // 构造数据
    std::vector<float> prices(num_elements);
    std::vector<float> ratings(num_elements);
    std::vector<std::string> categories(num_elements);
    std::vector<std::string> brands(num_elements);
    for (size_t i = 0; i < num_elements; ++i) {
        prices[i] = 10.0f + static_cast<float>(i % 100) * 0.25f;
        ratings[i] = 1.0f + static_cast<float>(i % 5) * 0.5f;
        categories[i] = (i % 2 == 0) ? std::string("electronics") : std::string("books");
        brands[i] = (i % 3 == 0) ? std::string("premium") : std::string("generic");
    }

    // 以名称添加列
    processor->addNamedFloatColumn("price", prices);
    processor->addNamedFloatColumn("rating", ratings);
    processor->addNamedStringColumn("category", categories);
    processor->addNamedStringColumn("brand", brands);

    // 使用Functor通过字段名访问
    bool ok = processor->computeWithFunctor(NamedPriceRatingSumFunctor{});
    processor->synchronize();
    if (!ok) {
        std::cout << "Named functor compute failed" << std::endl;
        return;
    }
    auto result_sum = processor->getResult();
    std::cout << "price+rating（前5个）: ";
    for (int i = 0; i < 5; ++i) {
        std::cout << std::fixed << std::setprecision(2) << result_sum[i] << (i < 4 ? ", " : "\n");
    }

    // 使用命名电商评分Functor
    ok = processor->computeWithFunctor(NamedEcommerceScoreFunctor{});
    processor->synchronize();
    if (ok) {
        auto scores = processor->getResult();
        std::cout << "Named EcommerceScore（前5个）: ";
        for (int i = 0; i < 5; ++i) {
            std::cout << std::fixed << std::setprecision(2) << scores[i] << (i < 4 ? ", " : "\n");
        }
    }
}

void demonstrateOperations() {
    std::cout << "\n=== 操作演示 ===" << std::endl;
    
    const size_t num_elements = 100000;
    
    auto processor = createMixedProcessor(num_elements);
    
    // 添加测试数据
    auto values1 = generateRandomFloats(num_elements, 1.0f, 100.0f);
    auto values2 = generateRandomFloats(num_elements, 1.0f, 50.0f);
    auto categories = generateRandomStrings(num_elements);
    
    processor->addFloatColumn(values1);
    processor->addFloatColumn(values2);
    processor->addStringColumn(categories);
    
    // 测试不同的操作
    std::vector<std::pair<std::string, MixedOperationFactory::OperationType>> operations = {
        {"浮点数加法", MixedOperationFactory::MIXED_ADD},
        {"字符串长度统计", MixedOperationFactory::STRING_LENGTH_SUM},
        {"混合加权求和", MixedOperationFactory::MIXED_SUM_WITH_STRING_WEIGHT},
        {"条件计算", MixedOperationFactory::CONDITIONAL_MIXED},
        {"字符串转数值", MixedOperationFactory::STRING_TO_NUMBER},
        {"字符串分类", MixedOperationFactory::STRING_CATEGORY}
    };
    
    for (const auto& op : operations) {
        const std::string& name = op.first;
        MixedOperationFactory::OperationType op_type = op.second;
        std::cout << "\n--- 执行操作: " << name << " ---" << std::endl;
        
        auto start = std::chrono::high_resolution_clock::now();
        
        bool success = processor->compute(op_type);
        processor->synchronize();
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration<float, std::milli>(end - start).count();
        
        if (success) {
            auto results = processor->getResult();
            
            // 显示前5个结果
            std::cout << "前5个结果: ";
            for (int i = 0; i < 5 && i < results.size(); ++i) {
                std::cout << std::fixed << std::setprecision(2) << results[i];
                if (i < 4) std::cout << ", ";
            }
            std::cout << std::endl;
            std::cout << "计算时间: " << duration << " ms" << std::endl;
            
            // 统计结果
            float sum = 0.0f, min_val = results[0], max_val = results[0];
            for (float val : results) {
                sum += val;
                min_val = std::min(min_val, val);
                max_val = std::max(max_val, val);
            }
            float avg = sum / results.size();
            
            std::cout << "统计: 最小=" << min_val << ", 最大=" << max_val 
                      << ", 平均=" << avg << ", 总和=" << sum << std::endl;
        } else {
            std::cout << "操作执行失败！" << std::endl;
        }
    }
}

void demonstrateEcommerceScenario() {
    std::cout << "\n=== 电商评分场景演示 ===" << std::endl;
    
    const size_t num_products = 50000;
    
    auto processor = createMixedProcessor(num_products);
    
    // 生成电商数据
    auto prices = generateRandomFloats(num_products, 5.0f, 2000.0f);
    auto ratings = generateRandomFloats(num_products, 1.0f, 5.0f);
    
    std::vector<std::string> categories;
    std::vector<std::string> brands;
    std::vector<std::string> product_types = {"electronics", "books", "clothing", "luxury"};
    std::vector<std::string> brand_types = {"premium", "popular", "generic"};
    
    std::random_device rd;
    std::mt19937 gen(rd());
    
    for (size_t i = 0; i < num_products; ++i) {
        categories.push_back(product_types[gen() % product_types.size()]);
        brands.push_back(brand_types[gen() % brand_types.size()]);
    }
    
    // 添加列（按照EcommerceScoreOperation的期望顺序）
    processor->addFloatColumn(prices);      // 第0列：价格
    processor->addFloatColumn(ratings);     // 第1列：评分
    processor->addStringColumn(categories); // 第2列：类别
    processor->addStringColumn(brands);     // 第3列：品牌
    
    std::cout << "生成了 " << num_products << " 个商品的数据" << std::endl;
    processor->printSampleData(8);
    
    // 执行电商评分计算
    std::cout << "\n执行电商评分计算..." << std::endl;
    
    auto start = std::chrono::high_resolution_clock::now();
    bool success = processor->compute(MixedOperationFactory::ECOMMERCE_SCORE);
    processor->synchronize();
    auto end = std::chrono::high_resolution_clock::now();
    
    if (success) {
        auto scores = processor->getResult();
        auto duration = std::chrono::duration<float, std::milli>(end - start).count();
        
        std::cout << "计算完成，用时: " << duration << " ms" << std::endl;
        
        // 分析结果
        float total_score = 0.0f;
        float min_score = scores[0], max_score = scores[0];
        size_t high_score_count = 0;
        
        for (float score : scores) {
            total_score += score;
            min_score = std::min(min_score, score);
            max_score = std::max(max_score, score);
            if (score > 50.0f) high_score_count++;
        }
        
        float avg_score = total_score / num_products;
        
        std::cout << "\n评分统计:" << std::endl;
        std::cout << "  平均评分: " << std::fixed << std::setprecision(2) << avg_score << std::endl;
        std::cout << "  最低评分: " << min_score << std::endl;
        std::cout << "  最高评分: " << max_score << std::endl;
        std::cout << "  高分商品数量 (>50): " << high_score_count 
                  << " (" << (100.0f * high_score_count / num_products) << "%)" << std::endl;
        
        // 显示前10个最高评分的商品
        std::vector<std::pair<float, size_t>> score_indices;
        for (size_t i = 0; i < scores.size(); ++i) {
            score_indices.emplace_back(scores[i], i);
        }
        std::sort(score_indices.begin(), score_indices.end(), std::greater<>());
        
        std::cout << "\n前10个最高评分商品:" << std::endl;
        std::cout << std::setw(6) << "排名" << std::setw(10) << "评分" << std::setw(10) << "价格" 
                  << std::setw(10) << "评级" << std::setw(15) << "类别" << std::setw(12) << "品牌" << std::endl;
        std::cout << std::string(67, '-') << std::endl;
        
        for (int i = 0; i < 10 && i < score_indices.size(); ++i) {
            size_t idx = score_indices[i].second;
            std::cout << std::setw(6) << (i + 1)
                      << std::setw(10) << std::fixed << std::setprecision(2) << scores[idx]
                      << std::setw(10) << std::fixed << std::setprecision(2) << prices[idx]
                      << std::setw(10) << std::fixed << std::setprecision(2) << ratings[idx]
                      << std::setw(15) << categories[idx]
                      << std::setw(12) << brands[idx] << std::endl;
        }
        
        std::cout << "\n性能指标:" << std::endl;
        std::cout << "  处理速度: " << (num_products / duration * 1000.0f) << " 商品/秒" << std::endl;
        std::cout << "  内存使用: " << (processor->getDeviceMemoryUsage() / (1024.0f * 1024.0f)) << " MB" << std::endl;
    } else {
        std::cout << "电商评分计算失败！" << std::endl;
    }
}

void demonstratePerformanceBenchmark() {
    std::cout << "\n=== 性能基准测试 ===" << std::endl;
    
    const size_t num_elements = 1000000;
    const int num_iterations = 50;
    
    auto processor = createMixedProcessor(num_elements);
    
    // 添加测试数据
    auto values1 = generateRandomFloats(num_elements, 1.0f, 100.0f);
    auto values2 = generateRandomFloats(num_elements, 1.0f, 100.0f);
    auto strings = generateRandomStrings(num_elements);
    
    processor->addFloatColumn(values1);
    processor->addFloatColumn(values2);
    processor->addStringColumn(strings);
    
    std::cout << "测试数据: " << num_elements << " 个元素, " << num_iterations << " 次迭代" << std::endl;
    
    // 测试不同操作的性能
    std::vector<std::pair<std::string, MixedOperationFactory::OperationType>> benchmark_ops = {
        {"浮点数加法", MixedOperationFactory::MIXED_ADD},
        {"条件计算", MixedOperationFactory::CONDITIONAL_MIXED},
        {"字符串长度统计", MixedOperationFactory::STRING_LENGTH_SUM},
        {"字符串分类", MixedOperationFactory::STRING_CATEGORY}
    };
    
    std::vector<MixedProcessorBenchmark::BenchmarkResult> results;
    std::vector<std::string> test_names;
    
    for (const auto& op : benchmark_ops) {
        const std::string& name = op.first;
        MixedOperationFactory::OperationType op_type = op.second;
        std::cout << "\n测试 " << name << "..." << std::endl;
        
        auto result = MixedProcessorBenchmark::benchmark(*processor, op_type, num_iterations);
        results.push_back(result);
        test_names.push_back(name);
        
        std::cout << "  平均时间: " << result.avg_compute_time_ms << " ms" << std::endl;
        std::cout << "  吞吐量: " << result.throughput_ops_per_sec << " ops/s" << std::endl;
    }
    
    // 显示比较结果
    MixedProcessorBenchmark::compareBenchmarks(results, test_names);
}

void demonstrateBuilderPattern() {
    std::cout << "\n=== 构建器模式演示 ===" << std::endl;
    
    const size_t num_elements = 10000;
    
    // 使用构建器模式创建处理器
    auto prices = generateRandomFloats(num_elements, 10.0f, 500.0f);
    auto categories = generateRandomStrings(num_elements);
    auto ratings = generateRandomFloats(num_elements, 1.0f, 5.0f);
    
    auto processor = MixedProcessorBuilder(num_elements)
        .addFloatColumn(prices)
        .addStringColumn(categories)
        .addFloatColumn(ratings)
        .build();
    
    std::cout << "使用构建器模式创建了混合处理器" << std::endl;
    processor->printColumnInfo();
    
    // 执行计算
    processor->compute(MixedOperationFactory::CONDITIONAL_MIXED);
    auto results = processor->getResult();
    
    std::cout << "计算结果示例: ";
    for (int i = 0; i < 5; ++i) {
        std::cout << std::fixed << std::setprecision(2) << results[i];
        if (i < 4) std::cout << ", ";
    }
    std::cout << std::endl;
}

int main() {
    std::cout << "ZMM 混合类型列数据处理框架演示" << std::endl;
    std::cout << "====================================" << std::endl;
    
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
    std::cout << "全局内存: " << prop.totalGlobalMem / (1024*1024*1024) << " GB" << std::endl;
    
    try {
        // 运行各种演示
        demonstrateBasicUsage();
        demonstrateOperations();
        demonstrateNamedAccess();
        demonstrateEcommerceScenario();
        demonstratePerformanceBenchmark();
        demonstrateBuilderPattern();
        
        std::cout << "\n=== 所有演示完成 ===" << std::endl;
        std::cout << "混合类型处理框架演示成功完成！" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "演示过程中发生错误: " << e.what() << std::endl;
        return -1;
    } catch (...) {
        std::cerr << "演示过程中发生未知错误！" << std::endl;
        return -1;
    }
    
    return 0;
}
