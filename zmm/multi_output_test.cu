#include "mixed_column_processor.cuh"
#include "mixed_operations.cuh"
#include <iostream>
#include <vector>
#include <iomanip>
#include <random>

using namespace zmm;

// 生成随机浮点数
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

// 生成随机字符串
std::vector<std::string> generateRandomStrings(size_t size) {
    std::vector<std::string> data(size);
    std::vector<std::string> categories = {"electronics", "books", "clothing", "luxury"};
    std::vector<std::string> brands = {"premium", "popular", "generic"};
    
    std::random_device rd;
    std::mt19937 gen(rd());
    
    for (size_t i = 0; i < size; ++i) {
        if (i % 2 == 0) {
            data[i] = categories[gen() % categories.size()];
        } else {
            data[i] = brands[gen() % brands.size()];
        }
    }
    return data;
}

// 测试1: 价格和评分分离输出
void testPriceRatingSplit() {
    std::cout << "\n=== 测试1: 价格和评分分离输出 ===" << std::endl;
    
    const size_t num_elements = 1000;
    
    // 准备输入数据
    auto prices = generateRandomFloats(num_elements, 10.0f, 1000.0f);
    auto ratings = generateRandomFloats(num_elements, 1.0f, 5.0f);
    
    // 创建处理器
    auto processor = createMixedProcessor(num_elements);
    processor->addNamedFloatColumn("price", prices);
    processor->addNamedFloatColumn("rating", ratings);
    
    // 定义输出类型
    std::vector<ColumnDataType> output_types = {
        ColumnDataType::FLOAT,  // 价格
        ColumnDataType::FLOAT   // 评分
    };
    
    // 执行多列输出计算
    bool success = processor->computeWithMultiOutput(PriceRatingSplitFunctor{}, output_types);
    
    if (success) {
        // 获取结果
        auto price_out = processor->getOutputFloatColumn(0);
        auto rating_out = processor->getOutputFloatColumn(1);
        
        std::cout << "处理成功，元素数量: " << num_elements << std::endl;
        std::cout << "\n前10个结果:" << std::endl;
        std::cout << std::setw(8) << "索引" << std::setw(15) << "价格(输入)" 
                  << std::setw(15) << "价格(输出)" << std::setw(15) << "评分(输入)" 
                  << std::setw(15) << "评分(输出)" << std::endl;
        std::cout << std::string(68, '-') << std::endl;
        
        for (int i = 0; i < 10; ++i) {
            std::cout << std::setw(8) << i 
                      << std::fixed << std::setprecision(2)
                      << std::setw(15) << prices[i]
                      << std::setw(15) << price_out[i]
                      << std::setw(15) << ratings[i]
                      << std::setw(15) << rating_out[i]
                      << std::endl;
        }
        
        // 验证结果
        bool all_match = true;
        for (size_t i = 0; i < num_elements; ++i) {
            if (std::abs(prices[i] - price_out[i]) > 1e-3f || 
                std::abs(ratings[i] - rating_out[i]) > 1e-3f) {
                all_match = false;
                break;
            }
        }
        
        if (all_match) {
            std::cout << "\n✓ 验证成功：所有结果匹配！" << std::endl;
        } else {
            std::cout << "\n✗ 验证失败：存在不匹配的结果！" << std::endl;
        }
    } else {
        std::cout << "计算失败！" << std::endl;
    }
}

// 测试2: 多统计值输出
void testMultiStatistics() {
    std::cout << "\n=== 测试2: 多统计值输出 (总和、最大值、最小值) ===" << std::endl;
    
    const size_t num_elements = 1000;
    
    // 准备输入数据 - 多个浮点列
    auto data1 = generateRandomFloats(num_elements, 1.0f, 100.0f);
    auto data2 = generateRandomFloats(num_elements, 1.0f, 100.0f);
    auto data3 = generateRandomFloats(num_elements, 1.0f, 100.0f);
    
    // 创建处理器
    auto processor = createMixedProcessor(num_elements);
    processor->addFloatColumn(data1);
    processor->addFloatColumn(data2);
    processor->addFloatColumn(data3);
    
    // 定义输出类型
    std::vector<ColumnDataType> output_types = {
        ColumnDataType::FLOAT,  // 总和
        ColumnDataType::FLOAT,  // 最大值
        ColumnDataType::FLOAT   // 最小值
    };
    
    // 执行多列输出计算
    bool success = processor->computeWithMultiOutput(MultiStatisticsFunctor{}, output_types);
    
    if (success) {
        // 获取结果
        auto sum_out = processor->getOutputFloatColumn(0);
        auto max_out = processor->getOutputFloatColumn(1);
        auto min_out = processor->getOutputFloatColumn(2);
        
        std::cout << "处理成功，元素数量: " << num_elements << std::endl;
        std::cout << "\n前10个结果:" << std::endl;
        std::cout << std::setw(8) << "索引" << std::setw(12) << "列1" 
                  << std::setw(12) << "列2" << std::setw(12) << "列3"
                  << std::setw(12) << "总和" << std::setw(12) << "最大值" 
                  << std::setw(12) << "最小值" << std::endl;
        std::cout << std::string(80, '-') << std::endl;
        
        for (int i = 0; i < 10; ++i) {
            std::cout << std::setw(8) << i 
                      << std::fixed << std::setprecision(2)
                      << std::setw(12) << data1[i]
                      << std::setw(12) << data2[i]
                      << std::setw(12) << data3[i]
                      << std::setw(12) << sum_out[i]
                      << std::setw(12) << max_out[i]
                      << std::setw(12) << min_out[i]
                      << std::endl;
        }
        
        // 验证结果
        bool all_match = true;
        for (size_t i = 0; i < num_elements; ++i) {
            float expected_sum = data1[i] + data2[i] + data3[i];
            float expected_max = std::max({data1[i], data2[i], data3[i]});
            float expected_min = std::min({data1[i], data2[i], data3[i]});
            
            if (std::abs(sum_out[i] - expected_sum) > 1e-3f ||
                std::abs(max_out[i] - expected_max) > 1e-3f ||
                std::abs(min_out[i] - expected_min) > 1e-3f) {
                all_match = false;
                std::cout << "不匹配 @" << i << ": sum=" << sum_out[i] 
                          << " (expected " << expected_sum << ")" << std::endl;
                break;
            }
        }
        
        if (all_match) {
            std::cout << "\n✓ 验证成功：所有统计值正确！" << std::endl;
        } else {
            std::cout << "\n✗ 验证失败：存在不匹配的统计值！" << std::endl;
        }
    } else {
        std::cout << "计算失败！" << std::endl;
    }
}

// 测试3: 多维度电商评分输出
void testMultiDimensionScore() {
    std::cout << "\n=== 测试3: 多维度电商评分输出 ===" << std::endl;
    
    const size_t num_elements = 500;
    
    // 准备输入数据
    auto prices = generateRandomFloats(num_elements, 10.0f, 500.0f);
    auto ratings = generateRandomFloats(num_elements, 1.0f, 5.0f);
    
    std::vector<std::string> categories(num_elements);
    std::vector<std::string> brands(num_elements);
    std::vector<std::string> cat_options = {"electronics", "books", "clothing", "luxury"};
    std::vector<std::string> brand_options = {"premium", "popular", "generic"};
    
    std::random_device rd;
    std::mt19937 gen(rd());
    
    for (size_t i = 0; i < num_elements; ++i) {
        categories[i] = cat_options[gen() % cat_options.size()];
        brands[i] = brand_options[gen() % brand_options.size()];
    }
    
    // 创建处理器
    auto processor = createMixedProcessor(num_elements);
    processor->addNamedFloatColumn("price", prices);
    processor->addNamedFloatColumn("rating", ratings);
    processor->addNamedStringColumn("category", categories);
    processor->addNamedStringColumn("brand", brands);
    
    // 定义输出类型
    std::vector<ColumnDataType> output_types = {
        ColumnDataType::FLOAT,  // 基础评分
        ColumnDataType::FLOAT,  // 价格评分
        ColumnDataType::FLOAT,  // 品牌评分
        ColumnDataType::FLOAT   // 综合评分
    };
    
    // 执行多列输出计算
    bool success = processor->computeWithMultiOutput(MultiDimensionScoreFunctor{}, output_types);
    
    if (success) {
        // 获取结果
        auto base_score = processor->getOutputFloatColumn(0);
        auto price_score = processor->getOutputFloatColumn(1);
        auto brand_score = processor->getOutputFloatColumn(2);
        auto total_score = processor->getOutputFloatColumn(3);
        
        std::cout << "处理成功，元素数量: " << num_elements << std::endl;
        std::cout << "\n前15个结果:" << std::endl;
        std::cout << std::setw(5) << "idx" << std::setw(10) << "价格" 
                  << std::setw(10) << "评分" << std::setw(15) << "类别"
                  << std::setw(12) << "品牌" << std::setw(12) << "基础分"
                  << std::setw(12) << "价格分" << std::setw(12) << "品牌分" 
                  << std::setw(12) << "综合分" << std::endl;
        std::cout << std::string(100, '-') << std::endl;
        
        for (int i = 0; i < 15; ++i) {
            std::cout << std::setw(5) << i 
                      << std::fixed << std::setprecision(2)
                      << std::setw(10) << prices[i]
                      << std::setw(10) << ratings[i]
                      << std::setw(15) << categories[i]
                      << std::setw(12) << brands[i]
                      << std::setw(12) << base_score[i]
                      << std::setw(12) << price_score[i]
                      << std::setw(12) << brand_score[i]
                      << std::setw(12) << total_score[i]
                      << std::endl;
        }
        
        // 统计分析
        float avg_base = 0.0f, avg_price = 0.0f, avg_brand = 0.0f, avg_total = 0.0f;
        for (size_t i = 0; i < num_elements; ++i) {
            avg_base += base_score[i];
            avg_price += price_score[i];
            avg_brand += brand_score[i];
            avg_total += total_score[i];
        }
        avg_base /= num_elements;
        avg_price /= num_elements;
        avg_brand /= num_elements;
        avg_total /= num_elements;
        
        std::cout << "\n统计摘要:" << std::endl;
        std::cout << "  平均基础评分: " << avg_base << std::endl;
        std::cout << "  平均价格评分: " << avg_price << std::endl;
        std::cout << "  平均品牌评分: " << avg_brand << std::endl;
        std::cout << "  平均综合评分: " << avg_total << std::endl;
        
        std::cout << "\n✓ 多维度评分计算成功！" << std::endl;
    } else {
        std::cout << "计算失败！" << std::endl;
    }
}

// 测试4: 使用MultiColumnResult批量获取所有输出
void testBatchResultRetrieval() {
    std::cout << "\n=== 测试4: 批量获取多列输出结果 ===" << std::endl;
    
    const size_t num_elements = 100;
    
    // 准备输入数据
    auto prices = generateRandomFloats(num_elements, 10.0f, 100.0f);
    auto ratings = generateRandomFloats(num_elements, 1.0f, 5.0f);
    
    // 创建处理器
    auto processor = createMixedProcessor(num_elements);
    processor->addNamedFloatColumn("price", prices);
    processor->addNamedFloatColumn("rating", ratings);
    
    // 定义输出类型
    std::vector<ColumnDataType> output_types = {
        ColumnDataType::FLOAT,
        ColumnDataType::FLOAT
    };
    
    // 执行计算
    bool success = processor->computeWithMultiOutput(PriceRatingSplitFunctor{}, output_types);
    
    if (success) {
        // 使用MultiColumnResult一次性获取所有输出
        auto results = processor->getMultiColumnResult();
        
        std::cout << "输出列数量: " << results.num_columns << std::endl;
        std::cout << "输出元素数量: " << results.num_elements << std::endl;
        std::cout << "Float列数量: " << results.float_columns.size() << std::endl;
        std::cout << "Int列数量: " << results.int_columns.size() << std::endl;
        std::cout << "Double列数量: " << results.double_columns.size() << std::endl;
        
        std::cout << "\n前5个结果:" << std::endl;
        for (int i = 0; i < 5; ++i) {
            std::cout << "  Row " << i << ": ";
            for (size_t col = 0; col < results.float_columns.size(); ++col) {
                std::cout << "col" << col << "=" << results.float_columns[col][i] << " ";
            }
            std::cout << std::endl;
        }
        
        std::cout << "\n✓ 批量结果获取成功！" << std::endl;
    } else {
        std::cout << "计算失败！" << std::endl;
    }
}

int main() {
    std::cout << "======================================================" << std::endl;
    std::cout << "ZMM 多列多类型输出功能测试" << std::endl;
    std::cout << "======================================================" << std::endl;
    
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
        // 运行所有测试
        testPriceRatingSplit();
        testMultiStatistics();
        testMultiDimensionScore();
        testBatchResultRetrieval();
        
        std::cout << "\n======================================================" << std::endl;
        std::cout << "所有测试完成！" << std::endl;
        std::cout << "======================================================" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "测试过程中发生错误: " << e.what() << std::endl;
        return -1;
    } catch (...) {
        std::cerr << "测试过程中发生未知错误！" << std::endl;
        return -1;
    }
    
    return 0;
}

