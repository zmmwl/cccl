#include "mixed_column_processor.cuh"
#include "mixed_operations.cuh"
#include <iostream>
#include <vector>
#include <string>
#include <cmath>

using namespace zmm;

static bool approx_equal(float a, float b, float eps = 1e-3f) {
    return std::fabs(a - b) <= eps * (1.0f + std::fabs(a) + std::fabs(b));
}

int main() {
    const size_t num_elements = 2048;

    // 构造数据
    std::vector<float> prices(num_elements);
    std::vector<float> ratings(num_elements);
    std::vector<std::string> categories(num_elements);
    std::vector<std::string> brands(num_elements);

    for (size_t i = 0; i < num_elements; ++i) {
        prices[i] = 20.0f + static_cast<float>(i % 64) * 0.75f;
        ratings[i] = 2.0f + static_cast<float>(i % 5) * 0.5f;
        categories[i] = (i % 4 == 0) ? std::string("electronics") : std::string("books");
        brands[i] = (i % 3 == 0) ? std::string("premium") : std::string("generic");
    }

    auto processor = createMixedProcessor(num_elements);
    processor->addNamedFloatColumn("price", prices);
    processor->addNamedFloatColumn("rating", ratings);
    processor->addNamedStringColumn("category", categories);
    processor->addNamedStringColumn("brand", brands);

    // 1) 测试 NamedPriceRatingSumOperation（类接口路径）
    NamedPriceRatingSumOperation op_sum;
    if (!processor->compute(op_sum)) {
        std::cerr << "compute(NamedPriceRatingSumOperation) failed" << std::endl;
        return 1;
    }
    processor->synchronize();
    std::vector<float> out_sum = processor->getResult();
    // 验证
    int mismatches = 0;
    for (size_t i = 0; i < num_elements; ++i) {
        float expected = prices[i] + ratings[i];
        if (!approx_equal(out_sum[i], expected)) {
            if (mismatches < 8) {
                std::cerr << "Sum mismatch @" << i << ": got=" << out_sum[i]
                          << ", expected=" << expected << std::endl;
            }
            ++mismatches;
        }
    }
    if (mismatches) {
        std::cerr << "NamedPriceRatingSumOperation mismatches: " << mismatches << std::endl;
        return 1;
    }

    // 2) 测试 NamedEcommerceScoreOperation（类接口路径）
    NamedEcommerceScoreOperation op_score;
    if (!processor->compute(op_score)) {
        std::cerr << "compute(NamedEcommerceScoreOperation) failed" << std::endl;
        return 1;
    }
    processor->synchronize();
    std::vector<float> out_score = processor->getResult();
    mismatches = 0;
    for (size_t i = 0; i < num_elements; ++i) {
        float price = prices[i];
        float rating = ratings[i];
        const std::string& cat = categories[i];
        const std::string& br = brands[i];
        float category_multiplier = (cat == "electronics") ? 1.2f : (cat == "luxury" ? 1.5f : 1.0f);
        float brand_bonus = (br == "premium") ? 10.0f : 0.0f;
        float base_score = rating * 7.0f;
        float price_factor = (price > 0) ? (100.0f / price) : 0.0f;
        float expected = (base_score + price_factor * 2.0f + brand_bonus) * category_multiplier;
        if (!approx_equal(out_score[i], expected, 1e-2f)) {
            if (mismatches < 8) {
                std::cerr << "Score mismatch @" << i << ": got=" << out_score[i]
                          << ", expected=" << expected << std::endl;
            }
            ++mismatches;
        }
    }
    if (mismatches) {
        std::cerr << "NamedEcommerceScoreOperation mismatches: " << mismatches << std::endl;
        return 1;
    }

    std::cout << "mixed_named_ops_test passed (" << num_elements << " rows)" << std::endl;
    return 0;
}


