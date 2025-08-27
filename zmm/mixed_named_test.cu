#include "mixed_column_processor.cuh"
#include <iostream>
#include <vector>
#include <string>
#include <cmath>

using namespace zmm;

struct NamedAccessFunctor {
    __device__ float operator()(const MixedRowData& row) const {
        float price = row.getFloat("price");
        float rating = row.getFloat("rating");
        GPUString cat = row.getString("category");
        // 使用名称访问的混合计算：price + rating + category.length
        return price + rating + static_cast<float>(cat.length);
    }
};

int main() {
    const size_t num_elements = 4096;

    // 构造测试数据
    std::vector<float> prices(num_elements);
    std::vector<float> ratings(num_elements);
    std::vector<std::string> categories(num_elements);

    for (size_t i = 0; i < num_elements; ++i) {
        prices[i] = 10.0f + static_cast<float>(i % 100) * 0.5f;
        ratings[i] = 1.0f + static_cast<float>(i % 5) * 0.25f;
        categories[i] = (i % 2 == 0) ? std::string("electronics") : std::string("books");
    }

    // 创建处理器并按名称添加列
    auto processor = createMixedProcessor(num_elements);
    if (processor->addNamedFloatColumn("price", prices) < 0) {
        std::cerr << "Failed to add named float column: price" << std::endl;
        return 1;
    }
    if (processor->addNamedFloatColumn("rating", ratings) < 0) {
        std::cerr << "Failed to add named float column: rating" << std::endl;
        return 1;
    }
    if (processor->addNamedStringColumn("category", categories) < 0) {
        std::cerr << "Failed to add named string column: category" << std::endl;
        return 1;
    }

    // 使用基于字段名访问的Functor执行计算
    if (!processor->computeWithFunctor(NamedAccessFunctor{})) {
        std::cerr << "computeWithFunctor failed" << std::endl;
        return 1;
    }
    processor->synchronize();

    // 获取结果并验证
    std::vector<float> result = processor->getResult();
    if (result.size() != num_elements) {
        std::cerr << "Unexpected result size" << std::endl;
        return 1;
    }

    int mismatches = 0;
    for (size_t i = 0; i < num_elements; ++i) {
        float expected = prices[i] + ratings[i] + static_cast<float>(categories[i].size());
        float diff = std::fabs(result[i] - expected);
        if (diff > 1e-3f) {
            if (mismatches < 10) {
                std::cerr << "Mismatch at " << i << ": got=" << result[i]
                          << ", expected=" << expected << ", diff=" << diff << std::endl;
            }
            mismatches++;
        }
    }

    if (mismatches > 0) {
        std::cerr << "Test failed with " << mismatches << " mismatches" << std::endl;
        return 1;
    }

    std::cout << "mixed_named_test passed (" << num_elements << " rows)" << std::endl;
    return 0;
}


