# ZMM GroupBy 功能快速入门

## 5分钟快速上手

### 第1步：编译和测试

```bash
cd /mnt/c/dev/ml/cuda/cccl-zmm-v2.8.5/zmm

# 运行自动构建和测试脚本
chmod +x build_and_test_groupby.sh
./build_and_test_groupby.sh
```

### 第2步：查看简单示例

最简单的使用示例：

```cpp
#include "mixed_column_processor.cuh"
using namespace zmm;

int main() {
    // 数据
    std::vector<int> ids = {1, 2, 1, 2, 1};          // 分组键
    std::vector<float> values = {10, 20, 30, 40, 50}; // 数值
    
    // 创建处理器
    auto processor = createMixedProcessor(5);
    int id_col = processor->addIntColumn(ids);
    processor->addFloatColumn(values);
    
    // 计算（这里简单返回原值）
    struct PassThrough {
        __device__ float operator()(const MixedRowData& row) const {
            return row.getFloat(1);
        }
    };
    processor->computeWithFunctor(PassThrough{});
    
    // 分组求和
    auto result = processor->groupBySum(id_col);
    
    // 输出结果
    // ID 1: Sum = 90.0 (10 + 30 + 50)
    // ID 2: Sum = 60.0 (20 + 40)
    for (size_t i = 0; i < result.num_groups; ++i) {
        std::cout << "ID " << result.unique_keys[i] 
                  << ": Sum = " << result.aggregated_values[i] << std::endl;
    }
    
    return 0;
}
```

### 第3步：运行现有测试

```bash
cd build_groupby

# 运行简单测试（20行数据，3个分组）
./bin/groupby_simple_test

# 运行完整示例（包含大数据测试）
./bin/groupby_example
```

## 常见使用场景

### 场景1：计算后分组求和

```cpp
// 有 ID、价格、数量三列，计算每个ID的总销售额
auto processor = createMixedProcessor(num_rows);
int id_col = processor->addIntColumn(ids);
processor->addFloatColumn(prices);
processor->addFloatColumn(quantities);

// 计算销售额 = 价格 × 数量
struct SalesAmount {
    __device__ float operator()(const MixedRowData& row) const {
        return row.getFloat(1) * row.getFloat(2);  // price * qty
    }
};

processor->computeWithFunctor(SalesAmount{});
auto result = processor->groupBySum(id_col);
```

### 场景2：多种聚合统计

```cpp
// 对同一数据进行多种聚合
auto sum_result = processor->groupByAggregate(id_col, AggregationType::SUM);
auto max_result = processor->groupByAggregate(id_col, AggregationType::MAX);
auto min_result = processor->groupByAggregate(id_col, AggregationType::MIN);

// 输出对比
for (size_t i = 0; i < sum_result.num_groups; ++i) {
    std::cout << "ID " << sum_result.unique_keys[i]
              << ": SUM=" << sum_result.aggregated_values[i]
              << ", MAX=" << max_result.aggregated_values[i]
              << ", MIN=" << min_result.aggregated_values[i]
              << std::endl;
}
```

### 场景3：直接对列数据分组

```cpp
// 不需要复杂计算，直接对某列分组求和
auto processor = createMixedProcessor(num_rows);
int id_col = processor->addIntColumn(ids);
processor->addFloatColumn(amounts);

// 简单传递
struct PassThrough {
    __device__ float operator()(const MixedRowData& row) const {
        return row.getFloat(1);  // 直接返回第二列
    }
};

processor->computeWithFunctor(PassThrough{});
auto result = processor->groupBySum(id_col);
```

## 快速参考

### API 速查

```cpp
// 添加整数列（用于分组）
int id_col = processor->addIntColumn(ids);

// 执行计算
processor->computeWithFunctor(YourFunctor{});

// 分组求和
auto result = processor->groupBySum(id_col);

// 分组聚合（指定类型）
auto result = processor->groupByAggregate(id_col, AggregationType::SUM);
```

### 结果结构

```cpp
struct GroupByResult {
    std::vector<int> unique_keys;           // 唯一的ID
    std::vector<float> aggregated_values;   // 聚合后的值
    size_t num_groups;                      // 分组数量
};
```

### 聚合类型

```cpp
AggregationType::SUM    // 求和
AggregationType::MAX    // 最大值
AggregationType::MIN    // 最小值
```

## 常见错误和解决

### 错误1：必须先 compute 再 groupBy

```cpp
❌ 错误:
auto result = processor->groupBySum(id_col);  // 还没有计算结果！

✅ 正确:
processor->computeWithFunctor(MyFunctor{});
auto result = processor->groupBySum(id_col);
```

### 错误2：分组列必须是 INT 类型

```cpp
❌ 错误:
int float_col = processor->addFloatColumn(values);
auto result = processor->groupBySum(float_col);  // 类型错误！

✅ 正确:
int id_col = processor->addIntColumn(ids);
auto result = processor->groupBySum(id_col);
```

### 错误3：列索引错误

```cpp
❌ 错误:
auto result = processor->groupBySum(999);  // 列不存在！

✅ 正确:
int id_col = processor->addIntColumn(ids);  // 记住列索引
auto result = processor->groupBySum(id_col);
```

## 性能建议

### ✅ 推荐做法

```cpp
// 1. 使用连续的整数ID
std::vector<int> ids = {0, 1, 2, 0, 1, 2, ...};  // 好

// 2. 控制分组数量
int num_groups = 100;  // 好 (< 10000)

// 3. 批量处理
processor->addIntColumn(ids);
processor->addFloatColumn(v1);
processor->addFloatColumn(v2);
// ... 一次性添加所有列
```

### ❌ 避免做法

```cpp
// 1. 避免稀疏的ID
std::vector<int> ids = {1, 10000, 1000000, ...};  // 不好

// 2. 避免过多分组
int num_groups = 100000;  // 不好 (> 10000)

// 3. 避免频繁分组
for (int i = 0; i < 1000; ++i) {
    processor->groupBySum(id_col);  // 不好：重复分配内存
}
```

## 下一步

### 深入学习
- 阅读完整文档：`cat GROUPBY_README.md`
- 查看实现总结：`cat GROUPBY_SUMMARY.md`
- 研究示例代码：`groupby_example.cu`

### 实践练习
1. 修改 `groupby_simple_test.cu`，尝试不同的计算逻辑
2. 使用自己的数据集测试性能
3. 尝试不同的聚合类型（SUM, MAX, MIN）

### 获取帮助
- 查看代码注释
- 参考混合类型文档：`MIXED_TYPES_README.md`
- 检查 CUB 文档：https://nvlabs.github.io/cub/

## 完整示例程序

保存为 `my_groupby_test.cu`：

```cpp
#include "mixed_column_processor.cuh"
#include <iostream>
#include <vector>

using namespace zmm;

int main() {
    // 模拟订单数据：商品ID、单价、数量
    std::vector<int> product_ids = {
        101, 102, 101, 103, 102,
        101, 103, 102, 101, 103
    };
    
    std::vector<float> prices = {
        9.99f, 19.99f, 9.99f, 29.99f, 19.99f,
        9.99f, 29.99f, 19.99f, 9.99f, 29.99f
    };
    
    std::vector<float> quantities = {
        2.0f, 1.0f, 3.0f, 1.0f, 2.0f,
        1.0f, 1.0f, 1.0f, 2.0f, 2.0f
    };
    
    // 创建处理器
    auto processor = createMixedProcessor(10);
    int product_col = processor->addIntColumn(product_ids);
    processor->addFloatColumn(prices);
    processor->addFloatColumn(quantities);
    
    // 计算订单金额 = 单价 × 数量
    struct OrderAmount {
        __device__ float operator()(const MixedRowData& row) const {
            float price = row.getFloat(1);
            float qty = row.getFloat(2);
            return price * qty;
        }
    };
    
    // 执行计算
    processor->computeWithFunctor(OrderAmount{});
    processor->synchronize();
    
    // 按商品ID统计总销售额
    auto sales_by_product = processor->groupBySum(product_col);
    
    // 输出结果
    std::cout << "各商品销售统计:" << std::endl;
    std::cout << "商品ID   总销售额" << std::endl;
    std::cout << "-------------------" << std::endl;
    for (size_t i = 0; i < sales_by_product.num_groups; ++i) {
        std::cout << sales_by_product.unique_keys[i] << "      "
                  << sales_by_product.aggregated_values[i] << std::endl;
    }
    
    return 0;
}
```

编译运行：

```bash
# 添加到 CMakeLists.txt:
add_executable(my_groupby_test my_groupby_test.cu)
target_link_libraries(my_groupby_test zmm_mixed_types CUDA::cudart)
target_compile_options(my_groupby_test PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

# 编译
cd build
cmake ..
make my_groupby_test

# 运行
./bin/my_groupby_test
```

预期输出：
```
各商品销售统计:
商品ID   总销售额
-------------------
101      69.93
102      79.95
103      119.96
```

---

🎉 恭喜！你已经掌握了 ZMM GroupBy 功能的基本使用！


