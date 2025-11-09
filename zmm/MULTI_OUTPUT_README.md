# ZMM 多列多类型输出功能文档

## 概述

本文档介绍ZMM框架的多列多类型输出功能。此功能允许CUDA内核操作返回多个不同类型的列，而不仅仅是单个float列。

## 新增功能

### 1. 多列输出支持

之前的版本只支持单列float类型的输出：
```cpp
// 旧版本 - 只能返回单个float
__device__ float operator()(const MixedRowData& row) const {
    return some_calculation;
}
```

新版本支持多列多类型输出：
```cpp
// 新版本 - 可以返回多列不同类型
__device__ void operator()(const MixedRowData& row, MultiColumnOutput& output) const {
    output.setFloat(0, price);
    output.setInt(1, quantity);
    output.setDouble(2, precision_value);
}
```

### 2. 支持的输出类型

- `ColumnDataType::FLOAT` - 单精度浮点数
- `ColumnDataType::INT` - 整数
- `ColumnDataType::DOUBLE` - 双精度浮点数

### 3. 核心组件

#### MultiColumnOutput 结构

用于在设备端写入多列输出：

```cpp
struct MultiColumnOutput {
    void** output_ptrs;             // 输出列的指针数组
    ColumnDataType* output_types;   // 输出列的类型数组
    int num_output_columns;        // 输出列数量
    size_t row_index;              // 当前行索引
    
    // 设置不同类型的值
    __device__ void setFloat(int column_index, float value);
    __device__ void setInt(int column_index, int value);
    __device__ void setDouble(int column_index, double value);
};
```

#### IMultiColumnOperation 接口

新的多列输出操作基类：

```cpp
class IMultiColumnOperation {
public:
    virtual int getNumOutputColumns() const = 0;
    virtual void getOutputTypes(ColumnDataType* types) const = 0;
    virtual __device__ void execute(const MixedRowData& row, MultiColumnOutput& output) const = 0;
    virtual const char* getName() const = 0;
};
```

## 使用示例

### 示例1: 价格和评分分离

将输入的价格和评分分别输出到两个float列：

```cpp
struct PriceRatingSplitFunctor {
    __device__ void operator()(const MixedRowData& row, MultiColumnOutput& output) const {
        float price = row.getFloat("price");
        float rating = row.getFloat("rating");
        
        output.setFloat(0, price);   // 第0列：价格
        output.setFloat(1, rating);  // 第1列：评分
    }
};

// 使用
auto processor = createMixedProcessor(num_elements);
processor->addNamedFloatColumn("price", prices);
processor->addNamedFloatColumn("rating", ratings);

std::vector<ColumnDataType> output_types = {
    ColumnDataType::FLOAT,  // 价格
    ColumnDataType::FLOAT   // 评分
};

processor->computeWithMultiOutput(PriceRatingSplitFunctor{}, output_types);

// 获取结果
auto price_out = processor->getOutputFloatColumn(0);
auto rating_out = processor->getOutputFloatColumn(1);
```

### 示例2: 多统计值计算

计算多个浮点列的总和、最大值、最小值：

```cpp
struct MultiStatisticsFunctor {
    __device__ void operator()(const MixedRowData& row, MultiColumnOutput& output) const {
        float sum = 0.0f;
        float max_val = -1e30f;
        float min_val = 1e30f;
        
        for (int i = 0; i < row.num_columns; ++i) {
            if (row.types[i] == ColumnDataType::FLOAT) {
                float val = row.getFloat(i);
                sum += val;
                max_val = (val > max_val) ? val : max_val;
                min_val = (val < min_val) ? val : min_val;
            }
        }
        
        output.setFloat(0, sum);
        output.setFloat(1, max_val);
        output.setFloat(2, min_val);
    }
};

// 使用
std::vector<ColumnDataType> output_types = {
    ColumnDataType::FLOAT,  // 总和
    ColumnDataType::FLOAT,  // 最大值
    ColumnDataType::FLOAT   // 最小值
};

processor->computeWithMultiOutput(MultiStatisticsFunctor{}, output_types);

auto sum_out = processor->getOutputFloatColumn(0);
auto max_out = processor->getOutputFloatColumn(1);
auto min_out = processor->getOutputFloatColumn(2);
```

### 示例3: 混合类型输出

输出不同类型的列（float和int）：

```cpp
struct PriceQuantityFunctor {
    __device__ void operator()(const MixedRowData& row, MultiColumnOutput& output) const {
        float price = row.getFloat("price");
        float quantity = row.getFloat("quantity");
        
        output.setFloat(0, price * quantity);          // 总价（float）
        output.setInt(1, static_cast<int>(quantity));  // 数量（int）
    }
};

// 使用
std::vector<ColumnDataType> output_types = {
    ColumnDataType::FLOAT,  // 总价
    ColumnDataType::INT     // 数量
};

processor->computeWithMultiOutput(PriceQuantityFunctor{}, output_types);

auto total_price = processor->getOutputFloatColumn(0);
auto quantity_int = processor->getOutputIntColumn(1);
```

### 示例4: 多维度评分分析

输出多个评分维度：

```cpp
struct MultiDimensionScoreFunctor {
    __device__ void operator()(const MixedRowData& row, MultiColumnOutput& output) const {
        float price = row.getFloat("price");
        float rating = row.getFloat("rating");
        GPUString category = row.getString("category");
        GPUString brand = row.getString("brand");
        
        float base_score = rating * 7.0f;
        float price_score = (price > 0) ? (100.0f / price) * 2.0f : 0.0f;
        
        float brand_score = 0.0f;
        if (gpu_string_equals(brand, "premium")) brand_score = 10.0f;
        else if (gpu_string_equals(brand, "popular")) brand_score = 5.0f;
        
        float category_multiplier = 1.0f;
        if (gpu_string_equals(category, "electronics")) category_multiplier = 1.2f;
        else if (gpu_string_equals(category, "luxury")) category_multiplier = 1.5f;
        
        float total_score = (base_score + price_score + brand_score) * category_multiplier;
        
        output.setFloat(0, base_score);
        output.setFloat(1, price_score);
        output.setFloat(2, brand_score);
        output.setFloat(3, total_score);
    }
};

// 使用
std::vector<ColumnDataType> output_types = {
    ColumnDataType::FLOAT,  // 基础评分
    ColumnDataType::FLOAT,  // 价格评分
    ColumnDataType::FLOAT,  // 品牌评分
    ColumnDataType::FLOAT   // 综合评分
};

processor->computeWithMultiOutput(MultiDimensionScoreFunctor{}, output_types);
```

## API 参考

### MixedColumnProcessor 新增方法

#### computeWithMultiOutput

```cpp
template<typename FunctorType>
bool computeWithMultiOutput(
    FunctorType functor,
    const std::vector<ColumnDataType>& output_types
);
```

使用多列输出的Functor执行计算。

**参数:**
- `functor`: 实现多列输出逻辑的函数对象
- `output_types`: 输出列的类型列表

**返回:** 计算是否成功

#### getMultiColumnResult

```cpp
MultiColumnResult getMultiColumnResult() const;
```

一次性获取所有输出列的结果。

**返回:** 包含所有输出列的结果结构

#### getOutputFloatColumn

```cpp
std::vector<float> getOutputFloatColumn(int output_column_index) const;
```

获取指定索引的float类型输出列。

**参数:**
- `output_column_index`: 输出列索引（从0开始）

**返回:** float向量

#### getOutputIntColumn

```cpp
std::vector<int> getOutputIntColumn(int output_column_index) const;
```

获取指定索引的int类型输出列。

#### getOutputDoubleColumn

```cpp
std::vector<double> getOutputDoubleColumn(int output_column_index) const;
```

获取指定索引的double类型输出列。

### MultiColumnResult 结构

```cpp
struct MultiColumnResult {
    std::vector<std::vector<float>> float_columns;
    std::vector<std::vector<int>> int_columns;
    std::vector<std::vector<double>> double_columns;
    std::vector<ColumnDataType> column_types;
    size_t num_elements;
    int num_columns;
};
```

## 向后兼容性

所有原有的单列float输出功能保持不变：

```cpp
// 原有的单列输出仍然完全支持
struct OldStyleFunctor {
    __device__ float operator()(const MixedRowData& row) const {
        return row.getFloat(0) + row.getFloat(1);
    }
};

processor->computeWithFunctor(OldStyleFunctor{});
auto result = processor->getResult();  // 仍然有效
```

## 预定义操作

框架提供了一些预定义的多列输出操作：

1. **PriceRatingSplitOperation** - 价格和评分分离
2. **MultiStatisticsOperation** - 多统计值（和、最大、最小）
3. **PriceQuantityOperation** - 价格数量混合类型输出
4. **MultiDimensionScoreOperation** - 多维度评分分析

## 编译和测试

### 编译多列输出测试

```bash
cd /path/to/cccl-zmm-v2.8.5/zmm
chmod +x build_multi_output_test.sh
./build_multi_output_test.sh
```

### 运行测试

```bash
cd build_multi_output
./multi_output_test
```

## 性能考虑

1. **内存分配**: 每个输出列都需要额外的GPU内存
2. **数据传输**: 多列输出需要更多的设备到主机数据传输
3. **类型转换**: 不同类型的输出可能涉及类型转换开销

## 最佳实践

1. **合理设计输出列**: 只输出必要的列，避免不必要的内存和带宽开销
2. **批量获取结果**: 使用 `getMultiColumnResult()` 而不是多次调用单列获取函数
3. **类型匹配**: 确保 `output_types` 与实际的 `setXxx()` 调用匹配
4. **错误检查**: 检查 `computeWithMultiOutput()` 的返回值

## 示例代码

完整的示例代码请参见 `multi_output_test.cu`。

## 故障排查

### 常见问题

1. **编译错误**: 确保使用 `-std=c++17` 和 `--extended-lambda` 编译选项
2. **运行时错误**: 检查输出类型是否匹配，以及列索引是否在有效范围内
3. **结果不正确**: 验证Functor中的逻辑和输出列索引

## 更新日志

### Version 2.8.5+

- ✅ 添加多列多类型输出支持
- ✅ 新增 `MultiColumnOutput` 结构
- ✅ 新增 `IMultiColumnOperation` 接口
- ✅ 新增 `computeWithMultiOutput()` 方法
- ✅ 新增多种获取结果的方法
- ✅ 保持向后兼容性
- ✅ 添加预定义多列输出操作
- ✅ 添加完整的测试用例

## 相关文档

- [MIXED_TYPES_README.md](MIXED_TYPES_README.md) - 混合类型系统文档
- [GROUPBY_README.md](GROUPBY_README.md) - GroupBy功能文档
- [README.md](README.md) - 主要文档

## 联系和支持

如有问题或建议，请提交Issue或Pull Request。

