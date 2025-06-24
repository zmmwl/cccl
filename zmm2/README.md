# ZMM2 列数据计算框架

一个高性能的CUDA模板化计算框架，专门用于处理大规模列数据的算术表达式计算。

## 特性

- 🚀 **高性能**: 基于CUDA的并行计算，支持10亿+元素的处理
- 🔧 **模板化设计**: 支持任意数量的输入列（编译时确定）
- 💡 **灵活表达式**: 通过继承基类轻松实现自定义算术表达式
- 🧠 **智能内存管理**: 自动处理GPU内存分配和批处理
- 📊 **批处理支持**: 自动分批处理超大数据集，避免内存溢出
- ⚡ **异步执行**: 使用CUDA流进行异步数据传输和计算
- 🛡️ **错误处理**: 完善的错误检查和异常处理机制

## 系统要求

- CUDA 11.0 或更高版本
- C++17 兼容编译器
- CMake 3.18 或更高版本
- 支持的GPU架构: Compute Capability 6.0+

## 快速开始

### 1. 编译

```bash
cd zmm2
mkdir build
cd build
cmake ..
make -j$(nproc)
```

### 2. 运行简单示例

```bash
./simple_example
```

### 3. 运行完整测试

```bash
./test_framework
```

## 核心组件

### 1. ArithmeticExpression<N> 基类

所有算术表达式的基类，业务逻辑需要继承此类：

```cpp
template<int N>
class ArithmeticExpression {
public:
    __device__ virtual float compute(const float values[N]) const = 0;
};
```

### 2. ColumnComputeFramework<N> 主框架

主要的计算框架类，负责内存管理和CUDA内核执行：

```cpp
template<int N>
class ColumnComputeFramework {
public:
    explicit ColumnComputeFramework(size_t num_elements);
    void setInputData(const std::vector<const float*>& host_columns);
    template<typename ExpressionType>
    void compute(const ExpressionType& expression);
    void getResult(float* host_output);
};
```

### 3. BatchProcessor<N> 批处理器

用于处理超大数据量的分批计算：

```cpp
template<int N>
class BatchProcessor {
public:
    BatchProcessor(size_t total_elements, size_t max_gpu_memory_mb = 4096);
    template<typename ExpressionType>
    void processBatches(/* parameters */);
};
```

## 使用示例

### 基本使用

```cpp
#include "column_compute_framework.cuh"
#include "business_expressions.cuh"

// 准备数据
std::vector<float> col1 = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
std::vector<float> col2 = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f};
std::vector<float> col3 = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f};

std::vector<const float*> input_columns = {col1.data(), col2.data(), col3.data()};
std::vector<float> output(5);

// 创建框架和表达式
ColumnComputeFramework<3> framework(5);
ThreeColumnSum expression;

// 执行计算
framework.setInputData(input_columns);
framework.compute(expression);
framework.getResult(output.data());
```

### 自定义表达式

```cpp
class CustomExpression : public ArithmeticExpression<2> {
public:
    __device__ float compute(const float values[2]) const override {
        return values[0] * values[0] + values[1] * values[1]; // 平方和
    }
};
```

### 处理大数据量

```cpp
// 使用便利函数自动选择最优处理方式
processColumns<3>(input_columns, output.data(), num_elements, expression,
                 4096, // 最大GPU内存4GB
                 [](size_t processed, size_t total) {
                     std::cout << "进度: " << (processed * 100 / total) << "%" << std::endl;
                 });
```

## 预定义表达式

框架提供了多种预定义的业务表达式：

| 表达式类 | 列数 | 功能描述 |
|---------|------|----------|
| `ThreeColumnSum` | 3 | 三列相加 |
| `ThreeColumnWeightedSum` | 3 | 三列加权和 |
| `TwoColumnMax` | 2 | 两列取最大值 |
| `FiveColumnAverage` | 5 | 五列平均值 |
| `ComplexMathExpression` | 3 | 复杂数学表达式 |
| `ConditionalExpression` | 3 | 条件表达式 |
| `StatisticalExpression` | 4 | 统计分析表达式 |
| `StandardizationExpression` | 3 | 标准化(z-score) |
| `RiskAssessmentExpression` | 3 | 风险评估 |
| `PolynomialExpression` | 1 | 多项式表达式 |

## 性能优化建议

### 1. 内存优化
- 对于大数据量，使用批处理模式
- 合理设置`max_gpu_memory_mb`参数
- 使用`MemoryInfo`类查询GPU内存状态

### 2. 计算优化
- 在表达式中使用`#pragma unroll`优化循环
- 避免在device函数中使用复杂的分支逻辑
- 使用CUDA内置数学函数（如`fmaxf`, `sqrtf`等）

### 3. 数据传输优化
- 框架自动使用异步内存传输
- 多个小批次比一个大批次更容易管理内存

## API 参考

### ColumnComputeFramework<N>

#### 构造函数
```cpp
explicit ColumnComputeFramework(size_t num_elements)
```
- `num_elements`: 每列的元素数量

#### 方法

##### setInputData
```cpp
void setInputData(const std::vector<const float*>& host_columns)
```
设置输入数据，异步传输到GPU。

##### compute
```cpp
template<typename ExpressionType>
void compute(const ExpressionType& expression)
```
执行计算，表达式类型必须继承自`ArithmeticExpression<N>`。

##### getResult
```cpp
void getResult(float* host_output)
```
获取计算结果，从GPU传输到主机内存。

### BatchProcessor<N>

#### 构造函数
```cpp
BatchProcessor(size_t total_elements, size_t max_gpu_memory_mb = 4096)
```
- `total_elements`: 总元素数量
- `max_gpu_memory_mb`: 最大GPU内存使用量（MB）

#### 方法

##### processBatches
```cpp
template<typename ExpressionType>
void processBatches(
    const std::vector<const float*>& host_input_columns,
    float* host_output_column,
    const ExpressionType& expression,
    std::function<void(size_t, size_t)> progress_callback = nullptr
)
```
执行批量处理，可选择提供进度回调。

### 便利函数

##### processColumns
```cpp
template<int N, typename ExpressionType>
void processColumns(
    const std::vector<const float*>& host_input_columns,
    float* host_output_column,
    size_t total_elements,
    const ExpressionType& expression,
    size_t max_gpu_memory_mb = 4096,
    std::function<void(size_t, size_t)> progress_callback = nullptr
)
```
自动选择最优处理方式的便利函数。

## 错误处理

框架提供完善的错误处理机制：

- `std::invalid_argument`: 输入参数错误
- `std::runtime_error`: CUDA运行时错误
- 自动CUDA错误检查和报告

## 扩展开发

### 创建自定义表达式

1. 继承`ArithmeticExpression<N>`基类
2. 在`__device__`函数中实现`compute`方法
3. 确保所有成员变量可以安全地拷贝到GPU

```cpp
class MyCustomExpression : public ArithmeticExpression<4> {
private:
    float parameter_;

public:
    __host__ __device__ MyCustomExpression(float param) : parameter_(param) {}
    
    __device__ float compute(const float values[4]) const override {
        // 实现自定义计算逻辑
        return (values[0] + values[1]) * parameter_ + 
               (values[2] - values[3]) / parameter_;
    }
};
```

## 许可证

本项目采用 MIT 许可证。

## 贡献

欢迎提交问题和拉取请求！

## 联系

如有问题或建议，请通过GitHub Issues联系。 