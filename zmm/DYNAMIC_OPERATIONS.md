# ZMM CUDA 动态操作框架

## 概述

本文档介绍如何将 ZMM 框架中的操作（如 `AddOperation`、`MultiplyOperation`）编译为独立的动态链接库，实现运行时动态加载和执行。

## 动态链接的优势

1. **模块化**: 每个操作独立编译，便于维护和更新
2. **按需加载**: 只加载实际需要的操作，节省内存
3. **插件架构**: 支持第三方开发者编写自定义操作
4. **版本管理**: 可以独立更新特定操作而不影响整个系统

## 架构设计

### 核心组件

1. **操作接口 (`operations_interface.h`)**
   - `IOperation<N>`: 操作基类接口
   - `IOperationFactory<N>`: 操作工厂接口
   - 统一的动态库导出宏

2. **动态处理器 (`dynamic_processor.h`)**
   - 封装动态库加载逻辑
   - 管理操作生命周期
   - 提供统一的执行接口

3. **操作实现 (`operations_*.cu`)**
   - 具体操作的 CUDA 内核实现
   - 工厂模式创建操作实例
   - 标准化的导出接口

## 实现原理

### 1. 接口抽象

```cpp
template<int N>
class IOperation {
public:
    virtual void execute(
        float* const* input_columns,
        float* output_column,
        size_t num_elements
    ) = 0;
};
```

### 2. 工厂模式

```cpp
template<int N>
class IOperationFactory {
public:
    virtual std::unique_ptr<IOperation<N>> createOperation() = 0;
};
```

### 3. 动态库导出

```cpp
extern "C" {
    __attribute__((visibility("default")))
    IOperationFactory<N>* createOperationFactory() {
        return new ConcreteOperationFactory<N>();
    }
}
```

## 使用方法

### 1. 构建动态库

```bash
# 使用专用构建脚本
./build_dynamic.sh

# 或者手动构建
mkdir build_dynamic && cd build_dynamic
cmake .. -f ../CMakeLists_dynamic.txt
make -j
```

### 2. 运行时加载

```cpp
#include "dynamic_processor.h"

// 创建动态处理器
auto processor = createDynamicProcessor<3>(num_elements);

// 加载操作动态库
processor->loadOperation("./lib/liboperations_add.so", "AddOperation");
processor->loadOperation("./lib/liboperations_multiply.so", "MultiplyOperation");

// 设置数据
processor->setInputColumn(0, data1);
processor->setInputColumn(1, data2);
processor->setInputColumn(2, data3);

// 执行操作
processor->executeOperation("AddOperation");
processor->getResult(result.data());
```

## 技术细节

### CUDA 代码动态链接的挑战

1. **模板实例化**: CUDA 模板函数需要在编译时实例化
2. **设备代码链接**: `__device__` 函数不能在运行时动态链接
3. **内核启动**: 内核函数必须在编译时可见

### 解决方案

1. **预编译实例化**: 为常用列数（3、5、10）预编译模板实例
2. **分离内核**: 将内核函数独立编译到动态库中
3. **接口封装**: 通过虚函数接口隐藏 CUDA 实现细节

### 内存管理

- 动态库生命周期由 `DynamicColumnProcessor` 管理
- 自动处理库加载和卸载
- 操作实例通过智能指针管理

## 性能考虑

### 优化策略

1. **模板特化**: 针对常用列数进行特化编译
2. **内联展开**: 在内核中使用 `#pragma unroll`
3. **内存对齐**: 确保数据访问对齐
4. **流同步**: 合理使用 CUDA 流

### 性能对比

| 方案 | 编译时间 | 运行时性能 | 内存占用 | 灵活性 |
|------|----------|------------|----------|--------|
| 静态链接 | 快 | 最优 | 高 | 低 |
| 动态链接 | 中等 | 略低 | 低 | 高 |

## 扩展新操作

### 1. 创建操作文件

```cpp
// operations_custom.cu
#include "operations_interface.h"

template<int N>
__global__ void custom_kernel(
    float* const* input_columns,
    float* output_column,
    size_t num_elements
) {
    // 自定义计算逻辑
}

template<int N>
class CustomOperationImpl : public IOperation<N> {
public:
    void execute(...) override {
        // 启动自定义内核
    }
};

// 导出工厂
EXPORT_OPERATION_FACTORY(3, CustomOperationFactory<3>)
```

### 2. 更新构建配置

```cmake
# 在 CMakeLists_dynamic.txt 中添加
add_library(operations_custom SHARED operations_custom.cu)
target_link_libraries(operations_custom CUDA::cudart)
```

### 3. 运行时使用

```cpp
processor->loadOperation("./lib/liboperations_custom.so", "CustomOperation");
processor->executeOperation("CustomOperation");
```

## 调试和故障排除

### 常见问题

1. **符号未找到**: 检查动态库导出符号
   ```bash
   nm -D ./lib/liboperations_add.so | grep createOperationFactory
   ```

2. **CUDA 错误**: 检查 GPU 架构兼容性
   ```bash
   nvidia-smi
   nvcc --version
   ```

3. **内存错误**: 使用 `cuda-memcheck` 检查
   ```bash
   cuda-memcheck ./bin/dynamic_example
   ```

### 调试技巧

1. **详细日志**: 在操作中添加调试输出
2. **单步调试**: 使用 `cuda-gdb` 调试 CUDA 代码
3. **性能分析**: 使用 `nvprof` 分析性能

## 限制和注意事项

### 当前限制

1. **列数固定**: 需要编译时确定支持的列数
2. **数据类型**: 当前只支持 `float` 类型
3. **GPU 架构**: 需要与编译时的架构兼容

### 最佳实践

1. **错误处理**: 总是检查动态库加载状态
2. **资源管理**: 及时释放不需要的操作
3. **版本控制**: 维护操作接口的版本兼容性

## 未来扩展

### 计划功能

1. **多数据类型**: 支持 `double`、`int` 等类型
2. **动态列数**: 运行时确定列数
3. **JIT 编译**: 使用 nvRTC 进行即时编译
4. **分布式计算**: 支持多 GPU 并行

### 技术探索

1. **CUDA 图**: 使用 CUDA 图优化执行
2. **内存池**: 减少内存分配开销
3. **异步执行**: 支持多操作并行执行

---

## 总结

ZMM 动态操作框架提供了一个灵活的插件架构，允许运行时动态加载和执行 CUDA 操作。虽然存在一些技术挑战，但通过合理的设计和实现，可以在保持高性能的同时获得良好的扩展性。 