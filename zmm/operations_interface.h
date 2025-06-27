#pragma once

#include <memory>

namespace zmm {

// 操作接口基类
template<int N>
class IOperation {
public:
    virtual ~IOperation() = default;
    virtual void execute(
        float* const* input_columns,
        float* output_column,
        size_t num_elements
    ) = 0;
    virtual const char* getName() const = 0;
};

// 操作工厂基类
template<int N>
class IOperationFactory {
public:
    virtual ~IOperationFactory() = default;
    virtual std::unique_ptr<IOperation<N>> createOperation() = 0;
    virtual const char* getOperationName() const = 0;
};

// 动态库入口点类型定义
template<int N>
using CreateOperationFactoryFunc = IOperationFactory<N>* (*)();

#define EXPORT_OPERATION_FACTORY(N, FactoryClass) \
    extern "C" { \
        __attribute__((visibility("default"))) \
        zmm::IOperationFactory<N>* createOperationFactory() { \
            return new FactoryClass(); \
        } \
    }

} // namespace zmm 