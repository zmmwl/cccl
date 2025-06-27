#pragma once

#include "operations_interface.h"
#include "column_processor.cuh"
#include <dlfcn.h>
#include <memory>
#include <string>
#include <unordered_map>

namespace zmm {

template<int N>
class DynamicColumnProcessor {
private:
    std::unique_ptr<ColumnProcessor<N>> core_processor_;
    std::unordered_map<std::string, void*> loaded_libraries_;
    std::unordered_map<std::string, std::unique_ptr<IOperationFactory<N>>> factories_;
    
public:
    DynamicColumnProcessor(size_t num_elements) 
        : core_processor_(std::make_unique<ColumnProcessor<N>>(num_elements)) {
    }
    
    ~DynamicColumnProcessor() {
        // 清理工厂
        factories_.clear();
        
        // 卸载动态库
        for (auto& lib : loaded_libraries_) {
            if (lib.second) {
                dlclose(lib.second);
            }
        }
    }
    
    // 加载操作动态库
    bool loadOperation(const std::string& library_path, const std::string& operation_name) {
        // 检查是否已经加载
        if (factories_.find(operation_name) != factories_.end()) {
            return true;
        }
        
        // 加载动态库
        void* handle = dlopen(library_path.c_str(), RTLD_LAZY);
        if (!handle) {
            std::cerr << "无法加载动态库 " << library_path << ": " << dlerror() << std::endl;
            return false;
        }
        
        // 获取工厂创建函数
        typedef IOperationFactory<N>* (*CreateFactoryFunc)();
        CreateFactoryFunc create_factory = (CreateFactoryFunc) dlsym(handle, "createOperationFactory");
        
        const char* dlsym_error = dlerror();
        if (dlsym_error) {
            std::cerr << "无法找到工厂函数 createOperationFactory: " << dlsym_error << std::endl;
            dlclose(handle);
            return false;
        }
        
        // 创建工厂实例
        auto factory = std::unique_ptr<IOperationFactory<N>>(create_factory());
        if (!factory) {
            std::cerr << "创建工厂失败" << std::endl;
            dlclose(handle);
            return false;
        }
        
        // 存储库句柄和工厂
        loaded_libraries_[operation_name] = handle;
        factories_[operation_name] = std::move(factory);
        
        std::cout << "成功加载操作: " << operation_name << std::endl;
        return true;
    }
    
    // 执行指定的操作
    bool executeOperation(const std::string& operation_name) {
        auto it = factories_.find(operation_name);
        if (it == factories_.end()) {
            std::cerr << "未找到操作: " << operation_name << std::endl;
            return false;
        }
        
        // 创建操作实例
        auto operation = it->second->createOperation();
        if (!operation) {
            std::cerr << "创建操作实例失败: " << operation_name << std::endl;
            return false;
        }
        
        // 执行操作（需要先获取设备指针）
        operation->execute(
            core_processor_->getInputColumnPointers(),
            core_processor_->getOutputPointer(),
            core_processor_->getNumElements()
        );
        
        return true;
    }
    
    // 代理方法到核心处理器
    void setInputColumn(int column_index, const float* host_data) {
        core_processor_->setInputColumn(column_index, host_data);
    }
    
    void getResult(float* host_output) {
        core_processor_->getResult(host_output);
    }
    
    size_t getNumElements() const {
        return core_processor_->getNumElements();
    }
    
    void synchronize() {
        core_processor_->synchronize();
    }
    
    // 列出已加载的操作
    std::vector<std::string> getLoadedOperations() const {
        std::vector<std::string> operations;
        for (const auto& factory : factories_) {
            operations.push_back(factory.first);
        }
        return operations;
    }
};

// 便捷工厂函数
template<int N>
std::unique_ptr<DynamicColumnProcessor<N>> createDynamicProcessor(size_t num_elements) {
    return std::make_unique<DynamicColumnProcessor<N>>(num_elements);
}

} // namespace zmm 