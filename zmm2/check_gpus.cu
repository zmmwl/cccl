#include <cuda_runtime.h>
#include <iostream>

int main() {
    std::cout << "=== CUDA GPU 设备检查 ===" << std::endl;
    
    // 获取CUDA设备数量
    int device_count;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess) {
        std::cout << "错误: 无法获取CUDA设备数量: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }
    
    std::cout << "检测到 " << device_count << " 个CUDA设备:" << std::endl;
    std::cout << std::endl;
    
    // 遍历所有设备
    for (int i = 0; i < device_count; ++i) {
        cudaDeviceProp device_prop;
        cudaGetDeviceProperties(&device_prop, i);
        
        std::cout << "设备 " << i << ":" << std::endl;
        std::cout << "  名称: " << device_prop.name << std::endl;
        std::cout << "  计算能力: " << device_prop.major << "." << device_prop.minor << std::endl;
        std::cout << "  全局内存: " << device_prop.totalGlobalMem / (1024*1024) << " MB" << std::endl;
        std::cout << "  多处理器数量: " << device_prop.multiProcessorCount << std::endl;
        std::cout << "  最大线程/块: " << device_prop.maxThreadsPerBlock << std::endl;
        std::cout << "  最大网格尺寸: " << device_prop.maxGridSize[0] << " x " 
                  << device_prop.maxGridSize[1] << " x " << device_prop.maxGridSize[2] << std::endl;
        std::cout << "  最大块尺寸: " << device_prop.maxThreadsDim[0] << " x " 
                  << device_prop.maxThreadsDim[1] << " x " << device_prop.maxThreadsDim[2] << std::endl;
        std::cout << "  时钟频率: " << device_prop.clockRate / 1000 << " MHz" << std::endl;
        std::cout << "  内存时钟频率: " << device_prop.memoryClockRate / 1000 << " MHz" << std::endl;
        std::cout << "  内存总线宽度: " << device_prop.memoryBusWidth << " bits" << std::endl;
        
        // 显示内存使用情况
        size_t free_mem, total_mem;
        cudaSetDevice(i);
        cudaMemGetInfo(&free_mem, &total_mem);
        std::cout << "  当前内存使用: " << (total_mem - free_mem) / (1024*1024) << " MB / " 
                  << total_mem / (1024*1024) << " MB" << std::endl;
        
        std::cout << std::endl;
    }
    
    // 获取当前使用的设备
    int current_device;
    cudaGetDevice(&current_device);
    std::cout << "当前默认使用的设备: GPU " << current_device << std::endl;
    
    // 显示CUDA运行时版本
    int runtime_version;
    cudaRuntimeGetVersion(&runtime_version);
    std::cout << "CUDA运行时版本: " << runtime_version / 1000 << "." << (runtime_version % 1000) / 10 << std::endl;
    
    // 显示驱动版本
    int driver_version;
    cudaDriverGetVersion(&driver_version);
    std::cout << "CUDA驱动版本: " << driver_version / 1000 << "." << (driver_version % 1000) / 10 << std::endl;
    
    return 0;
} 