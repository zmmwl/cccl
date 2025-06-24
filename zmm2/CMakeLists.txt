cmake_minimum_required(VERSION 3.18)
project(ZMM2_ColumnCompute LANGUAGES CXX CUDA)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置CUDA标准
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 查找必要的包
find_package(CUDAToolkit REQUIRED)

# 设置CUDA架构
set(CMAKE_CUDA_ARCHITECTURES "60;70;75;80;86")

# 编译选项
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --expt-relaxed-constexpr --extended-lambda")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# 包含当前目录作为头文件搜索路径
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# 添加CUDA包含目录
include_directories(${CUDAToolkit_INCLUDE_DIRS})

# 创建简单示例可执行文件
add_executable(simple_example simple_example.cu)
target_link_libraries(simple_example CUDA::cudart)

# 创建测试框架可执行文件
add_executable(test_framework test_framework.cu)
target_link_libraries(test_framework CUDA::cudart)

# 设置编译属性
set_target_properties(simple_example PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

set_target_properties(test_framework PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# 创建一个仅头文件的库目标（可选）
add_library(zmm2_framework INTERFACE)
target_include_directories(zmm2_framework INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

# 如果要安装（可选）
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/install" CACHE PATH "Installation directory" FORCE)
endif()

install(FILES
    column_compute_framework.cuh
    business_expressions.cuh
    batch_processor.cuh
    DESTINATION include/zmm2
)

install(TARGETS zmm2_framework
    EXPORT zmm2_frameworkConfig
    INCLUDES DESTINATION include
)

install(EXPORT zmm2_frameworkConfig
    FILE zmm2_frameworkConfig.cmake
    DESTINATION lib/cmake/zmm2_framework
)

# 显示构建信息
message(STATUS "ZMM2 Column Compute Framework")
message(STATUS "CUDA Version: ${CUDAToolkit_VERSION}")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "CUDA Standard: ${CMAKE_CUDA_STANDARD}") 