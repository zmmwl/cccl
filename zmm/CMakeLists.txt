cmake_minimum_required(VERSION 3.12)
project(ZMM_CUDA_Framework LANGUAGES CXX CUDA)

# 设置C++标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置CUDA标准
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 查找CUDA
find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)

# 包含CCCL路径
set(CCCL_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/..")
include_directories(${CCCL_ROOT})
include_directories(${CCCL_ROOT}/cub)
include_directories(${CCCL_ROOT}/thrust)
include_directories(${CCCL_ROOT}/libcudacxx/include)

# 设置CUDA编译选项
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G -g")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3")

# 检测GPU架构
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86")
endif()

# 添加可执行文件
add_executable(example example.cu)
add_executable(simple_example simple_example.cu)

# 链接CUDA库
target_link_libraries(example CUDA::cudart)
target_link_libraries(simple_example CUDA::cudart)

# 设置编译器特定选项
if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    target_compile_options(example PRIVATE 
        $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    )
    target_compile_options(simple_example PRIVATE 
        $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    )
endif()

# 设置输出目录
set_target_properties(example PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
set_target_properties(simple_example PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# 添加头文件到IDE
set(HEADER_FILES
    column_processor.cuh
)

# 创建一个仅包含头文件的目标，方便IDE显示
add_custom_target(headers SOURCES ${HEADER_FILES})

# 显示一些有用的信息
message(STATUS "CUDA Compiler: ${CMAKE_CUDA_COMPILER}")
message(STATUS "CUDA Version: ${CUDAToolkit_VERSION}")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")

# 添加测试目标（可选）
enable_testing()
add_test(NAME simple_test COMMAND simple_example)
add_test(NAME full_test COMMAND example) 