cmake_minimum_required(VERSION 3.12)
project(ZMM_Mixed_Types LANGUAGES CXX CUDA)

# 设置C++标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置CUDA标准
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 查找CUDA
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

# 设置输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

# 混合类型核心库
add_library(zmm_mixed_types STATIC 
    mixed_types.cu
    mixed_kernels.cu
    mixed_column_processor.cu
)
target_link_libraries(zmm_mixed_types CUDA::cudart)
set_target_properties(zmm_mixed_types PROPERTIES 
    CUDA_VISIBILITY_PRESET default
    CXX_VISIBILITY_PRESET default
)

# 混合类型示例程序
add_executable(mixed_example mixed_example.cu)
target_link_libraries(mixed_example zmm_mixed_types CUDA::cudart)

# 基于名称访问的测试程序
add_executable(mixed_named_test mixed_named_test.cu)
target_link_libraries(mixed_named_test zmm_mixed_types CUDA::cudart)

# 针对命名 Operation 类的测试程序
add_executable(mixed_named_ops_test mixed_named_ops_test.cu)
target_link_libraries(mixed_named_ops_test zmm_mixed_types CUDA::cudart)

# 保留原有示例程序（用于性能对比）
add_executable(example example.cu)
add_executable(simple_example simple_example.cu)
target_link_libraries(example CUDA::cudart)
target_link_libraries(simple_example CUDA::cudart)

# 设置编译器特定选项
if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    target_compile_options(zmm_mixed_types PRIVATE 
        $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    )
    target_compile_options(mixed_example PRIVATE 
        $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    )
    target_compile_options(example PRIVATE 
        $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    )
    target_compile_options(simple_example PRIVATE 
        $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    )
endif()

# 添加头文件到IDE
set(MIXED_HEADER_FILES
    mixed_types.h
    mixed_kernels.cuh
    mixed_operations.cuh
    mixed_column_processor.cuh
    # 原有头文件
    column_processor.cuh
    operations_interface.h
    dynamic_processor.h
)

# 创建一个仅包含头文件的目标，方便IDE显示
add_custom_target(mixed_headers SOURCES ${MIXED_HEADER_FILES})

# 显示一些有用的信息
message(STATUS "CUDA Compiler: ${CMAKE_CUDA_COMPILER}")
message(STATUS "CUDA Version: ${CUDAToolkit_VERSION}")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")

# 添加测试目标
enable_testing()
add_test(NAME mixed_test COMMAND mixed_example)
add_test(NAME mixed_named_test COMMAND mixed_named_test)
add_test(NAME mixed_named_ops_test COMMAND mixed_named_ops_test)
add_test(NAME simple_test COMMAND simple_example)
add_test(NAME full_test COMMAND example)

# 自定义目标：构建所有
add_custom_target(build_all_mixed 
    DEPENDS zmm_mixed_types mixed_example example simple_example
)

# 性能对比目标
add_custom_target(performance_comparison
    COMMAND echo "运行原始框架性能测试..."
    COMMAND ./bin/example
    COMMAND echo "运行混合类型框架性能测试..."
    COMMAND ./bin/mixed_example
    DEPENDS example mixed_example
)

# 帮助信息
add_custom_target(show_mixed_help
    COMMAND ${CMAKE_COMMAND} -E echo "可用目标："
    COMMAND ${CMAKE_COMMAND} -E echo "  zmm_mixed_types      - 混合类型核心库"
    COMMAND ${CMAKE_COMMAND} -E echo "  mixed_example        - 混合类型示例程序"
    COMMAND ${CMAKE_COMMAND} -E echo "  example              - 原始框架示例(对比用)"
    COMMAND ${CMAKE_COMMAND} -E echo "  simple_example       - 原始简单示例(对比用)"
    COMMAND ${CMAKE_COMMAND} -E echo "  build_all_mixed      - 构建所有混合类型目标"
    COMMAND ${CMAKE_COMMAND} -E echo "  performance_comparison - 运行性能对比测试"
)
