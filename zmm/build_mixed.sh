#!/bin/bash

# ZMM CUDA 混合类型框架构建脚本

echo "开始构建 ZMM CUDA 混合类型处理框架..."

# 检查CUDA是否可用
if ! command -v nvcc &> /dev/null; then
    echo "错误: 未找到 nvcc，请确保安装了CUDA Toolkit"
    exit 1
fi

# 显示CUDA版本
echo "CUDA版本信息:"
nvcc --version

# 创建构建目录
if [ ! -d "build_mixed" ]; then
    mkdir build_mixed
fi

cd build_mixed

# 配置CMake（使用混合类型配置）
echo "配置项目..."

# 创建专用的CMakeLists.txt文件用于混合类型
cat > ../CMakeLists_mixed.txt << 'EOF'
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

# 保留原有示例程序（用于性能对比）

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
endif()

# 添加头文件到IDE
set(MIXED_HEADER_FILES
    mixed_types.h
    mixed_kernels.cuh
    mixed_operations.cuh
    mixed_column_processor.cuh
    # 原有头文件
    # column_processor.cuh
    # operations_interface.h
    # dynamic_processor.h
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

# 自定义目标：构建所有
add_custom_target(build_all_mixed 
    DEPENDS zmm_mixed_types mixed_example 
)

# 性能对比目标
add_custom_target(performance_comparison
    COMMAND echo "运行原始框架性能测试..."
    COMMAND echo "运行混合类型框架性能测试..."
    COMMAND ./bin/mixed_example
)

# 帮助信息
add_custom_target(show_mixed_help
    COMMAND ${CMAKE_COMMAND} -E echo "可用目标："
    COMMAND ${CMAKE_COMMAND} -E echo "  zmm_mixed_types      - 混合类型核心库"
    COMMAND ${CMAKE_COMMAND} -E echo "  mixed_example        - 混合类型示例程序"
    COMMAND ${CMAKE_COMMAND} -E echo "  build_all_mixed      - 构建所有混合类型目标"
    COMMAND ${CMAKE_COMMAND} -E echo "  performance_comparison - 运行性能对比测试"
)
EOF

cp ../CMakeLists_mixed.txt ../CMakeLists.txt.backup
cp ../CMakeLists_mixed.txt ../CMakeLists.txt
cmake .. -DCMAKE_BUILD_TYPE=Release

# 编译
echo "开始编译..."
make -j$(nproc)

if [ $? -eq 0 ]; then
    echo "编译成功！"
    echo ""
    echo "生成的文件："
    echo "库文件："
    echo "  - lib/libzmm_mixed_types.a    - 混合类型核心库"
    echo ""
    echo "可执行文件："
    echo "  - bin/mixed_example          - 混合类型示例程序"
    echo ""
    echo "运行混合类型示例："
    echo "  cd build_mixed && ./bin/mixed_example"
    echo ""
    echo "运行性能对比测试："
    echo "  cd build_mixed && make performance_comparison"
    echo ""
    echo "显示帮助信息："
    echo "  cd build_mixed && make show_mixed_help"
else
    echo "编译失败！"
    exit 1
fi

# 运行一些基本检查
echo "进行基本检查..."

# 检查库文件
if [ -f "./lib/libzmm_mixed_types.a" ]; then
    echo "✓ 混合类型核心库已生成"
    size_info=$(ls -lh ./lib/libzmm_mixed_types.a | awk '{print $5}')
    echo "  库文件大小: $size_info"
else
    echo "✗ 混合类型核心库生成失败"
fi

# 检查可执行文件
if [ -f "./bin/mixed_example" ]; then
    echo "✓ 混合类型示例程序已生成"
    
    # 尝试运行一个简单的测试
    echo "运行快速验证测试..."
    timeout 10s ./bin/mixed_example > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo "✓ 混合类型示例程序运行正常"
    else
        echo "⚠ 混合类型示例程序运行可能有问题，请手动测试"
    fi
else
    echo "✗ 混合类型示例程序生成失败"
fi

echo "构建完成！"

# 恢复原始CMakeLists.txt
if [ -f "../CMakeLists.txt.backup" ]; then
    mv ../CMakeLists.txt.backup ../CMakeLists.txt
    echo "已恢复原始CMakeLists.txt"
fi

# 提供使用建议
echo ""
echo "使用建议："
echo "1. 运行混合类型示例："
echo "   ./bin/mixed_example"
echo ""
echo "2. 进行性能对比测试："
echo "   make performance_comparison"
echo ""
echo "3. 集成到现有项目："
echo "   - 链接库: libzmm_mixed_types.a"
echo "   - 包含头文件: mixed_column_processor.cuh"
echo ""
echo "4. 查看详细文档："
echo "   cat ../MIXED_TYPES_README.md"
