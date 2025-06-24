#!/bin/bash

# ZMM CUDA Framework 构建脚本

echo "开始构建 ZMM CUDA 框架..."

# 检查CUDA是否可用
if ! command -v nvcc &> /dev/null; then
    echo "错误: 未找到 nvcc，请确保安装了CUDA Toolkit"
    exit 1
fi

# 显示CUDA版本
echo "CUDA版本信息:"
nvcc --version

# 创建构建目录
if [ ! -d "build" ]; then
    mkdir build
fi

cd build

# 配置CMake
echo "配置项目..."
cmake .. -DCMAKE_BUILD_TYPE=Release

# 编译
echo "开始编译..."
make -j$(nproc)

if [ $? -eq 0 ]; then
    echo "编译成功！"
    echo "可执行文件位置:"
    echo "  - 简单示例: ./bin/simple_example"
    echo "  - 完整示例: ./bin/example"
    echo ""
    echo "运行简单示例:"
    echo "  ./bin/simple_example"
else
    echo "编译失败！"
    exit 1
fi 