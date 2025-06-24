#!/bin/bash

# ZMM2 列数据计算框架构建脚本

set -e  # 遇到错误时退出

echo "ZMM2 列数据计算框架构建脚本"
echo "=============================="

# 检查CUDA是否可用
if ! command -v nvcc &> /dev/null; then
    echo "错误: 未找到nvcc命令，请确保CUDA工具包已正确安装"
    exit 1
fi

# 显示CUDA版本
echo "CUDA版本:"
nvcc --version

# 检查CMake是否可用
if ! command -v cmake &> /dev/null; then
    echo "错误: 未找到cmake命令，请安装CMake 3.18或更高版本"
    exit 1
fi

# 显示CMake版本
echo "CMake版本:"
cmake --version

# 创建构建目录
BUILD_DIR="build"
if [ -d "$BUILD_DIR" ]; then
    echo "清理旧的构建目录..."
    rm -rf "$BUILD_DIR"
fi

echo "创建构建目录..."
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# 配置项目
echo "配置项目..."
cmake .. -DCMAKE_BUILD_TYPE=Release

# 编译项目
echo "编译项目..."
make -j$(nproc)

echo "构建完成!"
echo ""

# 检查GPU设备
echo "检查CUDA设备..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader,nounits
else
    echo "警告: 未找到nvidia-smi命令，无法查询GPU信息"
fi

echo ""
echo "可用的可执行文件:"
ls -la simple_example test_framework 2>/dev/null || echo "构建可能失败，请检查错误信息"

echo ""
echo "使用方法:"
echo "  ./simple_example    - 运行简单示例"
echo "  ./test_framework    - 运行完整测试"

echo ""
echo "构建脚本执行完成!" 