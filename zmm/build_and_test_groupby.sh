#!/bin/bash

# ZMM GroupBy 功能构建和测试脚本

echo "================================================"
echo "ZMM GroupBy 分组求和功能 - 构建和测试"
echo "================================================"

# 检查CUDA
if ! command -v nvcc &> /dev/null; then
    echo "错误: 未找到 nvcc，请确保安装了CUDA Toolkit"
    exit 1
fi

echo ""
echo "CUDA版本信息:"
nvcc --version

# 创建或使用构建目录
BUILD_DIR="build_groupby"
if [ ! -d "$BUILD_DIR" ]; then
    echo ""
    echo "创建构建目录: $BUILD_DIR"
    mkdir "$BUILD_DIR"
fi

cd "$BUILD_DIR"

# 配置CMake
echo ""
echo "配置CMake..."
cmake .. -DCMAKE_BUILD_TYPE=Release

if [ $? -ne 0 ]; then
    echo "CMake配置失败！"
    exit 1
fi

# 编译
echo ""
echo "开始编译..."
echo "  - zmm_mixed_types (核心库)"
echo "  - groupby_simple_test (简单测试)"
echo "  - groupby_example (完整示例)"

make -j$(nproc) zmm_mixed_types groupby_simple_test groupby_example

if [ $? -ne 0 ]; then
    echo ""
    echo "编译失败！"
    exit 1
fi

echo ""
echo "✓ 编译成功！"
echo ""

# 检查生成的文件
echo "生成的文件:"
if [ -f "./lib/libzmm_mixed_types.a" ]; then
    size_info=$(ls -lh ./lib/libzmm_mixed_types.a | awk '{print $5}')
    echo "  ✓ lib/libzmm_mixed_types.a ($size_info)"
else
    echo "  ✗ lib/libzmm_mixed_types.a (未找到)"
fi

if [ -f "./bin/groupby_simple_test" ]; then
    echo "  ✓ bin/groupby_simple_test"
else
    echo "  ✗ bin/groupby_simple_test (未找到)"
fi

if [ -f "./bin/groupby_example" ]; then
    echo "  ✓ bin/groupby_example"
else
    echo "  ✗ bin/groupby_example (未找到)"
fi

echo ""
echo "================================================"
echo "运行测试"
echo "================================================"

# 运行简单测试
if [ -f "./bin/groupby_simple_test" ]; then
    echo ""
    echo ">>> 运行简单测试 (groupby_simple_test) <<<"
    echo ""
    ./bin/groupby_simple_test
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ 简单测试通过！"
    else
        echo ""
        echo "✗ 简单测试失败！"
        exit 1
    fi
else
    echo "简单测试程序未找到，跳过"
fi

# 询问是否运行完整示例
echo ""
read -p "是否运行完整示例程序 (包含大数据测试)? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if [ -f "./bin/groupby_example" ]; then
        echo ""
        echo ">>> 运行完整示例 (groupby_example) <<<"
        echo ""
        ./bin/groupby_example
        
        if [ $? -eq 0 ]; then
            echo ""
            echo "✓ 完整示例运行成功！"
        else
            echo ""
            echo "✗ 完整示例运行失败！"
        fi
    else
        echo "完整示例程序未找到"
    fi
fi

echo ""
echo "================================================"
echo "构建和测试完成"
echo "================================================"
echo ""
echo "可用命令:"
echo "  ./bin/groupby_simple_test   - 运行简单测试（推荐）"
echo "  ./bin/groupby_example       - 运行完整示例（包含性能测试）"
echo ""
echo "文档:"
echo "  cat ../GROUPBY_README.md    - 查看详细使用文档"
echo ""


