#!/bin/bash

# ZMM CUDA Framework 动态库构建脚本

echo "开始构建 ZMM CUDA 动态操作框架..."

# 检查CUDA是否可用
if ! command -v nvcc &> /dev/null; then
    echo "错误: 未找到 nvcc，请确保安装了CUDA Toolkit"
    exit 1
fi

# 显示CUDA版本
echo "CUDA版本信息:"
nvcc --version

# 创建构建目录
if [ ! -d "build_dynamic" ]; then
    mkdir build_dynamic
fi

cd build_dynamic

# 配置CMake（使用动态库配置）
echo "配置项目..."
cp ../CMakeLists_dynamic.txt ../CMakeLists.txt.bak
cp ../CMakeLists_dynamic.txt ../CMakeLists.txt
cmake .. -DCMAKE_BUILD_TYPE=Release

# 编译
echo "开始编译..."
make -j$(nproc)

if [ $? -eq 0 ]; then
    echo "编译成功！"
    echo ""
    echo "生成的文件："
    echo "动态库："
    echo "  - lib/liboperations_add_3.so      - AddOperation动态库(3列)"
    echo "  - lib/liboperations_add_5.so      - AddOperation动态库(5列)"
    echo "  - lib/liboperations_multiply_3.so - MultiplyOperation动态库(3列)"
    echo ""
    echo "可执行文件："
    echo "  - bin/dynamic_example    - 动态加载示例"
    echo "  - bin/simple_example     - 简单示例"
    echo "  - bin/example            - 完整示例"
    echo ""
    echo "运行动态示例："
    echo "  cd build_dynamic && ./bin/dynamic_example"
    echo ""
    echo "测试动态库："
    echo "  ldd ./lib/liboperations_add.so"
    echo "  nm -D ./lib/liboperations_add.so | grep createOperationFactory"
else
    echo "编译失败！"
    exit 1
fi

# 运行一些基本检查
echo "进行基本检查..."

# 检查动态库符号
if [ -f "./lib/liboperations_add_3.so" ]; then
    echo "检查 AddOperation(3列) 动态库符号："
    nm -D ./lib/liboperations_add_3.so | grep createOperationFactory || echo "  未找到工厂函数符号"
fi

if [ -f "./lib/liboperations_add_5.so" ]; then
    echo "检查 AddOperation(5列) 动态库符号："
    nm -D ./lib/liboperations_add_5.so | grep createOperationFactory || echo "  未找到工厂函数符号"
fi

if [ -f "./lib/liboperations_multiply_3.so" ]; then
    echo "检查 MultiplyOperation(3列) 动态库符号："
    nm -D ./lib/liboperations_multiply_3.so | grep createOperationFactory || echo "  未找到工厂函数符号"
fi

echo "构建完成！"

# 恢复原始CMakeLists.txt
if [ -f "../CMakeLists.txt.bak" ]; then
    mv ../CMakeLists.txt.bak ../CMakeLists.txt
    echo "已恢复原始CMakeLists.txt"
fi 