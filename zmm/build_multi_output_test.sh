#!/bin/bash

# 多列输出测试编译脚本
echo "========================================"
echo "编译 ZMM 多列输出测试"
echo "========================================"

# 设置变量
CCCL_ROOT="../"
BUILD_DIR="build_multi_output"
SOURCE_FILE="multi_output_test.cu"
MIXED_TYPES_CU="mixed_types.cu"
MIXED_KERNELS_CU="mixed_kernels.cu"
MIXED_COLUMN_PROCESSOR_CU="mixed_column_processor.cu"
OUTPUT_BINARY="multi_output_test"

# 创建构建目录
mkdir -p ${BUILD_DIR}
cd ${BUILD_DIR}

echo ""
echo "使用 CCCL 路径: ${CCCL_ROOT}"
echo "构建目录: ${BUILD_DIR}"
echo ""

# 检测CUDA编译器
if ! command -v nvcc &> /dev/null; then
    echo "错误: 未找到 nvcc 编译器"
    echo "请确保CUDA已正确安装并在PATH中"
    exit 1
fi

echo "CUDA 编译器信息:"
nvcc --version
echo ""

# 编译命令
echo "开始编译..."
echo ""

COMPILE_CMD="nvcc \
    -I${CCCL_ROOT}/libcudacxx/include \
    -I${CCCL_ROOT}/cub \
    -I${CCCL_ROOT}/thrust \
    -I.. \
    --extended-lambda \
    --expt-relaxed-constexpr \
    -std=c++17 \
    -O3 \
    -arch=sm_86 \
    -o ${OUTPUT_BINARY} \
    ../${SOURCE_FILE} \
    ../${MIXED_TYPES_CU} \
    ../${MIXED_KERNELS_CU} \
    ../${MIXED_COLUMN_PROCESSOR_CU}"

echo "执行编译命令:"
echo "${COMPILE_CMD}"
echo ""

# 执行编译
eval ${COMPILE_CMD}

# 检查编译结果
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "✓ 编译成功!"
    echo "========================================"
    echo ""
    echo "可执行文件: ${BUILD_DIR}/${OUTPUT_BINARY}"
    echo ""
    echo "运行测试:"
    echo "  cd ${BUILD_DIR}"
    echo "  ./${OUTPUT_BINARY}"
    echo ""
    
    # 询问是否立即运行
    read -p "是否立即运行测试? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo ""
        echo "========================================"
        echo "运行多列输出测试"
        echo "========================================"
        echo ""
        ./${OUTPUT_BINARY}
        echo ""
        echo "========================================"
        echo "测试运行完成"
        echo "========================================"
    fi
else
    echo ""
    echo "========================================"
    echo "✗ 编译失败"
    echo "========================================"
    exit 1
fi

cd ..

