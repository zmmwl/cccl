#!/bin/bash

# ZMM GroupBy 功能 ARM 交叉编译脚本
# 用于交叉编译到 ARM 架构 (NVIDIA Orin/Jetson 等)

echo "================================================"
echo "ZMM GroupBy 分组求和功能 - ARM 交叉编译"
echo "================================================"

# 配置参数
ARCH="compute_87"           # 计算能力 (Orin: 87, Xavier: 72, Nano: 53)
CODE="sm_87"                # 目标代码
CROSS_COMPILER="aarch64-linux-gnu-g++"  # ARM64 交叉编译器
BUILD_DIR="build_groupby_arm"
OPTIMIZATION="-O3"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --arch)
            ARCH="$2"
            CODE="sm_${2#compute_}"
            shift 2
            ;;
        --compiler)
            CROSS_COMPILER="$2"
            shift 2
            ;;
        --debug)
            OPTIMIZATION="-g -G"
            shift
            ;;
        --help)
            echo "使用方法: $0 [选项]"
            echo "选项:"
            echo "  --arch <arch>       指定 GPU 架构 (默认: compute_87)"
            echo "                      Orin: compute_87, Xavier: compute_72, Nano: compute_53"
            echo "  --compiler <path>   指定交叉编译器 (默认: aarch64-linux-gnu-g++)"
            echo "  --debug            启用调试模式 (-g -G)"
            echo "  --help             显示此帮助信息"
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            echo "使用 --help 查看帮助"
            exit 1
            ;;
    esac
done

# 检查 nvcc
if ! command -v nvcc &> /dev/null; then
    echo "错误: 未找到 nvcc，请确保安装了 CUDA Toolkit"
    exit 1
fi

# 检查交叉编译器
if ! command -v $CROSS_COMPILER &> /dev/null; then
    echo "错误: 未找到交叉编译器 $CROSS_COMPILER"
    echo "请安装 ARM 交叉编译工具链:"
    echo "  Ubuntu/Debian: sudo apt-get install g++-aarch64-linux-gnu"
    exit 1
fi

echo ""
echo "编译配置:"
echo "  CUDA 版本: $(nvcc --version | grep release | awk '{print $5}' | sed 's/,//')"
echo "  GPU 架构: $ARCH -> $CODE"
echo "  交叉编译器: $CROSS_COMPILER"
echo "  优化级别: $OPTIMIZATION"
echo ""

# 创建构建目录
if [ ! -d "$BUILD_DIR" ]; then
    echo "创建构建目录: $BUILD_DIR"
    mkdir "$BUILD_DIR"
fi

cd "$BUILD_DIR"

# 设置路径
CCCL_ROOT=".."
INCLUDE_PATHS="-I${CCCL_ROOT} -I${CCCL_ROOT}/cub -I${CCCL_ROOT}/thrust -I${CCCL_ROOT}/libcudacxx/include"

# NVCC 编译选项
NVCC_FLAGS="-gencode arch=${ARCH},code=${CODE} -gencode arch=${ARCH},code=${ARCH}"
NVCC_FLAGS="${NVCC_FLAGS} -ccbin ${CROSS_COMPILER}"
NVCC_FLAGS="${NVCC_FLAGS} ${INCLUDE_PATHS}"
NVCC_FLAGS="${NVCC_FLAGS} --extended-lambda --expt-relaxed-constexpr"
NVCC_FLAGS="${NVCC_FLAGS} -std=c++14"
NVCC_FLAGS="${NVCC_FLAGS} ${OPTIMIZATION}"
NVCC_FLAGS="${NVCC_FLAGS} -lineinfo"

echo "NVCC 编译标志: $NVCC_FLAGS"
echo ""

# 创建 bin 和 lib 目录
mkdir -p bin
mkdir -p lib

echo "================================================"
echo "编译步骤 1/4: 编译 mixed_types.cu"
echo "================================================"
nvcc ${NVCC_FLAGS} -c ../mixed_types.cu -o lib/mixed_types.o
if [ $? -ne 0 ]; then
    echo "错误: mixed_types.cu 编译失败"
    exit 1
fi
echo "✓ mixed_types.o 编译成功"

echo ""
echo "================================================"
echo "编译步骤 2/4: 编译 mixed_kernels.cu"
echo "================================================"
nvcc ${NVCC_FLAGS} -c ../mixed_kernels.cu -o lib/mixed_kernels.o
if [ $? -ne 0 ]; then
    echo "错误: mixed_kernels.cu 编译失败"
    exit 1
fi
echo "✓ mixed_kernels.o 编译成功"

echo ""
echo "================================================"
echo "编译步骤 3/4: 编译 mixed_column_processor.cu"
echo "================================================"
nvcc ${NVCC_FLAGS} -c ../mixed_column_processor.cu -o lib/mixed_column_processor.o
if [ $? -ne 0 ]; then
    echo "错误: mixed_column_processor.cu 编译失败"
    exit 1
fi
echo "✓ mixed_column_processor.o 编译成功"

echo ""
echo "================================================"
echo "编译步骤 4/4: 链接可执行文件"
echo "================================================"

# 编译 groupby_simple_test
echo "链接 groupby_simple_test..."
nvcc ${NVCC_FLAGS} ../groupby_simple_test.cu \
    lib/mixed_types.o \
    lib/mixed_kernels.o \
    lib/mixed_column_processor.o \
    -o bin/groupby_simple_test_arm \
    -lcudart

if [ $? -ne 0 ]; then
    echo "错误: groupby_simple_test 链接失败"
    exit 1
fi
echo "✓ bin/groupby_simple_test_arm"

# 编译 groupby_example
echo "链接 groupby_example..."
nvcc ${NVCC_FLAGS} ../groupby_example.cu \
    lib/mixed_types.o \
    lib/mixed_kernels.o \
    lib/mixed_column_processor.o \
    -o bin/groupby_example_arm \
    -lcudart

if [ $? -ne 0 ]; then
    echo "错误: groupby_example 链接失败"
    exit 1
fi
echo "✓ bin/groupby_example_arm"

echo ""
echo "================================================"
echo "编译完成"
echo "================================================"
echo ""

# 显示生成的文件
echo "生成的文件:"
ls -lh bin/groupby_simple_test_arm bin/groupby_example_arm 2>/dev/null | awk '{printf "  %-35s %s\n", $9, $5}'

echo ""
echo "目标文件:"
ls -lh lib/*.o 2>/dev/null | awk '{printf "  %-35s %s\n", $9, $5}'

echo ""
echo "================================================"
echo "部署说明"
echo "================================================"
echo ""
echo "1. 将编译好的可执行文件传输到 ARM 设备:"
echo "   scp bin/groupby_simple_test_arm user@arm-device:/path/to/destination/"
echo "   scp bin/groupby_example_arm user@arm-device:/path/to/destination/"
echo ""
echo "2. 在 ARM 设备上运行:"
echo "   ./groupby_simple_test_arm"
echo "   ./groupby_example_arm"
echo ""
echo "3. 确保 ARM 设备上已安装 CUDA Runtime (JetPack SDK)"
echo ""

# 创建部署脚本
cat > deploy_to_arm.sh << 'EOF'
#!/bin/bash
# ARM 设备部署脚本

if [ $# -lt 1 ]; then
    echo "使用方法: $0 <user@arm-device>"
    echo "示例: $0 nvidia@192.168.1.100"
    exit 1
fi

TARGET=$1
REMOTE_DIR="/home/$(echo $TARGET | cut -d'@' -f1)/zmm_groupby"

echo "部署到 ARM 设备: $TARGET"
echo "目标目录: $REMOTE_DIR"

# 创建远程目录
ssh $TARGET "mkdir -p $REMOTE_DIR"

# 传输文件
echo "传输可执行文件..."
scp bin/groupby_simple_test_arm $TARGET:$REMOTE_DIR/
scp bin/groupby_example_arm $TARGET:$REMOTE_DIR/

# 设置执行权限
echo "设置执行权限..."
ssh $TARGET "chmod +x $REMOTE_DIR/groupby_simple_test_arm $REMOTE_DIR/groupby_example_arm"

echo ""
echo "部署完成！"
echo "登录 ARM 设备运行测试:"
echo "  ssh $TARGET"
echo "  cd $REMOTE_DIR"
echo "  ./groupby_simple_test_arm"
EOF

chmod +x deploy_to_arm.sh
echo "✓ 已创建部署脚本: deploy_to_arm.sh"
echo ""
echo "快速部署到 ARM 设备:"
echo "  ./deploy_to_arm.sh user@arm-device-ip"
echo ""

# 创建架构检测脚本
cat > detect_gpu_arch.sh << 'EOF'
#!/bin/bash
# GPU 架构检测脚本 (在 ARM 设备上运行)

echo "检测 GPU 架构..."

# 检查是否安装了 CUDA
if ! command -v nvidia-smi &> /dev/null; then
    echo "错误: 未找到 nvidia-smi，请确保安装了 CUDA 驱动"
    exit 1
fi

# 获取 GPU 信息
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
echo "GPU: $GPU_NAME"

# 根据 GPU 名称推荐架构
case $GPU_NAME in
    *"Orin"*)
        echo "推荐架构: compute_87 (Orin)"
        echo "编译命令: ./build_and_test_groupby_arm.sh --arch compute_87"
        ;;
    *"Xavier"*)
        echo "推荐架构: compute_72 (Xavier)"
        echo "编译命令: ./build_and_test_groupby_arm.sh --arch compute_72"
        ;;
    *"Nano"*)
        echo "推荐架构: compute_53 (Nano)"
        echo "编译命令: ./build_and_test_groupby_arm.sh --arch compute_53"
        ;;
    *)
        echo "未知 GPU 型号"
        echo "请查看 CUDA 文档确定正确的计算能力"
        ;;
esac

# 显示详细信息
echo ""
echo "详细 GPU 信息:"
nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv,noheader
EOF

chmod +x detect_gpu_arch.sh
echo "✓ 已创建架构检测脚本: detect_gpu_arch.sh"
echo "  (将此脚本复制到 ARM 设备运行以检测正确的架构)"

cd ..

echo ""
echo "================================================"
echo "常见 ARM GPU 架构参考"
echo "================================================"
echo "  NVIDIA Orin (AGX/NX):     compute_87"
echo "  NVIDIA Xavier (AGX/NX):   compute_72"
echo "  NVIDIA Jetson Nano:       compute_53"
echo "  NVIDIA Jetson TX2:        compute_62"
echo ""
echo "如需编译其他架构，使用:"
echo "  ./build_and_test_groupby_arm.sh --arch compute_XX"
echo ""

