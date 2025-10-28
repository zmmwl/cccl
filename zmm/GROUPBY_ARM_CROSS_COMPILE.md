# ZMM GroupBy ARM 交叉编译指南

本文档介绍如何在 x86_64 主机上交叉编译 ZMM GroupBy 功能到 ARM 架构（NVIDIA Jetson/Orin 平台）。

## 前置条件

### 主机端 (x86_64 开发机器)

1. **CUDA Toolkit**
   - 需要支持目标 ARM GPU 架构的 CUDA 版本
   - 推荐 CUDA 11.4+ 或 12.x

2. **ARM 交叉编译工具链**
   ```bash
   # Ubuntu/Debian
   sudo apt-get install g++-aarch64-linux-gnu gcc-aarch64-linux-gnu
   
   # 验证安装
   aarch64-linux-gnu-g++ --version
   ```

3. **CCCL 库**
   - 已包含在项目中（CUB, Thrust, libcudacxx）

### 目标设备端 (ARM 设备)

1. **NVIDIA JetPack SDK**
   - 根据设备型号安装对应版本的 JetPack
   - 包含 CUDA Runtime 和驱动

2. **支持的设备**
   - NVIDIA Orin (AGX/NX)
   - NVIDIA Xavier (AGX/NX)
   - NVIDIA Jetson Nano
   - NVIDIA Jetson TX2

## 快速开始

### 1. 基本编译

```bash
cd /mnt/c/dev/ml/cuda/cccl-zmm-v2.8.5/zmm

# 使用默认配置（Orin: compute_87）
./build_and_test_groupby_arm.sh
```

### 2. 指定目标架构

根据目标设备选择对应的 GPU 架构：

```bash
# NVIDIA Orin (默认)
./build_and_test_groupby_arm.sh --arch compute_87

# NVIDIA Xavier
./build_and_test_groupby_arm.sh --arch compute_72

# NVIDIA Jetson Nano
./build_and_test_groupby_arm.sh --arch compute_53

# NVIDIA Jetson TX2
./build_and_test_groupby_arm.sh --arch compute_62
```

### 3. 调试模式编译

```bash
# 启用调试符号和 GPU 调试信息
./build_and_test_groupby_arm.sh --arch compute_87 --debug
```

### 4. 自定义交叉编译器

```bash
# 使用自定义的交叉编译器路径
./build_and_test_groupby_arm.sh --compiler /path/to/aarch64-linux-gnu-g++
```

## 部署到 ARM 设备

### 方法 1: 使用自动部署脚本

编译完成后，会自动生成 `deploy_to_arm.sh` 脚本：

```bash
cd build_groupby_arm

# 部署到 ARM 设备
./deploy_to_arm.sh nvidia@192.168.1.100
```

### 方法 2: 手动部署

```bash
# 1. 传输可执行文件
scp build_groupby_arm/bin/groupby_simple_test_arm nvidia@arm-device:/home/nvidia/
scp build_groupby_arm/bin/groupby_example_arm nvidia@arm-device:/home/nvidia/

# 2. 登录 ARM 设备
ssh nvidia@arm-device

# 3. 设置执行权限
chmod +x groupby_simple_test_arm groupby_example_arm

# 4. 运行测试
./groupby_simple_test_arm
./groupby_example_arm
```

## GPU 架构检测

### 在 ARM 设备上检测 GPU 架构

编译会生成 `detect_gpu_arch.sh` 脚本，将其复制到 ARM 设备运行：

```bash
# 在开发机上
scp build_groupby_arm/detect_gpu_arch.sh nvidia@arm-device:/home/nvidia/

# 在 ARM 设备上
ssh nvidia@arm-device
./detect_gpu_arch.sh
```

输出示例：
```
检测 GPU 架构...
GPU: NVIDIA Orin
推荐架构: compute_87 (Orin)
编译命令: ./build_and_test_groupby_arm.sh --arch compute_87
```

## 编译输出

编译成功后，在 `build_groupby_arm` 目录下生成：

```
build_groupby_arm/
├── bin/
│   ├── groupby_simple_test_arm    # 简单测试程序
│   └── groupby_example_arm        # 完整示例程序
├── lib/
│   ├── mixed_types.o              # 目标文件
│   ├── mixed_kernels.o
│   └── mixed_column_processor.o
├── deploy_to_arm.sh               # 自动部署脚本
└── detect_gpu_arch.sh             # GPU 架构检测脚本
```

## 常见问题

### Q1: 交叉编译器未找到

**错误信息**:
```
错误: 未找到交叉编译器 aarch64-linux-gnu-g++
```

**解决方法**:
```bash
sudo apt-get update
sudo apt-get install g++-aarch64-linux-gnu gcc-aarch64-linux-gnu
```

### Q2: CUDA 版本不匹配

**错误信息**:
```
nvcc fatal : Unsupported gpu architecture 'compute_87'
```

**解决方法**:
- 升级 CUDA Toolkit 到 11.4 或更高版本
- 或使用较低的架构版本（如 compute_72 for Xavier）

### Q3: 在 ARM 设备上运行时报错

**错误信息**:
```
error while loading shared libraries: libcudart.so.XX.X
```

**解决方法**:
```bash
# 检查 CUDA 是否安装
nvidia-smi

# 添加 CUDA 库路径（如果需要）
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### Q4: 架构不匹配

**错误信息**:
```
no kernel image is available for execution on the device
```

**解决方法**:
1. 在 ARM 设备上运行 `nvidia-smi` 查看 GPU 型号
2. 根据 GPU 型号选择正确的 compute 架构重新编译
3. 使用 `detect_gpu_arch.sh` 脚本自动检测

## GPU 架构参考表

| 设备型号              | Compute Capability | 编译参数           |
|----------------------|-------------------|-------------------|
| NVIDIA Orin AGX/NX   | 8.7               | compute_87        |
| NVIDIA Xavier AGX/NX | 7.2               | compute_72        |
| NVIDIA Jetson TX2    | 6.2               | compute_62        |
| NVIDIA Jetson Nano   | 5.3               | compute_53        |

## 性能测试

### 在 ARM 设备上运行性能测试

```bash
# 简单测试（快速验证）
./groupby_simple_test_arm

# 完整性能测试
./groupby_example_arm
```

### 预期性能（参考）

| 设备              | 数据规模 | 分组数 | 总时间   |
|------------------|---------|--------|---------|
| Orin AGX (32GB)  | 1000万  | 1000   | ~150ms  |
| Xavier AGX       | 1000万  | 1000   | ~250ms  |
| Jetson Nano      | 100万   | 100    | ~300ms  |

## 编译选项详解

### --arch <architecture>

指定目标 GPU 的计算能力：

```bash
--arch compute_87  # Orin
--arch compute_72  # Xavier
--arch compute_62  # TX2
--arch compute_53  # Nano
```

### --compiler <path>

指定交叉编译器路径：

```bash
--compiler aarch64-linux-gnu-g++
--compiler /opt/gcc-arm/bin/aarch64-linux-gnu-g++
```

### --debug

启用调试模式：

```bash
--debug  # 添加 -g -G 标志，生成调试符号
```

## 高级用法

### 多架构编译

如果需要生成支持多种 GPU 的可执行文件：

```bash
# 编辑脚本，修改 NVCC_FLAGS，添加多个 -gencode 选项
NVCC_FLAGS="-gencode arch=compute_72,code=sm_72 \
            -gencode arch=compute_87,code=sm_87"
```

### 静态链接 CUDA Runtime

如果目标设备上 CUDA Runtime 版本不同，可以静态链接：

```bash
# 在脚本中修改链接选项
-lcudart_static -lculibos -lpthread -ldl -lrt
```

## 持续集成 (CI/CD)

### Docker 交叉编译环境

创建 Dockerfile:

```dockerfile
FROM nvidia/cuda:12.0.0-devel-ubuntu20.04

RUN apt-get update && apt-get install -y \
    g++-aarch64-linux-gnu \
    gcc-aarch64-linux-gnu \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY . .

RUN ./build_and_test_groupby_arm.sh --arch compute_87
```

### 自动化构建脚本

```bash
#!/bin/bash
# ci_build_arm.sh

ARCHS="compute_53 compute_62 compute_72 compute_87"

for arch in $ARCHS; do
    echo "Building for $arch..."
    ./build_and_test_groupby_arm.sh --arch $arch
    
    if [ $? -eq 0 ]; then
        echo "✓ $arch build successful"
    else
        echo "✗ $arch build failed"
        exit 1
    fi
done
```

## 故障排除日志

### 启用详细编译输出

编辑脚本，添加 `-v` 标志：

```bash
NVCC_FLAGS="${NVCC_FLAGS} -v"
```

### 检查依赖库

在 ARM 设备上：

```bash
# 检查可执行文件依赖
ldd groupby_simple_test_arm

# 检查 CUDA 库路径
ldconfig -p | grep cuda
```

## 参考资源

- [NVIDIA CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)
- [NVIDIA Jetson Documentation](https://developer.nvidia.com/embedded/develop/hardware)
- [CUDA GPU Architectures](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/)
- [ZMM GroupBy README](GROUPBY_README.md)
- [ZMM GroupBy Quick Start](GROUPBY_QUICKSTART.md)

## 支持

如遇问题，请提供以下信息：

1. 开发机器信息：
   - OS 版本
   - CUDA 版本 (`nvcc --version`)
   - GCC 版本 (`aarch64-linux-gnu-g++ --version`)

2. 目标设备信息：
   - 设备型号
   - JetPack 版本
   - GPU 信息 (`nvidia-smi`)

3. 完整的错误日志

---

**更新日期**: 2025年10月28日  
**测试平台**: NVIDIA Orin AGX, Xavier NX, Jetson Nano  
**CUDA 版本**: 11.4+, 12.x

