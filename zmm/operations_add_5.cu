#include "operations_interface.h"
#include "column_processor.cuh"
#include <cuda_runtime.h>

namespace zmm {

// AddOperation的内核函数
template<int N>
__global__ void add_kernel(
    float* const* input_columns,
    float* output_column,
    size_t num_elements
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    for (size_t i = idx; i < num_elements; i += stride) {
        float result = 0.0f;
        #pragma unroll
        for (int col = 0; col < N; ++col) {
            result += input_columns[col][i];
        }
        output_column[i] = result;
    }
}

// AddOperation实现
template<int N>
class AddOperationImpl : public IOperation<N> {
private:
    cudaStream_t stream_;
    
public:
    AddOperationImpl() {
        CUDA_CHECK(cudaStreamCreate(&stream_));
    }
    
    ~AddOperationImpl() {
        if (stream_) {
            cudaStreamDestroy(stream_);
        }
    }
    
    void execute(
        float* const* input_columns,
        float* output_column,
        size_t num_elements
    ) override {
        int blockSize = 256;
        int numBlocks = std::min(65535, static_cast<int>((num_elements + blockSize - 1) / blockSize));
        
        add_kernel<N><<<numBlocks, blockSize, 0, stream_>>>(
            input_columns, output_column, num_elements
        );
        
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(stream_));
    }
    
    const char* getName() const override {
        return "AddOperation";
    }
};

// 工厂实现
template<int N>
class AddOperationFactory : public IOperationFactory<N> {
public:
    std::unique_ptr<IOperation<N>> createOperation() override {
        return std::make_unique<AddOperationImpl<N>>();
    }
    
    const char* getOperationName() const override {
        return "AddOperation";
    }
};

// 显式实例化5列
template class AddOperationImpl<5>;
template class AddOperationFactory<5>;

} // namespace zmm

// 导出工厂函数
EXPORT_OPERATION_FACTORY(5, zmm::AddOperationFactory<5>) 