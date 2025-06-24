#pragma once

#include "column_compute_framework.cuh"
#include <cmath>

namespace zmm2 {

/**
 * 示例业务表达式1: 三列相加
 * 计算公式: result = col1 + col2 + col3
 */
struct ThreeColumnSum {
    __host__ __device__ float operator()(const float values[3]) const {
        return values[0] + values[1] + values[2];
    }
};

/**
 * 示例业务表达式2: 三列加权和
 * 计算公式: result = w1*col1 + w2*col2 + w3*col3
 */
struct ThreeColumnWeightedSum {
    float w1_, w2_, w3_;

    __host__ __device__ ThreeColumnWeightedSum(float w1, float w2, float w3) 
        : w1_(w1), w2_(w2), w3_(w3) {}
    
    __host__ __device__ float operator()(const float values[3]) const {
        return w1_ * values[0] + w2_ * values[1] + w3_ * values[2];
    }
};

/**
 * 示例业务表达式3: 复杂数学表达式
 * 计算公式: result = (col1 * col2) / (col3 + epsilon) + sqrt(col1)
 */
struct ComplexMathExpression {
    float epsilon_;

    __host__ __device__ ComplexMathExpression(float epsilon = 1e-6f) 
        : epsilon_(epsilon) {}
    
    __host__ __device__ float operator()(const float values[3]) const {
        float product = values[0] * values[1];
        float denominator = values[2] + epsilon_;
        float sqrt_term = sqrtf(fabsf(values[0])); // 使用fabsf确保非负
        return product / denominator + sqrt_term;
    }
};

/**
 * 示例业务表达式4: 两列比较运算
 * 计算公式: result = (col1 > col2) ? col1 : col2 (取最大值)
 */
struct TwoColumnMax {
    __host__ __device__ float operator()(const float values[2]) const {
        return fmaxf(values[0], values[1]);
    }
};

/**
 * 示例业务表达式5: 五列移动平均
 * 计算公式: result = (col1 + col2 + col3 + col4 + col5) / 5
 */
struct FiveColumnAverage {
    __host__ __device__ float operator()(const float values[5]) const {
        float sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < 5; ++i) {
            sum += values[i];
        }
        return sum / 5.0f;
    }
};

/**
 * 示例业务表达式6: 标准化表达式 (z-score)
 * 假设values[0]是数据值，values[1]是均值，values[2]是标准差
 * 计算公式: result = (col1 - col2) / (col3 + epsilon)
 */
struct StandardizationExpression {
    float epsilon_;

    __host__ __device__ StandardizationExpression(float epsilon = 1e-8f) 
        : epsilon_(epsilon) {}
    
    __host__ __device__ float operator()(const float values[3]) const {
        return (values[0] - values[1]) / (values[2] + epsilon_);
    }
};

/**
 * 示例业务表达式7: 风险评估表达式
 * 假设values[0]是收益率，values[1]是波动率，values[2]是基准收益率
 * 计算公式: result = (col1 - col3) / col2 (Sharpe ratio的简化版本)
 */
struct RiskAssessmentExpression {
    float min_volatility_;

    __host__ __device__ RiskAssessmentExpression(float min_vol = 1e-6f) 
        : min_volatility_(min_vol) {}
    
    __host__ __device__ float operator()(const float values[3]) const {
        float excess_return = values[0] - values[2];
        float volatility = fmaxf(values[1], min_volatility_);
        return excess_return / volatility;
    }
};

/**
 * 示例业务表达式8: 多项式表达式
 * 计算公式: result = a*x^3 + b*x^2 + c*x + d
 * 其中x是values[0]，a,b,c,d是系数
 */
struct PolynomialExpression {
    float a_, b_, c_, d_;

    __host__ __device__ PolynomialExpression(float a, float b, float c, float d) 
        : a_(a), b_(b), c_(c), d_(d) {}
    
    __host__ __device__ float operator()(const float values[1]) const {
        float x = values[0];
        float x2 = x * x;
        float x3 = x2 * x;
        return a_ * x3 + b_ * x2 + c_ * x + d_;
    }
};

/**
 * 示例业务表达式9: 条件表达式
 * 计算公式: result = (col1 > threshold) ? col2 * factor : col3 * factor
 */
struct ConditionalExpression {
    float threshold_;
    float factor_;

    __host__ __device__ ConditionalExpression(float threshold, float factor) 
        : threshold_(threshold), factor_(factor) {}
    
    __host__ __device__ float operator()(const float values[3]) const {
        return (values[0] > threshold_) ? (values[1] * factor_) : (values[2] * factor_);
    }
};

/**
 * 示例业务表达式10: 统计分析表达式
 * 计算四列数据的统计特征
 * 计算公式: result = mean + std_dev * skewness_factor
 */
struct StatisticalExpression {
    float skewness_factor_;

    __host__ __device__ StatisticalExpression(float skew_factor = 0.5f) 
        : skewness_factor_(skew_factor) {}
    
    __host__ __device__ float operator()(const float values[4]) const {
        // 计算均值
        float mean = (values[0] + values[1] + values[2] + values[3]) / 4.0f;
        
        // 计算标准差
        float sum_sq_diff = 0.0f;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            float diff = values[i] - mean;
            sum_sq_diff += diff * diff;
        }
        float std_dev = sqrtf(sum_sq_diff / 4.0f);
        
        return mean + std_dev * skewness_factor_;
    }
};

/**
 * 额外的简单表达式结构体
 */

// 两列相加
struct TwoColumnAdd {
    __host__ __device__ float operator()(const float values[2]) const {
        return values[0] + values[1];
    }
};

// 三列平均
struct ThreeColumnAverage {
    __host__ __device__ float operator()(const float values[3]) const {
        return (values[0] + values[1] + values[2]) / 3.0f;
    }
};

// 两列相乘
struct TwoColumnMultiply {
    __host__ __device__ float operator()(const float values[2]) const {
        return values[0] * values[1];
    }
};

/**
 * 通用的模板表达式，可以适用于任意列数的简单操作
 */
template<int N>
struct ColumnSum {
    __host__ __device__ float operator()(const float values[N]) const {
        float sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < N; ++i) {
            sum += values[i];
        }
        return sum;
    }
};

template<int N>
struct ColumnAverage {
    __host__ __device__ float operator()(const float values[N]) const {
        float sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < N; ++i) {
            sum += values[i];
        }
        return sum / N;
    }
};

template<int N>
struct ColumnMax {
    __host__ __device__ float operator()(const float values[N]) const {
        float max_val = values[0];
        #pragma unroll
        for (int i = 1; i < N; ++i) {
            max_val = fmaxf(max_val, values[i]);
        }
        return max_val;
    }
};

template<int N>
struct ColumnMin {
    __host__ __device__ float operator()(const float values[N]) const {
        float min_val = values[0];
        #pragma unroll
        for (int i = 1; i < N; ++i) {
            min_val = fminf(min_val, values[i]);
        }
        return min_val;
    }
};

} // namespace zmm2 