#pragma once

#include "mixed_types.h"
#include <functional>
#include <memory>

namespace zmm {

// 基础混合操作类
class BaseMixedOperation : public IMixedOperation {
public:
    virtual ~BaseMixedOperation() = default;
};

// 简单加法操作（只处理浮点列）
class MixedAddOperation : public BaseMixedOperation {
public:
    __device__ __host__ float execute(const MixedRowData& row) const override {
        float result = 0.0f;
        for (int i = 0; i < row.num_columns; ++i) {
            if (row.types[i] == ColumnDataType::FLOAT) {
                result += row.getFloat(i);
            }
        }
        return result;
    }
    
    const char* getName() const override {
        return "MixedAddOperation";
    }
};

// 字符串长度统计操作
class StringLengthSumOperation : public BaseMixedOperation {
public:
    __device__ __host__ float execute(const MixedRowData& row) const override {
        float result = 0.0f;
        for (int i = 0; i < row.num_columns; ++i) {
            if (row.types[i] == ColumnDataType::STRING) {
                GPUString str = row.getString(i);
                result += static_cast<float>(str.length);
            }
        }
        return result;
    }
    
    const char* getName() const override {
        return "StringLengthSumOperation";
    }
};

// 混合计算操作：浮点求和 + 字符串长度加权
class MixedSumWithStringWeightOperation : public BaseMixedOperation {
private:
    float string_weight_;
    
public:
    __device__ __host__ MixedSumWithStringWeightOperation(float string_weight = 0.1f) 
        : string_weight_(string_weight) {}
    
    __device__ __host__ float execute(const MixedRowData& row) const override {
        float float_sum = 0.0f;
        float string_sum = 0.0f;
        
        for (int i = 0; i < row.num_columns; ++i) {
            if (row.types[i] == ColumnDataType::FLOAT) {
                float_sum += row.getFloat(i);
            } else if (row.types[i] == ColumnDataType::STRING) {
                GPUString str = row.getString(i);
                string_sum += static_cast<float>(str.length);
            }
        }
        
        return float_sum + string_sum * string_weight_;
    }
    
    const char* getName() const override {
        return "MixedSumWithStringWeightOperation";
    }
};

// 条件计算操作：根据字符串内容调整浮点值
class ConditionalMixedOperation : public BaseMixedOperation {
public:
    __device__ __host__ float execute(const MixedRowData& row) const override {
        float base_value = 0.0f;
        float multiplier = 1.0f;
        
        // 计算基础浮点值
        for (int i = 0; i < row.num_columns; ++i) {
            if (row.types[i] == ColumnDataType::FLOAT) {
                base_value += row.getFloat(i);
            }
        }
        
        // 根据字符串内容调整乘数
        for (int i = 0; i < row.num_columns; ++i) {
            if (row.types[i] == ColumnDataType::STRING) {
                GPUString str = row.getString(i);
                
                if (gpu_string_equals(str, "high")) {
                    multiplier *= 1.5f;
                } else if (gpu_string_equals(str, "low")) {
                    multiplier *= 0.5f;
                } else if (gpu_string_equals(str, "medium")) {
                    multiplier *= 1.0f;
                } else if (gpu_string_equals(str, "premium")) {
                    multiplier *= 2.0f;
                } else if (gpu_string_equals(str, "discount")) {
                    multiplier *= 0.8f;
                }
            }
        }
        
        return base_value * multiplier;
    }
    
    const char* getName() const override {
        return "ConditionalMixedOperation";
    }
};

// 字符串转数值操作
class StringToNumberOperation : public BaseMixedOperation {
public:
    __device__ __host__ float execute(const MixedRowData& row) const override {
        float result = 0.0f;
        
        for (int i = 0; i < row.num_columns; ++i) {
            if (row.types[i] == ColumnDataType::FLOAT) {
                result += row.getFloat(i);
            } else if (row.types[i] == ColumnDataType::STRING) {
                GPUString str = row.getString(i);
                result += gpu_string_to_float(str);
            }
        }
        
        return result;
    }
    
    const char* getName() const override {
        return "StringToNumberOperation";
    }
};

// 字符串分类操作
class StringCategoryOperation : public BaseMixedOperation {
public:
    __device__ __host__ float execute(const MixedRowData& row) const override {
        float category_score = 0.0f;
        float float_bonus = 0.0f;
        
        // 计算浮点值奖励
        for (int i = 0; i < row.num_columns; ++i) {
            if (row.types[i] == ColumnDataType::FLOAT) {
                float_bonus += row.getFloat(i) * 0.01f; // 1%的奖励
            }
        }
        
        // 字符串分类评分
        for (int i = 0; i < row.num_columns; ++i) {
            if (row.types[i] == ColumnDataType::STRING) {
                GPUString str = row.getString(i);
                
                if (gpu_string_equals(str, "category_a") || gpu_string_equals(str, "excellent")) {
                    category_score += 10.0f;
                } else if (gpu_string_equals(str, "category_b") || gpu_string_equals(str, "good")) {
                    category_score += 7.0f;
                } else if (gpu_string_equals(str, "category_c") || gpu_string_equals(str, "average")) {
                    category_score += 5.0f;
                } else if (gpu_string_equals(str, "category_d") || gpu_string_equals(str, "poor")) {
                    category_score += 2.0f;
                } else {
                    category_score += 1.0f; // 默认分数
                }
            }
        }
        
        return category_score + float_bonus;
    }
    
    const char* getName() const override {
        return "StringCategoryOperation";
    }
};

// 固定长度字符串操作模板
template<int MAX_STRING_LENGTH = 256>
class FixedStringOperation : public BaseMixedOperation {
public:
    __device__ __host__ float execute(const MixedRowData& row) const override {
        float result = 0.0f;
        
        for (int i = 0; i < row.num_columns; ++i) {
            if (row.types[i] == ColumnDataType::FLOAT) {
                result += row.getFloat(i);
            } else if (row.types[i] == ColumnDataType::STRING) {
                // 假设使用固定长度字符串
                auto fixed_str = row.getFixedString<MAX_STRING_LENGTH>(i);
                result += static_cast<float>(fixed_str.length()) * 0.1f;
                
                // 特殊字符串处理
                if (gpu_string_equals(fixed_str, "special")) {
                    result *= 2.0f;
                }
            }
        }
        
        return result;
    }
    
    const char* getName() const override {
        return "FixedStringOperation";
    }
};

// 复合业务逻辑操作示例：电商评分计算
class EcommerceScoreOperation : public BaseMixedOperation {
public:
    __device__ __host__ float execute(const MixedRowData& row) const override {
        float price = 0.0f;
        float rating = 0.0f;
        float category_multiplier = 1.0f;
        float brand_bonus = 0.0f;
        
        // 假设列顺序：价格(float), 评分(float), 类别(string), 品牌(string)
        if (row.num_columns >= 4) {
            if (row.types[0] == ColumnDataType::FLOAT) {
                price = row.getFloat(0);
            }
            if (row.types[1] == ColumnDataType::FLOAT) {
                rating = row.getFloat(1);
            }
            
            if (row.types[2] == ColumnDataType::STRING) {
                GPUString category = row.getString(2);
                if (gpu_string_equals(category, "electronics")) {
                    category_multiplier = 1.2f;
                } else if (gpu_string_equals(category, "books")) {
                    category_multiplier = 1.0f;
                } else if (gpu_string_equals(category, "clothing")) {
                    category_multiplier = 1.1f;
                } else if (gpu_string_equals(category, "luxury")) {
                    category_multiplier = 1.5f;
                }
            }
            
            if (row.types[3] == ColumnDataType::STRING) {
                GPUString brand = row.getString(3);
                if (gpu_string_equals(brand, "premium")) {
                    brand_bonus = 10.0f;
                } else if (gpu_string_equals(brand, "popular")) {
                    brand_bonus = 5.0f;
                }
            }
        }
        
        // 计算综合评分：评分权重70%，价格因子20%，品牌奖励10%
        float base_score = rating * 7.0f;
        float price_factor = (price > 0) ? (100.0f / price) : 0.0f; // 价格越低分数越高
        float final_score = (base_score + price_factor * 2.0f + brand_bonus) * category_multiplier;
        
        return final_score;
    }
    
    const char* getName() const override {
        return "EcommerceScoreOperation";
    }
};

// Lambda风格的函数包装器
class LambdaMixedOperation : public BaseMixedOperation {
private:
    std::function<float(const MixedRowData&)> lambda_func_;
    const char* name_;
    
public:
    LambdaMixedOperation(
        std::function<float(const MixedRowData&)> func, 
        const char* name = "LambdaMixedOperation"
    ) : lambda_func_(func), name_(name) {}
    
    __device__ __host__ float execute(const MixedRowData& row) const override {
        // 注意：在GPU上无法直接调用std::function
        // 这个实现主要用于CPU端的测试和验证
        #ifdef __CUDA_ARCH__
        return 0.0f; // GPU上返回默认值
        #else
        return lambda_func_(row);
        #endif
    }
    
    const char* getName() const override {
        return name_;
    }
};

// 操作工厂类
class MixedOperationFactory {
public:
    enum OperationType {
        MIXED_ADD,
        STRING_LENGTH_SUM,
        MIXED_SUM_WITH_STRING_WEIGHT,
        CONDITIONAL_MIXED,
        STRING_TO_NUMBER,
        STRING_CATEGORY,
        FIXED_STRING,
        ECOMMERCE_SCORE
    };
    
    static std::unique_ptr<IMixedOperation> createOperation(OperationType type, float param = 1.0f) {
        switch (type) {
            case MIXED_ADD:
                return std::make_unique<MixedAddOperation>();
            case STRING_LENGTH_SUM:
                return std::make_unique<StringLengthSumOperation>();
            case MIXED_SUM_WITH_STRING_WEIGHT:
                return std::make_unique<MixedSumWithStringWeightOperation>(param);
            case CONDITIONAL_MIXED:
                return std::make_unique<ConditionalMixedOperation>();
            case STRING_TO_NUMBER:
                return std::make_unique<StringToNumberOperation>();
            case STRING_CATEGORY:
                return std::make_unique<StringCategoryOperation>();
            case FIXED_STRING:
                return std::make_unique<FixedStringOperation<256>>();
            case ECOMMERCE_SCORE:
                return std::make_unique<EcommerceScoreOperation>();
            default:
                return std::make_unique<MixedAddOperation>();
        }
    }
    
    // 创建自定义操作
    template<typename OperationType>
    static std::unique_ptr<IMixedOperation> createCustomOperation() {
        return std::make_unique<OperationType>();
    }
};

// GPU函数对象（用于内核调用）
struct MixedAddFunctor {
    __device__ float operator()(const MixedRowData& row) const {
        float result = 0.0f;
        for (int i = 0; i < row.num_columns; ++i) {
            if (row.types[i] == ColumnDataType::FLOAT) {
                result += row.getFloat(i);
            }
        }
        return result;
    }
};

struct StringLengthSumFunctor {
    __device__ float operator()(const MixedRowData& row) const {
        float result = 0.0f;
        for (int i = 0; i < row.num_columns; ++i) {
            if (row.types[i] == ColumnDataType::STRING) {
                GPUString str = row.getString(i);
                result += static_cast<float>(str.length);
            }
        }
        return result;
    }
};

struct ConditionalMixedFunctor {
    __device__ float operator()(const MixedRowData& row) const {
        float base_value = 0.0f;
        float multiplier = 1.0f;
        
        for (int i = 0; i < row.num_columns; ++i) {
            if (row.types[i] == ColumnDataType::FLOAT) {
                base_value += row.getFloat(i);
            } else if (row.types[i] == ColumnDataType::STRING) {
                GPUString str = row.getString(i);
                if (gpu_string_equals(str, "high")) {
                    multiplier *= 1.5f;
                } else if (gpu_string_equals(str, "low")) {
                    multiplier *= 0.5f;
                }
            }
        }
        
        return base_value * multiplier;
    }
};

struct EcommerceScoreFunctor {
    __device__ float operator()(const MixedRowData& row) const {
        if (row.num_columns < 4) return 0.0f;
        
        float price = (row.types[0] == ColumnDataType::FLOAT) ? row.getFloat(0) : 0.0f;
        float rating = (row.types[1] == ColumnDataType::FLOAT) ? row.getFloat(1) : 0.0f;
        
        float category_multiplier = 1.0f;
        if (row.types[2] == ColumnDataType::STRING) {
            GPUString category = row.getString(2);
            if (gpu_string_equals(category, "electronics")) {
                category_multiplier = 1.2f;
            } else if (gpu_string_equals(category, "luxury")) {
                category_multiplier = 1.5f;
            }
        }
        
        float brand_bonus = 0.0f;
        if (row.types[3] == ColumnDataType::STRING) {
            GPUString brand = row.getString(3);
            if (gpu_string_equals(brand, "premium")) {
                brand_bonus = 10.0f;
            }
        }
        
        float base_score = rating * 7.0f;
        float price_factor = (price > 0) ? (100.0f / price) : 0.0f;
        return (base_score + price_factor * 2.0f + brand_bonus) * category_multiplier;
    }
};

// ===== 使用字段名访问的操作（类 + Functor） =====

// 使用字段名访问的简单求和操作：price + rating
class NamedPriceRatingSumOperation : public BaseMixedOperation {
public:
    __device__ __host__ float execute(const MixedRowData& row) const override {
        float price = row.getFloat("price");
        float rating = row.getFloat("rating");
        return price + rating;
    }
    const char* getName() const override { return "NamedPriceRatingSumOperation"; }
};

// 使用字段名的电商评分：与EcommerceScoreOperation相同逻辑，但按名称取列
class NamedEcommerceScoreOperation : public BaseMixedOperation {
public:
    __device__ __host__ float execute(const MixedRowData& row) const override {
        float price = row.getFloat("price");
        float rating = row.getFloat("rating");
        GPUString category = row.getString("category");
        GPUString brand = row.getString("brand");

        float category_multiplier = 1.0f;
        if (gpu_string_equals(category, "electronics")) category_multiplier = 1.2f;
        else if (gpu_string_equals(category, "luxury")) category_multiplier = 1.5f;

        float brand_bonus = 0.0f;
        if (gpu_string_equals(brand, "premium")) brand_bonus = 10.0f;

        float base_score = rating * 7.0f;
        float price_factor = (price > 0) ? (100.0f / price) : 0.0f;
        return (base_score + price_factor * 2.0f + brand_bonus) * category_multiplier;
    }
    const char* getName() const override { return "NamedEcommerceScoreOperation"; }
};

// Device functor: 使用字段名访问的简单求和
struct NamedPriceRatingSumFunctor {
    __device__ float operator()(const MixedRowData& row) const {
        return row.getFloat("price") + row.getFloat("rating");
    }
};

// Device functor: 使用字段名访问的电商评分
struct NamedEcommerceScoreFunctor {
    __device__ float operator()(const MixedRowData& row) const {
        float price = row.getFloat("price");
        float rating = row.getFloat("rating");
        GPUString category = row.getString("category");
        GPUString brand = row.getString("brand");

        float category_multiplier = 1.0f;
        if (gpu_string_equals(category, "electronics")) category_multiplier = 1.2f;
        else if (gpu_string_equals(category, "luxury")) category_multiplier = 1.5f;

        float brand_bonus = 0.0f;
        if (gpu_string_equals(brand, "premium")) brand_bonus = 10.0f;

        float base_score = rating * 7.0f;
        float price_factor = (price > 0) ? (100.0f / price) : 0.0f;
        return (base_score + price_factor * 2.0f + brand_bonus) * category_multiplier;
    }
};

// ===== 多列输出操作示例 =====

// 示例1：将价格和评分分别输出到两列（float类型）
class PriceRatingSplitOperation : public IMultiColumnOperation {
public:
    int getNumOutputColumns() const override { return 2; }
    
    void getOutputTypes(ColumnDataType* types) const override {
        types[0] = ColumnDataType::FLOAT;  // 价格
        types[1] = ColumnDataType::FLOAT;  // 评分
    }
    
    __device__ __host__ void execute(const MixedRowData& row, MultiColumnOutput& output) const override {
        // 假设输入：第0列是价格，第1列是评分
        float price = row.getFloat("price");
        float rating = row.getFloat("rating");
        
        output.setFloat(0, price);
        output.setFloat(1, rating);
    }
    
    const char* getName() const override { return "PriceRatingSplitOperation"; }
};

// 示例2：计算多个统计值（总和、最大值、最小值）
class MultiStatisticsOperation : public IMultiColumnOperation {
public:
    int getNumOutputColumns() const override { return 3; }
    
    void getOutputTypes(ColumnDataType* types) const override {
        types[0] = ColumnDataType::FLOAT;  // 总和
        types[1] = ColumnDataType::FLOAT;  // 最大值
        types[2] = ColumnDataType::FLOAT;  // 最小值
    }
    
    __device__ __host__ void execute(const MixedRowData& row, MultiColumnOutput& output) const override {
        float sum = 0.0f;
        float max_val = -1e30f;
        float min_val = 1e30f;
        
        for (int i = 0; i < row.num_columns; ++i) {
            if (row.types[i] == ColumnDataType::FLOAT) {
                float val = row.getFloat(i);
                sum += val;
                max_val = (val > max_val) ? val : max_val;
                min_val = (val < min_val) ? val : min_val;
            }
        }
        
        output.setFloat(0, sum);
        output.setFloat(1, max_val);
        output.setFloat(2, min_val);
    }
    
    const char* getName() const override { return "MultiStatisticsOperation"; }
};

// 示例3：混合类型输出 - 计算总价(float)和数量(int)
class PriceQuantityOperation : public IMultiColumnOperation {
public:
    int getNumOutputColumns() const override { return 2; }
    
    void getOutputTypes(ColumnDataType* types) const override {
        types[0] = ColumnDataType::FLOAT;  // 总价
        types[1] = ColumnDataType::INT;    // 数量
    }
    
    __device__ __host__ void execute(const MixedRowData& row, MultiColumnOutput& output) const override {
        float price = row.getFloat("price");
        float quantity = row.getFloat("quantity");
        
        output.setFloat(0, price * quantity);  // 总价
        output.setInt(1, static_cast<int>(quantity));  // 数量转为整数
    }
    
    const char* getName() const override { return "PriceQuantityOperation"; }
};

// 示例4：电商评分的多维分析 - 返回多个评分维度
class MultiDimensionScoreOperation : public IMultiColumnOperation {
public:
    int getNumOutputColumns() const override { return 4; }
    
    void getOutputTypes(ColumnDataType* types) const override {
        types[0] = ColumnDataType::FLOAT;  // 基础评分
        types[1] = ColumnDataType::FLOAT;  // 价格评分
        types[2] = ColumnDataType::FLOAT;  // 品牌评分
        types[3] = ColumnDataType::FLOAT;  // 综合评分
    }
    
    __device__ __host__ void execute(const MixedRowData& row, MultiColumnOutput& output) const override {
        float price = row.getFloat("price");
        float rating = row.getFloat("rating");
        GPUString category = row.getString("category");
        GPUString brand = row.getString("brand");
        
        // 基础评分
        float base_score = rating * 7.0f;
        
        // 价格评分
        float price_score = (price > 0) ? (100.0f / price) * 2.0f : 0.0f;
        
        // 品牌评分
        float brand_score = 0.0f;
        if (gpu_string_equals(brand, "premium")) brand_score = 10.0f;
        else if (gpu_string_equals(brand, "popular")) brand_score = 5.0f;
        
        // 类别乘数
        float category_multiplier = 1.0f;
        if (gpu_string_equals(category, "electronics")) category_multiplier = 1.2f;
        else if (gpu_string_equals(category, "luxury")) category_multiplier = 1.5f;
        
        // 综合评分
        float total_score = (base_score + price_score + brand_score) * category_multiplier;
        
        output.setFloat(0, base_score);
        output.setFloat(1, price_score);
        output.setFloat(2, brand_score);
        output.setFloat(3, total_score);
    }
    
    const char* getName() const override { return "MultiDimensionScoreOperation"; }
};

// ===== 多列输出的Functor版本（用于模板内核） =====

// Functor: 价格和评分分离
struct PriceRatingSplitFunctor {
    __device__ void operator()(const MixedRowData& row, MultiColumnOutput& output) const {
        float price = row.getFloat("price");
        float rating = row.getFloat("rating");
        
        output.setFloat(0, price);
        output.setFloat(1, rating);
    }
};

// Functor: 多统计值
struct MultiStatisticsFunctor {
    __device__ void operator()(const MixedRowData& row, MultiColumnOutput& output) const {
        float sum = 0.0f;
        float max_val = -1e30f;
        float min_val = 1e30f;
        
        for (int i = 0; i < row.num_columns; ++i) {
            if (row.types[i] == ColumnDataType::FLOAT) {
                float val = row.getFloat(i);
                sum += val;
                max_val = (val > max_val) ? val : max_val;
                min_val = (val < min_val) ? val : min_val;
            }
        }
        
        output.setFloat(0, sum);
        output.setFloat(1, max_val);
        output.setFloat(2, min_val);
    }
};

// Functor: 多维评分
struct MultiDimensionScoreFunctor {
    __device__ void operator()(const MixedRowData& row, MultiColumnOutput& output) const {
        float price = row.getFloat("price");
        float rating = row.getFloat("rating");
        GPUString category = row.getString("category");
        GPUString brand = row.getString("brand");
        
        float base_score = rating * 7.0f;
        float price_score = (price > 0) ? (100.0f / price) * 2.0f : 0.0f;
        
        float brand_score = 0.0f;
        if (gpu_string_equals(brand, "premium")) brand_score = 10.0f;
        else if (gpu_string_equals(brand, "popular")) brand_score = 5.0f;
        
        float category_multiplier = 1.0f;
        if (gpu_string_equals(category, "electronics")) category_multiplier = 1.2f;
        else if (gpu_string_equals(category, "luxury")) category_multiplier = 1.5f;
        
        float total_score = (base_score + price_score + brand_score) * category_multiplier;
        
        output.setFloat(0, base_score);
        output.setFloat(1, price_score);
        output.setFloat(2, brand_score);
        output.setFloat(3, total_score);
    }
};

} // namespace zmm
