# SAE特征语义可解释性分析

## 问题背景

**核心问题**：SAE特征能否解析出语义？

基于EN和CN实验的对比分析，我们发现：
- ✅ 英文SAE表现更好（R²=0.113 vs 0.074）
- ⚠️ 但**两种语言的Top特征完全不同**（零重叠）
- ❓ 这引发疑问：特征是否真的编码了深层语义？

## 解决方案：最大激活样本分析

通过分析哪些样本最大程度地激活特定特征，判断特征是否对应特定语义概念。

### 期望发现

如果Feature 88728（EN顶级Y轴特征）真的编码"前后"语义：
- 最大激活样本应该都包含"in front of"、"behind"等词
- 或者目标对象都在Y轴上有特定位置关系

## 使用方法

### 1. 快速运行（已修复模型加载问题）

```bash
cd src/sae_relation

# 运行完整分析（英文+中文）
bash run_semantic_analysis.sh
```

**说明**：
- ✅ 现在会自动从`config.py`读取本地模型路径
- ✅ 不会重新下载模型
- ✅ 模型目录由仓库根目录的 `config.py` 或环境变量 `LLM_ROOT` 决定

### 2. 单独分析特定特征

```bash
# 只分析英文Y轴特征（最重要的）
python analyze_feature_semantics.py \
  --sae_path "../../artifacts/sae_relation/outputs_20260105_202027/sae_checkpoints/sae_layer20_exp32_20260105_202045/sae_final.pt" \
  --data_path "../data_generation/task_family_1/data_relation/spatial_reasoning_dataset_EN_test.json" \
  --feature_ids "88728" \
  --model_name "Qwen/Qwen2.5-7B-Instruct" \
  --layer 20 \
  --top_k 20 \
  --output_dir "./test_feature_88728" \
  --device cuda
```

## 输出结果

### 文件结构
```
feature_semantic_analysis_EN/
├── overall/
│   ├── feature_88728_analysis.json
│   ├── feature_81718_analysis.json
│   └── ...
├── y_axis/
├── x_axis/
└── z_axis/

feature_semantic_analysis_CN/
└── ...
```

### JSON内容示例

```json
{
  "top_samples": [
    {
      "sample": {...},
      "activation": 2.456,
      "text": "The ball is in front of the box."
    },
    ...
  ],
  "spatial_keywords": {
    "Y-front": 15,
    "Y-behind": 5
  },
  "activation_stats": {
    "top_mean": 2.134,
    "nonzero_ratio": 0.48
  }
}
```

## 预期分析结果

### 情况1：特征确实有语义

**表现**：
- Top-20激活样本中，80%+包含同类空间关键词
- 例如：Feature 88728 → 18/20样本含"front"或"behind"
- 不同特征激活不同类型的关键词

**结论**：✅ 特征捕捉到了特定空间关系的语义

### 情况2：特征只是统计模式

**表现**：
- Top激活样本没有明显的语义一致性
- 关键词分布随机
- 激活强度与语义无关

**结论**：❌ 特征更多是表层统计相关，非深层语义

### 情况3：分布式表征

**表现**：
- 单个特征语义模糊
- 但不同特征的组合可能有意义

**结论**：⚠️ 需要分析特征组合

## 关键问题将被回答

1. **维度特异性**：Y轴特征是否真的对应"前后"关系？
2. **跨语言一致性**：EN和CN是否用不同特征编码相同概念？
3. **语义vs统计**：特征编码的是深层概念还是词汇共现？

## 后续分析建议

### 如果发现有语义
1. 可视化特征激活热力图
2. 分析特征的因果作用（消融实验）
3. 寻找跨语言的功能相似特征对

### 如果发现无语义
1. 尝试更深的层（Layer 25, 28）
2. 分析特征组合的语义
3. 改进SAE训练（更多数据、调参）

## 技术细节

### 修复记录
- ❌ 初版：尝试重新下载HF模型
- ✅ 当前：使用`config.py`中的本地路径
- ✅ 方法：transformers加载 → 转换为HookedTransformer

### 依赖
- transformer_lens
- sae_lens
- transformers
- torch

## 相关文档

- 详细分析：`semantic_interpretability_analysis.md`
- 实验对比：`comparative_analysis_EN_vs_CN.md`
- 快速总结：`quick_summary_EN_vs_CN.txt`

