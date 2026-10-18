# SAE Analysis for Spatial Reasoning Tasks

基于 SAELens，对 Qwen 在空间推理任务上的早期 SAE 试验。三个任务族的正式流程在 `src/sae_relation`、`src/sae_orientation` 和 `src/sae_procedure`。

## 📋 项目概述

本项目使用稀疏自编码器（Sparse Autoencoder, SAE）来分析和解释Qwen模型在空间推理任务中的内部表示。通过SAE，我们可以：

1. **发现可解释的特征**：将模型的高维表示分解为稀疏的、可解释的特征
2. **识别关键特征**：找出哪些SAE特征对空间坐标推理最重要
3. **进行因果干预**：通过修改特定特征来测试其因果影响
4. **可视化分析**：生成全面的可视化报告

## 🗂️ 文件结构

```
src/sae_exploratory/
├── train_sae_relation.py          # SAE训练脚本
├── analyze_sae_features.py        # SAE特征分析脚本
├── intervene_sae_features.py      # SAE特征干预实验脚本
├── visualize_sae_results.py       # 结果可视化脚本
├── run_full_pipeline.sh           # 完整流程运行脚本
├── run_train_sae.sh               # 单独训练SAE
├── run_analyze_sae.sh             # 单独分析SAE特征
├── run_intervention.sh            # 单独运行干预实验
├── run_visualization.sh           # 单独生成可视化
└── README.md                      # 本文档
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install -r requirements.txt   # 仓库根目录
```

### 2. 运行完整流程

最简单的方法是运行完整的pipeline：

```bash
cd src/sae_exploratory
chmod +x run_full_pipeline.sh
./run_full_pipeline.sh
```

这将依次执行：
1. 训练SAE
2. 分析SAE特征
3. 运行特征干预实验
4. 生成可视化报告

### 3. 分步运行

如果需要更细粒度的控制，可以分步运行：

#### 步骤1: 训练SAE

```bash
python train_sae_relation.py \
    --model_name "Qwen/Qwen2.5-7B-Instruct" \
    --train_data_file "../data_generation/task_family_1/spatial_procedure_dataset_EN_with_prompt.json" \
    --layer 15 \
    --expansion_factor 8 \
    --batch_size 4 \
    --num_tokens 100000 \
    --l1_coefficient 0.001 \
    --output_dir "./sae_checkpoints" \
    --device "cuda:0"
```

**参数说明**：
- `--model_name`: 模型名称（需在config.py中配置路径）
- `--train_data_file`: 训练数据文件路径
- `--layer`: 要分析的层编号（通常选择probe实验中R²最高的层）
- `--expansion_factor`: SAE扩展因子（推荐4-16）
- `--batch_size`: 批次大小
- `--num_tokens`: 训练使用的token数量
- `--l1_coefficient`: L1稀疏性系数（控制稀疏度）
- `--output_dir`: 输出目录
- `--device`: 设备（cuda:0/cuda:1等）

#### 步骤2: 分析SAE特征

```bash
python analyze_sae_features.py \
    --model_name "Qwen/Qwen2.5-7B-Instruct" \
    --sae_path "./sae_checkpoints/sae_layer15_exp8_XXXXXX/sae_final.pt" \
    --train_data_file "../data_generation/task_family_1/spatial_procedure_dataset_EN_with_prompt.json" \
    --test_data_file "../data_generation/task_family_1/spatial_procedure_dataset_EN_test_with_prompt.json" \
    --output_dir "./sae_analysis" \
    --device "cuda:0" \
    --top_k 50
```

**参数说明**：
- `--sae_path`: 训练好的SAE模型路径
- `--train_data_file`: 训练数据（用于训练probe）
- `--test_data_file`: 测试数据（用于评估）
- `--top_k`: 保存和分析的top特征数量

**输出**：
- `sae_analysis_layerX.json`: 分析结果JSON
- `feature_importance_layerX.png`: 特征重要性分布图
- `coefficients_heatmap_layerX.png`: 各维度系数热力图
- `importance_vs_freq_layerX.png`: 重要性vs激活频率散点图

#### 步骤3: 特征干预实验

```bash
python intervene_sae_features.py \
    --model_name "Qwen/Qwen2.5-7B-Instruct" \
    --sae_path "./sae_checkpoints/sae_layer15_exp8_XXXXXX/sae_final.pt" \
    --test_data_file "../data_generation/task_family_1/spatial_procedure_dataset_EN_test_with_prompt.json" \
    --analysis_results "./sae_analysis/sae_analysis_layer15.json" \
    --output_dir "./sae_intervention" \
    --device "cuda:0" \
    --num_samples 100 \
    --intervention_magnitude 2.0
```

**参数说明**：
- `--analysis_results`: 步骤2生成的分析结果
- `--num_samples`: 测试样本数量
- `--intervention_magnitude`: 干预强度（倍数）

**输出**：
- `intervention_results_layerX.json`: 干预实验结果JSON
- `intervention_effects_layerX.png`: 干预效果对比图

#### 步骤4: 可视化

```bash
python visualize_sae_results.py \
    --analysis_results "./sae_analysis/sae_analysis_layer15.json" \
    --intervention_results "./sae_intervention/intervention_results_layer15.json" \
    --output_dir "./sae_visualizations"
```

**输出**：
- `performance_comparison.png`: 性能对比图
- `sparsity_analysis.png`: 稀疏性分析图
- `top_features_details.png`: Top特征详细信息
- `importance_vs_frequency.png`: 重要性vs频率散点图
- `intervention_results.png`: 干预实验结果
- `analysis_report.txt`: 文本格式的分析报告

## 📊 实验结果解读

### 1. 性能指标

- **R² Score**: 使用SAE特征进行Ridge回归的R²分数
  - > 0.8: 优秀，SAE特征很好地捕获了空间信息
  - 0.5-0.8: 良好
  - < 0.5: 需要调整参数或选择其他层

- **MAE**: 平均绝对误差（越小越好）

- **L0 (Sparsity)**: 平均激活的特征数量
  - 理想情况：远小于`d_sae`，表示稀疏性好
  - 如果接近`d_sae`，需要增大`l1_coefficient`

### 2. 特征重要性

- **Top Features**: 对空间推理最重要的特征
- **Per-Dimension Features**: 对X/Y/Z轴分别重要的特征
- **Activation Frequency**: 特征激活频率
  - 高重要性+低频率：可能是关键的"概念"特征
  - 高重要性+高频率：通用的重要特征

### 3. 干预实验

- **Change in Hidden State**: 修改特征后隐藏状态的变化
  - 高于random baseline：该特征确实重要（因果性）
  - 接近或低于baseline：该特征可能不是因果性的

## 🔧 参数调优建议

### SAE训练参数

1. **expansion_factor** (扩展因子)
   - 推荐值：4, 8, 16
   - 更大的值 → 更多特征，但可能过拟合
   - 更小的值 → 更少特征，但可能欠拟合

2. **l1_coefficient** (L1系数)
   - 推荐值：0.0001 - 0.01
   - 更大的值 → 更稀疏，但重构误差更大
   - 更小的值 → 更密集，重构更好但可解释性差

3. **num_tokens** (训练token数)
   - 推荐值：50K - 500K
   - 更多token → 更好的泛化，但训练时间更长

### 选择合适的层

根据probe实验结果，选择R²最高的层。通常：
- Qwen2.5-7B: 第10-20层
- 更大的模型: 中间偏后的层

### 干预实验参数

- **intervention_magnitude**: 1.5 - 3.0
  - 太小：效果不明显
  - 太大：可能破坏表示

## 📈 实验工作流建议

### 标准工作流

1. **Probe实验** (已完成)
   - 确定最佳层

2. **SAE训练与调优**
   - 先用小规模（num_tokens=10K）快速测试
   - 观察重构误差和稀疏性
   - 调整l1_coefficient找到平衡点
   - 确定参数后进行完整训练

3. **特征分析**
   - 识别top features
   - 分析每个维度的关键特征

4. **干预验证**
   - 验证特征的因果性
   - 对比与随机基线

5. **可视化与报告**
   - 生成综合报告
   - 解读结果

### 多层对比

如果想对比多个层的SAE：

```bash
for layer in 10 15 20; do
    python train_sae_relation.py --layer $layer ...
    python analyze_sae_features.py --sae_path ... --layer $layer ...
done
```

### 多语言对比

对比英文/中文/阿拉伯语数据：

```bash
# 英文
./run_full_pipeline.sh  # 使用EN数据

# 中文
# 修改脚本中的数据路径为CN数据，然后运行
./run_full_pipeline.sh

# 阿拉伯语
# 修改脚本中的数据路径为AR数据，然后运行
./run_full_pipeline.sh
```

## 🐛 常见问题

### Q1: CUDA out of memory

**解决方案**：
- 减小`batch_size`
- 减小`expansion_factor`
- 使用更大的GPU
- 分批处理数据

### Q2: 重构误差很大

**原因**：
- `l1_coefficient`太大

**解决方案**：
- 减小`l1_coefficient`（例如从0.01降到0.001）
- 增加`num_tokens`

### Q3: 稀疏性不够（L0太大）

**原因**：
- `l1_coefficient`太小

**解决方案**：
- 增大`l1_coefficient`
- 确保`expansion_factor`足够大

### Q4: SAE特征的R²远低于原始hidden state的R²

这是正常的！因为：
- SAE是有损压缩
- 稀疏性约束导致信息损失
- 如果SAE的R²是原始的70-90%，已经很好了

### Q5: 干预实验没有明显效果

**可能原因**：
1. `intervention_magnitude`太小
2. 选择的特征不是因果性的（只是相关性）
3. 模型的表示是分布式的，单个特征影响有限

**建议**：
- 增大`intervention_magnitude`
- 尝试同时干预多个相关特征
- 使用ablation（置零）而不是增强

## 📚 参考文献

1. **SAELens**: https://github.com/jbloomAus/SAELens
2. **Transformer Lens**: https://github.com/neelnanda-io/TransformerLens
3. **Sparse Autoencoders**: Anthropic's work on mechanistic interpretability

## 📝 引用

如果使用本代码，请引用：

```bibtex
@inproceedings{an2026spatial-repr,
  title     = {大语言模型空间推理能力的内部表征机制研究},
  author    = {安纪元 and 杨麟儿 and 王梦焰 and 鲁鹿鸣 and 安维华 and 杨尔弘},
  booktitle = {第二十五届中国计算语言学大会 (CCL 2026)},
  year      = {2026}
}
```

## 📞 联系

如有问题或建议，请提交issue或联系项目维护者。

---

**最后更新**: 2026-01-05




