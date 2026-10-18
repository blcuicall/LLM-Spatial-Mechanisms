# 补充实验脚本使用指南

## 📋 概述

本目录包含 6 个补充实验脚本，用于更全面地验证空间特征的因果作用。这些实验是对基础 ablation 实验的重要补充，能够提供更强的科学证据。

## 🎯 实验列表

| 实验 | 脚本名称 | 目的 | 核心价值 |
|-----|---------|------|---------|
| **1. Dose-Response** | `dose_response_experiment.py` | 证明性能单调下降 | 因果关系的剂量依赖性 |
| **2. Keep-Spatial-Only** | `keep_spatial_only_experiment.py` | 测试充分性 | 区分必要性 vs 充分性 |
| **3. Statistical Significance** | `statistical_significance_experiment.py` | 显著性检验 | 排除随机噪声 |
| **4. Ablation Robustness** | `ablation_robustness_experiment.py` | 测试稳健性 | 排除 OOD artifact |
| **5. Layer Localization** | `layer_localization_experiment.py` | 定位关键层 | 找到瓶颈层 |
| **6. Matched Random Control** | `matched_random_control_experiment.py` | 匹配对照 | 最严格的控制 |

## 🚀 快速开始

### 前置条件

1. **已完成基础实验**：
   ```bash
   # 确保已运行 SAE 训练和分析
   ls ../sae_saelens/sae_results/L8_F2048_*/sae_checkpoint.pt
   ls ../sae_saelens/sae_results/L8_F2048_*/analysis/dimension_features.json
   ```

2. **（可选）梯度归因结果**：
   ```bash
   # 如果要使用梯度排序，需要先运行
   cd ../sae_saelens
   python gradient_attribution.py -c ./sae_results/L8_F2048_*/ \
     -e ../../data_generation_task_3/spatial_procedure_dataset_EN_test_with_prompt.json \
     --max_samples 200
   ```

### 一键运行所有实验

```bash
cd src/supplementary

# 使用运行脚本（推荐）
bash run_all_experiments.sh ../sae_saelens/sae_results/L8_F2048_20241231_123456
```

### 逐个运行实验

#### 实验 1: Dose-Response（剂量-反应）

**目的**：证明越多空间特征被移除 → 性能越差（单调性）

```bash
python dose_response_experiment.py \
  -m "Qwen/Qwen2.5-7B-Instruct" \
  -c ../sae_saelens/sae_results/L8_F2048_*/ \
  -e ../../data_generation_task_3/spatial_procedure_dataset_EN_test_with_prompt.json \
  -g ../sae_saelens/sae_results/L8_F2048_*/gradient_attribution/gradient_attribution_grad_x_act_*.json \
  --k_values "5,10,20,40,80,136" \
  --ablation_type "mean" \
  --max_samples 200
```

**参数说明**：
- `--k_values`: 要测试的 top-k 值列表
- `--ablation_type`: `zero` / `mean` / `shuffle`
- `-g`: 梯度归因结果（用于排序特征），可选

**预期结果**：
- 性能随 k 单调下降
- 曲线可能有拐点（bottleneck）
- Margin 和 NLL 也应该恶化

---

#### 实验 2: Keep-Spatial-Only（充分性测试）

**目的**：测试"只保留空间特征"是否足够

```bash
python keep_spatial_only_experiment.py \
  -m "Qwen/Qwen2.5-7B-Instruct" \
  -c ../sae_saelens/sae_results/L8_F2048_*/ \
  -e ../../data_generation_task_3/spatial_procedure_dataset_EN_test_with_prompt.json \
  --ablation_type "mean" \
  --max_samples 200
```

**解读标准**：
- **接近随机 (≈25%)**：必要但不充分（门控作用）
- **明显高于随机 (40-60%)**：承载主要信息
- **接近基线 (>80%)**：几乎决定性（shortcut 风险）

---

#### 实验 3: Statistical Significance（统计检验）

**目的**：用显著性检验证明结果不是噪声

```bash
python statistical_significance_experiment.py \
  -m "Qwen/Qwen2.5-7B-Instruct" \
  -c ../sae_saelens/sae_results/L8_F2048_*/ \
  -e ../../data_generation_task_3/spatial_procedure_dataset_EN_test_with_prompt.json \
  --n_random_trials 5 \
  --max_samples 200
```

**输出指标**：
- Paired t-test 和 Wilcoxon test
- Cohen's d (效应量)
- p-value < 0.05 → 显著

**价值**：
- 提供 p-value 和置信区间
- 可以写进论文："p < 0.001, Cohen's d = 0.8"

---

#### 实验 4: Ablation Robustness（稳健性测试）

**目的**：排除"置零导致 OOD"的质疑

```bash
python ablation_robustness_experiment.py \
  -m "Qwen/Qwen2.5-7B-Instruct" \
  -c ../sae_saelens/sae_results/L8_F2048_*/ \
  -e ../../data_generation_task_3/spatial_procedure_dataset_EN_test_with_prompt.json \
  --max_samples 200
```

**测试 3 种 ablation 方式**：
1. **Zero**: 置零（可能 OOD）
2. **Mean**: 替换为均值（更自然）
3. **Shuffle**: 打乱（保留分布）

**预期**：
- 如果 3 种方式都显示"spatial 更伤" → 结论稳健
- 如果只有 zero 有效 → 可能是 OOD artifact

---

#### 实验 5: Layer Localization（层级定位）

**目的**：找出哪一层的空间特征最关键

```bash
python layer_localization_experiment.py \
  -m "Qwen/Qwen2.5-7B-Instruct" \
  -s ../sae_saelens/sae_results/ \
  -e ../../data_generation_task_3/spatial_procedure_dataset_EN_test_with_prompt.json \
  --ablation_type "mean" \
  --max_samples 200
```

**前置条件**：
- 需要多个层的 SAE 结果（例如 L4, L6, L8, L10）
- 目录格式：`sae_results/L{layer}_F{features}_*/`

**输出**：
- 每层的 spatial drop 和 random drop
- 按 effect size 排序
- 可视化：layer → ΔAcc 曲线

**价值**：
- 定位"关键瓶颈层"
- 论文可以写："Layer 8 is the critical bottleneck for spatial reasoning"

---

#### 实验 6: Matched Random Control（匹配对照）

**目的**：最严格的对照实验，控制激活统计量

```bash
python matched_random_control_experiment.py \
  -m "Qwen/Qwen2.5-7B-Instruct" \
  -c ../sae_saelens/sae_results/L8_F2048_*/ \
  -e ../../data_generation_task_3/spatial_procedure_dataset_EN_test_with_prompt.json \
  -g ../sae_saelens/sae_results/L8_F2048_*/gradient_attribution/gradient_attribution_grad_x_act_*.json \
  --n_random_samples 10 \
  --matching_method "both" \
  --max_samples 200
```

**匹配内容**：
- 激活值的均值、方差
- 稀疏度
- （可选）Attribution 分布

**价值**：
- 排除"spatial features 只是碰巧激活更强"
- 如果匹配后仍显著 → 一锤定音的证据

---

## 📊 结果解读

### 成功的实验应该看到：

1. **Dose-Response**:
   - 单调递减曲线
   - 最终 drop > 15%

2. **Keep-Spatial-Only**:
   - 保留率 40-70%（必要但不充分）

3. **Statistical Significance**:
   - p < 0.05 或 p < 0.01
   - Cohen's d > 0.5

4. **Ablation Robustness**:
   - 3 种方式一致

5. **Layer Localization**:
   - 清晰的瓶颈层

6. **Matched Random Control**:
   - Spatial / Matched ratio > 1.5x

## 🎓 论文写作建议

### Results Section 可以这样组织：

```
4. Supplementary Causal Experiments

4.1 Dose-Response Analysis
我们逐步增加被移除的空间特征数量（k=5,10,20,...），发现性能单调下降（图X）...

4.2 Sufficiency Test
我们测试了只保留空间特征的充分性。结果显示保留率为 X%，表明空间特征是
[必要但不充分 / 承载主要信息 / 几乎决定性]...

4.3 Statistical Significance
我们使用 paired t-test 验证了差异的显著性（p < 0.001, Cohen's d = 0.8）...

4.4 Robustness to Ablation Type
我们测试了 3 种 ablation 方式（zero, mean, shuffle），结果一致（图X）...

4.5 Layer-wise Localization
我们发现 Layer X 是空间推理的关键瓶颈层（图X）...

4.6 Matched Control Experiments
即使控制了激活统计量，空间特征仍显示特异性（ratio = X）...
```

### 可以这样写 Abstract：

```
"We conducted comprehensive causal experiments including dose-response 
analysis, statistical significance tests, and matched control experiments. 
Ablating spatial features led to a 27% accuracy drop (p < 0.001), 
significantly more than matched random controls (6% drop, p = 0.32). 
This provides strong evidence that LLMs develop dedicated, causally-relevant 
features for spatial reasoning."
```

## 🔧 故障排查

### 问题 1: 没有梯度归因文件

**症状**：`-g` 参数指向的文件不存在

**解决**：
```bash
# 先运行梯度归因
cd ../sae_saelens
python gradient_attribution.py -c ./sae_results/L8_F*/ -e <data_file> --max_samples 200
```

或者不使用 `-g` 参数（会使用随机排序）

### 问题 2: 找不到多层 SAE 结果

**症状**：Layer Localization 找不到其他层

**解决**：
- 确保训练了多个层的 SAE
- 或者跳过这个实验

### 问题 3: 运行时间太长

**解决**：
```bash
# 减少样本数
--max_samples 50  # 快速测试
--max_samples 100 # 中等
--max_samples 200 # 论文质量
```

### 问题 4: 显存不足

**解决**：
```bash
# 使用 CPU（较慢）
--device cpu

# 或者减少样本数
--max_samples 50
```

## 📁 输出文件结构

```
sae_results/L8_F2048_*/
├── dose_response_results/
│   ├── dose_response_mean_20260103_*.json
│   ├── dose_response_mean_20260103_*.png
│   └── dose_response_summary_*.txt
├── keep_spatial_only_results/
│   ├── keep_spatial_only_mean_*.json
│   └── keep_spatial_only_mean_*.png
├── statistical_significance_results/
│   ├── statistical_tests_*.json
│   ├── statistical_tests_*.png
│   └── statistical_summary_*.txt
├── ablation_robustness_results/
│   ├── ablation_robustness_*.json
│   └── ablation_robustness_*.png
└── matched_random_results/
    ├── matched_random_control_*.json
    └── matched_random_control_*.png
```

## ⏱️ 预计运行时间

| 实验 | 样本数 | GPU 时间 | CPU 时间 |
|-----|-------|---------|---------|
| Dose-Response | 200 | 15-20分钟 | 60-90分钟 |
| Keep-Spatial-Only | 200 | 10-15分钟 | 40-60分钟 |
| Statistical Significance | 200 | 15-20分钟 | 60-90分钟 |
| Ablation Robustness | 200 | 20-25分钟 | 80-120分钟 |
| Layer Localization | 200 (×N层) | 10N-15N分钟 | 40N-60N分钟 |
| Matched Random Control | 200 | 10-15分钟 | 40-60分钟 |

**总计**（单层）：约 **80-115 分钟 (GPU)** 或 **5-8 小时 (CPU)**

## 💡 最佳实践

1. **先用小样本测试**：
   ```bash
   --max_samples 20  # 确认脚本能运行
   ```

2. **再用中等样本**：
   ```bash
   --max_samples 100  # 查看初步结果
   ```

3. **最后全量运行**：
   ```bash
   --max_samples 500  # 论文质量
   ```

4. **保存结果**：
   ```bash
   # 实验完成后，备份结果
   cp -r sae_results backup_$(date +%Y%m%d)
   ```

5. **记录实验**：
   - 记录使用的参数
   - 记录运行时间
   - 截图关键结果

## 📚 参考文献

这些实验设计参考了：

1. **Causal Mediation Analysis**
   - Pearl, J. (2001). Direct and indirect effects.
   
2. **Feature Attribution**
   - Sundararajan et al. (2017). Axiomatic attribution for deep networks.

3. **Ablation Studies**
   - Meyes et al. (2019). Ablation studies in artificial neural networks.

## 🤝 贡献

如果发现 bug 或有改进建议，请：
1. 在 GitHub 开 issue
2. 提交 pull request
3. 联系维护者

## 📄 许可证

MIT License

---

**祝实验顺利！如果这些实验成功，你将拥有非常强的因果证据来支持你的论文！** 🚀




