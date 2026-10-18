# SAE分析快速开始指南

## 📁 项目结构

```
src/sae_exploratory/
├── train_sae_relation.py          # SAE训练脚本
├── analyze_sae_features.py        # SAE特征分析脚本  
├── intervene_sae_features.py      # SAE特征干预实验
├── visualize_sae_results.py       # 结果可视化
├── test_saelens.py               # SAELens测试脚本
├── run_full_pipeline.sh          # 完整流程
└── run_*.sh                      # 单步运行脚本
```

## 🚀 快速运行

### 方法1: 完整流程（推荐）

```bash
cd src/sae_exploratory
bash run_full_pipeline.sh
```

### 方法2: 分步运行

#### 步骤1: 训练SAE
```bash
$HOME/.conda/envs/SA/bin/python train_sae_relation.py \
  -m "Qwen/Qwen2.5-7B-Instruct" \
  -tr "../data_generation/task_family_1/spatial_procedure_dataset_EN_with_prompt.json" \
  -l 15 \
  -e 8 \
  -b 4 \
  -n 100000 \
  --l1_coefficient 0.001 \
  --output_dir "./sae_checkpoints" \
  --device "cuda:0"
```

**关键参数**:
- `-l`: 层编号（从probe实验选择R²最高的层）
- `-e`: 扩展因子（4-16，控制SAE特征数量）
- `-n`: 训练token数（越多越好，但更慢）
- `--l1_coefficient`: L1稀疏性系数（关键！）

#### 步骤2: 分析SAE特征
```bash
$HOME/.conda/envs/SA/bin/python analyze_sae_features.py \
  -m "Qwen/Qwen2.5-7B-Instruct" \
  -s "./sae_checkpoints/sae_layer15_exp8_XXXXXX/sae_final.pt" \
  -tr "../data_generation/task_family_1/spatial_procedure_dataset_EN_with_prompt.json" \
  -te "../data_generation/task_family_1/spatial_procedure_dataset_EN_test_with_prompt.json" \
  -o "./sae_analysis" \
  --device "cuda:0"
```

#### 步骤3: 特征干预实验
```bash
$HOME/.conda/envs/SA/bin/python intervene_sae_features.py \
  -m "Qwen/Qwen2.5-7B-Instruct" \
  -s "./sae_checkpoints/sae_layer15_exp8_XXXXXX/sae_final.pt" \
  -te "../data_generation/task_family_1/spatial_procedure_dataset_EN_test_with_prompt.json" \
  -a "./sae_analysis/sae_analysis_layer15.json" \
  -o "./sae_intervention" \
  --device "cuda:0"
```

#### 步骤4: 可视化
```bash
$HOME/.conda/envs/SA/bin/python visualize_sae_results.py \
  -a "./sae_analysis/sae_analysis_layer15.json" \
  -i "./sae_intervention/intervention_results_layer15.json" \
  -o "./sae_visualizations"
```

## ⚠️ 当前实验结果分析

根据你的干预实验输出，发现以下问题：

### 问题1: 特征激活为0
```
Feature 21920 | Change: 67.06 | Orig Activation: 0.0000
```
**原因**: 这些top特征在测试数据上的原始激活都是0，说明：
- SAE稀疏性太高（L1系数太大）
- 可能存在"特征死亡"问题

### 问题2: 特征重复
X/Y/Z轴的top特征高度重复（Feature 21920, 9549, 9550），说明：
- 特征未能有效分离不同维度的信息
- 可能需要更多特征或更好的训练

## 🔧 改进建议

### 方案1: 降低L1系数（推荐首先尝试）

```bash
$HOME/.conda/envs/SA/bin/python train_sae_relation.py \
  -m "Qwen/Qwen2.5-7B-Instruct" \
  -tr "../data_generation/task_family_1/spatial_procedure_dataset_EN_with_prompt.json" \
  -l 15 \
  -e 8 \
  -b 4 \
  -n 100000 \
  --l1_coefficient 0.0001 \  # 从0.001降到0.0001
  --output_dir "./sae_checkpoints" \
  --device "cuda:0"
```

### 方案2: 增加特征数量

```bash
# 使用更大的expansion factor
-e 16  # 从8改为16
```

### 方案3: 增加训练数据

```bash
-n 500000  # 从100000增加到500000
```

### 方案4: 尝试不同的层

根据probe实验结果，尝试R²较高的其他层：
```bash
-l 10  # 或其他层
```

## 📊 期望的良好结果

### 训练阶段
```
Reconstruction Error (MSE): 0.01-0.1  # 不要太高
Sparsity: 0.01-0.1                    # 1-10%的特征激活
L0: 50-500 / 28672                    # 平均激活特征数
```

### 分析阶段
```
Ridge R²: > 0.5                       # 至少要有0.5
Non-zero features: 100-1000           # Lasso选出的特征数
```

### 干预阶段
```
Feature Activation: > 0.001           # 特征应该有实际激活
Change > Random Baseline             # 干预效果应该显著
```

## 📈 推荐的实验流程

### 第一轮：快速验证
```bash
# 小规模快速测试
-n 10000 --l1_coefficient 0.0001
```
检查L0和重构误差，调整参数。

### 第二轮：中等规模
```bash
# 中等规模训练
-n 100000 --l1_coefficient 0.0001
```
进行完整分析，查看特征质量。

### 第三轮：完整训练
```bash
# 使用最佳参数进行完整训练
-n 500000 --l1_coefficient [最佳值]
```

## 🐛 常见问题排查

### Q1: 所有特征激活都是0
**解决**: 降低L1系数（从0.001 → 0.0001 → 0.00001）

### Q2: 重构误差很大
**解决**: 降低L1系数或增加训练token数

### Q3: L0太大（接近d_sae）
**解决**: 增大L1系数

### Q4: 训练很慢
**解决**: 
- 减少训练token数
- 减小batch size
- 使用更少的特征

## 📝 检查实验输出

### 训练完成后检查：
```bash
# 查看训练历史
cat sae_checkpoints/sae_layer15_*/training_history.json | grep -A 5 "l0"

# 查看评估指标
cat sae_checkpoints/sae_layer15_*/eval_results.json
```

### 分析完成后检查：
```bash
# 查看分析结果
cat sae_analysis/sae_analysis_layer15.json | grep "r2"

# 查看top特征
cat sae_analysis/sae_analysis_layer15.json | grep -A 10 "top_features"
```

## 💡 下一步建议

基于当前结果，建议：

1. **重新训练SAE**，使用更小的L1系数：
   ```bash
   --l1_coefficient 0.0001  # 或 0.00001
   ```

2. **检查probe实验结果**，确认layer 15是否最佳：
   ```bash
   cat ../probe/task_family_1/probe_results_*.json | grep "best_layer"
   ```

3. **对比不同配置**：
   - L1=0.001, 0.0001, 0.00001
   - expansion_factor=4, 8, 16
   - 选择最佳组合

4. **参考已有实验**：
   ```bash
   ls -la ../sae_new/task_family_1/sae_results*/
   ```
   查看之前实验的参数设置。

---

**祝实验顺利！如有问题请查看详细README.md或联系项目维护者。**




