# 快速开始指南

## 🚀 3 分钟快速测试

### Step 1: 进入目录
```bash
cd src/supplementary
```

### Step 2: 检查前置条件
```bash
# 确认 SAE 结果存在
ls ../sae_saelens/sae_results/L8_F2048_*/sae_checkpoint.pt

# 确认 spatial features 已识别
ls ../sae_saelens/sae_results/L8_F2048_*/analysis/dimension_features.json

# 确认评估数据存在
ls ../../data_generation_task_3/spatial_procedure_dataset_EN_test_with_prompt.json
```

如果以上文件都存在，继续 ✓

### Step 3: 快速测试（20 个样本，约 3 分钟）
```bash
# 获取 SAE checkpoint 路径
SAE_DIR=$(ls -d ../sae_saelens/sae_results/L8_F2048_* | head -n 1)

# 测试 Dose-Response
python dose_response_experiment.py \
  -c $SAE_DIR \
  -e ../../data_generation_task_3/spatial_procedure_dataset_EN_test_with_prompt.json \
  --max_samples 20 \
  --k_values "5,10,20" \
  --ablation_type mean
```

如果成功运行并生成结果，说明环境配置正确！

---

## 📋 完整实验运行（约 1-2 小时）

### 方法 1: 一键运行所有实验（推荐）

```bash
# 给脚本执行权限
chmod +x run_all_experiments.sh

# 运行所有实验（200 样本）
bash run_all_experiments.sh ../sae_saelens/sae_results/L8_F2048_* 200
```

### 方法 2: 手动运行每个实验

```bash
# 设置变量
SAE_DIR="../sae_saelens/sae_results/L8_F2048_*"
DATA_FILE="../../data_generation_task_3/spatial_procedure_dataset_EN_test_with_prompt.json"
MAX_SAMPLES=200

# 1. Dose-Response
python dose_response_experiment.py -c $SAE_DIR -e $DATA_FILE --max_samples $MAX_SAMPLES

# 2. Keep-Spatial-Only
python keep_spatial_only_experiment.py -c $SAE_DIR -e $DATA_FILE --max_samples $MAX_SAMPLES

# 3. Statistical Significance
python statistical_significance_experiment.py -c $SAE_DIR -e $DATA_FILE --max_samples $MAX_SAMPLES

# 4. Ablation Robustness
python ablation_robustness_experiment.py -c $SAE_DIR -e $DATA_FILE --max_samples $MAX_SAMPLES

# 5. Matched Random Control
python matched_random_control_experiment.py -c $SAE_DIR -e $DATA_FILE --max_samples $MAX_SAMPLES
```

---

## 📊 查看结果

```bash
# 进入结果目录
cd ../sae_saelens/sae_results/L8_F2048_*/

# 查看所有生成的图表
find . -name "*.png" -type f

# 查看摘要文件
find . -name "*summary*.txt" -type f | xargs cat
```

---

## 🎯 典型用例

### 用例 1: 只想快速验证（10 分钟）
```bash
# 用 50 个样本测试核心实验
python dose_response_experiment.py -c $SAE_DIR -e $DATA_FILE --max_samples 50
python statistical_significance_experiment.py -c $SAE_DIR -e $DATA_FILE --max_samples 50
```

### 用例 2: 准备论文图表（1-2 小时）
```bash
# 用 200-500 样本运行所有实验
bash run_all_experiments.sh $SAE_DIR 500
```

### 用例 3: 测试不同 ablation 类型
```bash
# 测试 zero, mean, shuffle 三种方式
for TYPE in zero mean shuffle; do
    python dose_response_experiment.py \
      -c $SAE_DIR -e $DATA_FILE \
      --ablation_type $TYPE \
      --max_samples 100
done
```

### 用例 4: 跨层分析（需要多层 SAE）
```bash
# 如果你训练了多个层的 SAE
python layer_localization_experiment.py \
  -s ../sae_saelens/sae_results/ \
  -e $DATA_FILE \
  --max_samples 200
```

---

## ⚠️ 常见问题

### Q1: 找不到 config.py
**A**: 确保在正确的目录，或创建软链接：
```bash
ln -s ../config.py .
```

### Q2: CUDA out of memory
**A**: 减少样本数或使用 CPU：
```bash
python xxx.py ... --max_samples 50 --device cpu
```

### Q3: 没有梯度归因结果
**A**: 先运行梯度归因（可选）：
```bash
cd ../sae_saelens
python gradient_attribution.py \
  -c ./sae_results/L8_F2048_*/ \
  -e ../../data_generation_task_3/spatial_procedure_dataset_EN_test_with_prompt.json \
  --max_samples 200
```

或者不使用 `-g` 参数。

### Q4: 实验运行很慢
**A**: 这是正常的，因为：
- 每个样本都要完整前向传播
- 要运行多次（baseline + spatial + random）
- 建议先用小样本测试

加速方法：
- 使用 GPU（比 CPU 快 5-10x）
- 减少样本数
- 使用 simple 版本

---

## 📈 预期结果示例

### Dose-Response:
```
k     Accuracy    Δ Acc      Margin     NLL
0     0.8500      -          2.3450     0.4123
5     0.8200      -0.0300    2.1234     0.4567
10    0.7800      -0.0700    1.8765     0.5234
20    0.7200      -0.1300    1.4567     0.6789
```

### Statistical Significance:
```
Baseline vs Spatial (Margin):
  Paired t-test: p=0.00012
  Cohen's d: 0.78
  ✓ Significant (p<0.05)
```

### Ablation Robustness:
```
Ablation Type    Spatial Drop    Random Drop    Ratio
zero             0.2300          0.0500         4.60
mean             0.2100          0.0600         3.50
shuffle          0.1900          0.0550         3.45
```

---

## 🎉 成功标志

如果看到以下结果，说明实验成功：

✅ **Dose-Response**: 单调递减曲线
✅ **Keep-Spatial-Only**: 保留率 40-70%
✅ **Statistical Tests**: p < 0.05
✅ **Robustness**: 3 种方式一致
✅ **Matched Control**: Ratio > 1.5x

**恭喜！你有了强有力的因果证据！** 🎊

---

## 📞 需要帮助？

1. 查看完整文档：`README.md`
2. 检查脚本注释
3. 查看原始论文/代码

**祝实验顺利！** 🚀




