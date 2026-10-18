"""
可视化SAE分析结果
生成综合报告和可视化图表
"""
import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# =========================
# 参数解析
# =========================
parser = argparse.ArgumentParser()
parser.add_argument("--analysis_results", "-a", type=str, required=True, help="SAE analysis results JSON")
parser.add_argument("--intervention_results", "-i", type=str, help="Intervention results JSON (optional)")
parser.add_argument("--output_dir", "-o", type=str, default="./sae_visualizations")
args = parser.parse_args()

# =========================
# 配置
# =========================
ANALYSIS_FILE = args.analysis_results
INTERVENTION_FILE = args.intervention_results
OUTPUT_DIR = args.output_dir

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*50)
print("SAE Results Visualization")
print("="*50)

# =========================
# 加载数据
# =========================
print("\nLoading analysis results...")
with open(ANALYSIS_FILE, 'r') as f:
    analysis = json.load(f)

intervention = None
if INTERVENTION_FILE and os.path.exists(INTERVENTION_FILE):
    print(f"Loading intervention results...")
    with open(INTERVENTION_FILE, 'r') as f:
        intervention = json.load(f)

# =========================
# 提取数据
# =========================
layer = analysis['layer']
d_sae = analysis['d_sae']

# 获取维度信息
num_target_dims = analysis.get('target_dimensions', 3)
dimension_names = analysis.get('dimension_names', ['x', 'y', 'z'])

# Ridge回归结果
ridge_r2 = analysis['ridge_regression']['r2']
ridge_mae = analysis['ridge_regression']['mae']
ridge_r2_per_dim = analysis['ridge_regression']['r2_per_dim']

# Lasso回归结果
lasso_r2 = analysis['lasso_regression']['r2']
lasso_mae = analysis['lasso_regression']['mae']
non_zero_features = analysis['lasso_regression']['non_zero_features']

# Top features
top_features = analysis['top_features']
top_feat_indices = top_features['indices'][:50]
top_feat_importance = top_features['importance'][:50]
top_feat_freq = top_features['activation_freq'][:50]

# Per-dimension features
feat_per_dim = analysis['top_features_per_dim']

# =========================
# 可视化1: 总体性能对比
# =========================
print("\nGenerating performance comparison...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# R² 对比
axes[0].bar(['Ridge', 'Lasso'], [ridge_r2, lasso_r2], color=['steelblue', 'coral'])
axes[0].set_ylabel('R² Score')
axes[0].set_title('Model Performance (R²)')
axes[0].set_ylim([0, 1])
for i, (name, val) in enumerate([('Ridge', ridge_r2), ('Lasso', lasso_r2)]):
    axes[0].text(i, val + 0.02, f'{val:.4f}', ha='center', va='bottom')

# MAE 对比
axes[1].bar(['Ridge', 'Lasso'], [ridge_mae, lasso_mae], color=['steelblue', 'coral'])
axes[1].set_ylabel('MAE')
axes[1].set_title('Model Performance (MAE)')
for i, (name, val) in enumerate([('Ridge', ridge_mae), ('Lasso', lasso_mae)]):
    axes[1].text(i, val + 0.01, f'{val:.4f}', ha='center', va='bottom')

# Per-dimension R²
x_pos = np.arange(len(dimension_names))
# 定义颜色
dim_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8'][:len(dimension_names)]
axes[2].bar(x_pos, ridge_r2_per_dim, color=dim_colors)
axes[2].set_ylabel('R² Score')
axes[2].set_title('Per-Dimension Performance (Ridge)')
axes[2].set_xticks(x_pos)
axes[2].set_xticklabels(dimension_names, rotation=15 if len(max(dimension_names, key=len)) > 3 else 0)
axes[2].set_ylim([0, 1])
for i, val in enumerate(ridge_r2_per_dim):
    axes[2].text(i, val + 0.02, f'{val:.4f}', ha='center', va='bottom', fontsize=9)

plt.suptitle(f'SAE Analysis Performance - Layer {layer}', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'performance_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()

# =========================
# 可视化2: 特征稀疏性
# =========================
print("\nGenerating sparsity analysis...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Lasso特征选择
axes[0].bar(['Selected', 'Not Selected'], 
           [non_zero_features, d_sae - non_zero_features],
           color=['steelblue', 'lightgray'])
axes[0].set_ylabel('Number of Features')
axes[0].set_title(f'Lasso Feature Selection\n({non_zero_features}/{d_sae} features selected)')
axes[0].text(0, non_zero_features/2, str(non_zero_features), 
            ha='center', va='center', fontsize=12, fontweight='bold', color='white')
axes[0].text(1, (d_sae - non_zero_features)/2, str(d_sae - non_zero_features), 
            ha='center', va='center', fontsize=12, fontweight='bold')

# Top features activation frequency
axes[1].bar(range(len(top_feat_freq)), top_feat_freq, color='coral')
axes[1].set_xlabel('Top 50 Features (ranked by importance)')
axes[1].set_ylabel('Activation Frequency')
axes[1].set_title('Activation Frequency of Top Features')
axes[1].axhline(y=np.mean(top_feat_freq), color='red', linestyle='--', 
               label=f'Mean: {np.mean(top_feat_freq):.3f}')
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'sparsity_analysis.png'), dpi=300, bbox_inches='tight')
plt.close()

# =========================
# 可视化3: Top Features详细信息
# =========================
print("\nGenerating top features details...")

# 根据维度数量调整布局
# 计算需要的行数：1行用于整体特征，剩余行用于各维度特征（每行2个）
num_dim_rows = (num_target_dims + 1) // 2  # 向上取整
total_rows = 1 + num_dim_rows

fig = plt.figure(figsize=(15, 5 * total_rows))
gs = fig.add_gridspec(total_rows, 2, hspace=0.3, wspace=0.3)

# Top 20 overall features
ax1 = fig.add_subplot(gs[0, :])
colors = plt.cm.viridis(np.linspace(0, 1, 20))
bars = ax1.barh(range(20), top_feat_importance[:20][::-1], color=colors)
ax1.set_yticks(range(20))
ax1.set_yticklabels([f'F{idx}' for idx in top_feat_indices[:20][::-1]])
ax1.set_xlabel('Feature Importance')
ax1.set_title('Top 20 Most Important Features (Overall)', fontsize=12, fontweight='bold')
ax1.grid(axis='x', alpha=0.3)

# Top 10 features per dimension
for idx, dim_name in enumerate(dimension_names):
    # 计算子图位置：从第二行开始，每行放2个
    row = (idx // 2) + 1
    col = idx % 2
    
    # 确保不超出网格范围
    if row >= total_rows:
        break
    
    ax = fig.add_subplot(gs[row, col])
    
    feat_data = feat_per_dim.get(dim_name, {})
    feat_indices = feat_data.get('indices', [])[:10]
    feat_coefs = feat_data.get('coefficients', [])[:10]
    
    color = dim_colors[idx % len(dim_colors)]
    
    if feat_coefs:
        bars = ax.barh(range(len(feat_coefs)), feat_coefs[::-1], color=color)
        ax.set_yticks(range(len(feat_coefs)))
        ax.set_yticklabels([f'F{idx}' for idx in feat_indices[::-1]])
        ax.set_xlabel('Coefficient')
        ax.set_title(f'Top {len(feat_coefs)} Features - {dim_name}', fontsize=11, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No data available', 
                ha='center', va='center', transform=ax.transAxes)

# 如果有空的子图位置，隐藏它们
if num_target_dims % 2 == 1:  # 如果维度数是奇数，最后一行会有一个空位置
    last_row = num_dim_rows
    last_col = 1
    if last_row < total_rows:
        ax_empty = fig.add_subplot(gs[last_row, last_col])
        ax_empty.axis('off')

plt.suptitle(f'Feature Importance Analysis - Layer {layer}', fontsize=14, fontweight='bold', y=0.995)
plt.savefig(os.path.join(OUTPUT_DIR, 'top_features_details.png'), dpi=300, bbox_inches='tight')
plt.close()

# =========================
# 可视化4: Feature importance vs activation frequency散点图
# =========================
print("\nGenerating importance vs frequency scatter...")

plt.figure(figsize=(10, 8))

# 绘制所有特征（如果数量不太多）
if len(top_feat_indices) <= 100:
    plt.scatter(top_feat_freq, top_feat_importance, 
               c=range(len(top_feat_indices)), cmap='viridis',
               s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
    
    # 标注前5个特征
    for i in range(min(5, len(top_feat_indices))):
        plt.annotate(f'F{top_feat_indices[i]}', 
                    (top_feat_freq[i], top_feat_importance[i]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
else:
    plt.scatter(top_feat_freq[:50], top_feat_importance[:50], 
               c=range(50), cmap='viridis',
               s=100, alpha=0.6, edgecolors='black', linewidth=0.5)

plt.xlabel('Activation Frequency', fontsize=12)
plt.ylabel('Feature Importance', fontsize=12)
plt.title(f'Feature Importance vs Activation Frequency\nLayer {layer}, Top {len(top_feat_indices)} Features', 
         fontsize=13, fontweight='bold')
plt.colorbar(label='Rank (by importance)')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'importance_vs_frequency.png'), dpi=300, bbox_inches='tight')
plt.close()

# =========================
# 可视化5: 干预实验结果（如果有）
# =========================
if intervention:
    print("\nGenerating intervention results...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 收集干预数据 - 动态获取所有键
    intervention_keys = [k for k in intervention.keys() if k not in ['random_baseline']]
    
    baseline = intervention.get('random_baseline', {}).get('avg_change', 0)
    
    for idx, key in enumerate(intervention_keys[:4]):  # 最多显示4个类别
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        data = intervention.get(key, {})
        
        if not data:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            continue
        
        feature_names = []
        changes = []
        
        for feat_key, feat_result in data.items():
            if isinstance(feat_result, dict) and 'feature_idx' in feat_result:
                feat_idx = feat_result['feature_idx']
                feature_names.append(f'F{feat_idx}')
                changes.append(feat_result['avg_change'])
        
        if changes:
            color = dim_colors[idx % len(dim_colors)]
            
            bars = ax.bar(range(len(changes)), changes, color=color, alpha=0.7)
            ax.axhline(y=baseline, color='red', linestyle='--', 
                      linewidth=2, label='Random Baseline')
            ax.set_xticks(range(len(changes)))
            ax.set_xticklabels(feature_names, rotation=45)
            ax.set_ylabel('Avg Change in Hidden State')
            ax.set_title(f'{key.replace("_", " ").title()} Features', fontweight='bold')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle(f'Feature Intervention Effects - Layer {layer}', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'intervention_results.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()

# =========================
# 生成文本报告
# =========================
print("\nGenerating text report...")

report_lines = [
    "="*70,
    f"SAE ANALYSIS REPORT - Layer {layer}",
    "="*70,
    "",
    "MODEL CONFIGURATION",
    "-"*70,
    f"Layer: {layer}",
    f"SAE Dimension: {d_sae}",
    f"Target Dimensions: {num_target_dims}",
    f"Dimension Names: {', '.join(dimension_names)}",
    "",
    "PERFORMANCE METRICS",
    "-"*70,
    f"Ridge Regression:",
    f"  - R²: {ridge_r2:.4f}",
    f"  - MAE: {ridge_mae:.4f}",
]

for i, dim_name in enumerate(dimension_names):
    report_lines.append(f"  - R²({dim_name}): {ridge_r2_per_dim[i]:.4f}")

report_lines.extend([
    "",
    f"Lasso Regression (Sparse):",
    f"  - R²: {lasso_r2:.4f}",
    f"  - MAE: {lasso_mae:.4f}",
    f"  - Non-zero features: {non_zero_features}/{d_sae} ({100*non_zero_features/d_sae:.2f}%)",
    "",
    "TOP 10 MOST IMPORTANT FEATURES",
    "-"*70,
])

for i in range(min(10, len(top_feat_indices))):
    feat_idx = top_feat_indices[i]
    importance = top_feat_importance[i]
    freq = top_feat_freq[i]
    report_lines.append(f"{i+1:2d}. Feature {feat_idx:4d} | Importance: {importance:.6f} | Freq: {freq:.4f}")

report_lines.extend([
    "",
    "TOP FEATURES PER DIMENSION",
    "-"*70,
])

for dim_name in dimension_names:
    feat_data = feat_per_dim.get(dim_name, {})
    indices = feat_data.get('indices', [])
    coefs = feat_data.get('coefficients', [])
    
    report_lines.append(f"\n{dim_name}:")
    for i in range(min(5, len(indices))):
        feat_idx = indices[i]
        coef = coefs[i]
        report_lines.append(f"  {i+1}. Feature {feat_idx:4d} | Coefficient: {coef:.6f}")

if intervention:
    report_lines.extend([
        "",
        "",
        "INTERVENTION EXPERIMENT RESULTS",
        "-"*70,
        f"Random Baseline (avg change): {intervention.get('random_baseline', {}).get('avg_change', 0):.4f}",
        "",
    ])
    
    intervention_keys = [k for k in intervention.keys() if k not in ['random_baseline']]
    
    for key in intervention_keys:
        data = intervention.get(key, {})
        
        if data:
            report_lines.append(f"\n{key.replace('_', ' ').title()} Features:")
            for feat_key, feat_result in list(data.items())[:5]:
                if isinstance(feat_result, dict) and 'feature_idx' in feat_result:
                    feat_idx = feat_result['feature_idx']
                    change = feat_result['avg_change']
                    orig_act = feat_result['avg_activation_orig']
                    report_lines.append(
                        f"  Feature {feat_idx:4d} | Change: {change:.4f} | Orig Activation: {orig_act:.4f}"
                    )

report_lines.extend([
    "",
    "="*70,
    "End of Report",
    "="*70,
])

report_text = "\n".join(report_lines)
report_file = os.path.join(OUTPUT_DIR, 'analysis_report.txt')
with open(report_file, 'w') as f:
    f.write(report_text)

print(f"\nReport saved to {report_file}")

# 也打印到终端
print("\n" + report_text)

print("\n" + "="*50)
print("Visualization Complete!")
print(f"All outputs saved to: {OUTPUT_DIR}")
print("="*50)

