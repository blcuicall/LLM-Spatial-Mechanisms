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
feat_x = analysis['top_features_per_dim']['x']
feat_y = analysis['top_features_per_dim']['y']
feat_z = analysis['top_features_per_dim']['z']

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
x_pos = np.arange(3)
axes[2].bar(x_pos, ridge_r2_per_dim, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
axes[2].set_ylabel('R² Score')
axes[2].set_title('Per-Dimension Performance (Ridge)')
axes[2].set_xticks(x_pos)
axes[2].set_xticklabels(['X', 'Y', 'Z'])
axes[2].set_ylim([0, 1])
for i, val in enumerate(ridge_r2_per_dim):
    axes[2].text(i, val + 0.02, f'{val:.4f}', ha='center', va='bottom')

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

fig = plt.figure(figsize=(15, 10))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

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
for idx, (dim_name, feat_data, color) in enumerate([
    ('X-axis', feat_x, '#FF6B6B'),
    ('Y-axis', feat_y, '#4ECDC4'),
    ('Z-axis', feat_z, '#45B7D1')
]):
    row = (idx // 2) + 1
    col = idx % 2
    ax = fig.add_subplot(gs[row, col])
    
    feat_indices = feat_data['indices'][:10]
    feat_coefs = feat_data['coefficients'][:10]
    
    bars = ax.barh(range(10), feat_coefs[::-1], color=color)
    ax.set_yticks(range(10))
    ax.set_yticklabels([f'F{idx}' for idx in feat_indices[::-1]])
    ax.set_xlabel('Coefficient')
    ax.set_title(f'Top 10 Features - {dim_name}', fontsize=11, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

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
    
    # 收集干预数据
    intervention_data = {
        'Overall': intervention.get('overall', {}),
        'X-axis': intervention.get('x_axis', {}),
        'Y-axis': intervention.get('y_axis', {}),
        'Z-axis': intervention.get('z_axis', {}),
    }
    
    baseline = intervention.get('random_baseline', {}).get('avg_change', 0)
    
    for idx, (category, data) in enumerate(intervention_data.items()):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        if not data:
            continue
        
        feature_names = []
        changes = []
        
        for feat_key, feat_result in data.items():
            feat_idx = feat_result['feature_idx']
            feature_names.append(f'F{feat_idx}')
            changes.append(feat_result['avg_change'])
        
        if changes:
            colors_map = {
                'Overall': 'steelblue',
                'X-axis': '#FF6B6B',
                'Y-axis': '#4ECDC4',
                'Z-axis': '#45B7D1',
            }
            
            bars = ax.bar(range(len(changes)), changes, 
                         color=colors_map.get(category, 'gray'), alpha=0.7)
            ax.axhline(y=baseline, color='red', linestyle='--', 
                      linewidth=2, label='Random Baseline')
            ax.set_xticks(range(len(changes)))
            ax.set_xticklabels(feature_names, rotation=45)
            ax.set_ylabel('Avg Change in Hidden State')
            ax.set_title(f'{category} Features', fontweight='bold')
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
    "",
    "PERFORMANCE METRICS",
    "-"*70,
    f"Ridge Regression:",
    f"  - R²: {ridge_r2:.4f}",
    f"  - MAE: {ridge_mae:.4f}",
    f"  - R²(X): {ridge_r2_per_dim[0]:.4f}",
    f"  - R²(Y): {ridge_r2_per_dim[1]:.4f}",
    f"  - R²(Z): {ridge_r2_per_dim[2]:.4f}",
    "",
    f"Lasso Regression (Sparse):",
    f"  - R²: {lasso_r2:.4f}",
    f"  - MAE: {lasso_mae:.4f}",
    f"  - Non-zero features: {non_zero_features}/{d_sae} ({100*non_zero_features/d_sae:.2f}%)",
    "",
    "TOP 10 MOST IMPORTANT FEATURES",
    "-"*70,
]

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

for dim_name, feat_data in [('X-axis', feat_x), ('Y-axis', feat_y), ('Z-axis', feat_z)]:
    report_lines.append(f"\n{dim_name}:")
    for i in range(min(5, len(feat_data['indices']))):
        feat_idx = feat_data['indices'][i]
        coef = feat_data['coefficients'][i]
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
    
    for category in ['Overall', 'X-axis', 'Y-axis', 'Z-axis']:
        key = category.lower().replace('-', '_')
        data = intervention.get(key, {})
        
        if data:
            report_lines.append(f"\n{category} Features:")
            for feat_key, feat_result in list(data.items())[:5]:
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




