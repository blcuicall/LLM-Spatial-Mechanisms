"""
对比线性模型(Ridge/Lasso)和非线性模型(Neural Network)的性能
生成详细的对比报告和可视化
"""
import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_results(linear_file, nonlinear_file):
    """加载分析结果"""
    with open(linear_file, 'r') as f:
        linear_results = json.load(f)
    
    with open(nonlinear_file, 'r') as f:
        nonlinear_results = json.load(f)
    
    return linear_results, nonlinear_results


def generate_comparison_report(linear_results, nonlinear_results, output_dir):
    """生成对比报告"""
    
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("LINEAR vs NONLINEAR MODEL COMPARISON REPORT")
    report_lines.append("="*80)
    report_lines.append("")
    
    # 模型配置
    report_lines.append("MODEL CONFIGURATIONS")
    report_lines.append("-"*80)
    report_lines.append(f"Language Model: {linear_results['model_name']}")
    report_lines.append(f"Layer: {linear_results['layer']}")
    report_lines.append(f"SAE Dimension: {linear_results['d_sae']:,}")
    report_lines.append("")
    
    # 线性模型
    report_lines.append("Linear Models:")
    report_lines.append("  - Ridge Regression (alpha=1.0)")
    report_lines.append("  - Lasso Regression (alpha=0.01)")
    report_lines.append("")
    
    # 非线性模型
    report_lines.append("Nonlinear Model:")
    report_lines.append(f"  - Type: {nonlinear_results['predictor_type'].upper()}")
    if nonlinear_results['predictor_type'] == 'mlp':
        report_lines.append(f"  - Architecture: {nonlinear_results['predictor_params']['hidden_dims']}")
    report_lines.append(f"  - Total Parameters: {nonlinear_results['predictor_params']['total_params']:,}")
    report_lines.append(f"  - Training Epochs: {nonlinear_results['training']['epochs_trained']}")
    report_lines.append("")
    
    # 性能对比
    report_lines.append("="*80)
    report_lines.append("PERFORMANCE COMPARISON")
    report_lines.append("="*80)
    report_lines.append("")
    
    # 总体R²
    ridge_r2 = linear_results['ridge_regression']['r2']
    lasso_r2 = linear_results['lasso_regression']['r2']
    nn_r2 = nonlinear_results['performance']['r2']
    
    report_lines.append("Overall R² Score:")
    report_lines.append("-"*80)
    report_lines.append(f"  Ridge Regression:     {ridge_r2:.4f}")
    report_lines.append(f"  Lasso Regression:     {lasso_r2:.4f}")
    report_lines.append(f"  Neural Network:       {nn_r2:.4f}")
    report_lines.append("")
    
    # 改进幅度
    ridge_improvement = ((nn_r2 - ridge_r2) / ridge_r2 * 100) if ridge_r2 > 0 else 0
    lasso_improvement = ((nn_r2 - lasso_r2) / lasso_r2 * 100) if lasso_r2 > 0 else 0
    
    report_lines.append("Improvement over Linear Models:")
    report_lines.append(f"  vs Ridge: {ridge_improvement:+.1f}%")
    report_lines.append(f"  vs Lasso: {lasso_improvement:+.1f}%")
    report_lines.append("")
    
    # MAE对比
    ridge_mae = linear_results['ridge_regression']['mae']
    lasso_mae = linear_results['lasso_regression']['mae']
    nn_mae = nonlinear_results['performance']['mae']
    
    report_lines.append("Mean Absolute Error (MAE):")
    report_lines.append("-"*80)
    report_lines.append(f"  Ridge Regression:     {ridge_mae:.4f}")
    report_lines.append(f"  Lasso Regression:     {lasso_mae:.4f}")
    report_lines.append(f"  Neural Network:       {nn_mae:.4f}")
    report_lines.append("")
    
    # 各维度R²
    report_lines.append("R² by Dimension:")
    report_lines.append("-"*80)
    ridge_r2_dims = linear_results['ridge_regression']['r2_per_dim']
    nn_r2_dims = nonlinear_results['performance']['r2_per_dim']
    
    dims = ['X', 'Y', 'Z']
    for i, dim in enumerate(dims):
        report_lines.append(f"  {dim}-axis:")
        report_lines.append(f"    Ridge:  {ridge_r2_dims[i]:.4f}")
        report_lines.append(f"    NN:     {nn_r2_dims[i]:.4f}")
        improvement = ((nn_r2_dims[i] - ridge_r2_dims[i]) / ridge_r2_dims[i] * 100) if ridge_r2_dims[i] > 0 else 0
        report_lines.append(f"    Δ:      {improvement:+.1f}%")
        report_lines.append("")
    
    # 稀疏性分析
    report_lines.append("="*80)
    report_lines.append("FEATURE SELECTION ANALYSIS")
    report_lines.append("="*80)
    report_lines.append("")
    
    non_zero = linear_results['lasso_regression']['non_zero_features']
    total_features = linear_results['d_sae']
    sparsity = (1 - non_zero / total_features) * 100
    
    report_lines.append(f"Lasso Selected Features: {non_zero} / {total_features:,} ({sparsity:.2f}% sparse)")
    report_lines.append("")
    report_lines.append("This suggests only a small subset of SAE features are critical")
    report_lines.append("for the task, indicating high redundancy in the feature space.")
    report_lines.append("")
    
    # 关键发现
    report_lines.append("="*80)
    report_lines.append("KEY FINDINGS")
    report_lines.append("="*80)
    report_lines.append("")
    
    if nn_r2 > ridge_r2 * 1.2:  # 20%以上提升
        report_lines.append("✓ SIGNIFICANT NONLINEAR EFFECTS DETECTED")
        report_lines.append(f"  Neural network achieved {ridge_improvement:.1f}% improvement over Ridge.")
        report_lines.append("  This indicates strong feature interactions and nonlinear patterns.")
    elif nn_r2 > ridge_r2 * 1.05:  # 5-20%提升
        report_lines.append("✓ MODERATE NONLINEAR EFFECTS DETECTED")
        report_lines.append(f"  Neural network achieved {ridge_improvement:.1f}% improvement over Ridge.")
        report_lines.append("  Some feature interactions exist but may not be dominant.")
    else:
        report_lines.append("⚠ LIMITED NONLINEAR EFFECTS")
        report_lines.append(f"  Neural network only achieved {ridge_improvement:.1f}% improvement.")
        report_lines.append("  The task may be largely linear, or more complex architectures needed.")
    
    report_lines.append("")
    
    if nn_r2 < 0.3:
        report_lines.append("⚠ LOW OVERALL PREDICTIVE POWER")
        report_lines.append(f"  Even the best model (R²={nn_r2:.3f}) explains <30% of variance.")
        report_lines.append("  Possible reasons:")
        report_lines.append("    - Task complexity requires deeper semantic understanding")
        report_lines.append("    - Information may be distributed across multiple layers")
        report_lines.append("    - SAE features may not capture task-relevant information")
    
    report_lines.append("")
    report_lines.append("="*80)
    report_lines.append("END OF REPORT")
    report_lines.append("="*80)
    
    # 保存报告
    report_text = "\n".join(report_lines)
    report_file = os.path.join(output_dir, "comparison_report.txt")
    with open(report_file, 'w') as f:
        f.write(report_text)
    
    print(report_text)
    print(f"\nReport saved to: {report_file}")
    
    return report_text


def generate_comparison_plots(linear_results, nonlinear_results, output_dir):
    """生成对比可视化"""
    
    # 1. R²对比（总体和分维度）
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 总体R²
    ax = axes[0]
    models = ['Ridge', 'Lasso', 'Neural Net']
    r2_scores = [
        linear_results['ridge_regression']['r2'],
        linear_results['lasso_regression']['r2'],
        nonlinear_results['performance']['r2']
    ]
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    bars = ax.bar(models, r2_scores, color=colors, alpha=0.8, edgecolor='black')
    ax.set_ylabel('R² Score', fontsize=12)
    ax.set_title('Overall R² Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim([0, max(r2_scores) * 1.2])
    ax.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for bar, score in zip(bars, r2_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.4f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 分维度R²
    ax = axes[1]
    dims = ['X', 'Y', 'Z']
    x = np.arange(len(dims))
    width = 0.35
    
    ridge_r2_dims = linear_results['ridge_regression']['r2_per_dim']
    nn_r2_dims = nonlinear_results['performance']['r2_per_dim']
    
    bars1 = ax.bar(x - width/2, ridge_r2_dims, width, label='Ridge', 
                   color='#3498db', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, nn_r2_dims, width, label='Neural Net', 
                   color='#2ecc71', alpha=0.8, edgecolor='black')
    
    ax.set_xlabel('Dimension', fontsize=12)
    ax.set_ylabel('R² Score', fontsize=12)
    ax.set_title('R² by Dimension', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(dims)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'r2_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. MAE对比
    fig, ax = plt.subplots(figsize=(10, 6))
    models = ['Ridge', 'Lasso', 'Neural Net']
    mae_scores = [
        linear_results['ridge_regression']['mae'],
        linear_results['lasso_regression']['mae'],
        nonlinear_results['performance']['mae']
    ]
    
    bars = ax.bar(models, mae_scores, color=colors, alpha=0.8, edgecolor='black')
    ax.set_ylabel('Mean Absolute Error', fontsize=12)
    ax.set_title('MAE Comparison (Lower is Better)', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    for bar, score in zip(bars, mae_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.4f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mae_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 性能提升百分比
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ridge_r2 = linear_results['ridge_regression']['r2']
    nn_r2 = nonlinear_results['performance']['r2']
    
    improvements = {
        'Overall': ((nn_r2 - ridge_r2) / ridge_r2 * 100) if ridge_r2 > 0 else 0,
        'X-axis': ((nn_r2_dims[0] - ridge_r2_dims[0]) / ridge_r2_dims[0] * 100) if ridge_r2_dims[0] > 0 else 0,
        'Y-axis': ((nn_r2_dims[1] - ridge_r2_dims[1]) / ridge_r2_dims[1] * 100) if ridge_r2_dims[1] > 0 else 0,
        'Z-axis': ((nn_r2_dims[2] - ridge_r2_dims[2]) / ridge_r2_dims[2] * 100) if ridge_r2_dims[2] > 0 else 0,
    }
    
    categories = list(improvements.keys())
    values = list(improvements.values())
    colors_imp = ['#2ecc71' if v > 0 else '#e74c3c' for v in values]
    
    bars = ax.barh(categories, values, color=colors_imp, alpha=0.8, edgecolor='black')
    ax.set_xlabel('Improvement (%)', fontsize=12)
    ax.set_title('Neural Network Improvement over Ridge', fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax.grid(axis='x', alpha=0.3)
    
    for bar, value in zip(bars, values):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2.,
                f'{value:+.1f}%',
                ha='left' if value > 0 else 'right',
                va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'improvement_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nVisualizations saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Compare linear and nonlinear SAE analysis results")
    parser.add_argument("--linear_results", "-l", type=str, required=True,
                       help="Path to linear analysis results JSON")
    parser.add_argument("--nonlinear_results", "-n", type=str, required=True,
                       help="Path to nonlinear analysis results JSON")
    parser.add_argument("--output_dir", "-o", type=str, default="./comparison_results",
                       help="Output directory for comparison report and plots")
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*80)
    print("COMPARING LINEAR AND NONLINEAR SAE ANALYSIS")
    print("="*80)
    print(f"Linear results: {args.linear_results}")
    print(f"Nonlinear results: {args.nonlinear_results}")
    print(f"Output directory: {args.output_dir}")
    print("="*80)
    
    # 加载结果
    print("\nLoading results...")
    linear_results, nonlinear_results = load_results(args.linear_results, args.nonlinear_results)
    
    # 生成报告
    print("\nGenerating comparison report...")
    generate_comparison_report(linear_results, nonlinear_results, args.output_dir)
    
    # 生成可视化
    print("\nGenerating comparison plots...")
    generate_comparison_plots(linear_results, nonlinear_results, args.output_dir)
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE!")
    print("="*80)
    print(f"\nAll outputs saved to: {args.output_dir}")
    print("  - comparison_report.txt")
    print("  - r2_comparison.png")
    print("  - mae_comparison.png")
    print("  - improvement_comparison.png")
    print("="*80)


if __name__ == "__main__":
    main()




