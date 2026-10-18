"""
跨语言SAE分析结果比较工具
比较同一模型在不同语言数据上的SAE特征分析结果
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class CrossLanguageAnalyzer:
    """跨语言SAE分析比较器"""
    
    def __init__(self, lang_results: Dict[str, Dict], output_dir: str):
        """
        Args:
            lang_results: 字典，key为语言名称，value为该语言的分析结果
            output_dir: 输出目录
        """
        self.lang_results = lang_results
        self.languages = list(lang_results.keys())
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Initialized CrossLanguageAnalyzer with {len(self.languages)} languages:")
        for lang in self.languages:
            print(f"  - {lang}")
    
    def compare_regression_performance(self) -> pd.DataFrame:
        """比较不同语言的回归模型性能"""
        print("\n" + "="*60)
        print("1. Regression Performance Comparison")
        print("="*60)
        
        metrics = []
        for lang, results in self.lang_results.items():
            analysis = results.get('analysis', {})
            
            # Ridge回归
            ridge = analysis.get('ridge_regression', {})
            metrics.append({
                'Language': lang,
                'Model': 'Ridge',
                'R²': ridge.get('r2', 0),
                'MAE': ridge.get('mae', 0)
            })
            
            # Lasso回归
            lasso = analysis.get('lasso_regression', {})
            metrics.append({
                'Language': lang,
                'Model': 'Lasso',
                'R²': lasso.get('r2', 0),
                'MAE': lasso.get('mae', 0),
                'Non-zero Features': lasso.get('non_zero_features', 0)
            })
        
        df = pd.DataFrame(metrics)
        print("\n回归性能对比:")
        print(df.to_string(index=False))
        
        # 保存为CSV
        df.to_csv(os.path.join(self.output_dir, 'regression_comparison.csv'), index=False)
        
        # 可视化
        self._plot_regression_comparison(df)
        
        return df
    
    def _plot_regression_comparison(self, df: pd.DataFrame):
        """绘制回归性能对比图"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # R² 比较
        ridge_data = df[df['Model'] == 'Ridge']
        lasso_data = df[df['Model'] == 'Lasso']
        
        x = np.arange(len(self.languages))
        width = 0.35
        
        axes[0].bar(x - width/2, ridge_data['R²'].values, width, label='Ridge', alpha=0.8)
        axes[0].bar(x + width/2, lasso_data['R²'].values, width, label='Lasso', alpha=0.8)
        axes[0].set_xlabel('Language')
        axes[0].set_ylabel('R² Score')
        axes[0].set_title('R² Score Comparison Across Languages')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(self.languages)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # MAE 比较
        axes[1].bar(x - width/2, ridge_data['MAE'].values, width, label='Ridge', alpha=0.8)
        axes[1].bar(x + width/2, lasso_data['MAE'].values, width, label='Lasso', alpha=0.8)
        axes[1].set_xlabel('Language')
        axes[1].set_ylabel('MAE')
        axes[1].set_title('MAE Comparison Across Languages')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(self.languages)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'regression_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: regression_comparison.png")
    
    def compare_top_features(self, top_k: int = 50) -> Dict:
        """比较不同语言的top特征重叠情况"""
        print("\n" + "="*60)
        print(f"2. Top-{top_k} Features Comparison")
        print("="*60)
        
        # 获取每个语言的top特征
        lang_top_features = {}
        for lang, results in self.lang_results.items():
            analysis = results.get('analysis', {})
            top_features = analysis.get('top_features', {})
            indices = top_features.get('indices', [])[:top_k]
            importance = top_features.get('importance', [])[:top_k]
            lang_top_features[lang] = {
                'indices': set(indices),
                'indices_list': indices,
                'importance': importance
            }
        
        # 计算重叠
        comparison = {}
        for i, lang1 in enumerate(self.languages):
            for lang2 in self.languages[i+1:]:
                features1 = lang_top_features[lang1]['indices']
                features2 = lang_top_features[lang2]['indices']
                
                overlap = features1.intersection(features2)
                overlap_ratio = len(overlap) / top_k
                
                key = f"{lang1} vs {lang2}"
                comparison[key] = {
                    'overlap_count': len(overlap),
                    'overlap_ratio': overlap_ratio,
                    'overlap_features': list(overlap)
                }
                
                print(f"\n{key}:")
                print(f"  Overlap: {len(overlap)}/{top_k} ({overlap_ratio*100:.1f}%)")
                print(f"  Unique to {lang1}: {len(features1 - features2)}")
                print(f"  Unique to {lang2}: {len(features2 - features1)}")
        
        # 保存结果
        with open(os.path.join(self.output_dir, 'top_features_comparison.json'), 'w', encoding='utf-8') as f:
            json.dump({
                'top_k': top_k,
                'comparison': comparison,
                'lang_top_features': {
                    lang: {'indices': list(data['indices']), 'importance': data['importance']}
                    for lang, data in lang_top_features.items()
                }
            }, f, indent=2, ensure_ascii=False)
        
        # 可视化
        self._plot_feature_overlap(comparison, lang_top_features, top_k)
        
        return comparison
    
    def _plot_feature_overlap(self, comparison: Dict, lang_top_features: Dict, top_k: int):
        """绘制特征重叠可视化"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # 1. 重叠比例条形图
        pairs = list(comparison.keys())
        overlap_ratios = [comparison[pair]['overlap_ratio'] * 100 for pair in pairs]
        
        axes[0].barh(pairs, overlap_ratios, alpha=0.7, color='steelblue')
        axes[0].set_xlabel('Overlap Ratio (%)')
        axes[0].set_title(f'Top-{top_k} Feature Overlap Between Languages')
        axes[0].grid(True, alpha=0.3, axis='x')
        
        # 2. Venn图式展示（仅适用于2种语言）
        if len(self.languages) == 2:
            lang1, lang2 = self.languages
            features1 = lang_top_features[lang1]['indices']
            features2 = lang_top_features[lang2]['indices']
            
            only_lang1 = len(features1 - features2)
            only_lang2 = len(features2 - features1)
            overlap = len(features1.intersection(features2))
            
            categories = [f'Only {lang1}', 'Overlap', f'Only {lang2}']
            values = [only_lang1, overlap, only_lang2]
            colors = ['lightcoral', 'lightgreen', 'lightblue']
            
            axes[1].bar(categories, values, color=colors, alpha=0.7)
            axes[1].set_ylabel('Number of Features')
            axes[1].set_title(f'Feature Distribution: {lang1} vs {lang2}')
            axes[1].grid(True, alpha=0.3, axis='y')
            
            # 添加数值标签
            for i, v in enumerate(values):
                axes[1].text(i, v + 0.5, str(v), ha='center', va='bottom', fontweight='bold')
        else:
            # 多语言情况：展示特征重要性分布
            for lang, data in lang_top_features.items():
                importance = data['importance'][:20]  # 只显示前20个
                axes[1].plot(range(1, len(importance)+1), importance, marker='o', label=lang, alpha=0.7)
            
            axes[1].set_xlabel('Feature Rank')
            axes[1].set_ylabel('Importance Score')
            axes[1].set_title('Top-20 Feature Importance Across Languages')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'feature_overlap.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: feature_overlap.png")
    
    def compare_intervention_effects(self, top_n: int = 10) -> pd.DataFrame:
        """比较不同语言的干预实验效果"""
        print("\n" + "="*60)
        print(f"3. Intervention Effects Comparison (Top-{top_n} Features)")
        print("="*60)
        
        intervention_data = []
        
        for lang, results in self.lang_results.items():
            intervention = results.get('intervention', {})
            overall = intervention.get('overall', {})
            
            # 获取前N个特征的干预效果
            feature_effects = []
            for feature_key, feature_data in overall.items():
                feature_effects.append({
                    'feature_idx': feature_data.get('feature_idx'),
                    'avg_change': feature_data.get('avg_change', 0)
                })
            
            # 按avg_change排序
            feature_effects.sort(key=lambda x: x['avg_change'], reverse=True)
            
            for i, effect in enumerate(feature_effects[:top_n]):
                intervention_data.append({
                    'Language': lang,
                    'Rank': i + 1,
                    'Feature_ID': effect['feature_idx'],
                    'Avg_Change': effect['avg_change']
                })
        
        df = pd.DataFrame(intervention_data)
        
        if not df.empty:
            print("\n干预效果对比:")
            print(df.to_string(index=False))
            
            df.to_csv(os.path.join(self.output_dir, 'intervention_comparison.csv'), index=False)
            
            # 可视化
            self._plot_intervention_comparison(df)
        else:
            print("  No intervention data available.")
        
        return df
    
    def _plot_intervention_comparison(self, df: pd.DataFrame):
        """绘制干预效果对比图"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # 1. 每个语言的top特征干预效果
        for lang in self.languages:
            lang_data = df[df['Language'] == lang]
            axes[0].plot(lang_data['Rank'], lang_data['Avg_Change'], 
                        marker='o', label=lang, alpha=0.7, linewidth=2)
        
        axes[0].set_xlabel('Feature Rank')
        axes[0].set_ylabel('Average Change Magnitude')
        axes[0].set_title('Intervention Effects: Top Features Across Languages')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. 平均干预效果对比
        avg_effects = df.groupby('Language')['Avg_Change'].mean()
        
        axes[1].bar(avg_effects.index, avg_effects.values, alpha=0.7, color='steelblue')
        axes[1].set_xlabel('Language')
        axes[1].set_ylabel('Average Change Magnitude')
        axes[1].set_title('Average Intervention Effect Across Languages')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for i, v in enumerate(avg_effects.values):
            axes[1].text(i, v + 0.1, f'{v:.2f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'intervention_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: intervention_comparison.png")
    
    def analyze_feature_importance_correlation(self) -> Dict:
        """分析不同语言间特征重要性的相关性"""
        print("\n" + "="*60)
        print("4. Feature Importance Correlation Analysis")
        print("="*60)
        
        # 获取所有特征的重要性
        lang_importance = {}
        all_features = set()
        
        for lang, results in self.lang_results.items():
            analysis = results.get('analysis', {})
            top_features = analysis.get('top_features', {})
            indices = top_features.get('indices', [])
            importance = top_features.get('importance', [])
            
            importance_dict = {idx: imp for idx, imp in zip(indices, importance)}
            lang_importance[lang] = importance_dict
            all_features.update(indices)
        
        # 计算相关性（对于共同的特征）
        correlations = {}
        for i, lang1 in enumerate(self.languages):
            for lang2 in self.languages[i+1:]:
                common_features = set(lang_importance[lang1].keys()).intersection(
                    set(lang_importance[lang2].keys())
                )
                
                if len(common_features) > 1:
                    imp1 = [lang_importance[lang1][f] for f in common_features]
                    imp2 = [lang_importance[lang2][f] for f in common_features]
                    
                    correlation = np.corrcoef(imp1, imp2)[0, 1]
                    
                    key = f"{lang1} vs {lang2}"
                    correlations[key] = {
                        'correlation': float(correlation),
                        'num_common_features': len(common_features)
                    }
                    
                    print(f"\n{key}:")
                    print(f"  Correlation: {correlation:.4f}")
                    print(f"  Common features: {len(common_features)}")
        
        # 保存结果
        with open(os.path.join(self.output_dir, 'importance_correlation.json'), 'w') as f:
            json.dump(correlations, f, indent=2)
        
        return correlations
    
    def generate_summary_report(self) -> str:
        """生成综合对比报告"""
        print("\n" + "="*60)
        print("5. Generating Summary Report")
        print("="*60)
        
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("跨语言SAE分析对比报告")
        report_lines.append("Cross-Language SAE Analysis Comparison Report")
        report_lines.append("="*80)
        report_lines.append("")
        
        # 基本信息
        report_lines.append("## 1. 基本信息 / Basic Information")
        report_lines.append("")
        for lang, results in self.lang_results.items():
            analysis = results.get('analysis', {})
            report_lines.append(f"Language: {lang}")
            report_lines.append(f"  - Model: {analysis.get('model_name', 'N/A')}")
            report_lines.append(f"  - Layer: {analysis.get('layer', 'N/A')}")
            report_lines.append(f"  - SAE Dimension: {analysis.get('d_sae', 'N/A')}")
            report_lines.append("")
        
        # 回归性能
        report_lines.append("## 2. 回归性能对比 / Regression Performance")
        report_lines.append("")
        for lang, results in self.lang_results.items():
            analysis = results.get('analysis', {})
            ridge = analysis.get('ridge_regression', {})
            lasso = analysis.get('lasso_regression', {})
            
            report_lines.append(f"{lang}:")
            report_lines.append(f"  Ridge - R²: {ridge.get('r2', 0):.4f}, MAE: {ridge.get('mae', 0):.4f}")
            report_lines.append(f"  Lasso - R²: {lasso.get('r2', 0):.4f}, MAE: {lasso.get('mae', 0):.4f}")
            report_lines.append(f"  Lasso Non-zero Features: {lasso.get('non_zero_features', 0)}")
            report_lines.append("")
        
        # 关键发现
        report_lines.append("## 3. 关键发现 / Key Findings")
        report_lines.append("")
        report_lines.append("详细分析结果请查看生成的可视化图表和CSV文件。")
        report_lines.append("Please refer to the generated visualizations and CSV files for detailed analysis.")
        report_lines.append("")
        
        # 生成的文件列表
        report_lines.append("## 4. 生成的文件 / Generated Files")
        report_lines.append("")
        files = [
            "- regression_comparison.csv",
            "- regression_comparison.png",
            "- top_features_comparison.json",
            "- feature_overlap.png",
            "- intervention_comparison.csv (if available)",
            "- intervention_comparison.png (if available)",
            "- importance_correlation.json",
            "- summary_report.txt"
        ]
        report_lines.extend(files)
        report_lines.append("")
        
        report_lines.append("="*80)
        
        report_text = "\n".join(report_lines)
        
        # 保存报告
        with open(os.path.join(self.output_dir, 'summary_report.txt'), 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"\n  Saved: summary_report.txt")
        print("\n" + report_text)
        
        return report_text
    
    def run_full_comparison(self, top_k: int = 50, top_n_intervention: int = 10):
        """运行完整的对比分析"""
        print("\n" + "="*80)
        print("Starting Full Cross-Language Comparison Analysis")
        print("="*80)
        
        # 1. 回归性能对比
        self.compare_regression_performance()
        
        # 2. Top特征对比
        self.compare_top_features(top_k=top_k)
        
        # 3. 干预效果对比
        self.compare_intervention_effects(top_n=top_n_intervention)
        
        # 4. 特征重要性相关性
        self.analyze_feature_importance_correlation()
        
        # 5. 生成综合报告
        self.generate_summary_report()
        
        print("\n" + "="*80)
        print(f"Analysis Complete! Results saved to: {self.output_dir}")
        print("="*80)


def load_language_results(lang_output_dirs: Dict[str, str]) -> Dict[str, Dict]:
    """
    加载不同语言的分析结果
    
    Args:
        lang_output_dirs: 字典，key为语言名称，value为该语言的输出目录
    
    Returns:
        字典，包含每个语言的所有分析结果
    """
    results = {}
    
    for lang, output_dir in lang_output_dirs.items():
        print(f"\nLoading results for {lang}...")
        lang_results = {}
        
        # 加载分析结果
        analysis_file = None
        for pattern in ['sae_analysis/sae_analysis_layer*.json', 'sae_analysis_layer*.json']:
            import glob
            files = glob.glob(os.path.join(output_dir, pattern))
            if files:
                analysis_file = files[0]
                break
        
        if analysis_file and os.path.exists(analysis_file):
            with open(analysis_file, 'r') as f:
                lang_results['analysis'] = json.load(f)
            print(f"  Loaded analysis: {analysis_file}")
        else:
            print(f"  WARNING: No analysis file found in {output_dir}")
        
        # 加载干预结果
        intervention_file = None
        for pattern in ['sae_intervention/intervention_results_layer*.json', 
                       'intervention_results_layer*.json']:
            files = glob.glob(os.path.join(output_dir, pattern))
            if files:
                intervention_file = files[0]
                break
        
        if intervention_file and os.path.exists(intervention_file):
            with open(intervention_file, 'r') as f:
                lang_results['intervention'] = json.load(f)
            print(f"  Loaded intervention: {intervention_file}")
        else:
            print(f"  WARNING: No intervention file found in {output_dir}")
        
        # 加载梯度归因结果（可选）
        gradient_file = None
        for pattern in ['gradient_attribution/gradient_attribution_layer*.json',
                       'gradient_attribution_layer*.json']:
            files = glob.glob(os.path.join(output_dir, pattern))
            if files:
                gradient_file = files[0]
                break
        
        if gradient_file and os.path.exists(gradient_file):
            with open(gradient_file, 'r') as f:
                lang_results['gradient'] = json.load(f)
            print(f"  Loaded gradient attribution: {gradient_file}")
        
        results[lang] = lang_results
    
    return results


def main():
    parser = argparse.ArgumentParser(description='跨语言SAE分析结果对比工具')
    parser.add_argument('--lang_dirs', type=str, nargs='+', required=True,
                       help='语言输出目录列表，格式: LANG1:DIR1 LANG2:DIR2 ...')
    parser.add_argument('--output_dir', type=str, default='./cross_language_comparison',
                       help='对比结果输出目录')
    parser.add_argument('--top_k', type=int, default=50,
                       help='比较的top特征数量')
    parser.add_argument('--top_n_intervention', type=int, default=10,
                       help='比较的top干预特征数量')
    
    args = parser.parse_args()
    
    # 解析语言目录
    lang_output_dirs = {}
    for lang_dir in args.lang_dirs:
        if ':' not in lang_dir:
            print(f"ERROR: Invalid format '{lang_dir}'. Expected format: LANG:DIR")
            return
        lang, dir_path = lang_dir.split(':', 1)
        lang_output_dirs[lang] = dir_path
    
    print("="*80)
    print("Cross-Language SAE Analysis Comparison")
    print("="*80)
    print("\nLanguage directories:")
    for lang, dir_path in lang_output_dirs.items():
        print(f"  {lang}: {dir_path}")
    
    # 加载结果
    lang_results = load_language_results(lang_output_dirs)
    
    # 运行对比分析
    analyzer = CrossLanguageAnalyzer(lang_results, args.output_dir)
    analyzer.run_full_comparison(
        top_k=args.top_k,
        top_n_intervention=args.top_n_intervention
    )


if __name__ == "__main__":
    main()

