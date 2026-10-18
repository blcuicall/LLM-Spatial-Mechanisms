import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.3)

# 路径配置：保证从任意工作目录运行都能正确读写
BASE_DIR = Path(__file__).parent
OUT_DIR = BASE_DIR / "visualization"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 定义实验配置
experiments = {
    'Qwen2.5-7B-Instruct': {
        'English': 'probe_procedure_results_fixed_20260104_184619.json',
        'Chinese': 'probe_procedure_results_fixed_20260105_000356.json',
        'Arabic': 'probe_procedure_results_fixed_20260105_000403.json',
    },
    'Llama3-8B-Instruct': {
        'English': 'probe_procedure_results_fixed_20260105_091427.json',
        'Chinese': 'probe_procedure_results_fixed_20260105_102529.json',
        'Arabic': 'probe_procedure_results_fixed_20260105_113547.json',
    },
    'Qwen2.5-7B-Base': {
        'English': 'probe_procedure_results_fixed_20260105_154242.json',
        'Chinese': 'probe_procedure_results_fixed_20260105_163651.json',
        'Arabic': 'probe_procedure_results_fixed_20260105_173019.json',
    },
    'Llama3-8B-Base': {
        'English': 'probe_results_procedure_Llama3-8B-Base_EN.json',
        'Chinese': 'probe_results_procedure_Llama3-8B-Base_CN.json',
        'Arabic': 'probe_results_procedure_Llama3-8B-Base_AR.json',
    }
}

# 颜色方案
colors = {
    'Qwen2.5-7B-Instruct': '#2E86AB',
    'Llama3-8B-Instruct': '#A23B72',
    'Qwen2.5-7B-Base': '#F18F01',
    'Llama3-8B-Base': '#6C5B7B',
}

line_styles = {
    'English': '-',
    'Chinese': '--',
    'Arabic': ':',
}

LANGUAGES = ['English', 'Chinese', 'Arabic']
N_MODELS = len(experiments)
N_LANGS = len(LANGUAGES)

# 读取所有数据
data = {}
for model, langs in experiments.items():
    data[model] = {}
    for lang, filename in langs.items():
        filepath = BASE_DIR / filename
        try:
            with open(filepath, 'r') as f:
                raw_data = json.load(f)
                # 提取最佳层的指标
                best_layer = raw_data['best_layer']
                raw_data['best_layer_metrics'] = {
                    'r2': raw_data['layer_r2'][best_layer],
                    'mae': raw_data['layer_mae'][best_layer],
                    'rmse': raw_data['layer_rmse'][best_layer]
                }
                data[model][lang] = raw_data
        except FileNotFoundError:
            print(f"Warning: {filepath} not found, skipping...")
            continue

# 图1: 各层R²变化 - 按模型分组
fig, axes = plt.subplots(1, N_MODELS, figsize=(6 * N_MODELS, 5))
if N_MODELS == 1:
    axes = [axes]
for idx, model in enumerate(experiments.keys()):
    ax = axes[idx]
    for lang in LANGUAGES:
        if lang not in data[model]:
            continue
        layer_r2 = data[model][lang]['layer_r2']
        layers = list(range(len(layer_r2)))
        ax.plot(layers, layer_r2, label=lang, linestyle=line_styles[lang],
                linewidth=2, color=colors[model], alpha=0.7 if lang != 'English' else 1.0)

    ax.set_xlabel('Layer Index', fontsize=12)
    ax.set_ylabel('R² Score', fontsize=12)
    ax.set_title(model, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    # 自动调整y轴范围
    available_langs = [l for l in LANGUAGES if l in data[model]]
    if available_langs:
        y_min = min(min(data[model][l]['layer_r2']) for l in available_langs)
        y_max = max(max(data[model][l]['layer_r2']) for l in available_langs)
        ax.set_ylim([min(-0.05, y_min), y_max + 0.05])

plt.tight_layout()
plt.savefig(OUT_DIR / 'layer_r2_by_model.png', dpi=300, bbox_inches='tight')
print("Saved: layer_r2_by_model.png")
plt.close()

# 图1b: 各层R²变化 - 全部合并到一张图（颜色=模型，线型=语言）
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
for model in experiments.keys():
    for lang in LANGUAGES:
        if lang not in data[model]:
            continue
        layer_r2 = data[model][lang]['layer_r2']
        layers = list(range(len(layer_r2)))
        ax.plot(
            layers,
            layer_r2,
            label=f"{model} | {lang}",
            linestyle=line_styles[lang],
            linewidth=2.2 if lang == 'English' else 2.0,
            color=colors[model],
            alpha=0.85 if lang == 'English' else 0.65,
        )

ax.set_xlabel('Layer Index', fontsize=12)
ax.set_ylabel('R² Score', fontsize=12)
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)

# 图例放到左上角，避免遮挡
ax.legend(fontsize=8, ncol=1, loc='upper left', bbox_to_anchor=(0.02, 1.0), frameon=True)
# 设置大小
fig.set_size_inches(6, 6)
plt.tight_layout()
plt.savefig(OUT_DIR / 'layer_r2_all_in_one.png', dpi=300, bbox_inches='tight')
print("Saved: layer_r2_all_in_one.png")
plt.close()

# 图2: 各层R²变化 - 按语言分组
fig, axes = plt.subplots(1, N_LANGS, figsize=(6 * N_LANGS, 5))
if N_LANGS == 1:
    axes = [axes]
for idx, lang in enumerate(LANGUAGES):
    ax = axes[idx]
    for model in experiments.keys():
        if lang not in data[model]:
            continue
        layer_r2 = data[model][lang]['layer_r2']
        layers = list(range(len(layer_r2)))
        ax.plot(layers, layer_r2, label=model, color=colors[model],
                linestyle=line_styles[lang], linewidth=2.5)

    ax.set_xlabel('Layer Index', fontsize=12)
    ax.set_ylabel('R² Score', fontsize=12)
    ax.set_title(f'{lang}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    # 自动调整y轴范围
    models_with_lang = [m for m in experiments.keys() if lang in data[m]]
    if models_with_lang:
        y_min = min(min(data[m][lang]['layer_r2']) for m in models_with_lang)
        y_max = max(max(data[m][lang]['layer_r2']) for m in models_with_lang)
        ax.set_ylim([min(-0.05, y_min), y_max + 0.05])

plt.tight_layout()
plt.savefig(OUT_DIR / 'layer_r2_by_language.png', dpi=300, bbox_inches='tight')
print("Saved: layer_r2_by_language.png")
plt.close()

# 图3: 最佳层性能对比 (R², MAE, RMSE)
metrics_names = ['r2', 'mae', 'rmse']
metrics_labels = ['R² Score', 'MAE', 'RMSE']

fig, axes = plt.subplots(1, len(metrics_names), figsize=(6 * len(metrics_names), 5))
for metric_idx, (metric, label) in enumerate(zip(metrics_names, metrics_labels)):
    ax = axes[metric_idx]

    models_list = []
    english_values = []
    chinese_values = []
    arabic_values = []

    for model in experiments.keys():
        if not all(lang in data[model] for lang in LANGUAGES):
            continue
        models_list.append(model.replace('-', '\n'))
        english_values.append(data[model]['English']['best_layer_metrics'][metric])
        chinese_values.append(data[model]['Chinese']['best_layer_metrics'][metric])
        arabic_values.append(data[model]['Arabic']['best_layer_metrics'][metric])

    x = np.arange(len(models_list))
    width = 0.25

    bars1 = ax.bar(x - width, english_values, width, label='English', color='#4A90E2', alpha=0.8)
    bars2 = ax.bar(x, chinese_values, width, label='Chinese', color='#E74C3C', alpha=0.8)
    bars3 = ax.bar(x + width, arabic_values, width, label='Arabic', color='#F39C12', alpha=0.8)

    # 添加数值标签
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel(label, fontsize=12)
    ax.set_title(f'Best Layer {label}', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models_list, fontsize=9)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(OUT_DIR / 'best_layer_metrics.png', dpi=300, bbox_inches='tight')
print("Saved: best_layer_metrics.png")
plt.close()

# 图4: 各层MAE变化 - 按模型分组
fig, axes = plt.subplots(1, N_MODELS, figsize=(6 * N_MODELS, 5))
if N_MODELS == 1:
    axes = [axes]
for idx, model in enumerate(experiments.keys()):
    ax = axes[idx]
    for lang in LANGUAGES:
        if lang not in data[model]:
            continue
        layer_mae = data[model][lang]['layer_mae']
        layers = list(range(len(layer_mae)))
        ax.plot(layers, layer_mae, label=lang, linestyle=line_styles[lang],
                linewidth=2, color=colors[model], alpha=0.7 if lang != 'English' else 1.0)

    ax.set_xlabel('Layer Index', fontsize=12)
    ax.set_ylabel('MAE', fontsize=12)
    ax.set_title(model, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_DIR / 'layer_mae_by_model.png', dpi=300, bbox_inches='tight')
print("Saved: layer_mae_by_model.png")
plt.close()

# 图5: 各层RMSE变化 - 按模型分组
fig, axes = plt.subplots(1, N_MODELS, figsize=(6 * N_MODELS, 5))
if N_MODELS == 1:
    axes = [axes]
for idx, model in enumerate(experiments.keys()):
    ax = axes[idx]
    for lang in LANGUAGES:
        if lang not in data[model]:
            continue
        layer_rmse = data[model][lang]['layer_rmse']
        layers = list(range(len(layer_rmse)))
        ax.plot(layers, layer_rmse, label=lang, linestyle=line_styles[lang],
                linewidth=2, color=colors[model], alpha=0.7 if lang != 'English' else 1.0)

    ax.set_xlabel('Layer Index', fontsize=12)
    ax.set_ylabel('RMSE', fontsize=12)
    ax.set_title(model, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_DIR / 'layer_rmse_by_model.png', dpi=300, bbox_inches='tight')
print("Saved: layer_rmse_by_model.png")
plt.close()

# 图6: 热力图 - 所有实验的性能矩阵
fig, axes = plt.subplots(1, len(metrics_names), figsize=(6 * len(metrics_names), 5))

for metric_idx, (metric, label) in enumerate(zip(metrics_names, metrics_labels)):
    ax = axes[metric_idx]

    # 创建数据矩阵
    matrix_data = []
    for model in experiments.keys():
        row = []
        for lang in LANGUAGES:
            if lang in data[model]:
                row.append(data[model][lang]['best_layer_metrics'][metric])
            else:
                row.append(0)
        matrix_data.append(row)

    matrix_data = np.array(matrix_data)

    # 绘制热力图
    im = ax.imshow(matrix_data, cmap='YlOrRd' if metric != 'r2' else 'RdYlGn', aspect='auto')

    # 设置刻度
    ax.set_xticks(np.arange(N_LANGS))
    ax.set_yticks(np.arange(N_MODELS))
    ax.set_xticklabels(LANGUAGES)
    ax.set_yticklabels([m.split('-')[0] for m in experiments.keys()])

    # 添加数值标签
    for i in range(N_MODELS):
        for j in range(N_LANGS):
            text = ax.text(j, i, f'{matrix_data[i, j]:.3f}',
                          ha="center", va="center", color="black", fontsize=10)

    ax.set_title(f'Best Layer {label}', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.savefig(OUT_DIR / 'performance_heatmap.png', dpi=300, bbox_inches='tight')
print("Saved: performance_heatmap.png")
plt.close()

# 图7: 最佳层位置对比
fig, ax = plt.subplots(figsize=(12, 6))

models_list = []
english_layers = []
chinese_layers = []
arabic_layers = []

for model in experiments.keys():
    if not all(lang in data[model] for lang in LANGUAGES):
        continue
    models_list.append(model.replace('-', '\n'))
    english_layers.append(data[model]['English']['best_layer'])
    chinese_layers.append(data[model]['Chinese']['best_layer'])
    arabic_layers.append(data[model]['Arabic']['best_layer'])

x = np.arange(len(models_list))
width = 0.25

bars1 = ax.bar(x - width, english_layers, width, label='English', color='#4A90E2', alpha=0.8)
bars2 = ax.bar(x, chinese_layers, width, label='Chinese', color='#E74C3C', alpha=0.8)
bars3 = ax.bar(x + width, arabic_layers, width, label='Arabic', color='#F39C12', alpha=0.8)

# 添加数值标签
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{int(height)}', ha='center', va='bottom', fontsize=10)

ax.set_xlabel('Model', fontsize=12)
ax.set_ylabel('Best Layer Index', fontsize=12)
ax.set_title('Best Layer Position Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models_list, fontsize=10)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(OUT_DIR / 'best_layer_position.png', dpi=300, bbox_inches='tight')
print("Saved: best_layer_position.png")
plt.close()

# 图8: 综合性能对比（雷达图）
from math import pi

fig, axes = plt.subplots(1, N_LANGS, figsize=(6 * N_LANGS, 6), subplot_kw=dict(projection='polar'))
if N_LANGS == 1:
    axes = [axes]

categories = ['R²', 'MAE\n(inv)', 'RMSE\n(inv)']
N = len(categories)

# 计算归一化用的最大值（只考虑有数据的模型）
all_models_with_data = [m for m in experiments.keys() if any(l in data[m] for l in LANGUAGES)]
if all_models_with_data:
    max_r2 = max(data[m][l]['best_layer_metrics']['r2']
                 for m in all_models_with_data for l in LANGUAGES if l in data[m])
    max_mae = max(data[m][l]['best_layer_metrics']['mae']
                  for m in all_models_with_data for l in LANGUAGES if l in data[m])
    max_rmse = max(data[m][l]['best_layer_metrics']['rmse']
                   for m in all_models_with_data for l in LANGUAGES if l in data[m])
else:
    max_r2 = max_mae = max_rmse = 1.0

for lang_idx, lang in enumerate(LANGUAGES):
    ax = axes[lang_idx]

    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    for model in experiments.keys():
        if lang not in data[model]:
            continue
        metrics = data[model][lang]['best_layer_metrics']

        values = [
            metrics['r2'] / max(max_r2, 0.01),  # 归一化到 [0, 1]
            (1 - metrics['mae'] / max_mae) if max_mae > 0 else 0,  # 反转并归一化
            (1 - metrics['rmse'] / max_rmse) if max_rmse > 0 else 0,  # 反转并归一化
        ]
        values += values[:1]

        ax.plot(angles, values, 'o-', linewidth=2, label=model.split('-')[0], color=colors[model])
        ax.fill(angles, values, alpha=0.15, color=colors[model])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title(lang, fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
    ax.grid(True)

plt.tight_layout()
plt.savefig(OUT_DIR / 'radar_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: radar_comparison.png")
plt.close()

# 图9: 综合对比图 - 各层三个指标同时展示
# Only include models that have data
models_with_data = [m for m in experiments.keys() if any(l in data[m] for l in LANGUAGES)]
if models_with_data:
    fig, axes = plt.subplots(len(models_with_data), N_LANGS, figsize=(6 * N_LANGS, 5 * len(models_with_data)))
    if len(models_with_data) == 1 and N_LANGS == 1:
        axes = np.array([[axes]])
    elif len(models_with_data) == 1:
        axes = np.array([axes])
    elif N_LANGS == 1:
        axes = axes[:, np.newaxis]

    for model_idx, model in enumerate(models_with_data):
        for lang_idx, lang in enumerate(LANGUAGES):
            ax = axes[model_idx, lang_idx]

            if lang not in data[model]:
                ax.set_title(f'{model.split("-")[0]} - {lang} (no data)', fontsize=11, fontweight='bold')
                continue

            layer_r2 = data[model][lang]['layer_r2']
            layer_mae = data[model][lang]['layer_mae']
            layer_rmse = data[model][lang]['layer_rmse']
            layers = list(range(len(layer_r2)))

            # 创建双y轴
            ax2 = ax.twinx()

            # 绘制R²
            line1 = ax.plot(layers, layer_r2, label='R²', color='#2E86AB', linewidth=2.5)
            ax.set_ylabel('R² Score', fontsize=10, color='#2E86AB')
            ax.tick_params(axis='y', labelcolor='#2E86AB')

            # 绘制MAE和RMSE
            line2 = ax2.plot(layers, layer_mae, label='MAE', color='#E74C3C', linewidth=2, linestyle='--')
            line3 = ax2.plot(layers, layer_rmse, label='RMSE', color='#F39C12', linewidth=2, linestyle=':')
            ax2.set_ylabel('MAE / RMSE', fontsize=10)

            # 标记最佳层
            best_layer = data[model][lang]['best_layer']
            ax.axvline(x=best_layer, color='gray', linestyle='-.', alpha=0.5, linewidth=1.5)
            ax.text(best_layer, ax.get_ylim()[1] * 0.95, f'Best: {best_layer}',
                    ha='center', fontsize=8, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            # 设置标题和网格
            ax.set_title(f'{model.split("-")[0]} - {lang}', fontsize=11, fontweight='bold')
            ax.set_xlabel('Layer Index', fontsize=10)
            ax.grid(True, alpha=0.3)

            # 合并图例
            lines = line1 + line2 + line3
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, loc='upper left', fontsize=8)

    plt.tight_layout()
    plt.savefig(OUT_DIR / 'comprehensive_layer_metrics.png', dpi=300, bbox_inches='tight')
    print("Saved: comprehensive_layer_metrics.png")
    plt.close()

# 生成统计摘要
print("\n" + "="*80)
print("Performance Summary - Task Family 3: Spatial Procedure Execution")
print("="*80)

for model in experiments.keys():
    print(f"\n{model}:")
    for lang in LANGUAGES:
        if lang not in data[model]:
            print(f"  {lang:10s}: No data available")
            continue
        metrics = data[model][lang]['best_layer_metrics']
        best_layer = data[model][lang]['best_layer']
        n_layers = data[model][lang]['n_layers']
        print(f"  {lang:10s}: Layer {best_layer:2d}/{n_layers} | "
              f"R²={metrics['r2']:.4f} | MAE={metrics['mae']:.4f} | RMSE={metrics['rmse']:.4f}")

# 计算跨语言平均性能
print("\n" + "="*80)
print("Cross-Language Average Performance")
print("="*80)

for model in experiments.keys():
    available = [lang for lang in LANGUAGES if lang in data[model]]
    if not available:
        print(f"{model:25s}: No data available")
        continue
    r2_avg = np.mean([data[model][lang]['best_layer_metrics']['r2'] for lang in available])
    mae_avg = np.mean([data[model][lang]['best_layer_metrics']['mae'] for lang in available])
    rmse_avg = np.mean([data[model][lang]['best_layer_metrics']['rmse'] for lang in available])
    print(f"{model:25s}: R²={r2_avg:.4f} | MAE={mae_avg:.4f} | RMSE={rmse_avg:.4f}")

# 找出每个语言的最佳模型
print("\n" + "="*80)
print("Best Model per Language")
print("="*80)

for lang in LANGUAGES:
    models_with_lang = [m for m in experiments.keys() if lang in data[m]]
    if not models_with_lang:
        print(f"{lang:10s}: No data available")
        continue
    best_model = max(models_with_lang,
                     key=lambda m: data[m][lang]['best_layer_metrics']['r2'])
    best_r2 = data[best_model][lang]['best_layer_metrics']['r2']
    print(f"{lang:10s}: {best_model:25s} (R²={best_r2:.4f})")

print("\n" + "="*80)
print("All plots have been generated successfully!")
print(f"Output directory: {OUT_DIR}")
print("="*80)
