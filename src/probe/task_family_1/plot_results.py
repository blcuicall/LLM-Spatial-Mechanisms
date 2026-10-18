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
        'English': 'probe_results_20260105_020118.json',
        'Chinese': 'probe_results_20260105_022758.json',
        'Arabic': 'probe_results_20260105_032239.json',
    },
    'Llama3-8B-Instruct': {
        'English': 'probe_results_20260105_090827.json',
        'Chinese': 'probe_results_20260105_101745.json',
        'Arabic': 'probe_results_20260105_112659.json',
    },
    'Qwen2.5-7B-Base': {
        'English': 'probe_results_20260105_142153.json',
        'Chinese': 'probe_results_20260105_102958.json',
        'Arabic': 'probe_results_20260105_110533.json',
    },
    'Llama3-8B-Base': {
        'English': 'probe_results_relation_Llama3-8B-Base_EN.json',
        'Chinese': 'probe_results_relation_Llama3-8B-Base_CN.json',
        'Arabic': 'probe_results_relation_Llama3-8B-Base_AR.json',
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

def _has_data(model, lang):
    return lang in data[model]

# 读取所有数据
data = {}
for model, langs in experiments.items():
    data[model] = {}
    for lang, filename in langs.items():
        filepath = BASE_DIR / filename
        try:
            with open(filepath, 'r') as f:
                data[model][lang] = json.load(f)
        except FileNotFoundError:
            print(f"Warning: {filepath} not found, skipping...")
            continue

N_MODELS = len(experiments)

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
    ax.set_ylim([-0.2, 0.5])

plt.tight_layout()
plt.savefig(OUT_DIR / 'layer_r2_by_model.png', dpi=300, bbox_inches='tight')
print("Saved: layer_r2_by_model.png")
plt.close()

# 图1b: 各层R²变化 - 全部合并到一张图（颜色=模型，线型=语言）
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
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
ax.set_ylim([-0.2, 0.5])

# 图例放到左角，避免遮挡
ax.legend(fontsize=8, ncol=1, loc='upper left', bbox_to_anchor=(0.02, 1.0), frameon=True)
# 设置大小
fig.set_size_inches(6, 6)
plt.tight_layout()
plt.savefig(OUT_DIR / 'layer_r2_all_in_one.png', dpi=300, bbox_inches='tight')
print("Saved: layer_r2_all_in_one.png")
plt.close()

# 图2: 各层R²变化 - 按语言分组
N_LANGS = len(LANGUAGES)
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
    ax.set_ylim([-0.2, 0.5])

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

# 图4: 最佳层各组件R²对比
fig, ax = plt.subplots(figsize=(14, 6))

components = ['r2_x', 'r2_y', 'r2_z']
component_labels = ['X Component', 'Y Component', 'Z Component']

x_positions = []
labels = []
current_pos = 0

for model_idx, model in enumerate(experiments.keys()):
    for lang in LANGUAGES:
        if lang not in data[model]:
            continue
        x_positions.append(current_pos)
        labels.append(f"{model.split('-')[0]}\n{lang[:3]}")
        current_pos += 1
    current_pos += 0.5  # 添加模型间的间隔

x_positions = np.array(x_positions)
width = 0.25

for comp_idx, (comp, comp_label) in enumerate(zip(components, component_labels)):
    values = []
    for model in experiments.keys():
        for lang in LANGUAGES:
            if lang not in data[model]:
                values.append(0)
            else:
                values.append(data[model][lang]['best_layer_metrics'][comp])

    offset = (comp_idx - 1) * width
    ax.bar(x_positions + offset, values, width, label=comp_label, alpha=0.8)

ax.set_xlabel('Model and Language', fontsize=12)
ax.set_ylabel('R² Score', fontsize=12)
ax.set_title('Component-wise R² Scores at Best Layer', fontsize=14, fontweight='bold')
ax.set_xticks(x_positions)
ax.set_xticklabels(labels, fontsize=8, rotation=0)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(OUT_DIR / 'component_r2_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: component_r2_comparison.png")
plt.close()

# 图5: 热力图 - 所有实验的性能矩阵
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
    ax.set_xticks(np.arange(len(LANGUAGES)))
    ax.set_yticks(np.arange(N_MODELS))
    ax.set_xticklabels(LANGUAGES)
    ax.set_yticklabels([m.split('-')[0] for m in experiments.keys()])

    # 添加数值标签
    for i in range(N_MODELS):
        for j in range(len(LANGUAGES)):
            text = ax.text(j, i, f'{matrix_data[i, j]:.3f}',
                          ha="center", va="center", color="black", fontsize=10)

    ax.set_title(f'Best Layer {label}', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.savefig(OUT_DIR / 'performance_heatmap.png', dpi=300, bbox_inches='tight')
print("Saved: performance_heatmap.png")
plt.close()

# 图6: 最佳层位置对比
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

# 图7: 综合性能对比（雷达图）
from math import pi

fig, axes = plt.subplots(1, N_LANGS, figsize=(6 * N_LANGS, 6), subplot_kw=dict(projection='polar'))
if N_LANGS == 1:
    axes = [axes]

categories = ['R²', 'MAE\n(inv)', 'RMSE\n(inv)', 'R²-X', 'R²-Y', 'R²-Z']
N = len(categories)

for lang_idx, lang in enumerate(LANGUAGES):
    ax = axes[lang_idx]

    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    for model in experiments.keys():
        if lang not in data[model]:
            continue
        metrics = data[model][lang]['best_layer_metrics']
        # 归一化数据（MAE和RMSE取倒数，因为越小越好）
        values = [
            metrics['r2'],
            1 - metrics['mae'],  # 反转，使得越大越好
            1 - metrics['rmse'],  # 反转
            metrics['r2_x'],
            metrics['r2_y'],
            metrics['r2_z']
        ]
        values += values[:1]

        ax.plot(angles, values, 'o-', linewidth=2, label=model.split('-')[0], color=colors[model])
        ax.fill(angles, values, alpha=0.15, color=colors[model])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 0.5)
    ax.set_title(lang, fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
    ax.grid(True)

plt.tight_layout()
plt.savefig(OUT_DIR / 'radar_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: radar_comparison.png")
plt.close()

# 生成统计摘要
print("\n" + "="*60)
print("Performance Summary")
print("="*60)

for model in experiments.keys():
    print(f"\n{model}:")
    for lang in LANGUAGES:
        if lang not in data[model]:
            print(f"  {lang:10s}: No data")
            continue
        metrics = data[model][lang]['best_layer_metrics']
        best_layer = data[model][lang]['best_layer']
        print(f"  {lang:10s}: Layer {best_layer:2d} | R²={metrics['r2']:.4f} | MAE={metrics['mae']:.4f} | RMSE={metrics['rmse']:.4f}")

print("\n" + "="*60)
print("All plots have been generated successfully!")
print("="*60)
