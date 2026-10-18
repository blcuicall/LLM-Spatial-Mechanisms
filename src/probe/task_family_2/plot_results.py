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

# 定义实验配置 - 基于 Readme.md 中的信息
experiments = {
    'Qwen2.5-7B-Instruct': {
        'English': 'probe_results_orientation_20260104_172839.json',
        'Chinese': 'probe_results_orientation_20260104_225806.json',
        'Arabic': 'probe_results_orientation_20260104_225809.json',
    },
    'Llama3-8B-Instruct': {
        'English': 'probe_results_orientation_20260105_153541.json',
        'Chinese': 'probe_results_orientation_20260105_164446.json',
        'Arabic': 'probe_results_orientation_20260105_175340.json',
    },
    'Qwen2.5-7B-Base': {
        'English': 'probe_results_orientation_20260106_030223.json',
        'Chinese': 'probe_results_orientation_20260106_035547.json',
        'Arabic': 'probe_results_orientation_20260106_044856.json',
    },
    'Llama3-8B-Base': {
        'English': 'probe_results_orientation_Llama3-8B-Base_EN.json',
        'Chinese': 'probe_results_orientation_Llama3-8B-Base_CN.json',
        'Arabic': 'probe_results_orientation_Llama3-8B-Base_AR.json',
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
                data[model][lang] = json.load(f)
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
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)

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

# 图例放到左下角，避免遮挡
ax.legend(fontsize=8, ncol=1, loc='lower left', bbox_to_anchor=(0.02, 0.02), frameon=True)
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
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)

plt.tight_layout()
plt.savefig(OUT_DIR / 'layer_r2_by_language.png', dpi=300, bbox_inches='tight')
print("Saved: layer_r2_by_language.png")
plt.close()

# 图3: 最佳层R²对比
fig, ax = plt.subplots(figsize=(12, 6))

models_list = []
english_values = []
chinese_values = []
arabic_values = []

for model in experiments.keys():
    models_list.append(model.replace('-', '\n'))
    # 获取最佳层的R²值
    if 'English' in data[model]:
        best_layer = data[model]['English']['best_layer']
        english_values.append(data[model]['English']['layer_r2'][best_layer])
    else:
        english_values.append(0)

    if 'Chinese' in data[model]:
        best_layer = data[model]['Chinese']['best_layer']
        chinese_values.append(data[model]['Chinese']['layer_r2'][best_layer])
    else:
        chinese_values.append(0)

    if 'Arabic' in data[model]:
        best_layer = data[model]['Arabic']['best_layer']
        arabic_values.append(data[model]['Arabic']['layer_r2'][best_layer])
    else:
        arabic_values.append(0)

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
               f'{height:.4f}', ha='center', va='bottom', fontsize=9)

ax.set_xlabel('Model', fontsize=12)
ax.set_ylabel('R² Score', fontsize=12)
ax.set_title('Best Layer R² Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models_list, fontsize=10)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')
ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)

plt.tight_layout()
plt.savefig(OUT_DIR / 'best_layer_r2.png', dpi=300, bbox_inches='tight')
print("Saved: best_layer_r2.png")
plt.close()

# 图4: 最佳层位置对比
fig, ax = plt.subplots(figsize=(12, 6))

models_list = []
english_layers = []
chinese_layers = []
arabic_layers = []

for model in experiments.keys():
    models_list.append(model.replace('-', '\n'))
    if 'English' in data[model]:
        english_layers.append(data[model]['English']['best_layer'])
    else:
        english_layers.append(0)

    if 'Chinese' in data[model]:
        chinese_layers.append(data[model]['Chinese']['best_layer'])
    else:
        chinese_layers.append(0)

    if 'Arabic' in data[model]:
        arabic_layers.append(data[model]['Arabic']['best_layer'])
    else:
        arabic_layers.append(0)

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

# 图5: 热力图 - R²性能矩阵
fig, ax = plt.subplots(figsize=(8, 6))

# 创建数据矩阵
matrix_data = []
for model in experiments.keys():
    row = []
    for lang in LANGUAGES:
        if lang in data[model]:
            best_layer = data[model][lang]['best_layer']
            row.append(data[model][lang]['layer_r2'][best_layer])
        else:
            row.append(0)
    matrix_data.append(row)

matrix_data = np.array(matrix_data)

# 绘制热力图
im = ax.imshow(matrix_data, cmap='RdYlGn', aspect='auto', vmin=-0.1, vmax=0.1)

# 设置刻度
ax.set_xticks(np.arange(N_LANGS))
ax.set_yticks(np.arange(N_MODELS))
ax.set_xticklabels(LANGUAGES)
ax.set_yticklabels([m.split('-')[0] for m in experiments.keys()])

# 添加数值标签
for i in range(N_MODELS):
    for j in range(N_LANGS):
        text = ax.text(j, i, f'{matrix_data[i, j]:.4f}',
                      ha="center", va="center", color="black", fontsize=11)

ax.set_title('Best Layer R² Performance Heatmap', fontsize=14, fontweight='bold')
plt.colorbar(im, ax=ax, label='R² Score')

plt.tight_layout()
plt.savefig(OUT_DIR / 'performance_heatmap.png', dpi=300, bbox_inches='tight')
print("Saved: performance_heatmap.png")
plt.close()

# 图6: R²曲线汇总图（所有实验在一张图上）
fig, ax = plt.subplots(figsize=(14, 8))

for model in experiments.keys():
    for lang in LANGUAGES:
        if lang not in data[model]:
            continue
        layer_r2 = data[model][lang]['layer_r2']
        layers = list(range(len(layer_r2)))

        # 使用模型颜色和语言线型
        label = f"{model.split('-')[0]} - {lang[:3]}"
        ax.plot(layers, layer_r2, label=label,
                color=colors[model], linestyle=line_styles[lang],
                linewidth=1.8, alpha=0.8)

ax.set_xlabel('Layer Index', fontsize=13)
ax.set_ylabel('R² Score', fontsize=13)
ax.set_title('R² Across All Layers - All Experiments', fontsize=15, fontweight='bold')
ax.legend(fontsize=9, ncol=3, loc='best')
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label='Zero line')

plt.tight_layout()
plt.savefig(OUT_DIR / 'all_experiments_r2.png', dpi=300, bbox_inches='tight')
print("Saved: all_experiments_r2.png")
plt.close()

# 图7: 各模型最大R²对比
fig, ax = plt.subplots(figsize=(12, 6))

for model in experiments.keys():
    lang_max_r2 = []
    lang_labels = []

    for lang in LANGUAGES:
        if lang in data[model]:
            best_layer = data[model][lang]['best_layer']
            max_r2 = data[model][lang]['layer_r2'][best_layer]
            lang_max_r2.append(max_r2)
            lang_labels.append(lang)

    if not lang_max_r2:
        continue
    x_pos = np.arange(len(lang_labels))
    ax.plot(x_pos, lang_max_r2, 'o-', label=model, color=colors[model],
            linewidth=2.5, markersize=10)

ax.set_xlabel('Language', fontsize=12)
ax.set_ylabel('Maximum R² Score', fontsize=12)
ax.set_title('Maximum R² Score by Language and Model', fontsize=14, fontweight='bold')
ax.set_xticks(range(N_LANGS))
ax.set_xticklabels(LANGUAGES)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)

plt.tight_layout()
plt.savefig(OUT_DIR / 'max_r2_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: max_r2_comparison.png")
plt.close()

# 生成统计摘要
print("\n" + "="*70)
print("Performance Summary - Orientation Task (Task Family 2)")
print("="*70)

for model in experiments.keys():
    print(f"\n{model}:")
    for lang in LANGUAGES:
        if lang not in data[model]:
            print(f"  {lang:10s}: No data available")
            continue

        best_layer = data[model][lang]['best_layer']
        best_r2 = data[model][lang]['layer_r2'][best_layer]
        n_layers = data[model][lang]['n_layers']

        print(f"  {lang:10s}: Best Layer={best_layer:2d}/{n_layers} | R²={best_r2:.6f}")

print("\n" + "="*70)
print("All plots have been generated successfully!")
print(f"Output directory: {OUT_DIR}")
print("="*70)
