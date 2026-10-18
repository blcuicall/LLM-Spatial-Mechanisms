"""
SAE特征干预实验 - 空间过程执行任务
通过修改特定SAE特征来测试其对模型输出的因果影响
"""
import sys
sys.path.append("./")
sys.path.append("../")
sys.path.append("../../")
from config import PATHS

import os
import json
import torch
import argparse
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer
import matplotlib.pyplot as plt

# SAELens imports
try:
    from sae_lens import SAE
    from sae_lens.config import LanguageModelSAERunnerConfig, LoggingConfig
    from sae_lens.saes import TrainingSAE, StandardTrainingSAE, StandardTrainingSAEConfig
    SAELENS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: SAELens not installed or import failed: {e}")
    print("Please run: pip install sae-lens")
    SAELENS_AVAILABLE = False
    sys.exit(1)

# =========================
# 参数解析
# =========================
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", "-m", type=str, required=True)
parser.add_argument("--sae_path", "-s", type=str, required=True)
parser.add_argument("--test_data_file", "-te", type=str, required=True)
parser.add_argument("--analysis_results", "-a", type=str, required=True, help="SAE analysis results JSON")
parser.add_argument("--output_dir", "-o", type=str, default="./sae_intervention")
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--num_samples", "-n", type=int, default=100, help="Number of test samples")
parser.add_argument("--intervention_magnitude", type=float, default=2.0, help="Intervention magnitude")
args = parser.parse_args()

# =========================
# 配置
# =========================
MODEL_NAME = args.model_name
MODEL_PATH = PATHS[MODEL_NAME]
DEVICE = args.device
DTYPE = torch.float32
SAE_PATH = args.sae_path
TEST_DATA_FILE = args.test_data_file
ANALYSIS_RESULTS = args.analysis_results
OUTPUT_DIR = args.output_dir
NUM_SAMPLES = args.num_samples
INTERVENTION_MAG = args.intervention_magnitude

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*50)
print("SAE Feature Intervention (Spatial Procedure)")
print("="*50)
print(f"Model: {MODEL_NAME}")
print(f"SAE Path: {SAE_PATH}")
print(f"Intervention Magnitude: {INTERVENTION_MAG}")
print("="*50)

# =========================
# 加载模型
# =========================
print("\nLoading model...")
hf_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=DTYPE,
    trust_remote_code=True
)
hf_model = hf_model.to(DEVICE)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True
)

model = HookedTransformer.from_pretrained(
    MODEL_NAME,
    hf_model=hf_model,
    tokenizer=tokenizer,
    dtype=DTYPE,
    device=DEVICE,
    fold_ln=False,
    center_writing_weights=False,
    center_unembed=False,
    fold_value_biases=False,
)
model.eval()

# =========================
# 加载SAE
# =========================
print("\nLoading SAE...")
checkpoint = torch.load(SAE_PATH, map_location=DEVICE)
sae_config = checkpoint['config']
LAYER = sae_config['layer']

# 重建SAE
sae_cfg = StandardTrainingSAEConfig(
    d_in=sae_config['d_model'],
    d_sae=sae_config['d_sae'],
    l1_coefficient=sae_config.get('l1_coefficient', 0.001),
    dtype="float32",
    device=str(DEVICE),
    apply_b_dec_to_input=True,
    normalize_activations="none",
)

sae = StandardTrainingSAE(sae_cfg)
sae.load_state_dict(checkpoint['sae_state_dict'])
sae = sae.to(DEVICE)
sae.eval()

print(f"SAE loaded: layer={LAYER}")

# =========================
# 加载分析结果
# =========================
print("\nLoading analysis results...")
with open(ANALYSIS_RESULTS, 'r') as f:
    analysis = json.load(f)

# 获取top features
top_features_overall = analysis['top_features']['indices'][:10]
top_features_x = analysis['top_features_per_dim']['x']['indices'][:5]
top_features_y = analysis['top_features_per_dim']['y']['indices'][:5]
top_features_z = analysis['top_features_per_dim']['z']['indices'][:5]

print(f"Top features (overall): {top_features_overall}")
print(f"Top features (x): {top_features_x}")
print(f"Top features (y): {top_features_y}")
print(f"Top features (z): {top_features_z}")

# =========================
# 加载测试数据
# =========================
print("\nLoading test data...")
with open(TEST_DATA_FILE, 'r') as f:
    test_data = json.load(f)

# 随机采样
np.random.seed(42)
if len(test_data) > NUM_SAMPLES:
    indices = np.random.choice(len(test_data), NUM_SAMPLES, replace=False)
    test_data = [test_data[i] for i in indices]

print(f"Testing on {len(test_data)} samples")

# =========================
# 干预函数
# =========================
def get_representation_with_sae(prompt, layer_idx):
    """获取原始表示和SAE编码"""
    tokens = model.to_tokens(prompt, truncate=True)
    
    with torch.no_grad():
        _, cache = model.run_with_cache(
            tokens,
            names_filter=f"blocks.{layer_idx}.hook_resid_post"
        )
    
    hidden = cache[f"blocks.{layer_idx}.hook_resid_post"][0, -1]  # [d_model]
    
    # SAE编码
    with torch.no_grad():
        feature_acts = sae.encode(hidden.unsqueeze(0))[0]  # [d_sae]
        reconstructed = sae.decode(feature_acts.unsqueeze(0))[0]  # [d_model]
    
    return hidden, feature_acts, reconstructed

def intervene_feature(feature_acts, feature_idx, magnitude):
    """干预特定特征"""
    intervened = feature_acts.clone()
    
    # 方法1: 置零
    # intervened[feature_idx] = 0
    
    # 方法2: 增强或抑制
    if intervened[feature_idx] > 0:
        # 如果已激活，则增强
        intervened[feature_idx] = intervened[feature_idx] * magnitude
    else:
        # 如果未激活，则激活到平均水平
        intervened[feature_idx] = magnitude
    
    return intervened

def decode_sae_features(feature_acts):
    """使用SAE解码器重构hidden state"""
    with torch.no_grad():
        # 使用SAE的decode方法
        reconstructed = sae.decode(feature_acts.unsqueeze(0))[0]
    return reconstructed

# =========================
# 实验1: 单特征干预
# =========================
print("\n" + "="*50)
print("Experiment 1: Single Feature Intervention")
print("="*50)

intervention_results = {
    'overall': {},
    'x_axis': {},
    'y_axis': {},
    'z_axis': {},
}

def run_single_feature_intervention(feature_idx, feature_name):
    """运行单特征干预实验"""
    changes = []
    
    for sample in tqdm(test_data, desc=f"Intervening feature {feature_idx}", leave=False):
        prompt = sample['prompt']
        target = np.array(sample['target'])
        
        # 获取原始表示
        hidden_orig, feature_acts_orig, _ = get_representation_with_sae(prompt, LAYER)
        
        # 干预特征
        feature_acts_intervened = intervene_feature(feature_acts_orig, feature_idx, INTERVENTION_MAG)
        
        # 重构
        hidden_intervened = decode_sae_features(feature_acts_intervened)
        
        # 计算变化
        change = (hidden_intervened - hidden_orig).cpu().numpy()
        change_norm = np.linalg.norm(change)
        
        changes.append({
            'target': target.tolist(),
            'change_norm': float(change_norm),
            'feature_activation_orig': float(feature_acts_orig[feature_idx].cpu().numpy()),
            'feature_activation_intervened': float(feature_acts_intervened[feature_idx].cpu().numpy()),
        })
    
    avg_change = np.mean([c['change_norm'] for c in changes])
    avg_activation_orig = np.mean([c['feature_activation_orig'] for c in changes])
    
    print(f"\n{feature_name} (Feature {feature_idx}):")
    print(f"  Avg change in hidden state: {avg_change:.4f}")
    print(f"  Avg original activation: {avg_activation_orig:.4f}")
    
    return {
        'feature_idx': int(feature_idx),
        'avg_change': float(avg_change),
        'avg_activation_orig': float(avg_activation_orig),
        'details': changes[:10],  # 只保存前10个样本的详细信息
    }

# 测试top overall features
print("\nIntervening top overall features...")
for i, feat_idx in enumerate(top_features_overall[:5]):
    feat_idx = int(feat_idx)  # 确保是Python int
    result = run_single_feature_intervention(feat_idx, f"Top Overall Feature {i+1}")
    intervention_results['overall'][f'feature_{feat_idx}'] = result

# 测试每个维度的top features
print("\nIntervening top x-axis features...")
for i, feat_idx in enumerate(top_features_x[:3]):
    feat_idx = int(feat_idx)  # 确保是Python int
    result = run_single_feature_intervention(feat_idx, f"Top X Feature {i+1}")
    intervention_results['x_axis'][f'feature_{feat_idx}'] = result

print("\nIntervening top y-axis features...")
for i, feat_idx in enumerate(top_features_y[:3]):
    feat_idx = int(feat_idx)  # 确保是Python int
    result = run_single_feature_intervention(feat_idx, f"Top Y Feature {i+1}")
    intervention_results['y_axis'][f'feature_{feat_idx}'] = result

print("\nIntervening top z-axis features...")
for i, feat_idx in enumerate(top_features_z[:3]):
    feat_idx = int(feat_idx)  # 确保是Python int
    result = run_single_feature_intervention(feat_idx, f"Top Z Feature {i+1}")
    intervention_results['z_axis'][f'feature_{feat_idx}'] = result

# =========================
# 实验2: 对比基线（随机特征）
# =========================
print("\n" + "="*50)
print("Experiment 2: Random Feature Baseline")
print("="*50)

# 随机选择一些特征作为对照
random_features = np.random.choice(sae_config['d_sae'], 10, replace=False)
random_results = []

for feat_idx in random_features:
    feat_idx = int(feat_idx)  # 确保是Python int
    result = run_single_feature_intervention(feat_idx, f"Random Feature")
    random_results.append(result)

intervention_results['random_baseline'] = {
    'features': [int(f) for f in random_features],  # 转换为Python int列表
    'results': random_results,
    'avg_change': float(np.mean([r['avg_change'] for r in random_results])),
}

# =========================
# 保存结果
# =========================
print("\n" + "="*50)
print("Saving Results")
print("="*50)

output_file = os.path.join(OUTPUT_DIR, f"intervention_results_layer{LAYER}.json")
with open(output_file, 'w') as f:
    json.dump(intervention_results, f, indent=2)

print(f"Results saved to {output_file}")

# =========================
# 可视化
# =========================
print("\nGenerating visualizations...")

# 收集所有干预的平均变化
all_changes = []
all_labels = []

for feat_idx in top_features_overall[:5]:
    key = f'feature_{feat_idx}'
    if key in intervention_results['overall']:
        all_changes.append(intervention_results['overall'][key]['avg_change'])
        all_labels.append(f'Overall\nF{feat_idx}')

for feat_idx in top_features_x[:3]:
    key = f'feature_{feat_idx}'
    if key in intervention_results['x_axis']:
        all_changes.append(intervention_results['x_axis'][key]['avg_change'])
        all_labels.append(f'X-axis\nF{feat_idx}')

for feat_idx in top_features_y[:3]:
    key = f'feature_{feat_idx}'
    if key in intervention_results['y_axis']:
        all_changes.append(intervention_results['y_axis'][key]['avg_change'])
        all_labels.append(f'Y-axis\nF{feat_idx}')

for feat_idx in top_features_z[:3]:
    key = f'feature_{feat_idx}'
    if key in intervention_results['z_axis']:
        all_changes.append(intervention_results['z_axis'][key]['avg_change'])
        all_labels.append(f'Z-axis\nF{feat_idx}')

# 添加随机基线
baseline_change = intervention_results['random_baseline']['avg_change']

plt.figure(figsize=(12, 6))
bars = plt.bar(range(len(all_changes)), all_changes, alpha=0.7)
plt.axhline(y=baseline_change, color='r', linestyle='--', label='Random Feature Baseline')
plt.xticks(range(len(all_changes)), all_labels, rotation=45, ha='right')
plt.ylabel('Avg Change in Hidden State Norm')
plt.title('Effect of Single Feature Intervention')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, f'intervention_effects_layer{LAYER}.png'), dpi=300)
plt.close()

print(f"Visualizations saved to {OUTPUT_DIR}")

print("\n" + "="*50)
print("SAE Feature Intervention Complete!")
print("="*50)




