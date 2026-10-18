import sys
sys.path.append("./")
sys.path.append("../../")
sys.path.append("../../../")
from config import PATHS

import time
import torch
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", "-m", type=str)
parser.add_argument("--train_data_file_path", "-tr", type=str)
parser.add_argument("--test_data_file_path", "-te", type=str)
parser.add_argument("--output_file_path", "-o", type=str, default=f"probe_procedure_results_fixed_{time.strftime('%Y%m%d_%H%M%S')}.json")
parser.add_argument("--ridge_alpha", type=float, default=10.0, help="Ridge regularization strength (default: 10.0)")
parser.add_argument("--standardize", action="store_true", help="Standardize features and targets")
args = parser.parse_args()

MODEL_NAME = args.model_name
TRAIN_DATA_FILE_PATH = args.train_data_file_path
TEST_DATA_FILE_PATH = args.test_data_file_path
OUTPUT_FILE_PATH = args.output_file_path

# =========================
# 1. 基本配置
# =========================
MODEL_PATH = PATHS[MODEL_NAME]
DEVICE = "cuda:0"
DTYPE = torch.float16
BATCH_SIZE = 4
RIDGE_ALPHA = args.ridge_alpha  # 增加正则化强度
USE_STANDARDIZATION = args.standardize
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("=" * 70)
print("FIXED VERSION: Layer Sweep Probe with Standardization")
print("=" * 70)
print(f"Ridge Alpha: {RIDGE_ALPHA} (原版: 1.0)")
print(f"Standardization: {USE_STANDARDIZATION}")
print(f"Random Seed: {SEED}")

# =========================
# 2. 加载模型
# =========================
print("\nLoading model...")
print(f"Model path: {MODEL_PATH}")

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

print("Converting to HookedTransformer...")

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

n_layers = model.cfg.n_layers
d_model = model.cfg.d_model

print(f"Loaded model with {n_layers} layers, d_model={d_model}")

# =========================
# 3. 加载数据
# =========================
import json

train_data = []
with open(TRAIN_DATA_FILE_PATH, "r") as f:
    data = json.load(f)
    for sample in data:
        prompt = sample['prompt']
        train_data.append({
            "prompt": prompt,
            "target": np.array(sample["target"], dtype=np.float32)
        })

test_data = []
with open(TEST_DATA_FILE_PATH, "r") as f:
    data = json.load(f)
    for sample in data:
        prompt = sample['prompt']
        test_data.append({
            "prompt": prompt,
            "target": np.array(sample["target"], dtype=np.float32)
        })

print(f"Loaded {len(train_data)} training samples")
print(f"Loaded {len(test_data)} test samples")

# 数据统计
train_targets = np.array([s['target'] for s in train_data])
test_targets = np.array([s['target'] for s in test_data])

print("\n" + "=" * 70)
print("Data Statistics:")
print("=" * 70)
print(f"Train targets - Mean: {train_targets.mean(axis=0)}")
print(f"Train targets - Std:  {train_targets.std(axis=0)}")
print(f"Test targets  - Mean: {test_targets.mean(axis=0)}")
print(f"Test targets  - Std:  {test_targets.std(axis=0)}")
print(f"Test/Train std ratio: {test_targets.std(axis=0) / train_targets.std(axis=0)}")

# 检查 z 轴方差
if test_targets.std(axis=0)[2] > train_targets.std(axis=0)[2] * 1.5:
    print("\n⚠️  WARNING: Test set z-axis variance is much larger than train!")
    print(f"   This may cause poor performance without standardization.")

# =========================
# 4. 工具函数：提取 resid_post
# =========================
def collect_hidden_states(data, layer_idx):
    """
    对指定 layer_idx，收集 resid_post hidden states
    返回：
        X: [N, d_model]
        Y: [N, 3]
    """
    X, Y = [], []

    for sample in tqdm(data, desc=f"Layer {layer_idx}", leave=False):
        tokens = model.to_tokens(sample["prompt"], truncate=True)

        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens,
                names_filter=f"blocks.{layer_idx}.hook_resid_post"
            )

        hidden = cache[f"blocks.{layer_idx}.hook_resid_post"][0, -1]
        hidden = hidden.float().cpu().numpy()

        X.append(hidden)
        Y.append(sample["target"])

    return np.stack(X), np.stack(Y)

# =========================
# 5. Layer Sweep Probe
# =========================
print("\n===== Starting Layer Sweep =====")
layer_r2 = []
layer_mae = []
layer_rmse = []

for layer in range(n_layers):
    print(f"\nProcessing Layer {layer}/{n_layers-1}...")
    
    X_train, Y_train = collect_hidden_states(train_data, layer)
    X_test, Y_test = collect_hidden_states(test_data, layer)

    # 标准化（如果启用）
    if USE_STANDARDIZATION:
        scaler_X = StandardScaler()
        scaler_Y = StandardScaler()
        
        X_train_scaled = scaler_X.fit_transform(X_train)
        Y_train_scaled = scaler_Y.fit_transform(Y_train)
        X_test_scaled = scaler_X.transform(X_test)
        
        # Ridge 回归
        probe = Ridge(alpha=RIDGE_ALPHA)
        probe.fit(X_train_scaled, Y_train_scaled)
        
        Y_pred_scaled = probe.predict(X_test_scaled)
        Y_pred = scaler_Y.inverse_transform(Y_pred_scaled)
    else:
        # 不标准化（原版）
        probe = Ridge(alpha=RIDGE_ALPHA)
        probe.fit(X_train, Y_train)
        Y_pred = probe.predict(X_test)

    # 计算评估指标
    r2 = r2_score(Y_test, Y_pred, multioutput="uniform_average")
    mae = mean_absolute_error(Y_test, Y_pred)
    rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
    
    layer_r2.append(float(r2))
    layer_mae.append(float(mae))
    layer_rmse.append(float(rmse))

    print(f"Layer {layer:02d} | R² = {r2:.4f} | MAE = {mae:.4f} | RMSE = {rmse:.4f}")

# =========================
# 6. 结果输出
# =========================
print("\n" + "="*50)
print("===== Layer Sweep Result =====")
print("="*50)
for i, (r2, mae, rmse) in enumerate(zip(layer_r2, layer_mae, layer_rmse)):
    marker = " <<<" if i == np.argmax(layer_r2) else ""
    print(f"Layer {i:02d}: R² = {r2:.4f} | MAE = {mae:.4f} | RMSE = {rmse:.4f}{marker}")

best_layer = int(np.argmax(layer_r2))
print("\n" + "="*50)
print(f">>> Best layer: {best_layer} (R²={layer_r2[best_layer]:.4f})")
print("="*50)

# 对比原版结果（如果有）
print("\n" + "="*50)
print("Improvement Summary:")
print("="*50)
print(f"Configuration:")
print(f"  - Ridge alpha: {RIDGE_ALPHA} (原版: 1.0)")
print(f"  - Standardization: {USE_STANDARDIZATION}")
print(f"  - Best layer: {best_layer}")
print(f"  - Best R²: {layer_r2[best_layer]:.4f}")
print(f"\n如果原版结果是 R²=0.047 (layer 0)，改进为:")
print(f"  - R² 提升: {layer_r2[best_layer] - 0.047:.4f}")
print(f"  - 相对提升: {(layer_r2[best_layer] / 0.047 - 1) * 100:.1f}%")

# =========================
# 7. 保存结果
# =========================
results = {
    'layer_r2': layer_r2,
    'layer_mae': layer_mae,
    'layer_rmse': layer_rmse,
    'best_layer': best_layer,
    'n_layers': n_layers,
    'd_model': d_model,
    'ridge_alpha': RIDGE_ALPHA,
    'use_standardization': USE_STANDARDIZATION,
    'task_type': 'spatial_procedure_execution',
    'train_samples': len(train_data),
    'test_samples': len(test_data),
    'train_target_stats': {
        'mean': train_targets.mean(axis=0).tolist(),
        'std': train_targets.std(axis=0).tolist()
    },
    'test_target_stats': {
        'mean': test_targets.mean(axis=0).tolist(),
        'std': test_targets.std(axis=0).tolist()
    }
}

output_file = OUTPUT_FILE_PATH
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {output_file}")

