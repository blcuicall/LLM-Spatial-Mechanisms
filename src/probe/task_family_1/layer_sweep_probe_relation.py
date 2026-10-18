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
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", "-m", type=str)
parser.add_argument("--train_data_file_path", "-tr", type=str)
parser.add_argument("--test_data_file_path", "-te", type=str)
parser.add_argument("--output_file_path", "-o", type=str, default=f"probe_results_{time.strftime('%Y%m%d_%H%M%S')}.json")
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
RIDGE_ALPHA = 1.0
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

# =========================
# 2. 加载模型
# =========================
print("Loading model...")
print(f"Model path: {MODEL_PATH}")

# 先用 transformers 加载 HF 模型
hf_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=DTYPE,
    trust_remote_code=True
)
# 手动将模型移到 GPU
hf_model = hf_model.to(DEVICE)

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True
)

print("Converting to HookedTransformer...")

# 转换为 HookedTransformer
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
        # 构造完整的 prompt
        prompt = sample['prompt']
        train_data.append({
            "prompt": prompt,
            "target": np.array(sample["target"])
        })

test_data = []
with open(TEST_DATA_FILE_PATH, "r") as f:
    data = json.load(f)
    for sample in data:
        # 构造完整的 prompt
        prompt = sample['prompt']
        test_data.append({
            "prompt": prompt,
            "target": np.array(sample["target"])
        })

# =========================
# 4. 工具函数：提取 resid_post
# =========================
def collect_hidden_states(data, layer_idx):
    """
    对指定 layer_idx，收集 resid_post hidden states
    返回：
        X: [N, d_model]
        Y: [N, 3]  # 三维：[x(left/right), y(below/above), z(behind/front)]
    """
    X, Y = [], []

    for sample in tqdm(data, desc=f"Layer {layer_idx}", leave=False):
        tokens = model.to_tokens(sample["prompt"], truncate=True)

        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens,
                names_filter=f"blocks.{layer_idx}.hook_resid_post"
            )

        # 取最后一个 token 的 resid_post
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
layer_r2_per_component = []  # 每个分量的R²

for layer in range(n_layers):
    print(f"\nProcessing Layer {layer}/{n_layers-1}...")
    
    X_train, Y_train = collect_hidden_states(train_data, layer)
    X_test, Y_test = collect_hidden_states(test_data, layer)

    # Ridge 回归（预测 Δx, Δy, Δz）
    probe = Ridge(alpha=RIDGE_ALPHA)
    probe.fit(X_train, Y_train)

    Y_pred = probe.predict(X_test)

    # 计算整体指标
    r2 = r2_score(Y_test, Y_pred, multioutput="uniform_average")
    mae = mean_absolute_error(Y_test, Y_pred)
    rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
    
    # 计算每个分量的R²
    r2_per_component = []
    component_names = ['x', 'y', 'z']
    for i in range(Y_test.shape[1]):
        r2_comp = r2_score(Y_test[:, i], Y_pred[:, i])
        r2_per_component.append(r2_comp)
    
    layer_r2.append(r2)
    layer_mae.append(mae)
    layer_rmse.append(rmse)
    layer_r2_per_component.append(r2_per_component)

    print(f"Layer {layer:02d} | R² = {r2:.4f} | MAE = {mae:.4f} | RMSE = {rmse:.4f}")
    print(f"           | R²(x) = {r2_per_component[0]:.4f}, R²(y) = {r2_per_component[1]:.4f}, R²(z) = {r2_per_component[2]:.4f}")

# =========================
# 6. 结果输出
# =========================
print("\n" + "="*50)
print("===== Layer Sweep Result =====")
print("="*50)
for i, r2 in enumerate(layer_r2):
    marker = " <<<" if i == np.argmax(layer_r2) else ""
    print(f"Layer {i:02d}: R² = {r2:.4f} | MAE = {layer_mae[i]:.4f} | RMSE = {layer_rmse[i]:.4f}{marker}")
    print(f"           | R²(x) = {layer_r2_per_component[i][0]:.4f}, R²(y) = {layer_r2_per_component[i][1]:.4f}, R²(z) = {layer_r2_per_component[i][2]:.4f}")

best_layer = int(np.argmax(layer_r2))
print("\n" + "="*50)
print(f">>> Best layer: {best_layer}")
print(f"    R² = {layer_r2[best_layer]:.4f}, MAE = {layer_mae[best_layer]:.4f}, RMSE = {layer_rmse[best_layer]:.4f}")
print(f"    R²(x) = {layer_r2_per_component[best_layer][0]:.4f}, R²(y) = {layer_r2_per_component[best_layer][1]:.4f}, R²(z) = {layer_r2_per_component[best_layer][2]:.4f}")
print("="*50)

# =========================
# 7. 保存结果（可选）
# =========================
results = {
    'layer_r2': layer_r2,
    'layer_mae': layer_mae,
    'layer_rmse': layer_rmse,
    'layer_r2_per_component': layer_r2_per_component,
    'best_layer': best_layer,
    'best_layer_metrics': {
        'r2': layer_r2[best_layer],
        'mae': layer_mae[best_layer],
        'rmse': layer_rmse[best_layer],
        'r2_x': layer_r2_per_component[best_layer][0],
        'r2_y': layer_r2_per_component[best_layer][1],
        'r2_z': layer_r2_per_component[best_layer][2],
    },
    'n_layers': n_layers,
    'd_model': d_model,
}

import json
output_file = OUTPUT_FILE_PATH
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {output_file}")

