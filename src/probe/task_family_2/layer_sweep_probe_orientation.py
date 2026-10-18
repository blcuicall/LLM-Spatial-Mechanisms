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
from sklearn.metrics import r2_score
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", "-m", type=str)
parser.add_argument("--train_data_file_path", "-tr", type=str)
parser.add_argument("--test_data_file_path", "-te", type=str)
parser.add_argument("--output_file_path", "-o", type=str, default=f"probe_results_orientation_{time.strftime('%Y%m%d_%H%M%S')}.json")
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
        prompt = sample["prompt"]
        train_data.append({
            "prompt": prompt,
            "target": np.array(sample["target"])  # [cos(θ), sin(θ)]
        })

test_data = []
with open(TEST_DATA_FILE_PATH, "r") as f:
    data = json.load(f)
    for sample in data:
        # 构造完整的 prompt
        prompt = sample["prompt"]
        test_data.append({
            "prompt": prompt,
            "target": np.array(sample["target"])  # [cos(θ), sin(θ)]
        })

# =========================
# 4. 工具函数：提取 resid_post
# =========================
def collect_hidden_states(data, layer_idx):
    """
    对指定 layer_idx，收集 resid_post hidden states
    返回：
        X: [N, d_model]
        Y: [N, 2]  # 二维：[cos(θ), sin(θ)]
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

for layer in range(n_layers):
    print(f"\nProcessing Layer {layer}/{n_layers-1}...")
    
    X_train, Y_train = collect_hidden_states(train_data, layer)
    X_test, Y_test = collect_hidden_states(test_data, layer)

    # Ridge 回归（预测 cos(θ), sin(θ)）
    probe = Ridge(alpha=RIDGE_ALPHA)
    probe.fit(X_train, Y_train)

    # 在测试集上预测
    Y_pred = probe.predict(X_test)
    r2 = r2_score(Y_test, Y_pred, multioutput="uniform_average")
    layer_r2.append(r2)
    
    # 在训练集上预测（检查是否能拟合）
    Y_train_pred = probe.predict(X_train)
    r2_train = r2_score(Y_train, Y_train_pred, multioutput="uniform_average")
    
    # === 诊断信息（仅对第一层、中间层和最后一层） ===
    if layer == 0 or layer == n_layers // 2 or layer == n_layers - 1:
        # 计算基准（用均值预测的MSE）
        Y_mean_pred = np.tile(Y_train.mean(axis=0), (len(Y_test), 1))
        baseline_mse = np.mean((Y_test - Y_mean_pred) ** 2)
        model_mse = np.mean((Y_test - Y_pred) ** 2)
        
        # 分别计算cos和sin的R²
        r2_cos = r2_score(Y_test[:, 0], Y_pred[:, 0])
        r2_sin = r2_score(Y_test[:, 1], Y_pred[:, 1])
        
        print(f"\n{'='*50}")
        print(f"Layer {layer} 详细诊断:")
        print(f"  数据量: train={len(Y_train)}, test={len(Y_test)}")
        print(f"  训练集目标: 均值={Y_train.mean(axis=0)}, std={Y_train.std(axis=0)}")
        print(f"  测试集目标: 均值={Y_test.mean(axis=0)}, std={Y_test.std(axis=0)}")
        print(f"  预测值(test): 均值={Y_pred.mean(axis=0)}, std={Y_pred.std(axis=0)}")
        print(f"  MSE (baseline/model): {baseline_mse:.6f} / {model_mse:.6f}")
        print(f"  R² on TRAIN: {r2_train:.4f}  <- 能否拟合训练集？")
        print(f"  R² on TEST:  {r2:.4f}       <- 泛化能力")
        print(f"  R² by dim (cos/sin): {r2_cos:.4f} / {r2_sin:.4f}")
        print(f"{'='*50}\n")

    print(f"Layer {layer:02d} | R²_train = {r2_train:.4f}, R²_test = {r2:.4f}")

# =========================
# 6. 结果输出
# =========================
print("\n" + "="*50)
print("===== Layer Sweep Result =====")
print("="*50)
for i, r2 in enumerate(layer_r2):
    marker = " <<<" if i == np.argmax(layer_r2) else ""
    print(f"Layer {i:02d}: R² = {r2:.4f}{marker}")

best_layer = int(np.argmax(layer_r2))
print("\n" + "="*50)
print(f">>> Best layer: {best_layer} (R²={layer_r2[best_layer]:.4f})")
print("="*50)

# =========================
# 7. 保存结果（可选）
# =========================
results = {
    'layer_r2': layer_r2,
    'best_layer': best_layer,
    'n_layers': n_layers,
    'd_model': d_model,
}

import json
output_file = OUTPUT_FILE_PATH
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {output_file}")

