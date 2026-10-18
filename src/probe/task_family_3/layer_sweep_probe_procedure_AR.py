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
parser.add_argument("--output_file_path", "-o", type=str, default=f"probe_procedure_results_{time.strftime('%Y%m%d_%H%M%S')}.json")
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
# 1.5 Prompt 模板
# =========================
SYSTEM_PROMPT = """You are a spatial reasoning assistant."""

INSTRUCTION_TEMPLATE = """You are given a starting position and a sequence of spatial operations.
Each operation transforms the position in 3D space.

{procedure}

Question:
{question}

Options:
A. {option_A}
B. {option_B}
C. {option_C}
D. {option_D}

Instruction:
Output ONLY the letter of the correct option (A, B, C, or D).
Do NOT provide any explanation, reasoning steps, or additional text.
"""

def construct_prompt(sample):
    """构造完整的 prompt"""
    # 从 question 字段解析 procedure 和 question
    question_text = sample['question']
    lines = question_text.strip().split('\n')
    
    # 找到 "What is the final position?" 这一行
    question_line = ""
    procedure_lines = []
    
    for line in lines:
        if line.strip().startswith("What is"):
            question_line = line.strip()
        elif line.strip():  # 非空行
            procedure_lines.append(line)
    
    procedure = '\n'.join(procedure_lines)
    
    # 获取选项
    options = sample['options']
    option_A = options[0]
    option_B = options[1]
    option_C = options[2]
    option_D = options[3]
    
    # 构造 prompt
    prompt = INSTRUCTION_TEMPLATE.format(
        procedure=procedure,
        question=question_line,
        option_A=option_A,
        option_B=option_B,
        option_C=option_C,
        option_D=option_D
    )
    
    return prompt

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
        prompt = sample['prompt'] #construct_prompt(sample)
        train_data.append({
            "prompt": prompt,
            "target": np.array(sample["target"], dtype=np.float32)
        })

test_data = []
with open(TEST_DATA_FILE_PATH, "r") as f:
    data = json.load(f)
    for sample in data:
        # 构造完整的 prompt
        prompt = sample['prompt'] #construct_prompt(sample)
        test_data.append({
            "prompt": prompt,
            "target": np.array(sample["target"], dtype=np.float32)
        })

print(f"Loaded {len(train_data)} training samples")
print(f"Loaded {len(test_data)} test samples")

# =========================
# 4. 工具函数：提取 resid_post
# =========================
def collect_hidden_states(data, layer_idx):
    """
    对指定 layer_idx，收集 resid_post hidden states
    返回：
        X: [N, d_model]
        Y: [N, 3]  # 三维：[x, y, z] 表示最终位置
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

for layer in range(n_layers):
    print(f"\nProcessing Layer {layer}/{n_layers-1}...")
    
    X_train, Y_train = collect_hidden_states(train_data, layer)
    X_test, Y_test = collect_hidden_states(test_data, layer)

    # Ridge 回归（预测最终位置 x, y, z）
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
    'task_type': 'spatial_procedure_execution'
}

output_file = OUTPUT_FILE_PATH
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {output_file}")

