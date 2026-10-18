"""
Activation Patching for Orientation Reasoning (CN)
==================================================
流程：
  Step 1: 运行 clean prompt，保存每层 h_l(clean)
  Step 2: 运行 corrupted prompt，保存 h_l(corrupt) 与 corrupted_output
  Step 3: 对每一层做 activation patching: h_l(corrupt) <- h_l(clean)，然后继续 forward
  Step 4: 测量恢复程度: logit(correct) - logit(wrong) 或 answer probability
  Step 5: 计算 patch score: recovery_score = logit_patch - logit_corrupt
"""

import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from config import PATHS

import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer

# =========================
# 数据路径与配置
# =========================
def replace_question_in_full_prompt(full_prompt: str, new_question: str) -> str:
    """将完整 prompt 中的「初始方向+动作+问题」块替换为 new_question。"""
    block = new_question.replace(
        "\n\n你现在面向哪个方向？", "\n\n\n问题：\n你现在面向哪个方向？"
    )
    start_marker = "初始方向和动作：\n"
    option_marker = "\n\n选项："
    if start_marker not in full_prompt or option_marker not in full_prompt:
        return full_prompt
    i0 = full_prompt.index(start_marker) + len(start_marker)
    i1 = full_prompt.index(option_marker, i0)
    return full_prompt[:i0] + block + full_prompt[i1:]


def load_clean_by_id(clean_path: Path) -> Dict[int, dict]:
    """加载 clean JSON，返回 id -> sample（含 prompt, options, correct_option）。"""
    with open(clean_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        data = [data]
    return {int(item["id"]): item for item in data}


def load_corrupted_pairs(
    corrupted_path: Path,
    clean_by_id: Dict[int, dict],
    max_samples: Optional[int] = None,
) -> List[dict]:
    """
    加载 corrupted JSONL，与 clean 对齐，生成 (clean_full_prompt, corrupted_full_prompt, correct_letter, wrong_letter)。
    支持每行一个 JSON 或多行缩进 JSON（逐行累积直到解析成功）。
    """
    pairs = []
    buffer = ""
    with open(corrupted_path, "r", encoding="utf-8") as f:
        for line in f:
            buffer += line
            try:
                row = json.loads(buffer)
                buffer = ""
            except json.JSONDecodeError:
                continue
            if "error" in row:
                continue
            original_id = row.get("original_id")
            if original_id is None:
                continue
            clean_sample = clean_by_id.get(int(original_id))
            if not clean_sample:
                continue
            full_prompt = clean_sample.get("prompt")
            if not full_prompt:
                continue
            clean_full = full_prompt
            corrupted_full = replace_question_in_full_prompt(
                full_prompt, row["corrupted_prompt"]
            )
            correct_letter = clean_sample.get("correct_option", "A")
            options = clean_sample.get("options", [])
            corrupted_short = row.get("corrupted_answer_short") or row.get("corrupted_answer", "")
            if isinstance(corrupted_short, str) and "方" in corrupted_short:
                corrupted_short = corrupted_short.replace("方", "").strip() or corrupted_short
            wrong_letter = None
            if options and corrupted_short:
                try:
                    idx = options.index(corrupted_short)
                    wrong_letter = "ABCD"[idx]
                except (ValueError, IndexError):
                    pass
            if wrong_letter is None:
                wrong_letter = "ABCD"[0] if correct_letter != "A" else "B"
            pairs.append({
                "clean_full": clean_full,
                "corrupted_full": corrupted_full,
                "correct_letter": correct_letter,
                "wrong_letter": wrong_letter,
                "original_id": original_id,
                "pair_id": row.get("id", len(pairs)),
            })
            if max_samples and len(pairs) >= max_samples:
                break
    return pairs


def get_option_token_ids(tokenizer, option_letters: List[str]) -> Dict[str, int]:
    """获取选项字母对应 token id（单 token）。"""
    ids = {}
    for letter in option_letters:
        tok = tokenizer.encode(letter, add_special_tokens=False)
        if len(tok) == 1:
            ids[letter] = tok[0]
        else:
            # 多 token 时取第一个
            ids[letter] = tok[0] if tok else 0
    return ids


def logit_diff_metric(
    logits: torch.Tensor,
    correct_idx: int,
    wrong_idx: int,
    position: int = -1,
) -> float:
    """logits: [batch, seq, vocab]，取 position 位置的 logit(correct) - logit(wrong)。"""
    logits_at = logits[0, position, :].float()
    return (logits_at[correct_idx] - logits_at[wrong_idx]).item()


def main():
    parser = argparse.ArgumentParser(description="Activation Patching: Orientation CN")
    parser.add_argument("--model_name", "-m", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument(
        "--clean_data",
        type=str,
        default="src/data_generation/task_family_2/data_orientation/orientation_reasoning_dataset_CN_with_prompt.json",
        help="Clean prompt JSON 路径（相对项目根）",
    )
    parser.add_argument(
        "--corrupted_data",
        type=str,
        default="src/data_generation/task_family_2/data_orientation_corrupted/orientation_reasoning_dataset_CN_corrupted.jsonl",
        help="Corrupted JSONL 路径",
    )
    parser.add_argument("--output_dir", "-o", type=str, default="src/activation_patching/task_family_3/out_orientation_cn")
    parser.add_argument("--max_samples", type=int, default=None, help="最多使用的 corrupted 样本数")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[3]
    clean_path = project_root / args.clean_data
    corrupted_path = project_root / args.corrupted_data
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device if torch.cuda.is_available() else "cpu"
    dtype = torch.float16

    # -------------------------
    # 加载模型
    # -------------------------
    model_name = args.model_name
    model_path = PATHS.get(model_name, model_name)
    print("Loading model:", model_path)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, trust_remote_code=True
    )
    hf_model = hf_model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = HookedTransformer.from_pretrained(
        model_name,
        hf_model=hf_model,
        tokenizer=tokenizer,
        dtype=dtype,
        device=device,
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
        fold_value_biases=False,
    )
    model.eval()
    n_layers = model.cfg.n_layers
    print(f"Model: {n_layers} layers")

    # -------------------------
    # 加载数据
    # -------------------------
    clean_by_id = load_clean_by_id(clean_path)
    pairs = load_corrupted_pairs(corrupted_path, clean_by_id, args.max_samples)
    print(f"Loaded {len(pairs)} clean/corrupted pairs")

    option_ids = get_option_token_ids(tokenizer, ["A", "B", "C", "D"])

    # 收集每层 recovery 的聚合（按样本平均）
    layer_recovery_scores = np.zeros(n_layers)
    layer_counts = np.zeros(n_layers)

    # 可选：保存每个样本的详细结果
    per_sample_results = []

    for pair_idx, pair in enumerate(tqdm(pairs, desc="Activation Patching")):
        clean_full = pair["clean_full"]
        corrupted_full = pair["corrupted_full"]
        correct_letter = pair["correct_letter"]
        wrong_letter = pair["wrong_letter"]
        correct_tok = option_ids.get(correct_letter, 0)
        wrong_tok = option_ids.get(wrong_letter, 0)

        tokens_clean = model.to_tokens(clean_full, truncate=True)
        tokens_corrupt = model.to_tokens(corrupted_full, truncate=True)

        # Step 1: 运行 clean，保存每层 h_l(clean)
        with torch.no_grad():
            _, cache_clean = model.run_with_cache(tokens_clean)

        # Step 2: 运行 corrupted，保存 cache 与 logits
        with torch.no_grad():
            logits_corrupt, cache_corrupt = model.run_with_cache(
                tokens_corrupt, return_type="logits"
            )
        seq_len = logits_corrupt.shape[1]
        metric_corrupt = logit_diff_metric(
            logits_corrupt, correct_tok, wrong_tok, position=seq_len - 1
        )

        # Step 3 & 4 & 5: 对每一层做 patching，计算 recovery
        def make_patch_hook(layer: int, clean_cache):
            clean_act = clean_cache[f"blocks.{layer}.hook_resid_post"]

            def hook(act, hook):
                if act.shape == clean_act.shape:
                    return clean_act
                # 序列长度不同时只修补最后一位置（预测位置）
                out = act.clone()
                out[:, -1, :] = clean_act[:, -1, :]
                return out

            return hook

        sample_recovery = np.zeros(n_layers)
        for layer in range(n_layers):
            hook_name = f"blocks.{layer}.hook_resid_post"
            hook_fn = make_patch_hook(layer, cache_clean)
            with torch.no_grad():
                logits_patch = model.run_with_hooks(
                    tokens_corrupt,
                    fwd_hooks=[(hook_name, hook_fn)],
                    return_type="logits",
                )
            metric_patch = logit_diff_metric(
                logits_patch, correct_tok, wrong_tok, position=seq_len - 1
            )
            recovery = metric_patch - metric_corrupt
            sample_recovery[layer] = recovery
            layer_recovery_scores[layer] += recovery
            layer_counts[layer] += 1

        per_sample_results.append({
            "pair_id": pair.get("pair_id"),
            "original_id": pair.get("original_id"),
            "metric_corrupt": metric_corrupt,
            "recovery_per_layer": sample_recovery.tolist(),
        })

    # 平均
    for layer in range(n_layers):
        if layer_counts[layer] > 0:
            layer_recovery_scores[layer] /= layer_counts[layer]

    # -------------------------
    # 保存结果
    # -------------------------
    summary = {
        "n_layers": n_layers,
        "n_samples": len(pairs),
        "layer_recovery_score_mean": layer_recovery_scores.tolist(),
        "best_layer": int(np.argmax(layer_recovery_scores)),
        "model_name": model_name,
    }
    out_summary = output_dir / "activation_patching_summary.json"
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Summary saved to {out_summary}")
    print("Layer recovery (mean):", [round(x, 4) for x in layer_recovery_scores])
    print("Best layer:", summary["best_layer"])

    out_detail = output_dir / "activation_patching_per_sample.jsonl"
    with open(out_detail, "w", encoding="utf-8") as f:
        for r in per_sample_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Per-sample results saved to {out_detail}")


if __name__ == "__main__":
    main()
