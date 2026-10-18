"""
Activation Patching for Orientation Reasoning (CN) — 标准 recovery 版本
========================================================================
流程：
  Step 1: 运行 clean prompt，保存每层 h_l(clean) 与 logits_clean
  Step 2: 运行 corrupted prompt，保存 h_l(corrupt) 与 logits_corrupt
  Step 3: 对每一层做 activation patching: h_l(corrupt) <- h_l(clean)，然后继续 forward
  Step 4: 使用标准 metric（patching 文献常用）

Metric 定义：
  LD = logit(clean/correct answer) - logit(corrupt/wrong answer)
  即：LD = logit_diff_metric(logits, correct_idx, wrong_idx)

  归一化恢复率：
    recovery = (LD_patched - LD_corrupt) / (LD_clean - LD_corrupt)
  解释：
    0   = 没恢复
    1   = 完全恢复到 clean 水平
    <0  = 比 corrupt 还差
    >1  = 超过 clean（可能过补）
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
    """
    LD = logit(correct answer) - logit(wrong answer)。
    logits: [batch, seq, vocab]，取 position 位置的 logit diff。
    """
    logits_at = logits[0, position, :].float()
    return (logits_at[correct_idx] - logits_at[wrong_idx]).item()


def normalized_recovery(
    LD_patched: float,
    LD_corrupt: float,
    LD_clean: float,
    eps: float = 1e-10,
) -> float:
    """
    归一化恢复率（patching 文献常用）：
      recovery = (LD_patched - LD_corrupt) / (LD_clean - LD_corrupt)
    解释：0=没恢复，1=完全恢复，<0=变差，>1=超过 clean。
    分母为 0 时返回 np.nan（不参与后续平均时用 nanmean）。
    """
    denom = LD_clean - LD_corrupt
    if abs(denom) < eps:
        return np.nan
    return (LD_patched - LD_corrupt) / denom


def main():
    parser = argparse.ArgumentParser(description="Activation Patching: Orientation CN (normalized recovery)")
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
    parser.add_argument("--output_dir", "-o", type=str, default="src/activation_patching/task_family_2/out_orientation_cn_v2")
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

    # 按层收集归一化 recovery，用 list 以便 nanmean（部分样本分母可能为 0）
    layer_recovery_lists: List[List[float]] = [[] for _ in range(n_layers)]
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

        # Step 1: 运行 clean，保存 cache 与 logits，得到 LD_clean
        with torch.no_grad():
            logits_clean, cache_clean = model.run_with_cache(
                tokens_clean, return_type="logits"
            )
        seq_len_clean = logits_clean.shape[1]
        LD_clean = logit_diff_metric(
            logits_clean, correct_tok, wrong_tok, position=seq_len_clean - 1
        )

        # Step 2: 运行 corrupted，保存 cache 与 logits，得到 LD_corrupt
        with torch.no_grad():
            logits_corrupt, cache_corrupt = model.run_with_cache(
                tokens_corrupt, return_type="logits"
            )
        seq_len = logits_corrupt.shape[1]
        LD_corrupt = logit_diff_metric(
            logits_corrupt, correct_tok, wrong_tok, position=seq_len - 1
        )

        # Step 3 & 4: 对每一层做 patching，计算 LD_patched，再得归一化 recovery
        def make_patch_hook(layer: int, clean_cache):
            clean_act = clean_cache[f"blocks.{layer}.hook_resid_post"]

            def hook(act, hook):
                if act.shape == clean_act.shape:
                    return clean_act
                out = act.clone()
                out[:, -1, :] = clean_act[:, -1, :]
                return out

            return hook

        sample_recovery = np.full(n_layers, np.nan)
        for layer in range(n_layers):
            hook_name = f"blocks.{layer}.hook_resid_post"
            hook_fn = make_patch_hook(layer, cache_clean)
            with torch.no_grad():
                logits_patch = model.run_with_hooks(
                    tokens_corrupt,
                    fwd_hooks=[(hook_name, hook_fn)],
                    return_type="logits",
                )
            LD_patched = logit_diff_metric(
                logits_patch, correct_tok, wrong_tok, position=seq_len - 1
            )
            rec = normalized_recovery(LD_patched, LD_corrupt, LD_clean)
            sample_recovery[layer] = rec
            if not np.isnan(rec):
                layer_recovery_lists[layer].append(rec)

        per_sample_results.append({
            "pair_id": pair.get("pair_id"),
            "original_id": pair.get("original_id"),
            "LD_clean": LD_clean,
            "LD_corrupt": LD_corrupt,
            "recovery_per_layer": [float(x) if not np.isnan(x) else None for x in sample_recovery],
        })

    # 每层用 nanmean 等价：只对有效 recovery 求平均
    layer_recovery_scores = np.array([
        np.mean(v) if len(v) > 0 else np.nan
        for v in layer_recovery_lists
    ])
    valid_layers = ~np.isnan(layer_recovery_scores)
    best_layer = int(np.argmax(np.where(valid_layers, layer_recovery_scores, -np.inf)))

    # -------------------------
    # 保存结果
    # -------------------------
    summary = {
        "n_layers": n_layers,
        "n_samples": len(pairs),
        "metric_note": "recovery = (LD_patched - LD_corrupt) / (LD_clean - LD_corrupt); 0=no recovery, 1=full, <0=worse",
        "layer_recovery_score_mean": [float(x) if not np.isnan(x) else None for x in layer_recovery_scores],
        "best_layer": best_layer,
        "model_name": model_name,
    }
    out_summary = output_dir / "activation_patching_summary.json"
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Summary saved to {out_summary}")
    print("Layer recovery (mean, normalized):", [round(x, 4) if not np.isnan(x) else "nan" for x in layer_recovery_scores])
    print("Best layer:", best_layer)

    out_detail = output_dir / "activation_patching_per_sample.jsonl"
    with open(out_detail, "w", encoding="utf-8") as f:
        for r in per_sample_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Per-sample results saved to {out_detail}")


if __name__ == "__main__":
    main()
