import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional

import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer
from functools import partial

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from config import PATHS


def replace_question_in_full_prompt(full_prompt: str, new_question: str) -> str:
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
    with open(clean_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        data = [data]
    return {int(item["id"]): item for item in data}


def _iter_json_objects(file_handle):
    """按完整 JSON 对象迭代：支持多行美化的 JSON（每行不是一条记录）。"""
    buffer = []
    depth = 0
    in_string = False
    escape = False
    quote_char = None
    for line in file_handle:
        for c in line:
            if escape:
                escape = False
                continue
            if c == "\\" and in_string:
                escape = True
                continue
            if not in_string:
                if c == '"':
                    in_string = True
                    quote_char = c
                elif c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
            else:
                if c == quote_char:
                    in_string = False
            buffer.append(c)
        if depth == 0 and buffer:
            raw = "".join(buffer).strip()
            buffer = []
            if raw:
                try:
                    yield json.loads(raw)
                except json.JSONDecodeError:
                    pass
    if buffer:
        raw = "".join(buffer).strip()
        if raw:
            try:
                yield json.loads(raw)
            except json.JSONDecodeError:
                pass


def load_corrupted_pairs(
    corrupted_path: Path,
    clean_by_id: Dict[int, dict],
    max_samples: Optional[int] = None,
) -> List[dict]:
    pairs = []
    with open(corrupted_path, "r", encoding="utf-8") as f:
        for row in _iter_json_objects(f):
            if "error" in row:
                continue

            original_id = row.get("original_id")
            if original_id is None or int(original_id) not in clean_by_id:
                continue

            clean_sample = clean_by_id[int(original_id)]
            clean_full = clean_sample["prompt"]
            corrupted_full = replace_question_in_full_prompt(
                clean_full, row["corrupted_prompt"]
            )

            pairs.append(
                {
                    "pair_id": row.get("id"),
                    "original_id": original_id,
                    "clean_full": clean_full,
                    "corrupted_full": corrupted_full,
                    "correct_letter": clean_sample.get("correct_option", "A"),
                }
            )
            if max_samples and len(pairs) >= max_samples:
                break
    return pairs


def get_letter_token_id(tokenizer, letter: str) -> int:
    # 这里最好和你的实际答案模板保持一致；必要时改成 "\nA" 或 " A"
    toks = tokenizer.encode(letter, add_special_tokens=False)
    if len(toks) != 1:
        raise ValueError(f"Option {letter!r} is not a single token: {toks}")
    return toks[0]


def patch_resid_pre(corrupted_act, hook, clean_cache, pos: int):
    clean_act = clean_cache[hook.name]
    corrupted_act[:, pos, :] = clean_act[:, pos, :]
    return corrupted_act


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--clean_data", type=str, required=True)
    parser.add_argument("--corrupted_data", type=str, required=True)
    parser.add_argument("-o", "--output_dir", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    project_root = Path(__file__).resolve().parents[3]
    clean_path = project_root / args.clean_data
    corrupted_path = project_root / args.corrupted_data
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device if torch.cuda.is_available() else "cpu"
    # CPU 上 float16 支持差，易触发警告；CPU 用 bfloat16，GPU 用 float16
    dtype = torch.bfloat16 if device == "cpu" else torch.float16

    model_path = PATHS.get(args.model_name, args.model_name)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, trust_remote_code=True
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    model = HookedTransformer.from_pretrained(
        args.model_name,
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

    clean_by_id = load_clean_by_id(clean_path)
    pairs = load_corrupted_pairs(corrupted_path, clean_by_id, args.max_samples)

    # 先用第一条样本确定序列长度；如果后续长度不一致，直接跳过
    first_tokens = model.to_tokens(pairs[0]["clean_full"], truncate=True)
    n_pos = first_tokens.shape[1]

    # 存 layer x position 平均恢复值
    patch_scores = np.zeros((n_layers, n_pos), dtype=np.float64)
    patch_counts = np.zeros((n_layers, n_pos), dtype=np.int64)

    per_sample_results = []

    for pair in tqdm(pairs, desc="activation patching"):
        clean_tokens = model.to_tokens(pair["clean_full"], truncate=True)
        corrupt_tokens = model.to_tokens(pair["corrupted_full"], truncate=True)

        if clean_tokens.shape != corrupt_tokens.shape:
            # minimal pair 最好长度一致；不一致时先跳过
            continue

        if clean_tokens.shape[1] != n_pos:
            continue

        correct_tok = get_letter_token_id(tokenizer, pair["correct_letter"])

        with torch.no_grad():
            clean_logits, clean_cache = model.run_with_cache(clean_tokens, return_type="logits")
            corrupt_logits = model(corrupt_tokens)

        clean_correct = clean_logits[0, -1, correct_tok].item()
        corrupt_correct = corrupt_logits[0, -1, correct_tok].item()

        sample_grid = np.zeros((n_layers, n_pos), dtype=np.float32)

        for layer in range(n_layers):
            hook_name = f"blocks.{layer}.hook_resid_pre"
            for pos in range(n_pos):
                hook_fn = partial(patch_resid_pre, clean_cache=clean_cache, pos=pos)
                with torch.no_grad():
                    patched_logits = model.run_with_hooks(
                        corrupt_tokens,
                        fwd_hooks=[(hook_name, hook_fn)],
                        return_type="logits",
                    )

                patched_correct = patched_logits[0, -1, correct_tok].item()
                recovery = patched_correct - corrupt_correct

                patch_scores[layer, pos] += recovery
                patch_counts[layer, pos] += 1
                sample_grid[layer, pos] = recovery

        per_sample_results.append(
            {
                "pair_id": pair["pair_id"],
                "original_id": pair["original_id"],
                "clean_correct_logit": clean_correct,
                "corrupt_correct_logit": corrupt_correct,
                "patch_grid": sample_grid.tolist(),
            }
        )

    mean_patch_scores = np.divide(
        patch_scores,
        np.maximum(patch_counts, 1),
        where=(patch_counts > 0),
    )

    summary = {
        "model_name": args.model_name,
        "n_layers": n_layers,
        "n_positions": n_pos,
        "mean_patch_scores": mean_patch_scores.tolist(),
        "best_layer": int(np.unravel_index(np.argmax(mean_patch_scores), mean_patch_scores.shape)[0]),
        "best_position": int(np.unravel_index(np.argmax(mean_patch_scores), mean_patch_scores.shape)[1]),
    }

    with open(output_dir / "activation_patching_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(output_dir / "activation_patching_per_sample.jsonl", "w", encoding="utf-8") as f:
        for row in per_sample_results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()