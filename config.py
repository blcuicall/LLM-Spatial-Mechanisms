"""Resolve local paths for the models used in this repository.

Lookup order for each model:

1. A dedicated environment variable, such as ``MODEL_PATH_QWEN25_7B_INSTRUCT``.
2. ``$LLM_ROOT/<relative directory>`` when ``LLM_ROOT`` is set.
3. The Hugging Face repository id, so ``from_pretrained`` can download it.
"""

import os
from pathlib import Path

LLM_ROOT = os.environ.get("LLM_ROOT", "")

# repo id -> (env var, directory under $LLM_ROOT)
_MODELS = {
    "Qwen/Qwen2.5-7B": ("MODEL_PATH_QWEN25_7B", "Qwen2.5/Qwen2.5-7B"),
    "Qwen/Qwen2.5-7B-Instruct": ("MODEL_PATH_QWEN25_7B_INSTRUCT", "Qwen2.5/Qwen2.5-7B-Instruct"),
    "meta-llama/Meta-Llama-3-8B": ("MODEL_PATH_LLAMA3_8B", "Llama3/Meta-Llama-3-8B"),
    "meta-llama/Meta-Llama-3-8B-Instruct": (
        "MODEL_PATH_LLAMA3_8B_INSTRUCT",
        "Llama3/Meta-Llama-3-8B-Instruct",
    ),
    "Qwen2.5-72B": ("MODEL_PATH_QWEN25_72B", "Qwen2.5/Qwen2.5-72B"),
    "Qwen2.5-72B-Instruct": ("MODEL_PATH_QWEN25_72B_INSTRUCT", "Qwen2.5/Qwen2.5-72B-Instruct"),
}

def _resolve(repo_id: str, env_key: str, relative: str) -> str:
    override = os.environ.get(env_key)
    if override:
        return override
    if LLM_ROOT:
        return str(Path(LLM_ROOT) / relative)
    return repo_id


PATHS = {
    repo_id: _resolve(repo_id, env_key, relative)
    for repo_id, (env_key, relative) in _MODELS.items()
}
