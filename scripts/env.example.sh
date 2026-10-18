# Source this file after editing the paths for your machine:
#   source scripts/env.example.sh

# Root directory that contains the downloaded model weights.
# Expected layout:
#   $LLM_ROOT/Qwen2.5/Qwen2.5-7B
#   $LLM_ROOT/Qwen2.5/Qwen2.5-7B-Instruct
#   $LLM_ROOT/Llama3/Meta-Llama-3-8B
#   $LLM_ROOT/Llama3/Meta-Llama-3-8B-Instruct
export LLM_ROOT="${HOME}/models"

# OpenAI-compatible endpoint used by the behavioral evaluation scripts.
export INFERENCE_API_BASE="http://127.0.0.1:8010/v1"

# Optional per-model overrides. These take priority over $LLM_ROOT.
# export MODEL_PATH_QWEN25_7B_INSTRUCT="${LLM_ROOT}/Qwen2.5/Qwen2.5-7B-Instruct"
# export MODEL_PATH_LLAMA3_8B="${LLM_ROOT}/Llama3/Meta-Llama-3-8B"
