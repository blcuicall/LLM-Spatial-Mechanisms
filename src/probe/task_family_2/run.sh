REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
# # 英文视角变换任务
# python ./layer_sweep_probe_orientation.py \
#     -m "Qwen/Qwen2.5-7B-Instruct" \
#     -tr "${REPO_ROOT}/src/data_generation/task_family_2/data_orientation/orientation_reasoning_dataset_EN_with_prompt.json" \
#     -te "${REPO_ROOT}/src/data_generation/task_family_2/data_orientation/orientation_reasoning_dataset_EN_test_with_prompt.json"

# # 中文视角变换任务
# python ./layer_sweep_probe_orientation_CN.py \
#     -m "Qwen/Qwen2.5-7B-Instruct" \
#     -tr "${REPO_ROOT}/src/data_generation/task_family_2/data_orientation/orientation_reasoning_dataset_CN_with_prompt.json" \
#     -te "${REPO_ROOT}/src/data_generation/task_family_2/data_orientation/orientation_reasoning_dataset_CN_test_with_prompt.json"

# # 阿拉伯语视角变换任务
# python ./layer_sweep_probe_orientation_AR.py \
#     -m "Qwen/Qwen2.5-7B-Instruct" \
#     -tr "${REPO_ROOT}/src/data_generation/task_family_2/data_orientation/orientation_reasoning_dataset_AR_with_prompt.json" \
#     -te "${REPO_ROOT}/src/data_generation/task_family_2/data_orientation/orientation_reasoning_dataset_AR_test_with_prompt.json"

# # 英文视角变换任务
# python ./layer_sweep_probe_orientation.py \
#     -m "meta-llama/Meta-Llama-3-8B-Instruct" \
#     -tr "${REPO_ROOT}/src/data_generation/task_family_2/data_orientation/orientation_reasoning_dataset_EN_with_prompt.json" \
#     -te "${REPO_ROOT}/src/data_generation/task_family_2/data_orientation/orientation_reasoning_dataset_EN_test_with_prompt.json"

# # 中文视角变换任务
# python ./layer_sweep_probe_orientation.py \
#     -m "meta-llama/Meta-Llama-3-8B-Instruct" \
#     -tr "${REPO_ROOT}/src/data_generation/task_family_2/data_orientation/orientation_reasoning_dataset_CN_with_prompt.json" \
#     -te "${REPO_ROOT}/src/data_generation/task_family_2/data_orientation/orientation_reasoning_dataset_CN_test_with_prompt.json"

# # 阿拉伯语视角变换任务
# python ./layer_sweep_probe_orientation.py \
#     -m "meta-llama/Meta-Llama-3-8B-Instruct" \
#     -tr "${REPO_ROOT}/src/data_generation/task_family_2/data_orientation/orientation_reasoning_dataset_AR_with_prompt.json" \
#     -te "${REPO_ROOT}/src/data_generation/task_family_2/data_orientation/orientation_reasoning_dataset_AR_test_with_prompt.json"

# 英文视角变换任务
python ./layer_sweep_probe_orientation.py \
    -m "Qwen/Qwen2.5-7B" \
    -tr "${REPO_ROOT}/src/data_generation/task_family_2/data_orientation/orientation_reasoning_dataset_EN_with_prompt.json" \
    -te "${REPO_ROOT}/src/data_generation/task_family_2/data_orientation/orientation_reasoning_dataset_EN_test_with_prompt.json"

# 中文视角变换任务
python ./layer_sweep_probe_orientation.py \
    -m "Qwen/Qwen2.5-7B" \
    -tr "${REPO_ROOT}/src/data_generation/task_family_2/data_orientation/orientation_reasoning_dataset_CN_with_prompt.json" \
    -te "${REPO_ROOT}/src/data_generation/task_family_2/data_orientation/orientation_reasoning_dataset_CN_test_with_prompt.json"

# 阿拉伯语视角变换任务
python ./layer_sweep_probe_orientation.py \
    -m "Qwen/Qwen2.5-7B" \
    -tr "${REPO_ROOT}/src/data_generation/task_family_2/data_orientation/orientation_reasoning_dataset_AR_with_prompt.json" \
    -te "${REPO_ROOT}/src/data_generation/task_family_2/data_orientation/orientation_reasoning_dataset_AR_test_with_prompt.json"