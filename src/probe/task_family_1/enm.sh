REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
# 英文空间推理任务
python ./layer_sweep_probe_relation.py \
    -m "Qwen/Qwen2.5-7B" \
    -tr "${REPO_ROOT}/src/data_generation/task_family_1/data_relation/spatial_reasoning_dataset_EN_with_prompt.json" \
    -te "${REPO_ROOT}/src/data_generation/task_family_1/data_relation/spatial_reasoning_dataset_EN_test_with_prompt.json"

# 中文空间推理任务
python ./layer_sweep_probe_relation.py \
    -m "Qwen/Qwen2.5-7B" \
    -tr "${REPO_ROOT}/src/data_generation/task_family_1/data_relation/spatial_reasoning_dataset_CN_with_prompt.json" \
    -te "${REPO_ROOT}/src/data_generation/task_family_1/data_relation/spatial_reasoning_dataset_CN_test_with_prompt.json"

# 阿拉伯语空间推理任务
python ./layer_sweep_probe_relation.py \
    -m "Qwen/Qwen2.5-7B" \
    -tr "${REPO_ROOT}/src/data_generation/task_family_1/data_relation/spatial_reasoning_dataset_AR_with_prompt.json" \
    -te "${REPO_ROOT}/src/data_generation/task_family_1/data_relation/spatial_reasoning_dataset_AR_test_with_prompt.json"