# SA-of-LLM

**Spatial Awareness of Large Language Models**

CCL 2026 口头汇报：[论文](https://blcuicall.org/papers/ccl26_ajy.pdf) · [幻灯片](slides/大语言模型空间推理能力的内部表征机制研究汇报_20261017_安纪元_V9.pdf)

研究大语言模型的空间感知：模型能否完成多跳空间推理，以及空间信息如何编码在内部表征中。实验覆盖英文、中文和阿拉伯语，模型为 Qwen2.5-7B 与 Llama-3-8B 的 Base / Instruct 版本。

## 任务

三个任务族都由程序生成，并带有可计算的标准答案。

| 任务族 | 目录 | 问的是什么 | 监督信号 |
| --- | --- | --- | --- |
| 1. 空间关系 | `src/data_generation/task_family_1` | 根据一组左/右、前/后、上/下关系，推断两个实体的复合空间关系 | 三维关系向量 |
| 2. 方向转向 | `src/data_generation/task_family_2` | 从初始朝向出发，执行左转、右转、转身，判断最终朝向 | 方向角的 `(cos θ, sin θ)` |
| 3. 空间过程 | `src/data_generation/task_family_3` | 执行移动、旋转、反射、缩放、平移，求最终坐标 | 三维坐标 |

语言代码：`EN` 英文，`CN` 中文，`AR` 阿拉伯语。带 `_with_prompt` 的文件是送入模型的完整提示；不带该后缀的文件是结构化样本。

## 方法

1. **行为准确率**。通过 OpenAI 兼容接口让模型作答，再统计选项准确率。
2. **线性探针**。在每一层残差流上训练 Ridge 回归，报告 R²、MAE、RMSE，用来定位空间信息出现的层。
3. **稀疏自编码器（SAE）**。用 [SAELens](https://github.com/decoderesearch/SAELens) 在探针表现最好的层上分解表征，再做特征重要性、梯度归因、因果干预、跨语言对比和最大激活样本分析。
4. **激活修补**。在残差流、注意力输出和 MLP 输出上做 activation patching，区分空间信息来自哪一类组件。
5. **补充因果实验**。剂量反应、只保留空间特征、显著性检验、消融稳健性、层级定位、匹配随机对照。

## 仓库结构

```
.
├── config.py                 # 模型路径解析
├── requirements.txt
├── scripts/
│   ├── env.example.sh        # LLM_ROOT 与推理接口
│   ├── run_llama3_8b_base.sh # 五阶段实验入口
│   └── sync_results.sh       # 把探针结果汇总成图表和 LaTeX 表
├── src/
│   ├── data_generation/      # 数据生成、提示构造、推理与准确率
│   ├── probe/                # 逐层线性探针与可视化
│   ├── sae_relation/         # 任务族 1 的 SAE 流程
│   ├── sae_orientation/      # 任务族 2 的 SAE 流程
│   ├── sae_procedure/        # 任务族 3 的 SAE 流程
│   ├── sae_exploratory/      # 早期 SAE 试验，正式结果以上面三个目录为准
│   ├── activation_patching/  # 激活修补
│   └── supplementary/        # 补充因果实验
├── artifacts/                # 已完成实验的分析输出；*.pt 不入库
└── archive/                  # 已被替代的旧脚本
```

目录曾用过的名字：`src/S` → `sae_exploratory`，`src/S_new_task1/2/3` → `sae_relation` / `sae_orientation` / `sae_procedure`，`src/add_exm` → `supplementary`，`src/ativation_patching` → `activation_patching`。

## 环境

需要一块至少能放下 7B/8B 模型的 GPU。探针、SAE 和激活修补使用 PyTorch、Transformers、TransformerLens 和 SAELens。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

模型权重不包含在本仓库中。把权重放到统一目录后：

```bash
cp scripts/env.example.sh scripts/env.local.sh
# 编辑 scripts/env.local.sh 中的 LLM_ROOT，然后
source scripts/env.local.sh
```

`config.py` 按以下顺序查找模型：

1. 单个模型的环境变量，例如 `MODEL_PATH_QWEN25_7B_INSTRUCT`
2. `$LLM_ROOT` 下的相对目录
3. Hugging Face 仓库名，由 `from_pretrained` 自行下载

行为评估还需要一个已经加载目标模型的 OpenAI 兼容服务：

```bash
export INFERENCE_API_BASE="http://127.0.0.1:8010/v1"
```

## 快速开始

数据集已经在 `src/data_generation/task_family_*/` 中。若要重新生成任务族 1 的英文数据，在对应目录运行：

```bash
cd src/data_generation/task_family_1
python generate_relation_EN.py
python generate_relation_EN_test.py
python add_prompt_relation_EN.py
```

中文、阿拉伯语以及另外两个任务族有同名的 `run_EN.sh`、`run_CN.sh`、`run_AR.sh`。

对一层残差流做探针（在 `src/probe/task_family_1` 下执行）：

```bash
python layer_sweep_probe_relation.py \
  -m "Qwen/Qwen2.5-7B-Instruct" \
  -tr "../../data_generation/task_family_1/data_relation/spatial_reasoning_dataset_EN_with_prompt.json" \
  -te "../../data_generation/task_family_1/data_relation/spatial_reasoning_dataset_EN_test_with_prompt.json" \
  -o probe_results_relation_EN.json
```

任务族 2、3 的统一入口分别是 `layer_sweep_probe_orientation.py` 和 `layer_sweep_probe_procedure_fixed.py`。画图：

```bash
python plot_results.py
```

SAE 全流程以任务族 1 的英文为例：

```bash
bash src/sae_relation/run_full_pipeline_EN.sh
```

训练产物写到该目录下新的 `outputs_<时间戳>/`。权重文件 `sae_final.pt` 会被 Git 忽略。

## 复现实验

`scripts/run_llama3_8b_base.sh` 把一次模型实验分成五个阶段。把脚本顶部的 `MODEL_LOCAL`、`MODEL_API`、`MODEL_DISPLAY` 改成目标模型后：

```bash
bash scripts/run_llama3_8b_base.sh        # 全部阶段
bash scripts/run_llama3_8b_base.sh 1      # 只做推理和准确率
bash scripts/run_llama3_8b_base.sh 2 3    # 探针 + SAE
```

| 阶段 | 内容 | 主要代码 |
| --- | --- | --- |
| 1 | 推理与准确率 | `src/data_generation/task_family_*/call_inference_api_*.py` |
| 2 | 逐层线性探针 | `src/probe/task_family_*/layer_sweep_probe_*.py` |
| 3 | SAE 训练、分析、归因、干预、可视化 | `src/sae_{relation,orientation,procedure}/` |
| 4 | 补充因果实验 | `src/supplementary/` |
| 5 | LaTeX 总表和探针图 | `generate_accuracy_big_table.py`、`generate_probe_big_table.py`、`plot_results.py` |

阶段 1 结束后，可用 `bash scripts/sync_results.sh` 把探针 JSON 收回绘图目录并重新生成图表。

激活修补示例：

```bash
python src/activation_patching/task_family_2/run_activation_patching_orientation_CN_v2_by_components.py
```

## 已有结果

探针曲线、热力图和 LaTeX 表在 `src/probe/` 与 `src/data_generation/latex_tables/`。SAE 的数值分析和语义分析在 `artifacts/`。这些 JSON 会随仓库发布；对应的 SAE 权重只留在本地。

## 引用

安纪元, 杨麟儿, 王梦焰, 鲁鹿鸣, 安维华, 杨尔弘. [大语言模型空间推理能力的内部表征机制研究](https://blcuicall.org/papers/ccl26_ajy.pdf). 第二十五届中国计算语言学大会 (CCL 2026)，口头汇报. [幻灯片](slides/大语言模型空间推理能力的内部表征机制研究汇报_20261017_安纪元_V9.pdf).

```bibtex
@inproceedings{an2026spatial-repr,
  title     = {大语言模型空间推理能力的内部表征机制研究},
  author    = {安纪元 and 杨麟儿 and 王梦焰 and 鲁鹿鸣 and 安维华 and 杨尔弘},
  booktitle = {第二十五届中国计算语言学大会 (CCL 2026)},
  year      = {2026},
  url       = {https://blcuicall.org/papers/ccl26_ajy.pdf}
}
```

## 许可

本仓库代码以 [MIT License](LICENSE) 发布。模型权重遵循各自原模型的许可，不在本许可范围内。
