"""
从 task_family_3 的 probe 结果 JSON 绘制 R2 / MAE / RMSE 随 layer 变化的曲线。
参考原有三组数据绘图逻辑，支持 12 个 JSON 文件（llama8b/qwen7b × base/instruct × AR/CN/EN）。
"""
import json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path

# 脚本所在目录即 task_family_3
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR
OUT_DIR = SCRIPT_DIR / "line_charts"

# 12 个 JSON 文件名（固定顺序便于图例一致）
JSON_NAMES = [
    "task_family_3_llama8b_base_AR.json",
    "task_family_3_llama8b_base_CN.json",
    "task_family_3_llama8b_base_EN.json",
    "task_family_3_llama8b_instruct_AR.json",
    "task_family_3_llama8b_instruct_CN.json",
    "task_family_3_llama8b_instruct_EN.json",
    "task_family_3_qwen7b_base_AR.json",
    "task_family_3_qwen7b_base_CN.json",
    "task_family_3_qwen7b_base_EN.json",
    "task_family_3_qwen7b_instruct_AR.json",
    "task_family_3_qwen7b_instruct_CN.json",
    "task_family_3_qwen7b_instruct_EN.json",
]


def load_dataset(json_path: Path) -> dict | None:
    """加载单个 JSON，返回与参考代码兼容的 data 字典。"""
    if not json_path.exists():
        return None
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    # 显示名：去掉 task_family_3_ 和 .json
    stem = json_path.stem
    name = stem.replace("task_family_3_", "") if stem.startswith("task_family_3_") else stem
    n = raw.get("n_layers") or len(raw["layer_r2"])
    return {
        "name": name,
        "layer_r2": raw["layer_r2"],
        "layer_mae": raw["layer_mae"],
        "layer_rmse": raw["layer_rmse"],
        "best_layer": raw.get("best_layer", 0),
        "n_layers": n,
    }


def load_all_datasets() -> list[dict]:
    """按 JSON_NAMES 顺序加载所有可用数据。"""
    datasets = []
    for jname in JSON_NAMES:
        path = DATA_DIR / jname
        d = load_dataset(path)
        if d is not None:
            datasets.append(d)
    return datasets


def main():
    datasets = load_all_datasets()
    if not datasets:
        print("未找到任何 JSON 数据，请确认路径与文件名。")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 12 条曲线：使用多种 marker + 颜色
    markers = ["o", "s", "^", "D", "v", "<", ">", "p", "h", "P", "*", "X"]
    colors = plt.cm.tab20.colors[:12] if len(datasets) <= 20 else plt.cm.tab20.colors

    metrics = [
        ("layer_r2", "R2"),
        ("layer_mae", "MAE"),
        ("layer_rmse", "RMSE"),
    ]

    saved = []

    for key, title in metrics:
        fig, ax = plt.subplots(figsize=(12, 7))
        for i, ds in enumerate(datasets):
            m = markers[i % len(markers)]
            c = colors[i % len(colors)]
            x = list(range(ds["n_layers"]))
            y = ds[key]
            ax.plot(x, y, marker=m, linewidth=1.8, markersize=4, label=ds["name"], color=c)
            if key == "layer_r2":
                b = ds["best_layer"]
                if 0 <= b < len(y):
                    ax.scatter([b], [y[b]], s=80, color=c, zorder=5, edgecolors="black", linewidths=0.8)

        ax.set_title(f"{title} by Layer (task_family_3)")
        ax.set_xlabel("Layer")
        ax.set_ylabel(title)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8, ncol=2)
        fig.tight_layout()

        path = OUT_DIR / f"{key}_task_family_3.png"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)

    # 合并为一个 PDF
    pdf_path = OUT_DIR / "all_layer_metrics_task_family_3.pdf"
    with PdfPages(pdf_path) as pdf:
        for key, title in metrics:
            fig, ax = plt.subplots(figsize=(12, 7))
            for i, ds in enumerate(datasets):
                m = markers[i % len(markers)]
                c = colors[i % len(colors)]
                x = list(range(ds["n_layers"]))
                y = ds[key]
                ax.plot(x, y, marker=m, linewidth=1.8, markersize=4, label=ds["name"], color=c)
                if key == "layer_r2":
                    b = ds["best_layer"]
                    if 0 <= b < len(y):
                        ax.scatter([b], [y[b]], s=80, color=c, zorder=5, edgecolors="black", linewidths=0.8)
            ax.set_title(f"{title} by Layer (task_family_3)")
            ax.set_xlabel("Layer")
            ax.set_ylabel(title)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best", fontsize=8, ncol=2)
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print("已生成曲线图（12 组数据）：")
    for p in saved:
        print(p)
    print(pdf_path)


if __name__ == "__main__":
    main()
