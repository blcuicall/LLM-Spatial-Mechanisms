"""
读取 task_family_3 下 12 个 probe 结果 JSON，生成可勾选折线的交互式 HTML。
运行后得到 line_charts/layer_metrics_interactive.html，用浏览器打开即可。
"""
import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR
OUT_DIR = SCRIPT_DIR / "line_charts"

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

# ECharts 常用颜色，与多系列区分
COLORS = [
    "#5470c6", "#91cc75", "#fac858", "#ee6666", "#73c0de",
    "#3ba272", "#fc8452", "#9a60b4", "#ea7ccc", "#5470c6",
    "#91cc75", "#fac858",
]


def load_dataset(json_path: Path) -> dict | None:
    if not json_path.exists():
        return None
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
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


def load_all() -> list[dict]:
    out = []
    for jname in JSON_NAMES:
        d = load_dataset(DATA_DIR / jname)
        if d is not None:
            out.append(d)
    return out


def main():
    datasets = load_all()
    if not datasets:
        print("未找到任何 JSON，请确认路径。")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 将数据序列化为 JS 字面量（无需 escape，JSON 已是安全数字/数组）
    data_js = json.dumps(datasets, ensure_ascii=False)
    colors_js = json.dumps(COLORS[: len(datasets)])

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Task Family 3 — Layer 指标折线（可勾选）</title>
  <script src="https://cdn.jsdelivr.net/npm/echarts@5.4.3/dist/echarts.min.js"></script>
  <style>
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: system-ui, sans-serif; background: #1a1a1e; color: #e4e4e7; min-height: 100vh; }}
    .container {{ display: flex; min-height: 100vh; }}
    .sidebar {{ width: 260px; padding: 16px; background: #252529; border-right: 1px solid #3f3f46; overflow-y: auto; }}
    .sidebar h2 {{ margin: 0 0 12px 0; font-size: 1rem; font-weight: 600; }}
    .metric {{ margin-bottom: 16px; }}
    .metric label {{ display: block; margin-bottom: 6px; font-size: 0.875rem; }}
    .metric select {{ width: 100%; padding: 8px; background: #3f3f46; border: 1px solid #52525b; color: #e4e4e7; border-radius: 6px; font-size: 0.875rem; }}
    .series {{ margin-bottom: 12px; }}
    .series h3 {{ margin: 0 0 8px 0; font-size: 0.8125rem; color: #a1a1aa; font-weight: 500; }}
    .series label {{ display: flex; align-items: center; gap: 8px; padding: 4px 0; cursor: pointer; font-size: 0.8125rem; }}
    .series label:hover {{ color: #fff; }}
    .series input {{ width: 16px; height: 16px; accent-color: #3b82f6; }}
    .btns {{ margin-top: 8px; display: flex; gap: 8px; }}
    .btns button {{ flex: 1; padding: 6px 10px; background: #3f3f46; border: 1px solid #52525b; color: #e4e4e7; border-radius: 6px; cursor: pointer; font-size: 0.8125rem; }}
    .btns button:hover {{ background: #52525b; }}
    .chart-wrap {{ flex: 1; padding: 16px; min-width: 0; }}
    #chart {{ width: 100%; height: 100%; min-height: 480px; }}
  </style>
</head>
<body>
  <div class="container">
    <aside class="sidebar">
      <h2>指标</h2>
      <div class="metric">
        <label for="metric">当前指标</label>
        <select id="metric">
          <option value="layer_r2">R²</option>
          <option value="layer_mae">MAE</option>
          <option value="layer_rmse">RMSE</option>
        </select>
      </div>
      <div class="series">
        <h3>显示折线（勾选即显示）</h3>
        <div id="checkboxes"></div>
        <div class="btns">
          <button type="button" id="selectAll">全选</button>
          <button type="button" id="selectNone">全不选</button>
        </div>
      </div>
    </aside>
    <main class="chart-wrap">
      <div id="chart"></div>
    </main>
  </div>

  <script>
    const DATASETS = {data_js};
    const COLORS = {colors_js};

    const metricSelect = document.getElementById('metric');
    const checkboxesDiv = document.getElementById('checkboxes');
    const chartDom = document.getElementById('chart');
    const chart = echarts.init(chartDom);

    // 生成每条折线的 checkbox
    DATASETS.forEach((ds, i) => {{
      const id = 'series-' + i;
      const label = document.createElement('label');
      label.htmlFor = id;
      label.innerHTML = `<input type="checkbox" id="${{id}}" data-index="${{i}}" checked /> <span style="color:${{COLORS[i] || '#888'}}">■</span> ${{ds.name}}`;
      checkboxesDiv.appendChild(label);
    }});

    document.getElementById('selectAll').addEventListener('click', () => {{
      checkboxesDiv.querySelectorAll('input').forEach(cb => cb.checked = true);
      updateChart();
    }});
    document.getElementById('selectNone').addEventListener('click', () => {{
      checkboxesDiv.querySelectorAll('input').forEach(cb => cb.checked = false);
      updateChart();
    }});

    function getSelectedIndices() {{
      return Array.from(checkboxesDiv.querySelectorAll('input'))
        .filter(cb => cb.checked)
        .map(cb => parseInt(cb.dataset.index, 10));
    }}

    function buildOption() {{
      const metric = metricSelect.value;
      const indices = getSelectedIndices();
      const metricTitle = {{ layer_r2: 'R²', layer_mae: 'MAE', layer_rmse: 'RMSE' }}[metric];

      const series = indices.map(i => {{
        const ds = DATASETS[i];
        const y = ds[metric];
        const pts = ds.n_layers ? Array.from({{ length: ds.n_layers }}, (_, j) => [j, y[j]]) : [];
        const s = {{
          name: ds.name,
          type: 'line',
          data: pts,
          smooth: true,
          symbol: 'circle',
          symbolSize: 6,
          lineStyle: {{ width: 2 }},
          itemStyle: {{ color: COLORS[i] || '#888' }},
        }};
        if (metric === 'layer_r2' && ds.best_layer != null && y[ds.best_layer] != null) {{
          s.markPoint = {{
            data: [{{ value: y[ds.best_layer], coord: [ds.best_layer, y[ds.best_layer]], name: 'best' }}],
            symbol: 'pin',
            symbolSize: 40,
            itemStyle: {{ color: COLORS[i] || '#888' }},
            label: {{ show: true, formatter: 'best' }},
          }};
        }}
        return s;
      }});

      return {{
        title: {{ text: metricTitle + ' by Layer (task_family_3)', left: 'center', textStyle: {{ color: '#e4e4e7' }} }},
        tooltip: {{ trigger: 'axis' }},
        legend: {{ type: 'scroll', bottom: 0, textStyle: {{ color: '#a1a1aa' }} }},
        grid: {{ left: 60, right: 24, top: 40, bottom: 80 }},
        xAxis: {{ type: 'value', name: 'Layer', nameLocation: 'middle', nameGap: 28, axisLine: {{ lineStyle: {{ color: '#52525b' }} }}, axisLabel: {{ color: '#a1a1aa' }} }},
        yAxis: {{ type: 'value', name: metricTitle, nameTextStyle: {{ color: '#a1a1aa' }}, splitLine: {{ lineStyle: {{ color: '#3f3f46' }} }}, axisLine: {{ lineStyle: {{ color: '#52525b' }} }}, axisLabel: {{ color: '#a1a1aa' }} }},
        series: series,
      }};
    }}

    function updateChart() {{
      chart.setOption(buildOption(), {{ notMerge: true }});
    }}

    checkboxesDiv.querySelectorAll('input').forEach(cb => {{
      cb.addEventListener('change', updateChart);
    }});
    metricSelect.addEventListener('change', updateChart);

    window.addEventListener('resize', () => chart.resize());
    updateChart();
  </script>
</body>
</html>
"""

    out_path = OUT_DIR / "layer_metrics_interactive.html"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    print("已生成:", out_path)
    print("用浏览器打开该文件即可选择显示哪几条折线。")


if __name__ == "__main__":
    main()
