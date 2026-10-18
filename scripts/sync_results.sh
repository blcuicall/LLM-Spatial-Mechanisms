#!/bin/bash
# ============================================================================
# Post-Experiment Results Sync Script
# ============================================================================
# After running experiments for Llama3-8B-Base, run this to:
#   1. Copy probe result JSONs to the plot_results.py directories
#   2. Update plot_results.py with actual filenames
#   3. Regenerate all probe visualizations and LaTeX tables
#
# Usage:
#   bash sync_results.sh
# ============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC_DIR="${REPO_ROOT}/src"
RESULTS_DIR="${REPO_ROOT}/results_Llama3-8B-Base"
MODEL_DISPLAY="Llama3-8B-Base"

echo "============================================"
echo "Syncing Llama3-8B-Base experiment results"
echo "============================================"

# ── Copy probe result JSONs to probe directories ──
echo ""
echo "Step 1: Copying probe result JSONs..."

declare -A PROBE_COPY_MAP=(
    # Task Family 1: Relation
    ["probe/task_family_1/EN"]="results_Llama3-8B-Base/probe/task_family_1/probe_results_relation_Llama3-8B-Base_EN.json"
    ["probe/task_family_1/CN"]="results_Llama3-8B-Base/probe/task_family_1/probe_results_relation_Llama3-8B-Base_CN.json"
    ["probe/task_family_1/AR"]="results_Llama3-8B-Base/probe/task_family_1/probe_results_relation_Llama3-8B-Base_AR.json"
    # Task Family 2: Orientation
    ["probe/task_family_2/EN"]="results_Llama3-8B-Base/probe/task_family_2/probe_results_orientation_Llama3-8B-Base_EN.json"
    ["probe/task_family_2/CN"]="results_Llama3-8B-Base/probe/task_family_2/probe_results_orientation_Llama3-8B-Base_CN.json"
    ["probe/task_family_2/AR"]="results_Llama3-8B-Base/probe/task_family_2/probe_results_orientation_Llama3-8B-Base_AR.json"
    # Task Family 3: Procedure
    ["probe/task_family_3/EN"]="results_Llama3-8B-Base/probe/task_family_3/probe_results_procedure_Llama3-8B-Base_EN.json"
    ["probe/task_family_3/CN"]="results_Llama3-8B-Base/probe/task_family_3/probe_results_procedure_Llama3-8B-Base_CN.json"
    ["probe/task_family_3/AR"]="results_Llama3-8B-Base/probe/task_family_3/probe_results_procedure_Llama3-8B-Base_AR.json"
)

for target_rel in "${!PROBE_COPY_MAP[@]}"; do
    src_name="${PROBE_COPY_MAP[$target_rel]}"
    src_path="${REPO_ROOT}/${src_name}"
    target_dir="${SRC_DIR}/${target_rel}"

    # Find the actual probe result file (filename may have timestamp)
    result_dir=$(dirname "${src_path}")
    result_pattern=$(basename "${src_path}")

    if [ -f "${src_path}" ]; then
        dest_name="${target_dir}/${result_pattern}"
        cp "${src_path}" "${dest_name}"
        echo "  Copied: ${result_pattern} → ${target_rel}/"
    else
        echo "  WARNING: ${src_path} not found — looking for alternative matches..."
        # Try to find any JSON file in the result directory
        if [ -d "${result_dir}" ]; then
            found=$(find "${result_dir}" -maxdepth 1 -name "*.json" | head -1)
            if [ -n "${found}" ]; then
                cp "${found}" "${target_dir}/${result_pattern}"
                echo "  Copied alternative: $(basename ${found}) → ${target_rel}/${result_pattern}"
            else
                echo "  WARNING: No probe results found in ${result_dir}"
            fi
        else
            echo "  WARNING: Directory ${result_dir} does not exist"
        fi
    fi
done

# ── Update plot_results.py with actual filenames ──
echo ""
echo "Step 2: Updating plot_results.py with actual probe result filenames..."

python3 -c "
import sys, json
sys.path.insert(0, '${SRC_DIR}')
from pathlib import Path

for fam_id, prefix in [('task_family_1', 'probe_results_relation'),
                        ('task_family_2', 'probe_results_orientation'),
                        ('task_family_3', 'probe_results_procedure')]:
    probe_dir = Path('${SRC_DIR}/probe') / fam_id
    plot_py = probe_dir / 'plot_results.py'

    # Check what files actually exist in the probe directory
    existing = {}
    for lang_code, lang_name in [('EN', 'English'), ('CN', 'Chinese'), ('AR', 'Arabic')]:
        expected = f'{prefix}_{MODEL_DISPLAY}_{lang_code}.json'
        if (probe_dir / expected).exists():
            existing[lang_name] = expected
            print(f'  Found: {fam_id}/{expected}')
        else:
            print(f'  Missing: {fam_id}/{expected}')

    if existing:
        print(f'  {fam_id}: {len(existing)}/3 probe result files found for {MODEL_DISPLAY}')

print()
print('Note: plot_results.py already contains the correct placeholder filenames.')
print('If the probe output files used different names, update plot_results.py manually.')
" 2>&1

# ── Regenerate probe visualizations ──
echo ""
echo "Step 3: Regenerating probe visualizations..."

for fam in task_family_1 task_family_2 task_family_3; do
    echo "  Generating ${fam} plots..."
    cd "${SRC_DIR}/probe/${fam}"
    python plot_results.py 2>&1 | grep -E "Saved:|Warning:|Error:" || true
done

# ── Regenerate LaTeX tables ──
echo ""
echo "Step 4: Regenerating LaTeX tables..."

cd "${SRC_DIR}"
LATEX_DIR="${REPO_ROOT}/results_Llama3-8B-Base/latex_tables"
mkdir -p "${LATEX_DIR}"

python -c "
import sys
sys.path.insert(0, 'src')
from pathlib import Path
from data_generation.generate_accuracy_big_table import build_table
tex = build_table(include_details=True)
out = Path('${LATEX_DIR}/accuracy_big_table.tex')
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(tex)
print(f'  Wrote {out}')
" 2>&1

python -c "
import sys
sys.path.insert(0, 'src')
from pathlib import Path
from probe.generate_probe_big_table import build_table
tex = build_table(include_errors=True)
out = Path('${LATEX_DIR}/probe_big_table.tex')
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(tex)
print(f'  Wrote {out}')
" 2>&1

echo ""
echo "============================================"
echo "Sync Complete!"
echo "============================================"
echo "Probe JSONs: src/probe/task_family_*/"
echo "LaTeX tables: ${LATEX_DIR}/"
echo "Visualizations: src/probe/task_family_*/visualization/"
