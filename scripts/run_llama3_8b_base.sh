#!/bin/bash
# ============================================================================
# Master Experiment Script for Llama3-8B-Base
# ============================================================================
# Usage:
#   bash scripts/run_llama3_8b_base.sh [PHASE]
#
# Phases:
#   all      - Run everything (default)
#   1        - Inference + Accuracy Analysis only
#   2        - Linear Probes only
#   3        - SAE Pipeline (train, analyze, intervene, visualize)
#   4        - Additional Experiments (dose-response, ablation, etc.)
#   5        - Generate Reports (LaTeX tables, comparison plots)
#
# Examples:
#   bash scripts/run_llama3_8b_base.sh          # Run all phases
#   bash scripts/run_llama3_8b_base.sh 1        # Run only inference
#   bash scripts/run_llama3_8b_base.sh 2 3      # Run probe + SAE
#
# Prerequisites:
#   - GPU with CUDA (at least 24GB VRAM for 8B models)
#   - An OpenAI-compatible API at $INFERENCE_API_BASE (default http://127.0.0.1:8010/v1)
#   - conda environment with torch, transformers, transformer_lens, sae_lens
# ============================================================================

set -euo pipefail

# ── Configuration ───────────────────────────────────────────────────────────
# Model name for local loading (config.py key — used by probe, SAE scripts)
MODEL_LOCAL="meta-llama/Meta-Llama-3-8B"
# Model name for API inference (must match what the API server expects)
MODEL_API="Llama3-8B-Base"
MODEL_DISPLAY="Llama3-8B-Base"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export INFERENCE_API_BASE="${INFERENCE_API_BASE:-http://127.0.0.1:8010/v1}"
SRC_DIR="${REPO_ROOT}/src"

# Data directories
DATA_TF1="${SRC_DIR}/data_generation/task_family_1/data_relation"
DATA_TF2="${SRC_DIR}/data_generation/task_family_2/data_orientation"
DATA_TF3="${SRC_DIR}/data_generation/task_family_3/data_procedure"

# Probe script directories
PROBE_TF1="${SRC_DIR}/probe/task_family_1"
PROBE_TF2="${SRC_DIR}/probe/task_family_2"
PROBE_TF3="${SRC_DIR}/probe/task_family_3"

# SAE script directories
SAE_TF1="${SRC_DIR}/sae_relation"
SAE_TF2="${SRC_DIR}/sae_orientation"
SAE_TF3="${SRC_DIR}/sae_procedure"

# Output root for all results
RESULTS_DIR="${REPO_ROOT}/results_${MODEL_DISPLAY}"
mkdir -p "${RESULTS_DIR}"

# SAE hyperparameters (matching existing experiments)
SAE_LAYER=20
SAE_EXPANSION_FACTOR=32
SAE_BATCH_SIZE=1024
SAE_NUM_TOKENS=500000
SAE_L1_COEF=0.0001
SAE_TOP_K=50
SAE_NUM_TEST_SAMPLES=100
SAE_INTERVENTION_MAG=2.0

# Languages
LANGS=("EN" "CN" "AR")
LANG_NAMES=("English" "Chinese" "Arabic")

# Timestamp for this run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${RESULTS_DIR}/experiment_${TIMESTAMP}.log"

# ── Helper Functions ────────────────────────────────────────────────────────
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG_FILE}"
}

run_step() {
    local desc="$1"; shift
    log "▶ ${desc}"
    log "  CMD: $*"
    if "$@" >> "${LOG_FILE}" 2>&1; then
        log "✓ ${desc} — OK"
        return 0
    else
        log "✗ ${desc} — FAILED (exit code $?)"
        return 1
    fi
}

# ────────────────────────────────────────────────────────────────────────────
# Phase 1: Inference + Accuracy Analysis
# ────────────────────────────────────────────────────────────────────────────
phase1() {
    log "============================================"
    log "PHASE 1: Inference & Accuracy Analysis"
    log "============================================"

    # Task Family 1: Relation
    for lang in "${LANGS[@]}"; do
        log "TF1 Inference: ${lang}"
        cd "${SRC_DIR}/data_generation/task_family_1"
        run_step "TF1 ${lang} inference" \
            python call_inference_api_relation.py --language "${lang}" --model "${MODEL_API}"

        # Copy API results to data directory for analyze scripts
        cp "./results/api_results_${MODEL_API}_${lang}_test.json" \
           "./data_relation/api_results_${lang}_test.json"

        log "TF1 Accuracy Analysis: ${lang}"
        run_step "TF1 ${lang} accuracy" \
            python analyze_results_relation_${lang}.py
    done

    # Task Family 2: Orientation
    for lang in "${LANGS[@]}"; do
        log "TF2 Inference: ${lang}"
        cd "${SRC_DIR}/data_generation/task_family_2"
        run_step "TF2 ${lang} inference" \
            python call_inference_api_relation.py --language "${lang}" --model "${MODEL_API}"

        # Copy API results to data directory for analyze scripts
        cp "./results/api_results_${MODEL_API}_${lang}_test.json" \
           "./data_orientation/api_results_${lang}_test.json"

        log "TF2 Accuracy Analysis: ${lang}"
        run_step "TF2 ${lang} accuracy" \
            python analyze_results_orientation_${lang}.py
    done

    # Task Family 3: Procedure
    for lang in "${LANGS[@]}"; do
        log "TF3 Inference: ${lang}"
        cd "${SRC_DIR}/data_generation/task_family_3"
        run_step "TF3 ${lang} inference" \
            python call_inference_api_relation.py --language "${lang}" --model "${MODEL_API}"

        # Copy API results to data directory for analyze scripts
        cp "./results/api_results_${MODEL_API}_${lang}_test.json" \
           "./data_procedure/api_results_${lang}_test.json"

        log "TF3 Accuracy Analysis: ${lang}"
        run_step "TF3 ${lang} accuracy" \
            python analyze_results_procedure_${lang}.py
    done
}

# ────────────────────────────────────────────────────────────────────────────
# Phase 2: Linear Probes
# ────────────────────────────────────────────────────────────────────────────
phase2() {
    log "============================================"
    log "PHASE 2: Linear Probe Experiments"
    log "============================================"

    # Task Family 1: Relation
    PROBE_OUT_TF1="${RESULTS_DIR}/probe/task_family_1"
    mkdir -p "${PROBE_OUT_TF1}"
    cd "${PROBE_TF1}"

    for i in "${!LANGS[@]}"; do
        lang="${LANGS[$i]}"
        lang_name="${LANG_NAMES[$i]}"
        train_file="${DATA_TF1}/spatial_reasoning_dataset_${lang}_with_prompt.json"
        test_file="${DATA_TF1}/spatial_reasoning_dataset_${lang}_test_with_prompt.json"
        output_file="${PROBE_OUT_TF1}/probe_results_relation_${MODEL_DISPLAY}_${lang}.json"

        run_step "TF1 Probe ${lang_name}" \
            python layer_sweep_probe_relation.py \
                -m "${MODEL_LOCAL}" \
                -tr "${train_file}" \
                -te "${test_file}" \
                -o "${output_file}"
    done

    # Task Family 2: Orientation
    PROBE_OUT_TF2="${RESULTS_DIR}/probe/task_family_2"
    mkdir -p "${PROBE_OUT_TF2}"
    cd "${PROBE_TF2}"

    for i in "${!LANGS[@]}"; do
        lang="${LANGS[$i]}"
        lang_name="${LANG_NAMES[$i]}"
        train_file="${DATA_TF2}/orientation_reasoning_dataset_${lang}_with_prompt.json"
        test_file="${DATA_TF2}/orientation_reasoning_dataset_${lang}_test_with_prompt.json"
        output_file="${PROBE_OUT_TF2}/probe_results_orientation_${MODEL_DISPLAY}_${lang}.json"

        run_step "TF2 Probe ${lang_name}" \
            python layer_sweep_probe_orientation.py \
                -m "${MODEL_LOCAL}" \
                -tr "${train_file}" \
                -te "${test_file}" \
                -o "${output_file}"
    done

    # Task Family 3: Procedure
    PROBE_OUT_TF3="${RESULTS_DIR}/probe/task_family_3"
    mkdir -p "${PROBE_OUT_TF3}"
    cd "${PROBE_TF3}"

    for i in "${!LANGS[@]}"; do
        lang="${LANGS[$i]}"
        lang_name="${LANG_NAMES[$i]}"
        train_file="${DATA_TF3}/spatial_procedure_dataset_${lang}_with_prompt.json"
        test_file="${DATA_TF3}/spatial_procedure_dataset_${lang}_test_with_prompt.json"
        output_file="${PROBE_OUT_TF3}/probe_results_procedure_${MODEL_DISPLAY}_${lang}.json"

        run_step "TF3 Probe ${lang_name}" \
            python layer_sweep_probe_procedure_fixed.py \
                -m "${MODEL_LOCAL}" \
                -tr "${train_file}" \
                -te "${test_file}" \
                -o "${output_file}"
    done
}

# ────────────────────────────────────────────────────────────────────────────
# Phase 3: SAE Pipeline (per task family, per language)
# ────────────────────────────────────────────────────────────────────────────
phase3() {
    log "============================================"
    log "PHASE 3: SAE Pipeline (Train → Analyze → Attribute → Intervene → Visualize)"
    log "============================================"

    # ── Task Family 1: Relation ──
    SAE_OUT_TF1="${RESULTS_DIR}/sae/task_family_1"
    cd "${SAE_TF1}"

    for i in "${!LANGS[@]}"; do
        lang="${LANGS[$i]}"
        lang_name="${LANG_NAMES[$i]}"
        train_file="${DATA_TF1}/spatial_reasoning_dataset_${lang}_with_prompt.json"
        test_file="${DATA_TF1}/spatial_reasoning_dataset_${lang}_test_with_prompt.json"

        TF1_LANG_DIR="${SAE_OUT_TF1}/${lang}"
        mkdir -p "${TF1_LANG_DIR}"/{checkpoints,analysis,gradient,intervention,visualization}

        # Train SAE
        run_step "TF1 SAE Train ${lang_name}" \
            python train_sae_relation.py \
                --model_name "${MODEL_LOCAL}" \
                --train_data_file "${train_file}" \
                --layer ${SAE_LAYER} \
                --expansion_factor ${SAE_EXPANSION_FACTOR} \
                --batch_size ${SAE_BATCH_SIZE} \
                --num_tokens ${SAE_NUM_TOKENS} \
                --l1_coefficient ${SAE_L1_COEF} \
                --output_dir "${TF1_LANG_DIR}/checkpoints"

        SAE_PATH=$(find "${TF1_LANG_DIR}/checkpoints" -name "sae_final.pt" | head -n 1)

        # Analyze SAE features
        run_step "TF1 SAE Analyze ${lang_name}" \
            python analyze_sae_features.py \
                --model_name "${MODEL_LOCAL}" \
                --sae_path "${SAE_PATH}" \
                --train_data_file "${train_file}" \
                --test_data_file "${test_file}" \
                --output_dir "${TF1_LANG_DIR}/analysis" \
                --top_k ${SAE_TOP_K}

        ANALYSIS_JSON="${TF1_LANG_DIR}/analysis/sae_analysis_layer${SAE_LAYER}.json"

        # Gradient attribution
        run_step "TF1 SAE Gradient ${lang_name}" \
            python gradient_attribution_sae.py \
                --model_name "${MODEL_LOCAL}" \
                --sae_path "${SAE_PATH}" \
                --test_data_file "${test_file}" \
                --output_dir "${TF1_LANG_DIR}/gradient" \
                --num_samples ${SAE_NUM_TEST_SAMPLES} \
                --top_k ${SAE_TOP_K} || true

        # Intervention
        run_step "TF1 SAE Intervene ${lang_name}" \
            python intervene_sae_features.py \
                --model_name "${MODEL_LOCAL}" \
                --sae_path "${SAE_PATH}" \
                --test_data_file "${test_file}" \
                --analysis_results "${ANALYSIS_JSON}" \
                --output_dir "${TF1_LANG_DIR}/intervention" \
                --num_samples ${SAE_NUM_TEST_SAMPLES} \
                --intervention_magnitude ${SAE_INTERVENTION_MAG}

        INTERVENTION_JSON="${TF1_LANG_DIR}/intervention/intervention_results_layer${SAE_LAYER}.json"

        # Visualization
        run_step "TF1 SAE Visualize ${lang_name}" \
            python visualize_sae_results.py \
                --analysis_results "${ANALYSIS_JSON}" \
                --intervention_results "${INTERVENTION_JSON}" \
                --output_dir "${TF1_LANG_DIR}/visualization"
    done

    # Cross-language comparison (TF1 only)
    log "TF1 Cross-Language Comparison..."
    cd "${SAE_TF1}"
    run_step "TF1 Cross-Language Compare" \
        python compare_cross_language.py || true
    run_step "TF1 Feature Semantics" \
        python analyze_feature_semantics.py || true

    # ── Task Family 2: Orientation ──
    SAE_OUT_TF2="${RESULTS_DIR}/sae/task_family_2"
    cd "${SAE_TF2}"

    for i in "${!LANGS[@]}"; do
        lang="${LANGS[$i]}"
        lang_name="${LANG_NAMES[$i]}"
        train_file="${DATA_TF2}/orientation_reasoning_dataset_${lang}_with_prompt.json"
        test_file="${DATA_TF2}/orientation_reasoning_dataset_${lang}_test_with_prompt.json"

        TF2_LANG_DIR="${SAE_OUT_TF2}/${lang}"
        mkdir -p "${TF2_LANG_DIR}"/{checkpoints,analysis,gradient,intervention,visualization}

        run_step "TF2 SAE Train ${lang_name}" \
            python train_sae_relation.py \
                --model_name "${MODEL_LOCAL}" \
                --train_data_file "${train_file}" \
                --layer ${SAE_LAYER} \
                --expansion_factor ${SAE_EXPANSION_FACTOR} \
                --batch_size ${SAE_BATCH_SIZE} \
                --num_tokens ${SAE_NUM_TOKENS} \
                --l1_coefficient ${SAE_L1_COEF} \
                --output_dir "${TF2_LANG_DIR}/checkpoints"

        SAE_PATH=$(find "${TF2_LANG_DIR}/checkpoints" -name "sae_final.pt" | head -n 1)

        run_step "TF2 SAE Analyze ${lang_name}" \
            python analyze_sae_features.py \
                --model_name "${MODEL_LOCAL}" \
                --sae_path "${SAE_PATH}" \
                --train_data_file "${train_file}" \
                --test_data_file "${test_file}" \
                --output_dir "${TF2_LANG_DIR}/analysis" \
                --top_k ${SAE_TOP_K}

        ANALYSIS_JSON="${TF2_LANG_DIR}/analysis/sae_analysis_layer${SAE_LAYER}.json"

        run_step "TF2 SAE Gradient ${lang_name}" \
            python gradient_attribution_sae.py \
                --model_name "${MODEL_LOCAL}" \
                --sae_path "${SAE_PATH}" \
                --test_data_file "${test_file}" \
                --output_dir "${TF2_LANG_DIR}/gradient" \
                --num_samples ${SAE_NUM_TEST_SAMPLES} \
                --top_k ${SAE_TOP_K} || true

        run_step "TF2 SAE Intervene ${lang_name}" \
            python intervene_sae_features.py \
                --model_name "${MODEL_LOCAL}" \
                --sae_path "${SAE_PATH}" \
                --test_data_file "${test_file}" \
                --analysis_results "${ANALYSIS_JSON}" \
                --output_dir "${TF2_LANG_DIR}/intervention" \
                --num_samples ${SAE_NUM_TEST_SAMPLES} \
                --intervention_magnitude ${SAE_INTERVENTION_MAG}

        INTERVENTION_JSON="${TF2_LANG_DIR}/intervention/intervention_results_layer${SAE_LAYER}.json"

        run_step "TF2 SAE Visualize ${lang_name}" \
            python visualize_sae_results.py \
                --analysis_results "${ANALYSIS_JSON}" \
                --intervention_results "${INTERVENTION_JSON}" \
                --output_dir "${TF2_LANG_DIR}/visualization"
    done

    # ── Task Family 3: Procedure ──
    SAE_OUT_TF3="${RESULTS_DIR}/sae/task_family_3"
    cd "${SAE_TF3}"

    for i in "${!LANGS[@]}"; do
        lang="${LANGS[$i]}"
        lang_name="${LANG_NAMES[$i]}"
        train_file="${DATA_TF3}/spatial_procedure_dataset_${lang}_with_prompt.json"
        test_file="${DATA_TF3}/spatial_procedure_dataset_${lang}_test_with_prompt.json"

        TF3_LANG_DIR="${SAE_OUT_TF3}/${lang}"
        mkdir -p "${TF3_LANG_DIR}"/{checkpoints,analysis,gradient,intervention,visualization}

        run_step "TF3 SAE Train ${lang_name}" \
            python train_sae_procedure.py \
                --model_name "${MODEL_LOCAL}" \
                --train_data_file "${train_file}" \
                --layer ${SAE_LAYER} \
                --expansion_factor ${SAE_EXPANSION_FACTOR} \
                --batch_size ${SAE_BATCH_SIZE} \
                --num_tokens ${SAE_NUM_TOKENS} \
                --l1_coefficient ${SAE_L1_COEF} \
                --output_dir "${TF3_LANG_DIR}/checkpoints"

        SAE_PATH=$(find "${TF3_LANG_DIR}/checkpoints" -name "sae_final.pt" | head -n 1)

        run_step "TF3 SAE Analyze ${lang_name}" \
            python analyze_sae_features.py \
                --model_name "${MODEL_LOCAL}" \
                --sae_path "${SAE_PATH}" \
                --train_data_file "${train_file}" \
                --test_data_file "${test_file}" \
                --output_dir "${TF3_LANG_DIR}/analysis" \
                --top_k ${SAE_TOP_K}

        ANALYSIS_JSON="${TF3_LANG_DIR}/analysis/sae_analysis_layer${SAE_LAYER}.json"

        run_step "TF3 SAE Gradient ${lang_name}" \
            python gradient_attribution_sae.py \
                --model_name "${MODEL_LOCAL}" \
                --sae_path "${SAE_PATH}" \
                --test_data_file "${test_file}" \
                --output_dir "${TF3_LANG_DIR}/gradient" \
                --num_samples ${SAE_NUM_TEST_SAMPLES} \
                --top_k ${SAE_TOP_K} || true

        run_step "TF3 SAE Intervene ${lang_name}" \
            python intervene_sae_features.py \
                --model_name "${MODEL_LOCAL}" \
                --sae_path "${SAE_PATH}" \
                --test_data_file "${test_file}" \
                --analysis_results "${ANALYSIS_JSON}" \
                --output_dir "${TF3_LANG_DIR}/intervention" \
                --num_samples ${SAE_NUM_TEST_SAMPLES} \
                --intervention_magnitude ${SAE_INTERVENTION_MAG}

        INTERVENTION_JSON="${TF3_LANG_DIR}/intervention/intervention_results_layer${SAE_LAYER}.json"

        run_step "TF3 SAE Visualize ${lang_name}" \
            python visualize_sae_results.py \
                --analysis_results "${ANALYSIS_JSON}" \
                --intervention_results "${INTERVENTION_JSON}" \
                --output_dir "${TF3_LANG_DIR}/visualization"
    done
}

# ────────────────────────────────────────────────────────────────────────────
# Phase 4: Additional Experiments (Ablation, Dose-Response, etc.)
# ────────────────────────────────────────────────────────────────────────────
phase4() {
    log "============================================"
    log "PHASE 4: Additional Experiments"
    log "============================================"

    ADD_EXM_DIR="${SRC_DIR}/supplementary"
    ADD_OUT_DIR="${RESULTS_DIR}/additional_experiments"
    mkdir -p "${ADD_OUT_DIR}"
    cd "${ADD_EXM_DIR}"

    # For additional experiments, use English data from TF1 as representative
    EVAL_DATA="${DATA_TF1}/spatial_reasoning_dataset_EN_test_with_prompt.json"

    # Find the SAE checkpoint from TF1 English
    SAE_CKPT_DIR="${RESULTS_DIR}/sae/task_family_1/EN/checkpoints"
    SAE_PATH=$(find "${SAE_CKPT_DIR}" -name "sae_final.pt" 2>/dev/null | head -n 1)

    if [ -n "${SAE_PATH}" ]; then
        # Dose-Response Experiment
        run_step "Dose-Response Experiment" \
            python dose_response_experiment.py \
                --model_name "${MODEL_LOCAL}" \
                --sae_checkpoint "$(dirname ${SAE_PATH})" \
                --eval_data_file "${EVAL_DATA}" \
                --output_dir "${ADD_OUT_DIR}/dose_response" || true

        # Ablation Robustness Experiment
        run_step "Ablation Robustness Experiment" \
            python ablation_robustness_experiment.py \
                --model_name "${MODEL_LOCAL}" \
                --sae_checkpoint "$(dirname ${SAE_PATH})" \
                --eval_data_file "${EVAL_DATA}" \
                --output_dir "${ADD_OUT_DIR}/ablation_robustness" || true

        # Layer Localization Experiment
        run_step "Layer Localization Experiment" \
            python layer_localization_experiment.py \
                --model_name "${MODEL_LOCAL}" \
                --sae_checkpoint "$(dirname ${SAE_PATH})" \
                --eval_data_file "${EVAL_DATA}" \
                --output_dir "${ADD_OUT_DIR}/layer_localization" || true

        # Matched Random Control Experiment
        run_step "Matched Random Control" \
            python matched_random_control_experiment.py \
                --model_name "${MODEL_LOCAL}" \
                --sae_checkpoint "$(dirname ${SAE_PATH})" \
                --eval_data_file "${EVAL_DATA}" \
                --output_dir "${ADD_OUT_DIR}/random_control" || true

        # Statistical Significance Experiment
        run_step "Statistical Significance" \
            python statistical_significance_experiment.py \
                --model_name "${MODEL_LOCAL}" \
                --sae_checkpoint "$(dirname ${SAE_PATH})" \
                --eval_data_file "${EVAL_DATA}" \
                --output_dir "${ADD_OUT_DIR}/statistical" || true

        # Keep Spatial Only Experiment
        run_step "Keep Spatial Only" \
            python keep_spatial_only_experiment.py \
                --model_name "${MODEL_LOCAL}" \
                --sae_checkpoint "$(dirname ${SAE_PATH})" \
                --eval_data_file "${EVAL_DATA}" \
                --output_dir "${ADD_OUT_DIR}/keep_spatial" || true
    else
        log "WARNING: No SAE checkpoint found at ${SAE_CKPT_DIR} — skipping additional experiments"
    fi
}

# ────────────────────────────────────────────────────────────────────────────
# Phase 5: Generate Reports
# ────────────────────────────────────────────────────────────────────────────
phase5() {
    log "============================================"
    log "PHASE 5: Generate Reports & Tables"
    log "============================================"

    # ── Update plot_results.py with actual probe result filenames ──
    log "Updating plot_results.py with actual probe filenames..."

    # Task Family 1
    python -c "
import json, sys
sys.path.insert(0, '${SRC_DIR}')
from pathlib import Path

# Update plot_results.py for each task family
for fam_id, fam_name in [('task_family_1', 'relation'),
                          ('task_family_2', 'orientation'),
                          ('task_family_3', 'procedure')]:
    plot_py = Path('${SRC_DIR}/probe') / fam_id / 'plot_results.py'
    content = plot_py.read_text()
    # Replace placeholder filenames with actual ones from results
    for lang in ['EN', 'CN', 'AR']:
        old = f'probe_results_{fam_name}_{MODEL_DISPLAY}_{lang}.json'
        # Keep the same name pattern — user must copy results to probe dirs
    print(f'  Checked {plot_py}')
print('Done. Copy probe result JSONs to their respective probe/task_family_X/ directories.')
" 2>&1 | tee -a "${LOG_FILE}"

    # ── Generate LaTeX Tables ──
    log "Generating LaTeX accuracy big table..."
    cd "${SRC_DIR}"
    python -c "
import sys
sys.path.insert(0, 'src')
from data_generation.generate_accuracy_big_table import build_table
tex = build_table(include_details=True)
out = Path('${RESULTS_DIR}/accuracy_big_table.tex')
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(tex)
print(f'Wrote {out}')
" 2>&1 | tee -a "${LOG_FILE}"

    log "Generating LaTeX probe big table..."
    python -c "
import sys
sys.path.insert(0, 'src')
from probe.generate_probe_big_table import build_table
tex = build_table(include_errors=True)
out = Path('${RESULTS_DIR}/probe_big_table.tex')
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(tex)
print(f'Wrote {out}')
" 2>&1 | tee -a "${LOG_FILE}"

    # ── Generate Probe Visualizations ──
    log "Generating probe visualization plots..."
    for fam in task_family_1 task_family_2 task_family_3; do
        cd "${SRC_DIR}/probe/${fam}"
        python plot_results.py 2>&1 | tee -a "${LOG_FILE}" || true
    done
}

# ────────────────────────────────────────────────────────────────────────────
# Main Entry Point
# ────────────────────────────────────────────────────────────────────────────
main() {
    log "============================================"
    log "Llama3-8B-Base Full Experiment Pipeline"
    log "Model (local): ${MODEL_LOCAL}"
    log "Model (API):   ${MODEL_API}"
    log "Results: ${RESULTS_DIR}"
    log "Log: ${LOG_FILE}"
    log "Started at: $(date)"
    log "============================================"

    if [ $# -eq 0 ]; then
        PHASES="1 2 3 4 5"
    else
        PHASES="$*"
    fi

    for phase in ${PHASES}; do
        case ${phase} in
            all)
                phase1; phase2; phase3; phase4; phase5
                ;;
            1) phase1 ;;
            2) phase2 ;;
            3) phase3 ;;
            4) phase4 ;;
            5) phase5 ;;
            *)
                log "ERROR: Unknown phase '${phase}'. Use 1-5 or 'all'."
                exit 1
                ;;
        esac
    done

    log ""
    log "============================================"
    log "Pipeline Complete!"
    log "Results saved to: ${RESULTS_DIR}"
    log "Log file: ${LOG_FILE}"
    log "Finished at: $(date)"
    log "============================================"
}

main "$@"
