#!/bin/bash

set -euo pipefail
set -x

STAGE1_ROOT=dataset/ui_agile_grounding/data_process_1
STAGE2_ROOT=dataset/ui_agile_grounding/data_process_2
DATASET_NAME=ui_agile_grounding_qwen3vl

python -m recipe.gui_rlvr_baseline.data_process.data_process_1_format_ui_agile \
    --output-root "${STAGE1_ROOT}" \
    --val-ratio 0.0

python -m recipe.gui_rlvr_baseline.data_process.data_process_2_to_verl \
    --img-folder "${STAGE1_ROOT}/images" \
    --json-path "${STAGE1_ROOT}/train.json" \
    --output-path "${STAGE2_ROOT}/train.parquet" \
    --dataset-name "${DATASET_NAME}"

# python -m recipe.gui_rlvr_baseline.data_process.data_process_2_to_verl \
#     --img-folder "${STAGE1_ROOT}/images" \
#     --json-path "${STAGE1_ROOT}/val.json" \
#     --output-path "${STAGE2_ROOT}/val.parquet" \
#     --dataset-name "${DATASET_NAME}"

# ss-pro

# python -m recipe.gui_rlvr_baseline.data_process.data_process_1_format_screenspot_pro --val-ratio 1.0