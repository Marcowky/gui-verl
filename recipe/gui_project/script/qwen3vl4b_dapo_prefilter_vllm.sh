#!/bin/bash

set -xeuo pipefail

export NCCL_P2P_DISABLE=0
export NCCL_P2P_LEVEL=SYS
export NCCL_IB_DISABLE=1
export NCCL_PROTO=Simple
export NCCL_MIN_NCHANNELS=2
export NCCL_MAX_NCHANNELS=4
export CUDA_VISIBLE_DEVICES=0,1,2,3

DATE=$(date +%Y%m%d_%H%M%S)
GPU_NUMS=4
MODEL_PATH=/home/kaiyu/Model/Qwen/Qwen3-VL-4B-Instruct

PROJECT_NAME=gui_rlvr_dapo
EXPERIMENT_NAME=${DATE}-qwen3_vl_4b_ui_agile_dapo_prefilter_vllm

TRAIN_FILE=dataset/ui_agile_grounding/data_process_2/train.parquet
OUTPUT_DIR=outputs/dapo_prefilter/${PROJECT_NAME}/${EXPERIMENT_NAME}
FILTERED_TRAIN_FILE=${OUTPUT_DIR}/filtered_train.parquet
PROMPT_REPORT_JSONL=${OUTPUT_DIR}/prompt_report.jsonl
SUMMARY_JSON=${OUTPUT_DIR}/summary.json

REWARD_FUNCTION_PATH=recipe/gui_project/reward/in_box_reward.py
REWARD_FUNCTION_NAME=compute_score

mkdir -p "${OUTPUT_DIR}"
cp "$0" "${OUTPUT_DIR}/run.sh"
cp "${REWARD_FUNCTION_PATH}" "${OUTPUT_DIR}/reward.py"

PYTHONUNBUFFERED=1 python -m recipe.gui_project.dapo_prefilter.main_prefilter \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    algorithm.filter_groups.enable=True \
    algorithm.filter_groups.metric=score \
    algorithm.filter_groups.max_num_gen_batches=1 \
    data.train_files="${TRAIN_FILE}" \
    data.train_batch_size=128 \
    data.train_max_samples=-1 \
    data.gen_batch_size=128 \
    data.dataloader_num_workers=0 \
    data.max_prompt_length=7168 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=32 \
    data.truncation='error' \
    data.image_key=images \
    data.shuffle=False \
    custom_reward_function.path="${REWARD_FUNCTION_PATH}" \
    custom_reward_function.name="${REWARD_FUNCTION_NAME}" \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    reward_model.reward_manager=dapo \
    reward_model.overlong_buffer.enable=True \
    reward_model.overlong_buffer.len=256 \
    reward_model.overlong_buffer.penalty_factor=1.0 \
    trainer.logger='["console"]' \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node="${GPU_NUMS}" \
    trainer.nnodes=1 \
    prefilter.output_dir="${OUTPUT_DIR}" \
    prefilter.kept_train_parquet="${FILTERED_TRAIN_FILE}" \
    prefilter.prompt_report_jsonl="${PROMPT_REPORT_JSONL}" \
    prefilter.summary_json="${SUMMARY_JSON}" \
    | tee "${OUTPUT_DIR}/prefilter.log"
