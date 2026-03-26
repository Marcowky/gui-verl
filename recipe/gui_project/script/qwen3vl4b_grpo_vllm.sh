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

PROJECT_NAME=gui_rlvr_baseline
EXPERIMENT_NAME=${DATE}-qwen3_vl_4b_ui_agile

TRAIN_FILE=dataset/ui_agile_grounding/data_process_2/train.parquet
VAL_FILE=dataset/screenspot_pro_grounding/data_process_2/val_500.parquet

REWARD_FUNCTION_PATH=recipe/gui_project/reward/in_box_reward.py
REWARD_FUNCTION_NAME=compute_score

SAVE_CHECKPOINT_PATH=checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}
ROLL_OUT_DATA_DIR=${SAVE_CHECKPOINT_PATH}/rollout_data
VAL_DATA_DIR=${SAVE_CHECKPOINT_PATH}/validation_data

mkdir -p "${SAVE_CHECKPOINT_PATH}"
cp "$0" "${SAVE_CHECKPOINT_PATH}/run.sh"
cp "${REWARD_FUNCTION_PATH}" "${SAVE_CHECKPOINT_PATH}/reward.py"

PYTHONUNBUFFERED=1 python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.train_batch_size=128 \
    data.val_batch_size=128 \
    data.dataloader_num_workers=0 \
    data.max_prompt_length=7168 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=32 \
    data.truncation='error' \
    data.image_key=images \
    custom_reward_function.path="${REWARD_FUNCTION_PATH}" \
    custom_reward_function.name="${REWARD_FUNCTION_NAME}" \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.default_local_dir="${SAVE_CHECKPOINT_PATH}" \
    trainer.rollout_data_dir="${ROLL_OUT_DATA_DIR}" \
    trainer.validation_data_dir="${VAL_DATA_DIR}" \
    trainer.n_gpus_per_node="${GPU_NUMS}" \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=6 \
    | tee "${SAVE_CHECKPOINT_PATH}/train.log"
