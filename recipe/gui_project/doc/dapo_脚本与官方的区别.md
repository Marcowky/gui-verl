  启动方式
  | 项目 | GUI 脚本 | 官方 DAPO 脚本 | 解释 |
  |---|---|---|---|
  | 启动命令 | python -m recipe.dapo.main_dapo | ray job submit --no-wait ... -- python3 -m recipe.dapo.main_dapo | GUI 版是本地前台启动；官方版是提交到 Ray 集群跑 |
  | 集群工作目录 | 无 | --working-dir "${WORKING_DIR}" | 官方版会把代码目录打包给集群 |
  | runtime env | 无显式 --runtime-env | --runtime-env="${RUNTIME_ENV}" | 官方版给集群 worker 注入统一环境 |
  | 节点数 | trainer.nnodes=1 | trainer.nnodes=16 | GUI 版单机；官方版多机 |
  | 每节点 GPU | trainer.n_gpus_per_node=4 | trainer.n_gpus_per_node=8 | 资源规模完全不同 |

  脚本级环境变量
  | 项目 | GUI 脚本 | 官方 DAPO 脚本 | 解释 |
  |---|---|---|---|
  | CUDA_VISIBLE_DEVICES | 0,1,2,3 | 无 | GUI 版限定本机 4 卡 |
  | NCCL 环境变量 | 显式设置多项 | 无 | GUI 版更像单机调优脚本 |
  | RAY_ADDRESS | 无 | 有 | 官方版面向已有 Ray 集群 |
  | WORKING_DIR | 无 | 有 | 官方版用于提交代码 |
  | RUNTIME_ENV | 无 | 有 | 官方版用于 Ray Job |
  | RAY_DATA_HOME | 无 | 有 | 官方版默认从集群共享目录找模型和数据 |

  数据配置
  | 配置 | GUI 脚本 | 官方 DAPO 脚本 | 解释 |
  |---|---|---|---|
  | data.train_files | dataset/ui_agile_grounding/.../train.parquet | .../data/dapo-math-17k.parquet | GUI grounding 数据 vs 数学数据 |
  | data.val_files | dataset/screenspot_pro_grounding/.../val_500.parquet | .../data/aime-2024.parquet | GUI 验证集 vs AIME |
  | data.train_batch_size | 128 | 512 | GUI 版更小，适配 4B VLM 和 4 卡 |
  | data.gen_batch_size | 192 | 1536 | 官方版更激进，先多采样再筛 |
  | data.val_batch_size | 128 | 未显式设置 | GUI 版单独控制验证 batch |
  | data.dataloader_num_workers | 0 | 未显式设置 | GUI 版偏保守，减少本机 dataloader 并发问题 |
  | data.max_prompt_length | 7168 | 2048 | GUI 多模态 prompt 很长 |
  | data.max_response_length | 1024 | 20480 | GUI 输出是点坐标，官方数学长 CoT 很长 |
  | data.filter_overlong_prompts | True | 未显式设置 | GUI 版先过滤超长样本 |
  | data.filter_overlong_prompts_workers | 32 | 未显式设置 | 配合上面过滤用 |
  | data.truncation | error | left | GUI 版不允许静默截断；官方版允许左截断 |
  | data.image_key | images | 无 | GUI 版明确是多模态数据 |
  | data.prompt_key | 未显式设置 | prompt | GUI 版依赖默认值，实际等价 |

  奖励配置
  | 配置 | GUI 脚本 | 官方 DAPO 脚本 | 解释 |
  |---|---|---|---|
  | custom_reward_function.path | recipe/gui_project/reward/in_box_reward.py | 无 | GUI 版用自定义 bbox reward |
  | custom_reward_function.name | compute_score | 无 | 同上 |
  | reward_model.reward_manager | dapo | dapo | 这一项相同 |
  | reward_model.overlong_buffer.enable | True | True | 相同 |
  | reward_model.overlong_buffer.len | 256 | 4096 | GUI 输出短，只需要很小 buffer |
  | reward_model.overlong_buffer.penalty_factor | 1.0 | 1.0 | 相同 |
  | algorithm.filter_groups.metric | score | acc | GUI reward 返回的是连续/半连续 score 更自然；官方数学直接按正确率筛 |

  DAPO / 算法配置
  | 配置 | GUI 脚本 | 官方 DAPO 脚本 | 解释 |
  |---|---|---|---|
  | algorithm.adv_estimator | grpo | grpo | 相同，DAPO 本来就是基于 GRPO advantage |
  | algorithm.use_kl_in_reward | False | False | 相同 |
  | algorithm.kl_ctrl.kl_coef | 0.0 | 0.0 | 相同 |
  | algorithm.filter_groups.enable | True | True | 相同 |
  | algorithm.filter_groups.max_num_gen_batches | 8 | 10 | GUI 版允许的补采轮数更少 |
  | actor_rollout_ref.actor.use_kl_loss | False | False | 相同 |
  | actor_rollout_ref.actor.kl_loss_coef | 0.0 | 0.0 | 相同 |
  | actor_rollout_ref.actor.clip_ratio_low | 0.2 | 0.2 | 相同 |
  | actor_rollout_ref.actor.clip_ratio_high | 0.28 | 0.28 | 相同 |
  | actor_rollout_ref.actor.clip_ratio_c | 10.0 | 10.0 | 相同 |
  | actor_rollout_ref.actor.loss_agg_mode | token-mean | token-mean | 相同 |

  Actor / 优化配置
  | 配置 | GUI 脚本 | 官方 DAPO 脚本 | 解释 |
  |---|---|---|---|
  | actor_rollout_ref.actor.optim.lr | 1e-6 | 1e-6 | 相同 |
  | actor_rollout_ref.actor.ppo_mini_batch_size | 128 | 32 | GUI 版 mini-batch 更大 |
  | actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu | 4 | 未显式设置 | GUI 版直接按每卡 micro batch 控制 |
  | actor_rollout_ref.actor.fsdp_config.param_offload | False | True | GUI 版关 offload 换速度；官方版开 offload 撑 32B |
  | actor_rollout_ref.actor.fsdp_config.optimizer_offload | False | True | 同上 |
  | actor_rollout_ref.actor.entropy_coeff | 0 | 0 | 相同 |
  | actor_rollout_ref.actor.grad_clip | 未显式设置 | 1.0 | 官方版更明确约束梯度 |
  | actor_rollout_ref.actor.optim.lr_warmup_steps | 无 | 10 | 官方版加 warmup |
  | actor_rollout_ref.actor.optim.weight_decay | 无 | 0.1 | 官方版显式 weight decay |
  | actor_rollout_ref.actor.use_dynamic_bsz | 无 | True | 官方版大模型更依赖动态 batch |
  | actor_rollout_ref.actor.ppo_max_token_len_per_gpu | 无 | 22528 | 官方版显式按 token 上限控显存 |
  | actor_rollout_ref.actor.ulysses_sequence_parallel_size | 无 | 8 | 官方版开序列并行 |
  | actor_rollout_ref.actor.fsdp_config.fsdp_size | 无 | -1 | 官方版显式全量 FSDP shard |

  Model 配置
  | 配置 | GUI 脚本 | 官方 DAPO 脚本 | 解释 |
  |---|---|---|---|
  | actor_rollout_ref.model.path | Qwen3-VL-4B-Instruct | Qwen2.5-32B | 模型家族和规模都不同 |
  | actor_rollout_ref.model.use_remove_padding | True | True | 相同 |
  | actor_rollout_ref.model.use_fused_kernels | True | 无 | GUI 版显式开 fused kernels |
  | actor_rollout_ref.model.enable_gradient_checkpointing | True | True | 相同 |

  Rollout / 推理配置
  | 配置 | GUI 脚本 | 官方 DAPO 脚本 | 解释 |
  |---|---|---|---|
  | actor_rollout_ref.rollout.name | vllm | vllm | 相同 |
  | actor_rollout_ref.rollout.n | 8 | 16 | 官方版每个 prompt 采样更多 response |
  | actor_rollout_ref.rollout.max_num_batched_tokens | 8192 | 22528 | 由 prompt/response 长度决定 |
  | actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu | 16 | 未显式设置 | GUI 版直接按 micro batch 控 |
  | actor_rollout_ref.rollout.tensor_model_parallel_size | 1 | 4 | 官方版 32B rollout 用 TP=4 |
  | actor_rollout_ref.rollout.gpu_memory_utilization | 0.6 | 0.80 | GUI 版更保守 |
  | actor_rollout_ref.rollout.temperature | 1.0 | 1.0 | 相同 |
  | actor_rollout_ref.rollout.top_p | 1.0 | 1.0 | 相同 |
  | actor_rollout_ref.rollout.top_k | -1 | -1 | 相同 |
  | +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache | True | 无 | GUI 多模态专用，避免 mm cache 问题 |
  | actor_rollout_ref.rollout.enable_chunked_prefill | 无 | True | 官方版长上下文吞吐优化 |
  | actor_rollout_ref.rollout.val_kwargs.temperature | 无 | 1.0 | 官方版单独定义验证采样 |
  | actor_rollout_ref.rollout.val_kwargs.top_p | 无 | 0.7 | 官方版验证更收敛 |
  | actor_rollout_ref.rollout.val_kwargs.top_k | 无 | -1 | 官方版验证采样设置 |
  | actor_rollout_ref.rollout.val_kwargs.do_sample | 无 | True | 官方版显式验证采样 |
  | actor_rollout_ref.rollout.val_kwargs.n | 无 | 1 | 官方版显式验证采样数 |
  | actor_rollout_ref.rollout.log_prob_use_dynamic_bsz | 无 | True | 官方版大模型显存优化 |

  Reference 配置
  | 配置 | GUI 脚本 | 官方 DAPO 脚本 | 解释 |
  |---|---|---|---|
  | actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu | 16 | 未显式设置 | GUI 版直接指定 |
  | actor_rollout_ref.ref.fsdp_config.param_offload | True | True | 相同 |
  | actor_rollout_ref.ref.log_prob_use_dynamic_bsz | 无 | True | 官方版显存优化 |
  | actor_rollout_ref.ref.log_prob_max_token_len_per_gpu | 无 | 22528 | 官方版显式 token 上限 |
  | actor_rollout_ref.ref.ulysses_sequence_parallel_size | 无 | 8 | 官方版序列并行 |

  Trainer 配置
  | 配置 | GUI 脚本 | 官方 DAPO 脚本 | 解释 |
  |---|---|---|---|
  | trainer.project_name | gui_rlvr_baseline | DAPO | 实验归类不同 |
  | trainer.experiment_name | 时间戳 + qwen3_vl_4b_ui_agile_dapo_vllm | DAPO-Qwen2.5-32B | 命名策略不同 |
  | trainer.default_local_dir | checkpoints/... 本地目录 | ${RAY_DATA_HOME}/ckpts/... | 本地实验目录 vs 集群默认目录 |
  | trainer.rollout_data_dir | 有 | 无 | GUI 版保存 rollout 数据 |
  | trainer.validation_data_dir | 有 | 无 | GUI 版保存验证样本 |
  | trainer.save_freq | 10 | 5 | GUI 版保存更稀疏 |
  | trainer.test_freq | 10 | 5 | GUI 版验证更稀疏 |
  | trainer.max_actor_ckpt_to_keep | 2 | 无 | GUI 版控制本地磁盘占用 |
  | trainer.best_ckpt_topk | 3 | 无 | GUI 版保留最优 checkpoint |
  | trainer.best_ckpt_metric | val-core/screenspot_pro_grounding/reward/mean@1 | 无 | GUI 版显式按 GUI 任务 reward 选最好模型 |
  | trainer.best_ckpt_mode | max | 无 | 配合上面 |
  | trainer.total_epochs | 6 | 1 | GUI 版跑更多 epoch |
  | trainer.resume_mode | 无 | auto | 官方版显式自动续训 |
  | trainer.val_before_train | 无显式设置 | True | 这项 GUI 版走默认值，实际也是 True，所以最终行为相同 |

  只在 GUI 脚本里有的配置

  - data.val_batch_size
  - data.dataloader_num_workers
  - data.filter_overlong_prompts
  - data.filter_overlong_prompts_workers
  - data.image_key
  - custom_reward_function.path
  - custom_reward_function.name
  - actor_rollout_ref.model.use_fused_kernels
  - actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu
  - actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu
  - +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache
  - actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu
  - trainer.rollout_data_dir
  - trainer.validation_data_dir
  - trainer.max_actor_ckpt_to_keep
  - trainer.best_ckpt_topk
  - trainer.best_ckpt_metric
  - trainer.best_ckpt_mode

  这些基本都和“单机 GUI 多模态任务调试/分析”有关。

  只在官方脚本里有的配置

  - actor_rollout_ref.actor.use_dynamic_bsz
  - actor_rollout_ref.ref.log_prob_use_dynamic_bsz
  - actor_rollout_ref.rollout.log_prob_use_dynamic_bsz
  - actor_rollout_ref.actor.ppo_max_token_len_per_gpu
  - actor_rollout_ref.ref.log_prob_max_token_len_per_gpu
  - actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu
  - actor_rollout_ref.actor.optim.lr_warmup_steps
  - actor_rollout_ref.actor.optim.weight_decay
  - actor_rollout_ref.actor.grad_clip
  - actor_rollout_ref.actor.ulysses_sequence_parallel_size
  - actor_rollout_ref.rollout.enable_chunked_prefill
  - actor_rollout_ref.rollout.val_kwargs.*
  - actor_rollout_ref.ref.ulysses_sequence_parallel_size
  - actor_rollout_ref.actor.fsdp_config.fsdp_size
  - trainer.resume_mode

  这些基本都和“32B 大模型、16 节点、长上下文、集群复现”有关。

  一句话总结：

  - GUI 脚本是“把官方 DAPO 目标函数迁到 GUI 多模态任务上”的单机 VLM 版本，配置重心是多模态兼容、日志分析和本地可跑。
  - 官方脚本是“Qwen2.5-32B 数学 DAPO 复现”版本，配置重心是大规模分布式吞吐、长上下文和显存优化。

  如果你要，我下一步可以把这些差异进一步整理成“哪些是任务必需差异，哪些只是规模/工程差异，哪些可以继续对齐官方”的三类清单。