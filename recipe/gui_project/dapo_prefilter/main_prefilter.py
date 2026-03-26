# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import socket
import uuid
from collections import defaultdict

import hydra
import numpy as np
import ray
import torch
from omegaconf import OmegaConf
from torch.utils.data import Dataset, SequentialSampler
from tqdm import tqdm

from verl import DataProto
from verl.trainer.main_ppo import create_rl_dataset
from verl.trainer.ppo.reward import compute_reward, load_reward_manager
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.utils import Role
from verl.utils.device import is_cuda_available


class IndexedDataset(Dataset):
    """Attach dataset row indices so filtered prompts can be exported back to parquet."""

    def __init__(self, base_dataset: Dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        item["dataset_row_idx"] = idx
        return item


class RayDAPOPrefilter(RayPPOTrainer):
    """Offline DAPO-style prompt filtering without actor updates."""

    @staticmethod
    def _should_keep_prompt_group(metric_name: str, metric_vals: list[float]) -> bool:
        """Keep all groups except the all-correct ones for score/acc-style metrics.

        For the GUI grounding use case we only want to remove prompts whose sampled responses are all correct.
        All-wrong groups are kept on purpose because they still provide useful hard examples.

        For other metrics, fall back to the original DAPO-style non-zero-variance rule.
        """
        if len(metric_vals) <= 1:
            return True

        if metric_name in {"score", "acc"}:
            return not np.allclose(metric_vals, 1.0)

        return float(np.std(metric_vals)) > 0

    def run_prefilter(self):
        output_dir = self.config.prefilter.output_dir
        os.makedirs(output_dir, exist_ok=True)

        metric_name = self.config.algorithm.filter_groups.metric
        filter_enabled = self.config.algorithm.filter_groups.enable

        kept_row_indices = []
        prompt_reports = []

        total_prompt_count = 0
        kept_prompt_count = 0
        dropped_prompt_count = 0

        progress_bar = tqdm(total=len(self.train_dataloader), desc="DAPO Prefilter")

        for batch_dict in self.train_dataloader:
            batch: DataProto = DataProto.from_single_dict(batch_dict)

            if "multi_modal_data" in batch.non_tensor_batch.keys():
                gen_batch = batch.pop(
                    batch_keys=["input_ids", "attention_mask", "position_ids"],
                    non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
                )
            else:
                gen_batch = batch.pop(
                    batch_keys=["input_ids", "attention_mask", "position_ids"],
                    non_tensor_batch_keys=["raw_prompt_ids"],
                )

            prompt_uids = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
            batch.non_tensor_batch["uid"] = prompt_uids

            prompt_meta_by_uid = {}
            dataset_row_idxs = batch.non_tensor_batch["dataset_row_idx"].tolist()
            data_sources = batch.non_tensor_batch.get("data_source", np.array([None] * len(prompt_uids), dtype=object))
            reward_models = batch.non_tensor_batch.get("reward_model", np.array([None] * len(prompt_uids), dtype=object))
            extra_infos = batch.non_tensor_batch.get("extra_info", np.array([None] * len(prompt_uids), dtype=object))
            for uid, row_idx, data_source, reward_model, extra_info in zip(
                prompt_uids, dataset_row_idxs, data_sources, reward_models, extra_infos, strict=True
            ):
                prompt_meta_by_uid[uid] = {
                    "dataset_row_idx": int(row_idx),
                    "data_source": data_source,
                    "ground_truth": reward_model.get("ground_truth") if isinstance(reward_model, dict) else None,
                    "extra_info": extra_info if isinstance(extra_info, dict) else {},
                }

            gen_batch_output = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)

            batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
            batch = batch.union(gen_batch_output)

            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)
            batch.batch["token_level_scores"] = reward_tensor
            batch.batch["token_level_rewards"] = reward_tensor
            if reward_extra_infos_dict:
                batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

            if filter_enabled:
                if metric_name == "seq_final_reward":
                    batch.non_tensor_batch["seq_final_reward"] = batch.batch["token_level_rewards"].sum(dim=-1).cpu().numpy()
                elif metric_name == "seq_reward":
                    batch.non_tensor_batch["seq_reward"] = batch.batch["token_level_scores"].sum(dim=-1).cpu().numpy()

                if metric_name not in batch.non_tensor_batch:
                    raise KeyError(
                        f"filter_groups.metric={metric_name} not found in batch.non_tensor_batch after reward computation"
                    )

                prompt_uid2metric_vals = defaultdict(list)
                for uid, metric_val in zip(
                    batch.non_tensor_batch["uid"], batch.non_tensor_batch[metric_name], strict=True
                ):
                    prompt_uid2metric_vals[uid].append(float(metric_val))

                kept_prompt_uids = {
                    uid
                    for uid, metric_vals in prompt_uid2metric_vals.items()
                    if self._should_keep_prompt_group(metric_name, metric_vals)
                }
                prompt_uid2metric_std = {
                    prompt_uid: float(np.std(metric_vals)) for prompt_uid, metric_vals in prompt_uid2metric_vals.items()
                }
            else:
                prompt_uid2metric_vals = defaultdict(list)
                prompt_uid2metric_std = {}
                kept_prompt_uids = set(prompt_uids.tolist())

            total_prompt_count += len(prompt_uids)
            kept_prompt_count += len(kept_prompt_uids)
            dropped_prompt_count += len(prompt_uids) - len(kept_prompt_uids)

            for uid in prompt_uids:
                meta = prompt_meta_by_uid[uid]
                kept = uid in kept_prompt_uids
                if kept:
                    kept_row_indices.append(meta["dataset_row_idx"])

                prompt_reports.append(
                    {
                        "dataset_row_idx": meta["dataset_row_idx"],
                        "kept": kept,
                        "metric": metric_name,
                        "metric_values": prompt_uid2metric_vals.get(uid, []),
                        "metric_std": prompt_uid2metric_std.get(uid),
                        "data_source": meta["data_source"],
                        "ground_truth": meta["ground_truth"],
                        "img_filename": meta["extra_info"].get("img_filename", ""),
                    }
                )

            progress_bar.update(1)

        progress_bar.close()
        self._save_prefilter_outputs(
            kept_row_indices=kept_row_indices,
            prompt_reports=prompt_reports,
            total_prompt_count=total_prompt_count,
            kept_prompt_count=kept_prompt_count,
            dropped_prompt_count=dropped_prompt_count,
        )

    def _save_prefilter_outputs(
        self,
        kept_row_indices: list[int],
        prompt_reports: list[dict],
        total_prompt_count: int,
        kept_prompt_count: int,
        dropped_prompt_count: int,
    ) -> None:
        output_dir = self.config.prefilter.output_dir
        kept_train_parquet = self.config.prefilter.kept_train_parquet
        prompt_report_jsonl = self.config.prefilter.prompt_report_jsonl
        summary_json = self.config.prefilter.summary_json

        os.makedirs(output_dir, exist_ok=True)

        unique_kept_row_indices = sorted(set(kept_row_indices))
        base_dataset = self.train_dataset.base_dataset if isinstance(self.train_dataset, IndexedDataset) else self.train_dataset
        filtered_dataset = base_dataset.dataframe.select(unique_kept_row_indices)
        filtered_dataset.to_parquet(kept_train_parquet)

        with open(prompt_report_jsonl, "w", encoding="utf-8") as f:
            for record in prompt_reports:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        summary = {
            "input_train_files": self.config.data.train_files,
            "kept_train_parquet": kept_train_parquet,
            "prompt_report_jsonl": prompt_report_jsonl,
            "total_prompt_count": total_prompt_count,
            "kept_prompt_count": kept_prompt_count,
            "dropped_prompt_count": dropped_prompt_count,
            "keep_ratio": (kept_prompt_count / total_prompt_count) if total_prompt_count > 0 else 0.0,
            "metric": self.config.algorithm.filter_groups.metric,
            "rollout_n": self.config.actor_rollout_ref.rollout.n,
            "unique_kept_row_count": len(unique_kept_row_indices),
        }
        with open(summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"Saved filtered parquet to {kept_train_parquet}")
        print(f"Saved prompt report to {prompt_report_jsonl}")
        print(f"Saved summary to {summary_json}")


@hydra.main(config_path=".", config_name="dapo_prefilter", version_base=None)
def main(config):
    run_prefilter(config)


def run_prefilter(config) -> None:
    if not ray.is_initialized():
        default_runtime_env = {
            "env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN", "VLLM_LOGGING_LEVEL": "WARN"}
        }
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    try:
        if (
            is_cuda_available
            and config.global_profiler.tool == "nsys"
            and OmegaConf.select(config.global_profiler, "steps") is not None
            and len(OmegaConf.select(config.global_profiler, "steps")) > 0
        ):
            nsight_options = OmegaConf.to_container(
                config.global_profiler.global_tool_config.nsys.controller_nsight_options
            )
            runner = TaskRunner.options(runtime_env={"nsight": nsight_options}).remote()
        else:
            runner = TaskRunner.remote()
        ray.get(runner.run.remote(config))
    finally:
        if ray.is_initialized():
            ray.shutdown()


@ray.remote(num_cpus=1)
class TaskRunner:
    def run(self, config):
        from pprint import pprint

        from verl.single_controller.ray import RayWorkerGroup
        from verl.utils import hf_processor, hf_tokenizer
        from verl.utils.fs import copy_to_local

        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        local_path = copy_to_local(config.actor_rollout_ref.model.path)
        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        if config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}:
            from verl.workers.fsdp_workers import ActorRolloutRefWorker

            ray_worker_group_cls = RayWorkerGroup
        elif config.actor_rollout_ref.actor.strategy == "megatron":
            from verl.workers.megatron_workers import ActorRolloutRefWorker

            ray_worker_group_cls = RayWorkerGroup
        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        }

        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
        }

        if config.reward_model.enable:
            if config.reward_model.strategy in {"fsdp", "fsdp2"}:
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        reward_fn = load_reward_manager(
            config,
            tokenizer,
            0,
            max_resp_len=config.data.max_response_length,
            overlong_buffer_cfg=config.reward_model.overlong_buffer,
        )

        train_dataset = create_rl_dataset(
            config.data.train_files,
            config.data,
            tokenizer,
            processor,
            is_train=True,
            max_samples=config.data.get("train_max_samples", -1),
        )
        indexed_train_dataset = IndexedDataset(train_dataset)
        train_sampler = SequentialSampler(indexed_train_dataset)

        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
        prefilterer = RayDAPOPrefilter(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=None,
            train_dataset=indexed_train_dataset,
            val_dataset=indexed_train_dataset,
            train_sampler=train_sampler,
        )
        prefilterer.init_workers()
        prefilterer.run_prefilter()


if __name__ == "__main__":
    main()
