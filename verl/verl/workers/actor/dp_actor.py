# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
"""
Single Process Actor
"""

import logging
import os
import re
import copy
from collections import defaultdict

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss, get_policy_loss_fn, kl_penalty
from verl.utils.device import get_device_name, is_cuda_available, is_npu_available
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor
if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input


__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOActor(BasePPOActor):
    def __init__(self, config, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"Actor use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"Actor use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  #  use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()

    def _forward_micro_batch(
        self, micro_batch, temperature, calculate_entropy=False, add_self_rewarding_loss=False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            if "image_bound" in micro_batch["multi_modal_inputs"][0]:  # minicpm-o logic
                for key in micro_batch["multi_modal_inputs"][0].keys():
                    multi_modal_inputs[key] = [inputs[key] for inputs in micro_batch["multi_modal_inputs"]]
            else:
                for key in micro_batch["multi_modal_inputs"][0].keys():
                    multi_modal_inputs[key] = torch.cat(
                        [inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0
                    )

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"] # (bsz, seqlen)
            responses = micro_batch["responses"] # (bsz, response_len)
            attention_mask = micro_batch["attention_mask"] # (bsz, seqlen)
            position_ids = micro_batch["position_ids"] # (bsz, seqlen)
            if "individual_rewards" in micro_batch.keys():
                individual_rewards = micro_batch["individual_rewards"] # (bsz,)
            else:
                individual_rewards = None
            if self.config.use_self_reward_loss and add_self_rewarding_loss:
                replaced_input_ids = copy.deepcopy(input_ids)
                replaced_responses = copy.deepcopy(responses)
                replaced_attention_mask = copy.deepcopy(attention_mask)
                # For each row in input_ids, find the first token from the end that is not pad_token_id,
                pad_token_id = self.config.pad_token_id # which is also the eos token, so we should carefully replace it
                replace_token_id = self.config.self_reward_token_id

                for i in range(len(replaced_input_ids)):
                    row = replaced_input_ids[i]
                    # if unfinished, continue
                    if row[-1] != pad_token_id:
                        continue
                    # Traverse from the end to the start
                    for idx in range(len(row) - 1, -1, -1):
                        if row[idx] != pad_token_id:
                            next_idx = idx + 2 # we do not replace the first eos token for now
                            if next_idx < len(row):
                                replaced_input_ids[i, next_idx] = replace_token_id        
                                replaced_attention_mask[i, next_idx] = 1
                                replaced_responses[i, next_idx - len(input_ids[i]) + len(responses[i])] = replace_token_id
                            break
                input_ids = replaced_input_ids
                micro_batch["input_ids"] = replaced_input_ids
                micro_batch["responses"] = replaced_responses
                micro_batch["attention_mask"] = replaced_attention_mask
            elif self.config.use_sft_loss and individual_rewards is not None and add_self_rewarding_loss:
                replaced_input_ids = copy.deepcopy(input_ids)
                replaced_responses = copy.deepcopy(responses)
                replaced_attention_mask = copy.deepcopy(attention_mask)
                # For each row in input_ids, find the first token from the end that is not pad_token_id,
                pad_token_id = self.config.pad_token_id # which is also the eos token, so we should carefully replace it
                replace_token_id_yes = self.config.self_reward_token_id
                replace_token_id_no = self.config.self_reward_token_id + 1 # may be specially designed
                for i in range(len(replaced_input_ids)):
                    row = replaced_input_ids[i]
                    # if unfinished, continue
                    if row[-1] != pad_token_id:
                        continue
                    # Traverse from the end to the start
                    for idx in range(len(row) - 1, -1, -1):
                        if row[idx] != pad_token_id:
                            next_idx = idx + 2 # we should not replace the first eos token!!!
                            if next_idx < len(row):
                                if individual_rewards[i] == 1:
                                    replace_token_id = replace_token_id_yes
                                else:
                                    replace_token_id = replace_token_id_no
                                replaced_input_ids[i, next_idx] = replace_token_id        
                                replaced_responses[i, next_idx - len(input_ids[i]) + len(responses[i])] = replace_token_id
                                replaced_attention_mask[i, next_idx] = 1
                            break
                input_ids = replaced_input_ids
                micro_batch["input_ids"] = replaced_input_ids
                micro_batch["responses"] = replaced_responses
                micro_batch["attention_mask"] = replaced_attention_mask

            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)
                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = "multi_modal_inputs" in micro_batch.keys()
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                        else:
                            entropy_rmpad = torch.utils.checkpoint.checkpoint(
                                self.compute_entropy_from_logits, logits_rmpad
                            )

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)

            return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                )
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)

        return log_probs, entropys

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        add_self_rewarding_loss = data.meta_info["add_self_rewarding_loss"]
        add_self_rewarding_advantages_integration = data.meta_info["add_self_rewarding_advantages_integration"]

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        if self.config.use_self_reward_loss and add_self_rewarding_loss:
            select_keys.append("token_level_rewards")
            # select_keys.append("group_mean")
            select_keys.append("individual_rewards")
        if (self.config.use_self_reward_loss and self.config.self_reward_loss_balanced and add_self_rewarding_loss) or (self.config.use_sft_loss and self.config.sft_loss_balanced and add_self_rewarding_loss):
            select_keys.append("reweighting_factor")
        if self.config.use_self_reward_loss and self.config.self_rewarding_advantages_integration_coef > 0.0 and add_self_rewarding_advantages_integration:
            select_keys.append("self_rewarding_advantages")
        if self.config.use_sft_loss and add_self_rewarding_loss:
            if "individual_rewards" not in select_keys:
                select_keys.append("individual_rewards")

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        metrics = {}
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]
                    # advantage integration with self-rewarding advantages
                    if self.config.use_self_reward_loss and self.config.self_rewarding_advantages_integration_coef > 0.0 and add_self_rewarding_advantages_integration:
                        self_rewarding_advantages = model_inputs["self_rewarding_advantages"]
                        advantages = (1.0 - self.config.self_rewarding_advantages_integration_coef) * advantages + self.config.self_rewarding_advantages_integration_coef * self_rewarding_advantages

                    clip_ratio = self.config.clip_ratio
                    clip_ratio_low = (
                        self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                    )
                    clip_ratio_high = (
                        self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                    )
                    clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True
                    entropy, log_prob = self._forward_micro_batch(
                        model_inputs, temperature=temperature, calculate_entropy=calculate_entropy, add_self_rewarding_loss=add_self_rewarding_loss
                    )

                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")

                    if self.config.policy_loss.loss_mode == "vanilla":
                        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=response_mask,
                            cliprange=clip_ratio,
                            cliprange_low=clip_ratio_low,
                            cliprange_high=clip_ratio_high,
                            clip_ratio_c=clip_ratio_c,
                            loss_agg_mode=loss_agg_mode,
                        )

                    else:
                        policy_loss_fn = get_policy_loss_fn(loss_mode)
                        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_fn(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=response_mask,
                            loss_agg_mode=loss_agg_mode,
                            config=self.config,
                        )

                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item()
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef
                    
                    # add self reward loss
                    if self.config.use_self_reward_loss and add_self_rewarding_loss:
                        # For each row in response_mask (shape [bsz, response_len]), which looks like [1,1,1,...,1,0,0,...,0],
                        # find the index of the first 0 (i.e., the first position where response_mask==0),
                        # and extract the corresponding log_prob value at that position.

                        # response_mask: (bsz, response_len), log_prob: (bsz, response_len)
                        # Find the first zero in each row
                        is_zero = (response_mask == 0)
                        # For each row, get the index of the first zero
                        # If there is no zero, argmax returns 0, but in that case is_zero.sum(dim=1)==0, so we can mask it if needed
                        special_token_idx = is_zero.float().argmax(dim=1)
                        
                        # Gather the corresponding log_prob value
                        bsz = log_prob.size(0)
                        log_prob_special_token = log_prob[torch.arange(bsz, device=log_prob.device), special_token_idx]
                        ref_log_prob_special_token = ref_log_prob[torch.arange(bsz, device=ref_log_prob.device), special_token_idx]
                        # print(f"log_prob_first_zero: {log_prob_first_zero}")
                        # Add mask_indicator: if the last element of response_mask[i] is 0, mask_indicator[i]=1, else mask_indicator[i]=0
                        # response_mask: (bsz, response_len)
                        last_element = response_mask[:, -1]  # shape (bsz,)
                        mask_indicator = (last_element == 0).long()
                        # print(f"mask_indicator: {mask_indicator}")
                        masked_log_prob_special_token = log_prob_special_token * temperature # important
                        masked_ref_log_prob_special_token = ref_log_prob_special_token * temperature # important
                        if self.config.ref_model_log_ratio_constant < 0.0:
                            internal_reward_ratio = (masked_log_prob_special_token - self.config.ref_model_log_ratio_constant) * self.config.self_reward_kl_coef
                        else:
                            internal_reward_ratio = (masked_log_prob_special_token - masked_ref_log_prob_special_token) * self.config.self_reward_kl_coef
                        # Only compute internal_reward_loss where mask_indicator == 1
                        if not self.config.self_reward_loss_balanced:
                            internal_reward_loss = (((internal_reward_ratio - model_inputs["individual_rewards"]) ** 2) * mask_indicator).mean()
                        else:
                            internal_reward_loss = (((internal_reward_ratio - model_inputs["individual_rewards"]) ** 2) * mask_indicator * model_inputs["reweighting_factor"]).mean()
                        policy_loss = policy_loss + internal_reward_loss * self.config.self_reward_loss_coef
                        with torch.no_grad():
                            micro_batch_metrics["actor/self_reward_loss"] = internal_reward_loss.detach().item()

                        # Calculate predicted reward based on internal_reward_ratio
                        pred_reward = (internal_reward_ratio.detach() > 0.5).long()
                        # Get ground truth reward
                        individual_rewards = model_inputs["individual_rewards"].long()
                        # Only consider entries where mask_indicator == 1
                        valid_mask = (mask_indicator == 1)
                        # Overall accuracy (only count where mask_indicator == 1)
                        correct = (pred_reward == individual_rewards) & valid_mask
                        total_correct = correct.sum().item()
                        # Number of valid solutions (where mask_indicator == 1)
                        valid_solution_total = valid_mask.sum().item()
                        # Number of valid solutions where individual_rewards == 1 and mask_indicator == 1
                        valid_correct_solution_total = ((individual_rewards == 1) & valid_mask).sum().item()
                        # Number of valid solutions where individual_rewards == 0 and mask_indicator == 1
                        valid_incorrect_solution_total = ((individual_rewards == 0) & valid_mask).sum().item()
                        # Number of correct predictions when individual_rewards == 1 and mask_indicator == 1
                        mask_correct = (individual_rewards == 1) & valid_mask
                        if mask_correct.any():
                            correct_correct = ((pred_reward[mask_correct] == 1).sum().item())
                        else:
                            correct_correct = 0
                        # Number of correct predictions when individual_rewards == 0 and mask_indicator == 1
                        mask_incorrect = (individual_rewards == 0) & valid_mask
                        if mask_incorrect.any():
                            correct_incorrect = ((pred_reward[mask_incorrect] == 0).sum().item())
                        else:
                            correct_incorrect = 0
                        micro_batch_metrics["actor/internal_reward_total_valid_solutions"] = valid_solution_total
                        micro_batch_metrics["actor/internal_reward_valid_solutions_total_correct"] = total_correct
                        micro_batch_metrics["actor/internal_reward_total_valid_correct_solutions"] = valid_correct_solution_total
                        micro_batch_metrics["actor/internal_reward_total_valid_incorrect_solutions"] = valid_incorrect_solution_total
                        micro_batch_metrics["actor/internal_reward_valid_correct_solutions_total_correct"] = correct_correct
                        micro_batch_metrics["actor/internal_reward_valid_incorrect_solutions_total_correct"] = correct_incorrect

                        micro_batch_metrics["actor/self_reward_loss_coef"] = self.config.self_reward_loss_coef
                        micro_batch_metrics["actor/self_reward_kl_coef"] = self.config.self_reward_kl_coef
                        
                    elif self.config.use_sft_loss and add_self_rewarding_loss:
                        # For each row in response_mask (shape [bsz, response_len]), which looks like [1,1,1,...,1,0,0,...,0],
                        # find the index of the first 0 (i.e., the first position where response_mask==0),
                        # and extract the corresponding log_prob value at that position.

                        # response_mask: (bsz, response_len), log_prob: (bsz, response_len)
                        # Find the first zero in each row
                        is_zero = (response_mask == 0)
                        # For each row, get the index of the first zero
                        # If there is no zero, argmax returns 0, but in that case is_zero.sum(dim=1)==0, so we can mask it if needed
                        special_token_idx = is_zero.float().argmax(dim=1)
                        # Gather the corresponding log_prob value
                        bsz = log_prob.size(0)
                        log_prob_special_token = log_prob[torch.arange(bsz, device=log_prob.device), special_token_idx]
                        # Add mask_indicator: if the last element of response_mask[i] is 0, mask_indicator[i]=1, else mask_indicator[i]=0
                        # response_mask: (bsz, response_len)
                        last_element = response_mask[:, -1]  # shape (bsz,)
                        mask_indicator = (last_element == 0).long()
                        # print(f"mask_indicator: {mask_indicator}")
                        masked_log_prob_special_token = log_prob_special_token * temperature # important
                        # Only compute internal_reward_loss where mask_indicator == 1
                        if not self.config.sft_loss_balanced:
                            internal_reward_loss = (-1 * masked_log_prob_special_token * mask_indicator).mean()
                        else:
                            internal_reward_loss = (-1 * masked_log_prob_special_token * mask_indicator * model_inputs["reweighting_factor"]).mean()
                        policy_loss = policy_loss + internal_reward_loss * self.config.sft_loss_coef
                        with torch.no_grad():
                            micro_batch_metrics["actor/internal_reward_loss"] = internal_reward_loss.detach().item()

                        # Calculate predicted reward based on internal_reward_ratio
                        pred_reward = (torch.exp(masked_log_prob_special_token.detach()) > 0.5).long()
                        # Get ground truth reward
                        individual_rewards = model_inputs["individual_rewards"].long()
                        # Only consider entries where mask_indicator == 1
                        valid_mask = (mask_indicator == 1)
                        # Overall accuracy (only count where mask_indicator == 1)
                        correct = (pred_reward == individual_rewards) & valid_mask
                        total_correct = correct.sum().item()
                        # Number of valid solutions (where mask_indicator == 1)
                        valid_solution_total = valid_mask.sum().item()
                        # Number of valid solutions where individual_rewards == 1 and mask_indicator == 1
                        valid_correct_solution_total = ((individual_rewards == 1) & valid_mask).sum().item()
                        # Number of valid solutions where individual_rewards == 0 and mask_indicator == 1
                        valid_incorrect_solution_total = ((individual_rewards == 0) & valid_mask).sum().item()
                        # Number of correct predictions when individual_rewards == 1 and mask_indicator == 1
                        mask_correct = (individual_rewards == 1) & valid_mask
                        if mask_correct.any():
                            correct_correct = ((pred_reward[mask_correct] == 1).sum().item())
                        else:
                            correct_correct = 0
                        # Number of correct predictions when individual_rewards == 0 and mask_indicator == 1
                        mask_incorrect = (individual_rewards == 0) & valid_mask
                        if mask_incorrect.any():
                            correct_incorrect = ((pred_reward[mask_incorrect] == 0).sum().item())
                        else:
                            correct_incorrect = 0
                        micro_batch_metrics["actor/internal_reward_total_valid_solutions"] = valid_solution_total
                        micro_batch_metrics["actor/internal_reward_valid_solutions_total_correct"] = total_correct
                        micro_batch_metrics["actor/internal_reward_total_valid_correct_solutions"] = valid_correct_solution_total
                        micro_batch_metrics["actor/internal_reward_total_valid_incorrect_solutions"] = valid_incorrect_solution_total
                        micro_batch_metrics["actor/internal_reward_valid_correct_solutions_total_correct"] = correct_correct
                        micro_batch_metrics["actor/internal_reward_valid_incorrect_solutions_total_correct"] = correct_incorrect

                        micro_batch_metrics["actor/self_reward_loss_coef"] = self.config.self_reward_loss_coef
                        micro_batch_metrics["actor/self_reward_kl_coef"] = self.config.self_reward_kl_coef
                        

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * (response_mask.shape[0] / self.config.ppo_mini_batch_size)
                    else:
                        loss = policy_loss / self.gradient_accumulation
                    loss.backward()

                    micro_batch_metrics.update(
                        {
                            "actor/pg_loss": pg_loss.detach().item(),
                            "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                            "actor/ppo_kl": ppo_kl.detach().item(),
                            "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                        }
                    )
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()
        return metrics
