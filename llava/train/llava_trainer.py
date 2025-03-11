import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from transformers import PreTrainedModel
from torch import nn

from .ppo_trainer import PPOTrainer
from .grpo_config import GRPOConfig
from ..models import PreTrainedModelWrapper
  
class GRPOTrainer(PPOTrainer):
    """
    Group Relative Policy Optimization (Group RPO) Trainer.
    Extends PPOTrainer with group-based optimization techniques.
    """

    def __init__(
        self,
        config: GRPOConfig = None,
        model: PreTrainedModelWrapper = None,
        ref_model: Optional[PreTrainedModelWrapper] = None,
        **kwargs
    ):
        super().__init__(config=config, model=model, ref_model=ref_model, **kwargs)
        self.config: GRPOConfig = config

    def compute_group_advantages(
        self,
        values: torch.FloatTensor,
        rewards: torch.FloatTensor,
        mask: torch.FloatTensor,
        group_indices: torch.LongTensor
    ) -> torch.FloatTensor:
        """
        Compute advantages within groups for relative optimization.
        """
        if not self.config.use_group_advantages:
            return super().compute_advantages(values, rewards, mask)[1]

        advantages = []
        unique_groups = torch.unique(group_indices)
        
        for group_id in unique_groups:
            group_mask = group_indices == group_id
            group_values = values[group_mask]
            group_rewards = rewards[group_mask]
            group_mask = mask[group_mask]
            
            if self.config.normalize_group_rewards:
                group_rewards = (group_rewards - group_rewards.mean()) / (group_rewards.std() + 1e-8)
            
            _, group_advantages, _ = super().compute_advantages(
                group_values, 
                group_rewards,
                group_mask
            )
            advantages.append(group_advantages)
            
        return torch.cat(advantages)

    def compute_group_loss(
        self,
        logprobs: torch.FloatTensor,
        values: torch.FloatTensor,
        old_logprobs: torch.FloatTensor,
        advantages: torch.FloatTensor,
        group_indices: torch.LongTensor,
        mask: torch.FloatTensor
    ) -> Tuple[torch.FloatTensor, Dict[str, torch.FloatTensor]]:
        """
        Compute the Group RPO loss combining PPO and relative group losses.
        """
        # Ensure all tensors are properly shaped [batch_size, seq_len]
        batch_size = logprobs.size(0)
        seq_len = logprobs.size(1) if logprobs.dim() > 1 else 1
        
        # Reshape tensors if needed
        if logprobs.dim() == 1:
            logprobs = logprobs.view(batch_size, 1)
        if old_logprobs.dim() == 1:
            old_logprobs = old_logprobs.view(batch_size, 1)
        if values.dim() == 1:
            values = values.view(batch_size, 1)
        if advantages.dim() == 1:
            advantages = advantages.view(batch_size, 1)
        if mask.dim() == 1:
            mask = mask.view(batch_size, 1)
            
        # Compute policy gradient loss
        ratio = torch.exp(logprobs - old_logprobs)
        pg_losses = -advantages * ratio
        
        # Compute clipped version of policy gradient loss
        eps = self.config.group_margin
        pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - eps, 1.0 + eps)
        
        # Take maximum of clipped and unclipped versions
        pg_loss = torch.mean(torch.max(pg_losses, pg_losses2) * mask)
        
        # Value function loss (MSE)
        returns = advantages + values
        vf_losses = (values - returns) ** 2
        vf_loss = 0.5 * torch.mean(vf_losses * mask)
        
        # Create stats dictionary
        stats = {
            "policy_loss": pg_loss.item(),
            "value_loss": vf_loss.item(),
            "policy_ratio": ratio.mean().item(),
        }

        # Group relative loss
        group_loss = torch.tensor(0.0, device=logprobs.device)
        unique_groups = torch.unique(group_indices)
        
        # Compute mean values per sequence for group comparisons
        logprobs_mean = (logprobs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-8)
        advantages_mean = (advantages * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-8)
        
        for group_id in unique_groups:
            group_mask = group_indices == group_id
            group_logprobs = logprobs_mean[group_mask]
            group_advantages = advantages_mean[group_mask]
            
            # Compute pairwise differences within group
            diff_logprobs = group_logprobs.unsqueeze(1) - group_logprobs.unsqueeze(0)
            diff_advantages = group_advantages.unsqueeze(1) - group_advantages.unsqueeze(0)
            
            # Compute relative loss based on configured type
            if self.config.relative_loss_type == "hinge":
                rel_loss = torch.relu(
                    -diff_logprobs * torch.sign(diff_advantages) + self.config.group_margin
                )
            elif self.config.relative_loss_type == "log":
                rel_loss = -torch.log(torch.sigmoid(
                    diff_logprobs * diff_advantages / self.config.group_temperature
                ))
            else:  # exp
                rel_loss = torch.exp(
                    -diff_logprobs * diff_advantages / self.config.group_temperature
                )
            
            group_loss += rel_loss.mean()

        # Combine losses
        total_loss = pg_loss + self.config.vf_coef * vf_loss + \
                    self.config.group_weight * group_loss

        # Update stats
        stats["loss/group"] = group_loss.detach()
        stats["loss/total"] = total_loss.detach()
        
        return total_loss, stats

    def train_minibatch(
        self,
        old_logprobs: torch.FloatTensor,
        values: torch.FloatTensor,
        logprobs: torch.FloatTensor,
        logits: torch.FloatTensor,
        vpreds: torch.FloatTensor,
        mask: torch.LongTensor,
        advantages: torch.FloatTensor,
        returns: torch.FloatTensor
    ):
        """
        Train one Group RPO minibatch.
        """
        self.model.train()
        
        # Create random groups for this minibatch
        batch_size = logprobs.shape[0]
        num_groups = batch_size // self.config.group_size
        group_indices = torch.repeat_interleave(
            torch.arange(num_groups, device=logprobs.device),
            self.config.group_size
        )
        
        # Shuffle groups
        perm = torch.randperm(batch_size, device=logprobs.device)
        group_indices = group_indices[perm]
        
        # Compute group-based advantages
        advantages = self.compute_group_advantages(
            values, advantages, mask, group_indices
        )
        
        # Compute loss with group optimization
        loss, train_stats = self.compute_group_loss(
            logprobs, vpreds, old_logprobs, advantages,
            group_indices, mask
        )
        
        # Optimization step
        self.accelerator.backward(loss)
        if self.config.max_grad_norm is not None:
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(
                    self.model_params,
                    self.config.max_grad_norm
                )
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        return train_stats 
