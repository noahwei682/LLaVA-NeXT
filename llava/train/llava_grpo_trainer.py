from typing import Dict, Optional, Sequence
import torch
from transformers import Trainer
from trl.trainer import PPOTrainer

from .llava_trainer import LLaVATrainer
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.grpo_trainer import GRPOTrainer

class LLaVAGRPOTrainer(LLaVATrainer, GRPOTrainer):
    """
    LLaVA trainer with Group Relative Policy Optimization (Group RPO).
    Combines LLaVA's multimodal capabilities with group-based optimization.
    """
    
    def __init__(self, grpo_config: GRPOConfig = None, **kwargs):
        super().__init__(**kwargs)
        self.grpo_config = grpo_config or GRPOConfig()
        
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute training loss with Group RPO components.
        Handles both multimodal and text-only (GSM8K) inputs.
        """
        if hasattr(self, "use_dpo_data_collator") and self.use_dpo_data_collator:
            return super().compute_loss(model, inputs, return_outputs)
            
        # Get standard outputs and loss
        outputs = model(**inputs)
        loss = outputs.loss
        
        # Extract logits and values for Group RPO
        logits = outputs.logits
        if hasattr(outputs, "value"):
            values = outputs.value
        else:
            # For GSM8K, we can use loss as a proxy for value
            values = -loss.detach().unsqueeze(-1).expand(-1, logits.shape[1])
            
        # Create groups for relative optimization
        batch_size = logits.shape[0]
        num_groups = batch_size // self.grpo_config.group_size
        if num_groups == 0:
            # If batch is smaller than group_size, use the entire batch as one group
            num_groups = 1
            self.grpo_config.group_size = batch_size
            
        group_indices = torch.repeat_interleave(
            torch.arange(num_groups, device=logits.device),
            self.grpo_config.group_size
        )[:batch_size]  # Truncate if needed
        
        # Shuffle groups
        perm = torch.randperm(batch_size, device=logits.device)
        group_indices = group_indices[perm]
        
        # For GSM8K, we want to group similar problems together
        # We can use the sequence lengths as a proxy for problem similarity
        if "attention_mask" in inputs:
            seq_lengths = inputs["attention_mask"].sum(-1)
            sorted_indices = torch.argsort(seq_lengths)
            group_indices = group_indices[sorted_indices]
        
        # Compute group-based advantages and loss
        advantages = self.compute_group_advantages(
            values, 
            loss.detach(),  # Use loss as reward signal
            inputs.get("attention_mask", torch.ones_like(values)),
            group_indices
        )
        
        # Get log probabilities
        log_probs = torch.log_softmax(logits, dim=-1)
        old_log_probs = log_probs.detach()
        
        # Compute group RPO loss
        grpo_loss, stats = self.compute_group_loss(
            log_probs,
            values,
            old_log_probs,
            advantages,
            group_indices,
            inputs.get("attention_mask", torch.ones_like(log_probs))
        )
        
        # For GSM8K, we want to emphasize the final answer tokens more
        if "final_answer_mask" in inputs:
            final_answer_weight = 2.0  # Give more weight to final answer tokens
            grpo_loss = grpo_loss * (1 + (final_answer_weight - 1) * inputs["final_answer_mask"])
        
        # Combine losses
        total_loss = loss + self.grpo_config.group_weight * grpo_loss
        
        if return_outputs:
            outputs.loss = total_loss
            return total_loss, outputs
        
        return total_loss
        
    def create_optimizer(self):
        """
        Create optimizer with proper parameter groups and learning rates.
        """
        if self.optimizer is None:
            return super().create_optimizer()
        return self.optimizer 
