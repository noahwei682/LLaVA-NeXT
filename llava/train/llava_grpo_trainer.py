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
        # Store GRPO config first
        self.grpo_config = grpo_config or GRPOConfig()
        
        # Remove grpo_config from kwargs to avoid passing it to parent classes
        kwargs_without_grpo = {k: v for k, v in kwargs.items() if k != 'grpo_config'}
        
        # Initialize parent classes
        LLaVATrainer.__init__(self, **kwargs_without_grpo)
        
        # Store model config
        if hasattr(kwargs.get('model', None), 'config'):
            self.config = kwargs['model'].config
            # Add GRPO-specific attributes to model config
            for key, value in vars(self.grpo_config).items():
                setattr(self.config, key, value)
        
        # Initialize other necessary attributes
        self.use_dpo_data_collator = False
        
        
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
        logits = outputs.logits  # [batch_size, seq_len, vocab_size]
        if hasattr(outputs, "value"):
            values = outputs.value
        else:
            # For GSM8K, we use negative loss as value for each token
            # Reshape loss to [batch_size, 1] and expand to [batch_size, seq_len]
            values = -loss.detach().view(-1, 1).expand(-1, logits.shape[1])
            
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
        
        # Create attention mask if not provided
        attention_mask = inputs.get("attention_mask", torch.ones((batch_size, logits.shape[1]), device=logits.device))
        
        # Get log probabilities for each token
        log_probs = torch.log_softmax(logits, dim=-1)  # [batch_size, seq_len, vocab_size]
        
        # Get the log probabilities of the actual tokens
        labels = inputs["labels"]  # [batch_size, seq_len]
        labels_mask = (labels != -100)  # Create mask for valid tokens
        
        # Gather the log probabilities of the actual tokens
        token_log_probs = torch.gather(log_probs, -1, labels.unsqueeze(-1).expand(-1, -1, 1).clamp(min=0)).squeeze(-1)
        token_log_probs = token_log_probs * labels_mask  # Zero out invalid positions
        
        # Compute advantages
        rewards = -loss.detach()  # Use negative loss as reward [batch_size]
        if values.dim() == 2:
            values_mean = values.mean(dim=1)  # [batch_size]
        else:
            values_mean = values.mean()  # scalar
            
        # Reshape rewards and values for broadcasting
        rewards = rewards.view(-1)  # [batch_size]
        values_mean = values_mean.view(-1)  # [batch_size]
        advantages = (rewards - values_mean).unsqueeze(1)  # [batch_size, 1]
        
        # Ensure all tensors are on the same device and have the same dtype
        device = logits.device
        dtype = logits.dtype
        
        # Convert tensors to the correct device and dtype
        token_log_probs = token_log_probs.to(device=device, dtype=dtype)
        values = values.to(device=device, dtype=dtype)
        advantages = advantages.expand(-1, logits.shape[1]).to(device=device, dtype=dtype)  # [batch_size, seq_len]
        attention_mask = attention_mask.to(device=device, dtype=dtype)
        
        # Create clipping tensors for PPO
        eps = self.grpo_config.group_margin
        self.clip_range = torch.tensor(eps, device=device, dtype=dtype)
        self.clip_range_value = torch.tensor(eps, device=device, dtype=dtype)
        
        # Compute group RPO loss
        grpo_loss, stats = self.compute_group_loss(
            token_log_probs,
            values,
            token_log_probs.detach(),  # Use current log probs as old log probs
            advantages,
            group_indices,
            attention_mask
        )
        
        # For GSM8K, we want to emphasize the final answer tokens more
        if "final_answer_mask" in inputs:
            final_answer_weight = 2.0  # Give more weight to final answer tokens
            grpo_loss = grpo_loss * (1 + (final_answer_weight - 1) * inputs["final_answer_mask"])
        
        # Combine losses
        total_loss = loss + self.grpo_config.group_weight * grpo_loss.mean()
        
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
