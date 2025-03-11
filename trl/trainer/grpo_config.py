from dataclasses import dataclass
from typing import Optional, Literal
from .ppo_config import PPOConfig

@dataclass
class GRPOConfig(PPOConfig):
    """
    Configuration class for Group Relative Policy Optimization (Group RPO).
    Extends PPOConfig with additional parameters specific to group-based optimization.
    """
    
    group_size: int = 4
    """Size of groups for relative optimization (number of samples to compare)"""
    
    group_weight: float = 1.0
    """Weight factor for group-based loss component"""
    
    relative_loss_type: Literal["hinge", "log", "exp"] = "hinge"
    """Type of relative loss function to use between group members"""
    
    group_margin: float = 0.1
    """Margin for hinge loss in group comparisons"""
    
    use_group_advantages: bool = True
    """Whether to compute advantages within groups"""
    
    group_temperature: float = 1.0
    """Temperature parameter for softmax in group comparisons"""
    
    normalize_group_rewards: bool = True
    """Whether to normalize rewards within groups""" 
