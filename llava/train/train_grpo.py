import os
import sys
from dataclasses import dataclass, field
from typing import Optional
import torch
from transformers import HfArgumentParser, TrainingArguments

from llava.train.train import (
    ModelArguments,
    DataArguments,
)
from trl.trainer.grpo_config import GRPOConfig
from llava.train.llava_grpo_trainer import LLaVAGRPOTrainer
from llava.data.gsm8k_dataset import GSM8KDataset

@dataclass
class GSM8KTrainingArguments(TrainingArguments):
    dataset_name: str = field(default="modelscope/gsm8k")
    dataset_config: str = field(default="main")
    group_size: int = field(default=8)
    group_weight: float = field(default=0.5)
    relative_loss_type: str = field(default="log")
    group_margin: float = field(default=0.05)
    use_group_advantages: bool = field(default=True)
    group_temperature: float = field(default=0.5)
    normalize_group_rewards: bool = field(default=True)

def main():
    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataArguments, GSM8KTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Create GRPO config
    grpo_config = GRPOConfig(
        group_size=training_args.group_size,
        group_weight=training_args.group_weight,
        relative_loss_type=training_args.relative_loss_type,
        group_margin=training_args.group_margin,
        use_group_advantages=training_args.use_group_advantages,
        group_temperature=training_args.group_temperature,
        normalize_group_rewards=training_args.normalize_group_rewards
    )

    # Initialize model and tokenizer
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        padding_side="right",
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float16,
    )

    # Create datasets
    train_dataset = GSM8KDataset(
        tokenizer=tokenizer,
        data_path=training_args.dataset_name,
        split="train",
        max_length=training_args.model_max_length,
    )
    eval_dataset = GSM8KDataset(
        tokenizer=tokenizer,
        data_path=training_args.dataset_name,
        split="test",
        max_length=training_args.model_max_length,
    )

    # Initialize trainer
    trainer = LLaVAGRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        grpo_config=grpo_config,
    )

    # Train
    trainer.train()

if __name__ == "__main__":
    main() 
