from typing import Dict, Optional, Sequence
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
import json

from llava.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

XML_COT_FORMAT = """
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""

class GSM8KDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        data_path: str = "modelscope/gsm8k",
        split: str = "train",
        max_length: int = 2048,
    ):
        super().__init__()
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load GSM8K dataset
        self.dataset = load_dataset(data_path, split=split)
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, index) -> Dict:
        item = self.dataset[index]
        question = item["question"]
        answer = item["answer"]
        
        # Extract final answer from the solution
        final_answer = answer.split("####")[-1].strip()
        # Get step by step solution
        solution = answer.split("####")[0].strip()
        
        # Format messages in conversation format
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "数字10203040里面有几个0?"},
            {"role": "assistant", "content": XML_COT_FORMAT.format(
                reasoning="可以将数字拆开看，1、0、2、0、3、0、4、0，我们可以数出有4个0",
                answer="4"
            )},
            {"role": "user", "content": question},
            {"role": "assistant", "content": XML_COT_FORMAT.format(
                reasoning=solution,
                answer=final_answer
            )}
        ]
        
        # Convert messages to prompt
        prompt = ""
        for msg in messages:
            if msg["role"] == "system":
                prompt += f"{msg['content']}\n"
            elif msg["role"] == "user":
                prompt += f"User: {msg['content']}\n"
            else:  # assistant
                prompt += f"Assistant: {msg['content']}\n"
        
        # Tokenize
        prompt_tokens = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
        )
        
        # Create input_ids and labels
        input_ids = prompt_tokens["input_ids"][0]
        attention_mask = prompt_tokens["attention_mask"][0]
        
        # For training, we want the model to predict the entire sequence
        labels = input_ids.clone()
        # Mask out padding tokens
        labels[labels == self.tokenizer.pad_token_id] = IGNORE_INDEX
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        } 
