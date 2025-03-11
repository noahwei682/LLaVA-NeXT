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
        data_path: str = "openai/gsm8k",
        split: str = "train",
        max_length: int = 2048,
    ):
        super().__init__()
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load GSM8K dataset with 'main' config
        self.dataset = load_dataset(data_path, 'main', split=split)
        
        # Pre-compute lengths for each sample
        self.lengths = []
        for i in range(len(self.dataset)):
            item = self.dataset[i]
            question = item["question"]
            answer = item["answer"]
            
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
                    reasoning=answer.split("####")[0].strip(),
                    answer=answer.split("####")[-1].strip()
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
            
            # Get length of tokenized prompt
            length = len(self.tokenizer(prompt, truncation=False)["input_ids"])
            self.lengths.append(length)


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
        tokenized = self.tokenizer(
            prompt,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors=None,  # Return python lists instead of tensors
        )
        
        # Convert to tensors
        input_ids = torch.tensor(tokenized["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(tokenized["attention_mask"], dtype=torch.long)
        
        # For training, we want the model to predict the entire sequence
        labels = input_ids.clone()
        # Mask out padding tokens
        labels[labels == self.tokenizer.pad_token_id] = IGNORE_INDEX
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        } 
