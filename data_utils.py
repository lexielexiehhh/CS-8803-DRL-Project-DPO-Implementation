import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class DPODataLoader(Dataset):
    """
    A PyTorch Dataset class to load the dpo_dataset.jsonl file we synthesized.
    It will return the data needed for the DPO algorithm.
    """
    def __init__(self, jsonl_path, tokenizer, max_length=512):
        self.data = []
        with open(jsonl_path, "r") as f:
            for line in f:
                self.data.append(json.loads(line))
        
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        this function is the place where the DPO algorithm "gets data".
        it needs to return the tokenized prompt, chosen and rejected answers.
        """
        item = self.data[idx]
        
        prompt = item['prompt']
        chosen = item['chosen']
        rejected = item['rejected']
        

        chosen_tokenized = self.tokenizer(
            prompt + chosen + self.tokenizer.eos_token,
            max_length=self.max_length,
            padding="max_length", # fill to max length
            truncation=True,    
            return_tensors="pt"
        )
        
        rejected_tokenized = self.tokenizer(
            prompt + rejected + self.tokenizer.eos_token,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "chosen_input_ids": chosen_tokenized.input_ids.squeeze(0),
            "chosen_attention_mask": chosen_tokenized.attention_mask.squeeze(0),
            "rejected_input_ids": rejected_tokenized.input_ids.squeeze(0),
            "rejected_attention_mask": rejected_tokenized.attention_mask.squeeze(0)
        }
