import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig


# ================================
# 0. Argument parsing
# ================================
def parse_args():
    parser = argparse.ArgumentParser(description="Run DPO training on a numbered dataset file.")
    parser.add_argument(
        "--dataset-id",
        type=int,
        help="Numbered dataset identifier, e.g. 3 for files matching ultrafeedback_3_*.jsonl."
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        help="Optional explicit path to a dataset JSONL file. Overrides --dataset-id if provided."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Optional explicit output directory for this training run."
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="./outputs",
        help="Root directory for saving runs when --output-dir is not provided."
    )
    return parser.parse_args()


args = parse_args()
repo_dir = Path(__file__).resolve().parent

if args.dataset_path:
    data_path = Path(args.dataset_path).expanduser()
else:
    if args.dataset_id is None:
        raise ValueError("Provide either --dataset-path or --dataset-id so the dataset file can be located.")
    matches = sorted(repo_dir.glob(f"ultrafeedback_{args.dataset_id}_*.jsonl"))
    if not matches:
        raise FileNotFoundError(
            f"No dataset file found for id {args.dataset_id}. Expected files like ultrafeedback_{args.dataset_id}_*.jsonl"
        )
    data_path = matches[0]

dataset_stem = data_path.stem
if args.output_dir:
    output_dir = Path(args.output_dir).expanduser()
else:
    output_dir = Path(args.output_root).expanduser() / dataset_stem
output_dir.mkdir(parents=True, exist_ok=True)


# ================================
# 1. Paths and model config
# ================================

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DATA_PATH = str(data_path)
OUTPUT_DIR = str(output_dir)


# ================================
# 2. Load tokenizer
# ================================
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True,
    padding_side="left"
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# ================================
# 3. Load base model (FP16 or BF16)
# ================================
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    attn_implementation="sdpa",
)

model.gradient_checkpointing_enable()


# ================================
# 4. LoRA Configuration
# ================================
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    # r=4,
    # lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
)


# ================================
# 5. Load DPO dataset
# ================================
dataset = load_dataset("json", data_files=DATA_PATH, split="train")

def convert_to_conversational(example):
    return {
        "prompt": [
            {"role": "user", "content": example["prompt"]}
        ],
        "chosen": [
            {"role": "assistant", "content": example["chosen"]}
        ],
        "rejected": [
            {"role": "assistant", "content": example["rejected"]}
        ],
    }

dataset = dataset.map(convert_to_conversational)


# ================================
# 6. DPO Training Configuration
# ================================
dpo_config = DPOConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=5e-6,
    beta=0.1,
    num_train_epochs=1,
    logging_steps=20,
    save_steps=500,
    save_total_limit=2,
    max_steps=2000,

    bf16=True,
    max_length=2048,
    max_prompt_length=1024,
    remove_unused_columns=False,
    gradient_checkpointing=True,

    report_to=[]
)


# ================================
# 7. Initialize DPO Trainer
# ================================
trainer = DPOTrainer(
    model=model,
    ref_model=None,
    args=dpo_config,
    train_dataset=dataset,
    peft_config=lora_config
)

print("Starting DPO LoRA training...\n")
print(f"Dataset: {DATA_PATH}")
print(f"Output dir: {OUTPUT_DIR}")
print(dataset[0])
trainer.train()


# ================================
# 8. Save final model
# ================================
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("\nTraining completed!")
print("LoRA adapter saved to:", OUTPUT_DIR)
