from SimpleDPOTrainer import MyDPOTrainer, DPOPairwiseCollator
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import load_dataset
import argparse
import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--tokenizer_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--train_dataset_path", type=str, default="data/train.json")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--output_dir", type=str, default="outputs")
    return parser.parse_args()

def add_lora_to_model(model):
    return model

def main():
    args = parse_args()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name,
        trust_remote_code=True,
        use_fast=True,
    )
    # Ensure padding token exists for small models that may not define it
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if hasattr(tokenizer, "padding_side"):
        tokenizer.padding_side = "right"
    # Resize embeddings if we've added a pad token
    if hasattr(model, "resize_token_embeddings"):
        model.resize_token_embeddings(len(tokenizer))
    train_dataset = load_dataset("json", data_files={"train": args.train_dataset_path})["train"]
    collator = DPOPairwiseCollator(tokenizer, max_length=args.max_length)

    args = TrainingArguments(
        output_dir="outputs",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1e-5,
        num_train_epochs=1,
        logging_steps=10,
        save_steps=200,
        bf16=torch.cuda.is_available(),
        report_to=[],
    )
    trainer = MyDPOTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        data_collator=collator,
        args=args,
    )
    trainer.train()
    trainer.save_model("dpo_model")
if __name__ == "__main__":
    main()