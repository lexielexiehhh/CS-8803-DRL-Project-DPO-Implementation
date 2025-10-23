from SimpleDPOTrainer import MyDPOTrainer, DPOPairwiseCollator
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import argparse
# from typing import Any
import torch
import wandb
from datetime import datetime
show_local_log = True
use_wandb = True

def LOCAL_LOG(in_log: str):
    if show_local_log:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {in_log}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--tokenizer_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--train_dataset_path", type=str, default="data/dpo_dataset.jsonl")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--output_dir", type=str, default="outputs")
    # LoRA args (enabled by default)
    parser.add_argument("--use_lora", action="store_true", default=True, help="Use LoRA for training because it's faster")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank, no need to change for now")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha, no need to change for now")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout, no need to change for now")
    # logging args
    parser.add_argument("--use_wandb", action=argparse.BooleanOptionalAction, default=True, help="Use wandb for logging")
    parser.add_argument("--show_local_log", action=argparse.BooleanOptionalAction, default=True, help="Show local log")
    parser.add_argument("--local_show_loss", action=argparse.BooleanOptionalAction, default=True, help="Show loss in local log")
    # Comma-separated module name substrings to target
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    return parser.parse_args()

def add_lora_to_model(model, args):
    target_modules = [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_cfg)
    return model

# A callback function that logs the loss to wandb
def on_loss_computed(loss: float):
    LOCAL_LOG(f"Loss computed: {loss}")
    if use_wandb:
        wandb.log({
            "train_loss": loss.item(),
        })

def main():
    args = parse_args()
    global show_local_log, use_wandb
    show_local_log = args.show_local_log
    use_wandb = args.use_wandb
    if use_wandb:
        LOCAL_LOG(f"Initializing wandb")
        wandb.init(project="8803DRL-dpo-implementation", config=args)
    else:
        LOCAL_LOG(f"Wandb is disabled")

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
    )
    LOCAL_LOG(f"Model loaded from {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name,
        trust_remote_code=True,
        use_fast=True,
    )
    LOCAL_LOG(f"Tokenizer loaded from {args.tokenizer_name}")
    # Ensure padding token exists for small models that may not define it
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if hasattr(tokenizer, "padding_side"):
        tokenizer.padding_side = "right"
    # Resize embeddings if we've added a pad token
    if hasattr(model, "resize_token_embeddings"):
        model.resize_token_embeddings(len(tokenizer))

    # trying to minize running time, so lora is used by default    
    if args.use_lora:
        model = add_lora_to_model(model, args)
    LOCAL_LOG(f"LoRA applied to model")
    train_dataset = load_dataset("json", data_files={"train": args.train_dataset_path})["train"]
    collator = DPOPairwiseCollator(tokenizer, max_length=args.max_length)
    LOCAL_LOG(f"Collator created")
    train_args = TrainingArguments(
        output_dir="outputs",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        learning_rate=1e-5,
        num_train_epochs=1,
        logging_steps=1,
        logging_first_step=True,
        save_steps=200,
        bf16=torch.cuda.is_available(),
        report_to=["wandb"] if use_wandb else [],
        remove_unused_columns=False,
    )
    LOCAL_LOG(f"TrainingArguments created")
    trainer = MyDPOTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        data_collator=collator,
        args=train_args,
    )
    LOCAL_LOG(f"Trainer created")
    if args.local_show_loss:
        trainer.register_callback_functions(on_loss_computed)
    LOCAL_LOG(f"Starting training")
    trainer.train()
    LOCAL_LOG(f"Training completed")
    trainer.save_model("dpo_model")
    LOCAL_LOG(f"Model saved to dpo_model")
    if use_wandb:
        # Attempt to log final loss from history
        last_with_loss = None
        for record in reversed(trainer.state.log_history):
            if isinstance(record, dict) and "loss" in record:
                last_with_loss = record
                break
        if last_with_loss is not None:
            wandb.log({
                "final_train_loss": last_with_loss["loss"],
                "step": last_with_loss.get("step", trainer.state.global_step),
            })
        wandb.finish()
if __name__ == "__main__":
    main()