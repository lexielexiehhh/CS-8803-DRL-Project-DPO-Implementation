from SimpleDPOTrainer import MyDPOTrainer, DPOPairwiseCollator
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import load_dataset
import torch

def main():
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    train_dataset = load_dataset("json", data_files={"train": "data/train.json"})["train"]
    collator = DPOPairwiseCollator(tokenizer, max_length=1024)

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
        ref_model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        data_collator=collator,
        args=args,
    )
    trainer.train()
    trainer.save_model("dpo_model")
if __name__ == "__main__":
    main()