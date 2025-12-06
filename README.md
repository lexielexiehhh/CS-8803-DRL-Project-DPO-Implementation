# CS-8803-DRL-Project-DPO-Implementation

This repository contains our implementation and experiments for Direct Preference Optimization (DPO) for CS 8803 Deep Reinforcement Learning.  
The workflow mirrors the steps documented in `final_main.ipynb`.

## 0. Prerequistis

- A GPU is required.
- Please download and unzip the data and models first. The `final_main.ipynb` will talk about where to put those files and asserts.
- Please download preprocessed data from https://drive.google.com/drive/folders/15hqwcZ_TVrvt30Rmw5VLbHCKyQ2Ydhdj?usp=sharing
- Please download trained models from https://drive.google.com/drive/folders/1aRMeu6YbQQjsO2KWWFND0a3PiQd-0s00?usp=drive_link

## 1. Setup

- Create and activate a Python environment (e.g. conda or venv).
- Install dependencies:

```bash
pip install -r requirements.txt
```

## 2. Implementing the DPO trainer (section 0 in `final_main.ipynb`)

The from-scratch DPO trainer and collator implemented in the notebook are available as:

- `src/simple_dpo_trainer.py` – `MyDPOTrainer` and `DPOPairwiseCollator`
- `src/train.py` – example training script that uses the custom trainer on a JSONL preference dataset.

There is also a small synthetic data pipeline for toy experiments:

- `synthesize_data.py` – generates a synthetic preference dataset and saves `data/dpo_dataset.jsonl`.
- `data_utils.py` – helper Dataset class for working with JSONL preference data.

To reproduce the small-scale toy run with the custom trainer:

```bash
python synthesize_data.py                    # optional synthetic data
python src/train.py --train_dataset_path data/dpo_dataset.jsonl
```

## 3. Data preparation for main experiments (section 1 in `final_main.ipynb`)

For the main experiments we use preference data derived from the UltraFeedback dataset.

1. Preprocessing is documented in `data_preprocess.ipynb` (see the notebook and logs for details).
2. Use the preprocessed / aligned JSONL files (or download the prepared data as described in the notebook) and place them under `./data`, e.g.:
   - `./data/ultrafeedback_student_aligned.jsonl`
   - `./data/ultrafeedback_professor_aligned.jsonl`
   - `./data/ultrafeedback_swe_aligned.jsonl`
   - `./data/ultrafeedback_helpfulness_aligned.jsonl`
   - `./data/ultrafeedback_honesty_aligned.jsonl`
   - `./data/ultrafeedback_instruction_following_aligned.jsonl`
   - `./data/ultrafeedback_truthfulness_aligned.jsonl`

Make sure `./data` exists and contains the aligned JSONL files mentioned in `final_main.ipynb`.

## 4. Training Qwen LoRA DPO models (section 2 in `final_main.ipynb`)

The main training script is:

- `multi_dpo_training.py` – trains a LoRA adapter on one preference file using TRL’s `DPOTrainer` and Qwen/Qwen2.5-3B-Instruct.

Example: train a model on the “helpfulness” preference data:

```bash
python multi_dpo_training.py \
  --dataset-path ./data/ultrafeedback_helpfulness_aligned.jsonl \
  --output-dir ./models/ultrafeedback_4_helpfulness
```

You can repeat this for each aligned JSONL file to obtain one LoRA adapter per preference dimension, matching the `train_all()` pattern shown in `final_main.ipynb`.

The trained adapters and tokenizer are saved under the directory passed via `--output-dir` (e.g. `./models/ultrafeedback_3_swe`, etc.).

## 5. Evaluation (evaluation section in `final_main.ipynb`)

Evaluation code that mirrors the notebook is in:

- `src/evaluation/evaluation.py`

This script:

- Loads trained LoRA adapters on top of Qwen/Qwen2.5-3B-Instruct.
- Generates responses for a fixed set of prompts.
- Calls an OpenAI model as a “critic” to score each output on:
  - `instruction_following`
  - `honesty`
  - `truthfulness`
  - `helpfulness`
- Aggregates scores and can compute preference-weighted advantages.

Before running evaluation, ensure:

- Your trained adapters are in `./models/ultrafeedback_*` (as produced by `multi_dpo_training.py`).
- `OPENAI_API_KEY` is set in your environment.

Then you can run (for a full re-evaluation) by using `final_main.ipynb` or enabling the `main()` call in `src/evaluation/evaluation.py` or adapt the script as needed; the notebook `final_main.ipynb` shows the exact evaluation flow used in the report.
