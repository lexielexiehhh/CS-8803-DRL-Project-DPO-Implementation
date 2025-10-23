# CS-8803-DRL-Project-DPO-Implementation
This repository contains the implementation of the Direct Preference Optimization (DPO) method from the selected paper for the CS 8803 Deep Reinforcement Learning course.

## 0. Hardware requirement
GPU is needed

## 1. Setup

Install dependencies:
`pip install -r requirements.txt`

## 2. Data Preparation

To synthesize the preference dataset, run:
`python synthesize_data.py`

## 3. Train
To train, run:
`python src/train.py`

Custom arguments and configurations can be found in the file.

This will generate a file named `dpo_dataset.jsonl`.