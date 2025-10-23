import json
import itertools
import random
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from tqdm import tqdm

MODEL_NAME = "gpt2"
CLASSIFIER_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
NUM_PROMPTS = 100
NUM_COMPLETIONS_PER_PROMPT = 4
OUTPUT_FILE = "data/dpo_dataset.jsonl"


def synthesize_data():

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    # pre-trained sentiment clssifier
    sentiment_classifier = pipeline(
        "sentiment-analysis",
        model=CLASSIFIER_NAME,
        device=0 if torch.cuda.is_available() else -1 
    )
    model.to(sentiment_classifier.device) 
    
    imdb_dataset = load_dataset("imdb", split="train")
    
    # get first 100 reviews as prompts 
    prompts = []
    # the second 100 reviews as the target, just in case some reviews are too short
    for example in imdb_dataset.select(range(NUM_PROMPTS * 2)):
        # get the first 50 words（prefix）of the review as the prompt

        tokens = tokenizer(example['text']).input_ids
        
        # if the review is too short (less than 8 tokens), skip
        if len(tokens) <= 8:
            continue
            
        # random select a length between 2 and 8
        prefix_length = random.randint(2, 8)

        prefix_tokens = tokens[:prefix_length]
        
        prefix = tokenizer.decode(prefix_tokens)

        prompts.append(prefix)
        
        # ensure we only get 100 prompts
        if len(prompts) >= NUM_PROMPTS: 
            break 

    final_preference_dataset = []
    
    for prompt_text in tqdm(prompts, desc="processing Prompts"):
        
        # generation
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            num_return_sequences=NUM_COMPLETIONS_PER_PROMPT,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            pad_token_id=tokenizer.pad_token_id
        )
        

        completions_full_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        completions_only_y = [text[len(prompt_text):] for text in completions_full_text]

        # scoring (prompt + completion)
        valid_texts_for_scoring = []
        valid_completions_only_y = []
        
        for full_text, comp_only in zip(completions_full_text, completions_only_y):
            # filter out too short or strange completions
            if len(comp_only.strip()) > 5: 
                valid_texts_for_scoring.append(full_text)
                valid_completions_only_y.append(comp_only)

        if len(valid_texts_for_scoring) < 2:
            continue
            
        # sentiment classifier scores the (Prompt + Completion)
        scores = sentiment_classifier(valid_texts_for_scoring)

        scored_completions = []
        # map the score and the completion text (y)
        for text_y, score_dict in zip(valid_completions_only_y, scores):
            final_score = score_dict['score'] if score_dict['label'] == 'POSITIVE' else -score_dict['score']
            scored_completions.append( (text_y, final_score) ) # 存的是 (y, score)

        # pair the completions
        for pair in itertools.combinations(scored_completions, 2):
            text_i, score_i = pair[0]
            text_j, score_j = pair[1]

            if score_i > score_j:
                y_w, y_l = text_i, text_j 
            elif score_j > score_i:
                y_w, y_l = text_j, text_i 
            else:
                continue

            final_preference_dataset.append({
                "prompt": prompt_text,
                "chosen": y_w,   
                "rejected": y_l  
            })

    # save the data
    with open(OUTPUT_FILE, "w") as f:
        for item in final_preference_dataset:
            f.write(json.dumps(item) + "\n")
            
    print(f"--- completed! ---")
    print(f"successfully created the dataset {OUTPUT_FILE}, total {len(final_preference_dataset)} preference data.")

if __name__ == "__main__":
    synthesize_data()