import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from tqdm import tqdm

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen-7B')
tokenizer.eos_token = tokenizer.eos_token or '</s>'
eos_token = tokenizer.eos_token

# Load your data (modify as needed)
df = pd.read_csv('/structured_input/Annotations_Complete.csv')


# Prepare the fine-tuning data
fine_tuning_data = []

for idx, row in tqdm(df.iterrows(), total=len(df)):
    prompt = row['prompt']
    response = f"GROUP: {row['GROUP']}{eos_token}"
    
    fine_tuning_example = {
        "prompt": prompt,
        "response": response 
    }
    fine_tuning_data.append(fine_tuning_example)

# Split the data into training and validation sets
train_data, val_data = train_test_split(fine_tuning_data, test_size=0.1, random_state=42)

# Save training data to JSONL file
train_output_file = './outputs/fine_tuning_train_data.jsonl'
with open(train_output_file, 'w') as f:
    for item in train_data:
        f.write(f"{json.dumps(item)}\n")

print(f"Training data saved to {train_output_file}")

# Save validation data to JSONL file
val_output_file = './outputs/fine_tuning_valid_data.jsonl'
with open(val_output_file, 'w') as f:
        for item in val_data:
            f.write(f"{json.dumps(item)}\n")

print(f"Validation data saved to {val_output_file}")