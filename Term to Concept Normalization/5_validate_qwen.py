import os

# **Set CUDA_VISIBLE_DEVICES to exclude GPU 3**
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from tqdm import tqdm
import nltk
import json
from peft import PeftModel
import re
from sklearn.metrics import classification_report, accuracy_score

nltk.download('punkt')

# Paths
model_path = './outputs/qwen_finetuned_32b'
val_data_file = './outputs/fine_tuning_valid_data.jsonl'
output_file = './outputs/validation_results.csv'

# Configure quantization to 8-bit with CPU offloading
quant_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_skip_modules=None,
    llm_int8_enable_fp32_cpu_offload=True  # Enable CPU offloading
)

# Base model name (used for tokenizer and base model)
base_model_name = 'Qwen/Qwen2.5-32B-Instruct'

# Load tokenizer from the base model
tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Custom device map (allows CPU offloading)
device_map = 'auto'

# Load base model with CPU offloading enabled
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=quant_config,
    device_map=device_map,
    trust_remote_code=True
)

# Load the LoRA adapters from your fine-tuned model
model = PeftModel.from_pretrained(
    model,
    model_path
    # Do not pass device_map again here
)

# **Print the model's device map for debugging**
print("Model device map:", model.hf_device_map)

# Set model to evaluation mode
model.eval()

# Load validation data
val_data = []
with open(val_data_file, 'r') as f:
    for line in f:
        item = json.loads(line)
        val_data.append(item)

# Create a DataFrame for evaluation
df = pd.DataFrame(val_data)

# Generate predictions
def generate_predictions(prompts):
    predictions = []
    for idx, prompt in enumerate(tqdm(prompts)):
        # Tokenization
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        input_ids = inputs["input_ids"]
        input_length = input_ids.shape[1]

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=inputs["attention_mask"],
                max_new_tokens=50,
                do_sample=False,
                num_beams=1,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

        generated_ids = outputs[0]
        response_ids = generated_ids[input_length:]
        generated_text = tokenizer.decode(response_ids, skip_special_tokens=True).strip()

        # Debugging outputs
        print(f'Prompt {idx+1}: {prompt[:75]}...')  # Show first 75 chars of prompt
        print(f'Raw generated text: {generated_text}')

        # Extract the group name
        group_name = extract_group_name(generated_text)

        print(f'Full output: {generated_text}')
        print(f'Extracted group: {group_name}')

        predictions.append(group_name)
    return predictions

# Get prompts from validation data
prompts = df['prompt'].tolist()

def extract_group_name(response):
    # Use regex to find 'GROUP:' followed by the group name
    match = re.search(r'GROUP:\s*(.+)', response, re.IGNORECASE)
    if match:
        group_name = match.group(1)
        # Remove EOS token and any trailing whitespaces
        group_name = group_name.replace(tokenizer.eos_token, '').strip()
        # Remove any trailing special tokens or text after the group name
        group_name = re.split(r'[\n\r]+', group_name)[0]
        return group_name
    else:
        return "Invalid Output"

def clean_group_name(name):
    # Remove 'GROUP:' prefix if present
    name = re.sub(r'^GROUP:\s*', '', name, flags=re.IGNORECASE)
    # Remove EOS token and strip whitespaces
    name = name.replace(tokenizer.eos_token, '').replace('</s>', '').strip()
    # Remove any trailing control characters or whitespaces
    name = re.sub(r'[\n\r]+', '', name)
    return name

# Generate predictions
print("Generating predictions on validation set...")
predictions = generate_predictions(prompts)

# Clean and normalize predicted groups
predicted_groups = [clean_group_name(pred) for pred in predictions]
predicted_groups = [name.lower() for name in predicted_groups]

# Clean and normalize true groups
true_groups = df['response'].apply(clean_group_name).tolist()
true_groups = [name.lower() for name in true_groups]

# Compute accuracy
group_accuracy = accuracy_score(true_groups, predicted_groups)
print(f"\nGROUP Prediction Accuracy: {group_accuracy * 100:.2f}%")

# Print classification report
print("\nClassification Report for GROUP:")
print(classification_report(true_groups, predicted_groups, zero_division=1))

# Save the results to a CSV file
df['Predicted_GROUP'] = predicted_groups
df['True_GROUP'] = true_groups
df.to_csv(output_file, index=False)
print(f"\nValidation results saved to {output_file}")