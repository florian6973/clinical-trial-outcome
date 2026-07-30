# clinical_trials/12_fine_tune_qwen.py
import os
import json
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig,
    default_data_collator
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from accelerate import Accelerator
from tqdm import tqdm

# Disable tokenizer parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

# Set up Accelerator
accelerator = Accelerator()

# Set PYTORCH_CUDA_ALLOC_CONF for better memory management
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

model_name = "Qwen/Qwen2.5-32B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    max_len = 512  # Adjust as needed
    
    prompts = [p if p is not None else '' for p in examples['prompt']]
    responses = [r if r is not None else '' for r in examples['response']]
    
    input_ids_list = []
    labels_list = []
    
    for prompt, response in zip(prompts, responses):
        # Tokenize prompt and response separately without adding special tokens
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        response_ids = tokenizer.encode(response + tokenizer.eos_token, add_special_tokens=False)
        
        # Combine prompt and response
        input_ids = prompt_ids + response_ids
        
        # Create labels: mask prompt tokens
        labels = [-100]*len(prompt_ids) + response_ids
        
        # Truncate sequences if they exceed max_len
        input_ids = input_ids[:max_len]
        labels = labels[:max_len]
        
        # Pad sequences to max_len
        padding_length = max_len - len(input_ids)
        input_ids += [tokenizer.pad_token_id]*padding_length
        labels += [-100]*padding_length
        
        input_ids_list.append(input_ids)
        labels_list.append(labels)
    
    encoding = {
        'input_ids': torch.tensor(input_ids_list, dtype=torch.long),
        'labels': torch.tensor(labels_list, dtype=torch.long),
    }
    
    return encoding

# Load data
train_data_file = './outputs/fine_tuning_train_data.jsonl'
val_data_file = './outputs/fine_tuning_valid_data.jsonl'

def load_data_from_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON on line {i} in {file_path}: {e}")
    return data

train_data = load_data_from_jsonl(train_data_file)
val_data = load_data_from_jsonl(val_data_file)

train_dataset = Dataset.from_list(train_data)
val_dataset = Dataset.from_list(val_data)

# Tokenize datasets
tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_val_dataset = val_dataset.map(tokenize_function, batched=True)

# Configure quantization to 8-bit
quant_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_skip_modules=None
)

# Load the model with quantization and device mapping
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quant_config,
    device_map='auto',
    trust_remote_code=True
)

# Configure LoRA
peft_config = LoraConfig(
    task_type="CAUSAL_LM",
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1
)

# Apply LoRA to the model
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# Set up optimizer
from torch.optim import AdamW
optimizer = AdamW(model.parameters(), lr=1e-5)

# Prepare dataloaders
from torch.utils.data import DataLoader
train_dataloader = DataLoader(
    tokenized_train_dataset,
    shuffle=True,
    batch_size=1,
    collate_fn=default_data_collator
)
val_dataloader = DataLoader(
    tokenized_val_dataset,
    batch_size=1,
    collate_fn=default_data_collator
)

# Prepare everything with accelerator
model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, val_dataloader
)

# Training loop
num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    # Create a progress bar for the training loop
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", disable=not accelerator.is_main_process)
    for step, batch in enumerate(progress_bar):
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
        
        # Accumulate loss
        total_loss += loss.item()
        
        # Update progress bar
        progress_bar.set_postfix({'loss': loss.item()})
        
    # Calculate average loss for the epoch
    avg_loss = total_loss / len(train_dataloader)
    # Use accelerator.print to print from the main process only
    accelerator.print(f"Epoch {epoch+1}/{num_epochs} finished. Average Loss: {avg_loss:.4f}")

    # Validation loop
    model.eval()
    eval_loss = 0
    eval_steps = 0
    with torch.no_grad():
        for batch in val_dataloader:
            outputs = model(**batch)
            loss = outputs.loss
            eval_loss += loss.item()
            eval_steps += 1

    avg_eval_loss = eval_loss / eval_steps
    accelerator.print(f"Validation Loss after Epoch {epoch+1}: {avg_eval_loss:.4f}")

# Save the model
accelerator.wait_for_everyone()
unwrapped_model = accelerator.unwrap_model(model)
unwrapped_model.save_pretrained('./outputs/qwen_finetuned_32b', save_function=accelerator.save)