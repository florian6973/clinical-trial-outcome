from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
import torch
from llm import format_prompt
from annotations import dataset

# Paths to original base model and LoRA-adapted model
base_model_name = "/gpfs/commons/groups/gursoy_lab/fpollet/models/Llama-3.2-1B-Instruct"
lora_model_path = "./llama-lora-adapted"  # Path where you saved the adapter weights

# Load the tokenizer
base_model_name = "/gpfs/commons/groups/gursoy_lab/fpollet/models/Llama-3.2-1B-Instruct"
lora_model_path = "./llama-lora-adapted"  # Directory containing adapter_model.safetensors
from peft import PeftModel

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token = tokenizer.eos_token

# Load the base model
model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.bfloat16)

# Load the LoRA adapter weights (which are in safetensors format)
model = PeftModel.from_pretrained(model, lora_model_path, torch_dtype=torch.bfloat16)


device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Example user input
for sample in (dataset['test']):
    # use similar prompt https://huggingface.co/docs/trl/sft_trainer instruction template
    prompt = f"### Question EXTRACT INFO JSON: " + sample['origs'] + "\n ### Answer: "

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Generate a response
    # Adjust generation parameters as needed (max length, temperature, top_p, etc.)
    output_tokens = model.generate(
        **inputs,
        max_length=256,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )

    # Decode and print the response
    generated_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    print(generated_text)
    input()