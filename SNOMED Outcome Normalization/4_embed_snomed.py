import os
# Set CUDA_VISIBLE_DEVICES before importing torch
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

import csv
import torch
from transformers import AutoModel, AutoTokenizer
from huggingface_hub import login
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import pandas as pd

# Disable tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

print(torch.cuda.is_available())  # Should return True if CUDA is available

# Step 1: Authenticate with Hugging Face API key
api_key = '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'  # Replace with your actual API key
login(api_key)

model_name = 'nvidia/NV-Embed-v2'

# Step 2: Define directories
cache_dir = "/nlp/projects/llama/cache"
os.makedirs(cache_dir, exist_ok=True)

offload_folder = '/tmp/offload'  # Ensure this folder exists and has write permissions
os.makedirs(offload_folder, exist_ok=True)

# Step 3: Load the model with device_map='auto' without moving it to a single device
print("Loading the model...")
try:
    # Define max_memory per device (optional but recommended)
    max_memory = {i: '24000MB' for i in range(torch.cuda.device_count())}
    max_memory['cpu'] = '48GB'  # Adjust based on your system's RAM

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        trust_remote_code=True
    )

    # Load model with automatic device mapping
    model = AutoModel.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        trust_remote_code=True,
        device_map='auto',           # Automatically distribute the model
        max_memory=max_memory,
        offload_folder=offload_folder,
        torch_dtype=torch.float16    # Use half-precision
    )
    model.eval()  # Set model to evaluation mode
    print("Model loaded successfully.")
except Exception as e:
    print(f"An error occurred while loading the model: {e}")
    exit(1)

# Step 4: Get available devices (GPUs)
available_devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
print(f"Available devices: {available_devices}")

# Step 5: Define the path to your data file and output file
data_file = 'outputs/descendants_with_depth.csv'  # Adjust the path as needed

# Ensure the output directory exists
output_dir = os.path.dirname(data_file)
os.makedirs(output_dir, exist_ok=True)

# Define the output file path
output_file = 'outputs/snomed_embeddings.csv'

# Step 6: Verify the data file exists
if not os.path.isfile(data_file):
    print(f"Data file not found at: {data_file}")
    exit(1)

# Step 7: Read the CSV file
print("Reading the descendants_with_depth.csv file...")
df = pd.read_csv(data_file)
print(f"Loaded {len(df)} rows from {data_file}")

# Remove duplicate descriptions if necessary
df = df.drop_duplicates(subset='description').reset_index(drop=True)
print(f"After removing duplicates, {len(df)} descriptions remain.")

# Prepare the instruction
task_name_to_instruct = {"embedding": "Given a description, find the closest other descriptions"}
instruction = "Instruct: " + task_name_to_instruct["embedding"] + "\nDescription: "

# Determine appropriate max_length
# Compute max token length
token_lengths = [
    len(tokenizer.encode(description, add_special_tokens=True)) + len(tokenizer.encode(instruction, add_special_tokens=False))
    for description in df['description']
]
max_token_length = max(token_lengths)
print(f"Max token length in data: {max_token_length}")

# Set max_length slightly above the max token length to accommodate any variations
max_length = min(max_token_length + 20, 512)  # Ensure max_length is not excessively large

# Create a Dataset for descriptions
class DescriptionsDataset(Dataset):
    def __init__(self, descriptions, concept_ids, depths):
        self.descriptions = descriptions
        self.concept_ids = concept_ids
        self.depths = depths

    def __len__(self):
        return len(self.descriptions)

    def __getitem__(self, idx):
        return self.descriptions[idx], self.concept_ids[idx], self.depths[idx]

dataset = DescriptionsDataset(df['description'].tolist(), df['conceptId'].tolist(), df['depth'].tolist())
batch_size = 50  # Adjust based on your GPU memory and capacity

# Define a custom collate function
def custom_collate_fn(batch):
    batch_descriptions, batch_concept_ids, batch_depths = zip(*batch)
    return list(batch_descriptions), list(batch_concept_ids), list(batch_depths)

dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, collate_fn=custom_collate_fn)

def embed_batch(batch_data):
    batch_descriptions, batch_concept_ids, batch_depths = batch_data
    # Use the model's encode method to get embeddings
    embeddings = model.encode(
        batch_descriptions,
        instruction=instruction,
        max_length=max_length
    )
    
    # Normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)
    
    return batch_concept_ids, batch_descriptions, batch_depths, embeddings.cpu().numpy()

# Step 8: Generate embeddings and save to file
print("Generating embeddings and saving to file...")

with open(output_file, 'w', newline='', encoding='utf-8') as out_file:
    writer = csv.writer(out_file)
    writer.writerow(['conceptId', 'description', 'depth', 'embedding'])

    total_batches = len(dataloader)
    for batch_num, batch_data in enumerate(dataloader):
        batch_concept_ids, batch_descriptions, batch_depths, embeddings = embed_batch(batch_data)
        for concept_id, description, depth, embedding in zip(batch_concept_ids, batch_descriptions, batch_depths, embeddings):
            embedding_str = ','.join(map(str, embedding))
            writer.writerow([concept_id, description, depth, embedding_str])
        
        print(f"Processed batch {batch_num + 1}/{total_batches}")

print(f"Embeddings saved to {output_file}")