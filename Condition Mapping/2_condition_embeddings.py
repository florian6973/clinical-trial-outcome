# CONDITION NAME EMBEDDINGS

import os
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from huggingface_hub import login

print('Reading conditions.txt file')

# Read the conditions.txt file
conditions_df = pd.read_csv('./AACT/conditions.txt', sep='|')

print('Total conditions:', len(conditions_df))

# Remove duplicate condition names
unique_conditions_df = conditions_df[['name']].drop_duplicates().reset_index(drop=True)
print('Unique condition names:', len(unique_conditions_df))

# Now, we need to generate embeddings for the unique condition names

print('Setting up the model for embeddings')

# Step 1: Authenticate with Hugging Face API key
api_key = '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'  # Replace with your actual API key
login(api_key)

# Step 2: Define directories
cache_dir = "./cache"
os.makedirs(cache_dir, exist_ok=True)

offload_folder = './offload'  # Ensure this folder exists and has write permissions
os.makedirs(offload_folder, exist_ok=True)

# Step 3: Load the model with device_map='auto' without moving it to a single device
print("Loading the model...")
try:
    # Define max_memory per device (optional but recommended)
    max_memory = {i: '24000MB' for i in range(torch.cuda.device_count())}
    max_memory['cpu'] = '48GB'  # Adjust based on your system's RAM

    model_name = 'nvidia/NV-Embed-v2'

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

# Prepare the instruction
task_name_to_instruct = {"embedding": "Given a description, find the closest other descriptions"}
instruction = "Instruct: " + task_name_to_instruct["embedding"] + "\nDescription: "

# Determine appropriate max_length
print("Computing max token length...")
token_lengths = [
    len(tokenizer.encode(name, add_special_tokens=True)) + len(tokenizer.encode(instruction, add_special_tokens=False))
    for name in unique_conditions_df['name']
]
max_token_length = max(token_lengths)
print(f"Max token length in data: {max_token_length}")

# Set max_length slightly above the max token length to accommodate any variations
max_length = min(max_token_length + 20, 512)  # Ensure max_length is not excessively large

# Create a Dataset for unique condition names
class UniqueConditionsDataset(Dataset):
    def __init__(self, names):
        self.names = names

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        return self.names[idx]

dataset = UniqueConditionsDataset(
    unique_conditions_df['name'].tolist()
)

batch_size = 50  # Adjust based on your GPU memory and capacity

# Define a custom collate function
def custom_collate_fn(batch):
    return batch  # Returns a list of names

dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, collate_fn=custom_collate_fn)

def embed_batch(batch_names):
    # Use the model's encode method to get embeddings
    embeddings = model.encode(
        batch_names,
        instruction=instruction,
        max_length=max_length
    )
    
    # Normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)
    
    return batch_names, embeddings.cpu().numpy()

# Generate embeddings and collect data
print("Generating embeddings and collecting data...")

embedding_data = []

total_batches = len(dataloader)
for batch_num, batch_names in enumerate(dataloader):
    batch_names, embeddings = embed_batch(batch_names)
    for name, embedding in zip(batch_names, embeddings):
        embedding_data.append({
            'name': name,
            'Embedding': embedding.tolist()  # Convert numpy array to list
        })
    print(f"Processed batch {batch_num + 1}/{total_batches}")

# Create a DataFrame from the embedding data
embeddings_df = pd.DataFrame(embedding_data)
print('Embeddings generated for unique condition names.')

# Now, merge the embeddings back to the original conditions_df on 'name' to get the nct_ids
print('Merging embeddings back with conditions to associate NCT IDs.')
final_df = conditions_df.merge(embeddings_df, on='name', how='left')
# Save the embeddings to a Parquet file (embeddings as strings)
output_file = 'outputs/conditions_name_embeddings.parquet'

print('Saving embeddings to a Parquet file...')
final_df.to_parquet(output_file, index=False)
print(f'Embeddings saved to {output_file}')