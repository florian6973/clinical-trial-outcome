import os
# Set CUDA_VISIBLE_DEVICES before importing torch
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

import csv
import torch
from transformers import AutoModel, AutoTokenizer
from huggingface_hub import login
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# Disable tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

print(torch.cuda.is_available())  # Should return True if CUDA is available

# Step 1: Authenticate with Hugging Face API key
api_key = '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'  # Your actual API key
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

# Do not move the model to a specific device when using device_map='auto'
# Remove the following lines:
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model.to(device)

# Step 4: Get available devices (GPUs)
available_devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
print(f"Available devices: {available_devices}")

# Step 5: Define the path to your data file and output file
data_file = '/nlp/projects/llama/clinical_trials/AACT/outcomes.txt'  # Adjust the path as needed

# Ensure the output directory exists
output_dir = '/nlp/projects/llama/clinical_trials/outputs'
os.makedirs(output_dir, exist_ok=True)

# Define the output file path
output_file = os.path.join(output_dir, 'outcome_embeddings.csv')

# Step 6: Verify the data file exists
if not os.path.isfile(data_file):
    print(f"Data file not found at: {data_file}")
    exit(1)

# Step 7: Read the .txt file, extract distinct titles
print("Extracting distinct titles...")
distinct_titles = set()
with open(data_file, 'r') as file:
    reader = csv.DictReader(file, delimiter='|')
    for row in reader:
        distinct_titles.add(row['title'])

print(f"Found {len(distinct_titles)} distinct titles.")

# Convert set to list for indexing
titles_list = list(distinct_titles)

# Prepare the instruction
task_name_to_instruct = {"embedding": "Given a title, find the closest other titles"}
instruction = "Instruct: " + task_name_to_instruct["embedding"] + "\nTitle: "

# Determine appropriate max_length
# Compute max token length
token_lengths = [
    len(tokenizer.encode(title, add_special_tokens=True)) + len(tokenizer.encode(instruction, add_special_tokens=False))
    for title in titles_list
]
max_token_length = max(token_lengths)
print(f"Max token length in data: {max_token_length}")

# Set max_length slightly above the max token length to accommodate any variations
max_length = min(max_token_length + 20, 512)  # Ensure max_length is not excessively large

# Create a Dataset for titles
class TitlesDataset(Dataset):
    def __init__(self, titles):
        self.titles = titles

    def __len__(self):
        return len(self.titles)

    def __getitem__(self, idx):
        return self.titles[idx]

dataset = TitlesDataset(titles_list)
batch_size = 50  # Increase batch size if GPU memory allows
dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4)

def embed_batch(batch_titles):
    # Use the model's encode method to get embeddings
    embeddings = model.encode(
        batch_titles,
        instruction=instruction,
        max_length=max_length
    )
    
    # Normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)
    
    return embeddings.cpu().numpy()

# Step 8: Generate embeddings and save to file
print("Generating embeddings and saving to file...")

with open(output_file, 'w', newline='') as out_file:
    writer = csv.writer(out_file)
    writer.writerow(['Title', 'Embedding'])

    total_batches = len(dataloader)
    for batch_num, batch_titles in enumerate(dataloader):
        embeddings = embed_batch(batch_titles)
        for title, embedding in zip(batch_titles, embeddings):
            embedding_str = ','.join(map(str, embedding))
            writer.writerow([title, embedding_str])

        print(f"Processed batch {batch_num + 1}/{total_batches}")

print(f"Embeddings saved to {output_file}")