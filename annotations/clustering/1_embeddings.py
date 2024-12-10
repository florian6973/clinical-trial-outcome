import os
# Set CUDA_VISIBLE_DEVICES before importing torch
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

from tqdm import tqdm
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
# api_key = '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'  # Your actual API key
# login(api_key)

# model_name = 'dunzhang/stella_en_400M_v5'

# # Step 2: Define directories
# cache_dir = "/gpfs/commons/groups/gursoy_lab/fpollet/models"
# os.makedirs(cache_dir, exist_ok=True)

# offload_folder = '/gpfs/commons/groups/gursoy_lab/fpollet/tmp/offload'  # Ensure this folder exists and has write permissions
# os.makedirs(offload_folder, exist_ok=True)

# # Step 3: Load the model with device_map='auto' without moving it to a single device
# print("Loading the model...")
# try:
#     # Define max_memory per device (optional but recommended)
#     max_memory = {i: '45000MB' for i in range(torch.cuda.device_count())}
#     max_memory['cpu'] = '48GB'  # Adjust based on your system's RAM

#     # Load tokenizer
#     tokenizer = AutoTokenizer.from_pretrained(
#         model_name,
#         cache_dir=cache_dir,
#         trust_remote_code=True
#     )

#     # Load model with automatic device mapping
#     model = AutoModel.from_pretrained(
#         model_name,
#         cache_dir=cache_dir,
#         trust_remote_code=True,
#         device_map='auto',           # Automatically distribute the model
#         max_memory=max_memory,
#         offload_folder=offload_folder,
#         torch_dtype=torch.float16    # Use half-precision
#     )
#     model.eval()  # Set model to evaluation mode
#     print("Model loaded successfully.")
# except Exception as e:
#     print(f"An error occurred while loading the model: {e}")
#     exit(1)

# Do not move the model to a specific device when using device_map='auto'
# Remove the following lines:
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model.to(device)

# Step 4: Get available devices (GPUs)
# available_devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
# print(f"Available devices: {available_devices}")

# Step 5: Define the path to your data file and output file
data_file = 'outcomes.csv'  # Adjust the path as needed

# Ensure the output directory exists
output_dir = './outputs'
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
df = pd.read_csv(data_file)
distinct_titles = set(df['object'].values.tolist())

print(f"Found {len(distinct_titles)} distinct titles.")
from sentence_transformers import SentenceTransformer, SimilarityFunction

# Convert set to list for indexing
titles_list = list(distinct_titles)

model = SentenceTransformer("dunzhang/stella_en_400M_v5", trust_remote_code=True,
                           ).cuda()
model.similarity_fn_name = SimilarityFunction.COSINE

# results = []
# for i, title in tqdm(enumerate(titles_list), total=len(titles_list)):
#     emb = model.encode(title)
#     results.append(emb)
#     # if i == 4:
#     #     break

results = []
batch_size = 50

for i in tqdm(range(0, len(titles_list), batch_size), total=(len(titles_list) + batch_size - 1) // batch_size):
    batch = titles_list[i : i + batch_size]
    batch_embeddings = model.encode(batch)
    results.extend(batch_embeddings)

import numpy as np
np.savez_compressed('embs.npz', np.array(results))
# np.savetxt("embs.txt", np.array(results))

