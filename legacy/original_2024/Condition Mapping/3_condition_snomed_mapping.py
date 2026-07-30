# CONDITION-SNOMED MAPPING

import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import pyarrow.parquet as pq
import ast
import faiss  # Add this import at the top of your script if not already present

# Check if CUDA is available and set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Paths to the embedding files
condition_embeddings_file = 'outputs/conditions_name_embeddings.csv'
snomed_embeddings_file = 'outputs/top_level_descendants_embeddings.parquet'
output_file = 'outputs/condition_snomed_mapping.parquet'

# Ensure the embeddings files exist
if not os.path.isfile(condition_embeddings_file):
    print(f"Condition embeddings file not found at: {condition_embeddings_file}")
    exit(1)

if not os.path.isfile(snomed_embeddings_file):
    print(f"SNOMED embeddings file not found at: {snomed_embeddings_file}")
    exit(1)

# Load the condition embeddings
print("Loading condition embeddings...")

try:
    parquet_file = pq.ParquetFile(condition_embeddings_file)
    # Read the file in batches
    batch_size = 100000  # Adjust this based on your memory capacity
    batches = []
    for batch in parquet_file.iter_batches(batch_size=batch_size):
        print(f"Processing batch {len(batches) + 1} of size {batch_size}")
        # Convert batch to DataFrame
        df_batch = batch.to_pandas()
        batches.append(df_batch)
    # Concatenate all batches
    condition_df = pd.concat(batches, ignore_index=True)
except Exception as e:
    print(f"An error occurred while reading the Parquet file: {e}")
    exit(1)

print('Converting condition embeddings to tensor...')
condition_embeddings = torch.stack([torch.tensor(emb) for emb in condition_df['Embedding']])

# Load the SNOMED embeddings
print("Loading SNOMED embeddings...")
snomed_df = pd.read_parquet(snomed_embeddings_file)



print('Converting SNOMED embeddings to tensor...')
snomed_embeddings = torch.stack([torch.tensor(emb) for emb in snomed_df['Embedding']])

# Move SNOMED embeddings to device
snomed_embeddings = snomed_embeddings.to(device)

# Since embeddings are already normalized, we don't need to normalize them again
condition_embeddings_norm = condition_embeddings  # Assuming they are normalized
snomed_embeddings_norm = snomed_embeddings

# Compute cosine similarities and find top matches using FAISS
print("Computing cosine similarities and finding top matches using FAISS...")

# Convert embeddings to NumPy arrays and move to CPU if necessary
condition_embeddings_np = condition_embeddings_norm.cpu().numpy().astype('float32')
snomed_embeddings_np = snomed_embeddings_norm.cpu().numpy().astype('float32')

# Build the FAISS index with IVFFlat for faster search
print('Building FAISS index...')
dimension = snomed_embeddings_np.shape[1]
nlist = min(4096, snomed_embeddings_np.shape[0] // 30)  # number of clusters
quantizer = faiss.IndexFlatIP(dimension)
index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)

# Need to train the index
print('Training index...')
index.train(snomed_embeddings_np)
index.add(snomed_embeddings_np)

# Set the number of clusters to probe (trade-off between speed and accuracy)
index.nprobe = 64  # Increase this for better accuracy, decrease for speed

print('Performing search...')
similarity_scores, indices = index.search(condition_embeddings_np, k=1)

# Prepare results
print("Preparing results...")
results = []
snomed_concept_ids = snomed_df['Concept ID'].values
snomed_top_level_terms = snomed_df['Category'].values

for i in tqdm(range(len(condition_df)), desc="Processing conditions"):
    nct_id = condition_df.iloc[i]['nct_id']
    condition_name = condition_df.iloc[i]['name']
    top_idx = indices[i][0]
    similarity = similarity_scores[i][0]
    snomed_concept_id = snomed_concept_ids[top_idx]
    snomed_top_level_term = snomed_top_level_terms[top_idx]
    result = {
        'nct_id': nct_id,
        'condition_name': condition_name,
        'snomed_concept_id': snomed_concept_id,
        'snomed_top_level_term': snomed_top_level_term,
        'similarity': similarity
    }
    results.append(result)

# Create a DataFrame from the results
results_df = pd.DataFrame(results)

# Save the results to a Parquet file
results_df.to_parquet(output_file, index=False)
print(f"Condition-SNOMED mapping saved to {output_file}")