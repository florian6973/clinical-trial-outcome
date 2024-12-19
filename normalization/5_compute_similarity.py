import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

# Check if CUDA is available and set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Paths to the embedding files
outcome_embeddings_file = 'outputs/outcome_embeddings.parquet'
snomed_embeddings_file = 'outputs/snomed_embeddings.parquet'
output_file = 'outputs/top5_snomed_matches.csv'

# Ensure the embeddings files exist
if not os.path.isfile(outcome_embeddings_file):
    print(f"Outcome embeddings file not found at: {outcome_embeddings_file}")
    exit(1)

if not os.path.isfile(snomed_embeddings_file):
    print(f"SNOMED embeddings file not found at: {snomed_embeddings_file}")
    exit(1)

# Load the outcome embeddings
print("Loading outcome embeddings...")
outcome_df = pd.read_parquet(outcome_embeddings_file)
outcome_titles = outcome_df['Title'].tolist()
print('Converting from string to list')
outcome_embeddings = outcome_df['Embedding'].apply(lambda x: np.fromstring(x, sep=',')).tolist()
print('Converting to tensor')
outcome_embeddings = torch.stack([torch.tensor(emb) for emb in outcome_embeddings])

# Load the SNOMED embeddings
print("Loading SNOMED embeddings...")
snomed_df = pd.read_parquet(snomed_embeddings_file)
print('Converting from string to list')
snomed_concept_ids = snomed_df['conceptId'].tolist()
snomed_descriptions = snomed_df['description'].tolist()
snomed_depths = snomed_df['depth'].tolist()
snomed_embeddings = snomed_df['embedding'].apply(lambda x: np.fromstring(x, sep=',')).tolist()
print('Converting to tensor')
snomed_embeddings = torch.stack([torch.tensor(emb) for emb in snomed_embeddings]).to(device)

# Normalize SNOMED embeddings once and keep on GPU
print("Normalizing SNOMED embeddings...")
snomed_embeddings_norm = torch.nn.functional.normalize(snomed_embeddings, p=2, dim=1)

# Normalize outcome embeddings in batches
def normalize_outcome_in_batches(embeddings, batch_size=1000):
    normalized_embeddings = []
    for i in tqdm(range(0, len(embeddings), batch_size), desc="Normalizing outcome batches"):
        batch = embeddings[i:i+batch_size].to(device)
        normalized_batch = torch.nn.functional.normalize(batch, p=2, dim=1)
        normalized_embeddings.append(normalized_batch.cpu())
        del batch, normalized_batch
        torch.cuda.empty_cache()
    return torch.cat(normalized_embeddings, dim=0)

print("Normalizing outcome embeddings...")
outcome_embeddings_norm = normalize_outcome_in_batches(outcome_embeddings, batch_size=1000)

# Compute cosine similarities
print("Computing cosine similarities...")

batch_size = 100  # Adjust based on your memory capacity
top_k = 5  # Number of top matches to retrieve
results = []

# Convert lists to NumPy arrays for indexing
snomed_concept_ids = np.array(snomed_concept_ids)
snomed_descriptions = np.array(snomed_descriptions)
snomed_depths = np.array(snomed_depths)

for i in tqdm(range(0, len(outcome_embeddings_norm), batch_size), desc="Processing outcome batches"):
    batch_outcome_emb = outcome_embeddings_norm[i:i+batch_size].to(device)
    similarities = torch.mm(batch_outcome_emb, snomed_embeddings_norm.T)
    topk_similarities, topk_indices = similarities.topk(k=top_k, dim=1)
    topk_similarities = topk_similarities.cpu().numpy()
    topk_indices = topk_indices.cpu().numpy()
    for j in range(batch_outcome_emb.size(0)):
        outcome_idx = i + j
        title = outcome_titles[outcome_idx]
        matches = []
        for k in range(top_k):
            snomed_idx = topk_indices[j, k]
            similarity = topk_similarities[j, k]
            concept_id = snomed_concept_ids[snomed_idx]
            description = snomed_descriptions[snomed_idx]
            depth = snomed_depths[snomed_idx]
            matches.append((concept_id, description, depth, similarity))
        result = {
            'Title': title,
            'Match1_ConceptId': matches[0][0],
            'Match1_Description': matches[0][1],
            'Match1_Depth': matches[0][2],
            'Match1_Similarity': matches[0][3],
            'Match2_ConceptId': matches[1][0],
            'Match2_Description': matches[1][1],
            'Match2_Depth': matches[1][2],
            'Match2_Similarity': matches[1][3],
            'Match3_ConceptId': matches[2][0],
            'Match3_Description': matches[2][1],
            'Match3_Depth': matches[2][2],
            'Match3_Similarity': matches[2][3],
            'Match4_ConceptId': matches[3][0],
            'Match4_Description': matches[3][1],
            'Match4_Depth': matches[3][2],
            'Match4_Similarity': matches[3][3],
            'Match5_ConceptId': matches[4][0],
            'Match5_Description': matches[4][1],
            'Match5_Depth': matches[4][2],
            'Match5_Similarity': matches[4][3],
        }
        results.append(result)
    del batch_outcome_emb, similarities, topk_similarities, topk_indices
    torch.cuda.empty_cache()

# Create a DataFrame from the results and save it
results_df = pd.DataFrame(results)
results_df.to_csv(output_file, index=False)
print(f"Top 5 SNOMED matches for each outcome saved to {output_file}")