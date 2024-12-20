import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

# Define the outputs directory
outputs_dir = './outputs'

# Load the embeddings
print("Loading embeddings...")
df = pd.read_parquet(os.path.join(outputs_dir, 'outcome_objects_embeddings.parquet'))

# Separate embeddings and objects
embedding_cols = [col for col in df.columns if col.startswith('e')]
embeddings = df[embedding_cols].values
objects = df['Object'].values

# Load or initialize group embeddings
group_embeddings_path = os.path.join(outputs_dir, 'group_embeddings.parquet')
starting_groups = [
    'Acceptability',
    'Adverse Events (AEs)',
    'Antibody Titers',
    'Blood Pressure',
    'Body Weight',
    'Clinical Response',
    'Concentration',
    'Dermatological Condition',
    'Disease Activity',
    'Disease-Free Survival',
    'Dose',
    'ECG measurement',
    'Efficacy',
    'Event Occurrence',
    'Feasibility',
    'Forced Expiratory Volume in One Second (FEV1)',
    'Glucose',
    'Heart Rate',
    'Hemoglobin',
    'HRQoL',
    'Immunogenicity',
    'Improvement',
    'Lesions',
    'Lymphocyte Count',
    'Metabolities',
    'Mortality',
    'NAO',
    'Overall Survival (OS)',
    'Overall/Objective Response Rate (ORR/OR)',
    'Pain',
    'PASI Score',
    'Pharmacokinetics',
    'Platelet Count',
    'Progression-free Survival (PFS)',
    'QOL',
    'Radiation Measurement',
    'Remission',
    'Responders',
    'Safety',
    'Satisfaction',
    'Seizure Frequency',
    'Sensitivity',
    'Serious Adverse Events (SAEs)',
    'Seroconversion',
    'Severity',
    'Symptoms',
    'Time to Maximum',
    'Time to Progression (TTP)',
    'Tolerability',
    'Tumor Response',
    'Vital Signs',
    'Visual Acuity',
    'Viral Load'
]

if os.path.exists(group_embeddings_path):
    print("Loading group embeddings...")
    group_df = pd.read_parquet(group_embeddings_path)
    group_embedding_cols = [col for col in group_df.columns if col.startswith('e')]
    group_embeddings = group_df[group_embedding_cols].values
    group_names = group_df['Group'].values
else:
    print("Group embeddings not found. Initializing with starting examples.")
    # Initialize group_df with starting groups
    group_df = pd.DataFrame({'Group': starting_groups})
    group_names = group_df['Group'].values

# Load the embedding model using SentenceTransformer (from embed_outcome_objects.py)
print("Loading embedding model...")
model_name = 'nvidia/NV-Embed-v2'  # Same model as in embed_outcome_objects.py
embedding_model = SentenceTransformer(
    model_name,
    trust_remote_code=True
).cuda()

# Function to compute embeddings
def compute_embeddings(texts):
    """Compute embeddings for a list of texts using SentenceTransformer."""
    embeddings = embedding_model.encode(
        texts,
        show_progress_bar=False,
        truncation=True
    )
    return embeddings

# If group embeddings don't exist, compute them for starting groups
if not os.path.exists(group_embeddings_path):
    # Compute embeddings for starting groups
    group_embeddings = compute_embeddings(starting_groups)
    # Add embeddings to group_df
    embedding_dim = group_embeddings.shape[1]
    for idx, col in enumerate([f'e{i}' for i in range(embedding_dim)]):
        group_df[col] = group_embeddings[:, idx]
    # Save initial group embeddings
    group_df.to_parquet(group_embeddings_path, index=False)

# Ensure group_embeddings and group_names are up-to-date
group_embedding_cols = [col for col in group_df.columns if col.startswith('e')]
group_embeddings = group_df[group_embedding_cols].values
group_names = group_df['Group'].values

# Load Qwen2.5-32B-Instruct model
print("Loading Qwen2.5-32B-Instruct model...")
model_name = "Qwen/Qwen2.5-32B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.eos_token = tokenizer.eos_token or '</s>'
eos_token = tokenizer.eos_token

def find_similar_terms(query_embedding, k=5):
    """Find k most similar terms using cosine similarity."""
    similarities = cosine_similarity([query_embedding], embeddings)[0]
    top_indices = np.argsort(similarities)[-k:][::-1]
    return [(objects[idx], similarities[idx]) for idx in top_indices]

def find_similar_groups(query_embedding, k=5):
    """Find k most similar groups using cosine similarity."""
    if group_embeddings is None or len(group_embeddings) == 0:
        return []
    similarities = cosine_similarity([query_embedding], group_embeddings)[0]
    top_indices = np.argsort(similarities)[-k:][::-1]
    return [(group_names[idx], similarities[idx]) for idx in top_indices]

def get_qwen_response_batch(prompts, system_message="You are a helpful and harmless assistant. You should output the correct answer."):
    """Get responses from the model for a batch of prompts."""
    messages_batch = []
    for prompt in prompts:
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        messages_batch.append(text)

    # Tokenize and move to device
    model_inputs = tokenizer(messages_batch, return_tensors="pt", padding=True).to(model.device)

    # Use torch.amp.autocast for mixed precision
    with torch.amp.autocast(device_type='cuda'):
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=50,              # Reduced max new tokens
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,                # Disable sampling
            num_beams=1,                    # Use greedy decoding
            use_cache=True,                 # Enable caching
            top_p=None,                     # Unset top_p to eliminate warning
            top_k=None,                     # Unset top_k to eliminate warning
            temperature=None
        )

    # Remove the input tokens to get only the generated tokens
    generated_ids = generated_ids[:, model_inputs['input_ids'].shape[1]:]

    responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return responses

# Create the categorization prompt
def create_prompt(term, similar_terms, similar_groups):
    term_type = 'Clinical Trial Outcome Object'
    prompt = f"""Task: Categorize the following {term_type} into one of the standardized groups listed below or create a new one if none are suitable.

IMPORTANT: Respond ONLY with 'GROUP: [group name]' and nothing else.

Main term: {term}

Similar terms for context:
{', '.join([f"{t[0]} ({t[1]:.2f})" for t in similar_terms])}

"""

    if similar_groups:
        prompt += f"""Standardized groups:
{', '.join([f"{g[0]} ({g[1]:.2f})" for g in similar_groups])}

"""
    else:
        prompt += "No standardized groups available, so create a new group for the outcome.\n\n"

    prompt += """Provide your answer in this format (exactly as shown, with no explanation):
GROUP: [group name]"""

    # Append the EOS token to the prompt
    prompt += tokenizer.eos_token if tokenizer.eos_token else '</s>'
    return prompt

# Process terms in batches
batch_size = 5
results = []


print("Processing terms in batches...")

for start_idx in tqdm(range(0, len(df), batch_size)):
    end_idx = min(start_idx + batch_size, len(df))
    batch_rows = df.iloc[start_idx:end_idx]
    batch_terms = batch_rows['Object'].values
    batch_embeddings = embeddings[start_idx:end_idx]

    batch_similar_terms = []
    batch_similar_groups = []
    for embedding in batch_embeddings:
        similar_terms = find_similar_terms(embedding)
        batch_similar_terms.append(similar_terms)

        similar_groups = find_similar_groups(embedding)
        batch_similar_groups.append(similar_groups)

    # Create batch prompts
    batch_prompts = []
    for term, similar_terms, similar_groups in zip(batch_terms, batch_similar_terms, batch_similar_groups):
        prompt = create_prompt(term, similar_terms, similar_groups)
        batch_prompts.append(prompt)

    # Get responses for the batch
    batch_responses = get_qwen_response_batch(batch_prompts)

    # Save results and update group embeddings if new groups are found
    new_groups_list = []
    for term, embedding, similar_terms, similar_groups, prompt, response in zip(batch_terms, batch_embeddings, batch_similar_terms, batch_similar_groups, batch_prompts, batch_responses):
        results.append({
            'term': term,
            'prompt': prompt,  # Save the prompt
            'similar_terms': similar_terms,
            'standardized_groups': similar_groups,
            'qwen_response': response
        })

        # Extract the group from the response
        group_name = None
        lines = response.strip().split('\n')
        for line in lines:
            if line.startswith('GROUP:'):
                group_name = line[len('GROUP:'):].strip()
                break

        if group_name and (group_names is None or group_name not in group_names):
            # Add the new group
            print(f"Adding new group '{group_name}'")
            # Compute embedding for the new group using embedding_model
            group_embedding = compute_embeddings([group_name])[0]
            # Prepare new group data
            new_group_data = {'Group': group_name}
            for idx, value in enumerate(group_embedding):
                new_group_data[f'e{idx}'] = value
            new_groups_list.append(new_group_data)
            # Update group_names immediately to avoid duplicates
            group_names = np.append(group_names, group_name)

    # After processing the batch, add all new groups
    if new_groups_list:
        new_groups_df = pd.DataFrame(new_groups_list)
        group_df = pd.concat([group_df, new_groups_df], ignore_index=True)
        group_embeddings = group_df[[col for col in group_df.columns if col.startswith('e')]].values

    # Save intermediate results every 10 batches
    if (start_idx // batch_size + 1) % 10 == 0:
        print(f"Saving interim results for batch {start_idx // batch_size + 1}")
        interim_df = pd.DataFrame(results)
        interim_df.to_csv(os.path.join(outputs_dir, f'outcome_categories_interim_{start_idx+batch_size}.csv'), index=False)
        # Save updated group embeddings
        group_df.to_parquet(group_embeddings_path, index=False)

# Save final results
final_df = pd.DataFrame(results)
final_df.to_parquet(os.path.join(outputs_dir, 'outcome_categories.parquet'), index=False)
# Save final group embeddings
group_df.to_parquet(group_embeddings_path, index=False)
print("Processing complete!")
