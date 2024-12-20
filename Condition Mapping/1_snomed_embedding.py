# SNOMED ANALYSIS

# Information for working with the relevant SNOMED files

# Concept hierarchy: ./SNOMED/sct2_Relationship_Full_US1000124_20240901.txt
# id  effectiveTime  active  moduleId  sourceId  destinationId  relationshipGroup  typeId  characteristicTypeId  modifierId
# Key columns: active, sourceId, destinationId, typeId
# Should require active = 1
# typeId should be 'is a' to create tree: 116680003 | is a |
# Use to 1) find children or distant children concepts of Clinical Finding (destinationId = 404684003), 2) build hierarchy tree

# Term to concept: ./SNOMED/sct2_Description_Full-en_US1000124_20240901.txt
# id  effectiveTime  active  moduleId  conceptId  languageCode  typeId  term  caseSignificanceId
# Key columns: active, conceptId, term
# For definition type id for 900000000000003001 or 'Fully Specified Name'
# Should require active = 1
# Link term text to conceptIds

import os
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from huggingface_hub import login

print('Reading SNOMED files')
# Read the SNOMED files
relationship_df = pd.read_csv('./SNOMED/sct2_Relationship_Full_US1000124_20240901.txt', sep='\t')
definition_df = pd.read_csv('./SNOMED/sct2_Description_Full-en_US1000124_20240901.txt', sep='\t')

print('Filtering active relationships and definitions')
# Filter active relationships and definitions
active_relationships = relationship_df[
    (relationship_df['active'] == 1) & (relationship_df['typeId'] == 116680003)
]
active_definitions = definition_df[
    (definition_df['active'] == 1) & (definition_df['typeId'] == 900000000000003001)
]

print('Length of active relationships:', len(active_relationships))
print('Length of active definitions:', len(active_definitions))
print('Number of distinct conceptIds in active_definitions:', active_definitions['conceptId'].nunique())

# Create a mapping from conceptId to term
concept_to_term = dict(zip(active_definitions['conceptId'], active_definitions['term']))

# Your provided mapping of top-level codes (multiple codes for each category)
top_level_codes = {
    "Cardiology/Vascular Diseases": [106063007],
    "Dental and Oral Health": [423066003],
    "Dermatology": [106076001],
    "Endocrinology": [106176003, 362969004],
    "Gastroenterology": [386617003],
    "Genetic Disease": [409709004, 66091009, 782964007],
    "Hematology": [131148009, 414027002, 414022008, 362970003, 362971004],
    "Hepatology": [249565005],
    "Immunology": [414029004, 106182000],
    "Infections and Infectious Diseases": [40733004],
    "Metabolism and Nutrition": [75934005, 106089007, 2492009, 414916001, 238131007, 129861002, 415510005],
    "Miscellanea": [116223007, 362975008, 429054002, 737294004, 231532002, 118199002],
    "Neonatology": [414025005],
    "Nephrology": [249578005],
    "Neurology": [118940003, 102957003],
    "Obstetrics/Gynecology": [276477006, 248982007, 248842004, 129103003],
    "Occupational Diseases": [115966001],
    "Oncology": [55342001, 395557000],
    "Ophthalmology": [118235002],
    "Orthopedics": [106028002],
    "Otolaryngology": [118236001, 297268004],
    "Pain": [22253000],
    "Podiatry": [116316008],
    "Psychiatry/Psychology": [384821006, 74732009, 66214007],
    "Pulmonary/Respiratory Diseases": [106048009],
    "Rheumatology": [3723001, 85828009, 396332003, 203082005, 31996006, 31541009, 195295006, 90560007, 396230008, 5323003, 359789008],
    "Sleep": [106168000],
    "Trauma": [417746004],
    "Urology": [249230006, 106098005],
    "Toxicity": [75478009]
}

# Convert codes to integers for consistency
top_level_codes = {k: [int(v) for v in codes] for k, codes in top_level_codes.items()}

print('Finding descendants for each category')

# Function to find all descendants of a given conceptId
def get_descendants(concept_id, relationships_df):
    descendants = set()
    stack = [concept_id]
    while stack:
        parent = stack.pop()
        children = relationships_df[relationships_df['destinationId'] == parent]['sourceId'].tolist()
        for child in children:
            if child not in descendants:
                descendants.add(child)
                stack.append(child)
    return descendants

# Prepare a list to collect the data
data = []

# Iterate over each category
for category, code_list in top_level_codes.items():
    print(f"\nProcessing category: {category}")
    category_descendants = set()
    # For each code in the category, find descendants
    for top_code in code_list:
        print(f"  Processing top-level code: {top_code}")
        # Get descendants
        descendants = get_descendants(top_code, active_relationships)
        # If there are no descendants, include the top-level code itself
        if not descendants:
            descendants = set([top_code])
        else:
            # Include the top-level code in the descendants
            descendants.add(top_code)
        category_descendants.update(descendants)
    # For each descendant, get the term
    for desc_code in category_descendants:
        desc_term = concept_to_term.get(desc_code, "Unknown")
        data.append({
            'category': category,
            'top_level_codes': code_list,
            'descendant_term': desc_term,
            'descendant_code': desc_code
        })

# Create a DataFrame from the collected data
descendants_df = pd.DataFrame(data)
print('Total descendants found:', len(descendants_df))

# Remove duplicates if any
descendants_df = descendants_df.drop_duplicates(subset=['category', 'descendant_code'])
print('After removing duplicates:', len(descendants_df))

# Now, we need to generate embeddings for the descendant terms

print('Setting up the model for embeddings')

# Step 1: Authenticate with Hugging Face API key
api_key = 'TO FILL IN'  # Replace with your actual API key
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
# Compute max token length
print("Computing max token length...")
token_lengths = [
    len(tokenizer.encode(description, add_special_tokens=True)) + len(tokenizer.encode(instruction, add_special_tokens=False))
    for description in descendants_df['descendant_term']
]
max_token_length = max(token_lengths)
print(f"Max token length in data: {max_token_length}")

# Set max_length slightly above the max token length to accommodate any variations
max_length = min(max_token_length + 10, 512)  # Ensure max_length is not excessively large

# Create a Dataset for descriptions
class DescriptionsDataset(Dataset):
    def __init__(self, descriptions, concept_ids, categories):
        self.descriptions = descriptions
        self.concept_ids = concept_ids
        self.categories = categories

    def __len__(self):
        return len(self.descriptions)

    def __getitem__(self, idx):
        return self.descriptions[idx], self.concept_ids[idx], self.categories[idx]

dataset = DescriptionsDataset(
    descendants_df['descendant_term'].tolist(),
    descendants_df['descendant_code'].tolist(),
    descendants_df['category'].tolist()
)

batch_size = 50  # Adjust based on your GPU memory and capacity

# Define a custom collate function
def custom_collate_fn(batch):
    batch_descriptions, batch_concept_ids, batch_categories = zip(*batch)
    return list(batch_descriptions), list(batch_concept_ids), list(batch_categories)

dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, collate_fn=custom_collate_fn)

def embed_batch(batch_data):
    batch_descriptions, batch_concept_ids, batch_categories = batch_data
    # Use the model's encode method to get embeddings
    embeddings = model.encode(
        batch_descriptions,
        instruction=instruction,
        max_length=max_length
    )
    
    # Normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)
    
    return batch_concept_ids, batch_descriptions, batch_categories, embeddings.cpu().numpy()

# Generate embeddings and collect data
print("Generating embeddings and collecting data...")

output_data = []

total_batches = len(dataloader)
for batch_num, batch_data in enumerate(dataloader):
    batch_concept_ids, batch_descriptions, batch_categories, embeddings = embed_batch(batch_data)
    for concept_id, description, category, embedding in zip(batch_concept_ids, batch_descriptions, batch_categories, embeddings):
        output_data.append({
            'Category': category,
            'Descendant Term': description,
            'Embedding': embedding.tolist(),  # Convert numpy array to list
            'Concept ID': concept_id
        })
    print(f"Processed batch {batch_num + 1}/{total_batches}")

# Create a DataFrame from the output data
final_df = pd.DataFrame(output_data)

# Save the embeddings to a CSV file (embeddings as strings)
output_file = './outputs/top_level_descendants_embeddings.parquet'
# Convert embedding list to string
final_df.to_parquet(output_file, index=False)
print(f'Embeddings saved to {output_file}')
