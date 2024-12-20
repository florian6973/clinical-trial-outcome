import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer
from huggingface_hub import login
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# Authentication and model setup
api_key = 'TO FILL IN'
login(api_key)
model_name = 'nvidia/NV-Embed-v2'

# Directory setup
cache_dir = "/cache"
output_dir = '/outputs'
input_file = '/outcomes.json'
output_file = os.path.join(output_dir, 'outcome_objects_embeddings.parquet')
os.makedirs(cache_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

def load_and_process_json():
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Extract key and objects of interest
    processed_data = []
    for item in data:
        if len(item) >= 2:  # Ensure item has both text and dictionary
            text = item[0]
            metadata = item[1]
            objects = metadata.get('Quantity or Object of Interest', [])
            
            # Create a row for each object
            for obj in objects:
                processed_data.append({
                    'key': text,
                    'object': obj
                })
    
    # Convert to DataFrame and drop duplicates
    df = pd.DataFrame(processed_data)
    df = df.drop_duplicates()
    return df

class ObjectsDataset(Dataset):
    def __init__(self, objects):
        self.objects = objects

    def __len__(self):
        return len(self.objects)

    def __getitem__(self, idx):
        return self.objects[idx]

def setup_model():
    max_memory = {i: '24000MB' for i in range(torch.cuda.device_count())}
    max_memory['cpu'] = '48GB'

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        trust_remote_code=True
    )

    model = AutoModel.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        trust_remote_code=True,
        device_map='auto',
        max_memory=max_memory,
        torch_dtype=torch.float16
    )
    model.eval()
    return model, tokenizer

def embed_batch(model, batch_objects, instruction):
    embeddings = model.encode(
        batch_objects,
        instruction=instruction,
        max_length=512
    )
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings.cpu().numpy()

def main():
    print("Loading and processing JSON data...")
    df = load_and_process_json()
    print(f"Found {len(df)} unique object entries")

    print("Setting up model...")
    model, tokenizer = setup_model()
    
    # Create dataset and dataloader
    dataset = ObjectsDataset(df['object'].tolist())
    dataloader = DataLoader(dataset, batch_size=50, num_workers=4)

    instruction = "Instruct: Represent the medical concept for similarity comparison\nConcept: "

    print("Generating embeddings...")
    all_embeddings = []
    
    for batch_num, batch_objects in enumerate(dataloader):
        embeddings = embed_batch(model, batch_objects, instruction)
        all_embeddings.extend(embeddings)
        print(f"Processed batch {batch_num + 1}/{len(dataloader)}")

    # Convert embeddings to list of lists instead of string
    df['embedding'] = all_embeddings
    
    # Save to parquet instead of CSV
    output_file = os.path.join(output_dir, 'outcome_objects_embeddings.parquet')
    df.to_parquet(output_file, index=False)
    print(f"Embeddings saved to {output_file}")

if __name__ == "__main__":
    main()