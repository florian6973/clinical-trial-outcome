import csv
import torch
from transformers import MllamaForConditionalGeneration, AutoProcessor
from huggingface_hub import login
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor

print(torch.cuda.is_available())  # Should return True if CUDA is available

# Step 1: Authenticate with Hugging Face API key
api_key = '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'  # Your actual API key
login(api_key)

model_name = 'meta-llama/Llama-3.2-11B-Vision-Instruct'

# Step 2: Define directories
cache_dir = "/nlp/projects/llama/checkpoints/Llama-3.2-11B-Vision-Instruct"
os.makedirs(cache_dir, exist_ok=True)

offload_folder = '/tmp/offload'  # Ensure this folder exists and has write permissions
os.makedirs(offload_folder, exist_ok=True)

# Step 3: Define maximum memory per GPU
# Each RTX 3090 has 24GB of VRAM. Allocating 23GB to leave some overhead.
max_memory = {i: '24500MB' for i in range(torch.cuda.device_count())}

# Step 4: Load the model across GPUs using from_pretrained
print("Loading the model and processor across GPUs...")
try:
    # Load model and distribute across GPUs using device_map='auto'
    model = MllamaForConditionalGeneration.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        torch_dtype=torch.float16,    # Use float16 for memory efficiency on GPUs
        device_map="auto",            # Automatically distribute model across available GPUs
        offload_folder=offload_folder # Offload layers if necessary
    )

    # Load processor
    processor = AutoProcessor.from_pretrained(model_name, cache_dir=cache_dir)

    print("Model and processor loaded successfully across GPUs.")
except Exception as e:
    print(f"An error occurred while loading the model: {e}")
    exit(1)

def embed_titles_batch(titles, device):
    """
    Generates embeddings for a batch of titles using the loaded model on a specific device.
    
    Args:
        titles (list): A list of title strings.
        device (str): The GPU device to use for this batch (e.g., 'cuda:0').
    
    Returns:
        list: A list of embedding vectors, one for each title.
    """
    # Prepare the messages for batch processing
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": title}
            ]
        } for title in titles
    ]

    # Tokenize the input texts for the batch
    try:
        input_texts = [processor.apply_chat_template([msg], add_generation_prompt=False) for msg in messages]
    except AttributeError:
        input_texts = titles  # If apply_chat_template is unavailable, use simple text

    # Process inputs in a batch, move them to the specified GPU
    inputs = processor(
        None,  # No image
        input_texts,
        add_special_tokens=False,
        return_tensors="pt",
        padding=True,  # Pad to the longest sequence in the batch
        truncation=True
    ).to(device)  # Move to the specified GPU

    with torch.no_grad():
        try:
            outputs = model(**inputs, output_hidden_states=True, return_dict=True)
        except Exception as e:
            print(f"An error occurred during model inference: {e}")
            return None

    # Extract the last hidden state and compute the mean for each title
    hidden_states = outputs.hidden_states[-1]
    embeddings = hidden_states.mean(dim=1).to(torch.float32).cpu().numpy()
    
    return embeddings

# Function to process and write a batch of embeddings
def process_and_write_batch(batch_titles, device, writer, batch_idx, total_batches):
    embeddings = embed_titles_batch(batch_titles, device)
    if embeddings is not None:
        for title, embedding in zip(batch_titles, embeddings):
            embedding_str = ','.join(map(str, embedding))
            writer.writerow([title, embedding_str])
    # Print progress after processing each batch
    print(f"Processed batch {batch_idx}/{total_batches} ({batch_idx * len(batch_titles)} titles completed)")

# Parallel batch processing function
def process_batches_in_parallel(titles_list, batch_size, num_gpus, writer):
    total_batches = (len(titles_list) + batch_size - 1) // batch_size  # Calculate total number of batches
    with ThreadPoolExecutor(max_workers=num_gpus) as executor:
        futures = []
        for i in range(0, len(titles_list), batch_size):
            batch_idx = i // batch_size + 1  # Track the current batch number
            batch_titles = titles_list[i:i + batch_size]
            device = f'cuda:{i % num_gpus}'  # Alternate between GPUs
            futures.append(executor.submit(process_and_write_batch, batch_titles, device, writer, batch_idx, total_batches))

        for future in futures:
            future.result()  # This will raise an exception if one occurs


# Step 5: Define the path to your data file and output file
data_file = '/nlp/projects/llama/clinical_trials/AACT/outcomes.txt'  # Adjust the path as needed
output_file = '/nlp/projects/llama/clinical_trials/outputs/outcome_embeddings.csv'

# Step 6: Verify the data file exists
if not os.path.isfile(data_file):
    print(f"Data file not found at: {data_file}")
    exit(1)

# Step 7: Read the .txt file, extract distinct titles, and process each row, saving embeddings to output_file
print("Extracting distinct titles and generating embeddings...")

# Extract distinct titles
distinct_titles = set()
with open(data_file, 'r') as file:
    reader = csv.DictReader(file, delimiter='|')
    for row in reader:
        distinct_titles.add(row['title'])

print(f"Found {len(distinct_titles)} distinct titles.")

# Now process each distinct title in batches and track progress
batch_size = 50  # Adjust based on memory capacity
titles_list = list(distinct_titles)
num_gpus = torch.cuda.device_count()

with open(output_file, 'w', newline='') as out_file:
    writer = csv.writer(out_file)
    
    # Write headers for output file
    writer.writerow(['Title', 'Embedding'])

    # Process batches in parallel across GPUs
    process_batches_in_parallel(titles_list, batch_size, num_gpus, writer)

print(f"Embeddings saved to {output_file}")
