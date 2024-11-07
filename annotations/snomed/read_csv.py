# import pandas as pd
# embeddings = pd.read_parquet('sample.parquet')
# print(len(embeddings['Embedding'].values[0].split(',')))

import pandas as pd
from tqdm import tqdm
import numpy as np

# array = np.memmap('array.npz').reshape(-1, 4096)
# print(array.shape)

# type_emb = 
type_emb = "outcome" # "outcome"
name_column = "Title" #'embedding' # "Embedding"
type_col = "label"
# embeddings = pd.read_parquet( "/nlp/projects/llama/outputs/outcome_embeddings.parquet")#'sample.parquet')
embeddings = pd.read_parquet(f"/nlp/projects/llama/outputs/{type_emb}_embeddings.parquet")#'sample.parquet')
# embeddings = pd.read_parquet('sample.parquet')

if type_col == "array":
    arrs = np.memmap(f"{type_col}-{type_emb}.dat", dtype='float16', mode='w+', shape=(len(embeddings), 4096))
elif type_col == "label":
    arrs = np.memmap(f"{type_col}-{type_emb}.dat", dtype='U1000', mode='w+', shape=(len(embeddings),))

for i in tqdm(range(len(embeddings))):
    if type_col == "array":
        arrs[i, :] = (embeddings[name_column].iloc[i].split(','))
    
    elif type_col == "label":
        text = embeddings[name_column].iloc[i]
        if len(text) > 1000:
            raise ValueError(f"too long string {len(text)}")
        arrs[i] = text
    if i % 10000 == 0:
        print("Flushing")
        arrs.flush()
arrs.flush()

# print(arrs[0, 0])

# print("Array")
# embs = np.array(arrs)
# print(embs.shape)
# np.savez('array-c.npz', embeddings=embs)
print("End")
#https://www.linkedin.com/pulse/using-numpymemmap-memory-mapped-file-storage-mohamed-riyaz-khan-7ihqc