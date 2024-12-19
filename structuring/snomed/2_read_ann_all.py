import pandas as pd
import numpy as np
embeddings = pd.read_parquet("/nlp/projects/llama/outputs/snomed_embeddings.parquet") # embedding

print(embeddings.columns)
print(embeddings.iloc[0])

# import numpy as np
# # mm = np.memmap("array-outcome.dat", mode='r', shape=(438026, 4096), dtype='float16')
# mm = np.memmap("label-outcome.dat", mode='r', shape=(438026,), dtype='U1000')
# mm = np.memmap("array-snomed.dat", mode='r', shape=(172582, 4096), dtype='float16')
# mm = np.memmap("label-snomed.dat", mode='r', shape=(172582,), dtype='U1000')
# print(mm[100])
# print(mm[0,0])