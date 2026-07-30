import pandas as pd

# # Convert CSV to Parquet
# csv_file = 'outputs/outcome_embeddings.csv'
# parquet_file = 'outputs/outcome_embeddings.parquet'

# print('Converting outcome embeddings to Parquet')
# df = pd.read_csv(csv_file)
# df.to_parquet(parquet_file)

# Convert Parquet to CSV
csv_file = 'outputs/snomed_embeddings.csv'
parquet_file = 'outputs/snomed_embeddings.parquet'

print('Converting SNOMED embeddings to Parquet')
df = pd.read_csv(csv_file)
df.to_parquet(parquet_file)
