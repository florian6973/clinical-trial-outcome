# Joining AACT studies, condition mappings, and outcome mappings by NCTId

import pandas as pd

# Paths to your files
studies_file = './AACT/studies.txt'
condition_mapping_file = './outputs/condition_snomed_mapping.parquet'
outcomes_file = './outputs/outcomes.csv'

# Read the studies file
print("Reading studies file...")
studies_df = pd.read_csv(studies_file, sep='|')
print("Number of distinct nct_ids in studies_df:", studies_df['nct_id'].nunique())

# Display the columns (optional)
print("Columns in studies_df:", studies_df.columns)

# Select relevant columns
studies_df = studies_df[['nct_id', 'start_month_year']]

# Read the condition mapping file
print("\nReading condition mapping file...")
condition_mapping_df = pd.read_parquet(condition_mapping_file)
print("Number of distinct nct_ids in condition_mapping_df:", condition_mapping_df['nct_id'].nunique())

# Read the outcomes mapping file
print("\nReading outcomes mapping file...")
outcomes_df = pd.read_csv(outcomes_file)
print("Number of distinct nct_ids in outcomes_df:", outcomes_df['nct_id'].nunique())

# Merge studies and condition mappings
print("\nMerging studies and condition mappings...")
studies_conditions_df = pd.merge(
    studies_df,
    condition_mapping_df,
    on='nct_id',
    how='left'  # Use 'left' join to keep all studies
)

# Merge the above with outcomes mappings
print("Merging the previous result with outcomes mappings...")
final_df = pd.merge(
    studies_conditions_df,
    outcomes_df,
    on='nct_id',
    how='left'  # Use 'left' join to keep all studies
)

# Display the final DataFrame
print("\nFinal merged DataFrame:")
print(final_df.head())

# Filter so that 'object' is not NaN
final_df = final_df[final_df['object'].notna()]
final_df = final_df[final_df['object'] != 'NaN']

# Save the final DataFrame to a CSV file
output_file = './outputs/final_merged_data.csv'
final_df.to_csv(output_file, index=False)
print(f"Final merged data saved to {output_file}")