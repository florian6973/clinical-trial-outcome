rel_path = 'results_cls-0-2053.csv'

import pandas as pd

df = pd.read_csv(rel_path)

# Assuming 'df' is your DataFrame
sampled_df = df.sample(n=50)

# Display the sampled rows
print(sampled_df)
sampled_df.to_csv('test_set.csv')