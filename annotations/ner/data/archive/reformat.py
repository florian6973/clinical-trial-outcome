path_1 = r'/gpfs/commons/groups/gursoy_lab/fpollet/Git/clinical-trial-outcome/annotations/ner/data/manual-ann-ner-0_100.txt'
path_2 = r'/gpfs/commons/groups/gursoy_lab/fpollet/Git/clinical-trial-outcome/annotations/ner/data/train_raw.csv'

import pandas as pd

df_1 = pd.read_csv(path_1, delimiter='|', encoding='utf-8', header=None)
df_1.columns = ['title']
df_2 = pd.read_csv(path_2, index_col=0).iloc[:150]

print(df_1)
print(df_2)

df = pd.concat([df_1, df_2], ignore_index=True)
print(df)
df.to_csv("/gpfs/commons/groups/gursoy_lab/fpollet/Git/clinical-trial-outcome/annotations/ner/data/manual-ann-ner-150.csv")


import yaml

import yaml

# Load the first YAML file
with open('/gpfs/commons/groups/gursoy_lab/fpollet/Git/clinical-trial-outcome/annotations/ner/data/manual-ann-ner.yaml', 'r') as file:
    data1 = yaml.safe_load(file)

# Load the second YAML file
with open('/gpfs/commons/groups/gursoy_lab/fpollet/Git/clinical-trial-outcome/annotations/ner/data/manual-ann-ner-extended.yaml', 'r') as file:
    data2 = yaml.safe_load(file)

# Extract annotations from both files
annotations1 = data1.get('annotations', {})
annotations2 = data2.get('annotations', {})

# Combine the two annotations, updating the indices
combined_annotations = {}
index = 0

# Add entries from the first file
for key in sorted(annotations1.keys(), key=int):
    combined_annotations[index] = annotations1[key]
    index += 1

# Add entries from the second file
for key in sorted(annotations2.keys(), key=int):
    combined_annotations[index] = annotations2[key]
    index += 1

# Combine everything into the final structure
combined_data = {'annotations': combined_annotations}

# Save the result to a new YAML file
with open('/gpfs/commons/groups/gursoy_lab/fpollet/Git/clinical-trial-outcome/annotations/ner/data/manual-ann-ner-all.yaml', 'w') as file:
    yaml.dump(combined_data, file, sort_keys=False)

print("YAML files have been combined and saved as 'combined.yaml'")