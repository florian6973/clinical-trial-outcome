import json
from tqdm import tqdm

file_bv = '../llm/outputs/outcomes_extracted_finetuned_bert-base-uncased_True.json'
# file_bv = '../llm/outputs/llm_infer_ministral-8b.json'
# file_bv = '../llm/outcomes_extracted_finetuned_bert-base-uncased_True.json'
# file = '../../outputs_True_json.json'
with open(file_bv, 'r') as f:
    data = json.load(f)


cats = {}
for item in tqdm(data):
    # print(item)
    # if "error" in item:
        # continue
    for key, lst in item[3].items():
        if key not in cats:
            cats[key] = []
        cats[key].extend([l.strip() for l in lst])

import pandas as pd

for key, val in cats.items():
    print("--------")
    # print(key)
    df = pd.DataFrame(val, columns=[key])

    # print(df)
    print(df.value_counts())
    df.value_counts().to_csv(f'outputs/stats_2_{key}.csv')