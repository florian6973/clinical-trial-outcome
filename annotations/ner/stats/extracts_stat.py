import json
import pandas as pd
from tqdm import tqdm
from itertools import product

import sys
sys.path.append('../llm')  # Replace with the actual path

from read_outcomes import read_outcomes
import numpy as np


file = '../../outputs_True_json.json'
with open(file, 'r') as f:
    data = json.load(f)
file = '../../outputs_True.json'
with open(file, 'r') as f:
    data_raw = json.load(f)

rows = read_outcomes()
# print(rows.value_counts(sort=True, ascending=False))
# exit()

data_final = []
errs = 0
max_nb = 0
data_max = None
k_max = 0
lengths_distrib = []

final_format = []

print(len(data))
# print(rows.iloc[:5])
# input()

for k, d in tqdm(enumerate(data), total=len(data)):
    if "error" in d:
        errs += 1
        lengths_distrib.append(0)
        continue

    # remove duplicates
    # check match the right 
    d = {key: value for key, value in d.items() if value != []}
    # d = {key: value for key, value in d.items() if value != [] and key == 'Quantity or Object of Interest'}

    row_idx = data_raw[k].find("Text to find entities:")+23
    row_idx_end = data_raw[k].find("<|eot_id|>", row_idx)
    # print(d)
    # print(data_raw[k][row_idx:row_idx_end].strip())
    # input()
    row = data_raw[k][row_idx:row_idx_end].strip()

    final_format.append((row, d))

    tmp_ls = [dict(zip(d.keys(), values)) for values in product(*d.values())]

    if len(tmp_ls) >= 1:
        # print("Cleaning")
        # print("-----------")
        # print(len(tmp_ls))
        d_clean = {}
        original_text = row
        # print(original_text)
        # print(d)
        for col, vals in d.items():
            if col not in d_clean:
                d_clean[col] = []
            
            for val in vals:
                common_word = False
                try:
                    if isinstance(val, dict): # can happen, wrong format
                        print(val)
                    for word in val.split(' '):
                        if word in original_text:
                            common_word = True
                    if common_word:
                        if val not in d_clean[col]:
                            d_clean[col].append(val)
                except:
                    print("Failed to check")

        d_clean = {key: value for key, value in d_clean.items() if value != []}
        # print(d_clean)
        tmp_ls = [dict(zip(d_clean.keys(), values)) for values in product(*d_clean.values())]
        # print(len(tmp_ls))
        # input()





    lengths_distrib.append(len(tmp_ls))

    # clean hallucinations by removing all elements that are not in the original string

    if len(tmp_ls) > 100:
        continue
    if len(tmp_ls) > max_nb:
        data_max = d
        max_nb = len(tmp_ls)
        k_max = k
    data_final.extend(tmp_ls)

print(sorted(lengths_distrib, reverse=True)[:250])
print(np.where(np.array(lengths_distrib) == 32400)) # 42196

print(max_nb)
print(data_max)
print(errs)
# print(rows[k_max])
# print(rows[42196])
# print(data[42196])

df = pd.DataFrame(data_final)
# df.to_csv('outcomes-extracted.csv')
for col in df.columns:
    df[col] = df[col].apply(lambda x: str(x).lower())
print(df)

# df = df.lower
import os
os.makedirs('outputs_2', exist_ok=True)
for col in df.columns:
    # print(df[col].value_counts())
    df[col].value_counts().to_csv(f"outputs_2/stats_{col}.csv")

with open("outputs_2/outcomes.json", 'w') as f:
    json.dump(final_format, f, indent=4)
# df[col].value_counts().to_csv(f"outputs_2/outcomes.json")