file = r'/gpfs/commons/groups/gursoy_lab/fpollet/Git/clinical-trial-outcome/annotations/ner/llm/outputs/outcomes_extracted_finetuned_bert-base-uncased_True.json'

import json
from tqdm import tqdm

with open(file, 'r') as f:
    data = json.load(f)

import pandas as pd

outcomes = []
for outcome in tqdm(data):
    if "object" in outcome[3]:
        normalized_outcomes = outcome[3]["object"]
        no_id = outcome[1]
        nct_id = outcome[2]
        for normalized_outcome in normalized_outcomes:
            # print(no_id, normalized_outcome)
            if "##" in normalized_outcome:
                continue
            if normalized_outcome in ['change from baseline', 'change', 'mean change from baseline']:
                continue
            outcomes.append((no_id, nct_id, normalized_outcome))
            # break
    # break

pd.DataFrame(outcomes, columns=['id', 'nct_id', 'object']).to_csv("outcomes.csv")