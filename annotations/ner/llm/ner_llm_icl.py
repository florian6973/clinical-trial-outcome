
from llm import load_model_processor, build_pipeline, build_pipeline_batch

# https://arxiv.org/pdf/2304.10428

import numpy as np
import json
from tqdm import tqdm
import pandas as pd
import sys
import gc
import torch


mapping = {
    'object': 'Quantity or object of Interest'.lower(),
    'specifier': 'Additional constraints'.lower(),
    'measure': 'Quantity Measure'.lower(), #'Quantity Specifier',
    'time': 'Time'.lower(),
    'unit': 'Quantity unit'.lower(),
    'range': 'Quantity range'.lower()
}

def convert_yaml_to_json(row_gt):
    gt_tmp = {
        "Time".lower(): [],
        "Quantity unit".lower(): [],
        # "Quantity Specifier": [],
        "Quantity Measure".lower(): [],
        "Quantity or object of Interest".lower(): [],
        "Additional constraints".lower(): [],
        "Quantity range".lower(): []
    }
    for item in row_gt['structured']:
        for key, val in item.items():
            if "norm" in key:
                continue
            if isinstance(val, str):
                gt_tmp[mapping[key]].append(val.lower())
            else:
                gt_tmp[mapping[key]].extend([v.lower() for v in val])
    return gt_tmp

# rows = np.load("data-ann.npz", allow_pickle=True)['rows']
with open('../data/manual-ann-ner-0_100.txt', 'r') as f:
    rows = pd.Series(f.readlines())

# print(rows)


examples = pd.read_csv("../data/manual-ann-ner-100_200.csv", index_col=0)['title']
# print(examples)

import yaml
with open('../data/manual-ann-ner-extended.yaml', 'r') as file:
    # Step 2: Load the contents of the file
    data_gt = yaml.safe_load(file)['annotations']


all_mode = False
if len(sys.argv) > 1:
    if sys.argv[1] == 'all':
        all_mode = True
        outcomes = pd.read_csv('outcomes.txt', sep='|')
        outcomes['length'] = outcomes['title'].apply(lambda x: len(x))
        outcomes.sort_values('length', ascending=False, inplace=True)
        rows = outcomes['title']
        print(len(rows))
        rows = rows.apply(lambda x: x.lower())
        rows = rows.drop_duplicates()
        rows = rows.reset_index(drop=True)
        print(len(rows))
        print(rows)


llm = load_model_processor("cuda:0")
pipeline = build_pipeline(*llm)
pipeline_batch = build_pipeline_batch(*llm)

system_msg = "You are a clinical expert in named entity extraction."

outputs = []
outputs_json = []
batch_size = 1
# nb_examples = 5

import sys
if len(sys.argv) > 1:
    nb_examples = int(sys.argv[1])
else:
    nb_examples = 10

# prompt = """Extract following entities, if it exists in following text input (outcomes):
#              - Time: when
#              - Quantity Unit: measurement unit
#              - Quantity Measure (examples: Percentage of Participants, Number of Participants, Mean Change from Baseline, CHange...)
#              - Quantity or object of Interest (examples: Toxicity, Survival, specific element): the main concept(s) describing the outcome
#              - Additional constraints: details that are not measure or quantity/object of interest
#              - Quantity range: range for measurements
#              Do not duplicate elements between categories. Do not hallucinate words or introduce new concepts. Make sure to separate concepts of the same type.
#              Additional Constraints should be used only if it does not fit in Time or Range.
#              Avoid abbreviations.
#              Provide the output in JSON form like 
#              {
#   "time": [],
#   "quantity unit": [],
#   "quantity measure": [],
#   "quantity or object of Interest": [],
#   "additional constraints": [],
#   "quantity range": []
# }
# Examples:""" 
prompt = """Follow the examples to extract the main components of the given outcome, with the same JSON format:
Examples:"""

for i in range(nb_examples):
    print(examples[i])
    print(convert_yaml_to_json(data_gt[i]))
    ex = str(convert_yaml_to_json(data_gt[i])).replace("'", '"')
    
    prompt += f'\n\nOutcome: {examples[i]}\nOutput: {ex}'

prompt += '\n\nOutcome to process: '

def save():
    with open(f'outputs/outputs_ICL_{all_mode}_{nb_examples}.json', 'w') as f:
        json.dump(outputs, f)
        
    with open(f'outputs/outputs_ICL_{all_mode}_{nb_examples}_json.json', 'w') as f:
        json.dump(outputs_json, f)

# for i, row in tqdm(enumerate(rows), total=len(rows)):
for i in tqdm(range(0, len(rows), batch_size)):
    txts = []
    for k in range(i, i+batch_size):
        row = rows.iloc[k]
        if not all_mode:
            print(k)
            print("------")
            print(row)
            txt = row
        else:
            txt = row
        txts.append(txt)
    


    # res = pipeline(
    #     system_msg,
    #     prompt + txt) #  Make sure to associate everything to a category.

    reses = pipeline_batch(
        [(system_msg, prompt + txt + '\nOutput: ') for txt in txts]
    )

    # print(prompt + txt + '\nOutput: ')

    for res in reses:
        # print(res[res.find('<|end_header_id|>'):])
        try:
            # print(txt)
            txt = res[res.find('assistant<|end_header_id|>')+26:-10].strip()
            txt = txt.replace('<|start_header_id|>', '')
            print("BEGIN", txt, "END")
            res_json = json.loads(txt)
            outputs_json.append(res_json)
        except Exception as e:
            print(e)
            print("Could not parse")
            outputs_json.append({"error": True})
        # input()
        outputs.append(res)
        # break

    if i % 100 == 0:
        save()

    # gc.collect()
    # torch.cuda.empty_cache()

save()