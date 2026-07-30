import numpy as np

# hallucination?
# duplicate NER
# https://medium.com/@ameydhote007/fine-tuning-language-models-for-ner-a-hands-on-step-by-step-guide-408cfee1e93b

import yaml
import pandas as pd



    

# Step 1: Open the YAML file
with open('../data/manual-ann-ner.yaml', 'r') as file:
    # Step 2: Load the contents of the file
    data_gt = yaml.safe_load(file)['annotations']

# Step 3: Access the data
# print(data_gt)
# input()

import sys
if len(sys.argv) > 1:
    nb_ex = int(sys.argv[1])
else:
    nb_ex = 10

import json
with open(f'../llm/outputs/outputs_ICL_False_{nb_ex}_json.json', 'r') as file:
    data_pred = json.load(file)

failed = 0
successed = 0
# print(data_pred)
data_pred_lw = []
for elt in data_pred:
    dico = {}

    # print(elt)
    if "error" in elt:
        dico = {
            'Quantity or object of Interest'.lower(): [],
            'Additional constraints'.lower(): [],
            'Quantity Measure'.lower(): [],
            'Time'.lower(): [],
            'Quantity unit'.lower(): [],
            'Quantity range'.lower(): [],
            # 'error': ['True']
        }
        failed += 1
        # dico = {"error":[]}
        data_pred_lw.append(dico)
        continue
    successed += 1
    for key, vals in elt.items():
        # dico[key.lower()] = [v.lower() for v in vals]
        # dico[key.lower()] = [v.lower() for v in vals]
        dico[key.lower()] = []
        # print(key,vals)
        for v in vals:
            if v.lower() not in dico[key.lower()]:
                dico[key.lower()].append(v.lower())

    data_pred_lw.append(dico)

data_pred = data_pred_lw

# llm_evaluate_llama-8b_{'only_object': False, 'example_selection': True, 'n_examples': 8}

# print(data_pred)
# exit()

# rows = np.load("data-ann.npz", allow_pickle=True)['rows']
# rows = np.load("data-ann.npz", allow_pickle=True)['rows']
with open('../data/manual-ann-ner-0_100.txt', 'r') as f:
    rows = pd.Series(f.readlines())

mapping = {
    'object': 'Quantity or object of Interest'.lower(),
    'specifier': 'Additional constraints'.lower(),
    'measure': 'Quantity Measure'.lower(), #'Quantity Specifier',
    'time': 'Time'.lower(),
    'unit': 'Quantity unit'.lower(),
    'range': 'Quantity range'.lower()
}

# duplicates as well

hallucinations = 0 # need initial string
exacts = {}
partials = {}
exacts_tot = {}
length = 0
counts = 0

multi_outcomes_gt = 0
multi_outcomes = 0

# https://www.davidsbatista.net/blog/2018/05/09/Named_Entity_Evaluation/
for i in range(len(data_gt)):
    # if "error" in data_pred[i]:
    #     failed += 1
    # else:
   
    gt_tmp = {
        "Time".lower(): [],
        "Quantity unit".lower(): [],
        # "Quantity Specifier": [],
        "Quantity Measure".lower(): [],
        "Quantity or object of Interest".lower(): [],
        "Additional constraints".lower(): [],
        "Quantity range".lower(): []
    }
    for item in data_gt[i]['structured']:
        for key, val in item.items():
            if "norm" in key:
                continue
            if mapping[key] not in exacts_tot:
                exacts_tot[mapping[key]] = 0
            if isinstance(val, str):
                gt_tmp[mapping[key]].append(val.lower())
                exacts_tot[mapping[key]] += 1
            else:
                gt_tmp[mapping[key]].extend([v.lower() for v in val])
                exacts_tot[mapping[key]] += len(val)

    multi = False
    for key, arr_val in gt_tmp.items():
        # print(arr_val)
        if len(arr_val) > 1:
            multi = True
    if multi:
        multi_outcomes_gt += 1

    multi = False
    for key, arr_val in data_pred[i].items():
        if len(arr_val) > 1:
            multi = True
    if multi:
        multi_outcomes += 1

    
    for key, arr_val in gt_tmp.items():
        for val in arr_val:
            partial = False
            for word in [x for sentence in data_pred[i][key] for x in sentence.split(' ')]:
                for word_gt in val.split(' '):
                    if word == word_gt:
                        partial = True
            if partial:
                if key not in partials:
                    partials[key] = 0
                partials[key] += 1

    txt = rows[i].lower()
    hallucinate = False
    for key, arr_val in data_pred[i].items():
        counts += len(arr_val)
        for val in arr_val:
            for word in val.split(' '):
                if word not in txt:
                    hallucinate = True
                    print(word, "-->", txt)
                    # input()
            if val in gt_tmp[key]:
                if key not in exacts:
                    exacts[key] = 0
                exacts[key] += 1

            # partial = False
            # for word in val.split(' '):
            #     for word_gt in [x for sentence in gt_tmp[key] for x in sentence.split(' ')]:
            #         if word == word_gt:
            #             partial = True
            # if partial:
            #     if key not in partials:
            #         partials[key] = 0
            #     partials[key] += 1

            #     if key == 'Additional Constraints':
            #         print(gt_tmp)
            #         print(data_pred[i])
            #         input()
    if hallucinate:
        hallucinations += 1

    length += len(txt)
        # input()

    # print(
    #     data_gt[i]['structured']
    # )
    # print(gt_tmp)
    # print(
    #     data_pred[i]
    # )

    # print("Exacts", exacts)
    # print("Exacts", exacts_tot)
    # input()

print(successed, failed)
print()
print("Hallucinations", hallucinations)
print("Average outcome length", length/len(data_gt))
print()
print("Multiple outcomes", multi_outcomes)
print("Multiple outcomes GT", multi_outcomes_gt)
print("Average # of elements", counts/len(data_gt))
print("Average # of elements GT", sum(exacts_tot.values())/len(data_gt))

for val in mapping.values():
    print()

    if val not in exacts:
        exacts[val] = 0
    if val not in exacts_tot:
        exacts_tot[val] = 0
    if val not in partials:
        partials[val] = 0

    print(val, exacts[val], exacts_tot[val], 100*exacts[val]/exacts_tot[val])


    print(val, partials[val], exacts_tot[val], 100*partials[val]/exacts_tot[val])

    print("Average # of elements", exacts_tot[val]/len(data_gt))

print()
print("Final score", sum(exacts.values())/sum(exacts_tot.values()))
print("Final score partial", sum(partials.values())/sum(exacts_tot.values()))
# For example, to access the database host
# print("Database Host:", config['database']['host'])

# data augmetnations

# internal LLM consistency


# NER GPT
# https://academic.oup.com/jamia/article/31/9/1812/7590607
# clinical NER

# https://github.com/MantisAI/nervaluate

# https://www.davidsbatista.net/blog/2018/05/09/Named_Entity_Evaluation/

# https://towardsdatascience.com/advanced-ner-with-gpt-3-and-gpt-j-ce43dc6cdb9c