import json
import yaml
from tqdm import tqdm

from Levenshtein import distance # or editdistance library
import sys
import pandas as pd
import numpy as np
import os

import argparse

sys.path.append("../llm")
from annotations import dataset, preproc_bert


# file_ann = "../data/manual-ann-ner-all.yaml"
# file_json = "../llm/outputs/llm_evaluate_ministral-8b.json"

# with open(file_ann, 'r') as file:
#     data_gt = yaml.safe_load(file)['annotations']


def main():
    # Initialize the parser
    parser = argparse.ArgumentParser(description="Select between NER and LLM.")

    # Add argument for selecting the mode
    parser.add_argument(
        "--type",
        choices=["ner", "llm-llama", "llm-ministral", "plot"],
        required=True,
        help="Select the mode of operation: 'ner' for Named Entity Recognition or 'llm' for Large Language Model.",
    )

    # Parse the arguments
    args = parser.parse_args()

    # Execute based on the selected mode
    if args.type == "ner":
        print("NER mode selected. Running Named Entity Recognition tasks...")

        file_json = "../llm/outputs/ner_infer_finetuned_bert-base-uncased_False.json"
        compute_distance("ner", file_json, "object", "object")
    elif args.type == "llm-ministral":
        print("LLM mode selected. Running Large Language Model tasks...")

        file_json = "../llm/outputs/llm_evaluate_ministral-8b.json"
        # file_json = "../llm/outputs/llm_evaluate_ministral-8b.json"
        compute_distance("llm-ministral", file_json, "Quantity or Object of Interest", "object")
    elif args.type == "llm-llama":
        print("LLM mode selected. Running Large Language Model tasks...")

        file_json = "../llm/outputs/llm_evaluate_llama-8b.json"
        # file_json = "../llm/outputs/llm_evaluate_ministral-8b.json"
        compute_distance("llm-llama", file_json, "Quantity or Object of Interest", "object")
    elif args.type == "plot":
        plot()



# https://stackoverflow.com/questions/45783385/normalizing-the-edit-distance
# https://stackoverflow.com/questions/64113621/how-to-normalize-levenshtein-distance-between-0-to-1

# # Calculate the Levenshtein distance
# str1 = "kitten"
# str2 = "sitting"
# distance = Levenshtein.distance(str1, str2)

# print(f"The Levenshtein distance between '{str1}' and '{str2}' is {distance}.")

def plot():
    import matplotlib.pyplot as plt
    
    files = [
        ("outputs/llm-ministral.txt", "LLM Ministral"),
        ("outputs/llm-llama.txt", "LLM Llama"),
        ("outputs/ner.txt", "NER")
    ]
    for file, name in files:
        data = np.loadtxt(file)
        plt.hist(data, alpha=0.3, label=name, bins=20)
    plt.xlabel("Normalized Levenshtein distance")
    plt.ylabel("Count")
    plt.legend()
    plt.savefig('outputs/perf.png')

def compute_distance(name, file_json, category_json, category_annotation):
    # print(data_all)
    with open(file_json, 'r') as file:
        data_pred = json.load(file)

    min_dist = []
    for i in tqdm(range(len(data_pred))):
        # category_json
        # category_annotation

        # print(dataset['test'][i])

        gts = []
        for elt in dataset['test']['true_labels'][i]:
            # print(elt)
            if category_annotation in elt:
                gts.append(preproc_bert(elt[category_annotation].lower()))
        
        if data_pred[i][1] is not None:
            print(data_pred[i][1])
            pred_tmp = data_pred[i][1][category_json]
            if type(pred_tmp) != list:
                pred_tmp = [pred_tmp]
            pred = []
            for pred_t in pred_tmp:
                pred.append(pred_t.strip().lower())
        else:
            pred = None

        print(gts, pred)
        if pred is not None and len(pred) > 0 and len(gts) > 0:

            # matrix of levenstein distance
            distance_matrix = [[distance(item1.lower(), item2.lower())/max(len(item1), len(item2)) for item2 in gts] for item1 in pred]

            df = pd.DataFrame(distance_matrix, index=pred, columns=gts)
            print(df)
            min_dist.append(df.min().values[0])
        else:
            min_dist.append(1.)
    
    print(min_dist)
    print(sum(min_dist)/len(min_dist))
    print(np.count_nonzero(np.array(min_dist) == 0.)/len(min_dist))
    print(np.count_nonzero(np.array(min_dist) < 0.1)/len(min_dist))

    # save min_dist
    np.savetxt(f"outputs/{name}.txt", min_dist)

# def convert_bio_to_json():
    # pass

# convert_bio_to_json("")
# https://towardsdatascience.com/a-pathbreaking-evaluation-technique-for-named-entity-recognition-ner-93da4406930c



if __name__ == "__main__":
    main()
