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
from check_similarity import load_model, compute_sim

from scipy.optimize import linear_sum_assignment


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
        choices=["bert", 
                 "llm-llama", 
                 "llm-llama-object",
                 "llm-ministral",
                 "llm-ministral-object",
                 "llm-openai-4",
                 "llm-openai-4-all",
                 "comp-llm",
                 "plot",
                 "all"],
        required=True,
        help="Select the mode of operation: 'ner' for Named Entity Recognition or 'llm' for Large Language Model.",
    )

    # Parse the arguments
    args = parser.parse_args()

    # Execute based on the selected mode
    if args.type == "bert":
        print("BERT mode selected. Running Named Entity Recognition tasks...")

        file_json = "../llm/outputs/ner_infer_finetuned_bert-base-uncased_False.json"
        compute_distance("ner", file_json, "object", "object")
    elif args.type == "llm-ministral":
        print("LLM mode selected. Running Large Language Model tasks...")

        file_json = "../llm/outputs/llm_evaluate_ministral-8b_False.json"
        # file_json = "../llm/outputs/llm_evaluate_ministral-8b.json"
        compute_distance("llm-ministral", file_json, "Quantity or Object of Interest", "object")
    elif args.type == "llm-ministral-object":
        print("LLM mode selected. Running Large Language Model tasks...")

        file_json = "../llm/outputs/llm_evaluate_ministral-8b_True.json"
        # file_json = "../llm/outputs/llm_evaluate_ministral-8b.json"
        compute_distance("llm-ministral-object", file_json, "Quantity or Object of Interest", "object")
    elif args.type == "llm-llama":
        print("LLM mode selected. Running Large Language Model tasks...")

        file_json = "../llm/outputs/llm_evaluate_llama-8b_False.json"
        # file_json = "../llm/outputs/llm_evaluate_ministral-8b.json"
        compute_distance("llm-llama", file_json, "Quantity or Object of Interest", "object")
    elif args.type == "llm-llama-object":
        print("LLM mode selected. Running Large Language Model tasks...")

        file_json = "../llm/outputs/llm_evaluate_llama-8b_True.json"
        # file_json = "../llm/outputs/llm_evaluate_ministral-8b.json"
        compute_distance("llm-llama-object", file_json, "Quantity or Object of Interest", "object")
    elif args.type == "llm-openai-4":
        print("LLM mode selected. Running Large Language Model tasks...")

        file_json = "../llm/outputs/chatgpt_zero_4_raw_preprocessed.json"
        # file_json = "../llm/outputs/llm_evaluate_ministral-8b.json"
        compute_distance("llm-openai-4", file_json, "Quantity or Object of Interest", "object")
    elif args.type == "llm-openai-4-all":
        print("LLM mode selected. Running Large Language Model tasks...")

        file_json = "../llm/outputs/chatgpt_zero_4_all_raw_preprocessed.json"
        # file_json = "../llm/outputs/llm_evaluate_ministral-8b.json"
        compute_distance("llm-openai-4-all", file_json, "Quantity or Object of Interest", "object")
    elif args.type == "comp-llm":
        print("LLM mode selected. Running Large Language Model tasks...")

        files = [
            "llm_evaluate_llama-8b_{'only_object': True, 'example_selection': 'random', 'n_examples': 0}",
            "llm_evaluate_llama-8b_{'only_object': True, 'example_selection': 'random', 'n_examples': 8}",
            "llm_evaluate_llama-8b_{'only_object': True, 'example_selection': 'random', 'n_examples': 50}",
            "llm_evaluate_llama-8b_{'only_object': True, 'example_selection': 'random', 'n_examples': 100}",
        ]
        for i, file in enumerate(files):
        # file_json = "../llm/outputs/llm_evaluate_ministral-8b.json"
            compute_distance(f"llm-all-{i}", "../llm/outputs/" + file + ".json", "Quantity or Object of Interest", "object")
    elif args.type == "plot":
        plot(True)
        plot(False)
    elif args.type == "all":
        print("BERT mode selected. Running Named Entity Recognition tasks...")

        file_json = "../llm/outputs/ner_infer_finetuned_bert-base-uncased_False.json"
        compute_distance("ner", file_json, "object", "object")
        print("LLM mode selected. Running Large Language Model tasks...")

        file_json = "../llm/outputs/llm_evaluate_ministral-8b_False.json"
        # file_json = "../llm/outputs/llm_evaluate_ministral-8b.json"
        compute_distance("llm-ministral", file_json, "Quantity or Object of Interest", "object")
        print("LLM mode selected. Running Large Language Model tasks...")

        file_json = "../llm/outputs/llm_evaluate_ministral-8b_True.json"
        # file_json = "../llm/outputs/llm_evaluate_ministral-8b.json"
        compute_distance("llm-ministral-object", file_json, "Quantity or Object of Interest", "object")
        print("LLM mode selected. Running Large Language Model tasks...")

        file_json = "../llm/outputs/llm_evaluate_llama-8b_False.json"
        # file_json = "../llm/outputs/llm_evaluate_ministral-8b.json"
        compute_distance("llm-llama", file_json, "Quantity or Object of Interest", "object")
        print("LLM mode selected. Running Large Language Model tasks...")

        file_json = "../llm/outputs/llm_evaluate_llama-8b_True.json"
        # file_json = "../llm/outputs/llm_evaluate_ministral-8b.json"
        compute_distance("llm-llama-object", file_json, "Quantity or Object of Interest", "object")
        print("LLM mode selected. Running Large Language Model tasks...")

        file_json = "../llm/outputs/chatgpt_zero_4_raw_preprocessed.json"
        # file_json = "../llm/outputs/llm_evaluate_ministral-8b.json"
        compute_distance("llm-openai-4", file_json, "Quantity or Object of Interest", "object")
        print("LLM mode selected. Running Large Language Model tasks...")

        file_json = "../llm/outputs/chatgpt_zero_4_all_raw_preprocessed.json"
        # file_json = "../llm/outputs/llm_evaluate_ministral-8b.json"
        compute_distance("llm-openai-4-all", file_json, "Quantity or Object of Interest", "object")
    



# https://stackoverflow.com/questions/45783385/normalizing-the-edit-distance
# https://stackoverflow.com/questions/64113621/how-to-normalize-levenshtein-distance-between-0-to-1

# # Calculate the Levenshtein distance
# str1 = "kitten"
# str2 = "sitting"
# distance = Levenshtein.distance(str1, str2)

# print(f"The Levenshtein distance between '{str1}' and '{str2}' is {distance}.")

def plot(all=True):
    import matplotlib.pyplot as plt
    
    
    if all:
         files = [
            ("outputs/llm-all-0.txt", "LLM Llama Object 0 FS"),
            ("outputs/llm-all-1.txt", "LLM Llama Object 8 FS"),
            ("outputs/llm-all-2.txt", "LLM Llama Object 50 FS"),
            ("outputs/llm-all-3.txt", "LLM Llama Object 100 FS")
        ]
    else:
        files = [
            ("outputs/llm-ministral.txt", "Ministral"),
            ("outputs/llm-ministral-object.txt", "Ministral Object Only"),
            ("outputs/llm-llama.txt", "Llama"),
            ("outputs/llm-llama-object.txt", "Llama Object Only"),
            ("outputs/llm-openai-4.txt", "OpenAI 4 Object Only"),
            ("outputs/ner.txt", "BERT")
        ]

    # factor = 3
    if all:
        factor = 2
    else:
        factor = 3

    fig1, ax1 = plt.subplots(2, factor, figsize=(12, 7))  # For histograms
    fig2, ax2 = plt.subplots(2, factor, figsize=(12, 7))  # For boxplots

    for i, (file, name) in enumerate(files):
        data_all = np.loadtxt(file)
        data = data_all[:, 0]
        diffs = data_all[:, 1]

        # Histogram on fig1
        row, col = divmod(i, factor)
        ax1[row, col].hist(data, alpha=0.3, label=name + f" - avg {np.mean(data):.2f}", bins=20, ec="k")
        ax1[row, col].set_xlabel("Normalized Levenshtein distance")
        ax1[row, col].set_ylabel("Count")
        ax1[row, col].set_xlim([-0.1, 1.1])
        ax1[row, col].set_ylim([-1, 30])
        ax1[row, col].legend()

        # Boxplot on fig2
        ax2[row, col].hist(diffs, alpha=0.3, label=name + f" - avg {np.mean(diffs):.2f}", bins=20, ec="k")
        ax2[row, col].set_xlabel("Differences")
        ax2[row, col].set_ylabel("Count")        
        # ax2[row, col].set_title(name + f" - avg {np.mean(diffs):.2f}")
        # ax2[row, col].set_ylabel("Differences")
        ax2[row, col].set_ylim([-1, 50])
        ax2[row, col].set_xlim([-1.1, 5])
        ax2[row, col].legend()


        # plt.subplot(2, 3, i+1)
        # plt.hist(data, alpha=0.3, label=name + f" {np.mean(data):.2f} - {np.mean(diffs):.2f}", bins=20, ec="k")
        # plt.xlabel("Normalized Levenshtein distance")
        # plt.ylabel("Count")
        # plt.xlim([-0.1, 1.1])
        # plt.legend()
        # # fig 2
        # plt.boxplot(diffs)
    fig1.tight_layout()
    fig2.tight_layout()
    fig1.savefig(f'outputs/perf-{all}-new.png')
    fig2.savefig(f'outputs/perf-{all}-new2.png')

def compute_distance(name, file_json, category_json, category_annotation):
    # print(data_all)
    with open(file_json, 'r') as file:
        data_pred = json.load(file)

    metric = "sim" #"sim"
    # metric = "levenshtein"
    if metric == "sim":
        model = load_model()
    

    min_dist = []
    diffs = []
    report = ""
    for i in tqdm(range(len(data_pred))):
        # category_json
        # category_annotation

        # print(dataset['test'][i])

        gts = []
        for elt in dataset['test']['true_labels'][i]:
            # print(elt)
            if category_annotation in elt:
                gts.append(preproc_bert(elt[category_annotation].lower()))
        
        try:
            if data_pred[i][1] is not None:
                print(data_pred[i][1])
                pred_tmp_low = {k.lower(): v for k, v in data_pred[i][1].items()}
                pred_tmp = pred_tmp_low[category_json.lower()]
                if type(pred_tmp) != list:
                    pred_tmp = [pred_tmp]
                pred = []
                for pred_t in pred_tmp:
                    pred.append(pred_t.strip().lower())
            else:
                pred = None
        except:
            print("Failed")
            pred = None

        print(gts, pred)
        if pred is not None and len(pred) > 0 and len(gts) > 0:

            # matrix of levenstein distance
            distance_matrix = [[distance(item1.lower(), item2.lower())/max(len(item1), len(item2)) for item2 in gts] for item1 in pred]
            
            if metric == "sim":
                distance_matrix = compute_sim(model, gts, pred).T

            df = pd.DataFrame(distance_matrix, index=pred, columns=gts)
            print(df)
            report += f"{df}\n\n"

            if metric == "sim":
                cost_matrix = -df.values

                # Compute the best assignment
                row_indices, col_indices = linear_sum_assignment(cost_matrix)

                # Map the assignments to the DataFrame indices
                assignments = list(zip(df.index[row_indices], df.columns[col_indices]))

                # Compute the total score of the best assignment
                total_score = df.values[row_indices, col_indices].mean()

                # Display the results
                print("Best Assignments:")
                for gt, predd in assignments:
                    print(f"Ground Truth: {gt} -> Prediction: {predd}")

                print(f"\nTotal Score: {total_score}")
                min_dist.append(total_score)
            else:
                min_dist.append(df.min().values[0])
            # compute assignment, and take the score
            # compute difference

            diffs.append(np.abs(len(gts) - len(pred)))
        else:
            if metric == "sim":
                min_dist.append(0.)
            else:
                min_dist.append(1.)
            diffs.append(len(gts))

    with open('report.txt', 'w') as f:
        f.write(report)
    
    print(min_dist)
    print(sum(min_dist)/len(min_dist))
    print(np.count_nonzero(np.array(min_dist) == 0.)/len(min_dist))
    print(np.count_nonzero(np.array(min_dist) < 0.1)/len(min_dist))
    print(np.mean(diffs))

    # save min_dist
    np.savetxt(f"outputs/{name}.txt", np.c_[min_dist, diffs])

# def convert_bio_to_json():
    # pass

# convert_bio_to_json("")
# https://towardsdatascience.com/a-pathbreaking-evaluation-technique-for-named-entity-recognition-ner-93da4406930c



if __name__ == "__main__":
    main()
