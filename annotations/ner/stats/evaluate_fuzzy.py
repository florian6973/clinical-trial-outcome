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
                 "gatortron",
                 "llm-llama", 
                 "llm-llama-object",
                 "llm-ministral",
                 "llm-ministral-object",
                 "llm-openai-4",
                 "llm-openai-4-all",
                 "comp-llm",
                 "plot",
                 "plot_together",
                 "plot_complexity",
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
    elif args.type == "gatortron":
        print("BERT mode selected. Running Named Entity Recognition tasks...")

        file_json = "../llm/outputs/ner_infer_finetuned_UFNLP-gatortron-base-fp_False.json"
        compute_distance("gatortron", file_json, "object", "object")
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
    elif args.type == "plot_together":
        plot_together()
    elif args.type == "plot_complexity":
        # plot(True)
        plot_complexity(False)
    elif args.type == "all":
        print("BERT mode selected. Running Named Entity Recognition tasks...")

        file_json = "../llm/outputs/ner_infer_finetuned_UFNLP-gatortron-base-fp_False.json"
        compute_distance("gatortron", file_json, "object", "object")
        # print("LLM mode selected. Running Large Language Model tasks...")

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
    
        files = [
            "llm_evaluate_llama-8b_{'only_object': True, 'example_selection': 'random', 'n_examples': 0}",
            "llm_evaluate_llama-8b_{'only_object': True, 'example_selection': 'random', 'n_examples': 8}",
            "llm_evaluate_llama-8b_{'only_object': True, 'example_selection': 'random', 'n_examples': 50}",
            "llm_evaluate_llama-8b_{'only_object': True, 'example_selection': 'random', 'n_examples': 100}",
        ]
        for i, file in enumerate(files):
        # file_json = "../llm/outputs/llm_evaluate_ministral-8b.json"
            compute_distance(f"llm-all-{i}", "../llm/outputs/" + file + ".json", "Quantity or Object of Interest", "object")



# https://stackoverflow.com/questions/45783385/normalizing-the-edit-distance
# https://stackoverflow.com/questions/64113621/how-to-normalize-levenshtein-distance-between-0-to-1

# # Calculate the Levenshtein distance
# str1 = "kitten"
# str2 = "sitting"
# distance = Levenshtein.distance(str1, str2)

# print(f"The Levenshtein distance between '{str1}' and '{str2}' is {distance}.")


def rand_jitter(arr):
    stdev = .01 * (max(arr) - min(arr))
    return arr + np.random.randn(len(arr)) * stdev

def jitter(ax, x, y, s=20, c='b', marker='o', cmap=None, norm=None, vmin=None, vmax=None, alpha=None, linewidths=None, verts=None, hold=None, **kwargs):
    return ax.scatter(rand_jitter(x), rand_jitter(y), s=s, c=c, marker=marker, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax, alpha=alpha, linewidths=linewidths, **kwargs)


metric = "levenshtein"

def plot(all=True):
    import matplotlib.pyplot as plt

    if all:
         files = [
            (f"outputs/{metric}/llm-all-0.txt", "LLM Llama Object 0 FS"),
            (f"outputs/{metric}/llm-all-1.txt", "LLM Llama Object 8 FS"),
            (f"outputs/{metric}/llm-all-2.txt", "LLM Llama Object 50 FS"),
            (f"outputs/{metric}/llm-all-3.txt", "LLM Llama Object 100 FS")
        ]
    else:
        files = [
            
            # (f"outputs/{metric}/llm-ministral.txt", "Ministral"),
            (f"outputs/{metric}/llm-ministral-object.txt", "Ministral Object Only"),
            (f"outputs/{metric}/llm-llama.txt", "Llama"),
            (f"outputs/{metric}/llm-llama-object.txt", "Llama Object Only"),
            (f"outputs/{metric}/llm-openai-4.txt", "OpenAI 4 Object Only"),
            (f"outputs/{metric}/ner.txt", "BERT"),
            (f"outputs/{metric}/gatortron.txt", "Gatortron"),
            # BETTER F1 BUT WORSE LEVENSHTEIN...
        ]

    # factor = 3
    if all:
        factor = 2
    else:
        factor = 3

    fig1, ax1 = plt.subplots(2, factor, figsize=(12, 7))  # For histograms
    fig2, ax2 = plt.subplots(2, factor, figsize=(12, 7))  # For boxplots
    fig3, ax3 = plt.subplots(2, factor, figsize=(12, 7))  # For boxplots
    fig4, ax4 = plt.subplots(2, factor, figsize=(12, 7))  # For boxplots

    for i, (file, name) in enumerate(files):
        data_all = np.loadtxt(file)
        data = data_all[:, 0]
        diffs = data_all[:, 1]
        print(file)

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

        try:
            complexities = data_all[:, 2]
            # ax3[row, col].scatter(complexities, diffs, label=name + f" - avg {np.mean(data):.2f}")
            jitter(ax3[row, col], complexities, diffs, label=name + f" - avg {np.mean(diffs):.2f}")
            ax3[row, col].set_xlabel("Complexity")
            ax3[row, col].set_ylabel("Differences")  
            ax3[row, col].legend()      

            jitter(ax4[row, col], complexities, data, label=name + f" - avg {np.mean(data):.2f}")
            ax4[row, col].set_xlabel("Complexity")
            ax4[row, col].set_ylabel("Normalized Levenshtein distance")  
            ax4[row, col].legend()
        except:
            print("no complexity available")

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
    fig3.tight_layout()
    fig4.tight_layout()
    fig1.savefig(f'outputs/{metric}/perf-{all}-new.png')
    fig2.savefig(f'outputs/{metric}/perf-{all}-new2.png')
    fig3.savefig(f'outputs/{metric}/perf-{all}-new3.png')
    fig4.savefig(f'outputs/{metric}/perf-{all}-new4.png')


def plot_together():
    import matplotlib.pyplot as plt

    
    files = [
        # (f"outputs/{metric}/llm-all-0.txt", "LLM Llama Object 0 FS"),
        (f"outputs/{metric}/ner.txt", "BERT"),
        (f"outputs/{metric}/llm-openai-4.txt", "GPT-4 Object ZS"),
        (f"outputs/{metric}/gatortron.txt", "Gatortron"),
        (f"outputs/{metric}/llm-llama-object.txt", "Llama Object ZS"),
        (f"outputs/{metric}/llm-llama.txt", "Llama ZS"),
        (f"outputs/{metric}/llm-ministral-object.txt", "Ministral Object ZS"),

        (f"outputs/{metric}/llm-all-1.txt", "Llama Object 8 FS"),
        (f"outputs/{metric}/llm-all-2.txt", "Llama Object 50 FS"),
        (f"outputs/{metric}/llm-all-3.txt", "Llama Object 100 FS"),
        # (f"outputs/{metric}/llm-ministral.txt", "Ministral"),

        # BETTER F1 BUT WORSE LEVENSHTEIN...
    ]
    colors = ['yellow', 'red', 'yellow',
              'red', 'red', 'red',
              'blue', 'blue', 'blue']


    fig1, ax1 = plt.subplots(2, 1, figsize=(15, 9))  # For histograms
    fig2, ax2 = plt.subplots(1, 1, figsize=(15, 9))  # For histograms

    bps_nld = []
    bps_d = []
    labels = []
    number_of_objects = {}
    counts = {}
    baseline = 0
    for i, (file, name) in enumerate(files):
        if name not in number_of_objects:
            number_of_objects[name] = 0
            counts[name] = 0

        data_all = np.loadtxt(file)
        data = data_all[:, 0]
        diffs = data_all[:, 1]
        n_objs = data_all[:, 3]
        n_objs_gt = data_all[:, 4]
        print(file)

        bps_nld.append(data)
        bps_d.append(diffs)
        labels.append(name)
        number_of_objects[name] += sum(n_objs)
        counts[name] += np.count_nonzero(np.array(n_objs) != -1)
        if i == 0:
            baseline += sum(n_objs_gt)

    avgs = {}
    for key, val in number_of_objects.items():
        avgs[key] = val / counts[key]

    baseline = baseline / len(n_objs_gt)

    # print(number_of_objects)

    box1 = ax1[0].boxplot(bps_nld, patch_artist=True, notch=True)
    box2 = ax1[1].boxplot(bps_d, patch_artist=True, notch=True)


    # Customize each box
    for patch, color in zip(box1['boxes'], colors):
        patch.set_facecolor(color)
    for patch, color in zip(box2['boxes'], colors):
        patch.set_facecolor(color)

    # # Optionally, customize other elements like whiskers, medians, caps, etc.
    for k, patch in enumerate(box2['boxes']):
        if k == 0:
            patch.set_edgecolor('red')  # Set edge color
            patch.set_linewidth(2)  # Set line width
    for k, patch in enumerate(box1['boxes']):
        if k == 0:
            patch.set_edgecolor('red')  # Set edge color
            patch.set_linewidth(2)  # Set line width


    ax1[0].set_xticks(list(range(1, len(labels)+1)), labels)
    ax1[0].set_ylabel("Normalized Levenshtein Distance")
    ax1[0].set_xlabel("Model")
    ax1[1].set_xticks(list(range(1, len(labels)+1)), labels)
    ax1[1].set_ylabel("Differences")
    ax1[1].set_xlabel("Model")

    # ax2.bar(list(range(1, len(labels)+1)), list(number_of_objects.values()))

    for i, (label, value) in enumerate(avgs.items(), start=1):
        ax2.bar(
            i, 
            value, 
            color=colors[i-1], 
            edgecolor='red' if i == 1 else None,  # Red contour for 'A', black (default) for others,
            linewidth=2
    )

    # ax2.bar(list(range(1, len(labels)+1)), list(avgs.values()), color=colors)
    ax2.axhline(baseline, label='Groundtruth average', c='gray')
    ax2.set_xticks(list(range(1, len(labels)+1)), labels)
    ax2.set_ylabel("Average number of objects per outcome")
    # ax2.set_yscale('log')
    ax2.legend()

    fig1.tight_layout()
    fig1.savefig(f'outputs/{metric}/perf-together.png')
    fig2.tight_layout()
    fig2.savefig(f'outputs/{metric}/perf-together-outcomes.png')

def plot_complexity(all=True):
    import matplotlib.pyplot as plt

    # metric = "levenshtein"
    
    
    if all:
         files = [
            (f"outputs/{metric}/llm-all-0.txt", "LLM Llama Object 0 FS"),
            (f"outputs/{metric}/llm-all-1.txt", "LLM Llama Object 8 FS"),
            (f"outputs/{metric}/llm-all-2.txt", "LLM Llama Object 50 FS"),
            (f"outputs/{metric}/llm-all-3.txt", "LLM Llama Object 100 FS")
        ]
    else:
        files = [
            (f"outputs/{metric}/llm-ministral.txt", "Ministral"),
            (f"outputs/{metric}/llm-ministral-object.txt", "Ministral Object Only"),
            (f"outputs/{metric}/llm-llama.txt", "Llama"),
            (f"outputs/{metric}/llm-llama-object.txt", "Llama Object Only"),
            (f"outputs/{metric}/llm-openai-4.txt", "OpenAI 4 Object Only"),
            (f"outputs/{metric}/ner.txt", "BERT"),
            (f"outputs/{metric}/gatortron.txt", "Gatortron"),
        ]

    fig1, ax1 = plt.subplots(2, 1, figsize=(12, 7))  
    ax1 = ax1.flatten()

    for i, (file, name) in enumerate(files):
        data_all = np.loadtxt(file)
        data = data_all[:, 0]
        diffs = data_all[:, 1]
        print(file)
        complexities = data_all[:, 2]

        df = pd.DataFrame({"levenshtein": data, "differences": diffs, "complexities": complexities})
        # dfg = df.groupby("complexities").mean()
        dfg = df.groupby("complexities").agg(['mean', 'std'])

        dfg.dropna(inplace=True)

        print(dfg)

        ax1[0].errorbar(dfg.index, dfg[("levenshtein", "mean")], yerr=dfg[("levenshtein", "std")], label=name, linestyle='--',  # Dashed line
    marker='x')      # Cross markers)
        ax1[1].errorbar(dfg.index, dfg[("differences", "mean")], yerr=dfg[("differences", "std")], label=name,
                        linestyle='--',  # Dashed line
    marker='x')      # Cross markers)

        # exit()

    ax1[0].legend()
    ax1[0].set_xlabel("Complexity")
    ax1[0].set_ylabel('Normalized Levenshtein Distance')

    ax1[1].set_ylabel('Differences')
    ax1[1].set_xlabel("Complexity")
    ax1[1].legend()


    fig1.tight_layout()
    fig1.savefig(f'outputs/{metric}/perf-{all}-comp.png')

def compute_distance(name, file_json, category_json, category_annotation):
    # print(data_all)
    with open(file_json, 'r') as file:
        data_pred = json.load(file)

    # metric = "sim" #"sim"
    # metric = "sim"
    if metric == "sim":
        model = load_model()
    

    min_dist = []
    diffs = []
    complexities = []
    n_objects = []
    total_objs = []
    # len as proxy
    # number of elements of data structure by linking with the initial dataset
    report = ""
    for i in tqdm(range(len(data_pred))):
        # category_json
        # category_annotation

        # print(dataset['test'][i])
        # exit()

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

            # complexities.append(len(dataset['test'][i]['origs']))
            complexities.append(sum([sum([1 if x is not None else 0 for x in dico.values()]) for dico in dataset['test'][i]['true_labels']]))
            # print(dataset['test'][i]['true_labels'])
            # exit()
            # n_objects.append(sum([1 if 'object' in dico else 0 for dico in dataset['test'][i]['true_labels']])) # len(dico['object'])
            # print("OK", n_objects)
            # input()
        except:
            print("Failed", data_pred[i][1]) # often when no object in result
            # exit()/
            pred = None
            complexities.append(np.nan)
            # n_objects.append(-1)

            
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
                # min_dist.append(df.min().values[0])
                print(df)
                # if len(df) > 1:
                    # exit()
                min_dist.append(df.min().mean())
            # compute assignment, and take the score
            # compute difference

            diffs.append(np.abs(len(gts) - len(pred)))
            n_objects.append(len(pred))
            # if len(gts) > len(pred):
            #     print("Sup")
            #     exit()
        else:
            if metric == "sim":
                min_dist.append(0.)
            else:
                min_dist.append(1.)
            diffs.append(len(gts))
            n_objects.append(0)
        total_objs.append(len(gts))
            # diffs.append(0)

    with open('report.txt', 'w') as f:
        f.write(report)
    
    print(min_dist)
    print(sum(min_dist)/len(min_dist))
    print(np.count_nonzero(np.array(min_dist) == 0.)/len(min_dist))
    print(np.count_nonzero(np.array(min_dist) < 0.1)/len(min_dist))
    print(np.mean(diffs))

    # save min_dist
    np.savetxt(f"outputs/{metric}/{name}.txt", np.c_[min_dist, diffs, complexities, n_objects, total_objs])

# def convert_bio_to_json():
    # pass

# convert_bio_to_json("")
# https://towardsdatascience.com/a-pathbreaking-evaluation-technique-for-named-entity-recognition-ner-93da4406930c



if __name__ == "__main__":
    main()
