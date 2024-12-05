# from llm import load_model_processor, build_pipeline, build_pipeline_batch

# # llm = load_model_processor("cuda:0", '3b')
# https://www.youtube.com/watch?v=Q1i4bIIFOFc



import os
import json
import sys

import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import DataCollatorForTokenClassification
from transformers import TrainingArguments
from transformers import Trainer, EarlyStoppingCallback
from datasets import DatasetDict, concatenate_datasets, Dataset

from seqeval.metrics import accuracy_score, f1_score, classification_report
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score

from annotations import get_outcomes, load_dataset_as_bio, preproc_bert, mapping, label_list, num_labels

# https://cs229.stanford.edu/proj2005/KrishnanGanapathy-NamedEntityRecognition.pdf
# https://arxiv.org/html/2401.11431v1


def compute_metrics(pred, object_only=False):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=2)

    print(len(labels), labels[0])
    print(len(preds), preds[0])

    # Convert predictions and labels to the required format
    # keep only object
    true_labels = [[str(label_list[l]) for l in label if l != -100] for label in labels]
    pred_labels = [[str(label_list[p]) for (p, l) in zip(pred, label) if l != -100]
                for pred, label in zip(preds, labels)]
    
    # print(true_labels, pred_labels)

    # overfits
    if object_only: 
        true_labels = [[l if l == 'I-object' else 'O' for l in true_label] for true_label in true_labels]
        pred_labels = [[l if l == 'I-object' else 'O' for l in pred_label] for pred_label in pred_labels]
        # print(true_labels, pred_labels)
    
    print(len(true_labels), true_labels[0])
    print(len(pred_labels), pred_labels[0])

    # Calculate F1, precision, recall
    f1 = f1_score(true_labels, pred_labels)
    return {
        "f1": f1,
        "precision": precision_score(true_labels, pred_labels),
        "recall": recall_score(true_labels, pred_labels)
    }


def tokenize_method(tokenizer, label_all_tokens=True):
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(examples['tokens'], truncation=True, is_split_into_words=True)
        # Align labels
        labels = []
        for i, label in enumerate(examples['ner_tags']):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            label_ids = []
            previous_word_idx = None
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(label[word_idx] if label_all_tokens else -100)
                previous_word_idx = word_idx
            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs
    return tokenize_and_align_labels

def train_bert(model_name, folder, dataset):
    evals = {}
    # for ds_size in [10, 30, 50, 70, 90, 110, 130, 150]:
        # dataset_train
        # dataset = concatenate_datasets([train_dataset, test_dataset])
        # dataset = DatasetDict({
        #     'train': Dataset.from_dict( dataset[:ds_size]),
        #     'test': Dataset.from_dict( dataset[150:])
        # })
        

    # sweep over the number of samples
    # try different bert models
    # model_name = "emilyalsentzer/Bio_ClinicalBERT"
    # model_name = 'bert-base-uncased'
    # model_name = 'dslim/bert-base-NER'

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=num_labels)
    data_collator = DataCollatorForTokenClassification(tokenizer)

    tokenize_and_align_labels = tokenize_method(tokenizer)
    tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)
    print(tokenized_datasets['train'][0])
    print(tokenized_datasets['train'][1])

    training_args = TrainingArguments(
        output_dir=folder + "_temp",
        # evaluation_strategy="epoch",
        evaluation_strategy="steps",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        # per_device_train_batch_size=100,
        # per_device_eval_batch_size=100,
        num_train_epochs=200,#100,
        eval_steps=20,
        save_total_limit=5,
        metric_for_best_model='eval_f1',
        greater_is_better=True,
        weight_decay=0.01,
        load_best_model_at_end=True,
    )




    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]
    )
    # https://stackoverflow.com/questions/69087044/early-stopping-in-bert-trainer-instances

    trainer.train()

    model.save_pretrained(folder)
    tokenizer.save_pretrained(folder)

    eval = trainer.evaluate()
    print(eval)
    # evals[ds_size] = eval
    evals[model_name] = eval

    print(evals)
    with open(f'perf_{model_name}.json', 'w') as f:
        json.dump(evals, f)

def eval_bert(folder, dataset):
    # load huggingface model from folder
    tokenizer, model = load_model(folder)

    tokenize_and_align_labels = tokenize_method(tokenizer, True)
    tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)
    data_collator = DataCollatorForTokenClassification(tokenizer)

    # 5. Initialize Trainer
    training_args = TrainingArguments(
        output_dir=folder + "_temp_eval",  # Path to save results
        evaluation_strategy="epoch",  # Define evaluation strategy (ignored for eval-only usage)
        per_device_eval_batch_size=32,  # Batch size for evaluation
        logging_dir="./logs"  # Directory for logs
    )

    trainer = Trainer(
        model=model,  # The pre-trained model
        args=training_args,
        eval_dataset=tokenized_datasets["test"], # 1.0, completely overfits
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics  # Compute metrics function
    )

    # 6. Run evaluation
    results = trainer.evaluate()
    print(results)

def load_model(folder):
    tokenizer = AutoTokenizer.from_pretrained(folder, device_map='cuda:0')
    model = AutoModelForTokenClassification.from_pretrained(folder, num_labels=num_labels, device_map='cuda:0')

    return tokenizer, model


def infer_bert(folder, dataset, all=True): # for test set
    tokenizer, model = load_model(folder)
    # pass
    # folder_inf = "./finetuned_clinicalbiobert_inf"
    if all:
        # outcomes = pd.read_csv('../../snomed/raw/outcomes.txt', sep='|')
        # outcomes['length'] = outcomes['title'].apply(lambda x: len(x))
        # outcomes.sort_values('length', ascending=False, inplace=True)
        # rows = outcomes['title']
        # print(len(rows))
        # rows = rows.apply(lambda x: x.lower())
        # rows = rows.drop_duplicates()
        # rows = rows.reset_index(drop=True)
        # print(len(rows))
        # print(rows)
        rows = get_outcomes()
    else:
        # convert back tokens to words
        rows = dataset['test']
        
        print(rows)
        rows = [row['origs'] for row in rows]
        print(rows)

    # outcome2 = [rows[k] for k in range(10)]
    # print(outcome2)
    batch_size = 100 #100 # 1000 batch size on gpu possible
    outcomes_json = []
    for i in tqdm(range(0, len(rows), batch_size)):
        outcome2 = [rows.iloc[k+i] for k in range(batch_size)]

        outcome2_pp = [preproc_bert(outcome['title']) for outcome in outcome2]
        outcome_words = [outcome_pp.split(' ') for outcome_pp in outcome2_pp]
        
        tokens = tokenizer(outcome_words, return_tensors="pt", truncation=True, is_split_into_words=True, padding=True)
        outputs = model(**tokens.to('cuda:0'))
        predictions = torch.argmax(outputs.logits, dim=2).detach().cpu()
        # print(predictions)
        # print(tokens)

        # tokens vs words to align, if change in the middle what to do?
        for i in range(len(outcome2)):
            tokens_tmp = tokenizer.convert_ids_to_tokens(tokens["input_ids"][i])
            inv_map = {v: k for k, v in mapping.items()}

            predicted_labels = [inv_map[p.item()] for p in predictions[i]]

            # Combine tokens and their predicted labels
            result = [(token, label) for token, label in zip(tokens_tmp, predicted_labels) if token not in ["[CLS]", "[SEP]"]]

        # print(predictions)
        # print(predicted_labels)
        # print(result)

        # # post process
        # # batch implementation to do

        # print(len(predictions[0]))
        # print(len(outcome_words))

            inf_st = {}
            for key, val in mapping.items():
                acc = []
                last = None
                for k, (word, pred) in enumerate(result):
                    # print(pred, last, key)
                    if pred != last:
                        if len(acc) > 0 and key == last:
                            if key not in inf_st:
                                inf_st[key]= []
                            str_words = tokenizer.convert_tokens_to_string(acc).replace('[PAD]', '').strip()
                            if str_words != "":
                                inf_st[key].append(str_words)
                        # print(acc)
                        last = pred
                        acc = []
                    acc.append(word)

            # print(inf_st)

            outcomes_json.append((outcome2[i]['title'], int(outcome2[i]['id']), outcome2[i]['nct_id'], inf_st))

            if i % 100 == 0:
                with open(f'outputs/outcomes_extracted_{os.path.basename(folder)}_{all}.json', 'w') as f:
                    json.dump(outcomes_json, f)

# # Convert token IDs to actual tokens and predictions to labels
# tokens = tokenizer.convert_ids_to_tokens(tokens["input_ids"][0])

    
