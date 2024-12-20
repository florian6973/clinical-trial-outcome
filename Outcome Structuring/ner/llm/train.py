import os

import argparse


from annotations import dataset, dataset_jb
from ner_bert import train_bert, infer_bert, eval_bert
from ner_llm import eval_llm, infer_llm, train_llm

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# For LLM
# Train is only finetuning for LLM
# EVAL on the examples -- including in-context learning
# while inference is running on all the samples

def main():
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Script for selecting method and type of operation.")
    
    # Add arguments
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["ner_bert", "ner_llm"],
        help="Specify the method to use: 'ner_llm' for Named Entity Recognition or 'ner_llm' for Large Language Model."
    )
    parser.add_argument(
        "--type",
        type=str,
        required=True,
        choices=["train", "inference", "evaluate"],
        help="Specify the type of operation: 'train' for training or 'inference' for running inference."
    )
    parser.add_argument(
        '--gpu', 
        type=int, 
        default=0, 
        help="Index of the GPU to use (default: 0)."
    )
    parser.add_argument(
        '--dataset', 
        type=str, 
        default="fp", 
        help="Name of the dataset to use."
    )
    
    # Parse the arguments
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    llm_name = 'llama-8b' 
    # llm_name = 'ministral-8b'

    ds = dataset
    if args.dataset == "jb":
        ds = dataset_jb

    # model_name = 'bert-base-uncased'    
    model_name = 'UFNLP/gatortron-base'
    # model_name = 'UFNLP/gatortron-medium'  
    # model_name = "emilyalsentzer/Bio_ClinicalBERT"

    model_name_path =  model_name.replace('/', '-')
    
    # Logic based on parsed arguments
    if args.method == "ner_bert":
        if args.type == "train":
            print("Training the NER model...")
            # Add your training code here
            train_bert(model_name, f"./finetuned_{model_name_path}-" + args.dataset, dataset)
        elif args.type == "inference":
            print("Running inference with the NER model...")
            infer_bert(f"./finetuned_{model_name_path}-" + args.dataset, ds)
            # infer_bert("./finetuned_clinicalbiobert_inf", ds)
            # Add your inference code here
        elif args.type == "evaluate":
            print("Running evaluation with the NER model...")
            eval_bert(f"./finetuned_{model_name_path}-" + args.dataset, ds)
            # eval_bert("./finetuned_clinicalbiobert_inf", ds)
    elif args.method == "ner_llm":
        if args.type == "train":
            print("Training the LLM model...")
            # Add your training code here
            train_llm(llm_name, "lora", ds)
        elif args.type == "inference":
            print("Running inference with the LLM model...")
            # Add your inference code here
            infer_llm(llm_name, ds)

        elif args.type == "evaluate":
            print("Running evaluation with the LLM model...")
            # eval_bert("./finetuned_bert-base-uncased")
            settings = {
                "only_object": True,
                "example_selection": "random",
                "n_examples": 8
            }
            eval_llm(llm_name, ds, settings)
            settings = {
                "only_object": True,
                "example_selection": "random",
                "n_examples": 50
            }
            eval_llm(llm_name, ds, settings)
            settings = {
                "only_object": True,
                "example_selection": "random",
                "n_examples": 100
            }
            eval_llm(llm_name, ds, settings)
            settings = {
                "only_object": True,
                "example_selection": "random",
                "n_examples": 0
            }
            eval_llm(llm_name, ds, settings)

if __name__ == "__main__":
    main()

# readme then