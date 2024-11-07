# from llm import load_model_processor, build_pipeline, build_pipeline_batch

# # llm = load_model_processor("cuda:0", '3b')

if __name__ == "__main__":
    import pandas as pd
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"



    # build dataset
    examples = pd.read_csv("../data/manual-ann-ner-100_200.csv", index_col=0)['title']
    # print(examples)

    import yaml
    with open('../data/manual-ann-ner-extended.yaml', 'r') as file:
        # Step 2: Load the contents of the file
        data_gt = yaml.safe_load(file)['annotations']


    tokens = []
    ner_tags = []
    mapping = {
        'object': 0,
        'specifier': 1,
        'measure': 2,
        'time': 3,
        'unit': 4,
        'range': 5
    }

    def preproc(txt):
        return txt.replace(',', ' ').replace('(', ' ').replace(')', ' ').replace('-', ' ').replace(';', ' ').replace(':', ' ').replace('.', ' ').replace('/', ' ').lower()

    for i in range(len(data_gt)):
        print(examples[i])
        txt = preproc(examples[i])
        print(txt)
        words = txt.split(' ')
        # print(examples[i])
        nb_words = len(words)
        # # print(convert_yaml_to_json(data_gt[i]))
        labels = [-100] * nb_words
        for anns in data_gt[i]['structured']:
            # print(anns)
            for key, val in anns.items():
                words_loc = preproc(val).split(' ')
                for word in words_loc:
                    idx = words.index(word)
                    # print(idx)
                    labels[idx] = mapping[key]

        print(labels)
        print(words)

        tokens.append(words)
        ner_tags.append(labels)
        # input()

        # ex = str(convert_yaml_to_json(data_gt[i])).replace("'", '"')
        
    df = pd.DataFrame({'tokens': tokens, "ner_tags": ner_tags})

    from datasets import Dataset

    dataset = Dataset.from_pandas(df)
    print(dataset)


    from transformers import AutoTokenizer, AutoModelForTokenClassification
    from transformers import DataCollatorForTokenClassification
    num_labels = 6

    model_name = "emilyalsentzer/Bio_ClinicalBERT"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=num_labels)


    data_collator = DataCollatorForTokenClassification(tokenizer)

    label_all_tokens = True

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

    tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)


    from transformers import TrainingArguments

    training_args = TrainingArguments(
        output_dir="./results",
        # evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
    )

    from transformers import Trainer

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        # eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    model.save_pretrained("./finetuned_clinicalbiobert")
    tokenizer.save_pretrained("./finetuned_clinicalbiobert")
