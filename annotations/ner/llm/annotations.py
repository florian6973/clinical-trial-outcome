import pandas as pd
from datasets import Dataset
import yaml

mapping = {
    'nothing': 0,
    'object': 1,
    'specifier': 2,
    'measure': 3,
    'time': 4,
    'unit': 5,
    'range': 6,
}

label_list = list(mapping.values())
label_list[0] = 'O'
for i in range(1, len(label_list)):
    label_list[i] = 'I-' + list(mapping.keys())[i]
num_labels = len(mapping)



def preproc_bert(txt):
    return txt.replace(',', ' ').replace('[', ' ').replace(']', ' ').replace('(', ' ').replace(')', ' ').replace('-', ' ').replace(';', ' ').replace(':', ' ').replace('.', ' ').replace('/', ' ').replace('\n', '').lower()


def load_dataset_as_bio(
    sentences_path,
    labels_path
):
    if sentences_path.endswith('csv'):
        examples = pd.read_csv(sentences_path, index_col=0)['title']
    else:
        with open(sentences_path, 'r') as f:
            examples = f.readlines()
        # print(examples)
        # exit()
    # print(examples)

    with open(labels_path, 'r') as file:
        # Step 2: Load the contents of the file
        data_gt = yaml.safe_load(file)['annotations']


    tokens = []
    ner_tags = []
    origs = []
    true_labels = []

    for i in range(len(data_gt)):
        print(examples[i])
        txt = preproc_bert(examples[i])
        print(txt)
        words = txt.split(' ')
        # print(examples[i])
        nb_words = len(words)
        # # print(convert_yaml_to_json(data_gt[i]))
        labels = [0] * nb_words
        if 'structured' in data_gt[i]:
            data_gt[i] = data_gt[i]['structured']
        for anns in data_gt[i]:
            # print(anns)
            for key, val in anns.items():
                if 'norm' in key:
                    continue
                words_loc = preproc_bert(val).split(' ')
                for word in words_loc:
                    idx = words.index(word)
                    # print(idx)
                    labels[idx] = mapping[key]

        print(labels)
        print(words)

        tokens.append(words)
        ner_tags.append(labels)
        origs.append(txt)
        true_labels.append(data_gt[i])
        # input()

        # ex = str(convert_yaml_to_json(data_gt[i])).replace("'", '"')
        
    df = pd.DataFrame({'tokens': tokens, "ner_tags": ner_tags, "origs": origs,
                       "true_labels": true_labels})

    dataset = Dataset.from_pandas(df)#.train_test_split(test_size=0.2)
    # print(dataset)
    return dataset


def get_outcomes():
    outcomes = pd.read_csv('../../snomed/raw/outcomes.txt', sep='|')
    outcomes['length'] = outcomes['title'].apply(lambda x: len(x))
    outcomes.sort_values('length', ascending=False, inplace=True)
    # print(outcomes.columns)
    rows = outcomes[['title', 'id', 'nct_id']]
    print(len(rows))
    rows.loc[:, 'title'] = rows['title'].apply(lambda x: x.lower())
    # rows = rows.drop_duplicates()
    rows = rows.reset_index(drop=True)
    # print(len(rows))
    # print(rows)
    return rows


dataset = load_dataset_as_bio(
    "../data/manual-ann-ner-250.csv",
    "../data/manual-ann-ner-all.yaml"
).train_test_split(0.2, seed=0)
dataset_jb = load_dataset_as_bio(
    "../data/manual-ann-ner-250.csv",
    "../data/manual-ann-ner-all-100-jb.yaml"
).train_test_split(0.2, seed=0)

# To TEST the file
if __name__ == '__main__':
    # dataset = load_dataset_as_bio(
    #     '/gpfs/commons/groups/gursoy_lab/fpollet/Git/clinical-trial-outcome/annotations/ner/data/manual-ann-ner-250.csv',
    #     '/gpfs/commons/groups/gursoy_lab/fpollet/Git/clinical-trial-outcome/annotations/ner/data/manual-ann-ner-all.yaml'
    # )
    print(dataset)
    print(dataset_jb)