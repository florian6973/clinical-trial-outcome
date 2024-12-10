# need bio labels for kappa assignment for classification problems
# token level

# load dataset as bio from me and him
import sys
import numpy as np
import sklearn.metrics as sm
import pandas as pd
sys.path.append("../llm")
from annotations import dataset, dataset_jb, load_dataset_as_bio

tags = load_dataset_as_bio(
    "../data/manual-ann-ner-250.csv",
    "../data/manual-ann-ner-fp.yaml"
)[:101]['ner_tags']
#['ner_tags'] #[:101]['ner_tags']
tags_10 = [[int(tag == 1) for tag in tag_list] for tag_list in tags]
tags_10f = [item for sublist in tags_10 for item in sublist]


tags_jb = load_dataset_as_bio(
    "../data/manual-ann-ner-250.csv",
    "../data/manual-ann-ner-all-100-jb-m.yaml"
    # "../data/manual-ann-ner-all-100-jb-m.yaml"
    # "../data/manual-ann-ner-all-100-jb-m.yaml"
)['ner_tags']
tags_10jb = [[int(tag == 1) for tag in tag_list] for tag_list in tags_jb]
tags_jbf = [item for sublist in tags_10jb for item in sublist]

print(tags_10f)
print(tags_jbf)

print(sm.cohen_kappa_score(tags_10f, tags_jbf))
# exit()


file_jb =  "../data/group-ann-jb.csv"
file_fp = "../data/group-ann-fp.csv"

fjb = pd.read_csv(file_jb)['GROUP']
ffp = pd.read_csv(file_fp)['GROUP']


all_categories = pd.Categorical(pd.concat([fjb, ffp])).categories

# Create a categorical mapping
category_mapping = {category: code for code, category in enumerate(all_categories)}


# Apply the mapping to both columns
fjbf = fjb.map(category_mapping)
ffpf = ffp.map(category_mapping)
fjbf = fjbf.fillna(-1).astype(int)
ffpf = ffpf.fillna(-1).astype(int)

fjbf.to_csv('test.csv')



print(fjbf)
print(ffpf)


print(sm.cohen_kappa_score(fjbf, ffpf))