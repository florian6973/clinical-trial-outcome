# need bio labels for kappa assignment for classification problems
# token level

# load dataset as bio from me and him
import sys
import numpy as np
import sklearn.metrics as sm
sys.path.append("../llm")
from annotations import dataset, dataset_jb, load_dataset_as_bio

tags = load_dataset_as_bio(
    "../data/manual-ann-ner-250.csv",
    "../data/manual-ann-ner-all.yaml"
)[:101]['ner_tags']
tags_10 = [[int(tag == 1) for tag in tag_list] for tag_list in tags]
tags_10f = [item for sublist in tags_10 for item in sublist]


tags_jb = load_dataset_as_bio(
    "../data/manual-ann-ner-250.csv",
    "../data/manual-ann-ner-all-100-jb-m.yaml"
)['ner_tags']
tags_jbf = [item for sublist in tags_jb for item in sublist]

print(tags_10f)
print(tags_jbf)

print(sm.cohen_kappa_score(tags_10f, tags_jbf))