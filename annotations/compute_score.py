import yaml
import numpy as np

with open('manual-ann-snomed.yaml', 'r') as f:
    anns = yaml.safe_load(f)

rows = np.load("data-ann.npz", allow_pickle=True)['rows']

def compute_complete_concepts(anns):
    count = 0
    for idx in anns['annotations']:
        ann = anns['annotations'][idx]
        if not ann['missing']:
            count += 1
    return count, count / len(anns['annotations']) * 100

def compute_score_mapping(anns, rows, top_k=1):
    count = 0
    count_resolved = 0

    for idx, row in enumerate(rows):
        if idx not in anns['annotations']:
            break

        has_match = False

        # print(idx, row[1], row[1+4])
        if 'concepts' in anns['annotations'][idx] and anns['annotations'][idx]['concepts'] is not None and len(anns['annotations'][idx]['concepts']) > 0:
            count_resolved += 1
            tgt_concepts = [concept['concept_id'] for concept in anns['annotations'][idx]['concepts']]
            # print(tgt_concepts)
            for k, i in enumerate(range(1, len(row), 4)):
                if k >= top_k:
                    break
                if row[i] in tgt_concepts:
                    has_match = True
        
        if has_match:
            count += 1

    return count, count / count_resolved * 100, count_resolved, count_resolved / len(anns['annotations']) * 100

    

print("Complete concepts: {0}, {1:.2f} %".format(*compute_complete_concepts(anns)))
print("Found concepts (manual): {2}, {3:.2f} %".format(*compute_score_mapping(anns, rows, 1)))
print("Found concepts (top-1): {0}, {1:.2f} %".format(*compute_score_mapping(anns, rows, 1)))
print("Found concepts (top-5): {0}, {1:.2f} %".format(*compute_score_mapping(anns, rows, 5)))