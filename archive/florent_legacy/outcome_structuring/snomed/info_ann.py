# import numpy as np

# file = '/nlp/projects/llama/outputs/top5_snomed_matches.csv'

# import pandas as pd

# csv = pd.read_csv(file)
# # print(csv.index)

# idxes = np.loadtxt("selected-idxes.txt").astype(int)
# array_src = np.memmap("array-outcome.dat", mode='r', shape=(438026, 4096), dtype='float16')
# label_src = np.memmap("label-outcome.dat", mode='r', shape=(438026,), dtype='U1000')

# rows = []
# for idx in idxes:
#     print(label_src[idx])
#     rows.append(csv.loc[idx].values)
#     print(rows[-1])
#     print()

# np.savez("data-ann.npz", rows=rows)

import numpy as np
rows = np.load("data-ann.npz", allow_pickle=True)['rows']

for i, row in enumerate(rows):
    print(i)
    print("------")
    print(row)
    # input()