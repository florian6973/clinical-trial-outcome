import json

with open('outputs.json', 'r') as f:
    data = json.load(f)

import numpy as np
rows = np.load("data-ann.npz", allow_pickle=True)['rows']

with open('data-ann.txt', 'w') as f:
    for res in rows:
        f.write(res[0] + '\n')


for i, res in enumerate(data):
    print('---------------')
    print(i, rows[i][0])
    print(res[res.find('<|start_header_id|>assistant<|end_header_id|>'):])