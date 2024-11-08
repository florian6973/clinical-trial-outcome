import json

with open('perf_4.json', 'r') as f:
    data = json.load(f)

import matplotlib.pyplot as plt

xs = [int(k) for k in data.keys()]

for yl in ['eval_f1', 'eval_recall', 'eval_precision']:
    ys = [v[yl] for v in data.values()]

    plt.plot(xs, ys, label=yl)

plt.xlabel('Number of training samples')
plt.ylabel('Performance on 50 test samples')
plt.legend()
plt.tight_layout()
plt.savefig('perf_4.png')