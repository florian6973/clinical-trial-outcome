import json

with open('outputs_True_json.json', 'r') as f:
    data = json.load(f)

print(len(data))