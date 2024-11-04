
from llm import load_model_processor, build_pipeline

import numpy as np
import json

rows = np.load("data-ann.npz", allow_pickle=True)['rows']


llm = load_model_processor()
pipeline = build_pipeline(*llm)

system_msg = "You are a clinical expert."

outputs = []
for i, row in enumerate(rows):
    print(i)
    print("------")
    print(row[0])

    res = pipeline(
        system_msg,
        """Extract following entities if exist in following text input:
             - Time
             - Unit
             - Quantity specifier
             - Quantity of Interest
             - Additional constraints
             - Range
             Provide the output in JSON form
Text to find entities: """ + str(i) + " = " + row[0])
    
    print(res[res.find('<|end_header_id|>'):])
    # input()
    outputs.append(res)
    # break
    
with open('outputs.json', 'w') as f:
    json.dump(outputs, f)