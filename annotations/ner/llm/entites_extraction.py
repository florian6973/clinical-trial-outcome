
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
        """Based on these examples:
  0: Composite Endpoint of CHD Death, Non-fatal MI, or Ischemic Stroke
    partial: false
    structured:
      - object: CHD Death
      - object: Non-fatal MI
      - object: Ischemic Stroke
  1: Number of Acute Febrile Illness Due to Laboratory Confirmed Dengue Presenting a Sign or a Symptom of Interest (Grade 3) During the 7-day Period Following the Onset of Each Episode of AFI Due to LCD
    partial: true
    structured:
      - object: Acute Febrile Illness
        time: 7-day Period Following the Onset
        measure: Number
  100: AFI a weeks 5 and 10
    partial: false
    structured:
      - object: Acute Febrile Illness
        time: Week 5
      - object: Acute Febrile Illness
        time: Week 10

Extract the main information from the outcome and format it as YAML. The general structure to KEEP and follow is (some fields can be empty.
Partial is true if you only covered part of the outcome in your structured section.
You can include several items in the structured section, for instance for different objects or timeframes.
Make sure to remove any necessary pronouns or connectors.
id:
    partial: true/false
    structured:
        - object:
          measure:
          time:
          specifier:
          range:
          unit:


Outcome to tackle: """ + str(i) + " = " + row[0])
    
    print(res[res.find('<|end_header_id|>'):])
    # input()
    outputs.append(res)
    # break
    
with open('outputs.json', 'w') as f:
    json.dump(outputs, f)