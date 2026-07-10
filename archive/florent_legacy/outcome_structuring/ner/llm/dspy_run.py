# loaded_people_extractor(tokens=["Italy", "recalled", "Marcello", "Cuttitta"]).extracted_people



import os
import tempfile
from datasets import load_dataset
from typing import Dict, Any, List
import dspy


from annotations import dataset
from Levenshtein import distance

from dspy_test import ObjectExtraction, test_set
from typing import List

# lm = dspy.LM("openai/meta-llama/Llama-3.1-8B-Instruct",
#              api_base="http://localhost:7501/v1",  # ensure this points to your port
#              api_key="local", model_type='chat')
# dspy.configure(lm=lm)

loaded_people_extractor = dspy.ChainOfThought(ObjectExtraction)
loaded_people_extractor.load("optimized_extractor.json")

for row in test_set:
    print(" ".join(row['tokens']), loaded_people_extractor(tokens=row['tokens']).extracted_objects)
