

categories = """
    0 Mortality - This outcome category focuses on death rates or time until death as the primary measure. Clinical trials often assess whether an intervention reduces overall mortality or the risk of death from a specific cause.
    1 Infection - This category includes the incidence, rate, or severity of infections. Studies might measure how well an intervention prevents infections, reduces their frequency, or shortens their duration.
    2 Pain - Pain-related outcomes assess changes in pain intensity, frequency, or severity. Trials evaluate if a treatment alleviates pain, improves pain management, or reduces patients' reliance on pain-related medications.
    3 Mental health - Outcomes in this category cover emotional well-being, including changes in depression, anxiety, mood disorders, or psychological distress. Researchers examine whether an intervention improves mental health scores or reduces symptoms of mental illness.
    4 Function - Function encompasses patients' ability to perform daily activities or maintain mobility. This can include improvements in walking, self-care, work capability, or performance in activities of daily living.
    5 Quality of Life - Quality of Life measures focus on overall well-being, satisfaction, and comfort. Trials may use standardized questionnaires to assess changes in patients' physical, emotional, and social functioning, as well as their perceived life satisfaction.
    6 Psychosocial - Psychosocial outcomes look at social interaction, family dynamics, community engagement, self-esteem, and coping skills. These outcomes help determine if a treatment influences a patient's social roles, relationships, and personal identity.
    7 Compliance with treatment - This category measures how consistently and accurately patients follow prescribed therapies, medication regimens, or lifestyle recommendations. It can also examine factors that improve adherence and barriers that reduce compliance.
    8 Adverse events - Adverse event outcomes track the occurrence, severity, and types of unwanted side effects or complications related to an intervention. By assessing adverse events, researchers gauge the safety profile of a treatment.
    9 Satisfaction with care - This outcome is about patients' subjective assessment of their treatment experience. It includes satisfaction with healthcare providers, communication, convenience, and the overall healthcare process.
    10 Resource use - These outcomes focus on healthcare utilization, such as hospital stays, outpatient visits, medication consumption, and costs. Trials may measure whether an intervention is more or less resource-intensive compared to standard care.
    11 Device/intervention failure - This category examines the reliability and durability of a medical device or the effectiveness of a specific procedure. It tracks how often a device malfunctions, requires replacement, or if an intervention does not produce the intended effect.
"""

# category_clean = """
#     0 Mortality
#     1 Infection
#     2 Pain
#     3 Mental health
#     4 Function
#     5 Quality of Life
#     6 Psychosocial
#     7 Compliance with treatment 
#     8 Adverse events
#     9 Satisfaction with care
#     10 Resource use
#     11 Device/intervention failure 
#     12 Physiological or clinical
# """

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, MllamaForCausalLM, AutoProcessor

from vllm import LLM
from vllm.sampling_params import SamplingParams
import random

from pathlib import Path
from tqdm import tqdm

# vllm

llama_models_path = Path('/gpfs/commons/groups/gursoy_lab/fpollet/models/Meta-Llama-3.1-8B-Instruct')
llm = LLM(model=(llama_models_path))

prompt = """
The outcomes of clinical trials have been extracted. They can be classified in the following categories

    0 Mortality - This outcome category focuses on death rates or time until death as the primary measure. Clinical trials often assess whether an intervention reduces overall mortality or the risk of death from a specific cause.
    1 Infection - This category includes the incidence, rate, or severity of infections. Studies might measure how well an intervention prevents infections, reduces their frequency, or shortens their duration.
    2 Pain - Pain-related outcomes assess changes in pain intensity, frequency, or severity. Trials evaluate if a treatment alleviates pain, improves pain management, or reduces patients' reliance on pain-related medications.
    3 Mental health - Outcomes in this category cover emotional well-being, including changes in depression, anxiety, mood disorders, or psychological distress. Researchers examine whether an intervention improves mental health scores or reduces symptoms of mental illness.
    4 Function - Function encompasses patients' ability to perform daily activities or maintain mobility. This can include improvements in walking, self-care, work capability, or performance in activities of daily living.
    5 Quality of Life - Quality of Life measures focus on overall well-being, satisfaction, and comfort. Trials may use standardized questionnaires to assess changes in patients' physical, emotional, and social functioning, as well as their perceived life satisfaction.
    6 Psychosocial - Psychosocial outcomes look at social interaction, family dynamics, community engagement, self-esteem, and coping skills. These outcomes help determine if a treatment influences a patient's social roles, relationships, and personal identity.
    7 Compliance with treatment - This category measures how consistently and accurately patients follow prescribed therapies, medication regimens, or lifestyle recommendations. It can also examine factors that improve adherence and barriers that reduce compliance.
    8 Adverse events - Adverse event outcomes track the occurrence, severity, and types of unwanted side effects or complications related to an intervention. By assessing adverse events, researchers gauge the safety profile of a treatment.
    9 Satisfaction with care - This outcome is about patients' subjective assessment of their treatment experience. It includes satisfaction with healthcare providers, communication, convenience, and the overall healthcare process.
    10 Resource use - These outcomes focus on healthcare utilization, such as hospital stays, outpatient visits, medication consumption, and costs. Trials may measure whether an intervention is more or less resource-intensive compared to standard care.
    11 Device/intervention failure - This category examines the reliability and durability of a medical device or the effectiveness of a specific procedure. It tracks how often a device malfunctions, requires replacement, or if an intervention does not produce the intended effect.

Predict the category of the following object::\t`[[sentence]]`
Only answer with the category number, no explanation and no comment.
Do not be confused if any number is included in the object to classify.
"""

import pandas as pd

outcomes_df = pd.read_csv('outcomes.csv')
sampling_params = SamplingParams(max_tokens=8192)

clses = []
# for index, row in tqdm(outcomes_df.iterrows(), total=len(outcomes_df)):
#     nct_id = row['nct_id']
#     obj = row['object']

#     messages = []
#     conversation = [
#         {  
#             "role": "user",
#             "content": prompt.replace("[[sentence]]", obj)
#         },
#     ]
#     messages.append(conversation)

#     outputs = llm.chat(messages, sampling_params=sampling_params)
#     outputs_clean = []
#     for i in range(len(messages)):
#         outputs_clean.append(outputs[i].outputs[0].text)
#     clses.append((nct_id, obj, outputs_clean[-1]))    
import numpy as np
batch_size = 100  # Batch size for inference

# Split data into batches
num_batches = int(np.ceil(len(outcomes_df) / batch_size))

output_file = "results_cls-11.csv"

for batch_idx in tqdm(range(num_batches), desc="Processing Batches"):
    # if batch_idx < 2000:
    #     continue
    # Get the batch data
    start_idx = batch_idx * batch_size
    end_idx = min((batch_idx + 1) * batch_size, len(outcomes_df))
    batch_df = outcomes_df.iloc[start_idx:end_idx]

    # Prepare messages for the batch
    messages = []
    nct_ids = []
    objs = []

    for index, row in batch_df.iterrows():
        nct_id = row['nct_id']
        obj = row['object']
        nct_ids.append(nct_id)
        objs.append(obj)

        conversation = [
            {
                "role": "user",
                "content": prompt.replace("[[sentence]]", obj)
            }
        ]
        messages.append(conversation)

    # Perform inference for the batch
    outputs = llm.chat(messages, sampling_params=sampling_params)

    # Collect and clean outputs
    for i in range(len(messages)):
        output_clean = outputs[i].outputs[0].text
        clses.append((nct_ids[i], objs[i], output_clean))

    if batch_idx % 10 == 0:
        pd.DataFrame(clses, columns=['nct_id', 'object', 'class']).to_csv(output_file)

pd.DataFrame(clses, columns=['nct_id', 'object', 'class']).to_csv(output_file)
