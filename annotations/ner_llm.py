
from llm import load_model_processor, build_pipeline, build_pipeline_batch

# https://arxiv.org/pdf/2304.10428

import numpy as np
import json
from tqdm import tqdm
import pandas as pd
import sys

rows = np.load("data-ann.npz", allow_pickle=True)['rows']

all_mode = False
if len(sys.argv) > 1:
    if sys.argv[1] == 'all':
        all_mode = True
        outcomes = pd.read_csv('outcomes.txt', sep='|')
        rows = outcomes['title']


llm = load_model_processor()
pipeline = build_pipeline(*llm)
pipeline_batch = build_pipeline_batch(*llm)

system_msg = "You are a clinical expert in named entity extraction."

outputs = []
outputs_json = []
batch_size = 5

def save():
    with open(f'outputs_{all_mode}.json', 'w') as f:
        json.dump(outputs, f)
        
    with open(f'outputs_{all_mode}_json.json', 'w') as f:
        json.dump(outputs_json, f)

# for i, row in tqdm(enumerate(rows), total=len(rows)):
for i in tqdm(range(len(rows), batch_size)):
    row = rows[i]
    if not all_mode:
        print(i)
        print("------")
        print(row[0])
        txt = row[0]
    else:
        txt = row
    

    res = pipeline(
        system_msg,
        """Extract following entities, if it exists in following text input (outcomes):
             - Time: when
             - Quantity Unit: measurement unit
             - Quantity Measure (examples: Percentage of Participants, Number of Participants, Mean Change from Baseline, CHange...)
             - Quantity or object of Interest (examples: Toxicity, Survival, specific element): the main concept(s) describing the outcome
             - Additional constraints: details that are not measure or quantity/object of interest
             - Quantity range: range for measurements
             Do not duplicate elements between categories, each word have maximum one label. Do not hallucinate words. 
             Additional Constraints should be used only if it does not fit in Time or Range.
             Avoid abbreviations.
             Provide the output in JSON form like 
             {
  "Time": [],
  "Quantity Unit": [],
  "Quantity Measure": [],
  "Quantity or Object of Interest": [],
  "Additional Constraints": [],
  "Quantity Range": []
}
Text to find entities: """ + txt) #  Make sure to associate everything to a category.
    
    # print(res[res.find('<|end_header_id|>'):])
    try:
        print(txt)
        txt = res[res.find('assistant<|end_header_id|>')+26:-10].strip()
        print("BEGIN", txt, "END")
        res_json = json.loads(txt)
        outputs_json.append(res_json)
    except Exception as e:
        print(e)
        print("Could not parse")
        outputs_json.append({"error": True})
    # input()
    outputs.append(res)
    # break
    if i % 100 == 0:
        save()

save()


# https://arxiv.org/pdf/2304.10428
# Improving large language models for clinical named entity
# recognition via prompt engineering https://watermark.silverchair.com/ocad259.pdf?token=AQECAHi208BE49Ooan9kkhW_Ercy7Dm3ZL_9Cf3qfKAc485ysgAAA3AwggNsBgkqhkiG9w0BBwagggNdMIIDWQIBADCCA1IGCSqGSIb3DQEHATAeBglghkgBZQMEAS4wEQQMqH46CjuDauXu1oy2AgEQgIIDI8qt7RKX9QdpBoAtTaaVxHoOvc_kwuGzNaB9usOwYoQ5b1qYMKIJcbHuGE1jwSlBVgjbnA9xvbeh5metxDdDgP52iI78IZTxJATDwY4vf3jA1SiE0HiltB4Pb9GGYHBfVGUy9YgBDcE2JSc-QtRD_VeujKl-R4CrAcclG9o0JBZ6uEPDM45WuUr_8HzavsOITDkEaexs0SBkYkPk1gDvcoWkbJBn0Gx5u6rMbByHm6-_8AMNvtf_9XRLk90Fy-ZYEcxbuVlSDiDXYBIeSBi_d0NJQTZ2-TccOkDh2UrjtExmhGzsgFhHzcVLK2_Jo0UFWgcLg8TiU0j0EON7ot2-EHhYJmqmANkU5kUDkQcSJ9jKXCbmmeqfHtlQogmVGpdg4Ej9bd50oLuOwwKvOK07gGXP7Ukx_5eGjTkAHzzNH_ZoRT5EbB_D9cINCsp2Ez4SzSSa8I2TgyYmSmI8MovpDUjVMINU8YV3HGSBAFkSKzdhuSvVkhSzj_iygcQ_0QrOZPyaz1ZxKpg26kQihk6Y-4vPVESlWLD65vjTifbGSSzs4lrUsFhkln3rTJmJs-56orzKMy-aE5uNZ0ynifyQZ7FRmWI6hlcCVDHzXGrKXluMgPB6qRQ4fv2rfKm0iEFH1tc3Qv3s30ZRjtowsTL1klSdiNKZ6tP3fFJYcHokIgIxVTLySngh_vXJYD8XwR5hMrjWHPMOtgIrCdOwNYvB7t5yBkeemRyazF3hB3e4DpwAsquGAT8Frm6V_oXUqoxhyDpytbujNJpr0TDOOowP_ovmAaNhNRFtvC-PviSc3MSR5_IWCv1sA7XBM0Xmlh85JtOyeyGuiWoLGp77N72nhgLoB4pS0zC1Hum63i1pEKvR5piCY5fyPAhgiva7sLgLk06XlSF_epsxg401UitxaCYIQ2x6zFVbzS8f-lqRpURpu4HM391QdpZoEcbbE5w_dDrOjLMlhTP7HQzE0twUNh_1_qMkIkIUYJ03INwzozAwFuxff34OYhP49hO1Fi4pB93vFDvA1Q35V_PfIh4OMGPj7rrMFFlh6Fke7Hc6k0yoB5ct