
from llm import load_model_processor, build_pipeline

# https://arxiv.org/pdf/2304.10428

import numpy as np
import json

rows = np.load("data-ann.npz", allow_pickle=True)['rows']
with open('prompt.txt', 'r') as f:
    prompt = f.read()

llm = load_model_processor()
pipeline = build_pipeline(*llm)

system_msg = "You are a clinical expert in named entity extraction."

outputs = []
outputs_json = []
for i, row in enumerate(rows):
    print(i)
    print("------")
    print(row[0])

    res = pipeline(
        system_msg,
        prompt.replace('[[text]]', row[0])) #  Make sure to associate everything to a category.
    
    # print(res[res.find('<|end_header_id|>'):])
    try:
        txt = res[res.find('assistant<|end_header_id|>')+26:-10].strip()
        print("BEGIN", txt, "END")
        # res_json = json.loads(txt)
        outputs_json.append(txt)
    except Exception as e:
        print(e)
        print("Could not parse")
        outputs_json.append({"error": True})
    # input()
    outputs.append(res)
    # break
    
with open('outputs.json', 'w') as f:
    json.dump(outputs, f)
    
with open('outputs_html.json', 'w') as f:
    json.dump(outputs_json, f)

# https://arxiv.org/pdf/2304.10428
# Improving large language models for clinical named entity
# recognition via prompt engineering https://watermark.silverchair.com/ocad259.pdf?token=AQECAHi208BE49Ooan9kkhW_Ercy7Dm3ZL_9Cf3qfKAc485ysgAAA3AwggNsBgkqhkiG9w0BBwagggNdMIIDWQIBADCCA1IGCSqGSIb3DQEHATAeBglghkgBZQMEAS4wEQQMqH46CjuDauXu1oy2AgEQgIIDI8qt7RKX9QdpBoAtTaaVxHoOvc_kwuGzNaB9usOwYoQ5b1qYMKIJcbHuGE1jwSlBVgjbnA9xvbeh5metxDdDgP52iI78IZTxJATDwY4vf3jA1SiE0HiltB4Pb9GGYHBfVGUy9YgBDcE2JSc-QtRD_VeujKl-R4CrAcclG9o0JBZ6uEPDM45WuUr_8HzavsOITDkEaexs0SBkYkPk1gDvcoWkbJBn0Gx5u6rMbByHm6-_8AMNvtf_9XRLk90Fy-ZYEcxbuVlSDiDXYBIeSBi_d0NJQTZ2-TccOkDh2UrjtExmhGzsgFhHzcVLK2_Jo0UFWgcLg8TiU0j0EON7ot2-EHhYJmqmANkU5kUDkQcSJ9jKXCbmmeqfHtlQogmVGpdg4Ej9bd50oLuOwwKvOK07gGXP7Ukx_5eGjTkAHzzNH_ZoRT5EbB_D9cINCsp2Ez4SzSSa8I2TgyYmSmI8MovpDUjVMINU8YV3HGSBAFkSKzdhuSvVkhSzj_iygcQ_0QrOZPyaz1ZxKpg26kQihk6Y-4vPVESlWLD65vjTifbGSSzs4lrUsFhkln3rTJmJs-56orzKMy-aE5uNZ0ynifyQZ7FRmWI6hlcCVDHzXGrKXluMgPB6qRQ4fv2rfKm0iEFH1tc3Qv3s30ZRjtowsTL1klSdiNKZ6tP3fFJYcHokIgIxVTLySngh_vXJYD8XwR5hMrjWHPMOtgIrCdOwNYvB7t5yBkeemRyazF3hB3e4DpwAsquGAT8Frm6V_oXUqoxhyDpytbujNJpr0TDOOowP_ovmAaNhNRFtvC-PviSc3MSR5_IWCv1sA7XBM0Xmlh85JtOyeyGuiWoLGp77N72nhgLoB4pS0zC1Hum63i1pEKvR5piCY5fyPAhgiva7sLgLk06XlSF_epsxg401UitxaCYIQ2x6zFVbzS8f-lqRpURpu4HM391QdpZoEcbbE5w_dDrOjLMlhTP7HQzE0twUNh_1_qMkIkIUYJ03INwzozAwFuxff34OYhP49hO1Fi4pB93vFDvA1Q35V_PfIh4OMGPj7rrMFFlh6Fke7Hc6k0yoB5ct