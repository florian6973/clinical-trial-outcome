from llm import load_model_processor, compute, build_pipeline, build_pipeline_batch, build_messages, parse_outputs
from annotations import get_outcomes

from vllm import LLM
from vllm.sampling_params import SamplingParams

import json

from tqdm import tqdm


# batch inference https://medium.com/@wearegap/a-brief-introduction-to-optimized-batched-inference-with-vllm-deddf5423d0c

def train_llm(model_name, method, dataset):
    # finetuning
    pass

def infer_llm(model_name, dataset):
    # dataset is for ICL
    rows = get_outcomes()
    
    llm, _ = load_model_processor(model=model_name)

    batch_size = 50
    outputs_outcomes = []
    outputs_parsed = []
    outcomes = []
    metadata = []

    # batch of 50
    # for row in dataset['test']:
    for i in tqdm(range(0, len(rows), batch_size)):
        dataset_tmp = rows.iloc[i:min(len(rows), i+batch_size)]
        # print(dataset_tmp)
        outcomes_tmp = dataset_tmp['title']
        msgs = build_messages(outcomes_tmp)

        outputs = compute(llm, msgs)

        outcomes.extend(outcomes_tmp)
        outputs_parsed.extend(parse_outputs(outputs))
        outputs_outcomes.extend(outputs)
        metadata.extend(list(zip(dataset_tmp['id'].astype(int), dataset_tmp['nct_id'])))

        with open(f'outputs/llm_infer_{model_name}.json', 'w') as f:
            json.dump(list(zip(outcomes, outputs_parsed, outputs_outcomes, metadata)), f)

# variant with object only 

def eval_llm(model_name, dataset, settings):
    # dataset is for ICL
    llm, _ = load_model_processor(model=model_name)

    batch_size = 50
    outputs_outcomes = []
    outputs_parsed = []
    outcomes = []

    # batch of 50
    # for row in dataset['test']:
    for i in tqdm(range(0, len(dataset['test']), batch_size)):
        dataset_tmp = dataset['test'][i:min(len(dataset['test']), i+batch_size)]
        
        # print(dataset_tmp)
        
        outcomes_tmp = dataset_tmp['origs']
        msgs = build_messages(outcomes_tmp, dataset, **settings)

        outputs = compute(llm, msgs)

        outcomes.extend(outcomes_tmp)
        outputs_parsed.extend(parse_outputs(outputs))
        outputs_outcomes.extend(outputs)

    with open(f'outputs/llm_evaluate_{model_name}_{settings}.json', 'w') as f:
        json.dump(list(zip(outcomes, outputs_parsed, outputs_outcomes, [None]*len(outcomes))), f)

    # # chat vs generate method
    # outputs = llm.chat(messages, sampling_params=sampling_params)

    # print(outputs[0].outputs[0].text)
    # print(outputs[1].outputs[0].text)

    # special characters
    # \u2264 <=
    # \u2265 >=