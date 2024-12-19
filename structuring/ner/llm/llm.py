# from clinical sdoh project

from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, MllamaForCausalLM, AutoProcessor

from vllm import LLM
from vllm.sampling_params import SamplingParams
import random

import json

import torch

# https://discuss.huggingface.co/t/the-effect-of-padding-side/67188

def load_model_processor(device='cuda:0', model='llama-11b'):
    if model == 'llama-11b':
        cache_dir = '/gpfs/commons/groups/gursoy_lab/fpollet/models/Llama-11B-Vision-Instruct'
        model_name = 'meta-llama/Llama-3.2-11B-Vision-Instruct'
        model = MllamaForCausalLM.from_pretrained(model_name, device_map=device, torch_dtype=torch.bfloat16, cache_dir=cache_dir)
        processor = AutoProcessor.from_pretrained(model_name, cache_dir=cache_dir, padding_side='left')

    elif model == '1b':
        # tokenizer_path = 
        # processor = AutoTokenizer.from_pretrained(tokenizer_path)
        # model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="cuda:0")
        pass
    elif model == 'llama-8b': # requires 24gb vram
        # tokenizer_path = 
        # processor = AutoTokenizer.from_pretrained(tokenizer_path)
        # model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="cuda:0")
        llama_models_path = Path('/gpfs/commons/groups/gursoy_lab/fpollet/models/Meta-Llama-3.1-8B-Instruct')
        model = LLM(model=(llama_models_path))
        processor = None

    elif model == 'ministral-8b': # requires 24gb vram
        # tokenizer_path = 
        # processor = AutoTokenizer.from_pretrained(tokenizer_path)
        # model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="cuda:0")
        mistral_models_path = Path('/gpfs/commons/groups/gursoy_lab/fpollet/models/Mistral/8B-Instruct')
        model = LLM(tokenizer_mode="mistral", config_format="mistral", load_format="mistral", model=mistral_models_path)
        processor = None

    
    return model, processor

def build_messages(outcomes, dataset=None, only_object=False, example_selection=None, n_examples=0):
    if dataset is None:
        if only_object:
            # prompt = """
            #     Extract the main objects from the given outcome, to get the general idea of it.
            #     Provide the output in JSON form with the unique key 'Quantity or Object of Interest' with a list of strings (often of length one) as value:
            #                     {
            #         "Quantity or Object of Interest": [],
            #         }
            #         Do not make any comments, give just the JSON.
            #         Text to analyze: [[sentence]] 
            # """
            # https://pmc.ncbi.nlm.nih.gov/articles/PMC11015372/
            prompt = """
    Instruction::   Extract key medical, scientific, or measurable terms or phrases ("objects") from each title. These objects should represent the primary focus, condition, measurement, or outcome mentioned in the title.
    Do not extract:
    - Descriptive words like "number of," "percentage of," "mean change," or similar modifiers.
    - Temporal or procedural phrases such as "from baseline," "over the up-titration period," or "following the onset.", "month 5", "day"
    - General terms like "participants," "subjects," or "patients" unless directly linked to a measurable or specific outcome.
    - Only keep the highest level concept. There can be several only in composite endpoints.
    Format as JSON with only one key "Quantity or Object of Interest" and for value a list of strings (often one). Do not make any comments.

    Text to analyse::  [[sentence]]"""
        else:
            prompt = """
            Extract following entities, if it exists in following text input (outcomes):
                    - Time: when
                    - Quantity Unit: measurement unit
                    - Quantity Measure (examples: Percentage of Participants, Number of Participants, Mean Change from Baseline, CHange...)
                    - Quantity or object of Interest (examples: Toxicity, Survival, specific element): the main concept(s) describing the outcome
                    - Additional constraints: details that are not measure or quantity/object of interest
                    - Quantity range: range for measurements
                    Do not duplicate elements between categories. Do not hallucinate words. Make sure to separate concepts of the same type.
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
        Text to find entities: [[sentence]]"""
    else:
        prompt = """
    Instruction::   Extract key medical, scientific, or measurable terms or phrases ("objects") from each title. These objects should represent the primary focus, condition, measurement, or outcome mentioned in the title.
    Do not extract:
    - Descriptive words like "number of," "percentage of," "mean change," or similar modifiers.
    - Temporal or procedural phrases such as "from baseline," "over the up-titration period," or "following the onset.", "month 5", "day"
    - General terms like "participants," "subjects," or "patients" unless directly linked to a measurable or specific outcome.
    - Only keep the highest level concept. There can be several only in composite endpoints.
    Format as JSON with only one key "Quantity or Object of Interest" and for value a list of strings (often one). Do not make any comments.

   """
        if example_selection == "random" and n_examples > 0:
            dataset_length = len(dataset['train'])
            print(dataset)
            print(dataset_length)
            indices = random.sample(range(dataset_length), n_examples)
            samples = dataset['train'][indices]
            # print(samples)

            examples_text = ""
            for sample in zip(samples['origs'], samples['true_labels']):
                print(sample)
                examples_text += "Outcome::\t" + sample[0] + "\n"
                if only_object:
                    examples_text += "Output::\t[" + ",".join(['{"Quantity or Object of Interest":"'+ (outcome["object"] if outcome["object"] is not None else "") + '"}'
                                                           for outcome in sample[1]]) + "] \n"
                else:
                    raise NotImplementedError()
                examples_text += "\n\n"
            # print(examples_text)

            prompt += f"Please follow the examples below:\n{examples_text}"

        prompt += f"Text to find entities: [[sentence]]"
        
        # select random examples first
        # try o1 version
        # only 50 samples
    messages = []
    for outcome in outcomes:
        conversation = [
            {  
                "role": "user",
                "content": prompt.replace("[[sentence]]", outcome)
            },
        ]
        messages.append(conversation)

    return messages

def compute(llm, messages):
    sampling_params = SamplingParams(max_tokens=8192)
    outputs = llm.chat(messages, sampling_params=sampling_params)
    outputs_clean = []
    for i in range(len(messages)):
        outputs_clean.append(outputs[i].outputs[0].text)
    return outputs_clean

def parse_outputs(outputs):
    outputs_parsed = []
    
    for output in outputs:
        try:
            i1 = output.index("{")
            i2 = output.rfind("}")
            outputs_parsed.append(json.loads(output[i1:i2+1]))
        except:
            outputs_parsed.append(None)
    
    return outputs_parsed

def build_pipeline(model, processor):
    def pipeline(sys, msg):
        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": sys}
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": msg}
                ]
            }
        ]
        text = processor.apply_chat_template(messages, add_generation_prompt=True) # https://huggingface.co/docs/transformers/main/chat_templating
        inputs = processor(text=text, return_tensors="pt").to(model.device)
        output = model.generate(**inputs, max_new_tokens=4000, do_sample=False, num_beams=1)
        return processor.decode(output[0])
    return pipeline


def format_prompt(example):
    output_texts = []
    for i in range(len(example['origs'])):
        text = f"### Question EXTRACT INFO JSON: {example['origs'][i]}\n ### Answer: {json.dumps(example['true_labels'][i], indent=4)}"
        output_texts.append(text)
    return output_texts


def build_pipeline_batch(model, processor):
    def pipeline(prompts):
        messages = [[
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": sys}
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": msg}
                ]
            }
        ] for (sys, msg) in prompts]
        # print(messages)
        formatted_prompts = [
            processor.apply_chat_template(messages, add_generation_prompt=True) for messages in messages
        ]
        # print(formatted_prompts)
        inputs = processor(text=formatted_prompts, return_tensors="pt", padding=True).to(model.device)

        # text = processor.apply_chat_template(messages, add_generation_prompt=True) # https://huggingface.co/docs/transformers/main/chat_templating
        # inputs = processor(text=text, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=4000, do_sample=False, num_beams=1)
        return [processor.decode(output).replace('<|finetune_right_pad_id|>','').replace('<|finetune_left_pad_id|>','') for output in outputs]
    return pipeline
    
# https://docs.unsloth.ai/basics/continued-pretraining
# https://huggingface.co/unsloth/Llama-3.2-11B-Vision-Instruct
# llm = load_model_processor()
# pipeline = build_pipeline_batch(*llm)
# print(pipeline([("AI expert", "count up to three"), ("AI expert", "what is white a cow related")]))

if __name__ == "__main__":
    from annotations import dataset
    messages = build_messages(
        ['a'], dataset, True, "random", 8
    )
    print(messages)
    print(format_prompt(dataset['test']))