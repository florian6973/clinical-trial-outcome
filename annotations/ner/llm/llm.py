# from clinical sdoh project

from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, MllamaForCausalLM, AutoProcessor

from vllm import LLM
from vllm.sampling_params import SamplingParams

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

def build_messages(outcomes):
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