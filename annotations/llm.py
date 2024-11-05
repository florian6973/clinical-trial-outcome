# from clinical sdoh project

from transformers import AutoModelForCausalLM, BitsAndBytesConfig, MllamaForCausalLM, AutoProcessor

import torch

# https://discuss.huggingface.co/t/the-effect-of-padding-side/67188

def load_model_processor():
    cache_dir = '/gpfs/commons/groups/gursoy_lab/fpollet/models/Llama-11B-Vision-Instruct'
    model_name = 'meta-llama/Llama-3.2-11B-Vision-Instruct'
    model = MllamaForCausalLM.from_pretrained(model_name, device_map="cuda:1", torch_dtype=torch.bfloat16, cache_dir=cache_dir)
    processor = AutoProcessor.from_pretrained(model_name, cache_dir=cache_dir, padding_side='left')

    return model, processor

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