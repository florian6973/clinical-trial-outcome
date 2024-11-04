# from clinical sdoh project

from transformers import AutoModelForCausalLM, BitsAndBytesConfig, MllamaForCausalLM, AutoProcessor

import torch

def load_model_processor():
    cache_dir = '/nlp/tools/Llama-3.2-11B-Vision-Instruct/'
    model_name = 'meta-llama/Llama-3.2-11B-Vision-Instruct'
    model = MllamaForCausalLM.from_pretrained(model_name, device_map="cuda:0", torch_dtype=torch.bfloat16, cache_dir=cache_dir)
    processor = AutoProcessor.from_pretrained(model_name, cache_dir=cache_dir)

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
    
