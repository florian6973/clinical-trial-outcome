from huggingface_hub import snapshot_download
from pathlib import Path

from vllm import LLM
from vllm.sampling_params import SamplingParams

llama_models_path = Path('/gpfs/commons/groups/gursoy_lab/fpollet/models/Meta-Llama-3.1-8B-Instruct')
llama_models_path.mkdir(parents=True, exist_ok=True)

# snapshot_download(repo_id="mistralai/Ministral-8B-Instruct-2410", allow_patterns=["params.json", "consolidated.safetensors", "tekken.json"], local_dir=mistral_models_path)



# model_name = "mistralai/Ministral-8B-Instruct-2410"

sampling_params = SamplingParams(max_tokens=8192)

# note that running Ministral 8B on a single GPU requires 24 GB of GPU RAM
# If you want to divide the GPU requirement over multiple devices, please add *e.g.* `tensor_parallel=2`
llm = LLM(model=llama_models_path)

prompt = [
"Do we need to think for 10 seconds to find the answer of 1 + 1?",
"what is the meaning of life?"
]

messages = [[
    {
        "role": "user",
        "content": prompt[0]
    },
],[
    {
        "role": "user",
        "content": prompt[1]
    },
]
]

# chat vs generate method
outputs = llm.chat(messages, sampling_params=sampling_params)

print(outputs[0].outputs[0].text)
print(outputs[1].outputs[0].text)