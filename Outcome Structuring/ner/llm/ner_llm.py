from llm import load_model_processor, compute, build_pipeline, build_pipeline_batch, build_messages, parse_outputs, format_prompt
from annotations import get_outcomes

from vllm import LLM
from vllm.sampling_params import SamplingParams

import json

from tqdm import tqdm


# batch inference https://medium.com/@wearegap/a-brief-introduction-to-optimized-batched-inference-with-vllm-deddf5423d0c

def train_llm(model_name, method, dataset):
    # supervised finetuning with lora
    # TODO
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
    import torch

    # Load pre-trained LLaMA model and tokenizer
    # model_name = "/gpfs/commons/groups/gursoy_lab/fpollet/models/Meta-Llama-3.1-8B-Instruct"  # Replace with a suitable model
    model_name = "/gpfs/commons/groups/gursoy_lab/fpollet/models/Llama-3.2-1B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name,  torch_dtype=torch.bfloat16)
    tokenizer.pad_token = tokenizer.eos_token 
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)

    # Configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,  # Type of task
        inference_mode=False,          # Set to False for training
        r=64,                           # Low-rank parameter
        lora_alpha=128,                 # Scaling factor
        lora_dropout=0.1               # Dropout rate
    )

    # Apply LoRA configuration to the model
    model = get_peft_model(model, lora_config)

    # reply as json
    # example: stanfordnlp/imdb

    # Load dataset
    # dataset = load_dataset("yelp_review_full")  # Replace with your dataset
    # tokenized_dataset = dataset.map(
    #     lambda x: tokenizer(x["text"], truncation=True, padding="max_length"), 
    #     batched=True
    # )
    tokenized_dataset = dataset
    # https://huggingface.co/docs/trl/sft_trainer

    # Set training arguments
    # training_args = TrainingArguments(
    #     output_dir="./llama-lora",
    #     evaluation_strategy="epoch",
    #     save_strategy="epoch",
    #     learning_rate=5e-5,
    #     per_device_train_batch_size=8,
    #     num_train_epochs=3,
    #     weight_decay=0.01,
    #     logging_dir="./logs",
    #     logging_steps=10,
    #     save_total_limit=2,
    #     report_to="none"
    # )

    # # Initialize SFTTrainer
    # trainer = SFTTrainer(
    #     model=model,
    #     train_dataset=tokenized_dataset["train"],
    #     eval_dataset=tokenized_dataset["test"],
    #     tokenizer=tokenizer,
    #     args=training_args
    # )

    # Train the model
    # trainer.train()

    response_template = " ### Answer:"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    trainer = SFTTrainer(
        model,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        args=SFTConfig(output_dir="./tmp-sft",
                       dataset_batch_size=10,
                       num_train_epochs=10,
                       fp16=True,
                    #    output_dir="./llama-lora",
                        evaluation_strategy="epoch",
                        save_strategy="epoch",
                        # learning_rate=5e-5,
                        # per_device_train_batch_size=8,
                        # num_train_epochs=3,
                        # weight_decay=0.01,
                        # logging_dir="./logs",
                        # logging_steps=10,
            max_seq_length=512,),
        # args=training_args,
        formatting_func=format_prompt,
        data_collator=collator,

    )
    trainer.train()

    # Save the LoRA-adapted model
    model.save_pretrained("./llama-lora-adapted")

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