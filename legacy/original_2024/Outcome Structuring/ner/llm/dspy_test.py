import os
import tempfile
from datasets import load_dataset
from typing import Dict, Any, List
import dspy

# CUDA_VISIBLE_DEVICES=1 python -m sglang.launch_server --port 7501 --model-path /gpfs/commons/groups/gursoy_lab/fpollet/models/Meta-Llama-3.1-8B-Instruct

# https://github.com/stanfordnlp/dspy/blob/main/examples/migration.ipynb
# https://learnbybuilding.ai/tutorials/a-gentle-introduction-to-dspy

from annotations import dataset
from Levenshtein import distance

from typing import List

lm = dspy.LM("openai/meta-llama/Llama-3.1-8B-Instruct",
             api_base="http://localhost:7501/v1",  # ensure this points to your port
             api_key="local", model_type='chat')
dspy.configure(lm=lm)


def extract_people_entities(data_row: Dict[str, Any]) -> List[str]:
    """
    Extracts entities referring to people from a row of the CoNLL-2003 dataset.
    
    Args:
        data_row (Dict[str, Any]): A row from the dataset containing tokens and NER tags.
    
    Returns:
        List[str]: List of tokens tagged as people.
    """
    return [
        token
        for token, ner_tag in zip(data_row["tokens"], data_row["ner_tags"])
        if ner_tag in [1]  # CoNLL entity codes 1 and 2 refer to people
    ]

def prepare_dataset(data_split, start: int, end: int) -> List[dspy.Example]:
    """
    Prepares a sliced dataset split for use with DSPy.
    
    Args:
        data_split: The dataset split (e.g., train or test).
        start (int): Starting index of the slice.
        end (int): Ending index of the slice.
    
    Returns:
        List[dspy.Example]: List of DSPy Examples with tokens and expected labels.
    """
    return [
        dspy.Example(
            tokens=row["tokens"],
            expected_extracted_objects=extract_people_entities(row)
        ).with_inputs("tokens")
        for row in data_split.select(range(start, end))
    ]

# Prepare the training and test sets
train_set = prepare_dataset(dataset["train"], 0, 200)
test_set = prepare_dataset(dataset["test"], 0, 50)

# original prompt is the docstirng?
class ObjectExtraction(dspy.Signature):
    """
    Extract contiguous tokens referring to specific objects of interest in these outcomes, for normalization, if any, from a list of string tokens.
    Exclude units, metrics (like number of participants, change from baseline), specifiers, timeframes and keep only the concept. 
    Output a list of tokens. In other words, do not combine multiple tokens into a single value.
    """
    tokens: list[str] = dspy.InputField(desc="tokenized text")
    extracted_objects: list[str] = dspy.OutputField(desc="all tokens referring to outcome objects extracted from the tokenized text")


def extraction_correctness_metric(example: dspy.Example, prediction: dspy.Prediction, trace=None) -> bool:
    """
    Computes correctness of entity extraction predictions.
    
    Args:
        example (dspy.Example): The dataset example containing expected people entities.
        prediction (dspy.Prediction): The prediction from the DSPy people extraction program.
        trace: Optional trace object for debugging.
    
    Returns:
        bool: True if predictions match expectations, False otherwise.
    """
    # return prediction.extracted_objects == example.expected_extracted_objects # use fuzzy matching
    #  https://github.com/stanfordnlp/dspy/issues/298
    d = distance(prediction.extracted_objects, example.expected_extracted_objects)/max(len(example.expected_extracted_objects), len(prediction.extracted_objects))
    if trace is not None:
        return d < 0.1
    return 1-d

if __name__ == "__main__":
    object_extractor = dspy.ChainOfThought(ObjectExtraction)
    evaluate_correctness = dspy.Evaluate(
        devset=test_set,
        metric=extraction_correctness_metric,
        num_threads=24,
        display_progress=True,
        display_table=True
    )

    print(evaluate_correctness(object_extractor, devset=test_set))

    mipro_optimizer = dspy.MIPROv2(
        metric=extraction_correctness_metric,
        auto="medium",
    )
    optimized_people_extractor = mipro_optimizer.compile(
        object_extractor,
        trainset=train_set,
        max_bootstrapped_demos=4,
        requires_permission_to_run=False,
        minibatch=False
    )
    print(evaluate_correctness(optimized_people_extractor, devset=test_set))
    dspy.inspect_history(n=1)

    optimized_people_extractor.save("optimized_extractor.json")

    # loaded_people_extractor = dspy.ChainOfThought(ObjectExtraction)
    # loaded_people_extractor.load("optimized_extractor.json")

    # loaded_people_extractor(tokens=["Italy", "recalled", "Marcello", "Cuttitta"]).extracted_people
    