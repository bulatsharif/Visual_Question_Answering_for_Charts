"""
Dataset preprocessing helpers used by training/test pipelines.

Exports `map_dataset_for_tixs` which maps preprocessing functions over
datasets using `datasets.Dataset.map` with multiple processes and
`preprocess_for_tixs` which converts raw ChartQA examples to the format
expected by the training connectors (pixel_values, input_ids, labels).
"""

import torch
from datasets import DatasetDict
import os
from chartvqa.models.TiQS.prompting import build_prompt


def map_dataset_for_tixs(dataset, preprocessor, map_fn_kwargs):
    # Somewhat dummy logic to define available number of CPU for processing.
    num_cpus = min(min(max(os.cpu_count() - 2, 1), 32), os.cpu_count() // 4)
    print(f"Using {num_cpus} cores for dataset mapping...")
    if isinstance(dataset, DatasetDict):
        mapped_splits = {}
        for split_name, split_dataset in dataset.items():
            mapped_splits[split_name] = split_dataset.map(
                preprocessor,
                batched=False,
                remove_columns=split_dataset.column_names,
                num_proc=num_cpus,
                fn_kwargs=map_fn_kwargs,
            )
        return DatasetDict(mapped_splits)

    return dataset.map(
        preprocessor,
        batched=False,
        remove_columns=dataset.column_names,
        fn_kwargs=map_fn_kwargs,
        num_proc=num_cpus,
    )


def preprocess_for_tixs(example, **kwargs):
    """Preprocess a single example for the TiQS connector training.

    Returns a dict containing `pixel_values`, `input_ids` and `labels`.
    The label tensor is offset such that the text prompt tokens are
    ignored (set to -100) during loss computation.
    """

    tiny_clip_processor = kwargs["processor"]
    smol_tokenizer = kwargs["tokenizer"]


    pil_img = example["image"].convert('RGB')     
    clip_inputs = tiny_clip_processor(images=pil_img, return_tensors="pt")
    pixel_values = clip_inputs.pixel_values[0]  # (3, H, W)


    question = example.get("query") or example.get("question") or ""
    answer_field = example.get("label") or example.get("answer") or ""

    if isinstance(answer_field, (list, tuple)):
        answer_text = answer_field[0]
    else:
        answer_text = answer_field
    answer_text = str(answer_text)

    prompt = build_prompt(question)


    tok_prompt = smol_tokenizer(
        prompt,  
        return_tensors="pt",
        add_special_tokens=True  
    )

    tok_answers = smol_tokenizer(
        " " + answer_text + smol_tokenizer.eos_token, 
        return_tensors="pt",
        add_special_tokens=False  
    )

    input_ids = torch.cat([tok_prompt.input_ids, tok_answers.input_ids], dim=1)[0]

    labels = input_ids.clone()

    answer_start = tok_prompt.input_ids.size(1)
    labels[:answer_start] = -100

    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "labels": labels,
    }
