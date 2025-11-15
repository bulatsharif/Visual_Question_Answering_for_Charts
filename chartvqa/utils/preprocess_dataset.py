import torch
from datasets import DatasetDict

from chartvqa.models.TiQS.prompting import build_prompt


def map_dataset_for_tixs(dataset, preprocessor, map_fn_kwargs):
    if isinstance(dataset, DatasetDict):
        mapped_splits = {}
        for split_name, split_dataset in dataset.items():
            mapped_splits[split_name] = split_dataset.map(
                preprocessor,
                batched=False,
                remove_columns=split_dataset.column_names,
                fn_kwargs=map_fn_kwargs,
            )
        return DatasetDict(mapped_splits)

    return dataset.map(
        preprocessor,
        batched=False,
        remove_columns=dataset.column_names,
        fn_kwargs=map_fn_kwargs,
    )


def preprocess_for_tixs(example, **kwargs):
    # 0) Get needed components
    tiny_clip_processor = kwargs["processor"]
    smol_tokenizer = kwargs["tokenizer"]

    # 1) Vision side
    pil_img = example["image"]        # ensure it's a PIL.Image
    clip_inputs = tiny_clip_processor(images=pil_img, return_tensors="pt")
    pixel_values = clip_inputs.pixel_values[0]  # (3, H, W)

    # 2) Text side: format prompt
    question = example.get("query") or example.get("question") or ""
    answer_field = example.get("label") or example.get("answer") or ""

    if isinstance(answer_field, (list, tuple)):
        answer_text = answer_field[0]
    else:
        answer_text = answer_field
    answer_text = str(answer_text)

    prompt = build_prompt(question)
    # We want the model to generate `answer_text` after this prompt

    # Tokenize prompt and answer separately to reliably locate the answer span.
    tok_prompt = smol_tokenizer(
        prompt + " ",
        return_tensors="pt"
    )

    tok_answers = smol_tokenizer(
        answer_text,
        return_tensors="pt"
    )

    input_ids = torch.cat([tok_prompt.input_ids, tok_answers.input_ids], dim=1)[0]

    # Mask loss on the prompt tokens, keep loss only on the answer segment
    labels = input_ids.clone()

    answer_start = tok_prompt.input_ids.size(1)
    labels[:answer_start] = -100

    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "labels": labels,
    }
