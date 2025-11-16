import torch
from datasets import DatasetDict

from chartvqa.models.TiQS.prompting import build_prompt


def map_dataset_for_tixs(dataset, preprocessor, map_fn_kwargs):
    if isinstance(dataset, DatasetDict):
        mapped_splits = {}
        for split_name, split_dataset in dataset.items():
            mapped_splits[split_name] = split_dataset.map(
                preprocessor,
                batched=True,
                remove_columns=split_dataset.column_names,
                fn_kwargs=map_fn_kwargs,
            )
        return DatasetDict(mapped_splits)

    return dataset.map(
        preprocessor,
        batched=True,
        remove_columns=dataset.column_names,
        fn_kwargs=map_fn_kwargs,
    )


def preprocess_for_tixs(batch, **kwargs):
    """
    Batched version.

    batch["image"]  -> list of PIL.Images
    batch["query"]  -> list of strings (or possibly missing)
    batch["label"]  -> list of answers (string/int/list, etc.)
    """
    tiny_clip_processor = kwargs["processor"]
    smol_tokenizer = kwargs["tokenizer"]

    batch_size = len(batch["image"])

    # 1) Vision side (batched)
    pil_imgs = [img.convert("RGB") for img in batch["image"]]
    clip_inputs = tiny_clip_processor(images=pil_imgs, return_tensors="pt")
    pixel_values_batch = clip_inputs.pixel_values  # (B, 3, H, W)

    out_pixel_values = []
    out_input_ids = []
    out_labels = []

    # 2) Text side: per-example loop (cheap, keeps behavior identical)
    for i in range(batch_size):
        question = (
            (batch.get("query") or [None])[i]
            if "query" in batch
            else None
        )
        if question is None and "question" in batch:
            question = batch["question"][i]
        if question is None:
            question = ""

        answer_field = None
        if "label" in batch:
            answer_field = batch["label"][i]
        elif "answer" in batch:
            answer_field = batch["answer"][i]

        if isinstance(answer_field, (list, tuple)):
            answer_text = answer_field[0]
        else:
            answer_text = answer_field
        if answer_text is None:
            answer_text = ""
        answer_text = str(answer_text)

        prompt = build_prompt(question)

        # Tokenize prompt and answer separately
        tok_prompt = smol_tokenizer(
            prompt + " ",
            return_tensors="pt",
        )
        tok_answers = smol_tokenizer(
            answer_text,
            return_tensors="pt",
        )

        # Concatenate
        input_ids = torch.cat(
            [tok_prompt.input_ids, tok_answers.input_ids], dim=1
        )[0]  # (L,)
        labels = input_ids.clone()

        answer_start = tok_prompt.input_ids.size(1)
        labels[:answer_start] = -100  # mask prompt

        # Collect outputs
        out_pixel_values.append(pixel_values_batch[i])  # (3, H, W) tensor
        out_input_ids.append(input_ids)                 # 1D tensor
        out_labels.append(labels)                       # 1D tensor

    return {
        "pixel_values": out_pixel_values,
        "input_ids": out_input_ids,
        "labels": out_labels,
    }
