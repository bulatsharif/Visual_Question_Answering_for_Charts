import torch
from typing import List, Dict, Any
from transformers import ProcessorMixin
from chartvqa.utils.text import normalize_answer

class ChartQADataCollator:
    """
    Data Collator for training the Florence-2.
    """
    def __init__(self, processor: ProcessorMixin):
        self.processor = processor

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        valid_examples = [e for e in examples if e.get("image") is not None]
        images = [example["image"].convert("RGB") for example in valid_examples]
        prompts = ["<VQA> " + example["query"] for example in valid_examples]
        
        answers = []
        for example in examples:
            lbl = example["label"]
            if isinstance(lbl, list):
                answers.append(lbl[0] if lbl else "")
            else:
                answers.append(str(lbl))

        inputs = self.processor(
            text=prompts,
            images=images,
            return_tensors="pt",
            padding=True
        )

        text_labels = self.processor.tokenizer(
            text=answers,
            return_tensors="pt",
            padding=True,
            return_token_type_ids=False
        ).input_ids

        text_labels[text_labels == self.processor.tokenizer.pad_token_id] = -100
        inputs["labels"] = text_labels
        return inputs


def collate_fn_eval(batch_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for validation. Normalizes all labels.
    """
    images = []
    questions = []
    labels = []

    for item in batch_items:
        img = item.get("image")
        q = item.get("query") or item.get("question")
        lbl = item.get("label")
        if img is not None and q is not None and lbl is not None:
            images.append(img.convert('RGB')) 
            questions.append(q)
            labels.append(normalize_answer(lbl))
    
    return {
        "image": images,
        "query": questions,
        "label": labels
    }
