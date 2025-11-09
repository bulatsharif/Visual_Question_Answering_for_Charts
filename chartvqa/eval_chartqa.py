"""
Evaluation script for ChartQA using a ViLT VQA model.

Config is managed by Hydra. Defaults live in `configs/` at the repo root.

Notes:
- Dataset: HuggingFaceM4/ChartQA
- Model: ViLT VQA (dandelin/vilt-b32-finetuned-vqa) by default
"""

from typing import Any, Dict, List, Tuple
import json
import os
import random
from PIL import Image
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
import hydra
import wandb
from datasets import load_dataset
from transformers import ViltForQuestionAnswering, ViltProcessor
import re


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_device(device_cfg: str):
    if device_cfg == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_cfg)


def _normalize_answer(ans: Any) -> List[str]:
    """Normalize ground-truth answers to a list of lowercase strings."""
    if ans is None:
        return []
    if isinstance(ans, str):
        return [ans.strip().lower()]
    if isinstance(ans, (list, tuple)):
        out: List[str] = []
        for a in ans:
            if a is None:
                continue
            out.append(str(a).strip().lower())
        return out
    return [str(ans).strip().lower()]

def _inference_vilt(model, processor, image: Image.Image, question: str, device: torch.device) -> str:
    """Run inference for a single (image, question) pair using ViLT."""
    encoding = processor(image, question, return_tensors="pt")
    encoding = {k: v.to(device) for k, v in encoding.items()}

    outputs = model(**encoding)
    pred_idx = outputs.logits.argmax(-1).item()
    pred_text = model.config.id2label.get(pred_idx, str(pred_idx)).strip().lower()
    return pred_text

def _inference_vision2seq(model, processor, image: Image.Image, question: str, device: torch.device) -> str:
    """
    Run inference for a single (image, question) pair using Vision2Seq models.
    Assumes the model uses a chat-style prompt.
    
    Args:
        model: The Vision2Seq model.
        processor: The corresponding processor.
        image: The input image (PIL Image).
        question: The question string.
        device: The torch device to run on.
    Returns:
        The generated answer string.
    """
    
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": f"Answer the following question about the image: {question}"}
            ]
        },
    ]
    
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt")
    inputs = inputs.to(device)
    
    
    
    
    generated_ids = model.generate(**inputs, max_new_tokens=32)
    generated_texts = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )

    raw = generated_texts[0]
    # Strategy:
    # 1. Find the last occurrence of 'assistant:' (lowercase to be safe)
    # 2. Take everything after it until end.
    # 3. Remove any leading whitespace/newlines.
    # 4. Strip common trailing stop punctuation or residual prompt echoes.
    # 5. Collapse internal whitespace.
    lower_raw = raw.lower()
    key = 'assistant:'
    pos = lower_raw.rfind(key)
    if pos == -1:
        return ""
    answer_fragment = raw[pos + len(key):].strip()
    # Remove a leading comma or period if model added punctuation right after colon
    answer_fragment = re.sub(r'^[,.:\s]+', '', answer_fragment)
    # If the model echoed the entire instruction again, split on common delimiters.
    # Heuristic: take first sentence/clause before double newline or line break.
    answer_fragment = answer_fragment.split('\n')[0].strip()
    # Remove trailing artifacts like stray prompt words
    # Example pattern: "green.," -> "green"
    answer_fragment = re.sub(r'[\s]*[.,]+$', '', answer_fragment)
    # Keep only the first token if it appears to be a single-word answer (common for ChartQA)
    # but avoid cutting if looks like a phrase containing spaces and digits.
    simple_match = re.match(r'^[A-Za-z]+$', answer_fragment)
    if not simple_match:
        # If sentence contains multiple words, we might still want first word when others look like prompt residue.
        # Example: "green. answer the following question ..." -> take first word before such phrases.
        cleanup_split = re.split(r'\b(answer the following question|question:)', answer_fragment, flags=re.IGNORECASE)
        if cleanup_split:
            answer_fragment = cleanup_split[0].strip()
    # Final normalization
    answer_fragment = answer_fragment.lower()
    answer_fragment = re.sub(r'%', '', answer_fragment)
    return answer_fragment


def evaluate(
    model: ViltForQuestionAnswering,
    processor: ViltProcessor,
    dataset,
    device: torch.device,
    max_samples: int | None = None,
    print_examples: bool = True,
    progress_every: int = 50,
    model_type: str = "other",
    wandb_run: wandb.sdk.wandb_run.Run | None = None,
) -> Tuple[int, int, float, List[Dict[str, Any]]]:
    """Run evaluation loop and return (correct, total, accuracy, examples)."""
    model.eval()
    correct = 0
    total = 0
    examples: List[Dict[str, Any]] = []

    with torch.no_grad():
        for idx, example in enumerate(dataset):
            if max_samples is not None and idx >= max_samples:
                break
            
            image = example.get("image")
            question = example.get("query") or example.get("question")
            labels = _normalize_answer(example.get("label"))

            if image is None or question is None:
                continue
            image = image.convert('RGB')

            match model_type:
                case "AutoModelForVision2Seq":
                    pred_text = _inference_vision2seq(model, processor, image, question, device)
                case _:
                    pred_text = _inference_vilt(model, processor, image, question, device)


            is_correct = pred_text in labels
            if is_correct:
                correct += 1
            total += 1
            
            
            if print_examples:
                print(f"""
                      Question: {question},
                      Ground truth answer: {labels},
                      Prediction: {pred_text},
                      Is correct?: {is_correct}
                      
                      """)
                examples.append({
                    "question": question,
                    "ground_truth": labels,
                    "prediction": pred_text,
                    "is_correct": is_correct,
                })


            if progress_every and total % progress_every == 0:
                acc = correct / total if total else 0.
                if wandb_run is not None:   
                    wandb_run.log({"accuracy": acc, "samples_evaluated": total})
                print(f"Processed {total} samples | Running accuracy: {acc:.3f}")

    accuracy = correct / total if total else 0.0
    return correct, total, accuracy, examples


@hydra.main(version_base="1.3", config_path="../configs", config_name="default")
def main(cfg: DictConfig) -> None:
    print("Resolved config:\n" + OmegaConf.to_yaml(cfg))

    # Reproducibility & device
    set_seed(int(cfg.seed))
    device = prepare_device(str(cfg.device))
    print(f"Using device: {device}")

    # Load dataset
    dataset_path: str = cfg.dataset.dataset_path
    split: str = getattr(cfg.eval, "split", "val")
    max_samples: int | None = getattr(cfg.eval, "max_samples", None)
    print_examples: int = getattr(cfg.eval, "print_examples", True)
    progress_every: int = getattr(cfg.eval, "progress_every", 50)
    wandb_logging_on: bool = getattr(cfg.eval, "wandb_log", False)
    model_type: str = getattr(cfg.model, "model_type", "other")

    print(f"Loading dataset: {dataset_path} [{split}]")
    ds = load_dataset(dataset_path, split=split)

    # Load model & processor
    model_path: str = cfg.model.model_path
    print(f"Loading model: {model_path}")
    
    match model_type:
        case "AutoModelForVision2Seq":
            from transformers import AutoModelForVision2Seq, AutoProcessor
            processor = AutoProcessor.from_pretrained(model_path)
            model = AutoModelForVision2Seq.from_pretrained(model_path,
                                                            torch_dtype=torch.bfloat16,
                                                            _attn_implementation="flash_attention_2" if device == "cuda" else "eager",
                                                            )
            model.to(device)
        case _:
            processor = ViltProcessor.from_pretrained(model_path)
            model = ViltForQuestionAnswering.from_pretrained(model_path)
    model.to(device)

    if wandb_logging_on:
        # Set up W&B run
        run = wandb.init(
            entity="b-sharipov-innopolis-university",
            project="chart-vqa-evaluation",
            config=OmegaConf.to_container(cfg, resolve=True)
        )

        print(f"Setting up Weights & Biases logging, the Run ID: {run.id}")

    # Evaluate
    correct, total, accuracy, examples = evaluate(
        model=model,
        processor=processor,
        dataset=ds,
        device=device,
        max_samples=max_samples,
        print_examples=print_examples,
        progress_every=progress_every,
        model_type=model_type,
        wandb_run=run if wandb_logging_on else None,
    )
    

    # Report
    report: Dict[str, Any] = {
        "dataset": dataset_path,
        "split": split,
        "model": model_path,
        "samples_evaluated": total,
        "correct": correct,
        "accuracy": accuracy,
        "examples": examples,
    }

    if wandb_logging_on:
        run.log({"final_accuracy": accuracy, "correct": correct, "total_samples": total})
        run.finish()


    out_path = getattr(cfg.eval, "report_path", "eval_report.json")
    # Hydra changes the working directory to a run dir; saving relative is fine
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\nEvaluation complete:")
    print(json.dumps({k: report[k] for k in ["samples_evaluated", "correct", "accuracy"]}, indent=2))
    print(f"Report saved to: {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()