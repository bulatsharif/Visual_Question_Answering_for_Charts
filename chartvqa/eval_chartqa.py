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


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_device(device_cfg: str):
    if device_cfg == "auto":
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


def evaluate(
    model: ViltForQuestionAnswering,
    processor: ViltProcessor,
    dataset,
    device: torch.device,
    max_samples: int | None = None,
    print_examples: int = 0,
    progress_every: int = 50,
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
            

            encoding = processor(image, question, return_tensors="pt")
            encoding = {k: v.to(device) for k, v in encoding.items()}

            outputs = model(**encoding)
            pred_idx = outputs.logits.argmax(-1).item()
            pred_text = model.config.id2label.get(pred_idx, str(pred_idx)).strip().lower()

            is_correct = pred_text in labels
            if is_correct:
                correct += 1
            total += 1

            if len(examples) < max(print_examples, 0):
                examples.append(
                    {
                        "idx": idx,
                        "question": question,
                        "ground_truth": labels,
                        "prediction": pred_text,
                        "correct": is_correct,
                    }
                )

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
    print_examples: int = getattr(cfg.eval, "print_examples", 5)
    progress_every: int = getattr(cfg.eval, "progress_every", 50)
    wandb_logging_on: bool = getattr(cfg.eval, "wandb_log", False)

    print(f"Loading dataset: {dataset_path} [{split}]")
    ds = load_dataset(dataset_path, split=split)

    # Load model & processor
    model_path: str = cfg.model.model_path
    print(f"Loading model: {model_path}")
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
        wandb_run=run if wandb_logging_on else None
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