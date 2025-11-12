import json
import os
import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from datasets import load_dataset
from typing import Dict, Any
from torch.utils.data import DataLoader

from chartvqa.utils.setup import set_seed, prepare_device
from chartvqa.models.base import VQAModel
from chartvqa.evaluate import evaluate
from chartvqa.utils.logging import WandbLogger
from chartvqa.utils.text import normalize_answer


def collate_fn_eval(batch_items):
    """
    Collate function for validation DataLoader.
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


@hydra.main(version_base="1.3", config_path="../configs", config_name="default")
def main(cfg: DictConfig) -> None:
    print("Resolved config:\n" + OmegaConf.to_yaml(cfg))

    set_seed(int(cfg.seed))
    device = prepare_device(str(cfg.device))
    print(f"Using device: {device}")

    logger = WandbLogger(cfg)

    try:
       static_info = {"device_name": str(device)}
       if torch.cuda.is_available() and device.type == "cuda":
           static_info["gpu_name"] = torch.cuda.get_device_name(device)
           static_info["gpu_total_memory_gb"] = round(torch.cuda.get_device_properties(device).total_memory / (1024**3), 2)
       logger.log(static_info)
    except Exception as e:
       print(f"Warning: Could not log static hardware info: {e}")

    print(f"Loading dataset: {cfg.dataset.dataset_path} [{cfg.eval.split}]")
    ds = load_dataset(cfg.dataset.dataset_path, split=cfg.eval.split)

    print(f"Setting up DataLoader (Batch size: {cfg.eval.batch_size}, Num workers: 4)")
    dataloader = DataLoader(
        ds,
        batch_size=cfg.eval.batch_size,
        collate_fn=collate_fn_eval,
        num_workers=4,
        pin_memory=True
    )

    print(f"Loading model: {cfg.model.model_path}")
    model = VQAModel.load_specific_model_from_config(cfg.model, device)

    correct, total, accuracy, examples = evaluate(
        model=model,
        dataloader=dataloader,
        device=device,
        eval_cfg=cfg.eval,
        logger=logger
    )

    report: Dict[str, Any] = {
        "dataset": cfg.dataset.dataset_path,
        "split": cfg.eval.split,
        "model": cfg.model.model_path,
        "samples_evaluated": total,
        "correct": correct,
        "accuracy": accuracy,
        "examples": examples,
    }

    logger.log({
        "final_accuracy": accuracy, 
        "correct": correct, 
        "total_samples": total
    })
    logger.finish()

    out_path = cfg.eval.report_path
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print("\nEvaluation complete:")
    print(json.dumps({k: report[k] for k in ["samples_evaluated", "correct", "accuracy"]}, indent=2))
    print(f"Report saved to: {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
