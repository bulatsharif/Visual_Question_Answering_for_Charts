import json
import os
import hydra
from omegaconf import DictConfig, OmegaConf
from datasets import load_dataset
from typing import Dict, Any

from chartvqa.utils.setup import set_seed, prepare_device
from chartvqa.models.base import VQAModel
from chartvqa.evaluate import evaluate
from chartvqa.utils.logging import WandbLogger

@hydra.main(version_base="1.3", config_path="../configs", config_name="default")
def main(cfg: DictConfig) -> None:
    print("Resolved config:\n" + OmegaConf.to_yaml(cfg))

    set_seed(int(cfg.seed))
    device = prepare_device(str(cfg.device))
    print(f"Using device: {device}")

    logger = WandbLogger(cfg)

    print(f"Loading dataset: {cfg.dataset.dataset_path} [{cfg.eval.split}]")
    ds = load_dataset(cfg.dataset.dataset_path, split=cfg.eval.split)

    print(f"Loading model: {cfg.model.model_path}")
    model = VQAModel.load_specific_model_from_config(cfg.model, device)

    correct, total, accuracy, examples = evaluate(
        model=model,
        dataset=ds,
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
