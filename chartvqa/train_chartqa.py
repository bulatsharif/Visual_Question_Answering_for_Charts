import json
from pathlib import Path

import hydra
import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

from chartvqa.models.base import VQAModel
from chartvqa.train import train_model
from chartvqa.utils.logging import WandbLogger
from chartvqa.utils.setup import prepare_device, set_seed


def _json_default(obj):
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


@hydra.main(version_base="1.3", config_path="../configs", config_name="train_chartqa")
def main(cfg: DictConfig) -> None:
    print("Resolved config:\n" + OmegaConf.to_yaml(cfg))

    set_seed(int(cfg.seed))
    device = prepare_device(str(cfg.device))
    print(f"Using device: {device}")

    logger = WandbLogger(cfg, section="train")
    try:
        static_info = {"device_name": str(device)}
        if torch.cuda.is_available() and device.type == "cuda":
            static_info["gpu_name"] = torch.cuda.get_device_name(device)
            static_info["gpu_total_memory_gb"] = round(
                torch.cuda.get_device_properties(device).total_memory / (1024**3), 2
            )
        logger.log(static_info)
    except Exception as exc:
        print(f"Warning: Could not log static hardware info: {exc}")

    project_root = Path(get_original_cwd())
    print("Loading TiQS model...")
    model = VQAModel.load_specific_model_from_config(cfg.model, device, wandb_logger=logger)

    print("Starting connector training...")
    try:
        summary = train_model(
            model_instance=model,
            cfg=cfg,
            project_root=project_root,
            logger=logger,
        )
    finally:
        logger.finish()

    report_path = Path(cfg.train.report_path)
    if not report_path.is_absolute():
        report_path = project_root / report_path
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=_json_default)

    print("\nTraining summary:")
    print(json.dumps(summary, indent=2, default=_json_default))
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()
