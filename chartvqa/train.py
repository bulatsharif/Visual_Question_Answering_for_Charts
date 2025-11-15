from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
import time
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import Trainer, TrainingArguments
# from transformers.trainer_callback import TrainerCallback

from .utils.preprocess_dataset import map_dataset_for_tixs, preprocess_for_tixs


class VLMDataCollator:
    """
    Data collator for TinyCLIP + SmolLM connectors.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        # 1) pixel_values: (3, H, W) → (B, 3, H, W)
        pixel_values = torch.stack(
            [torch.tensor(example["pixel_values"]) for example in batch]
        )

        # 2) prepare lists (ensure tensors)
        input_ids_list = []
        labels_list = []

        for example in batch:
            ids = example["input_ids"]
            lbl = example["labels"]

            if torch.is_tensor(ids):
                ids = ids.squeeze(0)
            else:
                ids = torch.tensor(ids, dtype=torch.long)

            if torch.is_tensor(lbl):
                lbl = lbl.squeeze(0)
            else:
                lbl = torch.tensor(lbl, dtype=torch.long)

            input_ids_list.append(ids)
            labels_list.append(lbl)

        # 3) Pad sequences
        pad_token_id = self.tokenizer.pad_token_id
        input_ids = pad_sequence(
            input_ids_list,
            batch_first=True,
            padding_value=pad_token_id,
        )

        labels = pad_sequence(
            labels_list,
            batch_first=True,
            padding_value=-100,
        )

        attention_mask = (input_ids != pad_token_id).long()

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }


# class TimeLimitCallback(TrainerCallback):
#     """
#     Stops training once the configured wall-clock time (in seconds) elapses.
#     """

#     def __init__(self, max_seconds: float):
#         self.max_seconds = max_seconds
#         self._start_time: Optional[float] = None

#     def on_train_begin(self, args, state, control, **kwargs):
#         self._start_time = time.time()
#         return control

#     def on_step_end(self, args, state, control, **kwargs):
#         if self._start_time is None:
#             return control

#         elapsed = time.time() - self._start_time
#         if elapsed >= self.max_seconds:
#             control.should_training_stop = True
#             control.should_evaluate = True
#         return control


def _resolve_path(path_like: Union[str, Path], project_root: Path) -> Path:
    path = Path(path_like)
    if not path.is_absolute():
        path = project_root / path
    return path


def _prepare_chartqa_splits(
    dataset_path: str,
    tokenizer,
    processor,
    train_split: str,
    eval_split: Optional[str],
    max_train_samples: Optional[int],
    max_eval_samples: Optional[int],
) -> Tuple[Any, Optional[Any]]:
    dataset_dict = load_dataset(dataset_path)
    dataset_dict = map_dataset_for_tixs(
        dataset_dict,
        preprocess_for_tixs,
        {
            "tokenizer": tokenizer,
            "processor": processor,
        },
    )

    if train_split not in dataset_dict:
        available = ", ".join(dataset_dict.keys())
        raise ValueError(
            f"Split '{train_split}' not found in dataset. Available splits: {available}"
        )

    train_dataset = dataset_dict[train_split]
    if max_train_samples is not None:
        train_dataset = train_dataset.select(range(min(len(train_dataset), max_train_samples)))

    eval_dataset = None
    if eval_split:
        if eval_split not in dataset_dict:
            available = ", ".join(dataset_dict.keys())
            raise ValueError(
                f"Split '{eval_split}' not found in dataset. Available splits: {available}"
            )
        eval_dataset = dataset_dict[eval_split]
        if max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(min(len(eval_dataset), max_eval_samples)))

    return train_dataset, eval_dataset


def _build_training_args(train_cfg, output_dir: Path) -> TrainingArguments:
    return TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        per_device_train_batch_size=train_cfg.get("train_batch_size", 1),
        per_device_eval_batch_size=train_cfg.get("eval_batch_size", 1),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 1),
        num_train_epochs=train_cfg.get("num_epochs", 1),
        learning_rate=train_cfg.get("learning_rate", 1e-4),
        weight_decay=train_cfg.get("weight_decay", 0.0),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.0),
        fp16=train_cfg.get("is_fp16", False),
        bf16=train_cfg.get("is_bf16", False),
        logging_steps=train_cfg.get("logging_steps", 50),
        save_steps=train_cfg.get("save_steps", 500),
        save_total_limit=train_cfg.get("save_total_limit", 1),
        eval_strategy=train_cfg.get("evaluation_strategy", "steps"),
        eval_steps=train_cfg.get("eval_steps", train_cfg.get("save_steps", 500)),
        report_to=train_cfg.get("report_to", None),
        remove_unused_columns=False,
        dataloader_num_workers=train_cfg.get("dataloader_num_workers", 4),
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", False),
        load_best_model_at_end=train_cfg.get("load_best_model_at_end", False),
        metric_for_best_model=train_cfg.get("metric_for_best_model"),
    )


def _save_connector_weights(model, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.qformer.state_dict(), output_path)


def train_model(model_instance, cfg, project_root: Optional[Path] = None, logger=None) -> Dict[str, Any]:
    """
    Train the TiQS connector on ChartQA.
    """
    if cfg.model.model_type != "CustomVLM":
        raise NotImplementedError("Training is only implemented for CustomVLM models.")

    project_root = project_root or Path.cwd()
    train_cfg = cfg.train

    train_dataset, eval_dataset = _prepare_chartqa_splits(
        dataset_path=cfg.dataset.dataset_path,
        tokenizer=model_instance.tokenizer,
        processor=model_instance.processor,
        train_split=train_cfg.get("train_split", "train"),
        eval_split=train_cfg.get("eval_split", "validation"),
        max_train_samples=train_cfg.get("max_train_samples"),
        max_eval_samples=train_cfg.get("max_eval_samples"),
    )

    if len(train_dataset) == 0:
        raise ValueError("Training split is empty after preprocessing.")

    checkpoint_dir = _resolve_path(train_cfg.get("checkpoint_dir", "./outputs/tiqs-qformer"), project_root)
    connector_output_path = _resolve_path(
        train_cfg.get("connector_output_path", "chartvqa/models/TiQS/chartqa-qformer-adapter.pt"),
        project_root,
    )

    training_args = _build_training_args(train_cfg, checkpoint_dir)

    data_collator = VLMDataCollator(model_instance.tokenizer)

    # callbacks = []
    # max_train_seconds = train_cfg.get("max_train_seconds")
    # if max_train_seconds:
    #     callbacks.append(TimeLimitCallback(float(max_train_seconds)))
    
    # if max_train_seconds:
    #     trainer = Trainer(
    #         model=model_instance.model,
    #         args=training_args,
    #         train_dataset=train_dataset,
    #         eval_dataset=eval_dataset,
    #         tokenizer=model_instance.tokenizer,
    #         data_collator=data_collator,
    #         # callbacks=callbacks,
    #     )
    # else:
    trainer = Trainer(
            model=model_instance.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=model_instance.tokenizer,
            data_collator=data_collator,
    )

    resume_from = train_cfg.get("resume_from_checkpoint")
    train_output = trainer.train(resume_from_checkpoint=resume_from)
    trainer.save_state()

    eval_metrics: Dict[str, Any] = {}
    if eval_dataset is not None:
        eval_metrics = trainer.evaluate(eval_dataset=eval_dataset)

    _save_connector_weights(model_instance.model, connector_output_path)
    model_instance.model.eval()

    if logger:
        payload = {
            "train_loss": train_output.metrics.get("train_loss"),
            "train_runtime": train_output.metrics.get("train_runtime"),
        }
        if eval_metrics:
            payload["eval_loss"] = eval_metrics.get("eval_loss")
        logger.log({k: v for k, v in payload.items() if v is not None})

        artifact_name = train_cfg.get("wandb_artifact_name")
        if not artifact_name:
            artifact_name = Path(connector_output_path).stem
        artifact_type = train_cfg.get("wandb_artifact_type", "connector")
        logger.log_artifact(
            name=artifact_name,
            path=connector_output_path,
            type_name=artifact_type,
            metadata={
                "train_samples": len(train_dataset),
                "eval_samples": len(eval_dataset) if eval_dataset is not None else 0,
            },
        )

    summary = {
        "train_metrics": train_output.metrics,
        "eval_metrics": eval_metrics,
        "train_samples": len(train_dataset),
        "eval_samples": len(eval_dataset) if eval_dataset is not None else 0,
        "checkpoint_dir": str(checkpoint_dir),
        "connector_path": str(connector_output_path),
    }
    return summary
