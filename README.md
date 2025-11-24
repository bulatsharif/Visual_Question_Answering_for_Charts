# Visual Question Answering for Charts

This repository contains reproducible training and evaluation pipelines for ChartQA-style visual question answering experiments. It standardizes dataset preprocessing, Hydra-based configuration, and optional Weights & Biases tracking across multiple model families (SmolVLM, Florence-2, and a custom TiQS connector).

## Contents

* [Setup](#setup)
* [Configuration system](#configuration-system)
* [Evaluation](#evaluation)
* [Training the TiQS connector](#training-the-tiqs-connector)
* [Training Florence-2 with LoRA](#training-florence-2-with-lora)
* [Repository layout](#repository-layout)

## Setup

1. Create and activate a virtual environment (Python 3.10+ recommended):

   ```bash
   uv venv
   source .venv/bin/activate
   ```
2. Install dependencies:

   ```bash
   uv pip install -r requirements.txt
   ```
3. (Optional) Authenticate to Weights & Biases if you want logging and artifact tracking:

   ```bash
   wandb login
   ```

## Configuration system

Hydra composes runtime settings from the YAML files under `configs/`. The default entrypoint uses `configs/default.yaml`, which chains in dataset, model, evaluation, and training defaults. You can override any field via the command line, e.g. `eval.max_samples=null` to evaluate the full split or `model=florence` to swap architectures.

Key config groups:

* **Model** (`configs/model/`): choose `smolvlm` (AutoModelForVision2Seq), `florence` (Florence-2), or `model` (CustomVLM/TiQS). Quantization flags live here as well.
* **Dataset** (`configs/dataset/data.yaml`): points to the HuggingFaceM4/ChartQA dataset.
* **Evaluation** (`configs/eval/eval.yaml`): split selection, batching, metrics list, sample limits, and W&B toggle.
* **Training** (`configs/train/` and `configs/train_chartqa.yaml`): optimizer and checkpointing settings for connector training, plus a separate LoRA recipe for Florence-2.

Because Hydra writes outputs under a timestamped working directory, set paths in configs as relative to the project root; scripts resolve them for you during execution.

## Evaluation

Run evaluation from the repository root:

```bash
python -m chartvqa.eval_chartvqa
```

The script loads ChartQA via `datasets`, builds a dataloader, instantiates the requested model, and reports accuracy with optional example printing.

Common overrides:

* Evaluate the full validation split:
  `PYTHONPATH=. python eval_chartvqa.py eval.split=val eval.max_samples=null`
* Switch models:
  `PYTHONPATH=. python eval_chartvqa.py model=florence` or `model=smolvlm`
* Turn off W&B logging:
  `PYTHONPATH=. python eval_chartvqa.py eval.wandb_log=false`

If W&B logging is enabled, hardware info, progress metrics, and the final report are pushed automatically.
A JSON summary is also written locally to the path from `eval.report_path` (default `eval_report.json`).

## Training the TiQS connector

Use `train_chartqa.py` to fine-tune the TiQS connector (CustomVLM) on ChartQA:

```bash
cd chartvqa
PYTHONPATH=. python train_chartqa.py
```

The runner seeds the job, prepares the device, downloads the configured model, and then calls `train_model` to launch a Hugging Face `Trainer` with ChartQA splits, a custom data collator, and optional evaluation.
Connector weights are saved to `train.connector_output_path` and can be uploaded as W&B artifacts when logging is active.

Useful overrides:

* Resume from a connector checkpoint:
  `PYTHONPATH=. python train_chartqa.py train.resume_from_checkpoint=outputs/tiqs-qformer/checkpoint-1000`
* Short runs for debugging:
  `PYTHONPATH=. python train_chartqa.py train.num_epochs=1 train.max_train_samples=128 train.max_eval_samples=64`
* Disable artifact upload:
  `PYTHONPATH=. python train_chartqa.py train.wandb_log=false`

The training report is written to `train.report_path` (default `train_report.json`) with train/eval metrics and output locations.

## Training Florence-2 with LoRA

`train_florence.py` provides a LoRA fine-tuning path for Florence-2:

```bash
cd chartvqa
PYTHONPATH=. python train_florence.py model=florence train=florence_lora
```

The script loads Florence-2 with the appropriate processor, applies the LoRA adapter configuration from `train.florence_lora`, and trains with a custom `Seq2SeqTrainer` that supports mixed precision, generation during eval callbacks, and W&B logging.
Checkpoints and processor assets are saved to `train.output_dir` (default `./checkpoints/florence2-chartqa-lora`).

Key toggles:

* Mixed precision: `train.fp16=true` or `train.bf16=true`
* Generation settings: `eval.max_samples`, `eval.split`, and `train.generation_num_beams`
* LoRA target modules: adjust `train.lora.target_modules` to refine which layers are adapted.

## Repository layout

* `chartvqa/eval_chartvqa.py` – Hydra-driven entrypoint for evaluation.
* `chartvqa/train_chartqa.py` – Hydra-driven entrypoint for TiQS connector training.
* `chartvqa/train_florence.py` – LoRA finetuning flow for Florence-2.
* `chartvqa/train.py` – Training utilities, dataloading, and collators for connector finetuning.
* `chartvqa/evaluate.py` – Core evaluation loop and metric logging.
* `chartvqa/models/` – Model wrappers (SmolVLM, Florence-2, TiQS, VILT) implementing the `VQAModel` interface.
* `configs/` – Hydra config groups for datasets, models, evaluation, and training presets.

With these pieces, you can benchmark off-the-shelf VLMs on ChartQA or continue research by adapting connectors and LoRA heads using consistent logging and configuration management.
