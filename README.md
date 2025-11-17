# Visual Question Answering for Charts

Minimal instructions to run evaluation on ChartQA using a ViLT VQA checkpoint.

## Setup

From repo root do:

```bash
uv venv 
source .venv/bin/activate
uv pip install -r requirements.txt
```


## Run evaluation

```bash
cd chartvqa
PYTHONPATH=. python eval_chartqa.py
```

This will:
- load dataset HuggingFaceM4/ChartQA (split=val)
- load model dandelin/vilt-b32-finetuned-vqa
- evaluate a sample of 100 examples and write `eval_report.json` in the run directory

### Common overrides (Hydra)

```bash
# evaluate all samples in the validation split
PYTHONPATH=. python eval_chartqa.py eval.max_samples=null

# evaluate on the test split
PYTHONPATH=. python eval_chartqa.py eval.split=test

# change the model checkpoint
PYTHONPATH=. python eval_chartqa.py model.model_path=dandelin/vilt-b32-mlm

# evaluate more samples and print a few examples
PYTHONPATH=. python eval_chartqa.py eval.max_samples=1000 eval.print_examples=10
```

Config files live in `configs/` and are merged at runtime via Hydra.

## Train TiQS connector

```bash
cd chartvqa
PYTHONPATH=. python train_chartqa.py
```

Training uses the ChartQA training split to finetune only the TiQS connector. Hyperparameters / paths can be overridden via Hydra, e.g. `python train_chartqa.py train.num_epochs=2 train.learning_rate=1e-4`.
