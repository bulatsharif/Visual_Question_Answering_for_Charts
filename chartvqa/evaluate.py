import torch
from typing import Any, Dict, List, Tuple
from omegaconf import DictConfig

from chartvqa.models.base import VQAModel
from chartvqa.utils.text import normalize_answer
from chartvqa.utils.logging import WandbLogger

def evaluate(
    model: VQAModel,
    dataset,
    device: torch.device,
    eval_cfg: DictConfig,
    logger: WandbLogger
) -> Tuple[int, int, float, List[Dict[str, Any]]]:
    """
    Evaluates the model on the dataset with logging.
    """
    model.model.eval()
    correct = 0
    total = 0
    examples: List[Dict[str, Any]] = []

    # Read parameters from config
    print_examples = eval_cfg.print_examples
    progress_every = eval_cfg.progress_every
    batch_size = eval_cfg.get("batch_size", 8)

    with torch.no_grad():
        for batch in dataset.iter(batch_size=batch_size):
            images = [img.convert('RGB') for img in batch.get("image") if img is not None]
            questions = batch.get("query") or batch.get("question")
            labels_batch = [normalize_answer(lbl) for lbl in batch.get("label")]

            if not images or not questions or len(images) != len(questions):
                print(f"Skipping corrupt batch of size {len(labels_batch)}")
                total += len(labels_batch)
                continue

            pred_texts = model.infer_batch(images, questions)

            for i in range(len(pred_texts)):
                pred_text = pred_texts[i]
                labels = labels_batch[i]
                is_correct = pred_text in labels
                if is_correct:
                    correct += 1
            
            total += len(labels_batch)

            if print_examples:
                pred_text = pred_texts[0]
                labels = labels_batch[0]
                question = questions[0]
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
            if progress_every and (total // batch_size) % progress_every == 0 and total > 0:
                acc = correct / total
                logger.log({"accuracy": acc, "samples_evaluated": total})
                print(f"Processed {total} samples | Running accuracy: {acc:.3f}")
            
    accuracy = correct / total if total else 0.0
    return correct, total, accuracy, examples
