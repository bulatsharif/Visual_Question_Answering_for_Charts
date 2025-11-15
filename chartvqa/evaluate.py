import torch
import time
from typing import Any, Dict, List, Tuple
from omegaconf import DictConfig

from chartvqa.models.base import VQAModel
from chartvqa.utils.logging import WandbLogger

def evaluate(
    model: VQAModel,
    dataloader,
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
    batch_size = eval_cfg.get("batch_size", 4)

    batch_processing_times = []

    with torch.no_grad():
        for batch in dataloader:
            batch_start_time = time.perf_counter()
            images = batch.get("image")
            questions = batch.get("query")
            labels_batch = batch.get("label")

            current_batch_size = len(images)

            if not current_batch_size:
                print("Skipping empty/corrupt batch")
                continue

            pred_texts = model.infer_batch(images, questions)
            print(pred_texts)

            for i in range(len(pred_texts)):
                pred_text = pred_texts[i]
                labels = labels_batch[i]
                is_correct = pred_text in labels
                if is_correct:
                    correct += 1

            batch_time_sec = time.perf_counter() - batch_start_time
            batch_processing_times.append(batch_time_sec)
            
            total += current_batch_size

            if print_examples:
                pred_text = pred_texts[0]
                labels = labels_batch[0]
                question = questions[0]
                is_correct = pred_text in labels
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

                avg_batch_time = sum(batch_processing_times) / len(batch_processing_times) if batch_processing_times else 0
                logger.log({
                    "accuracy": acc, 
                    "samples_evaluated": total,
                    "batch_processing_time_sec": batch_time_sec,
                    "avg_batch_processing_time_sec": avg_batch_time
                })

                print(f"Processed {total} samples | Running accuracy: {acc:.3f} | Last batch time: {batch_time_sec:.3f}s")
            
    accuracy = correct / total if total else 0.0
    return correct, total, accuracy, examples
