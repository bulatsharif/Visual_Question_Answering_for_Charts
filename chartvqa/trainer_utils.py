import torch
from torch.utils.data import DataLoader
from transformers import TrainerCallback
from datasets import load_dataset
from typing import List
from PIL import Image

from chartvqa.models.base import VQAModel
from chartvqa.evaluate import evaluate
from chartvqa.utils.data_utils import collate_fn_eval
from chartvqa.utils.logging import WandbLogger


class PeftModelWrapper(VQAModel):
    """
    Wrapper to make PEFT model consistent with VQAModel interface.
    """
    def __init__(self, model, processor, device):
        self.model = model
        self.processor = processor
        self.device = device

    def _load_model(self):
        pass

    def infer_batch(self, images: List[Image.Image], questions: List[str]) -> List[str]:
        task_prompt = "<VQA>"
        prompts = [task_prompt + " " + q for q in questions]
        
        inputs = self.processor(
            text=prompts, 
            images=images, 
            return_tensors="pt",
            padding=True
        )

        input_ids = inputs["input_ids"].to(self.device)
        pixel_values = inputs["pixel_values"].to(self.device)

        if self.model.dtype == torch.float16:
            pixel_values = pixel_values.to(torch.float16)
        elif self.model.dtype == torch.bfloat16:
            pixel_values = pixel_values.to(torch.bfloat16)

        gen_kwargs = {
            "max_new_tokens": 32,
            "num_beams": 3,
            "early_stopping": False,
            "do_sample": False,
            "use_cache": False
        }

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                **gen_kwargs
            )

        generated_texts = self.processor.batch_decode(
            generated_ids, 
            skip_special_tokens=False
        )
        
        cleaned_answers = []
        for text, img in zip(generated_texts, images):
            parsed_answer = self.processor.post_process_generation(
                text, 
                task=task_prompt, 
                image_size=(img.width, img.height)
            )
            ans = parsed_answer.get(task_prompt, "")
            ans = ans.replace("<s>", "").replace("</s>", "").replace("<pad>", "").strip()
            cleaned_answers.append(ans)

        return cleaned_answers


class EvaluationCallback(TrainerCallback):
    """
    Callback function for Trainer to execute validation at the end of each epoch.
    """
    def __init__(self, processor, device, eval_cfg, dataset_path, logger: WandbLogger):
        self.processor = processor
        self.device = device
        self.eval_cfg = eval_cfg
        self.dataset_path = dataset_path
        self.logger = logger
        self.vqa_model_wrapper = None
        
        print("EvaluationCallback: Loading validation dataset...")
        val_dataset = load_dataset(self.dataset_path, split=self.eval_cfg.split)
        
        self.val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.eval_cfg.batch_size,
            collate_fn=collate_fn_eval,
            num_workers=4, 
            pin_memory=True
        )
        print("EvaluationCallback: Validation dataset loaded.")

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if self.vqa_model_wrapper is None:
            self.vqa_model_wrapper = PeftModelWrapper(model, self.processor, self.device)
        else:
            self.vqa_model_wrapper.model = model 
        
        print("\nRunning Validation")
        
        correct, total, accuracy, examples = evaluate(
            model=self.vqa_model_wrapper,
            dataloader=self.val_dataloader,
            device=self.device,
            eval_cfg=self.eval_cfg,
            logger=self.logger
        )
        
        print(f"Epoch {state.epoch:.0f} Validation Accuracy: {accuracy:.4f}")
        
        self.logger.log({
            f"eval/epoch_accuracy": accuracy,
            f"eval/epoch": state.epoch,
            f"eval/correct": correct,
            f"eval/total": total
        })
