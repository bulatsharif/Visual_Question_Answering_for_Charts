from transformers import AutoProcessor, AutoModelForCausalLM
from typing import List
from PIL import Image
import torch
from .base import VQAModel

class Florence2Model(VQAModel):
    """
    Implementation of VQAModel for Florence-2.
    """
    
    def _load_model(self):
        """
        Download Florence-2 model and processor.
        """
        self.processor = AutoProcessor.from_pretrained(
            self.model_cfg.model_path, 
            trust_remote_code=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_cfg.model_path,
            dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            trust_remote_code=True,
            attn_implementation="eager"
        ).to(self.device)

    def infer_batch(self, images: List[Image.Image], questions: List[str]) -> List[str]:
        """
        Implement batch inference for Florence-2.
        """
        task_prompt = "<VQA>"
        prompts = [task_prompt + " " + q for q in questions]
        inputs = self.processor(
            text=prompts, 
            images=images, 
            return_tensors="pt",
            padding=True
        ).to(self.device, torch.float16 if self.device.type == "cuda" else torch.float32)

        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=32,
            early_stopping=False,
            do_sample=False,
            num_beams=3,
            use_cache=False
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
