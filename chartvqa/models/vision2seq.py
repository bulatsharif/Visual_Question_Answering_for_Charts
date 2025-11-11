from transformers import AutoModelForVision2Seq, AutoProcessor
from chartvqa.utils.text import parse_assistant_response
from typing import List
from .base import VQAModel
from PIL import Image
import torch
import re

class Vision2SeqModel(VQAModel):
    def _load_model(self):
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            _attn_implementation="sdpa" if self.device.type == "cuda" else "eager",
        ).to(self.device)

    def infer(self, image: Image.Image, question: str) -> str:
        """
        Implement inference.
        """
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": f"Answer the following question about the image: {question}"}
            ]}
        ]
        
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=[image], return_tensors="pt").to(self.device)
        
        generated_ids = self.model.generate(**inputs, max_new_tokens=32)
        generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

        if not generated_texts:
            return ""

        return parse_assistant_response(generated_texts[0])
    
    def infer_batch(self, images: List[Image.Image], questions: List[str]) -> List[str]:
        """
        Implement batch inference.
        """
        prompts = []
        for question in questions:
            messages = [
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"Answer the following question about the image: {question}"}
                ]}
            ]
            prompts.append(self.processor.apply_chat_template(messages, add_generation_prompt=True))
        
        inputs = self.processor(
            text=prompts, 
            images=images, 
            return_tensors="pt", 
            padding=True
        ).to(self.device)
        
        generated_ids = self.model.generate(**inputs, max_new_tokens=32)
        
        generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        if not generated_texts:
            return [""] * len(questions)

        return [parse_assistant_response(text) for text in generated_texts]
