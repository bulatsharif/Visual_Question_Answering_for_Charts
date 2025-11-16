from transformers import AutoModelForVision2Seq, AutoProcessor
from chartvqa.utils.text import parse_assistant_response
from typing import List
from .base import VQAModel
from PIL import Image
import torch
import re

class Vision2SeqModel(VQAModel):
    def _load_model(self):
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            padding_side='left'
        )
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
        
        input_token_len = inputs['input_ids'].shape[1]
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=32)
        
        new_tokens = generated_ids[:, input_token_len:]
        generated_texts = self.processor.batch_decode(new_tokens, skip_special_tokens=True)

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
                    {"type": "text", "text": f"You are a Vision Language Model specialized in interpreting visual data from chart images. Your task is to analyze the provided chart image and respond to queries with concise answers, usually a single word, number, or short phrase. The charts include a variety of types (e.g., line charts, bar charts) and contain colors, labels, and text. Focus on delivering accurate, succinct answers based on the visual information. Avoid additional explanation unless absolutely necessary. Answer the following question about the image: {question}"}
                ]}
            ]
            prompts.append(self.processor.apply_chat_template(messages, add_generation_prompt=True))
        
        inputs = self.processor(
            text=prompts, 
            images=images, 
            return_tensors="pt", 
            padding=True
        ).to(self.device)
        
        input_token_len = inputs['input_ids'].shape[1]
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=32)
        
        new_tokens = generated_ids[:, input_token_len:]
        generated_texts = self.processor.batch_decode(new_tokens, skip_special_tokens=True)
        
        if not generated_texts:
            return [""] * len(questions)

        return [parse_assistant_response(text) for text in generated_texts]
