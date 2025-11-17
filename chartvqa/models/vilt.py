from transformers import ViltForQuestionAnswering, ViltProcessor
from .base import VQAModel
from typing import List
from PIL import Image
import torch

class ViltModel(VQAModel):
    """
    Implementation of VQAModel for ViLT.
    """
    
    def _load_model(self):
        """
        Download ViLT model and processor.
        """
        self.processor = ViltProcessor.from_pretrained(self.model_cfg.model_path)
        self.model = ViltForQuestionAnswering.from_pretrained(self.model_cfg.model_path)
        self.model.to(self.device)

    def infer(self, image: Image.Image, question: str) -> str:
        """
        Implement inference.
        """
        encoding = self.processor(image, question, return_tensors="pt")
        encoding = {k: v.to(self.device) for k, v in encoding.items()} 

        with torch.no_grad():
            outputs = self.model(**encoding)
        
        pred_idx = outputs.logits.argmax(-1).item()
        pred_text = self.model.config.id2label.get(pred_idx, str(pred_idx)).strip().lower()
        return pred_text
    
    def infer_batch(self, images: List[Image.Image], questions: List[str]) -> List[str]:
        """
        Implement batch inference.
        """
        encoding = self.processor(images, questions, return_tensors="pt", padding=True)
        encoding = {k: v.to(self.device) for k, v in encoding.items()} 

        with torch.no_grad():
            outputs = self.model(**encoding)
        
        pred_indices = outputs.logits.argmax(-1)
        pred_texts = [
            self.model.config.id2label.get(pred_idx.item(), str(pred_idx.item())).strip().lower()
            for pred_idx in pred_indices
        ]
        return pred_texts
