from abc import ABC, abstractmethod
from omegaconf import DictConfig
from typing import List
from PIL import Image
import torch

class VQAModel(ABC):
    """Abstract class for all VQA models."""
    
    def __init__(self, model_cfg: str, device: torch.device, wandb_logger=None):
        self.model_cfg = model_cfg
        self.device = device
        self.model = None
        self.processor = None
        self.wandb_logger = wandb_logger
        # quantization-related helpers
        self.quantized: bool = bool(getattr(model_cfg, "quantized", False))
        self.quant_bits: int = int(getattr(model_cfg, "quant_bits", 0) or 0)
        self.quant_backend: str | None = getattr(model_cfg, "quant_backend", None)
        # loading of the model and processor are implemented in inherited classes
        self._load_model()

    @abstractmethod
    def _load_model(self):
        """Download model and processor."""
        pass

    @abstractmethod
    def infer_batch(self, images: List[Image.Image], questions: List[str]) -> List[str]:
        """
        Implement batch inference for lists of (image, question) pairs.
        Return a list of cleaned answer strings.
        """
        pass

    @staticmethod
    def load_specific_model_from_config(model_cfg: DictConfig, device: torch.device, wandb_logger=None) -> 'VQAModel':
        """
        Creates and returns the instance of the specified model.
        """
        model_type = model_cfg.model_type
        
        if model_type == "AutoModelForVision2Seq":
            from .vision2seq import Vision2SeqModel
            return Vision2SeqModel(model_cfg, device)
        
        elif model_type == "ViltForQuestionAnswering":
            from .vilt import ViltModel
            return ViltModel(model_cfg, device)
        
        elif model_type == "Florence2":
            from .florence import Florence2Model
            return Florence2Model(model_cfg.model_path, device)
        elif model_type == "CustomVLM":
            if model_cfg.model_name == "TiQS":
                from .TiQS.TiQSModel import TiQSModel
                return TiQSModel(model_cfg, device, wandb_logger=wandb_logger)
            else:
                raise NotImplementedError("The asked model is not implemented yet.")
        
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
