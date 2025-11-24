"""
Utility helpers for environment and device setup.

This small module provides helpers to consistently set random seeds and
to choose execution devices (`cpu`, `cuda`, or `mps`) based on a simple
configuration value.
"""

import torch
import random
import numpy as np

def set_seed(seed: int):
    """Set the random seed for Python, NumPy, and PyTorch.

    Ensures experiment reproducibility across multiple libraries. If CUDA
    is available, sets the manual seed for all CUDA devices as well.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def prepare_device(device_cfg: str):
    """Return a `torch.device` based on `device_cfg`.

    When `device_cfg` is `auto`, the function prefers MPS (Apple Silicon)
    if available, otherwise it falls back to CUDA if present or CPU.
    If `device_cfg` is a valid `torch.device` string (e.g., `cpu`, `cuda:0`),
    it returns the corresponding `torch.device` instance.
    """
    if device_cfg == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_cfg)
