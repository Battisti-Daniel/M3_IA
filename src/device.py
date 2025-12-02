from __future__ import annotations

import torch


def get_device() -> torch.device:
    """Return best available device preferring CUDA."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def use_half_precision(device: torch.device) -> bool:
    return device.type == "cuda"
