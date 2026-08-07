from __future__ import annotations

import torch


def resolve_device(device: str) -> torch.device:
    """Resolve 'auto' to the best available device, or pass through explicit values."""
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
