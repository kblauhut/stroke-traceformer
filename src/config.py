from dataclasses import dataclass

@dataclass
class ModelConfig:
    embed_size: int = 128
    num_heads: int = 8
    dropout: float = 0
    num_layers: int = 3
    context_length: int = 150
    img_size: int = 100
    patch_size: int = 16

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")
