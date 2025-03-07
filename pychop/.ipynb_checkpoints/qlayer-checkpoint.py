import torch
from .lightchop import LightChop

class QuantizedLayer(torch.nn.Module):
    """Example of a quantized linear layer"""
    def __init__(self, 
                 exp_bits: int,
                 sig_bits: int,
                 rmode: str = "nearest"):
        
        super().__init__()
        self.quantizer = LightChop(exp_bits, sig_bits)
        self.rmode = rmode
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.quantizer.quantize(x, self.rmode)
