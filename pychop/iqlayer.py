import torch

class IntQuantizedLayer(torch.nn.Module):
    """
    __init__(config)
        Apply ``pychop`` to quantization aware training, 
        One can feed [quant | chop | fixed_point] module as base for quantization.

    """

    def __init__(self, config):
        super(IntQuantizedLayer, self).__init__()
        self.quant = config
        
    def forward(self, x):
        return self.quant(x)
        