import torch
import torch.nn as nn
from .chop import *
from torch.autograd import Variable

class rounding(nn.Module):
    def __init__(
        self, prec='s', subnormal=None, rmode=1, flip=0, explim=1, device='cpu', 
        p=0.5, randfunc=None, customs=None, random_state=0
    ):
        super(rounding, self).__init__()
        self.chop = chop(prec=prec, subnormal=subnormal, rmode=rmode, flip=flip, explim=explim, device=device, 
            p=p, randfunc= randfunc, customs=customs, random_state=random_state)

    def forward(self, x):
        return Variable(self.chop(x), requires_grad=True)