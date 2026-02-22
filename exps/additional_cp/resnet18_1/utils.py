import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
import os

from pychop import ChopSTE
from pychop import Chop

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("class_images", exist_ok=True)
os.makedirs("qat_models", exist_ok=True)  


class ResNet18(nn.Module):
    """ResNet18 backbone adapted for custom input channels and number of classes."""
    def __init__(self, input_channels, num_classes):
        super().__init__()
        # Load pretrained weights
        weights = ResNet18_Weights.IMAGENET1K_V1
        self.backbone = resnet18(weights=weights)

        # Modify first conv for non-3-channel input (e.g., MNIST/FashionMNIST)
        if input_channels != 3:
            old_conv = self.backbone.conv1
            self.backbone.conv1 = nn.Conv2d(
                input_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False
            )
            # Approximate weight copy: average across RGB channels
            with torch.no_grad():
                new_weight = old_conv.weight.mean(dim=1, keepdim=True)  # (64, 3, 7, 7) -> (64, 1, 7, 7)
                self.backbone.conv1.weight.copy_(new_weight.repeat(1, input_channels, 1, 1))

        # Replace final fc layer
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


# ==================== QAT Modules (using ChopSTE) ====================
class QATConv2d(nn.Module):
    def __init__(self, original_conv: nn.Conv2d, chop: Chop, quant_act: bool = True):
        super().__init__()
        self.weight = nn.Parameter(original_conv.weight.data.clone())
        self.bias = nn.Parameter(original_conv.bias.data.clone()) if original_conv.bias is not None else None
        self.stride = original_conv.stride
        self.padding = original_conv.padding
        self.dilation = original_conv.dilation
        self.groups = original_conv.groups

        self.weight_quant = ChopSTE(
            exp_bits=chop.exp_bits,
            sig_bits=chop.sig_bits,
            rmode=chop.rmode,
            subnormal=False
        )

        # Activation quantization (can be disabled per-layer)
        self.act_quant = ChopSTE(
            exp_bits=chop.exp_bits,
            sig_bits=chop.sig_bits,
            rmode=chop.rmode,
            subnormal=False
        ) if quant_act else None

    def forward(self, x):
        if self.act_quant is not None:
            x = self.act_quant(x)
        w_quant = self.weight_quant(self.weight)
        b_quant = self.weight_quant(self.bias) if self.bias is not None else None
        return F.conv2d(x, w_quant, b_quant, self.stride, self.padding, self.dilation, self.groups)


class QATLinear(nn.Module):
    def __init__(self, original_linear: nn.Linear, chop: Chop, quant_act: bool = True):
        super().__init__()
        self.weight = nn.Parameter(original_linear.weight.data.clone())
        self.bias = nn.Parameter(original_linear.bias.data.clone()) if original_linear.bias is not None else None

        self.weight_quant = ChopSTE(
            exp_bits=chop.exp_bits,
            sig_bits=chop.sig_bits,
            rmode=chop.rmode,
            subnormal=False
        )

        self.act_quant = ChopSTE(
            exp_bits=chop.exp_bits,
            sig_bits=chop.sig_bits,
            rmode=chop.rmode,
            subnormal=False
        ) if quant_act else None

    def forward(self, x):
        if self.act_quant is not None:
            x = self.act_quant(x)
        w_quant = self.weight_quant(self.weight)
        b_quant = self.weight_quant(self.bias) if self.bias is not None else None
        return F.linear(x, w_quant, b_quant)


def replace_for_qat(model: nn.Module, chop: Chop, quant_act: bool = False):
    """
    Recursively replace Conv2d and Linear layers with QAT versions.
    By default quant_act=False (only quantize weights).
    Special handling: first conv (conv1) and final fc do not quantize activations.
    """
    def _replace(module, seen_first_conv):
        for name, child in list(module.named_children()):
            replaced = False
            if isinstance(child, nn.Conv2d):
                q_act = quant_act if seen_first_conv[0] else False
                q_module = QATConv2d(child, chop, quant_act=q_act)
                setattr(module, name, q_module)
                replaced = True
                seen_first_conv[0] = True  # Any subsequent convs get activation quantization
            elif isinstance(child, nn.Linear):
                # No activation quantization before the final classifier
                q_module = QATLinear(child, chop, quant_act=False)
                setattr(module, name, q_module)
                replaced = True
            
            if not replaced:
                _replace(child, seen_first_conv)
    
    _replace(model, [False])  # seen_first_conv as mutable list