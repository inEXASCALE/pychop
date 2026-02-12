import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import os

from pychop.tch.lightchop import LightChopSTE
from pychop import LightChop
# pychop.backend("torch")  # 如果需要的话保留

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("class_images", exist_ok=True)
os.makedirs("qat_models", exist_ok=True)  # 保存 QAT 模型

class ResNet(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(ResNet, self).__init__()
        self.backbone = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
        if input_channels == 1:
            self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

class QATConv2d(nn.Module):
    def __init__(self, original_conv: nn.Conv2d, chop: LightChop, quant_act: bool = True):
        super().__init__()
        # 複製原始權重（保持參數可訓練）
        self.weight = nn.Parameter(original_conv.weight.data.clone())
        if original_conv.bias is not None:
            self.bias = nn.Parameter(original_conv.bias.data.clone())
        else:
            self.bias = None

        self.stride = original_conv.stride
        self.padding = original_conv.padding
        self.dilation = original_conv.dilation
        self.groups = original_conv.groups

        # 創建量化模塊（作為子模塊）
        self.weight_quant = LightChopSTE(
            exp_bits=chop.exp_bits,
            sig_bits=chop.sig_bits,
            rmode=chop.rmode,
            subnormal=True   # 根據你的需求決定是否允許 subnormal
        )

        self.act_quant = LightChopSTE(
            exp_bits=chop.exp_bits,
            sig_bits=chop.sig_bits,
            rmode=chop.rmode,
            subnormal=True
        ) if quant_act else None

    def forward(self, x):
        # 量化激活（如果啟用）
        if self.act_quant is not None:
            x = self.act_quant(x)

        # 量化權重（訓練時使用 STE，eval 時也會套用量化）
        w_quant = self.weight_quant(self.weight)

        # 偏置也量化（如果有）
        b_quant = None
        if self.bias is not None:
            b_quant = self.weight_quant(self.bias)   # 注意：偏置通常也用同樣的格式量化

        # 執行卷積
        out = F.conv2d(
            x, w_quant, b_quant,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups
        )
        return out


class QATLinear(nn.Module):
    def __init__(self, original_linear: nn.Linear, chop: LightChop, quant_act: bool = True):
        super().__init__()
        self.weight = nn.Parameter(original_linear.weight.data.clone())
        if original_linear.bias is not None:
            self.bias = nn.Parameter(original_linear.bias.data.clone())
        else:
            self.bias = None

        self.weight_quant = LightChopSTE(
            exp_bits=chop.exp_bits,
            sig_bits=chop.sig_bits,
            rmode=chop.rmode,
            subnormal=True
        )

        self.act_quant = LightChopSTE(
            exp_bits=chop.exp_bits,
            sig_bits=chop.sig_bits,
            rmode=chop.rmode,
            subnormal=True
        ) if quant_act else None

    def forward(self, x):
        if self.act_quant is not None:
            x = self.act_quant(x)

        w_quant = self.weight_quant(self.weight)

        b_quant = None
        if self.bias is not None:
            b_quant = self.weight_quant(self.bias)

        return F.linear(x, w_quant, b_quant)


def replace_for_qat(model: nn.Module, chop: LightChop, quant_act: bool = True):
    """
    遞迴地把所有 nn.Conv2d 和 nn.Linear 替換成 QAT 版本
    注意：這裡會直接修改 model 的結構
    """
    for name, module in list(model.named_children()):
        if isinstance(module, nn.Conv2d):
            qat_module = QATConv2d(module, chop, quant_act=quant_act)
            setattr(model, name, qat_module)
        elif isinstance(module, nn.Linear):
            qat_module = QATLinear(module, chop, quant_act=quant_act)
            setattr(model, name, qat_module)
        else:
            # 遞迴處理子模塊（例如 BasicBlock 裡面的 conv）
            replace_for_qat(module, chop, quant_act)