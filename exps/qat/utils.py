import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
import os

from pychop.tch.lightchop import LightChopSTE
from pychop import LightChop

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("class_images", exist_ok=True)
os.makedirs("qat_models", exist_ok=True)  # 保存 QAT 模型

from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

class MobileNetV3Small(nn.Module):
    def __init__(self, input_channels, num_classes):
        super().__init__()
        # Load pretrained weights
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
        self.backbone = mobilenet_v3_small(weights=weights)

        # Modify first conv for 1-channel input (e.g., MNIST/FashionMNIST)
        if input_channels != 3:
            old_conv = self.backbone.features[0][0]
            self.backbone.features[0][0] = nn.Conv2d(
                input_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False
            )
            # Approximate weight copy: average across RGB channels
            with torch.no_grad():
                new_weight = self.backbone.features[0][0].weight.mean(dim=1, keepdim=True)
                self.backbone.features[0][0].weight.copy_(new_weight)

        # Replace the final classifier Linear layer
        # Find the last Linear layer's in_features
        in_features = None
        for layer in reversed(self.backbone.classifier):
            if isinstance(layer, nn.Linear):
                in_features = layer.in_features
                break
        if in_features is None:
            raise ValueError("No Linear layer found in MobileNetV3 classifier")

        # Replace the last Linear with new one for your num_classes
        self.backbone.classifier[-1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

# ==================== QAT 模塊（使用 nn.Module 版本的 LightChopSTE） ====================
class QATConv2d(nn.Module):
    def __init__(self, original_conv: nn.Conv2d, chop: LightChop, quant_act: bool = True):
        super().__init__()
        self.weight = nn.Parameter(original_conv.weight.data.clone())
        self.bias = nn.Parameter(original_conv.bias.data.clone()) if original_conv.bias is not None else None
        self.stride = original_conv.stride
        self.padding = original_conv.padding
        self.dilation = original_conv.dilation
        self.groups = original_conv.groups

        self.weight_quant = LightChopSTE(
            exp_bits=chop.exp_bits,
            sig_bits=chop.sig_bits,
            rmode=chop.rmode,
            subnormal=True
        )

        # 關鍵：即使外部傳 quant_act=True，也允許內部強制關閉 act_quant
        self.act_quant = LightChopSTE(
            exp_bits=chop.exp_bits,
            sig_bits=chop.sig_bits,
            rmode=chop.rmode,
            subnormal=True
        ) if quant_act else None

    def forward(self, x):
        # 強制保護：如果層名包含 "features.0" 或 "classifier"，不量化激活
        # （但因為這裡沒有 name，需要在 replace 時額外處理，或統一關閉 act_quant）
        if self.act_quant is not None:
            x = self.act_quant(x)

        w_quant = self.weight_quant(self.weight)

        b_quant = None
        if self.bias is not None:
            b_quant = self.weight_quant(self.bias)

        return F.conv2d(x, w_quant, b_quant, self.stride, self.padding, self.dilation, self.groups)


class QATLinear(nn.Module):
    def __init__(self, original_linear: nn.Linear, chop: LightChop, quant_act: bool = True):
        super().__init__()
        self.weight = nn.Parameter(original_linear.weight.data.clone())
        self.bias = nn.Parameter(original_linear.bias.data.clone()) if original_linear.bias is not None else None

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


def replace_for_qat(model: nn.Module, chop: LightChop, quant_act: bool = False):
    """
    現在預設 quant_act=False，只量化權重
    """
    for name, module in list(model.named_children()):
        replaced = False
        if isinstance(module, nn.Conv2d):
            # 特殊處理：第一層不量化激活（即使 quant_act=True 也強制關）
            q_act = quant_act and 'features.0' not in name
            q_module = QATConv2d(module, chop, quant_act=q_act)
            setattr(model, name, q_module)
            replaced = True
        elif isinstance(module, nn.Linear):
            # 最後分類層不量化激活
            q_act = quant_act and 'classifier' not in name
            q_module = QATLinear(module, chop, quant_act=q_act)
            setattr(model, name, q_module)
            replaced = True
        
        if not replaced:
            replace_for_qat(module, chop, quant_act)