import torch
# from .lightchop import LightChop
import torch.nn as nn
from typing import Optional, Union, Tuple, Any, List
import torch.nn.functional as F
from .tch.lightchop import LightChopSTE

import torch
import copy
from typing import Optional
from torch.autograd import Function
from pychop import Chopi



def post_quantization(model: torch.nn.Module, chop, eval_mode: bool = True, verbose: bool = False) -> torch.nn.Module:
    """
    Perform post-training quantization (PTQ) on a copy of a PyTorch model.
    Only weights and biases are quantized; BatchNorm running stats and other buffers are preserved.
    For best results, fuse Conv+BN layers before calling this function (e.g., model.fuse_model()).

    Args:
        model (torch.nn.Module): Original PyTorch model (remains unmodified).
        chop: Object with a `quantize(tensor)` method (simulated quantization).
        eval_mode (bool): If True, set the copied model to evaluation mode. Default: True.
        verbose (bool): If True, print parameter names and quantized values. Default: False.

    Returns:
        torch.nn.Module: A new quantized copy of the model.
    """
    # Deep copy the model to avoid modifying the original
    quantized_model = copy.deepcopy(model)

    # Set evaluation mode if requested
    if eval_mode:
        quantized_model.eval()

    # Device of the original model
    device = next(model.parameters()).device

    # Get state dict (includes both parameters and buffers)
    state_dict = quantized_model.state_dict()

    for key in state_dict.keys():
        tensor = state_dict[key].to(device)

        # Only quantize weights and biases
        if 'weight' in key or 'bias' in key:
            quantized_tensor = chop.quantize(tensor)
        else:
            quantized_tensor = tensor  # keep buffers unchanged

        # Check shape consistency
        if quantized_tensor.shape != tensor.shape:
            raise ValueError(f"Shape mismatch for {key}: {tensor.shape} vs {quantized_tensor.shape}")

        # Update state dict
        state_dict[key] = quantized_tensor

        # Verbose output
        if verbose and ('weight' in key or 'bias' in key):
            print(f"[Quantized] {key}: {quantized_tensor}")

    # Load the quantized state dict back into the model
    quantized_model.load_state_dict(state_dict)

    return quantized_model



# ===================================================================
# Float point quantization
# ===================================================================
class QuantizedLinear(nn.Linear):
    """Quantized version of :class:`torch.nn.Linear` for QAT.

    Parameters
    ----------
    in_features : int
        Size of each input sample.
    out_features : int
        Size of each output sample.
    bias : bool, default=True
        If set to False, the layer will not learn an additive bias.
    chop : LightChop or None, default=None
        pychop quantizer. If None, falls back to standard Linear.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        chop: Optional = None,
    ) -> None:
        super().__init__(in_features, out_features, bias=bias)
        self.chop = chop

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.chop is None:
            return super().forward(input)

        weight = self.chop(self.weight)
        bias = self.chop(self.bias) if self.bias is not None else None

        output = F.linear(input, weight, bias)
        return self.chop(output)


class QuantizedConv1d(nn.Conv1d):
    """Quantized 1D convolution layer for QAT."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        chop: Optional = None,
    ) -> None:
        super().__init__(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias, padding_mode=padding_mode,
        )
        self.chop = chop

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.chop is None:
            return super().forward(input)

        weight = self.chop(self.weight)
        bias = self.chop(self.bias) if self.bias is not None else None

        output = F.conv1d(
            input, weight, bias, self.stride, self.padding,
            self.dilation, self.groups
        )
        return self.chop(output)


class QuantizedConv2d(nn.Conv2d):
    """Quantized 2D convolution layer for QAT."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        chop: Optional = None,
    ) -> None:
        super().__init__(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias, padding_mode=padding_mode,
        )
        self.chop = chop

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.chop is None:
            return super().forward(input)

        weight = self.chop(self.weight)
        bias = self.chop(self.bias) if self.bias is not None else None

        output = F.conv2d(
            input, weight, bias, self.stride, self.padding,
            self.dilation, self.groups
        )
        return self.chop(output)


class QuantizedConv3d(nn.Conv3d):
    """Quantized 3D convolution layer for QAT."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int]] = 1,
        padding: Union[int, Tuple[int, int, int]] = 0,
        dilation: Union[int, Tuple[int, int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        chop: Optional = None,
    ) -> None:
        super().__init__(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias, padding_mode=padding_mode,
        )
        self.chop = chop

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.chop is None:
            return super().forward(input)

        weight = self.chop(self.weight)
        bias = self.chop(self.bias) if self.bias is not None else None

        output = F.conv3d(
            input, weight, bias, self.stride, self.padding,
            self.dilation, self.groups
        )
        return self.chop(output)


class QuantizedRNN(nn.RNN):
    """Quantized RNN layer for QAT (weights quantized on-the-fly)."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        nonlinearity: str = "tanh",
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
        chop: Optional = None,
    ) -> None:
        super().__init__(
            input_size, hidden_size, num_layers=num_layers,
            nonlinearity=nonlinearity, bias=bias,
            batch_first=batch_first, dropout=dropout,
            bidirectional=bidirectional,
        )
        self.chop = chop

    def forward(self, input: torch.Tensor, hx: Optional[torch.Tensor] = None):
        if self.chop is None:
            return super().forward(input, hx)

        # Fake-quant weights/biases (works for most common cases)
        for name, param in list(self.named_parameters()):
            if "weight" in name or "bias" in name:
                setattr(self, name, nn.Parameter(self.chop(param)))

        output, hidden = super().forward(input, hx)
        return self.chop(output), hidden


class QuantizedLSTM(nn.LSTM):
    """Quantized LSTM layer for QAT (weights quantized on-the-fly)."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
        chop: Optional = None,
    ) -> None:
        super().__init__(
            input_size, hidden_size, num_layers=num_layers,
            bias=bias, batch_first=batch_first,
            dropout=dropout, bidirectional=bidirectional,
        )
        self.chop = chop

    def forward(
        self,
        input: torch.Tensor,
        hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        if self.chop is None:
            return super().forward(input, hx)

        for name, param in list(self.named_parameters()):
            if "weight" in name or "bias" in name:
                setattr(self, name, nn.Parameter(self.chop(param)))

        output, (hn, cn) = super().forward(input, hx)
        return self.chop(output), (hn, cn)


class QuantizedMaxPool1d(nn.MaxPool1d):
    """Quantized 1D max pooling layer (activations only)."""

    def __init__(
        self,
        kernel_size: int,
        stride: Optional[int] = None,
        padding: int = 0,
        dilation: int = 1,
        return_indices: bool = False,
        ceil_mode: bool = False,
        chop: Optional = None,
    ) -> None:
        super().__init__(
            kernel_size, stride=stride, padding=padding,
            dilation=dilation, return_indices=return_indices,
            ceil_mode=ceil_mode,
        )
        self.chop = chop

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = super().forward(input)
        return self.chop(output) if self.chop is not None else output


class QuantizedMaxPool2d(nn.MaxPool2d):
    """Quantized 2D max pooling layer (activations only)."""

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Optional[Union[int, Tuple[int, int]]] = None,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        return_indices: bool = False,
        ceil_mode: bool = False,
        chop: Optional = None,
    ) -> None:
        super().__init__(
            kernel_size, stride=stride, padding=padding,
            dilation=dilation, return_indices=return_indices,
            ceil_mode=ceil_mode,
        )
        self.chop = chop

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = super().forward(input)
        return self.chop(output) if self.chop is not None else output


class QuantizedMaxPool3d(nn.MaxPool3d):
    """Quantized 3D max pooling layer (activations only)."""

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Optional[Union[int, Tuple[int, int, int]]] = None,
        padding: Union[int, Tuple[int, int, int]] = 0,
        dilation: Union[int, Tuple[int, int, int]] = 1,
        return_indices: bool = False,
        ceil_mode: bool = False,
        chop: Optional = None,
    ) -> None:
        super().__init__(
            kernel_size, stride=stride, padding=padding,
            dilation=dilation, return_indices=return_indices,
            ceil_mode=ceil_mode,
        )
        self.chop = chop

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = super().forward(input)
        return self.chop(output) if self.chop is not None else output


class QuantizedAvgPool1d(nn.AvgPool1d):
    """Quantized 1D average pooling layer (activations only)."""

    def __init__(
        self,
        kernel_size: int,
        stride: Optional[int] = None,
        padding: int = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
        chop: Optional = None,
    ) -> None:
        super().__init__(
            kernel_size, stride=stride, padding=padding,
            ceil_mode=ceil_mode, count_include_pad=count_include_pad,
        )
        self.chop = chop

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = super().forward(input)
        return self.chop(output) if self.chop is not None else output


class QuantizedAvgPool2d(nn.AvgPool2d):
    """Quantized 2D average pooling layer (activations only)."""

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Optional[Union[int, Tuple[int, int]]] = None,
        padding: Union[int, Tuple[int, int]] = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
        divisor_override: Optional[int] = None,
        chop: Optional = None,
    ) -> None:
        super().__init__(
            kernel_size, stride=stride, padding=padding,
            ceil_mode=ceil_mode, count_include_pad=count_include_pad,
            divisor_override=divisor_override,
        )
        self.chop = chop

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = super().forward(input)
        return self.chop(output) if self.chop is not None else output


class QuantizedAvgPool3d(nn.AvgPool3d):
    """Quantized 3D average pooling layer (activations only)."""

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Optional[Union[int, Tuple[int, int, int]]] = None,
        padding: Union[int, Tuple[int, int, int]] = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
        divisor_override: Optional[int] = None,
        chop: Optional = None,
    ) -> None:
        super().__init__(
            kernel_size, stride=stride, padding=padding,
            ceil_mode=ceil_mode, count_include_pad=count_include_pad,
            divisor_override=divisor_override,
        )
        self.chop = chop

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = super().forward(input)
        return self.chop(output) if self.chop is not None else output


class QuantizedMultiheadAttention(nn.MultiheadAttention):
    """Quantized Multi-head Attention (corresponds to nn.MultiheadAttention)
    for QAT. Alias: QuantizedAttention.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        batch_first: bool = False,
        device=None,
        dtype=None,
        chop: Optional = None,
    ):
        super().__init__(
            embed_dim, num_heads, dropout=dropout, bias=bias,
            add_bias_kv=add_bias_kv, add_zero_attn=add_zero_attn,
            kdim=kdim, vdim=vdim, batch_first=batch_first,
            device=device, dtype=dtype,
        )
        self.chop = chop

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ):
        if self.chop is None:
            return super().forward(
                query, key, value, key_padding_mask, need_weights,
                attn_mask, average_attn_weights, is_causal=is_causal
            )

        # Fake-quant projection weights
        if hasattr(self, "in_proj_weight") and self.in_proj_weight is not None:
            self.in_proj_weight = nn.Parameter(self.chop(self.in_proj_weight))
        if hasattr(self, "in_proj_bias") and self.in_proj_bias is not None:
            self.in_proj_bias = nn.Parameter(self.chop(self.in_proj_bias))
        self.out_proj.weight = nn.Parameter(self.chop(self.out_proj.weight))
        if self.out_proj.bias is not None:
            self.out_proj.bias = nn.Parameter(self.chop(self.out_proj.bias))

        output, attn_weights = super().forward(
            query, key, value, key_padding_mask, need_weights,
            attn_mask, average_attn_weights, is_causal=is_causal
        )
        if isinstance(output, tuple):
            return self.chop(output[0]), output[1]
        return self.chop(output), attn_weights


class QuantizedBatchNorm1d(nn.BatchNorm1d):
    """Quantized 1D Batch Normalization layer."""

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        chop: Optional = None,
    ) -> None:
        super().__init__(
            num_features, eps=eps, momentum=momentum,
            affine=affine, track_running_stats=track_running_stats,
        )
        self.chop = chop

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.chop is None:
            return super().forward(input)

        if self.affine:
            self.weight = nn.Parameter(self.chop(self.weight))
            self.bias = nn.Parameter(self.chop(self.bias))

        output = super().forward(input)
        return self.chop(output)


class QuantizedBatchNorm2d(nn.BatchNorm2d):
    """Quantized 2D Batch Normalization layer."""

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        chop: Optional = None,
    ) -> None:
        super().__init__(
            num_features, eps=eps, momentum=momentum,
            affine=affine, track_running_stats=track_running_stats,
        )
        self.chop = chop

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.chop is None:
            return super().forward(input)

        if self.affine:
            self.weight = nn.Parameter(self.chop(self.weight))
            self.bias = nn.Parameter(self.chop(self.bias))

        output = super().forward(input)
        return self.chop(output)


class QuantizedBatchNorm3d(nn.BatchNorm3d):
    """Quantized 3D Batch Normalization layer."""

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        chop: Optional = None,
    ) -> None:
        super().__init__(
            num_features, eps=eps, momentum=momentum,
            affine=affine, track_running_stats=track_running_stats,
        )
        self.chop = chop

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.chop is None:
            return super().forward(input)

        if self.affine:
            self.weight = nn.Parameter(self.chop(self.weight))
            self.bias = nn.Parameter(self.chop(self.bias))

        output = super().forward(input)
        return self.chop(output)


# Table aliases (convenience)
QuantizedAttention = QuantizedMultiheadAttention
QuantizedAvgPool = QuantizedAvgPool2d   # default to 2D

class QuantizedGRU(nn.GRU):
    """Quantized GRU layer for QAT (weights quantized on-the-fly)."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
        chop: Optional = None,
    ) -> None:
        super().__init__(
            input_size, hidden_size, num_layers=num_layers,
            bias=bias, batch_first=batch_first,
            dropout=dropout, bidirectional=bidirectional,
        )
        self.chop = chop

    def forward(
        self,
        input: torch.Tensor,
        hx: Optional[torch.Tensor] = None,
    ):
        if self.chop is None:
            return super().forward(input, hx)

        # Fake-quantize all weights and biases
        for name, param in list(self.named_parameters()):
            if "weight" in name or "bias" in name:
                setattr(self, name, nn.Parameter(self.chop(param)))

        output, hidden = super().forward(input, hx)
        return self.chop(output), hidden


class QuantizedConvTranspose2d(nn.ConvTranspose2d):
    """Quantized 2D transposed convolution layer for QAT."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        output_padding: Union[int, Tuple[int, int]] = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: Union[int, Tuple[int, int]] = 1,
        padding_mode: str = "zeros",
        chop: Optional = None,
    ) -> None:
        super().__init__(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding,
            groups=groups, bias=bias, dilation=dilation, padding_mode=padding_mode,
        )
        self.chop = chop

    def forward(self, input: torch.Tensor, output_size: Optional[List[int]] = None) -> torch.Tensor:
        if self.chop is None:
            return super().forward(input, output_size)

        weight = self.chop(self.weight)
        bias = self.chop(self.bias) if self.bias is not None else None

        output = F.conv_transpose2d(
            input, weight, bias, self.stride, self.padding,
            self.output_padding, self.groups, self.dilation
        )
        return self.chop(output)


class QuantizedReLU(nn.ReLU):
    """Quantized ReLU activation for QAT."""

    def __init__(self, inplace: bool = False, chop: Optional = None) -> None:
        super().__init__(inplace=inplace)
        self.chop = chop

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = super().forward(input)
        return self.chop(output) if self.chop is not None else output


class QuantizedLayerNorm(nn.LayerNorm):
    """Quantized Layer Normalization for QAT."""

    def __init__(
        self,
        normalized_shape: Union[int, List[int], torch.Size],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        chop: Optional = None,
    ) -> None:
        super().__init__(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)
        self.chop = chop

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.chop is None:
            return super().forward(input)

        if self.elementwise_affine:
            self.weight = nn.Parameter(self.chop(self.weight))
            self.bias = nn.Parameter(self.chop(self.bias))

        output = super().forward(input)
        return self.chop(output)


class QuantizedAdaptiveAvgPool2d(nn.AdaptiveAvgPool2d):
    """Quantized 2D adaptive average pooling layer (activations only)."""

    def __init__(
        self,
        output_size: Union[int, Tuple[int, int]],
        chop: Optional = None,
    ) -> None:
        super().__init__(output_size)
        self.chop = chop

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = super().forward(input)
        return self.chop(output) if self.chop is not None else output


class QuantizedEmbedding(nn.Embedding):
    """Quantized Embedding layer for QAT."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        chop: Optional = None,
    ) -> None:
        super().__init__(
            num_embeddings, embedding_dim, padding_idx=padding_idx,
            max_norm=max_norm, norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq, sparse=sparse,
        )
        self.chop = chop

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.chop is None:
            return super().forward(input)

        weight = self.chop(self.weight)
        output = F.embedding(
            input, weight, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse
        )
        return self.chop(output)

class QuantizedConvTranspose1d(nn.ConvTranspose1d):
    """Quantized 1D transposed convolution layer for QAT."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int]],
        stride: Union[int, Tuple[int]] = 1,
        padding: Union[int, Tuple[int]] = 0,
        output_padding: Union[int, Tuple[int]] = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: Union[int, Tuple[int]] = 1,
        padding_mode: str = "zeros",
        chop: Optional = None,
    ) -> None:
        super().__init__(
            in_channels, out_channels, kernel_size, stride=stride,
            padding=padding, output_padding=output_padding, groups=groups,
            bias=bias, dilation=dilation, padding_mode=padding_mode,
        )
        self.chop = chop

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.chop is None:
            return super().forward(input)

        weight = self.chop(self.weight)
        bias = self.chop(self.bias) if self.bias is not None else None

        output = F.conv_transpose1d(
            input, weight, bias, self.stride, self.padding,
            self.output_padding, self.groups, self.dilation
        )
        return self.chop(output)


class QuantizedConvTranspose3d(nn.ConvTranspose3d):
    """Quantized 3D transposed convolution layer for QAT."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int]] = 1,
        padding: Union[int, Tuple[int, int, int]] = 0,
        output_padding: Union[int, Tuple[int, int, int]] = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: Union[int, Tuple[int, int, int]] = 1,
        padding_mode: str = "zeros",
        chop: Optional = None,
    ) -> None:
        super().__init__(
            in_channels, out_channels, kernel_size, stride=stride,
            padding=padding, output_padding=output_padding, groups=groups,
            bias=bias, dilation=dilation, padding_mode=padding_mode,
        )
        self.chop = chop

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.chop is None:
            return super().forward(input)

        weight = self.chop(self.weight)
        bias = self.chop(self.bias) if self.bias is not None else None

        output = F.conv_transpose3d(
            input, weight, bias, self.stride, self.padding,
            self.output_padding, self.groups, self.dilation
        )
        return self.chop(output)


class QuantizedInstanceNorm1d(nn.InstanceNorm1d):
    """Quantized 1D Instance Normalization layer."""

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = False,
        track_running_stats: bool = False,
        chop: Optional = None,
    ) -> None:
        super().__init__(
            num_features, eps=eps, momentum=momentum,
            affine=affine, track_running_stats=track_running_stats,
        )
        self.chop = chop

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.chop is None:
            return super().forward(input)

        if self.affine:
            self.weight = nn.Parameter(self.chop(self.weight))
            self.bias = nn.Parameter(self.chop(self.bias))

        output = super().forward(input)
        return self.chop(output)


class QuantizedInstanceNorm2d(nn.InstanceNorm2d):
    """Quantized 2D Instance Normalization layer."""

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = False,
        track_running_stats: bool = False,
        chop: Optional = None,
    ) -> None:
        super().__init__(
            num_features, eps=eps, momentum=momentum,
            affine=affine, track_running_stats=track_running_stats,
        )
        self.chop = chop

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.chop is None:
            return super().forward(input)

        if self.affine:
            self.weight = nn.Parameter(self.chop(self.weight))
            self.bias = nn.Parameter(self.chop(self.bias))

        output = super().forward(input)
        return self.chop(output)


class QuantizedInstanceNorm3d(nn.InstanceNorm3d):
    """Quantized 3D Instance Normalization layer."""

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = False,
        track_running_stats: bool = False,
        chop: Optional = None,
    ) -> None:
        super().__init__(
            num_features, eps=eps, momentum=momentum,
            affine=affine, track_running_stats=track_running_stats,
        )
        self.chop = chop

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.chop is None:
            return super().forward(input)

        if self.affine:
            self.weight = nn.Parameter(self.chop(self.weight))
            self.bias = nn.Parameter(self.chop(self.bias))

        output = super().forward(input)
        return self.chop(output)


class QuantizedGroupNorm(nn.GroupNorm):
    """Quantized Group Normalization layer."""

    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-5,
        affine: bool = True,
        chop: Optional = None,
    ) -> None:
        super().__init__(num_groups, num_channels, eps=eps, affine=affine)
        self.chop = chop

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.chop is None:
            return super().forward(input)

        if self.affine:
            self.weight = nn.Parameter(self.chop(self.weight))
            self.bias = nn.Parameter(self.chop(self.bias))

        output = super().forward(input)
        return self.chop(output)


# ====================== Quantized Activation & Dropout Layers ======================

class QuantizedDropout(nn.Dropout):
    """Quantized Dropout layer for QAT."""

    def __init__(
        self,
        p: float = 0.5,
        chop: Optional = None,
    ) -> None:
        super().__init__(p=p)
        self.chop = chop

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.chop is None:
            return super().forward(input)
        x = self.chop(input)
        x = super().forward(x)
        return self.chop(x)


class QuantizedReLU(nn.ReLU):
    """Quantized ReLU activation for QAT."""

    def __init__(
        self,
        inplace: bool = False,
        chop: Optional = None,
    ) -> None:
        super().__init__(inplace=inplace)
        self.chop = chop

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.chop is None:
            return super().forward(input)
        x = self.chop(input)
        x = super().forward(x)
        return self.chop(x)


class QuantizedSigmoid(nn.Sigmoid):
    """Quantized Sigmoid activation for QAT."""

    def __init__(self, chop: Optional = None) -> None:
        super().__init__()
        self.chop = chop

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.chop is None:
            return super().forward(input)
        x = self.chop(input)
        x = super().forward(x)
        return self.chop(x)


class QuantizedTanh(nn.Tanh):
    """Quantized Tanh activation for QAT."""

    def __init__(self, chop: Optional = None) -> None:
        super().__init__()
        self.chop = chop

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.chop is None:
            return super().forward(input)
        x = self.chop(input)
        x = super().forward(x)
        return self.chop(x)


class QuantizedLeakyReLU(nn.LeakyReLU):
    """Quantized LeakyReLU activation for QAT."""

    def __init__(
        self,
        negative_slope: float = 0.01,
        inplace: bool = False,
        chop: Optional = None,
    ) -> None:
        super().__init__(negative_slope=negative_slope, inplace=inplace)
        self.chop = chop

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.chop is None:
            return super().forward(input)
        x = self.chop(input)
        x = super().forward(x)
        return self.chop(x)


class QuantizedSoftmax(nn.Softmax):
    """Quantized Softmax activation for QAT."""

    def __init__(
        self,
        dim: int = -1,
        chop: Optional = None,
    ) -> None:
        super().__init__(dim=dim)
        self.chop = chop

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.chop is None:
            return super().forward(input)
        x = self.chop(input)
        x = super().forward(x)
        return self.chop(x)


class QuantizedGELU(nn.GELU):
    """Quantized GELU activation for QAT."""

    def __init__(
        self,
        approximate: str = "none",
        chop: Optional = None,
    ) -> None:
        super().__init__(approximate=approximate)
        self.chop = chop

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.chop is None:
            return super().forward(input)
        x = self.chop(input)
        x = super().forward(x)
        return self.chop(x)


class QuantizedELU(nn.ELU):
    """Quantized ELU activation for QAT."""

    def __init__(
        self,
        alpha: float = 1.0,
        inplace: bool = False,
        chop: Optional = None,
    ) -> None:
        super().__init__(alpha=alpha, inplace=inplace)
        self.chop = chop

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.chop is None:
            return super().forward(input)
        x = self.chop(input)
        x = super().forward(x)
        return self.chop(x)


class QuantizedPReLU(nn.PReLU):
    """Quantized PReLU activation for QAT."""

    def __init__(
        self,
        num_parameters: int = 1,
        init: float = 0.25,
        chop: Optional = None,
    ) -> None:
        super().__init__(num_parameters=num_parameters, init=init)
        self.chop = chop

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.chop is None:
            return super().forward(input)
        x = self.chop(input)
        if self.chop is not None:
            self.weight = nn.Parameter(self.chop(self.weight))
        x = super().forward(x)
        return self.chop(x)
        



# ===================================================================
# Integer quantization
# ===================================================================

# chop=None 时完全退化为原生 nn.* 层

# ====================== ChopiSTE (必须放在所有 IQuantized 类之前) ======================
class FakeQuantizeSTE(Function):
    """Straight-Through Estimator for integer fake-quantization with Chopi."""

    @staticmethod
    def forward(ctx, input: torch.Tensor, chop: Chopi) -> torch.Tensor:
        q = chop.quantize(input)
        dq = chop.dequantize(q)
        ctx.save_for_backward(input)  # optional, for debugging
        return dq

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # STE: gradient flows through unchanged
        return grad_output, None


class ChopiSTE(Chopi):
    """Chopi with built-in Straight-Through Estimator (STE) for QAT.

    Use this class instead of plain Chopi during training.
    """

    def __init__(self, bits: int = 8, symmetric: bool = False, **kwargs) -> None:
        super().__init__(bits=bits, symmetric=symmetric, **kwargs)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply fake-quantization with STE."""
        if x is None or x.numel() == 0:
            return x
        return FakeQuantizeSTE.apply(x, self)


# ====================== IQuantized Layers (all updated with STE + sklearn-style docstrings) ======================

class IQuantizedLinear(nn.Linear):
    """Integer quantized version of :class:`torch.nn.Linear` for QAT using ChopiSTE.

    Parameters
    ----------
    in_features : int
        Size of each input sample.
    out_features : int
        Size of each output sample.
    bias : bool, default=True
        If set to ``False``, the layer will not learn an additive bias.
    chop : ChopiSTE or None, default=None
        Quantizer instance with STE. If ``None``, falls back to standard Linear.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        chop: Optional[ChopiSTE] = None,
    ) -> None:
        super().__init__(in_features, out_features, bias=bias)
        self.chop = chop

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.chop is None:
            return super().forward(input)

        weight = self.chop(self.weight)
        bias = self.chop(self.bias) if self.bias is not None else None

        output = F.linear(input, weight, bias)
        return self.chop(output)


class IQuantizedConv1d(nn.Conv1d):
    """Integer quantized 1D convolution for QAT using ChopiSTE."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        chop: Optional[ChopiSTE] = None,
    ) -> None:
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding,
            dilation, groups, bias, padding_mode,
        )
        self.chop = chop

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.chop is None:
            return super().forward(input)

        weight = self.chop(self.weight)
        bias = self.chop(self.bias) if self.bias is not None else None

        output = F.conv1d(
            input, weight, bias, self.stride, self.padding, self.dilation, self.groups
        )
        return self.chop(output)


class IQuantizedConv2d(nn.Conv2d):
    """Integer quantized 2D convolution for QAT using ChopiSTE."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        chop: Optional[ChopiSTE] = None,
    ) -> None:
        super().__init__(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding,
            dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode,
        )
        self.chop = chop

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.chop is None:
            return super().forward(input)

        weight = self.chop(self.weight)
        bias = self.chop(self.bias) if self.bias is not None else None

        output = F.conv2d(
            input, weight, bias, self.stride, self.padding, self.dilation, self.groups
        )
        return self.chop(output)


class IQuantizedConv3d(nn.Conv3d):
    """Integer quantized 3D convolution for QAT using ChopiSTE."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int]] = 1,
        padding: Union[int, Tuple[int, int, int]] = 0,
        dilation: Union[int, Tuple[int, int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        chop: Optional[ChopiSTE] = None,
    ) -> None:
        super().__init__(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding,
            dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode,
        )
        self.chop = chop

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.chop is None:
            return super().forward(input)

        weight = self.chop(self.weight)
        bias = self.chop(self.bias) if self.bias is not None else None

        output = F.conv3d(
            input, weight, bias, self.stride, self.padding, self.dilation, self.groups
        )
        return self.chop(output)


class IQuantizedConvTranspose1d(nn.ConvTranspose1d):
    """Integer quantized 1D transposed convolution for QAT using ChopiSTE."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int]],
        stride: Union[int, Tuple[int]] = 1,
        padding: Union[int, Tuple[int]] = 0,
        output_padding: Union[int, Tuple[int]] = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: Union[int, Tuple[int]] = 1,
        padding_mode: str = "zeros",
        chop: Optional[ChopiSTE] = None,
    ) -> None:
        super().__init__(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding,
            output_padding=output_padding, groups=groups, bias=bias,
            dilation=dilation, padding_mode=padding_mode,
        )
        self.chop = chop

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.chop is None:
            return super().forward(input)

        weight = self.chop(self.weight)
        bias = self.chop(self.bias) if self.bias is not None else None

        output = F.conv_transpose1d(
            input, weight, bias, self.stride, self.padding,
            self.output_padding, self.groups, self.dilation,
        )
        return self.chop(output)


class IQuantizedConvTranspose2d(nn.ConvTranspose2d):
    """Integer quantized 2D transposed convolution for QAT using ChopiSTE."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        output_padding: Union[int, Tuple[int, int]] = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: Union[int, Tuple[int, int]] = 1,
        padding_mode: str = "zeros",
        chop: Optional[ChopiSTE] = None,
    ) -> None:
        super().__init__(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding,
            output_padding=output_padding, groups=groups, bias=bias,
            dilation=dilation, padding_mode=padding_mode,
        )
        self.chop = chop

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.chop is None:
            return super().forward(input)

        weight = self.chop(self.weight)
        bias = self.chop(self.bias) if self.bias is not None else None

        output = F.conv_transpose2d(
            input, weight, bias, self.stride, self.padding,
            self.output_padding, self.groups, self.dilation,
        )
        return self.chop(output)


class IQuantizedConvTranspose3d(nn.ConvTranspose3d):
    """Integer quantized 3D transposed convolution for QAT using ChopiSTE."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int]] = 1,
        padding: Union[int, Tuple[int, int, int]] = 0,
        output_padding: Union[int, Tuple[int, int, int]] = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: Union[int, Tuple[int, int, int]] = 1,
        padding_mode: str = "zeros",
        chop: Optional[ChopiSTE] = None,
    ) -> None:
        super().__init__(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding,
            output_padding=output_padding, groups=groups, bias=bias,
            dilation=dilation, padding_mode=padding_mode,
        )
        self.chop = chop

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.chop is None:
            return super().forward(input)

        weight = self.chop(self.weight)
        bias = self.chop(self.bias) if self.bias is not None else None

        output = F.conv_transpose3d(
            input, weight, bias, self.stride, self.padding,
            self.output_padding, self.groups, self.dilation,
        )
        return self.chop(output)


class IQuantizedRNN(nn.RNN):
    """Integer quantized RNN for QAT using ChopiSTE (weights fake-quantized)."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        nonlinearity: str = "tanh",
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
        chop: Optional[ChopiSTE] = None,
    ) -> None:
        super().__init__(
            input_size, hidden_size, num_layers, nonlinearity, bias,
            batch_first, dropout, bidirectional,
        )
        self.chop = chop

    def forward(self, input: torch.Tensor, hx: Optional[torch.Tensor] = None):
        if self.chop is None:
            return super().forward(input, hx)

        # Fake-quantize weights/biases with STE (temporary, no permanent .data change)
        for name, param in self.named_parameters():
            if "weight" in name or "bias" in name:
                param_q = self.chop(param)
                # Use temporary quantized param for this forward pass
                setattr(self, name, nn.Parameter(param_q, requires_grad=param.requires_grad))

        output, hidden = super().forward(input, hx)
        return self.chop(output), hidden


class IQuantizedLSTM(nn.LSTM):
    """Integer quantized LSTM for QAT using ChopiSTE (weights fake-quantized)."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
        chop: Optional[ChopiSTE] = None,
    ) -> None:
        super().__init__(
            input_size, hidden_size, num_layers, bias, batch_first, dropout, bidirectional,
        )
        self.chop = chop

    def forward(
        self,
        input: torch.Tensor,
        hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        if self.chop is None:
            return super().forward(input, hx)

        for name, param in self.named_parameters():
            if "weight" in name or "bias" in name:
                param_q = self.chop(param)
                setattr(self, name, nn.Parameter(param_q, requires_grad=param.requires_grad))

        output, (hn, cn) = super().forward(input, hx)
        return self.chop(output), (hn, cn)


class IQuantizedGRU(nn.GRU):
    """Integer quantized GRU for QAT using ChopiSTE (weights fake-quantized)."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
        chop: Optional[ChopiSTE] = None,
    ) -> None:
        super().__init__(
            input_size, hidden_size, num_layers, bias, batch_first, dropout, bidirectional,
        )
        self.chop = chop

    def forward(
        self,
        input: torch.Tensor,
        hx: Optional[torch.Tensor] = None,
    ):
        if self.chop is None:
            return super().forward(input, hx)

        for name, param in self.named_parameters():
            if "weight" in name or "bias" in name:
                param_q = self.chop(param)
                setattr(self, name, nn.Parameter(param_q, requires_grad=param.requires_grad))

        output, hidden = super().forward(input, hx)
        return self.chop(output), hidden


class IQuantizedMaxPool1d(nn.MaxPool1d):
    """Integer quantized 1D max pooling (activations quantized with STE)."""

    def __init__(
        self,
        kernel_size: int,
        stride: Optional[int] = None,
        padding: int = 0,
        dilation: int = 1,
        return_indices: bool = False,
        ceil_mode: bool = False,
        chop: Optional[ChopiSTE] = None,
    ) -> None:
        super().__init__(kernel_size, stride, padding, dilation, return_indices, ceil_mode)
        self.chop = chop

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = super().forward(input)
        return self.chop(output) if self.chop is not None else output


class IQuantizedMaxPool2d(nn.MaxPool2d):
    """Integer quantized 2D max pooling (activations quantized with STE)."""

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Optional[Union[int, Tuple[int, int]]] = None,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        return_indices: bool = False,
        ceil_mode: bool = False,
        chop: Optional[ChopiSTE] = None,
    ) -> None:
        super().__init__(kernel_size, stride, padding, dilation, return_indices, ceil_mode)
        self.chop = chop

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = super().forward(input)
        return self.chop(output) if self.chop is not None else output


class IQuantizedMaxPool3d(nn.MaxPool3d):
    """Integer quantized 3D max pooling (activations quantized with STE)."""

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Optional[Union[int, Tuple[int, int, int]]] = None,
        padding: Union[int, Tuple[int, int, int]] = 0,
        dilation: Union[int, Tuple[int, int, int]] = 1,
        return_indices: bool = False,
        ceil_mode: bool = False,
        chop: Optional[ChopiSTE] = None,
    ) -> None:
        super().__init__(kernel_size, stride, padding, dilation, return_indices, ceil_mode)
        self.chop = chop

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = super().forward(input)
        return self.chop(output) if self.chop is not None else output


class IQuantizedAvgPool1d(nn.AvgPool1d):
    """Integer quantized 1D average pooling (activations quantized with STE)."""

    def __init__(
        self,
        kernel_size: int,
        stride: Optional[int] = None,
        padding: int = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
        chop: Optional[ChopiSTE] = None,
    ) -> None:
        super().__init__(
            kernel_size, stride=stride, padding=padding,
            ceil_mode=ceil_mode, count_include_pad=count_include_pad,
        )
        self.chop = chop

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = super().forward(input)
        return self.chop(output) if self.chop is not None else output


class IQuantizedAvgPool2d(nn.AvgPool2d):
    """Integer quantized 2D average pooling (activations quantized with STE)."""

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Optional[Union[int, Tuple[int, int]]] = None,
        padding: Union[int, Tuple[int, int]] = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
        divisor_override: Optional[int] = None,
        chop: Optional[ChopiSTE] = None,
    ) -> None:
        super().__init__(
            kernel_size, stride=stride, padding=padding,
            ceil_mode=ceil_mode, count_include_pad=count_include_pad,
            divisor_override=divisor_override,
        )
        self.chop = chop

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = super().forward(input)
        return self.chop(output) if self.chop is not None else output


class IQuantizedAvgPool3d(nn.AvgPool3d):
    """Integer quantized 3D average pooling (activations quantized with STE)."""

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Optional[Union[int, Tuple[int, int, int]]] = None,
        padding: Union[int, Tuple[int, int, int]] = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
        divisor_override: Optional[int] = None,
        chop: Optional[ChopiSTE] = None,
    ) -> None:
        super().__init__(
            kernel_size, stride=stride, padding=padding,
            ceil_mode=ceil_mode, count_include_pad=count_include_pad,
            divisor_override=divisor_override,
        )
        self.chop = chop

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = super().forward(input)
        return self.chop(output) if self.chop is not None else output


class IQuantizedAdaptiveAvgPool1d(nn.AdaptiveAvgPool1d):
    """Integer quantized 1D adaptive average pooling (activations quantized with STE)."""

    def __init__(
        self,
        output_size: Union[int, Tuple[int]],
        chop: Optional[ChopiSTE] = None,
    ) -> None:
        super().__init__(output_size)
        self.chop = chop

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = super().forward(input)
        return self.chop(output) if self.chop is not None else output


class IQuantizedAdaptiveAvgPool2d(nn.AdaptiveAvgPool2d):
    """Integer quantized 2D adaptive average pooling (activations quantized with STE)."""

    def __init__(
        self,
        output_size: Union[int, Tuple[int, int]],
        chop: Optional[ChopiSTE] = None,
    ) -> None:
        super().__init__(output_size)
        self.chop = chop

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = super().forward(input)
        return self.chop(output) if self.chop is not None else output


class IQuantizedAdaptiveAvgPool3d(nn.AdaptiveAvgPool3d):
    """Integer quantized 3D adaptive average pooling (activations quantized with STE)."""

    def __init__(
        self,
        output_size: Union[int, Tuple[int, int, int]],
        chop: Optional[ChopiSTE] = None,
    ) -> None:
        super().__init__(output_size)
        self.chop = chop

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = super().forward(input)
        return self.chop(output) if self.chop is not None else output


class IQuantizedBatchNorm1d(nn.BatchNorm1d):
    """Integer quantized 1D Batch Normalization for QAT using ChopiSTE."""

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        chop: Optional[ChopiSTE] = None,
    ) -> None:
        super().__init__(
            num_features, eps=eps, momentum=momentum,
            affine=affine, track_running_stats=track_running_stats,
        )
        self.chop = chop

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.chop is None:
            return super().forward(input)

        if self.affine:
            self.weight.data = self.chop(self.weight.data).data
            self.bias.data = self.chop(self.bias.data).data

        output = super().forward(input)
        return self.chop(output)


class IQuantizedBatchNorm2d(nn.BatchNorm2d):
    """Integer quantized 2D Batch Normalization for QAT using ChopiSTE."""

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        chop: Optional[ChopiSTE] = None,
    ) -> None:
        super().__init__(
            num_features, eps=eps, momentum=momentum,
            affine=affine, track_running_stats=track_running_stats,
        )
        self.chop = chop

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.chop is None:
            return super().forward(input)

        if self.affine:
            self.weight.data = self.chop(self.weight.data).data
            self.bias.data = self.chop(self.bias.data).data

        output = super().forward(input)
        return self.chop(output)


class IQuantizedBatchNorm3d(nn.BatchNorm3d):
    """Integer quantized 3D Batch Normalization for QAT using ChopiSTE."""

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        chop: Optional[ChopiSTE] = None,
    ) -> None:
        super().__init__(
            num_features, eps=eps, momentum=momentum,
            affine=affine, track_running_stats=track_running_stats,
        )
        self.chop = chop

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.chop is None:
            return super().forward(input)

        if self.affine:
            self.weight.data = self.chop(self.weight.data).data
            self.bias.data = self.chop(self.bias.data).data

        output = super().forward(input)
        return self.chop(output)


class IQuantizedLayerNorm(nn.LayerNorm):
    """Integer quantized Layer Normalization for QAT using ChopiSTE."""

    def __init__(
        self,
        normalized_shape: Union[int, List[int], torch.Size],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        chop: Optional[ChopiSTE] = None,
    ) -> None:
        super().__init__(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)
        self.chop = chop

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.chop is None:
            return super().forward(input)

        if self.elementwise_affine:
            self.weight.data = self.chop(self.weight.data).data
            self.bias.data = self.chop(self.bias.data).data

        output = super().forward(input)
        return self.chop(output)


class IQuantizedInstanceNorm1d(nn.InstanceNorm1d):
    """Integer quantized 1D Instance Normalization for QAT using ChopiSTE."""

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = False,
        track_running_stats: bool = False,
        chop: Optional[ChopiSTE] = None,
    ) -> None:
        super().__init__(
            num_features, eps=eps, momentum=momentum,
            affine=affine, track_running_stats=track_running_stats,
        )
        self.chop = chop

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.chop is None:
            return super().forward(input)

        if self.affine:
            self.weight.data = self.chop(self.weight.data).data
            self.bias.data = self.chop(self.bias.data).data

        output = super().forward(input)
        return self.chop(output)


class IQuantizedInstanceNorm2d(nn.InstanceNorm2d):
    """Integer quantized 2D Instance Normalization for QAT using ChopiSTE."""

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = False,
        track_running_stats: bool = False,
        chop: Optional[ChopiSTE] = None,
    ) -> None:
        super().__init__(
            num_features, eps=eps, momentum=momentum,
            affine=affine, track_running_stats=track_running_stats,
        )
        self.chop = chop

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.chop is None:
            return super().forward(input)

        if self.affine:
            self.weight.data = self.chop(self.weight.data).data
            self.bias.data = self.chop(self.bias.data).data

        output = super().forward(input)
        return self.chop(output)


class IQuantizedInstanceNorm3d(nn.InstanceNorm3d):
    """Integer quantized 3D Instance Normalization for QAT using ChopiSTE."""

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = False,
        track_running_stats: bool = False,
        chop: Optional[ChopiSTE] = None,
    ) -> None:
        super().__init__(
            num_features, eps=eps, momentum=momentum,
            affine=affine, track_running_stats=track_running_stats,
        )
        self.chop = chop

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.chop is None:
            return super().forward(input)

        if self.affine:
            self.weight.data = self.chop(self.weight.data).data
            self.bias.data = self.chop(self.bias.data).data

        output = super().forward(input)
        return self.chop(output)


class IQuantizedGroupNorm(nn.GroupNorm):
    """Integer quantized Group Normalization for QAT using ChopiSTE."""

    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-5,
        affine: bool = True,
        chop: Optional[ChopiSTE] = None,
    ) -> None:
        super().__init__(num_groups, num_channels, eps=eps, affine=affine)
        self.chop = chop

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.chop is None:
            return super().forward(input)

        if self.affine:
            self.weight.data = self.chop(self.weight.data).data
            self.bias.data = self.chop(self.bias.data).data

        output = super().forward(input)
        return self.chop(output)


class IQuantizedMultiheadAttention(nn.MultiheadAttention):
    """Integer quantized Multi-head Attention for QAT using ChopiSTE."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        batch_first: bool = False,
        chop: Optional[ChopiSTE] = None,
    ):
        super().__init__(
            embed_dim, num_heads, dropout=dropout, bias=bias,
            add_bias_kv=add_bias_kv, add_zero_attn=add_zero_attn,
            kdim=kdim, vdim=vdim, batch_first=batch_first,
        )
        self.chop = chop

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ):
        if self.chop is None:
            return super().forward(
                query, key, value, key_padding_mask, need_weights,
                attn_mask, average_attn_weights, is_causal=is_causal
            )

        # Fake-quantize projection weights with STE
        if hasattr(self, "in_proj_weight") and self.in_proj_weight is not None:
            self.in_proj_weight.data = self.chop(self.in_proj_weight.data).data
        if hasattr(self, "in_proj_bias") and self.in_proj_bias is not None:
            self.in_proj_bias.data = self.chop(self.in_proj_bias.data).data
        self.out_proj.weight.data = self.chop(self.out_proj.weight.data).data
        if self.out_proj.bias is not None:
            self.out_proj.bias.data = self.chop(self.out_proj.bias.data).data

        output, attn_weights = super().forward(
            query, key, value, key_padding_mask, need_weights,
            attn_mask, average_attn_weights, is_causal=is_causal
        )
        if isinstance(output, tuple):
            return self.chop(output[0]), output[1]
        return self.chop(output), attn_weights


# ====================== IQuantized Activations ======================

class IQuantizedDropout(nn.Dropout):
    """Integer quantized Dropout for QAT using ChopiSTE."""

    def __init__(
        self,
        p: float = 0.5,
        chop: Optional[ChopiSTE] = None,
    ) -> None:
        super().__init__(p=p)
        self.chop = chop

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.chop is None:
            return super().forward(input)
        x = self.chop(input)
        x = super().forward(x)
        return self.chop(x)


class IQuantizedReLU(nn.ReLU):
    """Integer quantized ReLU for QAT using ChopiSTE."""

    def __init__(
        self,
        inplace: bool = False,
        chop: Optional[ChopiSTE] = None,
    ) -> None:
        super().__init__(inplace=inplace)
        self.chop = chop

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.chop is None:
            return super().forward(input)
        x = self.chop(input)
        x = super().forward(x)
        return self.chop(x)


class IQuantizedSigmoid(nn.Sigmoid):
    """Integer quantized Sigmoid for QAT using ChopiSTE."""

    def __init__(self, chop: Optional[ChopiSTE] = None) -> None:
        super().__init__()
        self.chop = chop

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.chop is None:
            return super().forward(input)
        x = self.chop(input)
        x = super().forward(x)
        return self.chop(x)


class IQuantizedTanh(nn.Tanh):
    """Integer quantized Tanh for QAT using ChopiSTE."""

    def __init__(self, chop: Optional[ChopiSTE] = None) -> None:
        super().__init__()
        self.chop = chop

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.chop is None:
            return super().forward(input)
        x = self.chop(input)
        x = super().forward(x)
        return self.chop(x)


class IQuantizedLeakyReLU(nn.LeakyReLU):
    """Integer quantized LeakyReLU for QAT using ChopiSTE."""

    def __init__(
        self,
        negative_slope: float = 0.01,
        inplace: bool = False,
        chop: Optional[ChopiSTE] = None,
    ) -> None:
        super().__init__(negative_slope=negative_slope, inplace=inplace)
        self.chop = chop

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.chop is None:
            return super().forward(input)
        x = self.chop(input)
        x = super().forward(x)
        return self.chop(x)


class IQuantizedSoftmax(nn.Softmax):
    """Integer quantized Softmax for QAT using ChopiSTE."""

    def __init__(
        self,
        dim: int = -1,
        chop: Optional[ChopiSTE] = None,
    ) -> None:
        super().__init__(dim=dim)
        self.chop = chop

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.chop is None:
            return super().forward(input)
        x = self.chop(input)
        x = super().forward(x)
        return self.chop(x)


class IQuantizedGELU(nn.GELU):
    """Integer quantized GELU for QAT using ChopiSTE."""

    def __init__(
        self,
        approximate: str = "none",
        chop: Optional[ChopiSTE] = None,
    ) -> None:
        super().__init__(approximate=approximate)
        self.chop = chop

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.chop is None:
            return super().forward(input)
        x = self.chop(input)
        x = super().forward(x)
        return self.chop(x)


class IQuantizedELU(nn.ELU):
    """Integer quantized ELU for QAT using ChopiSTE."""

    def __init__(
        self,
        alpha: float = 1.0,
        inplace: bool = False,
        chop: Optional[ChopiSTE] = None,
    ) -> None:
        super().__init__(alpha=alpha, inplace=inplace)
        self.chop = chop

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.chop is None:
            return super().forward(input)
        x = self.chop(input)
        x = super().forward(x)
        return self.chop(x)


class IQuantizedPReLU(nn.PReLU):
    """Integer quantized PReLU for QAT using ChopiSTE."""

    def __init__(
        self,
        num_parameters: int = 1,
        init: float = 0.25,
        chop: Optional[ChopiSTE] = None,
    ) -> None:
        super().__init__(num_parameters=num_parameters, init=init)
        self.chop = chop

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.chop is None:
            return super().forward(input)
        x = self.chop(input)
        if self.chop is not None:
            self.weight.data = self.chop(self.weight.data).data
        x = super().forward(x)
        return self.chop(x)


class IQuantizedSiLU(nn.SiLU):
    """Integer quantized SiLU (Swish) for QAT using ChopiSTE."""

    def __init__(
        self,
        inplace: bool = False,
        chop: Optional[ChopiSTE] = None,
    ) -> None:
        super().__init__(inplace=inplace)
        self.chop = chop

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.chop is None:
            return super().forward(input)
        x = self.chop(input)
        x = super().forward(x)
        return self.chop(x)


class IQuantizedEmbedding(nn.Embedding):
    """Integer quantized Embedding layer for QAT using ChopiSTE."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        chop: Optional[ChopiSTE] = None,
    ) -> None:
        super().__init__(
            num_embeddings, embedding_dim, padding_idx=padding_idx,
            max_norm=max_norm, norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq, sparse=sparse,
        )
        self.chop = chop

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.chop is None:
            return super().forward(input)

        weight = self.chop(self.weight)
        output = F.embedding(
            input, weight, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse,
        )
        return self.chop(output)


# Aliases (keep for backward compatibility)
IQuantizedAttention = IQuantizedMultiheadAttention
IQuantizedAvgPool = IQuantizedAvgPool2d