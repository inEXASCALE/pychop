import torch
# from .lightchop import LightChop
from .fixed_point import Chopf
from .tch.integer import Chopi

import torch.nn as nn
from typing import Tuple

class QuantizedLayer(torch.nn.Module):
    """Example of a quantized linear layer"""
    def __init__(self, 
                 exp_bits: int,
                 sig_bits: int,
                 rmode: int = 1):
        
        super().__init__()
        self.quantizer = FPRound(exp_bits, sig_bits, rmode)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.quantizer.quantize(x)


class IntQuantizedLayer(torch.nn.Module):
    """
    __init__(config)
        Apply ``pychop`` to quantization aware training, 
        One can feed [quant | chop | fixed_point] module as base for quantization.

    """

    def __init__(self, num_bits=8, symmetric=False, per_channel=False, channel_dim=0):
        super(IntQuantizedLayer, self).__init__()
        self.chopi = Chopi(num_bits=num_bits, symmetric=symmetric, per_channel=per_channel, channel_dim=channel_dim)
        
    def forward(self, x):
        return self.chopi(x)
        

class FQuantizedLayer(nn.Module):
    def __init__(self, 
                 in_dim: int, 
                 out_dim: int,
                 ibits: int,
                 fbits: int,
                 rmode: int = 1,
                 bias: bool = True):
        """
        A linear layer with fixed-point quantization for weights, bias, and inputs.
            
        Parameters
        ----------
        in_dim : int
            Number of input features
        
        out_dim : int
            Number of output features
        
        ibits : int
            Number of integer bits (including sign) for Qm.n format
        
        fbits : int
            Number of fractional bits for Qm.n format
        
        rmode : int
            Rounding mode to use when quantizing the significand. Options are:
            - 0: Round to nearest value, ties to odd.
            - 1: Round to nearest value, ties to even (IEEE 754 default).
            - 2: Round towards plus infinity (round up).
            - 3: Round towards minus infinity (round down).
            - 4: Truncate toward zero (no rounding up).
            - 5: Stochastic rounding proportional to the fractional part.
            - 6: Stochastic rounding with 50% probability.
            - 7: Round to nearest value, ties to zero.
            - 8: Round to nearest value, ties to away.

        bias : int
            Whether to include a bias term
        """
        super(FQuantizedLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.quantizer = Chopf(ibits, fbits)
        self.rmode = rmode

        # Initialize weights and bias as floating-point parameters
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_dim))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with fixed-point quantization.
        
        Parameters
        ----------
        x : numpy.ndarray | jax.Array | torch.Tensor,
            The input tensor (batch_size, in_dim)

        Returns
        ----------
        Output: numpy.ndarray | jax.Array | torch.Tensor,
            The input tensor (batch_size, out_dim)
        """
        
        return self.quantizer.quantize(x, self.rmode)




# Assume FPRound is defined elsewhere with the corrected _quantize_components
class QuantizedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, exp_bits: int, sig_bits: int, rmode: int = 1):
        super().__init__()
        self.quantizer = FPRound(exp_bits, sig_bits, rmode)
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q_weight = self.quantizer.quantize(self.weight)
        q_input = self.quantizer.quantize(x)
        output = torch.matmul(q_input, q_weight.t())
        q_bias = self.quantizer.quantize(self.bias)
        return output + q_bias



class QuantizedConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int or tuple, 
                 exp_bits: int, sig_bits: int, stride: int or tuple = 1, 
                 padding: int or tuple = 0, rmode: int = 1):
        super().__init__()
        self.quantizer = FPRound(exp_bits, sig_bits, rmode)
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, *([kernel_size] * 2 if isinstance(kernel_size, int) else kernel_size)))
        self.bias = nn.Parameter(torch.randn(out_channels))
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q_weight = self.quantizer.quantize(self.weight)
        q_input = self.quantizer.quantize(x)
        output = nn.functional.conv2d(q_input, q_weight, stride=self.stride, padding=self.padding)
        q_bias = self.quantizer.quantize(self.bias)
        return output + q_bias.view(1, -1, 1, 1)

class QuantizedRNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, exp_bits: int, sig_bits: int, 
                 num_layers: int = 1, bias: bool = True, nonlinearity: str = 'tanh', rmode: int = 1):
        super().__init__()
        self.quantizer = FPRound(exp_bits, sig_bits, rmode)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.nonlinearity = nonlinearity
        
        self.weight_ih = nn.ParameterList([nn.Parameter(torch.randn(hidden_size, input_size if i == 0 else hidden_size)) for i in range(num_layers)])
        self.weight_hh = nn.ParameterList([nn.Parameter(torch.randn(hidden_size, hidden_size)) for i in range(num_layers)])
        if bias:
            self.bias_ih = nn.ParameterList([nn.Parameter(torch.randn(hidden_size)) for i in range(num_layers)])
            self.bias_hh = nn.ParameterList([nn.Parameter(torch.randn(hidden_size)) for i in range(num_layers)])
        else:
            self.bias_ih = self.bias_hh = None
        
    def forward(self, x: torch.Tensor, h0: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.size()
        if h0 is None:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        
        q_input = self.quantizer.quantize(x)
        h = h0.clone()
        outputs = []
        
        for t in range(seq_len):
            x_t = q_input[:, t, :]
            new_h = h.clone()
            for layer in range(self.num_layers):
                w_ih = self.quantizer.quantize(self.weight_ih[layer])
                w_hh = self.quantizer.quantize(self.weight_hh[layer])
                h_prev = h[layer]
                
                if self.bias:
                    b_ih = self.quantizer.quantize(self.bias_ih[layer])
                    b_hh = self.quantizer.quantize(self.bias_hh[layer])
                    linear_input = torch.matmul(x_t, w_ih.t()) + b_ih + torch.matmul(h_prev, w_hh.t()) + b_hh
                else:
                    linear_input = torch.matmul(x_t, w_ih.t()) + torch.matmul(h_prev, w_hh.t())
                
                if self.nonlinearity == 'tanh':
                    new_h[layer] = torch.tanh(linear_input)
                elif self.nonlinearity == 'relu':
                    new_h[layer] = torch.relu(linear_input)
                x_t = new_h[layer]
            h = new_h
            outputs.append(x_t.unsqueeze(1))
        
        output = torch.cat(outputs, dim=1)
        return output, h


class QuantizedLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, exp_bits: int, sig_bits: int, 
                 num_layers: int = 1, bias: bool = True, rmode: int = 1):
        super().__init__()
        self.quantizer = FPRound(exp_bits, sig_bits, rmode)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        
        self.weight_ih = nn.ParameterList([nn.Parameter(torch.randn(4 * hidden_size, input_size if i == 0 else hidden_size)) for i in range(num_layers)])
        self.weight_hh = nn.ParameterList([nn.Parameter(torch.randn(4 * hidden_size, hidden_size)) for i in range(num_layers)])
        if bias:
            self.bias_ih = nn.ParameterList([nn.Parameter(torch.randn(4 * hidden_size)) for i in range(num_layers)])
            self.bias_hh = nn.ParameterList([nn.Parameter(torch.randn(4 * hidden_size)) for i in range(num_layers)])
        else:
            self.bias_ih = self.bias_hh = None
        
    def forward(self, x: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        batch_size, seq_len, _ = x.size()
        if hidden is None:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        else:
            h0, c0 = hidden
        
        q_input = self.quantizer.quantize(x)
        h, c = h0.clone(), c0.clone()
        outputs = []
        
        for t in range(seq_len):
            x_t = q_input[:, t, :]
            new_h, new_c = h.clone(), c.clone()
            for layer in range(self.num_layers):
                w_ih = self.quantizer.quantize(self.weight_ih[layer])
                w_hh = self.quantizer.quantize(self.weight_hh[layer])
                h_prev, c_prev = h[layer], c[layer]
                
                if self.bias:
                    b_ih = self.quantizer.quantize(self.bias_ih[layer])
                    b_hh = self.quantizer.quantize(self.bias_hh[layer])
                    gates = torch.matmul(x_t, w_ih.t()) + b_ih + torch.matmul(h_prev, w_hh.t()) + b_hh
                else:
                    gates = torch.matmul(x_t, w_ih.t()) + torch.matmul(h_prev, w_hh.t())
                
                i, f, g, o = gates.chunk(4, dim=1)
                i = torch.sigmoid(i)
                f = torch.sigmoid(f)
                g = torch.tanh(g)
                o = torch.sigmoid(o)
                
                new_c[layer] = f * c_prev + i * g
                new_h[layer] = o * torch.tanh(new_c[layer])
                x_t = new_h[layer]
            h, c = new_h, new_c
            outputs.append(x_t.unsqueeze(1))
        
        output = torch.cat(outputs, dim=1)
        return output, (h, c)


class QuantizedBatchNorm2d(nn.Module):
    def __init__(self, num_features: int, exp_bits: int, sig_bits: int, 
                 eps: float = 1e-5, momentum: float = 0.1, rmode: int = 1):
        super().__init__()
        self.quantizer = FPRound(exp_bits, sig_bits, rmode)
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            batch_mean = x.mean([0, 2, 3])
            batch_var = x.var([0, 2, 3], unbiased=False)
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
                self.num_batches_tracked += 1
        else:
            batch_mean = self.running_mean
            batch_var = self.running_var
        
        q_input = self.quantizer.quantize(x)
        q_weight = self.quantizer.quantize(self.weight)
        q_bias = self.quantizer.quantize(self.bias)
        q_mean = self.quantizer.quantize(batch_mean)
        q_var = self.quantizer.quantize(batch_var)
        
        normalized = (q_input - q_mean.view(1, -1, 1, 1)) / torch.sqrt(q_var.view(1, -1, 1, 1) + self.eps)
        return q_weight.view(1, -1, 1, 1) * normalized + q_bias.view(1, -1, 1, 1)


class QuantizedLayerNorm(nn.Module):
    def __init__(self, normalized_shape: int or tuple, exp_bits: int, sig_bits: int, 
                 eps: float = 1e-5, rmode: int = 1):
        super().__init__()
        self.quantizer = FPRound(exp_bits, sig_bits, rmode)
        self.normalized_shape = normalized_shape if isinstance(normalized_shape, tuple) else (normalized_shape,)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(self.normalized_shape))
        self.bias = nn.Parameter(torch.zeros(self.normalized_shape))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        
        q_input = self.quantizer.quantize(x)
        q_weight = self.quantizer.quantize(self.weight)
        q_bias = self.quantizer.quantize(self.bias)
        q_mean = self.quantizer.quantize(mean)
        q_var = self.quantizer.quantize(var)
        
        normalized = (q_input - q_mean) / torch.sqrt(q_var + self.eps)
        return q_weight * normalized + q_bias


class QuantizedGRU(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, exp_bits: int, sig_bits: int, 
                 num_layers: int = 1, bias: bool = True, rmode: int = 1):
        super().__init__()
        self.quantizer = FPRound(exp_bits, sig_bits, rmode)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        
        self.weight_ih = nn.ParameterList([nn.Parameter(torch.randn(3 * hidden_size, input_size if i == 0 else hidden_size)) for i in range(num_layers)])
        self.weight_hh = nn.ParameterList([nn.Parameter(torch.randn(3 * hidden_size, hidden_size)) for i in range(num_layers)])
        if bias:
            self.bias_ih = nn.ParameterList([nn.Parameter(torch.randn(3 * hidden_size)) for i in range(num_layers)])
            self.bias_hh = nn.ParameterList([nn.Parameter(torch.randn(3 * hidden_size)) for i in range(num_layers)])
        else:
            self.bias_ih = self.bias_hh = None
        
    def forward(self, x: torch.Tensor, h0: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.size()
        if h0 is None:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        
        q_input = self.quantizer.quantize(x)
        h = h0.clone()
        outputs = []
        
        for t in range(seq_len):
            x_t = q_input[:, t, :]
            new_h = h.clone()
            for layer in range(self.num_layers):
                w_ih = self.quantizer.quantize(self.weight_ih[layer])
                w_hh = self.quantizer.quantize(self.weight_hh[layer])
                h_prev = h[layer]
                
                if self.bias:
                    b_ih = self.quantizer.quantize(self.bias_ih[layer])
                    b_hh = self.quantizer.quantize(self.bias_hh[layer])
                    gates = torch.matmul(x_t, w_ih.t()) + b_ih + torch.matmul(h_prev, w_hh.t()) + b_hh
                else:
                    gates = torch.matmul(x_t, w_ih.t()) + torch.matmul(h_prev, w_hh.t())
                
                r, z, n = gates.chunk(3, dim=1)
                r = torch.sigmoid(r)
                z = torch.sigmoid(z)
                n = torch.tanh(n)
                
                new_h[layer] = (1 - z) * n + z * h_prev
                x_t = new_h[layer]
            h = new_h
            outputs.append(x_t.unsqueeze(1))
        
        output = torch.cat(outputs, dim=1)
        return output, h

class QuantizedMultiheadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, exp_bits: int, sig_bits: int, 
                 dropout: float = 0.0, rmode: int = 1):
        super().__init__()
        self.quantizer = FPRound(exp_bits, sig_bits, rmode)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.q_proj = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.k_proj = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.v_proj = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.out_proj = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                attn_mask: torch.Tensor = None, key_padding_mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len_q, _ = query.size()
        seq_len_k = key.size(1)
        
        q = self.quantizer.quantize(torch.matmul(self.quantizer.quantize(query), self.quantizer.quantize(self.q_proj)))
        k = self.quantizer.quantize(torch.matmul(self.quantizer.quantize(key), self.quantizer.quantize(self.k_proj)))
        v = self.quantizer.quantize(torch.matmul(self.quantizer.quantize(value), self.quantizer.quantize(self.v_proj)))
        
        q = q.view(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask
        if key_padding_mask is not None:
            attn_scores = attn_scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, v)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.embed_dim)
        output = self.quantizer.quantize(torch.matmul(attn_output, self.quantizer.quantize(self.out_proj)))
        
        return output, attn_weights




class FPRound:
    def __init__(self, exp_bits: int, sig_bits: int, rmode: int = 1):
        """Initialize float precision simulator with custom format and rounding mode."""
        self.exp_bits = exp_bits
        self.sig_bits = sig_bits
        self.rmode = rmode
        self.max_exp = 2 ** (exp_bits - 1) - 1
        self.min_exp = -self.max_exp + 1
        self.bias = 2 ** (exp_bits - 1) - 1  # Bias for IEEE 754-like format


    def _to_custom_float(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, 
                                                        torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert to custom float representation with proper IEEE 754 handling."""
        sign = torch.sign(x)
        abs_x = torch.abs(x)
        
        zero_mask = (abs_x == 0)
        inf_mask = torch.isinf(x)
        nan_mask = torch.isnan(x)
        
        exponent = torch.floor(torch.log2(abs_x.clamp(min=2.0**-24)))  # Minimum denormal
        significand = abs_x / (2.0 ** exponent)
        
        subnormal_mask = (exponent < self.min_exp)
        significand = torch.where(subnormal_mask, abs_x / (2.0 ** self.min_exp), significand)
        exponent = torch.where(subnormal_mask, self.min_exp, exponent)
        
        return sign, exponent + self.bias, significand, zero_mask, inf_mask, nan_mask
    
    def _quantize_components(self, 
                           x: torch.Tensor,
                           sign: torch.Tensor, 
                           exponent: torch.Tensor, 
                           significand: torch.Tensor,
                           zero_mask: torch.Tensor,
                           inf_mask: torch.Tensor,
                           nan_mask: torch.Tensor) -> torch.Tensor:
        """Quantize components according to IEEE 754 FP16 rules with specified rounding mode."""

        exp_min = 0  
        exp_max = 2 ** self.exp_bits - 1
        exponent = exponent.clamp(min=exp_min, max=exp_max)
        
        significand_steps = 2 ** self.sig_bits
        normal_mask = (exponent > 0) & (exponent < exp_max)
        subnormal_mask = (exponent == 0)
        significand_normal = significand - 1.0  
        
        if self.rmode == 1:
            significand_q = torch.round(significand_normal * significand_steps) / significand_steps
            significand_q = torch.where(subnormal_mask, 
                                   torch.round(significand * significand_steps) / significand_steps, 
                                   significand_q)
            
        elif self.rmode == 2:
            significand_q = torch.where(sign > 0, 
                                   torch.ceil(significand_normal * significand_steps),
                                   torch.floor(significand_normal * significand_steps)) / significand_steps
            significand_q = torch.where(subnormal_mask, 
                                   torch.where(sign > 0, 
                                             torch.ceil(significand * significand_steps), 
                                             torch.floor(significand * significand_steps)) / significand_steps, 
                                   significand_q)
            
        elif self.rmode == 3:
            significand_q = torch.where(sign > 0,
                                   torch.floor(significand_normal * significand_steps),
                                   torch.ceil(significand_normal * significand_steps)) / significand_steps
            significand_q = torch.where(subnormal_mask, 
                                   torch.where(sign > 0, 
                                             torch.floor(significand * significand_steps), 
                                             torch.ceil(significand * significand_steps)) / significand_steps, 
                                   significand_q)
            
        elif self.rmode == 4:
            significand_q = torch.floor(significand_normal * significand_steps) / significand_steps
            significand_q = torch.where(subnormal_mask, 
                                   torch.floor(significand * significand_steps) / significand_steps, 
                                   significand_q)
            
        elif self.rmode == 5:
            significand_scaled = significand_normal * significand_steps
            floor_val = torch.floor(significand_scaled)
            fraction = significand_scaled - floor_val
            prob = torch.rand_like(significand_scaled)
            significand_q = torch.where(prob < fraction, floor_val + 1, floor_val) / significand_steps
            significand_q = torch.where(subnormal_mask, 
                                   torch.where(torch.rand_like(significand) < (significand * significand_steps - torch.floor(significand * significand_steps)), 
                                             torch.ceil(significand * significand_steps), 
                                             torch.floor(significand * significand_steps)) / significand_steps, 
                                   significand_q)
            
        elif self.rmode == 6:
            significand_scaled = significand_normal * significand_steps
            floor_val = torch.floor(significand_scaled)
            prob = torch.rand_like(significand_scaled)
            significand_q = torch.where(prob < 0.5, floor_val, floor_val + 1) / significand_steps
            significand_q = torch.where(subnormal_mask, 
                                   torch.where(torch.rand_like(significand) < 0.5, 
                                             torch.floor(significand * significand_steps), 
                                             torch.ceil(significand * significand_steps)) / significand_steps, 
                                   significand_q)
            
        elif self.rmode == 7:
            significand_scaled = significand_normal * significand_steps
            floor_val = torch.floor(significand_scaled)
            ceil_val = torch.ceil(significand_scaled)
            is_half = torch.abs(significand_scaled - floor_val - 0.5) < 1e-6  # Robust tie check
            significand_q = torch.where(
                is_half,
                torch.where(sign >= 0, floor_val, ceil_val),  # Toward zero: positive floor, negative ceil
                torch.round(significand_scaled)
            ) / significand_steps
            significand_subnormal = significand * significand_steps
            sub_floor = torch.floor(significand_subnormal)
            sub_ceil = torch.ceil(significand_subnormal)
            sub_is_half = torch.abs(significand_subnormal - sub_floor - 0.5) < 1e-6
            significand_q = torch.where(
                subnormal_mask,
                torch.where(
                    sub_is_half,
                    torch.where(sign >= 0, sub_floor, sub_ceil),
                    torch.round(significand_subnormal)
                ) / significand_steps,
                significand_q
            )
            
        elif self.rmode == 8:
            significand_scaled = significand_normal * significand_steps
            floor_val = torch.floor(significand_scaled)
            ceil_val = torch.ceil(significand_scaled)
            is_half = torch.abs(significand_scaled - floor_val - 0.5) < 1e-6  # Robust tie check
            significand_q = torch.where(
                is_half,
                torch.where(sign >= 0, ceil_val, floor_val),  # Away from zero: positive ceil, negative floor
                torch.round(significand_scaled)
            ) / significand_steps
            significand_subnormal = significand * significand_steps
            sub_floor = torch.floor(significand_subnormal)
            sub_ceil = torch.ceil(significand_subnormal)
            sub_is_half = torch.abs(significand_subnormal - sub_floor - 0.5) < 1e-6
            significand_q = torch.where(
                subnormal_mask,
                torch.where(
                    sub_is_half,
                    torch.where(sign >= 0, sub_ceil, sub_floor),
                    torch.round(significand_subnormal)
                ) / significand_steps,
                significand_q
            )

        else:
            raise ValueError(f"Unsupported rounding mode: {self.rmode}")
        
        normal_result = sign * (1.0 + significand_q) * (2.0 ** (exponent - self.bias))
        subnormal_result = sign * significand_q * (2.0 ** self.min_exp)
        special_result = torch.where(inf_mask, torch.sign(x) * float('inf'), 
                                   torch.where(nan_mask, float('nan'), 0.0))
        
        result = torch.where(normal_mask, normal_result, 
                           torch.where(subnormal_mask, subnormal_result, 
                                     torch.where(zero_mask, 0.0, special_result)))
        
        return result

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize tensor to specified precision using the initialized rounding mode."""
        sign, exponent, significand, zero_mask, inf_mask, nan_mask = self._to_custom_float(x)
        return self._quantize_components(x, sign, exponent, significand, zero_mask, inf_mask, nan_mask)
