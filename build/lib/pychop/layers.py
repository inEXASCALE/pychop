import torch
# from .lightchop import LightChop
from .tch.fixed_point import FPRound
from .tch.integer import Chopi
import torch.nn as nn
from typing import Tuple, Union
import torch.nn.functional as F
from .tch.lightchop import LightChopSTE

def post_quantization(model, chop, eval_mode=True, verbose=False):
    """
    Perform post-training quantization on a copy of a PyTorch model for simulation purposes.
    
    Args:
        model (torch.nn.Module): The PyTorch model to quantize (remains unchanged).
        chop: Object with a quantize method (e.g., rounds to specific floating-point values).
        eval_mode (bool): If True, set the copied model to evaluation mode. Default: True.
        verbose (bool): If True, print parameter names and quantized data. Default: False.
    
    Returns:
        torch.nn.Module: A new quantized copy of the model.
    """
    import copy
    # Create a deep copy of the model to avoid modifying the original
    quantized_model = copy.deepcopy(model)
    
    # Set the copied model to evaluation mode if specified
    if eval_mode:
        quantized_model.eval()
    
    # Get the state dict of the copied model
    state_dict = quantized_model.state_dict()
    
    # Quantize each tensor in the state dict
    for key in state_dict.keys():
        original_tensor = state_dict[key]
        quantized_tensor = chop.quantize(original_tensor)
        
        # Ensure the quantized tensor matches the original's shape
        if quantized_tensor.shape != original_tensor.shape:
            raise ValueError(f"Shape mismatch for {key}: {original_tensor.shape} vs {quantized_tensor.shape}")
        
        state_dict[key] = quantized_tensor
    
    # Load the quantized state dict back into the copied model
    quantized_model.load_state_dict(state_dict)
    
    # Optional verbose output for debugging
    if verbose:
        print("Post-Quantization Results:")
        for name, param in quantized_model.named_parameters():
            print(f"{name}: {param.data}")
    
    return quantized_model



class QuantizedLayer(torch.nn.Module):
    """Quantize each element of neural networks"""
    def __init__(self, 
                 exp_bits: int,
                 sig_bits: int,
                 rmode: int = 1):
        
        super().__init__()
        self.quantizer = LightChopSTE(exp_bits, sig_bits, rmode)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.quantizer.quantize(x)


class IntQuantizedLayer(torch.nn.Module):
    """
    __init__(config)
        Apply ``pychop`` to quantization aware training, 
        One can feed [quant | chop | fixed_point] module as base for quantization.

    """

    def __init__(self, num_bits=8, symmetric=True, per_channel=False, channel_dim=0):
        super(IntQuantizedLayer, self).__init__()
        self.chopi = Chopi(num_bits=num_bits, symmetric=symmetric, per_channel=per_channel, channel_dim=channel_dim)
        
    def forward(self, x):
        return self.chopi.quantize(x).to(dtype=torch.float32)
        

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
        self.quantizer = FPRound(ibits, fbits)
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



# ===================================================================
# ===================================================================

# Quantized Convolutional Layers
class QuantizedConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, tuple], 
                 exp_bits: int, sig_bits: int, stride: int = 1, padding: int = 0, 
                 rmode: int = 1):
        super().__init__()
        self.quantizer = LightChopSTE(exp_bits, sig_bits, rmode)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)

        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q_weight = self.quantizer.quantize(self.conv.weight)
        q_bias = self.quantizer.quantize(self.conv.bias)
        q_input = self.quantizer.quantize(x)
        return F.conv1d(q_input, q_weight, q_bias, self.conv.stride, self.conv.padding)


class QuantizedConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int or tuple, 
                 exp_bits: int, sig_bits: int, stride: int or tuple = 1, 
                 padding: int or tuple = 0, rmode: int = 1):
        super().__init__()
        self.quantizer = LightChopSTE(exp_bits, sig_bits, rmode)
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


class QuantizedConv3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, tuple], 
                 exp_bits: int, sig_bits: int, stride: int = 1, padding: int = 0, 
                 rmode: int = 1):
        super().__init__()
        self.quantizer = LightChopSTE(exp_bits, sig_bits, rmode)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q_weight = self.quantizer.quantize(self.conv.weight)
        q_bias = self.quantizer.quantize(self.conv.bias)
        q_input = self.quantizer.quantize(x)
        return F.conv3d(q_input, q_weight, q_bias, self.conv.stride, self.conv.padding)

# Quantized Pooling Layers
class QuantizedMaxPool1d(nn.Module):
    def __init__(self, kernel_size: int, exp_bits: int, sig_bits: int, stride: int = None, 
                 rmode: int = 1):
        super().__init__()
        self.quantizer = LightChopSTE(exp_bits, sig_bits, rmode)
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q_input = self.quantizer.quantize(x)
        return F.max_pool1d(q_input, self.kernel_size, self.stride)

class QuantizedMaxPool2d(nn.Module):
    def __init__(self, kernel_size: int, exp_bits: int, sig_bits: int, stride: int = None, 
                 rmode: int = 1):
        super().__init__()
        self.quantizer = LightChopSTE(exp_bits, sig_bits, rmode)
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q_input = self.quantizer.quantize(x)
        return F.max_pool2d(q_input, self.kernel_size, self.stride)

class QuantizedMaxPool3d(nn.Module):
    def __init__(self, kernel_size: int, exp_bits: int, sig_bits: int, stride: int = None, 
                 rmode: int = 1):
        super().__init__()
        self.quantizer = LightChopSTE(exp_bits, sig_bits, rmode)
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q_input = self.quantizer.quantize(x)
        return F.max_pool3d(q_input, self.kernel_size, self.stride)

class QuantizedAvgPool(nn.Module):
    def __init__(self, kernel_size: int, exp_bits: int, sig_bits: int, stride: int = None, 
                 rmode: int = 1):
        super().__init__()
        self.quantizer = LightChopSTE(exp_bits, sig_bits, rmode)
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q_input = self.quantizer.quantize(x)
        return F.avg_pool2d(q_input, self.kernel_size, self.stride)

class QuantizedAvgPool1d(nn.Module):
    def __init__(self, kernel_size: int, exp_bits: int, sig_bits: int, stride: int = None, 
                 rmode: int = 1):
        super().__init__()
        self.quantizer = LightChopSTE(exp_bits, sig_bits, rmode)
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q_input = self.quantizer.quantize(x)
        return F.avg_pool1d(q_input, self.kernel_size, self.stride)

class QuantizedAvgPool2d(nn.Module):
    def __init__(self, kernel_size: int, exp_bits: int, sig_bits: int, stride: int = None, 
                 rmode: int = 1):
        super().__init__()
        self.quantizer = LightChopSTE(exp_bits, sig_bits, rmode)
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q_input = self.quantizer.quantize(x)
        return F.avg_pool2d(q_input, self.kernel_size, self.stride)

# Quantized Attention
class QuantizedAttention(nn.Module):
    def __init__(self, hidden_size: int, exp_bits: int, sig_bits: int, 
                 rmode: int = 1):
        super().__init__()
        self.quantizer = LightChopSTE(exp_bits, sig_bits, rmode)
        self.hidden_size = hidden_size
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.quantizer.quantize(self.query(x))
        k = self.quantizer.quantize(self.key(x))
        v = self.quantizer.quantize(self.value(x))
        scores = self.quantizer.quantize(torch.matmul(q, k.transpose(-2, -1)) / (self.hidden_size ** 0.5))
        attn = self.quantizer.quantize(F.softmax(scores, dim=-1))
        return torch.matmul(attn, v)

    

class QuantizedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, exp_bits: int, sig_bits: int, rmode: int = 1):
        super().__init__()
        self.quantizer = LightChopSTE(exp_bits, sig_bits, rmode)
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q_weight = self.weight # self.quantizer.quantize(self.weight)
        q_input = self.quantizer.quantize(x)
        output = torch.matmul(q_input, q_weight.t())
        q_bias = self.bias # self.quantizer.quantize(self.bias)
        return output + q_bias



class QuantizedRNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, exp_bits: int, sig_bits: int, 
                 num_layers: int = 1, bias: bool = True, nonlinearity: str = 'tanh', rmode: int = 1):
        super().__init__()
        self.quantizer = LightChopSTE(exp_bits, sig_bits, rmode)
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
        self.quantizer = LightChopSTE(exp_bits, sig_bits, rmode)
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


# Quantized Dropout
class QuantizedDropout(nn.Module):
    def __init__(self, p: float, exp_bits: int, sig_bits: int, 
                 rmode: int = 1):
        super().__init__()
        self.quantizer = LightChopSTE(exp_bits, sig_bits, rmode)
        self.dropout = nn.Dropout(p)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q_input = self.quantizer.quantize(x)
        return self.dropout(q_input)
    

class QuantizedReLU(nn.Module):
    """Quantized ReLU activation: applies quantization followed by ReLU."""
    def __init__(self, exp_bits: int, sig_bits: int, rmode: int = 1):
        super().__init__()
        self.quantizer = LightChopSTE(exp_bits, sig_bits, rmode)
        self.relu = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q_input = self.quantizer.quantize(x)
        return self.relu(q_input)


class QuantizedSigmoid(nn.Module):
    """Quantized Sigmoid activation: applies quantization followed by Sigmoid."""
    def __init__(self, exp_bits: int, sig_bits: int, rmode: int = 1):
        super().__init__()
        self.quantizer = LightChopSTE(exp_bits, sig_bits, rmode)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q_input = self.quantizer.quantize(x)
        return self.sigmoid(q_input)


class QuantizedTanh(nn.Module):
    """Quantized Tanh activation: applies quantization followed by Tanh."""
    def __init__(self, exp_bits: int, sig_bits: int, rmode: int = 1):
        super().__init__()
        self.quantizer = LightChopSTE(exp_bits, sig_bits, rmode)
        self.tanh = nn.Tanh()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q_input = self.quantizer.quantize(x)
        return self.tanh(q_input)


class QuantizedLeakyReLU(nn.Module):
    """Quantized LeakyReLU activation: applies quantization followed by LeakyReLU."""
    def __init__(self, exp_bits: int, sig_bits: int, rmode: int = 1, negative_slope: float = 0.01):
        super().__init__()
        self.quantizer = LightChopSTE(exp_bits, sig_bits, rmode)
        self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q_input = self.quantizer.quantize(x)
        return self.leaky_relu(q_input)


class QuantizedSoftmax(nn.Module):
    """Quantized Softmax activation: applies quantization followed by Softmax."""
    def __init__(self, exp_bits: int, sig_bits: int, rmode: int = 1, dim: int = -1):
        super().__init__()
        self.quantizer = LightChopSTE(exp_bits, sig_bits, rmode)
        self.softmax = nn.Softmax(dim=dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q_input = self.quantizer.quantize(x)
        return self.softmax(q_input)

class QuantizedGELU(nn.Module):
    """Quantized GELU activation: applies quantization followed by GELU."""
    def __init__(self, exp_bits: int, sig_bits: int, rmode: int = 1):
        super().__init__()
        self.quantizer = LightChopSTE(exp_bits, sig_bits, rmode)
        self.gelu = nn.GELU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q_input = self.quantizer.quantize(x)
        return self.gelu(q_input)

class QuantizedELU(nn.Module):
    """Quantized ELU activation: applies quantization followed by ELU."""
    def __init__(self, exp_bits: int, sig_bits: int, rmode: int = 1, alpha: float = 1.0):
        super().__init__()
        self.quantizer = LightChopSTE(exp_bits, sig_bits, rmode)
        self.elu = nn.ELU(alpha=alpha)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q_input = self.quantizer.quantize(x)
        return self.elu(q_input)

class QuantizedPReLU(nn.Module):
    """Quantized PReLU activation: applies quantization followed by PReLU."""
    def __init__(self, exp_bits: int, sig_bits: int, rmode: int = 1, num_parameters: int = 1, init: float = 0.25):
        super().__init__()
        self.quantizer = LightChopSTE(exp_bits, sig_bits, rmode)
        self.prelu = nn.PReLU(num_parameters=num_parameters, init=init)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q_input = self.quantizer.quantize(x)
        return self.prelu(q_input)
        

class QuantizedBatchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, exp_bits=5, sig_bits=10, rmode=1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.quantizer = LightChopSTE(exp_bits, sig_bits, rmode)

        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

    def forward(self, x):
        x_q = self.quantizer.quantize(x)

        # Handle input dimensions
        if x_q.dim() == 2:  # Shape: (N, C)
            dims = (0,)  # Mean over batch dimension
        elif x_q.dim() == 3:  # Shape: (N, C, L)
            dims = (0, 2)  # Mean over batch and length dimensions
        else:
            raise ValueError(f"Expected input tensor of 2 or 3 dimensions, got {x_q.dim()}")

        if self.training:
            # Compute batch statistics over appropriate dimensions
            batch_mean = x_q.mean(dim=dims)
            batch_var = x_q.var(dim=dims, unbiased=False)

            # Update running statistics
            with torch.no_grad():
                self.running_mean.mul_(1 - self.momentum).add_(batch_mean * self.momentum)
                self.running_var.mul_(1 - self.momentum).add_(batch_var * self.momentum)
                self.num_batches_tracked += 1

            mean = self.quantizer.quantize(batch_mean)
            var = self.quantizer.quantize(batch_var)
        else:
            mean = self.quantizer.quantize(self.running_mean)
            var = self.quantizer.quantize(self.running_var)

        # Quantize parameters
        weight_q = self.quantizer.quantize(self.weight)
        bias_q = self.quantizer.quantize(self.bias)

        # Normalize and scale with proper broadcasting
        if x_q.dim() == 2:
            x_normalized = (x_q - mean) / torch.sqrt(var + self.eps)
            x_q = weight_q * x_normalized + bias_q
        else:  # Shape: (N, C, L)
            x_normalized = (x_q - mean.view(1, -1, 1)) / torch.sqrt(var.view(1, -1, 1) + self.eps)
            x_q = weight_q.view(1, -1, 1) * x_normalized + bias_q.view(1, -1, 1)

        return self.quantizer.quantize(x_q)



class QuantizedBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, exp_bits=5, sig_bits=10, rmode=1):
        super().__init__()
        self.quantizer = LightChopSTE(exp_bits, sig_bits, rmode)
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


class QuantizedBatchNorm3d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, exp_bits=5, sig_bits=10, rmode=1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.quantizer = LightChopSTE(exp_bits, sig_bits, rmode)

        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

    def forward(self, x):
        x_q = self.quantizer.quantize(x)

        if self.training:
            batch_mean = x_q.mean(dim=(0, 2, 3, 4))
            batch_var = x_q.var(dim=(0, 2, 3, 4), unbiased=False)

            with torch.no_grad():
                self.running_mean.mul_(1 - self.momentum).add_(batch_mean * self.momentum)
                self.running_var.mul_(1 - self.momentum).add_(batch_var * self.momentum)
                self.num_batches_tracked += 1

            mean = self.quantizer.quantize(batch_mean)
            var = self.quantizer.quantize(batch_var)
        else:
            mean = self.quantizer.quantize(self.running_mean)
            var = self.quantizer.quantize(self.running_var)

        weight_q = self.quantizer.quantize(self.weight)
        bias_q = self.quantizer.quantize(self.bias)

        x_normalized = (x_q - mean.view(1, -1, 1, 1, 1)) / torch.sqrt(var.view(1, -1, 1, 1, 1) + self.eps)
        x_q = weight_q.view(1, -1, 1, 1, 1) * x_normalized + bias_q.view(1, -1, 1, 1, 1)

        return self.quantizer.quantize(x_q)
    

class QuantizedLayerNorm(nn.Module):
    def __init__(self, normalized_shape: int or tuple, exp_bits: int, sig_bits: int, 
                 eps: float = 1e-5, rmode: int = 1):
        super().__init__()
        self.quantizer = LightChopSTE(exp_bits, sig_bits, rmode)
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
        self.quantizer = LightChopSTE(exp_bits, sig_bits, rmode)
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
        self.quantizer = LightChopSTE(exp_bits, sig_bits, rmode)
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






# ===================================================================
# ===================================================================


# Quantized Linear Layer
class IntQuantizedLinear(nn.Module):
    def __init__(self, in_features, out_features, num_bits=8):
        super(IntQuantizedLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.quantizer_w = Chopi(num_bits=num_bits, symmetric=True)
        self.quantizer_x = Chopi(num_bits=num_bits, symmetric=True)
        self.quantizer_out = Chopi(num_bits=num_bits, symmetric=True)

    def forward(self, x):
        w = self.quantizer_w(self.linear.weight, training=self.training)
        b = self.quantizer_w(self.linear.bias, training=self.training) if self.linear.bias is not None else None
        x = self.quantizer_x(x, training=self.training)
        out = F.linear(x, w, b)
        return self.quantizer_out(out, training=self.training)

# Quantized Conv1d Layer
class IntQuantizedConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, num_bits=8):
        super(IntQuantizedConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.quantizer_w = Chopi(num_bits=num_bits, symmetric=True)
        self.quantizer_x = Chopi(num_bits=num_bits, symmetric=True)
        self.quantizer_out = Chopi(num_bits=num_bits, symmetric=True)

    def forward(self, x):
        w = self.quantizer_w(self.conv.weight, training=self.training)
        b = self.quantizer_w(self.conv.bias, training=self.training) if self.conv.bias is not None else None
        x = self.quantizer_x(x, training=self.training)
        out = self.conv._conv_forward(x, w, b)
        return self.quantizer_out(out, training=self.training)

# Quantized Conv2d Layer
class IntQuantizedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, num_bits=8):
        super(IntQuantizedConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.quantizer_w = Chopi(num_bits=num_bits, symmetric=True)
        self.quantizer_x = Chopi(num_bits=num_bits, symmetric=True)
        self.quantizer_out = Chopi(num_bits=num_bits, symmetric=True)

    def forward(self, x):
        w = self.quantizer_w(self.conv.weight, training=self.training)
        b = self.quantizer_w(self.conv.bias, training=self.training) if self.conv.bias is not None else None
        x = self.quantizer_x(x, training=self.training)
        out = self.conv._conv_forward(x, w, b)
        return self.quantizer_out(out, training=self.training)

# Quantized Conv3d Layer
class IntQuantizedConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, num_bits=8):
        super(IntQuantizedConv3d, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.quantizer_w = Chopi(num_bits=num_bits, symmetric=True)
        self.quantizer_x = Chopi(num_bits=num_bits, symmetric=True)
        self.quantizer_out = Chopi(num_bits=num_bits, symmetric=True)

    def forward(self, x):
        w = self.quantizer_w(self.conv.weight, training=self.training)
        b = self.quantizer_w(self.conv.bias, training=self.training) if self.conv.bias is not None else None
        x = self.quantizer_x(x, training=self.training)
        out = self.conv._conv_forward(x, w, b)
        return self.quantizer_out(out, training=self.training)

# Quantized LSTM Layer (Simplified, custom forward needed for full quantization)
class IntQuantizedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=False, num_bits=8):
        super(IntQuantizedLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=batch_first)
        self.quantizer_x = Chopi(num_bits=num_bits, symmetric=True)
        self.quantizer_h = Chopi(num_bits=num_bits, symmetric=True)
        self.quantizer_c = Chopi(num_bits=num_bits, symmetric=True)
        self.quantizer_w_ih = Chopi(num_bits=num_bits, symmetric=True)
        self.quantizer_w_hh = Chopi(num_bits=num_bits, symmetric=True)
        self.quantizer_out = Chopi(num_bits=num_bits, symmetric=True)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, x, hidden=None):
        x = self.quantizer_x(x, training=self.training)
        if hidden is None:
            batch_size = x.size(0)
            hidden = (torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device),
                      torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device))
        h, c = hidden
        h = self.quantizer_h(h, training=self.training)
        c = self.quantizer_c(c, training=self.training)
        # Note: nn.LSTM doesn't natively support quantized weights in forward
        # For simplicity, quantize weights but use standard forward (imperfect)
        w_ih = self.quantizer_w_ih(self.lstm.weight_ih_l0, training=self.training)
        w_hh = self.quantizer_w_hh(self.lstm.weight_hh_l0, training=self.training)
        b_ih = self.quantizer_w_ih(self.lstm.bias_ih_l0, training=self.training) if self.lstm.bias_ih_l0 is not None else None
        b_hh = self.quantizer_w_hh(self.lstm.bias_hh_l0, training=self.training) if self.lstm.bias_hh_l0 is not None else None
        out, (h_n, c_n) = self.lstm(x, (h, c))  # Weights not fully utilized here
        return self.quantizer_out(out, training=self.training), (h_n, c_n)

# Quantized GRU Layer (Simplified, custom forward needed)
class IntQuantizedGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=False, num_bits=8):
        super(IntQuantizedGRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=batch_first)
        self.quantizer_x = Chopi(num_bits=num_bits, symmetric=True)
        self.quantizer_h = Chopi(num_bits=num_bits, symmetric=True)
        self.quantizer_w_ih = Chopi(num_bits=num_bits, symmetric=True)
        self.quantizer_w_hh = Chopi(num_bits=num_bits, symmetric=True)
        self.quantizer_out = Chopi(num_bits=num_bits, symmetric=True)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, x, hidden=None):
        x = self.quantizer_x(x, training=self.training)
        if hidden is None:
            batch_size = x.size(0)
            hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        hidden = self.quantizer_h(hidden, training=self.training)
        # Quantize weights (imperfect without custom forward)
        w_ih = self.quantizer_w_ih(self.gru.weight_ih_l0, training=self.training)
        w_hh = self.quantizer_w_hh(self.gru.weight_hh_l0, training=self.training)
        b_ih = self.quantizer_w_ih(self.gru.bias_ih_l0, training=self.training) if self.gru.bias_ih_l0 is not None else None
        b_hh = self.quantizer_w_hh(self.gru.bias_hh_l0, training=self.training) if self.gru.bias_hh_l0 is not None else None
        out, h_n = self.gru(x, hidden)
        return self.quantizer_out(out, training=self.training), h_n

# Quantized Attention Layer (Simplified, custom forward needed)
class IntQuantizedAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, num_bits=8):
        super(IntQuantizedAttention, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.quantizer_q = Chopi(num_bits=num_bits, symmetric=True)
        self.quantizer_k = Chopi(num_bits=num_bits, symmetric=True)
        self.quantizer_v = Chopi(num_bits=num_bits, symmetric=True)
        self.quantizer_w = Chopi(num_bits=num_bits, symmetric=True)
        self.quantizer_out = Chopi(num_bits=num_bits, symmetric=True)

    def forward(self, query, key, value):
        q = self.quantizer_q(query, training=self.training)
        k = self.quantizer_k(key, training=self.training)
        v = self.quantizer_v(value, training=self.training)
        # Quantize projection weights (imperfect without custom forward)
        w_q = self.quantizer_w(self.attn.in_proj_weight[:self.attn.embed_dim], training=self.training)
        w_k = self.quantizer_w(self.attn.in_proj_weight[self.attn.embed_dim:2*self.attn.embed_dim], training=self.training)
        w_v = self.quantizer_w(self.attn.in_proj_weight[2*self.attn.embed_dim:], training=self.training)
        out, weights = self.attn(q, k, v)
        quantized_output = self.quantizer_out(out, training=self.training)
        return quantized_output, weights


# Quantized BatchNorm2d Layer
class IntQuantizedBatchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, num_bits=8):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.quantizer = Chopi(num_bits=num_bits, symmetric=True)

        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

    def forward(self, x):
        # Apply fake quantization during training, true quantization during inference
        x_q = self.quantizer(x, training=self.training)

        # Handle input dimensions
        if x_q.dim() == 2:  # Shape: (N, C)
            dims = (0,)  # Mean over batch dimension
        elif x_q.dim() == 3:  # Shape: (N, C, L)
            dims = (0, 2)  # Mean over batch and length dimensions
        else:
            raise ValueError(f"Expected input tensor of 2 or 3 dimensions, got {x_q.dim()}")

        if self.training:
            # Compute batch statistics on fake-quantized input (floating-point)
            batch_mean = x_q.mean(dim=dims)
            batch_var = x_q.var(dim=dims, unbiased=False)

            # Update running statistics with floating-point values
            with torch.no_grad():
                self.running_mean.mul_(1 - self.momentum).add_(batch_mean * self.momentum)
                self.running_var.mul_(1 - self.momentum).add_(batch_var * self.momentum)
                self.num_batches_tracked += 1

            mean = batch_mean
            var = batch_var
        else:
            mean = self.running_mean
            var = self.running_var

        # Quantize parameters using fake quantization during training
        weight_q = self.quantizer(self.weight, training=self.training)
        bias_q = self.quantizer(self.bias, training=self.training)

        # Normalize and scale with proper broadcasting
        if x_q.dim() == 2:
            x_normalized = (x_q - mean) / torch.sqrt(var + self.eps)
            x_out = weight_q * x_normalized + bias_q
        else:  # Shape: (N, C, L)
            x_normalized = (x_q - mean.view(1, -1, 1)) / torch.sqrt(var.view(1, -1, 1) + self.eps)
            x_out = weight_q.view(1, -1, 1) * x_normalized + bias_q.view(1, -1, 1)

        # Apply output quantization
        return self.quantizer(x_out, training=self.training)




class IntQuantizedBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, num_bits=8):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # Initialize the quantizer (assuming Chopi is your quantizer class)
        self.quantizer = Chopi(num_bits=num_bits, symmetric=True)

        # Learnable parameters
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        # Buffers for running statistics
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

    def forward(self, x):
        # Apply quantization to input
        # During training, this should return a fake-quantized float tensor
        x_q = self.quantizer(x, training=self.training)

        if self.training:
            # Compute batch statistics using the fake-quantized (float) input
            batch_mean = x_q.mean(dim=(0, 2, 3))
            batch_var = x_q.var(dim=(0, 2, 3), unbiased=False)

            # Update running statistics (in float)
            with torch.no_grad():
                self.running_mean.mul_(1 - self.momentum).add_(batch_mean * self.momentum)
                self.running_var.mul_(1 - self.momentum).add_(batch_var * self.momentum)
                self.num_batches_tracked += 1

            mean = batch_mean
            var = batch_var
        else:
            # Use running statistics during inference
            mean = self.running_mean
            var = self.running_var

        # Quantize parameters (weight and bias) with fake quantization during training
        weight_q = self.quantizer(self.weight, training=self.training)
        bias_q = self.quantizer(self.bias, training=self.training)

        # Perform batch normalization with proper broadcasting
        x_normalized = (x_q - mean.view(1, -1, 1, 1)) / torch.sqrt(var.view(1, -1, 1, 1) + self.eps)
        x_out = weight_q.view(1, -1, 1, 1) * x_normalized + bias_q.view(1, -1, 1, 1)

        # Apply output quantization
        return self.quantizer(x_out, training=self.training)

        

class IntQuantizedBatchNorm3d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, num_bits=8):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.quantizer = Chopi(num_bits=num_bits, symmetric=True)

        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

    def forward(self, x):
        # Apply quantization to input
        # During training, this should return a fake-quantized float tensor
        x_q = self.quantizer(x, training=self.training)

        if self.training:
            batch_mean = x_q.mean(dim=(0, 2, 3, 4))
            batch_var = x_q.var(dim=(0, 2, 3, 4), unbiased=False)

            with torch.no_grad():
                self.running_mean.mul_(1 - self.momentum).add_(batch_mean * self.momentum)
                self.running_var.mul_(1 - self.momentum).add_(batch_var * self.momentum)
                self.num_batches_tracked += 1

            mean = self.quantizer(batch_mean, training=self.training)
            var = self.quantizer(batch_var, training=self.training)
        else:
            mean = self.quantizer(self.running_mean, training=self.training)
            var = self.quantizer(self.running_var, training=self.training)

        weight_q = self.quantizer(self.weight, training=self.training)
        bias_q = self.quantizer(self.bias, training=self.training)

        x_normalized = (x_q - mean.view(1, -1, 1, 1, 1)) / torch.sqrt(var.view(1, -1, 1, 1, 1) + self.eps)
        x_q = weight_q.view(1, -1, 1, 1, 1) * x_normalized + bias_q.view(1, -1, 1, 1, 1)

        return self.quantizer(x_q, training=self.training)
    


# Quantized ReLU Layer
class IntQuantizedReLU(nn.Module):
    def __init__(self, num_bits=8):
        super(IntQuantizedReLU, self).__init__()
        self.relu = nn.ReLU()
        self.quantizer_x = Chopi(num_bits=num_bits, symmetric=True)
        self.quantizer_out = Chopi(num_bits=num_bits, symmetric=True)

    def forward(self, x):
        x = self.quantizer_x(x, training=self.training)
        out = self.relu(x)
        return self.quantizer_out(out, training=self.training)

# Quantized MaxPool2d Layer
class IntQuantizedMaxPool2d(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0, num_bits=8):
        super(IntQuantizedMaxPool2d, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size, stride=stride, padding=padding)
        self.quantizer_x = Chopi(num_bits=num_bits, symmetric=True)
        self.quantizer_out = Chopi(num_bits=num_bits, symmetric=True)

    def forward(self, x):
        x = self.quantizer_x(x, training=self.training)
        out = self.pool(x)
        return self.quantizer_out(out, training=self.training)

# Quantized AvgPool2d Layer
class IntQuantizedAvgPool2d(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0, num_bits=8):
        super(IntQuantizedAvgPool2d, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size, stride=stride, padding=padding)
        self.quantizer_x = Chopi(num_bits=num_bits, symmetric=True)
        self.quantizer_out = Chopi(num_bits=num_bits, symmetric=True)

    def forward(self, x):
        x = self.quantizer_x(x, training=self.training)
        out = self.pool(x)
        return self.quantizer_out(out, training=self.training)

# Quantized Dropout Layer
class IntQuantizedDropout(nn.Module):
    def __init__(self, p=0.5, num_bits=8):
        super(IntQuantizedDropout, self).__init__()
        self.dropout = nn.Dropout(p=p)
        self.quantizer_x = Chopi(num_bits=num_bits, symmetric=True)
        self.quantizer_out = Chopi(num_bits=num_bits, symmetric=True)

    def forward(self, x):
        x = self.quantizer_x(x, training=self.training)
        out = self.dropout(x)
        return self.quantizer_out(out, training=self.training)

# Quantized Flatten Layer
class IntQuantizedFlatten(nn.Module):
    def __init__(self, start_dim=1, end_dim=-1, num_bits=8):
        super(IntQuantizedFlatten, self).__init__()
        self.flatten = nn.Flatten(start_dim=start_dim, end_dim=end_dim)
        self.quantizer_x = Chopi(num_bits=num_bits, symmetric=True)
        self.quantizer_out = Chopi(num_bits=num_bits, symmetric=True)

    def forward(self, x):
        x = self.quantizer_x(x, training=self.training)
        out = self.flatten(x)
        return self.quantizer_out(out, training=self.training)
    






# ===================================================================
# ===================================================================

# ----- Fixed-point arithmetic

class FPQuantizedLinear(nn.Module):
    def __init__(self, in_features, out_features, ibits=8, fbits=8, rmode=1):
        super().__init__()
        self.quantizer = FPRound(ibits, fbits, rmode)
        self.linear = nn.Linear(in_features, out_features)
        self.initialize_weights()

    def initialize_weights(self):
        nn.init.kaiming_normal_(self.linear.weight, mode='fan_in', nonlinearity='linear')
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        q_x = self.quantizer.quantize(x)
        q_weight = self.quantizer.quantize(self.linear.weight)
        q_bias = self.quantizer.quantize(self.linear.bias)
        return F.linear(q_x, q_weight, q_bias)
        # return F.linear(q_x, self.linear.weight, self.linear.bias)


class FPQuantizedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=True, ibits=8, fbits=8, rmode=1):
        super().__init__()
        self.quantizer = FPRound(ibits, fbits, rmode)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=batch_first)
        self.initialize_weights()

    def initialize_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x, h0=None, c0=None):
        q_x = self.quantizer.quantize(x)
        if h0 is None:
            h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size, device=x.device)
        if c0 is None:
            c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size, device=x.device)
        q_h0 = self.quantizer.quantize(h0)
        q_c0 = self.quantizer.quantize(c0)
        for name, param in self.lstm.named_parameters():
            param.data = self.quantizer.quantize(param.data)
        return self.lstm(q_x, (q_h0, q_c0))


class FPQuantizedAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, ibits=8, fbits=8, rmode=1):
        super().__init__()
        self.quantizer = FPRound(ibits, fbits, rmode)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.initialize_weights()

    def initialize_weights(self):
        for name, param in self.attn.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, query, key, value, attn_mask=None):
        q_query = self.quantizer.quantize(query)
        q_key = self.quantizer.quantize(key)
        q_value = self.quantizer.quantize(value)
        for name, param in self.attn.named_parameters():
            param.data = self.quantizer.quantize(param.data)
        return self.attn(q_query, q_key, q_value, attn_mask=attn_mask)


class FPQuantizedGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=True, ibits=8, fbits=8, rmode=1):
        super().__init__()
        self.quantizer = FPRound(ibits, fbits, rmode)
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=batch_first)
        self.initialize_weights()

    def initialize_weights(self):
        for name, param in self.gru.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x, h0=None):
        q_x = self.quantizer.quantize(x)
        if h0 is None:
            h0 = torch.zeros(self.gru.num_layers, x.size(0), self.gru.hidden_size, device=x.device)
        q_h0 = self.quantizer.quantize(h0)
        for name, param in self.gru.named_parameters():
            param.data = self.quantizer.quantize(param.data)
        return self.gru(q_x, q_h0)


class FPQuantizedMaxPool2d(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0, ibits=8, fbits=8, rmode=1):
        super().__init__()
        self.quantizer = FPRound(ibits, fbits, rmode)
        self.pool = nn.MaxPool2d(kernel_size, stride, padding)

    def forward(self, x):
        q_x = self.quantizer.quantize(x)
        return self.pool(q_x)


class FPQuantizedConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, ibits=8, fbits=8, rmode=1):
        super().__init__()
        self.quantizer = FPRound(ibits, fbits, rmode)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.initialize_weights()

    def initialize_weights(self):
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_in', nonlinearity='relu')
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        q_x = self.quantizer.quantize(x)
        q_weight = self.quantizer.quantize(self.conv.weight)
        q_bias = self.quantizer.quantize(self.conv.bias) if self.conv.bias is not None else None
        return self.conv._conv_forward(q_x, q_weight, q_bias)


class FPQuantizedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, ibits=8, fbits=8, rmode=1):
        super().__init__()
        self.quantizer = FPRound(ibits, fbits, rmode)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.initialize_weights()

    def initialize_weights(self):
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_in', nonlinearity='relu')
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        q_x = self.quantizer.quantize(x)
        q_weight = self.quantizer.quantize(self.conv.weight)
        q_bias = self.quantizer.quantize(self.conv.bias) if self.conv.bias is not None else None
        return self.conv._conv_forward(q_x, q_weight, q_bias)


class FPQuantizedConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, ibits=8, fbits=8, rmode=1):
        super().__init__()
        self.quantizer = FPRound(ibits, fbits, rmode)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.initialize_weights()

    def initialize_weights(self):
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_in', nonlinearity='relu')
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        q_x = self.quantizer.quantize(x)
        q_weight = self.quantizer.quantize(self.conv.weight)
        q_bias = self.quantizer.quantize(self.conv.bias) if self.conv.bias is not None else None
        return self.conv._conv_forward(q_x, q_weight, q_bias)


class FPQuantizedBatchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, ibits=8, fbits=8, rmode=1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.quantizer = FPRound(ibits, fbits, rmode)

        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

    def forward(self, x):
        x_q = self.quantizer.quantize(x)

        # Handle input dimensions
        if x_q.dim() == 2:  # Shape: (N, C)
            dims = (0,)  # Mean over batch dimension
        elif x_q.dim() == 3:  # Shape: (N, C, L)
            dims = (0, 2)  # Mean over batch and length dimensions
        else:
            raise ValueError(f"Expected input tensor of 2 or 3 dimensions, got {x_q.dim()}")

        if self.training:
            # Compute batch statistics over appropriate dimensions
            batch_mean = x_q.mean(dim=dims)
            batch_var = x_q.var(dim=dims, unbiased=False)

            # Update running statistics
            with torch.no_grad():
                self.running_mean.mul_(1 - self.momentum).add_(batch_mean * self.momentum)
                self.running_var.mul_(1 - self.momentum).add_(batch_var * self.momentum)
                self.num_batches_tracked += 1

            mean = self.quantizer.quantize(batch_mean)
            var = self.quantizer.quantize(batch_var)
        else:
            mean = self.quantizer.quantize(self.running_mean)
            var = self.quantizer.quantize(self.running_var)

        # Quantize parameters
        weight_q = self.quantizer.quantize(self.weight)
        bias_q = self.quantizer.quantize(self.bias)

        # Normalize and scale with proper broadcasting
        if x_q.dim() == 2:
            x_normalized = (x_q - mean) / torch.sqrt(var + self.eps)
            x_q = weight_q * x_normalized + bias_q
        else:  # Shape: (N, C, L)
            x_normalized = (x_q - mean.view(1, -1, 1)) / torch.sqrt(var.view(1, -1, 1) + self.eps)
            x_q = weight_q.view(1, -1, 1) * x_normalized + bias_q.view(1, -1, 1)

        return self.quantizer.quantize(x_q)
    

class FPQuantizedBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, ibits=8, fbits=8, rmode=1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.quantizer = FPRound(ibits, fbits, rmode)

        # Learnable parameters
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        
        # Running statistics buffers
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

    def forward(self, x):
        # Quantize input
        x_q = self.quantizer.quantize(x)

        if self.training:
            # Compute batch statistics over N, H, W dimensions (N, C, H, W)
            batch_mean = x_q.mean(dim=(0, 2, 3))
            batch_var = x_q.var(dim=(0, 2, 3), unbiased=False)

            # Update running statistics
            with torch.no_grad():
                self.running_mean.mul_(1 - self.momentum).add_(batch_mean * self.momentum)
                self.running_var.mul_(1 - self.momentum).add_(batch_var * self.momentum)
                self.num_batches_tracked += 1

            # Quantize batch statistics
            mean = self.quantizer.quantize(batch_mean)
            var = self.quantizer.quantize(batch_var)
        else:
            # Use quantized running statistics during inference
            mean = self.quantizer.quantize(self.running_mean)
            var = self.quantizer.quantize(self.running_var)

        # Quantize parameters
        weight_q = self.quantizer.quantize(self.weight)
        bias_q = self.quantizer.quantize(self.bias)

        # Normalize and scale
        # Reshape mean and var to (1, C, 1, 1) for broadcasting
        x_normalized = (x_q - mean.view(1, -1, 1, 1)) / torch.sqrt(var.view(1, -1, 1, 1) + self.eps)
        x_q = weight_q.view(1, -1, 1, 1) * x_normalized + bias_q.view(1, -1, 1, 1)

        # Quantize output
        return self.quantizer.quantize(x_q)
    
    
#class FPQuantizedBatchNorm2d(nn.Module):
#    def __init__(self, num_features, ibits=8, fbits=8, rmode=1):
#        super().__init__()
#        self.quantizer = FPRound(ibits, fbits, rmode)
#        self.bn = nn.BatchNorm2d(num_features)
#        self.initialize_weights()
#
#    def initialize_weights(self):
#        nn.init.ones_(self.bn.weight)
#        nn.init.zeros_(self.bn.bias)
#
#    def forward(self, x):
#        q_x = self.quantizer.quantize(x)
#        q_weight = self.quantizer.quantize(self.bn.weight)
#        q_bias = self.quantizer.quantize(self.bn.bias)
#        return F.batch_norm(q_x, self.bn.running_mean, self.bn.running_var, 
#                            q_weight, q_bias, self.bn.training, self.bn.momentum, self.bn.eps)



class FPQuantizedBatchNorm3d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, ibits=8, fbits=8, rmode=1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.quantizer = FPRound(ibits, fbits, rmode)

        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

    def forward(self, x):
        x_q = self.quantizer.quantize(x)

        if self.training:
            # Compute batch statistics (N, C, D, H, W) -> mean/var over N, D, H, W
            batch_mean = x_q.mean(dim=(0, 2, 3, 4))
            batch_var = x_q.var(dim=(0, 2, 3, 4), unbiased=False)

            # Update running statistics
            with torch.no_grad():
                self.running_mean.mul_(1 - self.momentum).add_(batch_mean * self.momentum)
                self.running_var.mul_(1 - self.momentum).add_(batch_var * self.momentum)
                self.num_batches_tracked += 1

            mean = self.quantizer.quantize(batch_mean)
            var = self.quantizer.quantize(batch_var)
        else:
            mean = self.quantizer.quantize(self.running_mean)
            var = self.quantizer.quantize(self.running_var)

        # Quantize parameters
        weight_q = self.quantizer.quantize(self.weight)
        bias_q = self.quantizer.quantize(self.bias)

        # Normalize and scale
        x_normalized = (x_q - mean.view(1, -1, 1, 1, 1)) / torch.sqrt(var.view(1, -1, 1, 1, 1) + self.eps)
        x_q = weight_q.view(1, -1, 1, 1, 1) * x_normalized + bias_q.view(1, -1, 1, 1, 1)

        # Quantize output
        return self.quantizer.quantize(x_q)
    
    
class FPQuantizedAvgPool2d(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0, ibits=8, fbits=8, rmode=1):
        super().__init__()
        self.quantizer = FPRound(ibits, fbits, rmode)
        self.pool = nn.AvgPool2d(kernel_size, stride, padding)

    def forward(self, x):
        q_x = self.quantizer.quantize(x)
        return self.pool(q_x)


class FPQuantizedDropout(nn.Module):
    def __init__(self, p=0.5, ibits=8, fbits=8, rmode=1):
        super().__init__()
        self.quantizer = FPRound(ibits, fbits, rmode)
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        q_x = self.quantizer.quantize(x)
        return self.dropout(q_x)











