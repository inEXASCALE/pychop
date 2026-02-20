import sys
# appending a path
sys.path.append('../')

import torch
import torch.nn as nn
import torch.nn.functional as F
from pychop.layers import *

def test_layer(quantized_class, normal_class, input_tensor, target_tensor=None, requires_grad=True, **kwargs):
    # Extract initialization arguments
    init_args = kwargs.get('init_args', {})
    
    # Create arguments for normal layer by excluding 'num_bits'
    normal_init_args = {k: v for k, v in init_args.items() if k != 'num_bits'}
    
    # For quantized layer, add 'num_bits' if not present and if it's an IntQuantized layer
    quantized_init_args = init_args.copy()
    if quantized_class.__name__.startswith('IntQuantized') and 'num_bits' not in quantized_init_args:
        quantized_init_args['num_bits'] = 8  # Default to 8 bits
    
    # Instantiate both quantized and normal layers
    quantized_layer = quantized_class(**quantized_init_args).to(device)
    normal_layer = normal_class(**normal_init_args).to(device)
    
    # Sync weights between normal and quantized layers (only for trainable parameters)
    with torch.no_grad():
        for q_param, n_param in zip(quantized_layer.parameters(), normal_layer.parameters()):
            if q_param.requires_grad:
                n_param.copy_(q_param)

    # Forward pass for both layers
    quantized_layer.train()
    normal_layer.train()
    input_tensor.requires_grad_(requires_grad)
    
    # Handle layers with multiple inputs (e.g., Attention)
    if 'key' in kwargs and 'value' in kwargs:
        q_output = quantized_layer(input_tensor, key=kwargs['key'], value=kwargs['value'])
        n_output = normal_layer(input_tensor, key=kwargs['key'], value=kwargs['value'])
    elif 'h0' in kwargs and 'c0' in kwargs:
        q_output = quantized_layer(input_tensor, h0=kwargs['h0'], c0=kwargs['c0'])
        n_output = normal_layer(input_tensor, h0=kwargs['h0'], c0=kwargs['c0'])
    elif 'h0' in kwargs:
        q_output = quantized_layer(input_tensor, h0=kwargs['h0'])
        n_output = normal_layer(input_tensor, h0=kwargs['h0'])
    else:
        q_output = quantized_layer(input_tensor)
        n_output = normal_layer(input_tensor)
    
    # Handle tuple outputs
    if isinstance(q_output, tuple):
        q_primary_output = q_output[0]
        n_primary_output = n_output[0]
    else:
        q_primary_output = q_output
        n_primary_output = n_output
    
    # Forward pass checks
    assert q_primary_output is not None, "Quantized forward pass returned None"
    assert q_primary_output.shape[0] == input_tensor.shape[0], "Batch size mismatch in quantized layer"
    print(f"{quantized_class.__name__}: Forward pass shape: {q_primary_output.shape}")
    
    # Compute L2 norm difference
    diff_norm = torch.norm(q_primary_output - n_primary_output, p=2).item()
    print(f"{quantized_class.__name__}: L2 norm difference (IntQuantized vs normal): {diff_norm:.6f}")
    
    # Backward pass check (IntQuantized only, skip if no trainable parameters)
    if requires_grad and target_tensor is not None:
        loss = F.mse_loss(q_primary_output, target_tensor)
        loss.backward()
        assert input_tensor.grad is not None, "No gradients computed for input"
        # Check gradients only for trainable parameters, excluding quantizer params
        has_trainable_params = False
        for name, param in quantized_layer.named_parameters():
            if param.requires_grad and 'quantizer' not in name:
                has_trainable_params = True
                assert param.grad is not None, f"No gradients for {name}"
        if has_trainable_params:
            print(f"{quantized_class.__name__}: Backward pass gradients OK, Loss: {loss.item():.4f}")
        else:
            print(f"{quantized_class.__name__}: No trainable parameters, backward pass skipped")

if __name__ == "__main__":
    # Test Cases
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Conv1d
    x_conv1d = torch.randn(2, 3, 10).to(device)
    target_conv1d = torch.randn(2, 16, 8).to(device)
    test_layer(IntQuantizedConv1d, nn.Conv1d, x_conv1d, target_conv1d, 
               init_args={'in_channels': 3, 'out_channels': 16, 'kernel_size': 3})

    # 2. LSTM
    x_lstm = torch.randn(2, 5, 10).to(device)
    target_lstm = torch.randn(2, 5, 20).to(device)
    test_layer(IntQuantizedLSTM, nn.LSTM, x_lstm, target_lstm, 
               init_args={'input_size': 10, 'hidden_size': 20, 'num_layers': 1, 'batch_first': True})

    # 3. Attention
    x_attn = torch.randn(5, 2, 32).to(device)  # [seq_len, batch, embed_dim]
    target_attn = torch.randn(5, 2, 32).to(device)
    test_layer(IntQuantizedAttention, nn.MultiheadAttention, x_attn, target_attn, 
               init_args={'embed_dim': 32, 'num_heads': 4}, key=x_attn, value=x_attn)

    # 4. GRU
    x_gru = torch.randn(2, 5, 10).to(device)
    target_gru = torch.randn(2, 5, 20).to(device)
    test_layer(IntQuantizedGRU, nn.GRU, x_gru, target_gru, 
               init_args={'input_size': 10, 'hidden_size': 20, 'num_layers': 1, 'batch_first': True})

    # 5. Linear
    x_linear = torch.randn(2, 784).to(device)
    target_linear = torch.randn(2, 10).to(device)
    test_layer(IntQuantizedLinear, nn.Linear, x_linear, target_linear, 
               init_args={'in_features': 784, 'out_features': 10})

    # 6. MaxPool2d
    x_maxpool = torch.randn(2, 3, 16, 16).to(device)
    test_layer(IntQuantizedMaxPool2d, nn.MaxPool2d, x_maxpool, 
               init_args={'kernel_size': 2, 'stride': 2})

    # 7. Conv2d
    x_conv2d = torch.randn(2, 3, 32, 32).to(device)
    target_conv2d = torch.randn(2, 16, 30, 30).to(device)
    test_layer(IntQuantizedConv2d, nn.Conv2d, x_conv2d, target_conv2d, 
               init_args={'in_channels': 3, 'out_channels': 16, 'kernel_size': 3})

    # 8. Conv3d
    x_conv3d = torch.randn(2, 3, 10, 10, 10).to(device)
    target_conv3d = torch.randn(2, 16, 8, 8, 8).to(device)
    test_layer(IntQuantizedConv3d, nn.Conv3d, x_conv3d, target_conv3d, 
               init_args={'in_channels': 3, 'out_channels': 16, 'kernel_size': 3})

    # 9. BatchNorm1d
    x_bn = torch.randn(16, 32, 3).to(device)
    target_bn = torch.randn(16, 32, 3).to(device)
    test_layer(IntQuantizedBatchNorm1d, nn.BatchNorm1d, x_bn, target_bn, 
               init_args={'num_features': 32})

    # 10. BatchNorm2d
    x_bn = torch.randn(2, 3, 16, 16).to(device)
    target_bn = torch.randn(2, 3, 16, 16).to(device)
    test_layer(IntQuantizedBatchNorm2d, nn.BatchNorm2d, x_bn, target_bn, 
               init_args={'num_features': 3})

    # 11. BatchNorm3d
    x_bn = torch.randn(2, 3, 16, 16, 16).to(device)
    target_bn = torch.randn(2, 3, 16, 16, 16).to(device)
    test_layer(IntQuantizedBatchNorm3d, nn.BatchNorm3d, x_bn, target_bn, 
               init_args={'num_features': 3})

    # 12. AvgPool2d
    x_avgpool = torch.randn(2, 3, 16, 16).to(device)
    test_layer(IntQuantizedAvgPool2d, nn.AvgPool2d, x_avgpool, 
               init_args={'kernel_size': 2, 'stride': 2})

    # 13. Dropout
    x_dropout = torch.randn(2, 10).to(device)
    test_layer(IntQuantizedDropout, nn.Dropout, x_dropout, 
               init_args={'p': 0.1})