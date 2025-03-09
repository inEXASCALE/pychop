
# Test the layers
import torch 
import sys
sys.path.append('../')
from pychop.layers import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    linear_layer = QuantizedLinear(10, 5, 5, 10, 4).to(device)
    x_linear = torch.randn(2, 10, device=device, requires_grad=True)
    y_linear = linear_layer(x_linear)
    loss_linear = y_linear.sum()
    loss_linear.backward()
    print("Quantized Linear - Output shape:", y_linear.shape, "Gradients:", x_linear.grad is not None)
    
    # Test QuantizedConv2d
    conv_layer = QuantizedConv2d(3, 16, 3, 5, 10, stride=1, padding=0, rmode=1).to(device)
    x_conv = torch.randn(2, 3, 32, 32, device=device, requires_grad=True)
    y_conv = conv_layer(x_conv)
    loss_conv = y_conv.sum()
    loss_conv.backward()
    print("Quantized Conv2d - Output shape:", y_conv.shape, "Gradients:", x_conv.grad is not None)
    
    # Test QuantizedRNN
    rnn_layer = QuantizedRNN(10, 20, 5, 10, num_layers=1, nonlinearity='tanh', rmode=5).to(device)
    x_rnn = torch.randn(2, 5, 10, device=device, requires_grad=True)
    y_rnn, h_rnn = rnn_layer(x_rnn)
    loss_rnn = y_rnn.sum()
    loss_rnn.backward()
    print("Quantized RNN - Output shape:", y_rnn.shape, "Gradients:", x_rnn.grad is not None)
    
    # Test QuantizedLSTM
    lstm_layer = QuantizedLSTM(10, 20, 5, 10, num_layers=1, rmode=6).to(device)
    x_lstm = torch.randn(2, 5, 10, device=device, requires_grad=True)
    y_lstm, (h_lstm, c_lstm) = lstm_layer(x_lstm)
    loss_lstm = y_lstm.sum()
    loss_lstm.backward()
    print("Quantized LSTM - Output shape:", y_lstm.shape, "Gradients:", x_lstm.grad is not None)

    # Test QuantizedBatchNorm2d
    bn_layer = QuantizedBatchNorm2d(3, 5, 10, rmode=1).to(device)
    x_bn = torch.randn(2, 3, 32, 32, device=device, requires_grad=True)
    y_bn = bn_layer(x_bn)
    loss_bn = y_bn.sum()
    loss_bn.backward()
    print("Quantized BatchNorm2d - Output shape:", y_bn.shape, "Gradients:", x_bn.grad is not None)
    
    # Test QuantizedLayerNorm
    ln_layer = QuantizedLayerNorm(512, 5, 10, rmode=4).to(device)
    x_ln = torch.randn(2, 10, 512, device=device, requires_grad=True)
    y_ln = ln_layer(x_ln)
    loss_ln = y_ln.sum()
    loss_ln.backward()
    print("Quantized LayerNorm - Output shape:", y_ln.shape, "Gradients:", x_ln.grad is not None)
    
    # Test QuantizedGRU
    gru_layer = QuantizedGRU(10, 20, 5, 10, num_layers=1, rmode=6).to(device)
    x_gru = torch.randn(2, 5, 10, device=device, requires_grad=True)
    y_gru, h_gru = gru_layer(x_gru)
    loss_gru = y_gru.sum()
    loss_gru.backward()
    print("Quantized GRU - Output shape:", y_gru.shape, "Gradients:", x_gru.grad is not None)
    
    # Test QuantizedMultiheadAttention
    mha_layer = QuantizedMultiheadAttention(512, 8, 5, 10, rmode=1).to(device)
    x_mha = torch.randn(2, 10, 512, device=device, requires_grad=True)
    y_mha, attn_weights = mha_layer(x_mha, x_mha, x_mha)
    loss_mha = y_mha.sum()
    loss_mha.backward()
    print("Quantized MultiheadAttention - Output shape:", y_mha.shape, "Gradients:", x_mha.grad is not None)
