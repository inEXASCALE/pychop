import torch
from torch.optim import Optimizer
from .layers import FPRound

# 1. Quantized SGD with Momentum
class QuantizedSGD(Optimizer):
    """Custom SGD optimizer with quantized momentum."""
    def __init__(self, params, lr=0.01, momentum=0.9, weight_decay=0, 
                 exp_bits=8, sig_bits=7, rmode=1):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                        exp_bits=exp_bits, sig_bits=sig_bits, rmode=rmode)
        super(QuantizedSGD, self).__init__(params, defaults)
        self.quantizer = FPRound(exp_bits, sig_bits, rmode)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])

                state = self.state[p]
                if len(state) == 0 and group['momentum'] != 0:
                    state['momentum_buffer'] = torch.zeros_like(p.data)

                if group['momentum'] != 0:
                    buf = state['momentum_buffer']
                    buf.mul_(group['momentum']).add_(grad, alpha=1 - group['momentum'])  # Momentum update
                    q_buf = self.quantizer.quantize(buf)  # Quantize momentum
                    update = q_buf
                    state['momentum_buffer'] = q_buf  # Store quantized momentum
                else:
                    update = grad

                q_update = self.quantizer.quantize(update)  # Quantize update
                p.data.add_(-group['lr'] * q_update)

        return loss

# 2. Quantized RMSprop
class QuantizedRMSprop(Optimizer):
    """Custom RMSprop optimizer with quantized accumulator."""
    def __init__(self, params, lr=0.01, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0,
                 exp_bits=8, sig_bits=7, rmode=1):
        defaults = dict(lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay, momentum=momentum,
                        exp_bits=exp_bits, sig_bits=sig_bits, rmode=rmode)
        super(QuantizedRMSprop, self).__init__(params, defaults)
        self.quantizer = FPRound(exp_bits, sig_bits, rmode)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])

                state = self.state[p]
                if len(state) == 0:
                    state['square_avg'] = torch.zeros_like(p.data)
                    if group['momentum'] > 0:
                        state['momentum_buffer'] = torch.zeros_like(p.data)

                square_avg = state['square_avg']
                square_avg.mul_(group['alpha']).addcmul_(grad, grad, value=1 - group['alpha'])  # Accumulator update
                q_square_avg = self.quantizer.quantize(square_avg)  # Quantize accumulator

                avg = q_square_avg.sqrt().add_(group['eps'])  # Quantized denominator

                if group['momentum'] > 0:
                    buf = state['momentum_buffer']
                    buf.mul_(group['momentum']).addcdiv_(grad, avg, value=1 - group['momentum'])
                    q_buf = self.quantizer.quantize(buf)  # Quantize momentum
                    update = q_buf
                    state['momentum_buffer'] = q_buf
                else:
                    update = grad / avg

                q_update = self.quantizer.quantize(update)  # Quantize update
                p.data.add_(-group['lr'] * q_update)

                state['square_avg'] = q_square_avg  # Store quantized accumulator

        return loss

# 3. Quantized Adagrad
class QuantizedAdagrad(Optimizer):
    """Custom Adagrad optimizer with quantized accumulator."""
    def __init__(self, params, lr=0.01, lr_decay=0, weight_decay=0, eps=1e-10,
                 exp_bits=8, sig_bits=7, rmode=1):
        defaults = dict(lr=lr, lr_decay=lr_decay, weight_decay=weight_decay, eps=eps,
                        exp_bits=exp_bits, sig_bits=sig_bits, rmode=rmode)
        super(QuantizedAdagrad, self).__init__(params, defaults)
        self.quantizer = FPRound(exp_bits, sig_bits, rmode)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])

                state = self.state[p]
                if len(state) == 0:
                    state['sum'] = torch.zeros_like(p.data)
                    state['step'] = 0

                state['step'] += 1
                sum_sq = state['sum']
                sum_sq.addcmul_(grad, grad, value=1)  # Accumulator update (sum of squared gradients)
                q_sum_sq = self.quantizer.quantize(sum_sq)  # Quantize accumulator

                std = q_sum_sq.sqrt().add_(group['eps'])  # Quantized denominator
                update = grad / std

                q_update = self.quantizer.quantize(update)  # Quantize update
                lr = group['lr'] / (1 + group['lr_decay'] * state['step'])  # Apply learning rate decay
                p.data.add_(-lr * q_update)

                state['sum'] = q_sum_sq  # Store quantized accumulator

        return loss

# 4. Quantized Adam (from previous response, repeated for completeness)
class QuantizedAdam(Optimizer):
    """Custom Adam optimizer with quantized momentum and accumulators."""
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0,
                 exp_bits=8, sig_bits=7, rmode=1):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        exp_bits=exp_bits, sig_bits=sig_bits, rmode=rmode)
        super(QuantizedAdam, self).__init__(params, defaults)
        self.quantizer = FPRound(exp_bits, sig_bits, rmode)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)  # Momentum
                    state['exp_avg_sq'] = torch.zeros_like(p.data)  # Accumulator

                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                step = state['step']

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)  # Momentum update
                q_exp_avg = self.quantizer.quantize(exp_avg)      # Quantize momentum

                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)  # Accumulator update
                q_exp_avg_sq = self.quantizer.quantize(exp_avg_sq)            # Quantize accumulator

                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                step_size = group['lr'] * (bias_correction2 ** 0.5) / bias_correction1

                denom = q_exp_avg_sq.sqrt().add_(group['eps'])
                update = q_exp_avg / denom
                q_update = self.quantizer.quantize(update)  # Quantize update

                p.data.add_(-step_size * q_update)

                state['exp_avg'] = q_exp_avg
                state['exp_avg_sq'] = q_exp_avg_sq

        return loss
