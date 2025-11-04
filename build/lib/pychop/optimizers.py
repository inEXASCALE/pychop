import torch
from torch.optim import Optimizer
from .layers import BFPRound
from .tch import FPRound
import torch
from torch.optim import Optimizer
from typing import Dict, Any

# Quantized Adagrad Optimizer
class QuantizedAdagrad(Optimizer):
    def __init__(self, params, lr: float = 1e-2, lr_decay: float = 0, weight_decay: float = 0,
                 initial_accumulator_value: float = 0, eps: float = 1e-10,
                 exp_bits: int = 5, sig_bits: int = 10, rmode: int = 1):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if lr_decay < 0.0:
            raise ValueError(f"Invalid lr_decay value: {lr_decay}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")

        defaults = dict(
            lr=lr,
            lr_decay=lr_decay,
            weight_decay=weight_decay,
            initial_accumulator_value=initial_accumulator_value,
            eps=eps,
        )
        super().__init__(params, defaults)

        self.quantizer = BFPRound(exp_bits, sig_bits)
        

    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = self.quantizer.quantize(p.grad)
                state = self.state[p]

                # Initialize state
                if len(state) == 0:
                    state['sum'] = torch.full_like(p, group['initial_accumulator_value'])

                # Update sum of squared gradients
                state['sum'].addcmul_(grad, grad, value=1)
                sum_q = self.quantizer.quantize(state['sum'])
                state['sum'] = sum_q

                # Apply weight decay if specified
                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])
                    grad = self.quantizer.quantize(grad)

                # Compute update
                std = self.quantizer.quantize(sum_q.sqrt().add_(group['eps']))
                update = grad / std

                # Apply learning rate with decay
                if group['lr_decay'] > 0:
                    clr = group['lr'] / (1 + state.get('step', 0) * group['lr_decay'])
                else:
                    clr = group['lr']

                # Update parameters
                p.data.add_(-clr * update)
                p.data = self.quantizer.quantize(p.data)

                # Increment step
                state['step'] = state.get('step', 0) + 1

# Quantized Adadelta Optimizer
class QuantizedAdadelta(Optimizer):
    def __init__(self, params, lr: float = 1.0, rho: float = 0.9, eps: float = 1e-6,
                 weight_decay: float = 0, exp_bits: int = 5, sig_bits: int = 10,
                 rmode: int = 1):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if rho < 0.0 or rho > 1.0:
            raise ValueError(f"Invalid rho value: {rho}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, rho=rho, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

        self.quantizer = BFPRound(exp_bits, sig_bits, rmode)
        

    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = self.quantizer.quantize(p.grad)
                state = self.state[p]

                # Initialize state
                if len(state) == 0:
                    state['square_avg'] = torch.zeros_like(p)
                    state['acc_delta'] = torch.zeros_like(p)

                square_avg, acc_delta = state['square_avg'], state['acc_delta']
                rho = group['rho']

                # Apply weight decay
                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])
                    grad = self.quantizer.quantize(grad)

                # Update square average
                square_avg.mul_(rho).addcmul_(grad, grad, value=1 - rho)
                square_avg_q = self.quantizer.quantize(square_avg)
                state['square_avg'] = square_avg_q

                # Compute update
                std = self.quantizer.quantize(square_avg_q.sqrt().add_(group['eps']))
                delta = self.quantizer.quantize(acc_delta.sqrt().add_(group['eps']) / std * grad)

                # Update accumulated delta
                acc_delta.mul_(rho).addcmul_(delta, delta, value=1 - rho)
                acc_delta_q = self.quantizer.quantize(acc_delta)
                state['acc_delta'] = acc_delta_q

                # Update parameters
                p.data.add_(-group['lr'] * delta)
                p.data = self.quantizer.quantize(p.data)

# Quantized AdamW Optimizer
class QuantizedAdamW(Optimizer):
    def __init__(self, params, lr: float = 1e-3, betas: tuple = (0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 1e-2, exp_bits: int = 5, sig_bits: int = 10,
                 rmode: int = 1):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta_1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta_2: {betas[1]}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

        self.quantizer = BFPRound(exp_bits, sig_bits, rmode)
        

    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = self.quantizer.quantize(p.grad)
                state = self.state[p]

                # Initialize state
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1

                # Update moving averages
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Quantize moving averages
                exp_avg_q = self.quantizer.quantize(exp_avg)
                exp_avg_sq_q = self.quantizer.quantize(exp_avg_sq)

                state['exp_avg'] = exp_avg_q
                state['exp_avg_sq'] = exp_avg_sq_q

                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                step_size = group['lr'] * (bias_correction2 ** 0.5) / bias_correction1
                denom = self.quantizer.quantize(exp_avg_sq_q.sqrt().add_(group['eps']))
                update = self.quantizer.quantize(exp_avg_q / denom)

                # Apply weight decay
                if group['weight_decay'] != 0:
                    p.data.add_(-group['lr'] * group['weight_decay'], p.data)

                # Update parameters
                p.data.add_(-step_size * update)
                p.data = self.quantizer.quantize(p.data)

# Quantized SGD Optimizer
class QuantizedSGD(Optimizer):
    def __init__(self, params, lr: float, momentum: float = 0, exp_bits: int = 5, 
                 sig_bits: int = 10, rmode: int = 1):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
            
        defaults = dict(lr=lr, momentum=momentum)
        super().__init__(params, defaults)
        
        self.quantizer = BFPRound(exp_bits, sig_bits, rmode)
        
        
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                grad = self.quantizer.quantize(p.grad)
                state = self.state[p]
                if len(state) == 0 and momentum > 0:
                    state['momentum_buffer'] = torch.zeros_like(p)
                
                if momentum > 0:
                    buffer = state['momentum_buffer']
                    buffer.mul_(momentum).add_(grad)
                    buffer_q = self.quantizer.quantize(buffer)
                    state['momentum_buffer'] = buffer_q
                    update = buffer_q
                else:
                    update = grad
                
                p.data.add_(-lr * update)
                p.data = self.quantizer.quantize(p.data)



# Quantized Adam Optimizer
class QuantizedAdam(Optimizer):
    def __init__(self, params, lr: float = 1e-3, betas: tuple = (0.9, 0.999), eps: float = 1e-8,
                 exp_bits: int = 5, sig_bits: int = 10, rmode: int = 1):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta_1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta_2: {betas[1]}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon: {eps}")
            
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super().__init__(params, defaults)
        
        self.quantizer = BFPRound(exp_bits, sig_bits, rmode)
        
        
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                grad = self.quantizer.quantize(p.grad)
                state = self.state[p]
                
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1
                
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                exp_avg_q = self.quantizer.quantize(exp_avg)
                exp_avg_sq_q = self.quantizer.quantize(exp_avg_sq)
                
                state['exp_avg'] = exp_avg_q
                state['exp_avg_sq'] = exp_avg_sq_q
                
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                step_size = group['lr'] * (bias_correction2 ** 0.5) / bias_correction1
                denom = self.quantizer.quantize(exp_avg_sq_q.sqrt().add_(group['eps']))
                update = self.quantizer.quantize(exp_avg_q / denom)
                
                p.data.add_(-step_size * update)
                p.data = self.quantizer.quantize(p.data)



# Quantized RMSProp Optimizer
class QuantizedRMSProp(Optimizer):
    def __init__(self, params, lr: float = 1e-2, alpha: float = 0.99, eps: float = 1e-8,
                 momentum: float = 0, exp_bits: int = 5, sig_bits: int = 10,
                 rmode: int = 1):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if alpha < 0.0 or alpha > 1.0:
            raise ValueError(f"Invalid alpha: {alpha}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum: {momentum}")
            
        defaults = dict(lr=lr, alpha=alpha, eps=eps, momentum=momentum)
        super().__init__(params, defaults)
        
        self.quantizer = BFPRound(exp_bits, sig_bits, rmode)
        
        
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                grad = self.quantizer.quantize(p.grad)
                state = self.state[p]
                
                if len(state) == 0:
                    state['square_avg'] = torch.zeros_like(p)
                    if group['momentum'] > 0:
                        state['momentum_buffer'] = torch.zeros_like(p)
                
                square_avg = state['square_avg']
                alpha = group['alpha']
                
                square_avg.mul_(alpha).addcmul_(grad, grad, value=1 - alpha)
                square_avg_q = self.quantizer.quantize(square_avg)
                state['square_avg'] = square_avg_q
                
                avg = self.quantizer.quantize(square_avg_q.sqrt().add_(group['eps']))
                
                if group['momentum'] > 0:
                    buffer = state['momentum_buffer']
                    buffer.mul_(group['momentum']).addcdiv_(grad, avg)
                    buffer_q = self.quantizer.quantize(buffer)
                    state['momentum_buffer'] = buffer_q
                    update = buffer_q
                else:
                    update = grad / avg
                
                p.data.add_(-group['lr'] * update)
                p.data = self.quantizer.quantize(p.data)



# Quantized SGD Optimizer
class FPQuantizedSGD(Optimizer):
    def __init__(self, params, lr: float, momentum: float = 0, ibits: int = 8, 
                 fbits: int = 8, rmode: int = 1):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
            
        defaults = dict(lr=lr, momentum=momentum)
        super().__init__(params, defaults)
        
        self.quantizer = FPRound(ibits, fbits, rmode)
        
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                grad = self.quantizer.quantize(p.grad)
                state = self.state[p]
                if len(state) == 0 and momentum > 0:
                    state['momentum_buffer'] = torch.zeros_like(p)
                
                if momentum > 0:
                    buffer = state['momentum_buffer']
                    buffer.mul_(momentum).add_(grad)
                    buffer_q = self.quantizer.quantize(buffer)
                    state['momentum_buffer'] = buffer_q
                    update = buffer_q
                else:
                    update = grad
                
                p.data.add_(-lr * update)
                p.data = self.quantizer.quantize(p.data)

# Quantized Adam Optimizer
class FPQuantizedAdam(Optimizer):
    def __init__(self, params, lr: float = 1e-3, betas: tuple = (0.9, 0.999), eps: float = 1e-8,
                 ibits: int = 8, fbits: int = 8, rmode: int = 1):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta_1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta_2: {betas[1]}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon: {eps}")
            
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super().__init__(params, defaults)
        
        self.quantizer = FPRound(ibits, fbits, rmode)
        
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                grad = self.quantizer.quantize(p.grad)
                state = self.state[p]
                
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1
                
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                exp_avg_q = self.quantizer.quantize(exp_avg)
                exp_avg_sq_q = self.quantizer.quantize(exp_avg_sq)
                
                state['exp_avg'] = exp_avg_q
                state['exp_avg_sq'] = exp_avg_sq_q
                
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                step_size = group['lr'] * (bias_correction2 ** 0.5) / bias_correction1
                denom = self.quantizer.quantize(exp_avg_sq_q.sqrt().add_(group['eps']))
                update = self.quantizer.quantize(exp_avg_q / denom)
                
                p.data.add_(-step_size * update)
                p.data = self.quantizer.quantize(p.data)

# Quantized RMSProp Optimizer
class QuantizedRMSProp(Optimizer):
    def __init__(self, params, lr: float = 1e-2, alpha: float = 0.99, eps: float = 1e-8,
                 momentum: float = 0, ibits: int = 8, fbits: int = 8,
                 rmode: int = 1):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if alpha < 0.0 or alpha > 1.0:
            raise ValueError(f"Invalid alpha: {alpha}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum: {momentum}")
            
        defaults = dict(lr=lr, alpha=alpha, eps=eps, momentum=momentum)
        super().__init__(params, defaults)
        
        self.quantizer = FPRound(ibits, fbits, rmode)
        
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                grad = self.quantizer.quantize(p.grad)
                state = self.state[p]
                
                if len(state) == 0:
                    state['square_avg'] = torch.zeros_like(p)
                    if group['momentum'] > 0:
                        state['momentum_buffer'] = torch.zeros_like(p)
                
                square_avg = state['square_avg']
                alpha = group['alpha']
                
                square_avg.mul_(alpha).addcmul_(grad, grad, value=1 - alpha)
                square_avg_q = self.quantizer.quantize(square_avg)
                state['square_avg'] = square_avg_q
                
                avg = self.quantizer.quantize(square_avg_q.sqrt().add_(group['eps']))
                
                if group['momentum'] > 0:
                    buffer = state['momentum_buffer']
                    buffer.mul_(group['momentum']).addcdiv_(grad, avg)
                    buffer_q = self.quantizer.quantize(buffer)
                    state['momentum_buffer'] = buffer_q
                    update = buffer_q
                else:
                    update = self.quantizer.quantize(grad / avg)
                
                p.data.add_(-group['lr'] * update)
                p.data = self.quantizer.quantize(p.data)

# Quantized Adagrad Optimizer
class FPQuantizedAdagrad(Optimizer):
    def __init__(self, params, lr: float = 1e-2, lr_decay: float = 0, weight_decay: float = 0,
                 initial_accumulator_value: float = 0, eps: float = 1e-10,
                 ibits: int = 8, fbits: int = 8, rmode: int = 1):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if lr_decay < 0.0:
            raise ValueError(f"Invalid lr_decay value: {lr_decay}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")

        defaults = dict(
            lr=lr,
            lr_decay=lr_decay,
            weight_decay=weight_decay,
            initial_accumulator_value=initial_accumulator_value,
            eps=eps,
        )
        super().__init__(params, defaults)

        self.quantizer = FPRound(ibits, fbits, rmode)

    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = self.quantizer.quantize(p.grad)
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['sum'] = torch.full_like(p, group['initial_accumulator_value'])

                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])
                    grad = self.quantizer.quantize(grad)

                state['sum'].addcmul_(grad, grad, value=1)
                sum_q = self.quantizer.quantize(state['sum'])
                state['sum'] = sum_q

                std = self.quantizer.quantize(sum_q.sqrt().add_(group['eps']))
                update = self.quantizer.quantize(grad / std)

                if group['lr_decay'] > 0:
                    clr = group['lr'] / (1 + state.get('step', 0) * group['lr_decay'])
                else:
                    clr = group['lr']

                p.data.add_(-clr * update)
                p.data = self.quantizer.quantize(p.data)

                state['step'] += 1

# Quantized Adadelta Optimizer
class FPQuantizedAdadelta(Optimizer):
    def __init__(self, params, lr: float = 1.0, rho: float = 0.9, eps: float = 1e-6,
                 weight_decay: float = 0, ibits: int = 8, fbits: int = 8,
                 rmode: int = 1):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if rho < 0.0 or rho > 1.0:
            raise ValueError(f"Invalid rho value: {rho}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, rho=rho, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

        self.quantizer = FPRound(ibits, fbits, rmode)

    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = self.quantizer.quantize(p.grad)
                state = self.state[p]

                if len(state) == 0:
                    state['square_avg'] = torch.zeros_like(p)
                    state['acc_delta'] = torch.zeros_like(p)

                square_avg, acc_delta = state['square_avg'], state['acc_delta']
                rho = group['rho']

                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])
                    grad = self.quantizer.quantize(grad)

                square_avg.mul_(rho).addcmul_(grad, grad, value=1 - rho)
                square_avg_q = self.quantizer.quantize(square_avg)
                state['square_avg'] = square_avg_q

                std = self.quantizer.quantize(square_avg_q.sqrt().add_(group['eps']))
                delta = self.quantizer.quantize(acc_delta.sqrt().add_(group['eps']) / std * grad)

                acc_delta.mul_(rho).addcmul_(delta, delta, value=1 - rho)
                acc_delta_q = self.quantizer.quantize(acc_delta)
                state['acc_delta'] = acc_delta_q

                p.data.add_(-group['lr'] * delta)
                p.data = self.quantizer.quantize(p.data)

# Quantized AdamW Optimizer
class FPQuantizedAdamW(Optimizer):
    def __init__(self, params, lr: float = 1e-3, betas: tuple = (0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 1e-2, ibits: int = 8, fbits: int = 8,
                 rmode: int = 1):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta_1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta_2: {betas[1]}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

        self.quantizer = FPRound(ibits, fbits, rmode)

    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = self.quantizer.quantize(p.grad)
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                exp_avg_q = self.quantizer.quantize(exp_avg)
                exp_avg_sq_q = self.quantizer.quantize(exp_avg_sq)

                state['exp_avg'] = exp_avg_q
                state['exp_avg_sq'] = exp_avg_sq_q

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                step_size = group['lr'] * (bias_correction2 ** 0.5) / bias_correction1
                denom = self.quantizer.quantize(exp_avg_sq_q.sqrt().add_(group['eps']))
                update = self.quantizer.quantize(exp_avg_q / denom)

                if group['weight_decay'] != 0:
                    p.data.add_(-group['lr'] * group['weight_decay'], p.data)

                p.data.add_(-step_size * update)
                p.data = self.quantizer.quantize(p.data)
