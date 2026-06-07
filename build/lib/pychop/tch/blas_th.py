import torch
import numpy as np
import pychop
import logging

logging.basicConfig(level=logging.INFO) 

pychop.backend('torch')

precision_configs = {  # Precision configurations
    'q52': {'exp_bits': 5, 'sig_bits': 2, 'rmode': 1},
    'q43': {'exp_bits': 4, 'sig_bits': 3, 'rmode': 1},
    'bf16': {'exp_bits': 8, 'sig_bits': 7, 'rmode': 1},
    'half': {'exp_bits': 5, 'sig_bits': 10, 'rmode': 1},
    'tf32': {'exp_bits': 8, 'sig_bits': 10, 'rmode': 1},
    'fp32': {'exp_bits': 8, 'sig_bits': 23, 'rmode': 1},
    'fp64': {'exp_bits': 11, 'sig_bits': 52, 'rmode': 1}
}

precision_fallback = ['q52', 'q43', 'bf16', 'half', 'tf32', 'fp32', 'fp64']

def get_dtype(precision):
    if isinstance(precision, dict):
        return None, None
    d = {
        'fp64': (torch.float64, torch.complex128),
        'fp32': (torch.float32, torch.complex64),
        'half': (torch.float16, None),
        'bf16': (torch.bfloat16, None),
    }
    return d.get(precision, (None, None))

def chop(x, precision='q52', use_gpu=False):
    """Recursive chop function with support for custom precision"""
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    if not torch.is_tensor(x):
        dtype = x.type
        x = torch.tensor(x, dtype=dtype, device=device)
    if precision == 'fp64':
        type_default = torch.float64
        return x.to(device)
    
    elif precision == 'fp32':
        type_default = torch.float32

    else:
        type_default = torch.float64

    if isinstance(precision, dict):
        config = precision

    else:
        config = precision_configs[precision]
    
    ch = pychop.LightChop(**config)
    result = ch(x)
    dtype = type_default if not torch.is_complex(x) else torch.complex128
    return result.to(dtype).to(device)

def rounding(x, precision, use_gpu=False):
    """Round tensor to specified precision"""
    return chop(x, precision=precision, use_gpu=use_gpu)

# Level 1 BLAS: Vector-Vector Operations
def axpy(alpha, x, y, precision='fp64', exp_bits=None, sig_bits=None, rmode=None, use_gpu=False):
    """y = alpha * x + y"""
    if exp_bits is not None and sig_bits is not None:
        if exp_bits == 23 and sig_bits == 52:
            precision = 'fp64'
        elif exp_bits == 8 and sig_bits == 23:
            precision = 'fp32'
        else:
            precision = {'exp_bits': exp_bits, 'sig_bits': sig_bits, 'rmode': rmode or 1}
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    is_complex = np.iscomplexobj(alpha) or np.iscomplexobj(x) or np.iscomplexobj(y)
    dtype = torch.complex128 if is_complex else torch.float64
    x = torch.as_tensor(x, dtype=dtype, device=device)
    y = torch.as_tensor(y, dtype=dtype, device=device)
    alpha = torch.tensor(alpha, dtype=dtype, device=device)
    if x.shape != y.shape or x.dim() != 1:
        raise ValueError("x and y must be 1D vectors of same length")
    native_real, native_complex = get_dtype(precision)
    native_dtype = native_complex if is_complex else native_real
    if is_complex and precision in ['half', 'bf16']:
        native_dtype = None
    if native_dtype is not None:
        alpha = alpha.to(native_dtype)
        x = x.to(native_dtype)
        y = y.to(native_dtype)
        result = alpha * x + y
        result = result.to(dtype)
    else:
        alpha = rounding(alpha, precision, use_gpu)
        x = rounding(x, precision, use_gpu)
        y = rounding(y, precision, use_gpu)
        mul = rounding(alpha * x, precision, use_gpu)
        result = rounding(mul + y, precision, use_gpu)
    return result

def scal(alpha, x, precision='fp64', exp_bits=None, sig_bits=None, rmode=None, use_gpu=False):
    """x = alpha * x"""
    if exp_bits is not None and sig_bits is not None:
        if exp_bits == 23 and sig_bits == 52:
            precision = 'fp64'
        elif exp_bits == 8 and sig_bits == 23:
            precision = 'fp32'
        else:
            precision = {'exp_bits': exp_bits, 'sig_bits': sig_bits, 'rmode': rmode or 1}
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    is_complex = np.iscomplexobj(alpha) or np.iscomplexobj(x)
    dtype = torch.complex128 if is_complex else torch.float64
    x = torch.as_tensor(x, dtype=dtype, device=device)
    alpha = torch.tensor(alpha, dtype=dtype, device=device)
    if x.dim() != 1:
        raise ValueError("x must be a 1D vector")
    native_real, native_complex = get_dtype(precision)
    native_dtype = native_complex if is_complex else native_real
    if is_complex and precision in ['half', 'bf16']:
        native_dtype = None
    if native_dtype is not None:
        alpha = alpha.to(native_dtype)
        x = x.to(native_dtype)
        result = alpha * x
        result = result.to(dtype)
    else:
        alpha = rounding(alpha, precision, use_gpu)
        x = rounding(x, precision, use_gpu)
        result = rounding(alpha * x, precision, use_gpu)
    return result

def copy(x, y, precision='fp64', exp_bits=None, sig_bits=None, rmode=None, use_gpu=False):
    """y = x"""
    if exp_bits is not None and sig_bits is not None:
        if exp_bits == 23 and sig_bits == 52:
            precision = 'fp64'
        elif exp_bits == 8 and sig_bits == 23:
            precision = 'fp32'
        else:
            precision = {'exp_bits': exp_bits, 'sig_bits': sig_bits, 'rmode': rmode or 1}
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    is_complex = np.iscomplexobj(x) or np.iscomplexobj(y)
    dtype = torch.complex128 if is_complex else torch.float64
    x = torch.as_tensor(x, dtype=dtype, device=device)
    y = torch.as_tensor(y, dtype=dtype, device=device)
    if x.shape != y.shape or x.dim() != 1:
        raise ValueError("x and y must be 1D vectors of same length")
    native_real, native_complex = get_dtype(precision)
    native_dtype = native_complex if is_complex else native_real
    if is_complex and precision in ['half', 'bf16']:
        native_dtype = None
    if native_dtype is not None:
        x = x.to(native_dtype)
        result = x.clone()
        result = result.to(dtype)
    else:
        x = rounding(x, precision, use_gpu)
        result = x
    return result

def swap(x, y, precision='fp64', exp_bits=None, sig_bits=None, rmode=None, use_gpu=False):
    """x <-> y"""
    if exp_bits is not None and sig_bits is not None:
        if exp_bits == 23 and sig_bits == 52:
            precision = 'fp64'
        elif exp_bits == 8 and sig_bits == 23:
            precision = 'fp32'
        else:
            precision = {'exp_bits': exp_bits, 'sig_bits': sig_bits, 'rmode': rmode or 1}
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    is_complex = np.iscomplexobj(x) or np.iscomplexobj(y)
    dtype = torch.complex128 if is_complex else torch.float64
    x = torch.as_tensor(x, dtype=dtype, device=device)
    y = torch.as_tensor(y, dtype=dtype, device=device)
    if x.shape != y.shape or x.dim() != 1:
        raise ValueError("x and y must be 1D vectors of same length")
    native_real, native_complex = get_dtype(precision)
    native_dtype = native_complex if is_complex else native_real
    if is_complex and precision in ['half', 'bf16']:
        native_dtype = None
    if native_dtype is not None:
        x = x.to(native_dtype)
        y = y.to(native_dtype)
        x_new = y.to(dtype)
        y_new = x.to(dtype)
    else:
        x = rounding(x, precision, use_gpu)
        y = rounding(y, precision, use_gpu)
        x_new = y
        y_new = x
    return x_new, y_new

def dot(x, y, precision='fp64', exp_bits=None, sig_bits=None, rmode=None, use_gpu=False):
    """x . y"""
    if exp_bits is not None and sig_bits is not None:
        if exp_bits == 23 and sig_bits == 52:
            precision = 'fp64'
        elif exp_bits == 8 and sig_bits == 23:
            precision = 'fp32'
        else:
            precision = {'exp_bits': exp_bits, 'sig_bits': sig_bits, 'rmode': rmode or 1}
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    is_complex = np.iscomplexobj(x) or np.iscomplexobj(y)
    dtype = x.dtype
    x = torch.as_tensor(x, dtype=dtype, device=device)
    y = torch.as_tensor(y, dtype=dtype, device=device)
    if x.shape != y.shape or x.dim() != 1:
        raise ValueError("x and y must be 1D vectors of same length")
    native_real, native_complex = get_dtype(precision)
    native_dtype = native_complex if is_complex else native_real
    if is_complex and precision in ['half', 'bf16']:
        native_dtype = None
    if native_dtype is not None:
        x = x.to(native_dtype)
        y = y.to(native_dtype)
        result = torch.sum(x * y)
        result = result.to(dtype)
    else:
        x = rounding(x, precision, use_gpu)
        y = rounding(y, precision, use_gpu)
        acc = torch.zeros_like(x[0], device=device)
        for i in range(x.shape[0]):
            prod = rounding(x[i] * y[i], precision, use_gpu)
            acc = rounding(acc + prod, precision, use_gpu)
        result = acc
    return result.item()

def dotc(x, y, precision='fp64', exp_bits=None, sig_bits=None, rmode=None, use_gpu=False):
    """x . conj(y) (complex)"""
    if exp_bits is not None and sig_bits is not None:
        if exp_bits == 23 and sig_bits == 52:
            precision = 'fp64'
        elif exp_bits == 8 and sig_bits == 23:
            precision = 'fp32'
        else:
            precision = {'exp_bits': exp_bits, 'sig_bits': sig_bits, 'rmode': rmode or 1}
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    x = torch.as_tensor(x, dtype=torch.complex128, device=device)
    y = torch.as_tensor(y, dtype=torch.complex128, device=device)
    if x.shape != y.shape or x.dim() != 1:
        raise ValueError("x and y must be 1D vectors of same length")
    native_real, native_complex = get_dtype(precision)
    native_dtype = native_complex
    if precision in ['half', 'bf16']:
        native_dtype = None
    if native_dtype is not None:
        x = x.to(native_dtype)
        y = y.to(native_dtype)
        result = torch.sum(x * torch.conj(y))
        result = result.to(torch.complex128)
    else:
        x = rounding(x, precision, use_gpu)
        y = rounding(y, precision, use_gpu)
        acc = torch.zeros_like(x[0], device=device)
        for i in range(x.shape[0]):
            prod = rounding(x[i] * torch.conj(y[i]), precision, use_gpu)
            acc = rounding(acc + prod, precision, use_gpu)
        result = acc
    return result.item()

def dotu(x, y, precision='fp64', exp_bits=None, sig_bits=None, rmode=None, use_gpu=False):
    """x . y (complex, no conjugate)"""
    if exp_bits is not None and sig_bits is not None:
        if exp_bits == 23 and sig_bits == 52:
            precision = 'fp64'
        elif exp_bits == 8 and sig_bits == 23:
            precision = 'fp32'
        else:
            precision = {'exp_bits': exp_bits, 'sig_bits': sig_bits, 'rmode': rmode or 1}
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    x = torch.as_tensor(x, dtype=torch.complex128, device=device)
    y = torch.as_tensor(y, dtype=torch.complex128, device=device)
    if x.shape != y.shape or x.dim() != 1:
        raise ValueError("x and y must be 1D vectors of same length")
    native_real, native_complex = get_dtype(precision)
    native_dtype = native_complex
    if precision in ['half', 'bf16']:
        native_dtype = None
    if native_dtype is not None:
        x = x.to(native_dtype)
        y = y.to(native_dtype)
        result = torch.sum(x * y)
        result = result.to(torch.complex128)
    else:
        x = rounding(x, precision, use_gpu)
        y = rounding(y, precision, use_gpu)
        acc = torch.zeros_like(x[0], device=device)
        for i in range(x.shape[0]):
            prod = rounding(x[i] * y[i], precision, use_gpu)
            acc = rounding(acc + prod, precision, use_gpu)
        result = acc
    return result.item()

def nrm2(x, precision='fp64', exp_bits=None, sig_bits=None, rmode=None, use_gpu=False):
    """Euclidean norm of x"""
    if exp_bits is not None and sig_bits is not None:
        if exp_bits == 23 and sig_bits == 52:
            precision = 'fp64'
        elif exp_bits == 8 and sig_bits == 23:
            precision = 'fp32'
        else:
            precision = {'exp_bits': exp_bits, 'sig_bits': sig_bits, 'rmode': rmode or 1}
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    is_complex = np.iscomplexobj(x)
    dtype = torch.complex128 if is_complex else torch.float64
    x = torch.as_tensor(x, dtype=dtype, device=device)
    if x.dim() != 1:
        raise ValueError("x must be a 1D vector")
    native_real, native_complex = get_dtype(precision)
    native_dtype = native_complex if is_complex else native_real
    if is_complex and precision in ['half', 'bf16']:
        native_dtype = None
    if native_dtype is not None:
        x = x.to(native_dtype)
        result = torch.sqrt(torch.sum(torch.abs(x)**2))
        result = result.to(torch.float64)  # nrm2 is real
    else:
        x = rounding(x, precision, use_gpu)
        acc = torch.zeros(1, dtype=torch.float64, device=device)[0]
        for i in range(x.shape[0]):
            abs2 = x[i].conj() * x[i] if is_complex else x[i] * x[i]
            abs2_rounded = rounding(abs2, precision, use_gpu)
            acc = rounding(acc + torch.real(abs2_rounded), precision, use_gpu)
        result = rounding(torch.sqrt(acc), precision, use_gpu)
    return result.item()

def asum(x, precision='fp64', exp_bits=None, sig_bits=None, rmode=None, use_gpu=False):
    """Sum of absolute values of x"""
    if exp_bits is not None and sig_bits is not None:
        if exp_bits == 23 and sig_bits == 52:
            precision = 'fp64'
        elif exp_bits == 8 and sig_bits == 23:
            precision = 'fp32'
        else:
            precision = {'exp_bits': exp_bits, 'sig_bits': sig_bits, 'rmode': rmode or 1}
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    is_complex = np.iscomplexobj(x)
    dtype = torch.complex128 if is_complex else torch.float64
    x = torch.as_tensor(x, dtype=dtype, device=device)
    if x.dim() != 1:
        raise ValueError("x must be a 1D vector")
    native_real, native_complex = get_dtype(precision)
    native_dtype = native_complex if is_complex else native_real
    if is_complex and precision in ['half', 'bf16']:
        native_dtype = None
    if native_dtype is not None:
        x = x.to(native_dtype)
        result = torch.sum(torch.abs(x))
        result = result.to(torch.float64)  # asum is real
    else:
        x = rounding(x, precision, use_gpu)
        acc = torch.zeros(1, dtype=torch.float64, device=device)[0]
        for i in range(x.shape[0]):
            abs_val = torch.abs(x[i])
            acc = rounding(acc + abs_val, precision, use_gpu)
        result = acc
    return result.item()

def rot(x, y, c, s, precision='fp64', exp_bits=None, sig_bits=None, rmode=None, use_gpu=False):
    """Apply Givens rotation: x' = c*x + s*y, y' = -s*x + c*y"""
    if exp_bits is not None and sig_bits is not None:
        if exp_bits == 23 and sig_bits == 52:
            precision = 'fp64'
        elif exp_bits == 8 and sig_bits == 23:
            precision = 'fp32'
        else:
            precision = {'exp_bits': exp_bits, 'sig_bits': sig_bits, 'rmode': rmode or 1}
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    is_complex = np.iscomplexobj(x) or np.iscomplexobj(y) or np.iscomplexobj(s)
    dtype = torch.complex128 if is_complex else torch.float64
    x = torch.as_tensor(x, dtype=dtype, device=device)
    y = torch.as_tensor(y, dtype=dtype, device=device)
    c = torch.tensor(c, dtype=torch.float64, device=device)
    s = torch.tensor(s, dtype=dtype, device=device)
    if x.shape != y.shape or x.dim() != 1:
        raise ValueError("x and y must be 1D vectors of same length")
    native_real, native_complex = get_dtype(precision)
    native_dtype = native_complex if is_complex else native_real
    if is_complex and precision in ['half', 'bf16']:
        native_dtype = None
    if native_dtype is not None:
        x = x.to(native_dtype)
        y = y.to(native_dtype)
        c = c.to(native_real)  # c is real
        s = s.to(native_dtype)
        x_new = c * x + s * y
        y_new = -s * x + c * y
        x_new = x_new.to(dtype)
        y_new = y_new.to(dtype)
    else:
        x = rounding(x, precision, use_gpu)
        y = rounding(y, precision, use_gpu)
        c = rounding(c, precision, use_gpu)
        s = rounding(s, precision, use_gpu)
        cx = rounding(c * x, precision, use_gpu)
        sy = rounding(s * y, precision, use_gpu)
        x_new = rounding(cx + sy, precision, use_gpu)
        neg_s = rounding(-s, precision, use_gpu)
        sx = rounding(neg_s * x, precision, use_gpu)
        cy = rounding(c * y, precision, use_gpu)
        y_new = rounding(sx + cy, precision, use_gpu)
    return x_new, y_new

def rotg(a, b, precision='fp64', exp_bits=None, sig_bits=None, rmode=None, use_gpu=False):
    """Generate Givens rotation: compute c, s, r, z for rotation"""
    if exp_bits is not None and sig_bits is not None:
        if exp_bits == 23 and sig_bits == 52:
            precision = 'fp64'
        elif exp_bits == 8 and sig_bits == 23:
            precision = 'fp32'
        else:
            precision = {'exp_bits': exp_bits, 'sig_bits': sig_bits, 'rmode': rmode or 1}
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    a = torch.tensor(a, dtype=torch.float64, device=device)
    b = torch.tensor(b, dtype=torch.float64, device=device)
    native_real, native_complex = get_dtype(precision)
    native_dtype = native_real
    if native_dtype is not None:
        a = a.to(native_dtype)
        b = b.to(native_dtype)
        r = torch.sqrt(a**2 + b**2)
        c = a / r if r != 0 else 1.0
        s = b / r if r != 0 else 0.0
        z = s if c != 0 else 1.0
        r = r.to(torch.float64)
        c = c.to(torch.float64)
        s = s.to(torch.float64)
        z = z.to(torch.float64)
    else:
        a = rounding(a, precision, use_gpu)
        b = rounding(b, precision, use_gpu)
        a2 = rounding(a ** 2, precision, use_gpu)
        b2 = rounding(b ** 2, precision, use_gpu)
        sum_sq = rounding(a2 + b2, precision, use_gpu)
        r = rounding(torch.sqrt(sum_sq), precision, use_gpu)
        if r == 0:
            c = torch.tensor(1.0, device=device)
            s = torch.tensor(0.0, device=device)
        else:
            c = rounding(a / r, precision, use_gpu)
            s = rounding(b / r, precision, use_gpu)
        z = s if c != 0 else torch.tensor(1.0, device=device)
    return (r.item(), c.item(), s.item(), z.item())

def rotm(x, y, param, precision='fp64', exp_bits=None, sig_bits=None, rmode=None, use_gpu=False):
    """Apply modified rotation (Hessenberg)"""
    if exp_bits is not None and sig_bits is not None:
        if exp_bits == 23 and sig_bits == 52:
            precision = 'fp64'
        elif exp_bits == 8 and sig_bits == 23:
            precision = 'fp32'
        else:
            precision = {'exp_bits': exp_bits, 'sig_bits': sig_bits, 'rmode': rmode or 1}
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    x = torch.as_tensor(x, dtype=torch.float64, device=device)
    y = torch.as_tensor(y, dtype=torch.float64, device=device)
    param = torch.as_tensor(param, dtype=torch.float64, device=device)  # [flag, h11, h21, h12, h22]
    if x.shape != y.shape or x.dim() != 1 or param.shape != (5,):
        raise ValueError("x, y must be 1D vectors of same length, param must be 5-element vector")
    native_real, native_complex = get_dtype(precision)
    native_dtype = native_real
    if native_dtype is not None:
        x = x.to(native_dtype)
        y = y.to(native_dtype)
        param = param.to(native_dtype)
        flag = param[0]
        if flag == -1:
            h11, h21, h12, h22 = param[1:5]
            x_new = h11 * x + h12 * y
            y_new = h21 * x + h22 * y
        elif flag == 0:
            x_new, y_new = x, y
        elif flag == 1:
            h11, h12 = param[1], param[3]
            x_new = h11 * x + h12 * y
            y_new = y
        else:  # flag == 2
            h21, h22 = param[2], param[4]
            x_new = x
            y_new = h21 * x + h22 * y
        x_new = x_new.to(torch.float64)
        y_new = y_new.to(torch.float64)
    else:
        x = rounding(x, precision, use_gpu)
        y = rounding(y, precision, use_gpu)
        param = rounding(param, precision, use_gpu)
        flag = param[0]
        if flag == -1:
            h11, h21, h12, h22 = param[1:5]
            h11x = rounding(h11 * x, precision, use_gpu)
            h12y = rounding(h12 * y, precision, use_gpu)
            x_new = rounding(h11x + h12y, precision, use_gpu)
            h21x = rounding(h21 * x, precision, use_gpu)
            h22y = rounding(h22 * y, precision, use_gpu)
            y_new = rounding(h21x + h22y, precision, use_gpu)
        elif flag == 0:
            x_new, y_new = x, y
        elif flag == 1:
            h11, h12 = param[1], param[3]
            h11x = rounding(h11 * x, precision, use_gpu)
            h12y = rounding(h12 * y, precision, use_gpu)
            x_new = rounding(h11x + h12y, precision, use_gpu)
            y_new = y
        else:  # flag == 2
            h21, h22 = param[2], param[4]
            x_new = x
            h21x = rounding(h21 * x, precision, use_gpu)
            h22y = rounding(h22 * y, precision, use_gpu)
            y_new = rounding(h21x + h22y, precision, use_gpu)
    return x_new, y_new

def rotmg(d1, d2, x1, y1, precision='fp64', exp_bits=None, sig_bits=None, rmode=None, use_gpu=False):
    """Generate modified rotation parameters"""
    if exp_bits is not None and sig_bits is not None:
        if exp_bits == 23 and sig_bits == 52:
            precision = 'fp64'
        elif exp_bits == 8 and sig_bits == 23:
            precision = 'fp32'
        else:
            precision = {'exp_bits': exp_bits, 'sig_bits': sig_bits, 'rmode': rmode or 1}
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    d1 = torch.tensor(d1, dtype=torch.float64, device=device)
    d2 = torch.tensor(d2, dtype=torch.float64, device=device)
    x1 = torch.tensor(x1, dtype=torch.float64, device=device)
    y1 = torch.tensor(y1, dtype=torch.float64, device=device)
    native_real, native_complex = get_dtype(precision)
    native_dtype = native_real
    if native_dtype is not None:
        d1 = d1.to(native_dtype)
        d2 = d2.to(native_dtype)
        x1 = x1.to(native_dtype)
        y1 = y1.to(native_dtype)
        param = torch.zeros(5, dtype=native_dtype, device=device)
        if d1 == 0 or d2 == 0 or x1 == 0:
            param[0] = -1
        else:
            param[0] = -1  # Flag for full matrix
            param[1] = 1.0  # h11
            param[2] = 0.0  # h21
            param[3] = 0.0  # h12
            param[4] = 1.0  # h22
        param = param.to(torch.float64)
    else:
        d1 = rounding(d1, precision, use_gpu)
        d2 = rounding(d2, precision, use_gpu)
        x1 = rounding(x1, precision, use_gpu)
        y1 = rounding(y1, precision, use_gpu)
        param = torch.zeros(5, dtype=torch.float64, device=device)
        if d1 == 0 or d2 == 0 or x1 == 0:
            param[0] = -1
        else:
            param[0] = -1  # Flag for full matrix
            param[1] = 1.0  # h11
            param[2] = 0.0  # h21
            param[3] = 0.0  # h12
            param[4] = 1.0  # h22
        param = rounding(param, precision, use_gpu)
    return param

def iamax(x, precision='fp64', exp_bits=None, sig_bits=None, rmode=None, use_gpu=False):
    """Index of maximum absolute value"""
    if exp_bits is not None and sig_bits is not None:
        if exp_bits == 23 and sig_bits == 52:
            precision = 'fp64'
        elif exp_bits == 8 and sig_bits == 23:
            precision = 'fp32'
        else:
            precision = {'exp_bits': exp_bits, 'sig_bits': sig_bits, 'rmode': rmode or 1}
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    is_complex = np.iscomplexobj(x)
    dtype = torch.complex128 if is_complex else torch.float64
    x = torch.as_tensor(x, dtype=dtype, device=device)
    if x.dim() != 1:
        raise ValueError("x must be a 1D vector")
    native_real, native_complex = get_dtype(precision)
    native_dtype = native_complex if is_complex else native_real
    if is_complex and precision in ['half', 'bf16']:
        native_dtype = None
    if native_dtype is not None:
        x = x.to(native_dtype)
        idx = torch.argmax(torch.abs(x))
    else:
        x = rounding(x, precision, use_gpu)
        idx = torch.argmax(torch.abs(x))
    return idx.item()

def iamin(x, precision='fp64', exp_bits=None, sig_bits=None, rmode=None, use_gpu=False):
    """Index of minimum absolute value"""
    if exp_bits is not None and sig_bits is not None:
        if exp_bits == 23 and sig_bits == 52:
            precision = 'fp64'
        elif exp_bits == 8 and sig_bits == 23:
            precision = 'fp32'
        else:
            precision = {'exp_bits': exp_bits, 'sig_bits': sig_bits, 'rmode': rmode or 1}
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    is_complex = np.iscomplexobj(x)
    dtype = torch.complex128 if is_complex else torch.float64
    x = torch.as_tensor(x, dtype=dtype, device=device)
    if x.dim() != 1:
        raise ValueError("x must be a 1D vector")
    native_real, native_complex = get_dtype(precision)
    native_dtype = native_complex if is_complex else native_real
    if is_complex and precision in ['half', 'bf16']:
        native_dtype = None
    if native_dtype is not None:
        x = x.to(native_dtype)
        idx = torch.argmin(torch.abs(x))
    else:
        x = rounding(x, precision, use_gpu)
        idx = torch.argmin(torch.abs(x))
    return idx.item()

# Level 2 BLAS: Matrix-Vector Operations
def low_prec_mv(A, x, precision, use_gpu, conj=False):
    """Low precision matrix-vector multiply with intermediate rounding"""
    result = torch.zeros(A.shape[0], dtype=A.dtype, device=A.device)
    for row in range(A.shape[0]):
        acc = torch.zeros_like(result[0])
        for col in range(A.shape[1]):
            a_val = A[row, col]
            if conj:
                a_val = torch.conj(a_val)
            prod = rounding(a_val * x[col], precision, use_gpu)
            acc = rounding(acc + prod, precision, use_gpu)
        result[row] = acc
    return result

def gemv(alpha, A, x, beta, y, trans='N', precision='fp64', exp_bits=None, sig_bits=None, rmode=None, use_gpu=False):
    """y = alpha * op(A) * x + beta * y, op(A) = A or A^T or A^H"""
    if exp_bits is not None and sig_bits is not None:
        if exp_bits == 23 and sig_bits == 52:
            precision = 'fp64'
        elif exp_bits == 8 and sig_bits == 23:
            precision = 'fp32'
        else:
            precision = {'exp_bits': exp_bits, 'sig_bits': sig_bits, 'rmode': rmode or 1}
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    is_complex = np.iscomplexobj(A) or np.iscomplexobj(x) or np.iscomplexobj(y) or np.iscomplexobj(alpha) or np.iscomplexobj(beta)
    dtype = torch.complex128 if is_complex else torch.float64
    A = torch.as_tensor(A, dtype=dtype, device=device)
    x = torch.as_tensor(x, dtype=dtype, device=device)
    y = torch.as_tensor(y, dtype=dtype, device=device)
    alpha = torch.tensor(alpha, dtype=dtype, device=device)
    beta = torch.tensor(beta, dtype=dtype, device=device)
    if A.dim() != 2 or x.dim() != 1 or y.dim() != 1:
        raise ValueError("A must be 2D, x and y must be 1D")
    m, n = A.shape
    if trans == 'N':
        if n != x.shape[0] or m != y.shape[0]:
            raise ValueError("Incompatible dimensions for A*x")
    else:
        if m != x.shape[0] or n != y.shape[0]:
            raise ValueError("Incompatible dimensions for op(A)*x")
    native_real, native_complex = get_dtype(precision)
    native_dtype = native_complex if is_complex else native_real
    if is_complex and precision in ['half', 'bf16']:
        native_dtype = None
    if native_dtype is not None:
        alpha = alpha.to(native_dtype)
        A = A.to(native_dtype)
        x = x.to(native_dtype)
        beta = beta.to(native_dtype)
        y = y.to(native_dtype)
        opA = A if trans == 'N' else (A.T if trans == 'T' else A.conj().T)
        mv = torch.matmul(opA, x)
        result = alpha * mv + beta * y
        result = result.to(dtype)
    else:
        alpha = rounding(alpha, precision, use_gpu)
        beta = rounding(beta, precision, use_gpu)
        A = rounding(A, precision, use_gpu)
        x = rounding(x, precision, use_gpu)
        y = rounding(y, precision, use_gpu)
        conj_A = (trans == 'C')
        opA = A if trans == 'N' else A.T
        mv = low_prec_mv(opA, x, precision, use_gpu, conj=conj_A)
        alpha_mv = rounding(alpha * mv, precision, use_gpu)
        beta_y = rounding(beta * y, precision, use_gpu)
        result = rounding(alpha_mv + beta_y, precision, use_gpu)
    return result

def gbmv(alpha, A, x, beta, y, kl, ku, trans='N', precision='fp64', exp_bits=None, sig_bits=None, rmode=None, use_gpu=False):
    """y = alpha * op(A) * x + beta * y for band matrix A"""
    if exp_bits is not None and sig_bits is not None:
        if exp_bits == 23 and sig_bits == 52:
            precision = 'fp64'
        elif exp_bits == 8 and sig_bits == 23:
            precision = 'fp32'
        else:
            precision = {'exp_bits': exp_bits, 'sig_bits': sig_bits, 'rmode': rmode or 1}
    return gemv(alpha, A, x, beta, y, trans, precision=precision, use_gpu=use_gpu)

def symv(alpha, A, x, beta, y, uplo='U', precision='fp64', exp_bits=None, sig_bits=None, rmode=None, use_gpu=False):
    """y = alpha * A * x + beta * y, A symmetric"""
    if exp_bits is not None and sig_bits is not None:
        if exp_bits == 23 and sig_bits == 52:
            precision = 'fp64'
        elif exp_bits == 8 and sig_bits == 23:
            precision = 'fp32'
        else:
            precision = {'exp_bits': exp_bits, 'sig_bits': sig_bits, 'rmode': rmode or 1}
    return gemv(alpha, A, x, beta, y, 'N', precision=precision, use_gpu=use_gpu)

def sbmv(alpha, A, x, beta, y, k, uplo='U', precision='fp64', exp_bits=None, sig_bits=None, rmode=None, use_gpu=False):
    """y = alpha * A * x + beta * y, A symmetric band"""
    if exp_bits is not None and sig_bits is not None:
        if exp_bits == 23 and sig_bits == 52:
            precision = 'fp64'
        elif exp_bits == 8 and sig_bits == 23:
            precision = 'fp32'
        else:
            precision = {'exp_bits': exp_bits, 'sig_bits': sig_bits, 'rmode': rmode or 1}
    return gemv(alpha, A, x, beta, y, 'N', precision=precision, use_gpu=use_gpu)

def hemv(alpha, A, x, beta, y, uplo='U', precision='fp64', exp_bits=None, sig_bits=None, rmode=None, use_gpu=False):
    """y = alpha * A * x + beta * y, A Hermitian"""
    if exp_bits is not None and sig_bits is not None:
        if exp_bits == 23 and sig_bits == 52:
            precision = 'fp64'
        elif exp_bits == 8 and sig_bits == 23:
            precision = 'fp32'
        else:
            precision = {'exp_bits': exp_bits, 'sig_bits': sig_bits, 'rmode': rmode or 1}
    return gemv(alpha, A, x, beta, y, 'N', precision=precision, use_gpu=use_gpu)

def hbmv(alpha, A, x, beta, y, k, uplo='U', precision='fp64', exp_bits=None, sig_bits=None, rmode=None, use_gpu=False):
    """y = alpha * A * x + beta * y, A Hermitian band"""
    if exp_bits is not None and sig_bits is not None:
        if exp_bits == 23 and sig_bits == 52:
            precision = 'fp64'
        elif exp_bits == 8 and sig_bits == 23:
            precision = 'fp32'
        else:
            precision = {'exp_bits': exp_bits, 'sig_bits': sig_bits, 'rmode': rmode or 1}
    return gemv(alpha, A, x, beta, y, 'N', precision=precision, use_gpu=use_gpu)

def spmv(alpha, A, x, beta, y, uplo='U', precision='fp64', exp_bits=None, sig_bits=None, rmode=None, use_gpu=False):
    """y = alpha * A * x + beta * y, A symmetric packed"""
    if exp_bits is not None and sig_bits is not None:
        if exp_bits == 23 and sig_bits == 52:
            precision = 'fp64'
        elif exp_bits == 8 and sig_bits == 23:
            precision = 'fp32'
        else:
            precision = {'exp_bits': exp_bits, 'sig_bits': sig_bits, 'rmode': rmode or 1}
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    A = torch.as_tensor(A, dtype=torch.float64, device=device)
    x = torch.as_tensor(x, dtype=A.dtype, device=device)
    y = torch.as_tensor(y, dtype=A.dtype, device=device)
    n = int((np.sqrt(8 * A.shape[0] + 1) - 1) / 2)
    A_dense = torch.zeros((n, n), dtype=A.dtype, device=device)
    idx = 0
    for i in range(n):
        start = i if uplo == 'U' else 0
        end = n if uplo == 'U' else i + 1
        for j in range(start, end):
            A_dense[i, j] = A[idx]
            if i != j:
                A_dense[j, i] = A[idx]
            idx += 1
    return gemv(alpha, A_dense, x, beta, y, 'N', precision=precision, use_gpu=use_gpu)

def hpmv(alpha, A, x, beta, y, uplo='U', precision='fp64', exp_bits=None, sig_bits=None, rmode=None, use_gpu=False):
    """y = alpha * A * x + beta * y, A Hermitian packed"""
    if exp_bits is not None and sig_bits is not None:
        if exp_bits == 23 and sig_bits == 52:
            precision = 'fp64'
        elif exp_bits == 8 and sig_bits == 23:
            precision = 'fp32'
        else:
            precision = {'exp_bits': exp_bits, 'sig_bits': sig_bits, 'rmode': rmode or 1}
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    A = torch.as_tensor(A, dtype=torch.complex128, device=device)
    x = torch.as_tensor(x, dtype=A.dtype, device=device)
    y = torch.as_tensor(y, dtype=A.dtype, device=device)
    n = int((np.sqrt(8 * A.shape[0] + 1) - 1) / 2)
    A_dense = torch.zeros((n, n), dtype=A.dtype, device=device)
    idx = 0
    for i in range(n):
        start = i if uplo == 'U' else 0
        end = n if uplo == 'U' else i + 1
        for j in range(start, end):
            A_dense[i, j] = A[idx]
            if i != j:
                A_dense[j, i] = torch.conj(A[idx])
            idx += 1
    return gemv(alpha, A_dense, x, beta, y, 'N', precision=precision, use_gpu=use_gpu)

def trmv(A, x, uplo='U', trans='N', diag='N', precision='fp64', exp_bits=None, sig_bits=None, rmode=None, use_gpu=False):
    """x = op(A) * x, A triangular"""
    if exp_bits is not None and sig_bits is not None:
        if exp_bits == 23 and sig_bits == 52:
            precision = 'fp64'
        elif exp_bits == 8 and sig_bits == 23:
            precision = 'fp32'
        else:
            precision = {'exp_bits': exp_bits, 'sig_bits': sig_bits, 'rmode': rmode or 1}
    return gemv(1.0, A, x, 0.0, torch.zeros_like(x), trans, precision=precision, use_gpu=use_gpu)

def low_prec_trsv(A, b, uplo='U', trans='N', diag='N', precision='fp64', use_gpu=False):
    """Low precision triangular solve with intermediate rounding"""
    n = A.shape[0]
    x = torch.zeros_like(b)
    unit = (diag == 'U')
    if trans == 'N':
        # Forward solve for lower, back for upper
        upper = (uplo == 'U')
        if upper:
            # Back substitution
            for i in range(n-1, -1, -1):
                sum_ = torch.zeros_like(x[0])
                for j in range(i+1, n):
                    prod = rounding(A[i, j] * x[j], precision, use_gpu)
                    sum_ = rounding(sum_ + prod, precision, use_gpu)
                div = rounding(A[i, i], precision, use_gpu) if not unit else torch.tensor(1.0, dtype=A.dtype, device=A.device)
                diff = rounding(b[i] - sum_, precision, use_gpu)
                x[i] = rounding(diff / div, precision, use_gpu)
        else:
            # Forward substitution
            for i in range(n):
                sum_ = torch.zeros_like(x[0])
                for j in range(i):
                    prod = rounding(A[i, j] * x[j], precision, use_gpu)
                    sum_ = rounding(sum_ + prod, precision, use_gpu)
                div = rounding(A[i, i], precision, use_gpu) if not unit else torch.tensor(1.0, dtype=A.dtype, device=A.device)
                diff = rounding(b[i] - sum_, precision, use_gpu)
                x[i] = rounding(diff / div, precision, use_gpu)
    else:
        # Transpose, swap upper/lower
        upper = (uplo == 'U')
        if upper:
            # Forward for trans upper = lower non trans
            for i in range(n):
                sum_ = torch.zeros_like(x[0])
                for j in range(i):
                    prod = rounding(A[j, i] * x[j], precision, use_gpu)  # A.T[i,j] = A[j,i]
                    sum_ = rounding(sum_ + prod, precision, use_gpu)
                div = rounding(A[i, i], precision, use_gpu) if not unit else torch.tensor(1.0, dtype=A.dtype, device=A.device)
                diff = rounding(b[i] - sum_, precision, use_gpu)
                x[i] = rounding(diff / div, precision, use_gpu)
        else:
            # Back for trans lower = upper non trans
            for i in range(n-1, -1, -1):
                sum_ = torch.zeros_like(x[0])
                for j in range(i+1, n):
                    prod = rounding(A[j, i] * x[j], precision, use_gpu)  # A.T[i,j] = A[j,i]
                    sum_ = rounding(sum_ + prod, precision, use_gpu)
                div = rounding(A[i, i], precision, use_gpu) if not unit else torch.tensor(1.0, dtype=A.dtype, device=A.device)
                diff = rounding(b[i] - sum_, precision, use_gpu)
                x[i] = rounding(diff / div, precision, use_gpu)
    return x

def trsv(A, b, uplo='U', trans='N', diag='N', precision='fp64', exp_bits=None, sig_bits=None, rmode=None, use_gpu=False):
    """Solve op(A) * x = b, A triangular"""
    if exp_bits is not None and sig_bits is not None:
        if exp_bits == 23 and sig_bits == 52:
            precision = 'fp64'
        elif exp_bits == 8 and sig_bits == 23:
            precision = 'fp32'
        else:
            precision = {'exp_bits': exp_bits, 'sig_bits': sig_bits, 'rmode': rmode or 1}
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    is_complex = np.iscomplexobj(A) or np.iscomplexobj(b)
    dtype = torch.complex128 if is_complex else torch.float64
    A = torch.as_tensor(A, dtype=dtype, device=device)
    b = torch.as_tensor(b, dtype=dtype, device=device)
    if A.dim() != 2 or b.dim() != 1 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square 2D, b must be 1D")
    native_real, native_complex = get_dtype(precision)
    native_dtype = native_complex if is_complex else native_real
    if is_complex and precision in ['half', 'bf16']:
        native_dtype = None
    if native_dtype is not None:
        A = A.to(native_dtype)
        b = b.to(native_dtype)
        opA = A if trans == 'N' else A.T
        x = torch.linalg.solve_triangular(opA, b, upper=(uplo == 'U'), unitriangular=(diag == 'U'))
        x = x.to(dtype)
    else:
        A = rounding(A, precision, use_gpu)
        b = rounding(b, precision, use_gpu)
        x = low_prec_trsv(A, b, uplo, trans, diag, precision, use_gpu)
    return x

def tbmv(A, x, k, uplo='U', trans='N', diag='N', precision='fp64', exp_bits=None, sig_bits=None, rmode=None, use_gpu=False):
    """x = op(A) * x, A triangular band"""
    if exp_bits is not None and sig_bits is not None:
        if exp_bits == 23 and sig_bits == 52:
            precision = 'fp64'
        elif exp_bits == 8 and sig_bits == 23:
            precision = 'fp32'
        else:
            precision = {'exp_bits': exp_bits, 'sig_bits': sig_bits, 'rmode': rmode or 1}
    return trmv(A, x, uplo, trans, diag, precision=precision, use_gpu=use_gpu)

def tbsv(A, b, k, uplo='U', trans='N', diag='N', precision='fp64', exp_bits=None, sig_bits=None, rmode=None, use_gpu=False):
    """Solve op(A) * x = b, A triangular band"""
    if exp_bits is not None and sig_bits is not None:
        if exp_bits == 23 and sig_bits == 52:
            precision = 'fp64'
        elif exp_bits == 8 and sig_bits == 23:
            precision = 'fp32'
        else:
            precision = {'exp_bits': exp_bits, 'sig_bits': sig_bits, 'rmode': rmode or 1}
    return trsv(A, b, uplo, trans, diag, precision=precision, use_gpu=use_gpu)

def tpmv(A, x, uplo='U', trans='N', diag='N', precision='fp64', exp_bits=None, sig_bits=None, rmode=None, use_gpu=False):
    """x = op(A) * x, A triangular packed"""
    if exp_bits is not None and sig_bits is not None:
        if exp_bits == 23 and sig_bits == 52:
            precision = 'fp64'
        elif exp_bits == 8 and sig_bits == 23:
            precision = 'fp32'
        else:
            precision = {'exp_bits': exp_bits, 'sig_bits': sig_bits, 'rmode': rmode or 1}
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    is_complex = np.iscomplexobj(A) or np.iscomplexobj(x)
    dtype = torch.complex128 if is_complex else torch.float64
    A = torch.as_tensor(A, dtype=dtype, device=device)
    x = torch.as_tensor(x, dtype=dtype, device=device)
    n = int((np.sqrt(8 * A.shape[0] + 1) - 1) / 2)
    A_dense = torch.zeros((n, n), dtype=dtype, device=device)
    idx = 0
    for i in range(n):
        start = i if uplo == 'U' else 0
        end = n if uplo == 'U' else i + 1
        for j in range(start, end):
            A_dense[i, j] = A[idx]
            idx += 1
    return trmv(A_dense, x, uplo, trans, diag, precision=precision, use_gpu=use_gpu)

def tpsv(A, b, uplo='U', trans='N', diag='N', precision='fp64', exp_bits=None, sig_bits=None, rmode=None, use_gpu=False):
    """Solve op(A) * x = b, A triangular packed"""
    if exp_bits is not None and sig_bits is not None:
        if exp_bits == 23 and sig_bits == 52:
            precision = 'fp64'
        elif exp_bits == 8 and sig_bits == 23:
            precision = 'fp32'
        else:
            precision = {'exp_bits': exp_bits, 'sig_bits': sig_bits, 'rmode': rmode or 1}
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    is_complex = np.iscomplexobj(A) or np.iscomplexobj(b)
    dtype = torch.complex128 if is_complex else torch.float64
    A = torch.as_tensor(A, dtype=dtype, device=device)
    b = torch.as_tensor(b, dtype=dtype, device=device)
    n = int((np.sqrt(8 * A.shape[0] + 1) - 1) / 2)
    A_dense = torch.zeros((n, n), dtype=dtype, device=device)
    idx = 0
    for i in range(n):
        start = i if uplo == 'U' else 0
        end = n if uplo == 'U' else i + 1
        for j in range(start, end):
            A_dense[i, j] = A[idx]
            idx += 1
    return trsv(A_dense, b, uplo, trans, diag, precision=precision, use_gpu=use_gpu)

def low_prec_outer(x, y, precision, use_gpu, conj_y=False):
    """Low precision outer product with intermediate rounding"""
    m = x.shape[0]
    n = y.shape[0]
    result = torch.zeros(m, n, dtype=x.dtype, device=x.device)
    for i in range(m):
        for j in range(n):
            y_val = torch.conj(y[j]) if conj_y else y[j]
            prod = rounding(x[i] * y_val, precision, use_gpu)
            result[i, j] = prod
    return result

def ger(alpha, x, y, A, precision='fp64', exp_bits=None, sig_bits=None, rmode=None, use_gpu=False):
    """A = A + alpha * x * y^T"""
    if exp_bits is not None and sig_bits is not None:
        if exp_bits == 23 and sig_bits == 52:
            precision = 'fp64'
        elif exp_bits == 8 and sig_bits == 23:
            precision = 'fp32'
        else:
            precision = {'exp_bits': exp_bits, 'sig_bits': sig_bits, 'rmode': rmode or 1}
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    is_complex = np.iscomplexobj(A) or np.iscomplexobj(x) or np.iscomplexobj(y) or np.iscomplexobj(alpha)
    dtype = torch.complex128 if is_complex else torch.float64
    A = torch.as_tensor(A, dtype=dtype, device=device)
    x = torch.as_tensor(x, dtype=dtype, device=device)
    y = torch.as_tensor(y, dtype=dtype, device=device)
    alpha = torch.tensor(alpha, dtype=dtype, device=device)
    if A.dim() != 2 or x.dim() != 1 or y.dim() != 1 or A.shape[0] != x.shape[0] or A.shape[1] != y.shape[0]:
        raise ValueError("Incompatible dimensions")
    native_real, native_complex = get_dtype(precision)
    native_dtype = native_complex if is_complex else native_real
    if is_complex and precision in ['half', 'bf16']:
        native_dtype = None
    if native_dtype is not None:
        alpha = alpha.to(native_dtype)
        x = x.to(native_dtype)
        y = y.to(native_dtype)
        A = A.to(native_dtype)
        result = A + alpha * torch.outer(x, y)
        result = result.to(dtype)
    else:
        alpha = rounding(alpha, precision, use_gpu)
        x = rounding(x, precision, use_gpu)
        y = rounding(y, precision, use_gpu)
        A = rounding(A, precision, use_gpu)
        outer = low_prec_outer(x, y, precision, use_gpu, conj_y=False)
        alpha_outer = rounding(alpha * outer, precision, use_gpu)
        result = rounding(A + alpha_outer, precision, use_gpu)
    return result

def syr(alpha, x, A, uplo='U', precision='fp64', exp_bits=None, sig_bits=None, rmode=None, use_gpu=False):
    """A = A + alpha * x * x^T, A symmetric"""
    if exp_bits is not None and sig_bits is not None:
        if exp_bits == 23 and sig_bits == 52:
            precision = 'fp64'
        elif exp_bits == 8 and sig_bits == 23:
            precision = 'fp32'
        else:
            precision = {'exp_bits': exp_bits, 'sig_bits': sig_bits, 'rmode': rmode or 1}
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    A = torch.as_tensor(A, dtype=torch.float64, device=device)
    x = torch.as_tensor(x, dtype=A.dtype, device=device)
    alpha = torch.tensor(alpha, dtype=A.dtype, device=device)
    if A.dim() != 2 or x.dim() != 1 or A.shape[0] != A.shape[1] or A.shape[0] != x.shape[0]:
        raise ValueError("Incompatible dimensions")
    native_real, native_complex = get_dtype(precision)
    native_dtype = native_real
    if native_dtype is not None:
        alpha = alpha.to(native_dtype)
        x = x.to(native_dtype)
        A = A.to(native_dtype)
        result = A + alpha * torch.outer(x, x)
        result = result.to(torch.float64)
    else:
        alpha = rounding(alpha, precision, use_gpu)
        x = rounding(x, precision, use_gpu)
        A = rounding(A, precision, use_gpu)
        outer = low_prec_outer(x, x, precision, use_gpu, conj_y=False)
        alpha_outer = rounding(alpha * outer, precision, use_gpu)
        result = rounding(A + alpha_outer, precision, use_gpu)
    return result

def spr(alpha, x, A, uplo='U', precision='fp64', exp_bits=None, sig_bits=None, rmode=None, use_gpu=False):
    """A = A + alpha * x * x^T, A symmetric packed"""
    if exp_bits is not None and sig_bits is not None:
        if exp_bits == 23 and sig_bits == 52:
            precision = 'fp64'
        elif exp_bits == 8 and sig_bits == 23:
            precision = 'fp32'
        else:
            precision = {'exp_bits': exp_bits, 'sig_bits': sig_bits, 'rmode': rmode or 1}
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    A = torch.as_tensor(A, dtype=torch.float64, device=device)
    x = torch.as_tensor(x, dtype=A.dtype, device=device)
    n = int((np.sqrt(8 * A.shape[0] + 1) - 1) / 2)
    A_dense = torch.zeros((n, n), dtype=A.dtype, device=device)
    idx = 0
    for i in range(n):
        start = i if uplo == 'U' else 0
        end = n if uplo == 'U' else i + 1
        for j in range(start, end):
            A_dense[i, j] = A[idx]
            if i != j:
                A_dense[j, i] = A[idx]
            idx += 1
    A_new = syr(alpha, x, A_dense, uplo, precision=precision, use_gpu=use_gpu)
    A_packed = torch.zeros_like(A)
    idx = 0
    for i in range(n):
        start = i if uplo == 'U' else 0
        end = n if uplo == 'U' else i + 1
        for j in range(start, end):
            A_packed[idx] = A_new[i, j]
            idx += 1
    return A_packed

def syr2(alpha, x, y, A, uplo='U', precision='fp64', exp_bits=None, sig_bits=None, rmode=None, use_gpu=False):
    """A = A + alpha * x * y^T + alpha * y * x^T, A symmetric"""
    if exp_bits is not None and sig_bits is not None:
        if exp_bits == 23 and sig_bits == 52:
            precision = 'fp64'
        elif exp_bits == 8 and sig_bits == 23:
            precision = 'fp32'
        else:
            precision = {'exp_bits': exp_bits, 'sig_bits': sig_bits, 'rmode': rmode or 1}
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    A = torch.as_tensor(A, dtype=torch.float64, device=device)
    x = torch.as_tensor(x, dtype=A.dtype, device=device)
    y = torch.as_tensor(y, dtype=A.dtype, device=device)
    alpha = torch.tensor(alpha, dtype=A.dtype, device=device)
    if A.dim() != 2 or x.dim() != 1 or y.dim() != 1 or A.shape[0] != A.shape[1] or A.shape[0] != x.shape[0] or A.shape[0] != y.shape[0]:
        raise ValueError("Incompatible dimensions")
    native_real, native_complex = get_dtype(precision)
    native_dtype = native_real
    if native_dtype is not None:
        alpha = alpha.to(native_dtype)
        x = x.to(native_dtype)
        y = y.to(native_dtype)
        A = A.to(native_dtype)
        result = A + alpha * (torch.outer(x, y) + torch.outer(y, x))
        result = result.to(torch.float64)
    else:
        alpha = rounding(alpha, precision, use_gpu)
        x = rounding(x, precision, use_gpu)
        y = rounding(y, precision, use_gpu)
        A = rounding(A, precision, use_gpu)
        outer1 = low_prec_outer(x, y, precision, use_gpu)
        outer2 = low_prec_outer(y, x, precision, use_gpu)
        sum_outer = rounding(outer1 + outer2, precision, use_gpu)
        alpha_sum = rounding(alpha * sum_outer, precision, use_gpu)
        result = rounding(A + alpha_sum, precision, use_gpu)
    return result

def spr2(alpha, x, y, A, uplo='U', precision='fp64', exp_bits=None, sig_bits=None, rmode=None, use_gpu=False):
    """A = A + alpha * x * y^T + alpha * y * x^T, A symmetric packed"""
    if exp_bits is not None and sig_bits is not None:
        if exp_bits == 23 and sig_bits == 52:
            precision = 'fp64'
        elif exp_bits == 8 and sig_bits == 23:
            precision = 'fp32'
        else:
            precision = {'exp_bits': exp_bits, 'sig_bits': sig_bits, 'rmode': rmode or 1}
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    A = torch.as_tensor(A, dtype=torch.float64, device=device)
    x = torch.as_tensor(x, dtype=A.dtype, device=device)
    y = torch.as_tensor(y, dtype=A.dtype, device=device)
    n = int((np.sqrt(8 * A.shape[0] + 1) - 1) / 2)
    A_dense = torch.zeros((n, n), dtype=A.dtype, device=device)
    idx = 0
    for i in range(n):
        start = i if uplo == 'U' else 0
        end = n if uplo == 'U' else i + 1
        for j in range(start, end):
            A_dense[i, j] = A[idx]
            if i != j:
                A_dense[j, i] = A[idx]
            idx += 1
    A_new = syr2(alpha, x, y, A_dense, uplo, precision=precision, use_gpu=use_gpu)
    A_packed = torch.zeros_like(A)
    idx = 0
    for i in range(n):
        start = i if uplo == 'U' else 0
        end = n if uplo == 'U' else i + 1
        for j in range(start, end):
            A_packed[idx] = A_new[i, j]
            idx += 1
    return A_packed

def her(alpha, x, A, uplo='U', precision='fp64', exp_bits=None, sig_bits=None, rmode=None, use_gpu=False):
    """A = A + alpha * x * x^H, A Hermitian"""
    if exp_bits is not None and sig_bits is not None:
        if exp_bits == 23 and sig_bits == 52:
            precision = 'fp64'
        elif exp_bits == 8 and sig_bits == 23:
            precision = 'fp32'
        else:
            precision = {'exp_bits': exp_bits, 'sig_bits': sig_bits, 'rmode': rmode or 1}
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    A = torch.as_tensor(A, dtype=torch.complex128, device=device)
    x = torch.as_tensor(x, dtype=A.dtype, device=device)
    alpha = torch.tensor(alpha, dtype=torch.float64, device=device)
    if A.dim() != 2 or x.dim() != 1 or A.shape[0] != A.shape[1] or A.shape[0] != x.shape[0]:
        raise ValueError("Incompatible dimensions")
    native_real, native_complex = get_dtype(precision)
    native_dtype = native_complex
    if precision in ['half', 'bf16']:
        native_dtype = None
    if native_dtype is not None:
        alpha = alpha.to(native_real)  # alpha real
        x = x.to(native_dtype)
        A = A.to(native_dtype)
        result = A + alpha * torch.outer(x, torch.conj(x))
        result = result.to(torch.complex128)
    else:
        alpha = rounding(alpha, precision, use_gpu)
        x = rounding(x, precision, use_gpu)
        A = rounding(A, precision, use_gpu)
        outer = low_prec_outer(x, x, precision, use_gpu, conj_y=True)
        alpha_outer = rounding(alpha * outer, precision, use_gpu)
        result = rounding(A + alpha_outer, precision, use_gpu)
    return result

def hpr(alpha, x, A, uplo='U', precision='fp64', exp_bits=None, sig_bits=None, rmode=None, use_gpu=False):
    """A = A + alpha * x * x^H, A Hermitian packed"""
    if exp_bits is not None and sig_bits is not None:
        if exp_bits == 23 and sig_bits == 52:
            precision = 'fp64'
        elif exp_bits == 8 and sig_bits == 23:
            precision = 'fp32'
        else:
            precision = {'exp_bits': exp_bits, 'sig_bits': sig_bits, 'rmode': rmode or 1}
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    A = torch.as_tensor(A, dtype=torch.complex128, device=device)
    x = torch.as_tensor(x, dtype=A.dtype, device=device)
    n = int((np.sqrt(8 * A.shape[0] + 1) - 1) / 2)
    A_dense = torch.zeros((n, n), dtype=A.dtype, device=device)
    idx = 0
    for i in range(n):
        start = i if uplo == 'U' else 0
        end = n if uplo == 'U' else i + 1
        for j in range(start, end):
            A_dense[i, j] = A[idx]
            if i != j:
                A_dense[j, i] = torch.conj(A[idx])
            idx += 1
    A_new = her(alpha, x, A_dense, uplo, precision=precision, use_gpu=use_gpu)
    A_packed = torch.zeros_like(A)
    idx = 0
    for i in range(n):
        start = i if uplo == 'U' else 0
        end = n if uplo == 'U' else i + 1
        for j in range(start, end):
            A_packed[idx] = A_new[i, j]
            idx += 1
    return A_packed

def her2(alpha, x, y, A, uplo='U', precision='fp64', exp_bits=None, sig_bits=None, rmode=None, use_gpu=False):
    """A = A + alpha * x * y^H + alpha * y * x^H, A Hermitian"""
    if exp_bits is not None and sig_bits is not None:
        if exp_bits == 23 and sig_bits == 52:
            precision = 'fp64'
        elif exp_bits == 8 and sig_bits == 23:
            precision = 'fp32'
        else:
            precision = {'exp_bits': exp_bits, 'sig_bits': sig_bits, 'rmode': rmode or 1}
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    A = torch.as_tensor(A, dtype=torch.complex128, device=device)
    x = torch.as_tensor(x, dtype=A.dtype, device=device)
    y = torch.as_tensor(y, dtype=A.dtype, device=device)
    alpha = torch.tensor(alpha, dtype=A.dtype, device=device)
    if A.dim() != 2 or x.dim() != 1 or y.dim() != 1 or A.shape[0] != A.shape[1] or A.shape[0] != x.shape[0] or A.shape[0] != y.shape[0]:
        raise ValueError("Incompatible dimensions")
    native_real, native_complex = get_dtype(precision)
    native_dtype = native_complex
    if precision in ['half', 'bf16']:
        native_dtype = None
    if native_dtype is not None:
        alpha = alpha.to(native_dtype)
        x = x.to(native_dtype)
        y = y.to(native_dtype)
        A = A.to(native_dtype)
        result = A + alpha * (torch.outer(x, torch.conj(y)) + torch.outer(y, torch.conj(x)))
        result = result.to(torch.complex128)
    else:
        alpha = rounding(alpha, precision, use_gpu)
        x = rounding(x, precision, use_gpu)
        y = rounding(y, precision, use_gpu)
        A = rounding(A, precision, use_gpu)
        outer1 = low_prec_outer(x, y, precision, use_gpu, conj_y=True)
        outer2 = low_prec_outer(y, x, precision, use_gpu, conj_y=True)
        sum_outer = rounding(outer1 + outer2, precision, use_gpu)
        alpha_sum = rounding(alpha * sum_outer, precision, use_gpu)
        result = rounding(A + alpha_sum, precision, use_gpu)
    return result

def hpr2(alpha, x, y, A, uplo='U', precision='fp64', exp_bits=None, sig_bits=None, rmode=None, use_gpu=False):
    """A = A + alpha * x * y^H + alpha * y * x^H, A Hermitian packed"""
    if exp_bits is not None and sig_bits is not None:
        if exp_bits == 23 and sig_bits == 52:
            precision = 'fp64'
        elif exp_bits == 8 and sig_bits == 23:
            precision = 'fp32'
        else:
            precision = {'exp_bits': exp_bits, 'sig_bits': sig_bits, 'rmode': rmode or 1}
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    A = torch.as_tensor(A, dtype=torch.complex128, device=device)
    x = torch.as_tensor(x, dtype=A.dtype, device=device)
    y = torch.as_tensor(y, dtype=A.dtype, device=device)
    n = int((np.sqrt(8 * A.shape[0] + 1) - 1) / 2)
    A_dense = torch.zeros((n, n), dtype=A.dtype, device=device)
    idx = 0
    for i in range(n):
        start = i if uplo == 'U' else 0
        end = n if uplo == 'U' else i + 1
        for j in range(start, end):
            A_dense[i, j] = A[idx]
            if i != j:
                A_dense[j, i] = torch.conj(A[idx])
            idx += 1
    A_new = her2(alpha, x, y, A_dense, uplo, precision=precision, use_gpu=use_gpu)
    A_packed = torch.zeros_like(A)
    idx = 0
    for i in range(n):
        start = i if uplo == 'U' else 0
        end = n if uplo == 'U' else i + 1
        for j in range(start, end):
            A_packed[idx] = A_new[i, j]
            idx += 1
    return A_packed

# Level 3 BLAS: Matrix-Matrix Operations
def low_prec_mm(A, B, precision, use_gpu, conjA=False, conjB=False):
    """Low precision matrix-matrix multiply with intermediate rounding"""
    m, k = A.shape
    k2, n = B.shape
    if k != k2:
        raise ValueError("Incompatible dimensions for A * B")
    result = torch.zeros(m, n, dtype=A.dtype, device=A.device)
    for i in range(m):
        for j in range(n):
            acc = torch.zeros_like(result[0, 0])
            for l in range(k):
                a_val = torch.conj(A[i, l]) if conjA else A[i, l]
                b_val = torch.conj(B[l, j]) if conjB else B[l, j]
                prod = rounding(a_val * b_val, precision, use_gpu)
                acc = rounding(acc + prod, precision, use_gpu)
            result[i, j] = acc
    return result

def gemm(alpha, A, B, beta, C, transA='N', transB='N', precision='fp64', exp_bits=None, sig_bits=None, rmode=None, use_gpu=False):
    """C = alpha * op(A) * op(B) + beta * C"""
    if exp_bits is not None and sig_bits is not None:
        if exp_bits == 23 and sig_bits == 52:
            precision = 'fp64'
        elif exp_bits == 8 and sig_bits == 23:
            precision = 'fp32'
        else:
            precision = {'exp_bits': exp_bits, 'sig_bits': sig_bits, 'rmode': rmode or 1}
    
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    is_complex = np.iscomplexobj(A) or np.iscomplexobj(B) or np.iscomplexobj(C) or np.iscomplexobj(alpha) or np.iscomplexobj(beta)
    dtype = torch.complex128 if is_complex else x.dtype
    A = torch.as_tensor(A, dtype=dtype, device=device)
    B = torch.as_tensor(B, dtype=dtype, device=device)
    C = torch.as_tensor(C, dtype=dtype, device=device)
    alpha = torch.tensor(alpha, dtype=dtype, device=device)
    beta = torch.tensor(beta, dtype=dtype, device=device)
    if A.dim() != 2 or B.dim() != 2 or C.dim() != 2:
        raise ValueError("A, B, C must be 2D")
    opA = A if transA == 'N' else (A.T if transA == 'T' else A.conj().T)
    opB = B if transB == 'N' else (B.T if transB == 'T' else B.conj().T)
    if opA.shape[1] != opB.shape[0] or opA.shape[0] != C.shape[0] or opB.shape[1] != C.shape[1]:
        raise ValueError("Incompatible dimensions")
    native_real, native_complex = get_dtype(precision)
    native_dtype = native_complex if is_complex else native_real
    
    if is_complex and precision in ['half', 'bf16']:
        native_dtype = None
    if native_dtype is not None:
        alpha = alpha.to(native_dtype)
        opA = opA.to(native_dtype)
        opB = opB.to(native_dtype)
        beta = beta.to(native_dtype)
        C = C.to(native_dtype)
        mm = torch.matmul(opA, opB)
        result = alpha * mm + beta * C
        result = result.to(dtype)
    else:
        alpha = rounding(alpha, precision, use_gpu)
        beta = rounding(beta, precision, use_gpu)
        A = rounding(A, precision, use_gpu)
        B = rounding(B, precision, use_gpu)
        C = rounding(C, precision, use_gpu)
        conjA = (transA == 'C')
        conjB = (transB == 'C')
        opA = A if transA == 'N' else A.T
        opB = B if transB == 'N' else B.T
        mm = rounding(torch.matmul(opA, opB), precision, use_gpu)  # low_prec_mm(opA, opB, precision, use_gpu, conjA=conjA, conjB=conjB)
        alpha_mm = rounding(alpha * mm, precision, use_gpu)
        beta_C = rounding(beta * C, precision, use_gpu)
        result = rounding(alpha_mm + beta_C, precision, use_gpu)
    return result

def symm(alpha, A, B, beta, C, side='L', uplo='U', precision='fp64', exp_bits=None, sig_bits=None, rmode=None, use_gpu=False):
    """C = alpha * A * B + beta * C or alpha * B * A + beta * C, A symmetric"""
    if exp_bits is not None and sig_bits is not None:
        if exp_bits == 23 and sig_bits == 52:
            precision = 'fp64'
        elif exp_bits == 8 and sig_bits == 23:
            precision = 'fp32'
        else:
            precision = {'exp_bits': exp_bits, 'sig_bits': sig_bits, 'rmode': rmode or 1}
    transA = 'N' if side == 'L' else 'T'
    transB = 'N'
    if side == 'R':
        temp = A
        A = B
        B = temp
    return gemm(alpha, A, B, beta, C, transA=transA, transB=transB, precision=precision, use_gpu=use_gpu)

def hemm(alpha, A, B, beta, C, side='L', uplo='U', precision='fp64', exp_bits=None, sig_bits=None, rmode=None, use_gpu=False):
    """C = alpha * A * B + beta * C or alpha * B * A + beta * C, A Hermitian"""
    if exp_bits is not None and sig_bits is not None:
        if exp_bits == 23 and sig_bits == 52:
            precision = 'fp64'
        elif exp_bits == 8 and sig_bits == 23:
            precision = 'fp32'
        else:
            precision = {'exp_bits': exp_bits, 'sig_bits': sig_bits, 'rmode': rmode or 1}
    transA = 'N' if side == 'L' else 'C'
    transB = 'N'
    if side == 'R':
        temp = A
        A = B
        B = temp
    return gemm(alpha, A, B, beta, C, transA=transA, transB=transB, precision=precision, use_gpu=use_gpu)

def syrk(alpha, A, beta, C, trans='N', uplo='U', precision='fp64', exp_bits=None, sig_bits=None, rmode=None, use_gpu=False):
    """C = alpha * A * A^T + beta * C or alpha * A^T * A + beta * C, C symmetric"""
    if exp_bits is not None and sig_bits is not None:
        if exp_bits == 23 and sig_bits == 52:
            precision = 'fp64'
        elif exp_bits == 8 and sig_bits == 23:
            precision = 'fp32'
        else:
            precision = {'exp_bits': exp_bits, 'sig_bits': sig_bits, 'rmode': rmode or 1}
    transA = trans
    transB = 'T' if trans == 'N' else 'N'
    return gemm(alpha, A, A, beta, C, transA=transA, transB=transB, precision=precision, use_gpu=use_gpu)

def herk(alpha, A, beta, C, trans='N', uplo='U', precision='fp64', exp_bits=None, sig_bits=None, rmode=None, use_gpu=False):
    """C = alpha * A * A^H + beta * C or alpha * A^H * A + beta * C, C Hermitian"""
    if exp_bits is not None and sig_bits is not None:
        if exp_bits == 23 and sig_bits == 52:
            precision = 'fp64'
        elif exp_bits == 8 and sig_bits == 23:
            precision = 'fp32'
        else:
            precision = {'exp_bits': exp_bits, 'sig_bits': sig_bits, 'rmode': rmode or 1}
    transA = trans
    transB = 'C' if trans == 'N' else 'N'
    return gemm(alpha, A, A, beta, C, transA=transA, transB=transB, precision=precision, use_gpu=use_gpu)

def syr2k(alpha, A, B, beta, C, trans='N', uplo='U', precision='fp64', exp_bits=None, sig_bits=None, rmode=None, use_gpu=False):
    """C = alpha * A * B^T + alpha * B * A^T + beta * C, C symmetric"""
    if exp_bits is not None and sig_bits is not None:
        if exp_bits == 23 and sig_bits == 52:
            precision = 'fp64'
        elif exp_bits == 8 and sig_bits == 23:
            precision = 'fp32'
        else:
            precision = {'exp_bits': exp_bits, 'sig_bits': sig_bits, 'rmode': rmode or 1}
    if trans == 'N':
        mm1 = gemm(alpha, A, B, 0.0, torch.zeros_like(C), transA='N', transB='T', precision=precision, use_gpu=use_gpu)
        mm2 = gemm(alpha, B, A, 0.0, torch.zeros_like(C), transA='N', transB='T', precision=precision, use_gpu=use_gpu)
    else:
        mm1 = gemm(alpha, A, B, 0.0, torch.zeros_like(C), transA='T', transB='N', precision=precision, use_gpu=use_gpu)
        mm2 = gemm(alpha, B, A, 0.0, torch.zeros_like(C), transA='T', transB='N', precision=precision, use_gpu=use_gpu)
    sum_mm = rounding(mm1 + mm2, precision, use_gpu) if get_dtype(precision)[0] is None else mm1 + mm2
    beta_C = rounding(beta * C, precision, use_gpu) if get_dtype(precision)[0] is None else beta * C
    result = rounding(sum_mm + beta_C, precision, use_gpu) if get_dtype(precision)[0] is None else sum_mm + beta_C
    return result

def her2k(alpha, A, B, beta, C, trans='N', uplo='U', precision='fp64', exp_bits=None, sig_bits=None, rmode=None, use_gpu=False):
    """C = alpha * A * B^H + alpha * B * A^H + beta * C, C Hermitian"""
    if exp_bits is not None and sig_bits is not None:
        if exp_bits == 23 and sig_bits == 52:
            precision = 'fp64'
        elif exp_bits == 8 and sig_bits == 23:
            precision = 'fp32'
        else:
            precision = {'exp_bits': exp_bits, 'sig_bits': sig_bits, 'rmode': rmode or 1}
    if trans == 'N':
        mm1 = gemm(alpha, A, B, 0.0, torch.zeros_like(C), transA='N', transB='C', precision=precision, use_gpu=use_gpu)
        mm2 = gemm(alpha, B, A, 0.0, torch.zeros_like(C), transA='N', transB='C', precision=precision, use_gpu=use_gpu)
    else:
        mm1 = gemm(alpha, A, B, 0.0, torch.zeros_like(C), transA='C', transB='N', precision=precision, use_gpu=use_gpu)
        mm2 = gemm(alpha, B, A, 0.0, torch.zeros_like(C), transA='C', transB='N', precision=precision, use_gpu=use_gpu)
    sum_mm = rounding(mm1 + mm2, precision, use_gpu) if get_dtype(precision)[0] is None else mm1 + mm2
    beta_C = rounding(beta * C, precision, use_gpu) if get_dtype(precision)[0] is None else beta * C
    result = rounding(sum_mm + beta_C, precision, use_gpu) if get_dtype(precision)[0] is None else sum_mm + beta_C
    return result

def trmm(alpha, A, B, side='L', uplo='U', transA='N', diag='N', precision='fp64', exp_bits=None, sig_bits=None, rmode=None, use_gpu=False):
    """B = alpha * op(A) * B or alpha * B * op(A), A triangular"""
    if exp_bits is not None and sig_bits is not None:
        if exp_bits == 23 and sig_bits == 52:
            precision = 'fp64'
        elif exp_bits == 8 and sig_bits == 23:
            precision = 'fp32'
        else:
            precision = {'exp_bits': exp_bits, 'sig_bits': sig_bits, 'rmode': rmode or 1}
    if side == 'L':
        transB = 'N'
        return gemm(alpha, A, B, 0.0, torch.zeros_like(B), transA=transA, transB=transB, precision=precision, use_gpu=use_gpu)
    else:
        transB = transA
        transA = 'N'
        return gemm(alpha, B, A, 0.0, torch.zeros_like(B), transA=transA, transB=transB, precision=precision, use_gpu=use_gpu)

def low_prec_trsm(alpha, A, B, side='L', uplo='U', transA='N', diag='N', precision="fp64", use_gpu=False):
    """Low precision triangular solve for matrix B"""
    m, n = B.shape
    result = torch.zeros_like(B)
    for col in range(n):
        b = B[:, col]
        x = low_prec_trsv(A, b, uplo, transA, diag, precision, use_gpu)
        result[:, col] = x
    result = rounding(alpha * result, precision, use_gpu)
    return result

def trsm(alpha, A, B, side='L', uplo='U', transA='N', diag='N', precision='fp64', exp_bits=None, sig_bits=None, rmode=None, use_gpu=False):
    """Solve op(A) * X = alpha * B or X * op(A) = alpha * B, A triangular"""
    if exp_bits is not None and sig_bits is not None:
        if exp_bits == 23 and sig_bits == 52:
            precision = 'fp64'
        elif exp_bits == 8 and sig_bits == 23:
            precision = 'fp32'
        else:
            precision = {'exp_bits': exp_bits, 'sig_bits': sig_bits, 'rmode': rmode or 1}
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    is_complex = np.iscomplexobj(A) or np.iscomplexobj(B) or np.iscomplexobj(alpha)
    dtype = torch.complex128 if is_complex else torch.float64
    A = torch.as_tensor(A, dtype=dtype, device=device)
    B = torch.as_tensor(B, dtype=dtype, device=device)
    alpha = torch.tensor(alpha, dtype=dtype, device=device)
    if A.dim() != 2 or B.dim() != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square 2D, B must be 2D")
    native_real, native_complex = get_dtype(precision)
    native_dtype = native_complex if is_complex else native_real
    if is_complex and precision in ['half', 'bf16']:
        native_dtype = None
    if native_dtype is not None:
        alpha = alpha.to(native_dtype)
        A = A.to(native_dtype)
        B = B.to(native_dtype)
        opA = A if transA == 'N' else (A.T if transA == 'T' else A.conj().T)
        X = torch.linalg.solve_triangular(opA, alpha * B, upper=(uplo == 'U'), unitriangular=(diag == 'U'), left=(side == 'L'))
        X = X.to(dtype)
    else:
        A = rounding(A, precision, use_gpu)
        B = rounding(B, precision, use_gpu)
        alpha = rounding(alpha, precision, use_gpu)
        if side == 'L':
            X = low_prec_trsm(alpha, A, B, side, uplo, transA, diag, precision, use_gpu)
        else:
            # for right, X * op(A) = alpha * B, so trans to left
            transA_right = 'N' if transA == 'N' else ('T' if transA == 'C' else 'C')
            uplo_right = 'L' if uplo == 'U' else 'U'  # trans swaps uplo
            X = low_prec_trsm(alpha, A.T, B.T, 'L', uplo_right, transA_right, diag, precision, use_gpu).T
    return X

if __name__ == "__main__":
    device = torch.device('cpu') 

    torch.manual_seed(42)
    np.random.seed(42)

    # Test Level 1: axpy, dot, dotc
    n = 5
    x = torch.randn(n, dtype=torch.float64, device=device)
    y = torch.randn(n, dtype=torch.float64, device=device)
    x_c = torch.randn(n, dtype=torch.complex128, device=device) + 1j * torch.randn(n)
    y_c = torch.randn(n, dtype=torch.complex128, device=device) + 1j * torch.randn(n)
    alpha = 2.0

    print("Level 1 Tests:")
    axpy_fp64 = axpy(alpha, x, y, 'fp64')
    axpy_fp32 = axpy(alpha, x, y, 'fp32')
    print(f"axpy (fp64): {axpy_fp64[:3].numpy()}")
    print(f"axpy (fp32): {axpy_fp32[:3].numpy()}")

    dot_fp64 = dot(x, y, 'fp64')
    dot_fp32 = dot(x, y, 'fp32')
    print(f"dot (fp64): {dot_fp64:.6f}")
    print(f"dot (fp32): {dot_fp32:.6f}")

    dotc_fp64 = dotc(x_c, y_c, 'fp64')
    dotc_fp32 = dotc(x_c, y_c, 'fp32')
    print(f"dotc (fp64): {dotc_fp64:.6f}")
    print(f"dotc (fp32): {dotc_fp32:.6f}")

    # Test Level 2: gemv, her
    m, n = 4, 3
    A = torch.randn(m, n, dtype=torch.float64, device=device)
    A_c = torch.randn(m, m, dtype=torch.complex128, device=device)
    A_c = A_c + A_c.conj().T  # Make Hermitian
    x = torch.randn(n, dtype=torch.float64, device=device)
    y = torch.randn(m, dtype=torch.float64, device=device)
    x_c = torch.randn(m, dtype=torch.complex128, device=device)
    alpha, beta = 1.5, 0.5

    print("\nLevel 2 Tests:")
    gemv_fp64 = gemv(alpha, A, x, beta, y, 'N', 'fp64')
    gemv_fp32 = gemv(alpha, A, x, beta, y, 'N', 'fp32')
    print(f"gemv (fp64): {gemv_fp64[:3].numpy()}")
    print(f"gemv (fp32): {gemv_fp32[:3].numpy()}")

    her_fp64 = her(alpha, x_c, A_c, 'U', 'fp64')
    her_fp32 = her(alpha, x_c, A_c, 'U', 'fp32')
    print(f"her (fp64): \n{her_fp64[:2, :2].numpy()}")
    print(f"her (fp32): \n{her_fp32[:2, :2].numpy()}")

    # Test Level 3: gemm, herk
    m, n, k = 3, 3, 2
    A = torch.randn(m, k, dtype=torch.float64, device=device)
    B = torch.randn(k, n, dtype=torch.float64, device=device)
    C = torch.randn(m, n, dtype=torch.float64, device=device)
    A_c = torch.randn(m, k, dtype=torch.complex128, device=device)
    C_c = torch.randn(m, m, dtype=torch.complex128, device=device)
    C_c = C_c + C_c.conj().T  # Make Hermitian
    alpha, beta = 1.0, 0.5

    print("\nLevel 3 Tests:")
    gemm_fp64 = gemm(alpha, A, B, beta, C, 'N', 'N', 'fp64')
    gemm_fp32 = gemm(alpha, A, B, beta, C, 'N', 'N', 'fp32')
    print(f"gemm (fp64): \n{gemm_fp64.numpy()}")
    print(f"gemm (fp32): \n{gemm_fp32.numpy()}")

    herk_fp64 = herk(alpha, A_c, beta, C_c, 'N', 'U', 'fp64')
    herk_fp32 = herk(alpha, A_c, beta, C_c, 'N', 'U', 'fp32')
    print(f"herk (fp64): \n{herk_fp64.numpy()}")
    print(f"herk (fp32): \n{herk_fp32.numpy()}")

    # Additional tests for correctness and custom precision
    print("\nAdditional Tests for Correctness:")

    # Test dot with fp32 simulation vs native fp32
    n = 5
    x = torch.randn(n, dtype=torch.float64)
    y = torch.randn(n, dtype=torch.float64)
    dot_sim_fp32 = dot(x, y, precision='fp32')
    dot_custom_fp32 = dot(x, y, exp_bits=8, sig_bits=23, rmode=1)
    dot_native_fp32 = torch.sum(x.float() * y.float()).item()
    print(f"dot simulated fp32: {dot_sim_fp32:.6f}")
    print(f"dot custom fp32: {dot_custom_fp32:.6f}")
    print(f"dot native fp32: {dot_native_fp32:.6f}")

    # Test dot with half simulation vs native half
    dot_sim_half = dot(x, y, precision='half')

    dot_custom_half = dot(x, y, exp_bits=5, sig_bits=10, rmode=1)

    dot_native_half = torch.sum(x.half() * y.half()).item()

    print(f"dot simulated half: {dot_sim_half:.6f}")
    print(f"dot custom half: {dot_custom_half:.6f}")
    print(f"dot native half: {dot_native_half:.6f}")

    # Test gemm with fp32 simulation vs native fp32
    m, n, k = 3, 3, 2
    A = torch.randn(m, k, dtype=torch.float64)
    B = torch.randn(k, n, dtype=torch.float64)
    C = torch.randn(m, n, dtype=torch.float64)
    alpha, beta = 1.0, 0.5
    gemm_sim_fp32 = gemm(alpha, A, B, beta, C, 'N', 'N', 'fp32')
    gemm_custom_fp32 = gemm(alpha, A, B, beta, C, 'N', 'N', exp_bits=8, sig_bits=23, rmode=1)
    opA_native = A.float()
    opB_native = B.float()
    gemm_native_fp32 = alpha * torch.matmul(opA_native, opB_native) + beta * C.float()
    print(f"gemm simulated fp32:\n{gemm_sim_fp32.numpy()}")
    print(f"gemm custom fp32:\n{gemm_custom_fp32.numpy()}")
    print(f"gemm native fp32:\n{gemm_native_fp32.numpy()}")

    # Test with GPU if available
    if torch.cuda.is_available():
        print("\nTesting with GPU:")
        dot_gpu = dot(x, y, precision='fp32', use_gpu=True)
        print(f"dot on GPU fp32: {dot_gpu:.6f}")
    else:
        print("\nGPU not available for testing.")