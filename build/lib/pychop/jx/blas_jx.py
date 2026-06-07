import jax.numpy as jnp
import pychop
import logging

logging.basicConfig(level=logging.INFO)

pychop.backend('jax')

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
        'fp64': (jnp.float64, jnp.complex128),
        'fp32': (jnp.float32, jnp.complex64),
        'half': (jnp.float16, None),
        'bf16': (jnp.bfloat16, None),  # JAX supports bfloat16 natively
    }
    return d.get(precision, (None, None))

def chop(x, precision='q52'):
    """Recursive chop function with support for custom precision"""
    if not isinstance(x, jnp.ndarray):
        x = jnp.array(x)
    if precision == 'fp64':
        dtype = jnp.float64 if not jnp.iscomplexobj(x) else jnp.complex128
        return x.astype(dtype)
    
    elif precision == 'fp32':
        dtype = jnp.float32 if not jnp.iscomplexobj(x) else jnp.complex64
        return x.astype(dtype)
    
    if isinstance(precision, dict):
        config = precision
    else:
        config = precision_configs[precision]
    
    ch = pychop.LightChop(**config)

    return ch(x)

def rounding(x, precision):
    """Round array to specified precision"""
    return chop(x, precision=precision)

# Level 1 BLAS: Vector-Vector Operations
def axpy(alpha, x, y, precision='fp64', exp_bits=None, sig_bits=None, rmode=None):
    """y = alpha * x + y"""
    if exp_bits is not None and sig_bits is not None:
        if exp_bits == 23 and sig_bits == 52:
            precision = 'fp64'
        elif exp_bits == 8 and sig_bits == 23:
            precision = 'fp32'
        else:
            precision = {'exp_bits': exp_bits, 'sig_bits': sig_bits, 'rmode': rmode or 1}
    is_complex = jnp.iscomplexobj(alpha) or jnp.iscomplexobj(x) or jnp.iscomplexobj(y)
    dtype = jnp.complex128 if is_complex else x.dtype
    x = jnp.array(x, dtype=dtype)
    y = jnp.array(y, dtype=dtype)
    alpha = jnp.array(alpha, dtype=dtype)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("x and y must be 1D vectors of same length")
    native_real, native_complex = get_dtype(precision)
    native_dtype = native_complex if is_complex else native_real
    if is_complex and precision in ['half', 'bf16']:
        native_dtype = None
    if native_dtype is not None:
        alpha = alpha.astype(native_dtype)
        x = x.astype(native_dtype)
        y = y.astype(native_dtype)
        result = alpha * x + y
        result = result.astype(dtype)
    else:
        alpha = rounding(alpha, precision)
        x = rounding(x, precision)
        y = rounding(y, precision)
        mul = rounding(alpha * x, precision)
        result = rounding(mul + y, precision)
    return result

def scal(alpha, x, precision='fp64', exp_bits=None, sig_bits=None, rmode=None):
    """x = alpha * x"""
    if exp_bits is not None and sig_bits is not None:
        if exp_bits == 23 and sig_bits == 52:
            precision = 'fp64'
        elif exp_bits == 8 and sig_bits == 23:
            precision = 'fp32'
        else:
            precision = {'exp_bits': exp_bits, 'sig_bits': sig_bits, 'rmode': rmode or 1}
    is_complex = jnp.iscomplexobj(alpha) or jnp.iscomplexobj(x)
    dtype = jnp.complex128 if is_complex else x.dtype
    x = jnp.array(x, dtype=dtype)
    alpha = jnp.array(alpha, dtype=dtype)
    if x.ndim != 1:
        raise ValueError("x must be a 1D vector")
    native_real, native_complex = get_dtype(precision)
    native_dtype = native_complex if is_complex else native_real
    if is_complex and precision in ['half', 'bf16']:
        native_dtype = None
    if native_dtype is not None:
        alpha = alpha.astype(native_dtype)
        x = x.astype(native_dtype)
        result = alpha * x
        result = result.astype(dtype)
    else:
        alpha = rounding(alpha, precision)
        x = rounding(x, precision)
        result = rounding(alpha * x, precision)
    return result

def copy(x, y, precision='fp64', exp_bits=None, sig_bits=None, rmode=None):
    """y = x"""
    if exp_bits is not None and sig_bits is not None:
        if exp_bits == 23 and sig_bits == 52:
            precision = 'fp64'
        elif exp_bits == 8 and sig_bits == 23:
            precision = 'fp32'
        else:
            precision = {'exp_bits': exp_bits, 'sig_bits': sig_bits, 'rmode': rmode or 1}
    is_complex = jnp.iscomplexobj(x) or jnp.iscomplexobj(y)
    dtype = jnp.complex128 if is_complex else x.dtype
    x = jnp.array(x, dtype=dtype)
    y = jnp.array(y, dtype=dtype)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("x and y must be 1D vectors of same length")
    native_real, native_complex = get_dtype(precision)
    native_dtype = native_complex if is_complex else native_real
    if is_complex and precision in ['half', 'bf16']:
        native_dtype = None
    if native_dtype is not None:
        x = x.astype(native_dtype)
        result = x.copy()
        result = result.astype(dtype)
    else:
        x = rounding(x, precision)
        result = x
    return result

def swap(x, y, precision='fp64', exp_bits=None, sig_bits=None, rmode=None):
    """x <-> y"""
    if exp_bits is not None and sig_bits is not None:
        if exp_bits == 23 and sig_bits == 52:
            precision = 'fp64'
        elif exp_bits == 8 and sig_bits == 23:
            precision = 'fp32'
        else:
            precision = {'exp_bits': exp_bits, 'sig_bits': sig_bits, 'rmode': rmode or 1}
    is_complex = jnp.iscomplexobj(x) or jnp.iscomplexobj(y)
    dtype = jnp.complex128 if is_complex else x.dtype
    x = jnp.array(x, dtype=dtype)
    y = jnp.array(y, dtype=dtype)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("x and y must be 1D vectors of same length")
    native_real, native_complex = get_dtype(precision)
    native_dtype = native_complex if is_complex else native_real
    if is_complex and precision in ['half', 'bf16']:
        native_dtype = None
    if native_dtype is not None:
        x = x.astype(native_dtype)
        y = y.astype(native_dtype)
        x_new = y.astype(dtype)
        y_new = x.astype(dtype)
    else:
        x = rounding(x, precision)
        y = rounding(y, precision)
        x_new = y
        y_new = x
    return x_new, y_new

def dot(x, y, precision='fp64', exp_bits=None, sig_bits=None, rmode=None):
    """x . y"""
    if exp_bits is not None and sig_bits is not None:
        if exp_bits == 23 and sig_bits == 52:
            precision = 'fp64'
        elif exp_bits == 8 and sig_bits == 23:
            precision = 'fp32'
        else:
            precision = {'exp_bits': exp_bits, 'sig_bits': sig_bits, 'rmode': rmode or 1}
    is_complex = jnp.iscomplexobj(x) or jnp.iscomplexobj(y)
    dtype = jnp.complex128 if is_complex else x.dtype
    x = jnp.array(x, dtype=dtype)
    y = jnp.array(y, dtype=dtype)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("x and y must be 1D vectors of same length")
    native_real, native_complex = get_dtype(precision)
    native_dtype = native_complex if is_complex else native_real
    if is_complex and precision in ['half', 'bf16']:
        native_dtype = None
    if native_dtype is not None:
        x = x.astype(native_dtype)
        y = y.astype(native_dtype)
        result = jnp.sum(x * y)
        result = result.astype(dtype)
    else:
        x = rounding(x, precision)
        y = rounding(y, precision)
        acc = jnp.array(0.0, dtype=dtype)
        for i in range(x.shape[0]):
            print("i:", i, x[i], y[i])
            prod = rounding(x[i] * y[i], precision)
            acc = rounding(acc + prod, precision)
        result = acc
    return result.item()

def dotc(x, y, precision='fp64', exp_bits=None, sig_bits=None, rmode=None):
    """x . conj(y) (complex)"""
    if exp_bits is not None and sig_bits is not None:
        if exp_bits == 23 and sig_bits == 52:
            precision = 'fp64'
        elif exp_bits == 8 and sig_bits == 23:
            precision = 'fp32'
        else:
            precision = {'exp_bits': exp_bits, 'sig_bits': sig_bits, 'rmode': rmode or 1}
    x = jnp.array(x, dtype=jnp.complex128)
    y = jnp.array(y, dtype=jnp.complex128)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("x and y must be 1D vectors of same length")
    native_real, native_complex = get_dtype(precision)
    native_dtype = native_complex
    if precision in ['half', 'bf16']:
        native_dtype = None
    if native_dtype is not None:
        x = x.astype(native_dtype)
        y = y.astype(native_dtype)
        result = jnp.sum(x * jnp.conj(y))
        result = result.astype(jnp.complex128)
    else:
        x = rounding(x, precision)
        y = rounding(y, precision)
        acc = jnp.array(0.0, dtype=jnp.complex128)
        for i in range(x.shape[0]):
            prod = rounding(x[i] * jnp.conj(y[i]), precision)
            acc = rounding(acc + prod, precision)
        result = acc
    return result.item()

def dotu(x, y, precision='fp64', exp_bits=None, sig_bits=None, rmode=None):
    """x . y (complex, no conjugate)"""
    if exp_bits is not None and sig_bits is not None:
        if exp_bits == 23 and sig_bits == 52:
            precision = 'fp64'
        elif exp_bits == 8 and sig_bits == 23:
            precision = 'fp32'
        else:
            precision = {'exp_bits': exp_bits, 'sig_bits': sig_bits, 'rmode': rmode or 1}
    x = jnp.array(x, dtype=jnp.complex128)
    y = jnp.array(y, dtype=jnp.complex128)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("x and y must be 1D vectors of same length")
    native_real, native_complex = get_dtype(precision)
    native_dtype = native_complex
    if precision in ['half', 'bf16']:
        native_dtype = None
    if native_dtype is not None:
        x = x.astype(native_dtype)
        y = y.astype(native_dtype)
        result = jnp.sum(x * y)
        result = result.astype(jnp.complex128)
    else:
        x = rounding(x, precision)
        y = rounding(y, precision)
        acc = jnp.array(0.0, dtype=jnp.complex128)
        for i in range(x.shape[0]):
            prod = rounding(x[i] * y[i], precision)
            acc = rounding(acc + prod, precision)
        result = acc
    return result.item()

def nrm2(x, precision='fp64', exp_bits=None, sig_bits=None, rmode=None):
    """Euclidean norm of x"""
    if exp_bits is not None and sig_bits is not None:
        if exp_bits == 23 and sig_bits == 52:
            precision = 'fp64'
        elif exp_bits == 8 and sig_bits == 23:
            precision = 'fp32'
        else:
            precision = {'exp_bits': exp_bits, 'sig_bits': sig_bits, 'rmode': rmode or 1}
    is_complex = jnp.iscomplexobj(x)
    dtype = jnp.complex128 if is_complex else x.dtype
    x = jnp.array(x, dtype=dtype)
    if x.ndim != 1:
        raise ValueError("x must be a 1D vector")
    native_real, native_complex = get_dtype(precision)
    native_dtype = native_complex if is_complex else native_real
    if is_complex and precision in ['half', 'bf16']:
        native_dtype = None
    if native_dtype is not None:
        x = x.astype(native_dtype)
        result = jnp.sqrt(jnp.sum(jnp.abs(x)**2))
        result = result.astype(jnp.float64)
    else:
        x = rounding(x, precision)
        acc = jnp.array(0.0, dtype=jnp.float64)
        for i in range(x.shape[0]):
            abs2 = x[i].conj() * x[i] if is_complex else x[i] * x[i]
            abs2_rounded = rounding(abs2, precision)
            acc = rounding(acc + abs2_rounded.real, precision)
        result = rounding(jnp.sqrt(acc), precision)
    return result.item()

def asum(x, precision='fp64', exp_bits=None, sig_bits=None, rmode=None):
    """Sum of absolute values of x"""
    if exp_bits is not None and sig_bits is not None:
        if exp_bits == 23 and sig_bits == 52:
            precision = 'fp64'
        elif exp_bits == 8 and sig_bits == 23:
            precision = 'fp32'
        else:
            precision = {'exp_bits': exp_bits, 'sig_bits': sig_bits, 'rmode': rmode or 1}
    is_complex = jnp.iscomplexobj(x)
    dtype = jnp.complex128 if is_complex else x.dtype
    x = jnp.array(x, dtype=dtype)
    if x.ndim != 1:
        raise ValueError("x must be a 1D vector")
    native_real, native_complex = get_dtype(precision)
    native_dtype = native_complex if is_complex else native_real
    if is_complex and precision in ['half', 'bf16']:
        native_dtype = None
    if native_dtype is not None:
        x = x.astype(native_dtype)
        result = jnp.sum(jnp.abs(x))
        result = result.astype(jnp.float64)
    else:
        x = rounding(x, precision)
        acc = jnp.array(0.0, dtype=jnp.float64)
        for i in range(x.shape[0]):
            abs_val = jnp.abs(x[i])
            acc = rounding(acc + abs_val, precision)
        result = acc
    return result.item()

def rot(x, y, c, s, precision='fp64', exp_bits=None, sig_bits=None, rmode=None):
    """Apply Givens rotation: x' = c*x + s*y, y' = -s*x + c*y"""
    if exp_bits is not None and sig_bits is not None:
        if exp_bits == 23 and sig_bits == 52:
            precision = 'fp64'
        elif exp_bits == 8 and sig_bits == 23:
            precision = 'fp32'
        else:
            precision = {'exp_bits': exp_bits, 'sig_bits': sig_bits, 'rmode': rmode or 1}
    is_complex = jnp.iscomplexobj(x) or jnp.iscomplexobj(y) or jnp.iscomplexobj(s)
    dtype = jnp.complex128 if is_complex else x.dtype
    x = jnp.array(x, dtype=dtype)
    y = jnp.array(y, dtype=dtype)
    c = jnp.array(c, dtype=jnp.float64)
    s = jnp.array(s, dtype=dtype)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("x and y must be 1D vectors of same length")
    native_real, native_complex = get_dtype(precision)
    native_dtype = native_complex if is_complex else native_real
    if is_complex and precision in ['half', 'bf16']:
        native_dtype = None
    if native_dtype is not None:
        x = x.astype(native_dtype)
        y = y.astype(native_dtype)
        c = c.astype(native_real)
        s = s.astype(native_dtype)
        x_new = c * x + s * y
        y_new = -s * x + c * y
        x_new = x_new.astype(dtype)
        y_new = y_new.astype(dtype)
    else:
        x = rounding(x, precision)
        y = rounding(y, precision)
        c = rounding(c, precision)
        s = rounding(s, precision)
        cx = rounding(c * x, precision)
        sy = rounding(s * y, precision)
        x_new = rounding(cx + sy, precision)
        neg_s = rounding(-s, precision)
        sx = rounding(neg_s * x, precision)
        cy = rounding(c * y, precision)
        y_new = rounding(sx + cy, precision)
    return x_new, y_new

def rotg(a, b, precision='fp64', exp_bits=None, sig_bits=None, rmode=None):
    """Generate Givens rotation: compute c, s, r, z for rotation"""
    if exp_bits is not None and sig_bits is not None:
        if exp_bits == 23 and sig_bits == 52:
            precision = 'fp64'
        elif exp_bits == 8 and sig_bits == 23:
            precision = 'fp32'
        else:
            precision = {'exp_bits': exp_bits, 'sig_bits': sig_bits, 'rmode': rmode or 1}
    a = jnp.array(a, dtype=jnp.float64)
    b = jnp.array(b, dtype=jnp.float64)
    native_real, native_complex = get_dtype(precision)
    native_dtype = native_real
    if native_dtype is not None:
        a = a.astype(native_dtype)
        b = b.astype(native_dtype)
        r = jnp.sqrt(a**2 + b**2)
        c = a / r if r != 0 else 1.0
        s = b / r if r != 0 else 0.0
        z = s if c != 0 else 1.0
        r = r.astype(jnp.float64)
        c = c.astype(jnp.float64)
        s = s.astype(jnp.float64)
        z = z.astype(jnp.float64)
    else:
        a = rounding(a, precision)
        b = rounding(b, precision)
        a2 = rounding(a ** 2, precision)
        b2 = rounding(b ** 2, precision)
        sum_sq = rounding(a2 + b2, precision)
        r = rounding(jnp.sqrt(sum_sq), precision)
        if r == 0:
            c = jnp.array(1.0)
            s = jnp.array(0.0)
        else:
            c = rounding(a / r, precision)
            s = rounding(b / r, precision)
        z = s if c != 0 else jnp.array(1.0)
    return (r.item(), c.item(), s.item(), z.item())

def rotm(x, y, param, precision='fp64', exp_bits=None, sig_bits=None, rmode=None):
    """Apply modified rotation (Hessenberg)"""
    if exp_bits is not None and sig_bits is not None:
        if exp_bits == 23 and sig_bits == 52:
            precision = 'fp64'
        elif exp_bits == 8 and sig_bits == 23:
            precision = 'fp32'
        else:
            precision = {'exp_bits': exp_bits, 'sig_bits': sig_bits, 'rmode': rmode or 1}
    x = jnp.array(x, dtype=jnp.float64)
    y = jnp.array(y, dtype=jnp.float64)
    param = jnp.array(param, dtype=jnp.float64)
    if x.shape != y.shape or x.ndim != 1 or param.shape != (5,):
        raise ValueError("x, y must be 1D vectors of same length, param must be 5-element vector")
    native_real, native_complex = get_dtype(precision)
    native_dtype = native_real
    if native_dtype is not None:
        x = x.astype(native_dtype)
        y = y.astype(native_dtype)
        param = param.astype(native_dtype)
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
        else:
            h21, h22 = param[2], param[4]
            x_new = x
            y_new = h21 * x + h22 * y
        x_new = x_new.astype(jnp.float64)
        y_new = y_new.astype(jnp.float64)
    else:
        x = rounding(x, precision)
        y = rounding(y, precision)
        param = rounding(param, precision)
        flag = param[0]
        if flag == -1:
            h11, h21, h12, h22 = param[1:5]
            h11x = rounding(h11 * x, precision)
            h12y = rounding(h12 * y, precision)
            x_new = rounding(h11x + h12y, precision)
            h21x = rounding(h21 * x, precision)
            h22y = rounding(h22 * y, precision)
            y_new = rounding(h21x + h22y, precision)
        elif flag == 0:
            x_new, y_new = x, y
        elif flag == 1:
            h11, h12 = param[1], param[3]
            h11x = rounding(h11 * x, precision)
            h12y = rounding(h12 * y, precision)
            x_new = rounding(h11x + h12y, precision)
            y_new = y
        else:
            h21, h22 = param[2], param[4]
            x_new = x
            h21x = rounding(h21 * x, precision)
            h22y = rounding(h22 * y, precision)
            y_new = rounding(h21x + h22y, precision)
    return x_new, y_new

def rotmg(d1, d2, x1, y1, precision='fp64', exp_bits=None, sig_bits=None, rmode=None):
    """Generate modified rotation parameters"""
    if exp_bits is not None and sig_bits is not None:
        if exp_bits == 23 and sig_bits == 52:
            precision = 'fp64'
        elif exp_bits == 8 and sig_bits == 23:
            precision = 'fp32'
        else:
            precision = {'exp_bits': exp_bits, 'sig_bits': sig_bits, 'rmode': rmode or 1}
    d1 = jnp.array(d1, dtype=jnp.float64)
    d2 = jnp.array(d2, dtype=jnp.float64)
    x1 = jnp.array(x1, dtype=jnp.float64)
    y1 = jnp.array(y1, dtype=jnp.float64)
    native_real, native_complex = get_dtype(precision)
    native_dtype = native_real
    if native_dtype is not None:
        d1 = d1.astype(native_dtype)
        d2 = d2.astype(native_dtype)
        x1 = x1.astype(native_dtype)
        y1 = y1.astype(native_dtype)
        param = jnp.zeros(5, dtype=native_dtype)
        if d1 == 0 or d2 == 0 or x1 == 0:
            param = param.at[0].set(-1)
        else:
            param = param.at[0].set(-1)
            param = param.at[1].set(1.0)
            param = param.at[2].set(0.0)
            param = param.at[3].set(0.0)
            param = param.at[4].set(1.0)
        param = param.astype(jnp.float64)
    else:
        d1 = rounding(d1, precision)
        d2 = rounding(d2, precision)
        x1 = rounding(x1, precision)
        y1 = rounding(y1, precision)
        param = jnp.zeros(5, dtype=jnp.float64)
        if d1 == 0 or d2 == 0 or x1 == 0:
            param = param.at[0].set(-1)
        else:
            param = param.at[0].set(-1)
            param = param.at[1].set(1.0)
            param = param.at[2].set(0.0)
            param = param.at[3].set(0.0)
            param = param.at[4].set(1.0)
        param = rounding(param, precision)
    return param

def iamax(x, precision='fp64', exp_bits=None, sig_bits=None, rmode=None):
    """Index of maximum absolute value"""
    if exp_bits is not None and sig_bits is not None:
        if exp_bits == 23 and sig_bits == 52:
            precision = 'fp64'
        elif exp_bits == 8 and sig_bits == 23:
            precision = 'fp32'
        else:
            precision = {'exp_bits': exp_bits, 'sig_bits': sig_bits, 'rmode': rmode or 1}
    is_complex = jnp.iscomplexobj(x)
    dtype = jnp.complex128 if is_complex else x.dtype
    x = jnp.array(x, dtype=dtype)
    if x.ndim != 1:
        raise ValueError("x must be a 1D vector")
    native_real, native_complex = get_dtype(precision)
    native_dtype = native_complex if is_complex else native_real
    if is_complex and precision in ['half', 'bf16']:
        native_dtype = None
    if native_dtype is not None:
        x = x.astype(native_dtype)
        idx = jnp.argmax(jnp.abs(x))
    else:
        x = rounding(x, precision)
        idx = jnp.argmax(jnp.abs(x))
    return idx.item()

def iamin(x, precision='fp64', exp_bits=None, sig_bits=None, rmode=None):
    """Index of minimum absolute value"""
    if exp_bits is not None and sig_bits is not None:
        if exp_bits == 23 and sig_bits == 52:
            precision = 'fp64'
        elif exp_bits == 8 and sig_bits == 23:
            precision = 'fp32'
        else:
            precision = {'exp_bits': exp_bits, 'sig_bits': sig_bits, 'rmode': rmode or 1}
    is_complex = jnp.iscomplexobj(x)
    dtype = jnp.complex128 if is_complex else x.dtype
    x = jnp.array(x, dtype=dtype)
    if x.ndim != 1:
        raise ValueError("x must be a 1D vector")
    native_real, native_complex = get_dtype(precision)
    native_dtype = native_complex if is_complex else native_real
    if is_complex and precision in ['half', 'bf16']:
        native_dtype = None
    if native_dtype is not None:
        x = x.astype(native_dtype)
        idx = jnp.argmin(jnp.abs(x))
    else:
        x = rounding(x, precision)
        idx = jnp.argmin(jnp.abs(x))
    return idx.item()

# Level 2 BLAS: Matrix-Vector Operations
def low_prec_mv(A, x, precision, conj=False):
    """Low precision matrix-vector multiply with intermediate rounding"""
    result = jnp.zeros(A.shape[0], dtype=A.dtype)
    for row in range(A.shape[0]):
        acc = jnp.array(0.0, dtype=A.dtype)
        for col in range(A.shape[1]):
            a_val = jnp.conj(A[row, col]) if conj else A[row, col]
            prod = rounding(a_val * x[col], precision)
            acc = rounding(acc + prod, precision)
        result = result.at[row].set(acc)
    return result

def gemv(alpha, A, x, beta, y, trans='N', precision='fp64', exp_bits=None, sig_bits=None, rmode=None):
    """y = alpha * op(A) * x + beta * y, op(A) = A or A^T or A^H"""
    if exp_bits is not None and sig_bits is not None:
        if exp_bits == 23 and sig_bits == 52:
            precision = 'fp64'
        elif exp_bits == 8 and sig_bits == 23:
            precision = 'fp32'
        else:
            precision = {'exp_bits': exp_bits, 'sig_bits': sig_bits, 'rmode': rmode or 1}
    is_complex = jnp.iscomplexobj(A) or jnp.iscomplexobj(x) or jnp.iscomplexobj(y) or jnp.iscomplexobj(alpha) or jnp.iscomplexobj(beta)
    dtype = jnp.complex128 if is_complex else x.dtype
    A = jnp.array(A, dtype=dtype)
    x = jnp.array(x, dtype=dtype)
    y = jnp.array(y, dtype=dtype)
    alpha = jnp.array(alpha, dtype=dtype)
    beta = jnp.array(beta, dtype=dtype)
    if A.ndim != 2 or x.ndim != 1 or y.ndim != 1:
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
        alpha = alpha.astype(native_dtype)
        A = A.astype(native_dtype)
        x = x.astype(native_dtype)
        beta = beta.astype(native_dtype)
        y = y.astype(native_dtype)
        opA = A if trans == 'N' else (A.T if trans == 'T' else A.conj().T)
        mv = jnp.dot(opA, x)
        result = alpha * mv + beta * y
        result = result.astype(dtype)
    else:
        alpha = rounding(alpha, precision)
        beta = rounding(beta, precision)
        A = rounding(A, precision)
        x = rounding(x, precision)
        y = rounding(y, precision)
        conj_A = (trans == 'C')
        opA = A if trans == 'N' else A.T
        mv = low_prec_mv(opA, x, precision, conj=conj_A)
        alpha_mv = rounding(alpha * mv, precision)
        beta_y = rounding(beta * y, precision)
        result = rounding(alpha_mv + beta_y, precision)
    return result

def gbmv(alpha, A, x, beta, y, kl, ku, trans='N', precision='fp64', exp_bits=None, sig_bits=None, rmode=None):
    """y = alpha * op(A) * x + beta * y for band matrix A"""
    if exp_bits is not None and sig_bits is not None:
        if exp_bits == 23 and sig_bits == 52:
            precision = 'fp64'
        elif exp_bits == 8 and sig_bits == 23:
            precision = 'fp32'
        else:
            precision = {'exp_bits': exp_bits, 'sig_bits': sig_bits, 'rmode': rmode or 1}
    return gemv(alpha, A, x, beta, y, trans, precision=precision)

def symv(alpha, A, x, beta, y, uplo='U', precision='fp64', exp_bits=None, sig_bits=None, rmode=None):
    """y = alpha * A * x + beta * y, A symmetric"""
    if exp_bits is not None and sig_bits is not None:
        if exp_bits == 23 and sig_bits == 52:
            precision = 'fp64'
        elif exp_bits == 8 and sig_bits == 23:
            precision = 'fp32'
        else:
            precision = {'exp_bits': exp_bits, 'sig_bits': sig_bits, 'rmode': rmode or 1}
    return gemv(alpha, A, x, beta, y, 'N', precision=precision)

def sbmv(alpha, A, x, beta, y, k, uplo='U', precision='fp64', exp_bits=None, sig_bits=None, rmode=None):
    """y = alpha * A * x + beta * y, A symmetric band"""
    if exp_bits is not None and sig_bits is not None:
        if exp_bits == 23 and sig_bits == 52:
            precision = 'fp64'
        elif exp_bits == 8 and sig_bits == 23:
            precision = 'fp32'
        else:
            precision = {'exp_bits': exp_bits, 'sig_bits': sig_bits, 'rmode': rmode or 1}
    return gemv(alpha, A, x, beta, y, 'N', precision=precision)

def hemv(alpha, A, x, beta, y, uplo='U', precision='fp64', exp_bits=None, sig_bits=None, rmode=None):
    """y = alpha * A * x + beta * y, A Hermitian"""
    if exp_bits is not None and sig_bits is not None:
        if exp_bits == 23 and sig_bits == 52:
            precision = 'fp64'
        elif exp_bits == 8 and sig_bits == 23:
            precision = 'fp32'
        else:
            precision = {'exp_bits': exp_bits, 'sig_bits': sig_bits, 'rmode': rmode or 1}
    return gemv(alpha, A, x, beta, y, 'N', precision=precision)

def hbmv(alpha, A, x, beta, y, k, uplo='U', precision='fp64', exp_bits=None, sig_bits=None, rmode=None):
    """y = alpha * A * x + beta * y, A Hermitian band"""
    if exp_bits is not None and sig_bits is not None:
        if exp_bits == 23 and sig_bits == 52:
            precision = 'fp64'
        elif exp_bits == 8 and sig_bits == 23:
            precision = 'fp32'
        else:
            precision = {'exp_bits': exp_bits, 'sig_bits': sig_bits, 'rmode': rmode or 1}
    return gemv(alpha, A, x, beta, y, 'N', precision=precision)

def spmv(alpha, A, x, beta, y, uplo='U', precision='fp64', exp_bits=None, sig_bits=None, rmode=None):
    """y = alpha * A * x + beta * y, A symmetric packed"""
    if exp_bits is not None and sig_bits is not None:
        if exp_bits == 23 and sig_bits == 52:
            precision = 'fp64'
        elif exp_bits == 8 and sig_bits == 23:
            precision = 'fp32'
        else:
            precision = {'exp_bits': exp_bits, 'sig_bits': sig_bits, 'rmode': rmode or 1}
    A = jnp.array(A, dtype=jnp.float64)
    x = jnp.array(x, dtype=A.dtype)
    y = jnp.array(y, dtype=A.dtype)
    n = int((jnp.sqrt(8 * A.shape[0] + 1) - 1) / 2)
    A_dense = jnp.zeros((n, n), dtype=A.dtype)
    idx = 0
    for i in range(n):
        start = i if uplo == 'U' else 0
        end = n if uplo == 'U' else i + 1
        for j in range(start, end):
            A_dense = A_dense.at[i, j].set(A[idx])
            if i != j:
                A_dense = A_dense.at[j, i].set(A[idx])
            idx += 1
    return gemv(alpha, A_dense, x, beta, y, 'N', precision=precision)

def hpmv(alpha, A, x, beta, y, uplo='U', precision='fp64', exp_bits=None, sig_bits=None, rmode=None):
    """y = alpha * A * x + beta * y, A Hermitian packed"""
    if exp_bits is not None and sig_bits is not None:
        if exp_bits == 23 and sig_bits == 52:
            precision = 'fp64'
        elif exp_bits == 8 and sig_bits == 23:
            precision = 'fp32'
        else:
            precision = {'exp_bits': exp_bits, 'sig_bits': sig_bits, 'rmode': rmode or 1}
    A = jnp.array(A, dtype=jnp.complex128)
    x = jnp.array(x, dtype=A.dtype)
    y = jnp.array(y, dtype=A.dtype)
    n = int((jnp.sqrt(8 * A.shape[0] + 1) - 1) / 2)
    A_dense = jnp.zeros((n, n), dtype=A.dtype)
    idx = 0
    for i in range(n):
        start = i if uplo == 'U' else 0
        end = n if uplo == 'U' else i + 1
        for j in range(start, end):
            A_dense = A_dense.at[i, j].set(A[idx])
            if i != j:
                A_dense = A_dense.at[j, i].set(jnp.conj(A[idx]))
            idx += 1
    return gemv(alpha, A_dense, x, beta, y, 'N', precision=precision)

def trmv(A, x, uplo='U', trans='N', diag='N', precision='fp64', exp_bits=None, sig_bits=None, rmode=None):
    """x = op(A) * x, A triangular"""
    if exp_bits is not None and sig_bits is not None:
        if exp_bits == 23 and sig_bits == 52:
            precision = 'fp64'
        elif exp_bits == 8 and sig_bits == 23:
            precision = 'fp32'
        else:
            precision = {'exp_bits': exp_bits, 'sig_bits': sig_bits, 'rmode': rmode or 1}
    return gemv(1.0, A, x, 0.0, jnp.zeros_like(x), trans, precision=precision)

def low_prec_trsv(A, b, uplo='U', trans='N', diag='N', precision='fp64'):
    """Low precision triangular solve with intermediate rounding"""
    n = A.shape[0]
    x = jnp.zeros_like(b)
    unit = (diag == 'U')
    if trans == 'N':
        upper = (uplo == 'U')
        if upper:
            for i in range(n-1, -1, -1):
                sum_ = jnp.array(0.0, dtype=A.dtype)
                for j in range(i+1, n):
                    prod = rounding(A[i, j] * x[j], precision)
                    sum_ = rounding(sum_ + prod, precision)
                div = rounding(A[i, i], precision) if not unit else jnp.array(1.0, dtype=A.dtype)
                diff = rounding(b[i] - sum_, precision)
                x = x.at[i].set(rounding(diff / div, precision))
        else:
            for i in range(n):
                sum_ = jnp.array(0.0, dtype=A.dtype)
                for j in range(i):
                    prod = rounding(A[i, j] * x[j], precision)
                    sum_ = rounding(sum_ + prod, precision)
                div = rounding(A[i, i], precision) if not unit else jnp.array(1.0, dtype=A.dtype)
                diff = rounding(b[i] - sum_, precision)
                x = x.at[i].set(rounding(diff / div, precision))
    else:
        upper = (uplo == 'U')
        if upper:
            for i in range(n):
                sum_ = jnp.array(0.0, dtype=A.dtype)
                for j in range(i):
                    prod = rounding(A[j, i] * x[j], precision)
                    sum_ = rounding(sum_ + prod, precision)
                div = rounding(A[i, i], precision) if not unit else jnp.array(1.0, dtype=A.dtype)
                diff = rounding(b[i] - sum_, precision)
                x = x.at[i].set(rounding(diff / div, precision))
        else:
            for i in range(n-1, -1, -1):
                sum_ = jnp.array(0.0, dtype=A.dtype)
                for j in range(i+1, n):
                    prod = rounding(A[j, i] * x[j], precision)
                    sum_ = rounding(sum_ + prod, precision)
                div = rounding(A[i, i], precision) if not unit else jnp.array(1.0, dtype=A.dtype)
                diff = rounding(b[i] - sum_, precision)
                x = x.at[i].set(rounding(diff / div, precision))
    return x

def trsv(A, b, uplo='U', trans='N', diag='N', precision='fp64', exp_bits=None, sig_bits=None, rmode=None):
    """Solve op(A) * x = b, A triangular"""
    if exp_bits is not None and sig_bits is not None:
        if exp_bits == 23 and sig_bits == 52:
            precision = 'fp64'
        elif exp_bits == 8 and sig_bits == 23:
            precision = 'fp32'
        else:
            precision = {'exp_bits': exp_bits, 'sig_bits': sig_bits, 'rmode': rmode or 1}
    is_complex = jnp.iscomplexobj(A) or jnp.iscomplexobj(b)
    dtype = jnp.complex128 if is_complex else b.dtype
    A = jnp.array(A, dtype=dtype)
    b = jnp.array(b, dtype=dtype)
    if A.ndim != 2 or b.ndim != 1 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square 2D, b must be 1D")
    native_real, native_complex = get_dtype(precision)
    native_dtype = native_complex if is_complex else native_real
    if is_complex and precision in ['half', 'bf16']:
        native_dtype = None
    if native_dtype is not None:
        A = A.astype(native_dtype)
        b = b.astype(native_dtype)
        opA = A if trans == 'N' else A.T
        x = jnp.linalg.solve(opA, b) if not unit else jnp.linalg.solve(jnp.tril(opA, 0) + jnp.diag(jnp.ones(A.shape[0])), b)
        x = x.astype(dtype)
    else:
        A = rounding(A, precision)
        b = rounding(b, precision)
        x = low_prec_trsv(A, b, uplo, trans, diag, precision)
    return x

def tbmv(A, x, k, uplo='U', trans='N', diag='N', precision='fp64', exp_bits=None, sig_bits=None, rmode=None):
    """x = op(A) * x, A triangular band"""
    if exp_bits is not None and sig_bits is not None:
        if exp_bits == 23 and sig_bits == 52:
            precision = 'fp64'
        elif exp_bits == 8 and sig_bits == 23:
            precision = 'fp32'
        else:
            precision = {'exp_bits': exp_bits, 'sig_bits': sig_bits, 'rmode': rmode or 1}
    return trmv(A, x, uplo, trans, diag, precision=precision)

def tbsv(A, b, k, uplo='U', trans='N', diag='N', precision='fp64', exp_bits=None, sig_bits=None, rmode=None):
    """Solve op(A) * x = b, A triangular band"""
    if exp_bits is not None and sig_bits is not None:
        if exp_bits == 23 and sig_bits == 52:
            precision = 'fp64'
        elif exp_bits == 8 and sig_bits == 23:
            precision = 'fp32'
        else:
            precision = {'exp_bits': exp_bits, 'sig_bits': sig_bits, 'rmode': rmode or 1}
    return trsv(A, b, uplo, trans, diag, precision=precision)

def tpmv(A, x, uplo='U', trans='N', diag='N', precision='fp64', exp_bits=None, sig_bits=None, rmode=None):
    """x = op(A) * x, A triangular packed"""
    if exp_bits is not None and sig_bits is not None:
        if exp_bits == 23 and sig_bits == 52:
            precision = 'fp64'
        elif exp_bits == 8 and sig_bits == 23:
            precision = 'fp32'
        else:
            precision = {'exp_bits': exp_bits, 'sig_bits': sig_bits, 'rmode': rmode or 1}
    is_complex = jnp.iscomplexobj(A) or jnp.iscomplexobj(x)
    dtype = jnp.complex128 if is_complex else x.dtype
    A = jnp.array(A, dtype=dtype)
    x = jnp.array(x, dtype=dtype)
    n = int((jnp.sqrt(8 * A.shape[0] + 1) - 1) / 2)
    A_dense = jnp.zeros((n, n), dtype=dtype)
    idx = 0
    for i in range(n):
        start = i if uplo == 'U' else 0
        end = n if uplo == 'U' else i + 1
        for j in range(start, end):
            A_dense = A_dense.at[i, j].set(A[idx])
            idx += 1
    return trmv(A_dense, x, uplo, trans, diag, precision=precision)

def tpsv(A, b, uplo='U', trans='N', diag='N', precision='fp64', exp_bits=None, sig_bits=None, rmode=None):
    """Solve op(A) * x = b, A triangular packed"""
    if exp_bits is not None and sig_bits is not None:
        if exp_bits == 23 and sig_bits == 52:
            precision = 'fp64'
        elif exp_bits == 8 and sig_bits == 23:
            precision = 'fp32'
        else:
            precision = {'exp_bits': exp_bits, 'sig_bits': sig_bits, 'rmode': rmode or 1}
    is_complex = jnp.iscomplexobj(A) or jnp.iscomplexobj(b)
    dtype = jnp.complex128 if is_complex else b.dtype
    A = jnp.array(A, dtype=dtype)
    b = jnp.array(b, dtype=dtype)
    n = int((jnp.sqrt(8 * A.shape[0] + 1) - 1) / 2)
    A_dense = jnp.zeros((n, n), dtype=dtype)
    idx = 0
    for i in range(n):
        start = i if uplo == 'U' else 0
        end = n if uplo == 'U' else i + 1
        for j in range(start, end):
            A_dense = A_dense.at[i, j].set(A[idx])
            idx += 1
    return trsv(A_dense, b, uplo, trans, diag, precision=precision)

def low_prec_outer(x, y, precision, conj_y=False):
    """Low precision outer product with intermediate rounding"""
    m = x.shape[0]
    n = y.shape[0]
    result = jnp.zeros((m, n), dtype=x.dtype)
    for i in range(m):
        for j in range(n):
            y_val = jnp.conj(y[j]) if conj_y else y[j]
            prod = rounding(x[i] * y_val, precision)
            result = result.at[i, j].set(prod)
    return result

def ger(alpha, x, y, A, precision='fp64', exp_bits=None, sig_bits=None, rmode=None):
    """A = A + alpha * x * y^T"""
    if exp_bits is not None and sig_bits is not None:
        if exp_bits == 23 and sig_bits == 52:
            precision = 'fp64'
        elif exp_bits == 8 and sig_bits == 23:
            precision = 'fp32'
        else:
            precision = {'exp_bits': exp_bits, 'sig_bits': sig_bits, 'rmode': rmode or 1}
    is_complex = jnp.iscomplexobj(A) or jnp.iscomplexobj(x) or jnp.iscomplexobj(y) or jnp.iscomplexobj(alpha)
    dtype = jnp.complex128 if is_complex else x.dtype
    A = jnp.array(A, dtype=dtype)
    x = jnp.array(x, dtype=dtype)
    y = jnp.array(y, dtype=dtype)
    alpha = jnp.array(alpha, dtype=dtype)
    if A.ndim != 2 or x.ndim != 1 or y.ndim != 1 or A.shape[0] != x.shape[0] or A.shape[1] != y.shape[0]:
        raise ValueError("Incompatible dimensions")
    native_real, native_complex = get_dtype(precision)
    native_dtype = native_complex if is_complex else native_real
    if is_complex and precision in ['half', 'bf16']:
        native_dtype = None
    if native_dtype is not None:
        alpha = alpha.astype(native_dtype)
        x = x.astype(native_dtype)
        y = y.astype(native_dtype)
        A = A.astype(native_dtype)
        result = A + alpha * jnp.outer(x, y)
        result = result.astype(dtype)
    else:
        alpha = rounding(alpha, precision)
        x = rounding(x, precision)
        y = rounding(y, precision)
        A = rounding(A, precision)
        outer = low_prec_outer(x, y, precision, conj_y=False)
        alpha_outer = rounding(alpha * outer, precision)
        result = rounding(A + alpha_outer, precision)
    return result

def syr(alpha, x, A, uplo='U', precision='fp64', exp_bits=None, sig_bits=None, rmode=None):
    """A = A + alpha * x * x^T, A symmetric"""
    if exp_bits is not None and sig_bits is not None:
        if exp_bits == 23 and sig_bits == 52:
            precision = 'fp64'
        elif exp_bits == 8 and sig_bits == 23:
            precision = 'fp32'
        else:
            precision = {'exp_bits': exp_bits, 'sig_bits': sig_bits, 'rmode': rmode or 1}
    A = jnp.array(A, dtype=jnp.float64)
    x = jnp.array(x, dtype=A.dtype)
    alpha = jnp.array(alpha, dtype=A.dtype)
    if A.ndim != 2 or x.ndim != 1 or A.shape[0] != A.shape[1] or A.shape[0] != x.shape[0]:
        raise ValueError("Incompatible dimensions")
    native_real, native_complex = get_dtype(precision)
    native_dtype = native_real
    if native_dtype is not None:
        alpha = alpha.astype(native_dtype)
        x = x.astype(native_dtype)
        A = A.astype(native_dtype)
        result = A + alpha * jnp.outer(x, x)
        result = result.astype(jnp.float64)
    else:
        alpha = rounding(alpha, precision)
        x = rounding(x, precision)
        A = rounding(A, precision)
        outer = low_prec_outer(x, x, precision, conj_y=False)
        alpha_outer = rounding(alpha * outer, precision)
        result = rounding(A + alpha_outer, precision)
    return result

def spr(alpha, x, A, uplo='U', precision='fp64', exp_bits=None, sig_bits=None, rmode=None):
    """A = A + alpha * x * x^T, A symmetric packed"""
    if exp_bits is not None and sig_bits is not None:
        if exp_bits == 23 and sig_bits == 52:
            precision = 'fp64'
        elif exp_bits == 8 and sig_bits == 23:
            precision = 'fp32'
        else:
            precision = {'exp_bits': exp_bits, 'sig_bits': sig_bits, 'rmode': rmode or 1}
    A = jnp.array(A, dtype=jnp.float64)
    x = jnp.array(x, dtype=A.dtype)
    n = int((jnp.sqrt(8 * A.shape[0] + 1) - 1) / 2)
    A_dense = jnp.zeros((n, n), dtype=A.dtype)
    idx = 0
    for i in range(n):
        start = i if uplo == 'U' else 0
        end = n if uplo == 'U' else i + 1
        for j in range(start, end):
            A_dense = A_dense.at[i, j].set(A[idx])
            if i != j:
                A_dense = A_dense.at[j, i].set(A[idx])
            idx += 1
    A_new = syr(alpha, x, A_dense, uplo, precision=precision)
    A_packed = jnp.zeros_like(A)
    idx = 0
    for i in range(n):
        start = i if uplo == 'U' else 0
        end = n if uplo == 'U' else i + 1
        for j in range(start, end):
            A_packed = A_packed.at[idx].set(A_new[i, j])
            idx += 1
    return A_packed

def syr2(alpha, x, y, A, uplo='U', precision='fp64', exp_bits=None, sig_bits=None, rmode=None):
    """A = A + alpha * x * y^T + alpha * y * x^T, A symmetric"""
    if exp_bits is not None and sig_bits is not None:
        if exp_bits == 23 and sig_bits == 52:
            precision = 'fp64'
        elif exp_bits == 8 and sig_bits == 23:
            precision = 'fp32'
        else:
            precision = {'exp_bits': exp_bits, 'sig_bits': sig_bits, 'rmode': rmode or 1}
    A = jnp.array(A, dtype=jnp.float64)
    x = jnp.array(x, dtype=A.dtype)
    y = jnp.array(y, dtype=A.dtype)
    alpha = jnp.array(alpha, dtype=A.dtype)
    if A.ndim != 2 or x.ndim != 1 or y.ndim != 1 or A.shape[0] != A.shape[1] or A.shape[0] != x.shape[0] or A.shape[0] != y.shape[0]:
        raise ValueError("Incompatible dimensions")
    native_real, native_complex = get_dtype(precision)
    native_dtype = native_real
    if native_dtype is not None:
        alpha = alpha.astype(native_dtype)
        x = x.astype(native_dtype)
        y = y.astype(native_dtype)
        A = A.astype(native_dtype)
        result = A + alpha * (jnp.outer(x, y) + jnp.outer(y, x))
        result = result.astype(jnp.float64)
    else:
        alpha = rounding(alpha, precision)
        x = rounding(x, precision)
        y = rounding(y, precision)
        A = rounding(A, precision)
        outer1 = low_prec_outer(x, y, precision)
        outer2 = low_prec_outer(y, x, precision)
        sum_outer = rounding(outer1 + outer2, precision)
        alpha_sum = rounding(alpha * sum_outer, precision)
        result = rounding(A + alpha_sum, precision)
    return result

def spr2(alpha, x, y, A, uplo='U', precision='fp64', exp_bits=None, sig_bits=None, rmode=None):
    """A = A + alpha * x * y^T + alpha * y * x^T, A symmetric packed"""
    if exp_bits is not None and sig_bits is not None:
        if exp_bits == 23 and sig_bits == 52:
            precision = 'fp64'
        elif exp_bits == 8 and sig_bits == 23:
            precision = 'fp32'
        else:
            precision = {'exp_bits': exp_bits, 'sig_bits': sig_bits, 'rmode': rmode or 1}
    A = jnp.array(A, dtype=jnp.float64)
    x = jnp.array(x, dtype=A.dtype)
    y = jnp.array(y, dtype=A.dtype)
    n = int((jnp.sqrt(8 * A.shape[0] + 1) - 1) / 2)
    A_dense = jnp.zeros((n, n), dtype=A.dtype)
    idx = 0
    for i in range(n):
        start = i if uplo == 'U' else 0
        end = n if uplo == 'U' else i + 1
        for j in range(start, end):
            A_dense = A_dense.at[i, j].set(A[idx])
            if i != j:
                A_dense = A_dense.at[j, i].set(A[idx])
            idx += 1
    A_new = syr2(alpha, x, y, A_dense, uplo, precision=precision)
    A_packed = jnp.zeros_like(A)
    idx = 0
    for i in range(n):
        start = i if uplo == 'U' else 0
        end = n if uplo == 'U' else i + 1
        for j in range(start, end):
            A_packed = A_packed.at[idx].set(A_new[i, j])
            idx += 1
    return A_packed

def her(alpha, x, A, uplo='U', precision='fp64', exp_bits=None, sig_bits=None, rmode=None):
    """A = A + alpha * x * x^H, A Hermitian"""
    if exp_bits is not None and sig_bits is not None:
        if exp_bits == 23 and sig_bits == 52:
            precision = 'fp64'
        elif exp_bits == 8 and sig_bits == 23:
            precision = 'fp32'
        else:
            precision = {'exp_bits': exp_bits, 'sig_bits': sig_bits, 'rmode': rmode or 1}
    A = jnp.array(A, dtype=jnp.complex128)
    x = jnp.array(x, dtype=A.dtype)
    alpha = jnp.array(alpha, dtype=jnp.float64)
    if A.ndim != 2 or x.ndim != 1 or A.shape[0] != A.shape[1] or A.shape[0] != x.shape[0]:
        raise ValueError("Incompatible dimensions")
    native_real, native_complex = get_dtype(precision)
    native_dtype = native_complex
    if precision in ['half', 'bf16']:
        native_dtype = None
    if native_dtype is not None:
        alpha = alpha.astype(native_real)
        x = x.astype(native_dtype)
        A = A.astype(native_dtype)
        result = A + alpha * jnp.outer(x, jnp.conj(x))
        result = result.astype(jnp.complex128)
    else:
        alpha = rounding(alpha, precision)
        x = rounding(x, precision)
        A = rounding(A, precision)
        outer = low_prec_outer(x, x, precision, conj_y=True)
        alpha_outer = rounding(alpha * outer, precision)
        result = rounding(A + alpha_outer, precision)
    return result

def hpr(alpha, x, A, uplo='U', precision='fp64', exp_bits=None, sig_bits=None, rmode=None):
    """A = A + alpha * x * x^H, A Hermitian packed"""
    if exp_bits is not None and sig_bits is not None:
        if exp_bits == 23 and sig_bits == 52:
            precision = 'fp64'
        elif exp_bits == 8 and sig_bits == 23:
            precision = 'fp32'
        else:
            precision = {'exp_bits': exp_bits, 'sig_bits': sig_bits, 'rmode': rmode or 1}
    A = jnp.array(A, dtype=jnp.complex128)
    x = jnp.array(x, dtype=A.dtype)
    n = int((jnp.sqrt(8 * A.shape[0] + 1) - 1) / 2)
    A_dense = jnp.zeros((n, n), dtype=A.dtype)
    idx = 0
    for i in range(n):
        start = i if uplo == 'U' else 0
        end = n if uplo == 'U' else i + 1
        for j in range(start, end):
            A_dense = A_dense.at[i, j].set(A[idx])
            if i != j:
                A_dense = A_dense.at[j, i].set(jnp.conj(A[idx]))
            idx += 1
    A_new = her(alpha, x, A_dense, uplo, precision=precision)
    A_packed = jnp.zeros_like(A)
    idx = 0
    for i in range(n):
        start = i if uplo == 'U' else 0
        end = n if uplo == 'U' else i + 1
        for j in range(start, end):
            A_packed = A_packed.at[idx].set(A_new[i, j])
            idx += 1
    return A_packed

def her2(alpha, x, y, A, uplo='U', precision='fp64', exp_bits=None, sig_bits=None, rmode=None):
    """A = A + alpha * x * y^H + alpha * y * x^H, A Hermitian"""
    if exp_bits is not None and sig_bits is not None:
        if exp_bits == 23 and sig_bits == 52:
            precision = 'fp64'
        elif exp_bits == 8 and sig_bits == 23:
            precision = 'fp32'
        else:
            precision = {'exp_bits': exp_bits, 'sig_bits': sig_bits, 'rmode': rmode or 1}
    A = jnp.array(A, dtype=jnp.complex128)
    x = jnp.array(x, dtype=A.dtype)
    y = jnp.array(y, dtype=A.dtype)
    alpha = jnp.array(alpha, dtype=A.dtype)
    if A.ndim != 2 or x.ndim != 1 or y.ndim != 1 or A.shape[0] != A.shape[1] or A.shape[0] != x.shape[0] or A.shape[0] != y.shape[0]:
        raise ValueError("Incompatible dimensions")
    native_real, native_complex = get_dtype(precision)
    native_dtype = native_complex
    if precision in ['half', 'bf16']:
        native_dtype = None
    if native_dtype is not None:
        alpha = alpha.astype(native_dtype)
        x = x.astype(native_dtype)
        y = y.astype(native_dtype)
        A = A.astype(native_dtype)
        result = A + alpha * (jnp.outer(x, jnp.conj(y)) + jnp.outer(y, jnp.conj(x)))
        result = result.astype(jnp.complex128)
    else:
        alpha = rounding(alpha, precision)
        x = rounding(x, precision)
        y = rounding(y, precision)
        A = rounding(A, precision)
        outer1 = low_prec_outer(x, y, precision, conj_y=True)
        outer2 = low_prec_outer(y, x, precision, conj_y=True)
        sum_outer = rounding(outer1 + outer2, precision)
        alpha_sum = rounding(alpha * sum_outer, precision)
        result = rounding(A + alpha_sum, precision)
    return result

def hpr2(alpha, x, y, A, uplo='U', precision='fp64', exp_bits=None, sig_bits=None, rmode=None):
    """A = A + alpha * x * y^H + alpha * y * x^H, A Hermitian packed"""
    if exp_bits is not None and sig_bits is not None:
        if exp_bits == 23 and sig_bits == 52:
            precision = 'fp64'
        elif exp_bits == 8 and sig_bits == 23:
            precision = 'fp32'
        else:
            precision = {'exp_bits': exp_bits, 'sig_bits': sig_bits, 'rmode': rmode or 1}
    A = jnp.array(A, dtype=jnp.complex128)
    x = jnp.array(x, dtype=A.dtype)
    y = jnp.array(y, dtype=A.dtype)
    n = int((jnp.sqrt(8 * A.shape[0] + 1) - 1) / 2)
    A_dense = jnp.zeros((n, n), dtype=A.dtype)
    idx = 0
    for i in range(n):
        start = i if uplo == 'U' else 0
        end = n if uplo == 'U' else i + 1
        for j in range(start, end):
            A_dense = A_dense.at[i, j].set(A[idx])
            if i != j:
                A_dense = A_dense.at[j, i].set(jnp.conj(A[idx]))
            idx += 1
    A_new = her2(alpha, x, y, A_dense, uplo, precision=precision)
    A_packed = jnp.zeros_like(A)
    idx = 0
    for i in range(n):
        start = i if uplo == 'U' else 0
        end = n if uplo == 'U' else i + 1
        for j in range(start, end):
            A_packed = A_packed.at[idx].set(A_new[i, j])
            idx += 1
    return A_packed

# Level 3 BLAS: Matrix-Matrix Operations
def low_prec_mm(A, B, precision, conjA=False, conjB=False):
    """Low precision matrix-matrix multiply with intermediate rounding"""
    m, k = A.shape
    k2, n = B.shape
    if k != k2:
        raise ValueError("Incompatible dimensions for A * B")
    result = jnp.zeros((m, n), dtype=A.dtype)
    for i in range(m):
        for j in range(n):
            acc = jnp.array(0.0, dtype=A.dtype)
            for l in range(k):
                a_val = jnp.conj(A[i, l]) if conjA else A[i, l]
                b_val = jnp.conj(B[l, j]) if conjB else B[l, j]
                prod = rounding(a_val * b_val, precision)
                acc = rounding(acc + prod, precision)
            result = result.at[i, j].set(acc)
    return result

def gemm(alpha, A, B, beta, C, transA='N', transB='N', precision='fp64', exp_bits=None, sig_bits=None, rmode=None):
    """C = alpha * op(A) * op(B) + beta * C"""
    if exp_bits is not None and sig_bits is not None:
        if exp_bits == 23 and sig_bits == 52:
            precision = 'fp64'
        elif exp_bits == 8 and sig_bits == 23:
            precision = 'fp32'
        else:
            precision = {'exp_bits': exp_bits, 'sig_bits': sig_bits, 'rmode': rmode or 1}

    is_complex = jnp.iscomplexobj(A) or jnp.iscomplexobj(B) or jnp.iscomplexobj(C) or jnp.iscomplexobj(alpha) or jnp.iscomplexobj(beta)
    dtype = jnp.complex128 if is_complex else A.dtype
    A = jnp.array(A, dtype=dtype)
    B = jnp.array(B, dtype=dtype)
    C = jnp.array(C, dtype=dtype)
    alpha = jnp.array(alpha, dtype=dtype)
    beta = jnp.array(beta, dtype=dtype)
    if A.ndim != 2 or B.ndim != 2 or C.ndim != 2:
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
        alpha = alpha.astype(native_dtype)
        opA = opA.astype(native_dtype)
        opB = opB.astype(native_dtype)
        beta = beta.astype(native_dtype)
        C = C.astype(native_dtype)
        mm = jnp.dot(opA, opB)
        result = alpha * mm + beta * C
        result = result.astype(dtype)
    else:
        alpha = rounding(alpha, precision)
        beta = rounding(beta, precision)
        A = rounding(A, precision)
        B = rounding(B, precision)
        C = rounding(C, precision)
        conjA = (transA == 'C')
        conjB = (transB == 'C')
        opA = A if transA == 'N' else A.T
        opB = B if transB == 'N' else B.T
        mm = rounding(jnp.dot(opA, opB), precision)
        alpha_mm = rounding(alpha * mm, precision)
        beta_C = rounding(beta * C, precision)
        result = rounding(alpha_mm + beta_C, precision)
    return result.astype(dtype)

def symm(alpha, A, B, beta, C, side='L', uplo='U', precision='fp64', exp_bits=None, sig_bits=None, rmode=None):
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
    return gemm(alpha, A, B, beta, C, transA=transA, transB=transB, precision=precision)

def hemm(alpha, A, B, beta, C, side='L', uplo='U', precision='fp64', exp_bits=None, sig_bits=None, rmode=None):
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
    return gemm(alpha, A, B, beta, C, transA=transA, transB=transB, precision=precision)



def syrk(alpha, A, beta, C, trans='N', uplo='U', precision='fp64', exp_bits=None, sig_bits=None, rmode=None):
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
    return gemm(alpha, A, A, beta, C, transA=transA, transB=transB, precision=precision)


def herk(alpha, A, beta, C, trans='N', uplo='U', precision='fp64', exp_bits=None, sig_bits=None, rmode=None):
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
    return gemm(alpha, A, A, beta, C, transA=transA, transB=transB, precision=precision)

def syr2k(alpha, A, B, beta, C, trans='N', uplo='U', precision='fp64', exp_bits=None, sig_bits=None, rmode=None):
    """C = alpha * A * B^T + alpha * B * A^T + beta * C, C symmetric"""
    if exp_bits is not None and sig_bits is not None:
        if exp_bits == 23 and sig_bits == 52:
            precision = 'fp64'
        elif exp_bits == 8 and sig_bits == 23:
            precision = 'fp32'
        else:
            precision = {'exp_bits': exp_bits, 'sig_bits': sig_bits, 'rmode': rmode or 1}
    if trans == 'N':
        mm1 = gemm(alpha, A, B, 0.0, jnp.zeros_like(C), transA='N', transB='T', precision=precision)
        mm2 = gemm(alpha, B, A, 0.0, jnp.zeros_like(C), transA='N', transB='T', precision=precision)
    else:
        mm1 = gemm(alpha, A, B, 0.0, jnp.zeros_like(C), transA='T', transB='N', precision=precision)
        mm2 = gemm(alpha, B, A, 0.0, jnp.zeros_like(C), transA='T', transB='N', precision=precision)
    sum_mm = rounding(mm1 + mm2, precision) if get_dtype(precision)[0] is None else mm1 + mm2
    beta_C = rounding(beta * C, precision) if get_dtype(precision)[0] is None else beta * C
    result = rounding(sum_mm + beta_C, precision) if get_dtype(precision)[0] is None else sum_mm + beta_C
    return result

def her2k(alpha, A, B, beta, C, trans='N', uplo='U', precision='fp64', exp_bits=None, sig_bits=None, rmode=None):
    """C = alpha * A * B^H + alpha * B * A^H + beta * C, C Hermitian"""
    if exp_bits is not None and sig_bits is not None:
        if exp_bits == 23 and sig_bits == 52:
            precision = 'fp64'
        elif exp_bits == 8 and sig_bits == 23:
            precision = 'fp32'
        else:
            precision = {'exp_bits': exp_bits, 'sig_bits': sig_bits, 'rmode': rmode or 1}
    if trans == 'N':
        mm1 = gemm(alpha, A, B, 0.0, jnp.zeros_like(C), transA='N', transB='C', precision=precision)
        mm2 = gemm(alpha, B, A, 0.0, jnp.zeros_like(C), transA='N', transB='C', precision=precision)
    else:
        mm1 = gemm(alpha, A, B, 0.0, jnp.zeros_like(C), transA='C', transB='N', precision=precision)
        mm2 = gemm(alpha, B, A, 0.0, jnp.zeros_like(C), transA='C', transB='N', precision=precision)
    sum_mm = rounding(mm1 + mm2, precision) if get_dtype(precision)[0] is None else mm1 + mm2
    beta_C = rounding(beta * C, precision) if get_dtype(precision)[0] is None else beta * C
    result = rounding(sum_mm + beta_C, precision) if get_dtype(precision)[0] is None else sum_mm + beta_C
    return result

def trmm(alpha, A, B, side='L', uplo='U', transA='N', diag='N', precision='fp64', exp_bits=None, sig_bits=None, rmode=None):
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
        return gemm(alpha, A, B, 0.0, jnp.zeros_like(B), transA=transA, transB=transB, precision=precision)
    else:
        transB = transA
        transA = 'N'
        return gemm(alpha, B, A, 0.0, jnp.zeros_like(B), transA=transA, transB=transB, precision=precision)

def low_prec_trsm(alpha, A, B, side='L', uplo='U', transA='N', diag='N', precision='fp64'):
    """Low precision triangular solve for matrix B"""
    m, n = B.shape
    result = jnp.zeros_like(B)
    for col in range(n):
        b = B[:, col]
        x = low_prec_trsv(A, b, uplo, transA, diag, precision)
        result = result.at[:, col].set(x)
    result = rounding(alpha * result, precision)
    return result

def trsm(alpha, A, B, side='L', uplo='U', transA='N', diag='N', precision='fp64', exp_bits=None, sig_bits=None, rmode=None):
    """Solve op(A) * X = alpha * B or X * op(A) = alpha * B, A triangular"""
    if exp_bits is not None and sig_bits is not None:
        if exp_bits == 23 and sig_bits == 52:
            precision = 'fp64'
        elif exp_bits == 8 and sig_bits == 23:
            precision = 'fp32'
        else:
            precision = {'exp_bits': exp_bits, 'sig_bits': sig_bits, 'rmode': rmode or 1}
    is_complex = jnp.iscomplexobj(A) or jnp.iscomplexobj(B) or jnp.iscomplexobj(alpha)
    dtype = jnp.complex128 if is_complex else A.dtype
    A = jnp.array(A, dtype=dtype)
    B = jnp.array(B, dtype=dtype)
    alpha = jnp.array(alpha, dtype=dtype)
    if A.ndim != 2 or B.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square 2D, B must be 2D")
    native_real, native_complex = get_dtype(precision)
    native_dtype = native_complex if is_complex else native_real
    if is_complex and precision in ['half', 'bf16']:
        native_dtype = None
    if native_dtype is not None:
        alpha = alpha.astype(native_dtype)
        A = A.astype(native_dtype)
        B = B.astype(native_dtype)
        opA = A if transA == 'N' else (A.T if transA == 'T' else A.conj().T)
        X = jnp.linalg.solve(opA, alpha * B)
        X = X.astype(dtype)
    else:
        A = rounding(A, precision)
        B = rounding(B, precision)
        alpha = rounding(alpha, precision)
        if side == 'L':
            X = low_prec_trsm(alpha, A, B, side, uplo, transA, diag, precision)
        else:
            transA_right = 'N' if transA == 'N' else ('T' if transA == 'C' else 'C')
            uplo_right = 'L' if uplo == 'U' else 'U'
            X = low_prec_trsm(alpha, A.T, B.T, 'L', uplo_right, transA_right, diag, precision).T
    return X

if __name__ == "__main__":
    import jax
    jax.config.update("jax_enable_x64", True)  # Enable 64-bit precision for testing
    key = jax.random.PRNGKey(42)

    # Test Level 1: axpy, dot, dotc
    n = 5
    x = jax.random.normal(key, (n,))
    y = jax.random.normal(key, (n,))
    x_c = jax.random.normal(key, (n,)) + 1j * jax.random.normal(key, (n,))
    y_c = jax.random.normal(key, (n,)) + 1j * jax.random.normal(key, (n,))
    alpha = 2.0

    print("Level 1 Tests:")
    axpy_fp64 = axpy(alpha, x, y, 'fp64')
    axpy_fp32 = axpy(alpha, x, y, 'fp32')
    print(f"axpy (fp64): {axpy_fp64[:3]}")
    print(f"axpy (fp32): {axpy_fp32[:3]}")

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
    A = jax.random.normal(key, (m, n))
    A_c = jax.random.normal(key, (m, m)) + 1j * jax.random.normal(key, (m, m))
    A_c = A_c + A_c.conj().T  # Make Hermitian
    x = jax.random.normal(key, (n,))
    y = jax.random.normal(key, (m,))
    x_c = jax.random.normal(key, (m,)) + 1j * jax.random.normal(key, (m,))
    alpha, beta = 1.5, 0.5

    print("\nLevel 2 Tests:")
    gemv_fp64 = gemv(alpha, A, x, beta, y, 'N', 'fp64')
    gemv_fp32 = gemv(alpha, A, x, beta, y, 'N', 'fp32')
    print(f"gemv (fp64): {gemv_fp64[:3]}")
    print(f"gemv (fp32): {gemv_fp32[:3]}")

    her_fp64 = her(alpha, x_c, A_c, 'U', 'fp64')
    her_fp32 = her(alpha, x_c, A_c, 'U', 'fp32')
    print(f"her (fp64): \n{her_fp64[:2, :2]}")
    print(f"her (fp32): \n{her_fp32[:2, :2]}")

    # Test Level 3: gemm, herk
    m, n, k = 3, 3, 2
    A = jax.random.normal(key, (m, k))
    B = jax.random.normal(key, (k, n))
    C = jax.random.normal(key, (m, n))
    A_c = jax.random.normal(key, (m, k)) + 1j * jax.random.normal(key, (m, k))
    C_c = jax.random.normal(key, (m, m)) + 1j * jax.random.normal(key, (m, m))
    C_c = C_c + C_c.conj().T  # Make Hermitian
    alpha, beta = 1.0, 0.5

    print("\nLevel 3 Tests:")
    gemm_fp64 = gemm(alpha, A, B, beta, C, 'N', 'N', 'fp64')
    gemm_fp32 = gemm(alpha, A, B, beta, C, 'N', 'N', 'fp32')
    print(f"gemm (fp64): \n{gemm_fp64}")
    print(f"gemm (fp32): \n{gemm_fp32}")

    herk_fp64 = herk(alpha, A_c, beta, C_c, 'N', 'U', 'fp64')
    herk_fp32 = herk(alpha, A_c, beta, C_c, 'N', 'U', 'fp32')
    print(f"herk (fp64): \n{herk_fp64}")
    print(f"herk (fp32): \n{herk_fp32}")

    # Additional tests for correctness
    print("\nAdditional Tests for Correctness:")

    # Test dot with fp32 simulation vs native fp32
    n = 5
    x = jax.random.normal(key, (n,))
    y = jax.random.normal(key, (n,))
    dot_sim_fp32 = dot(x, y, precision='fp32')
    dot_custom_fp32 = dot(x, y, exp_bits=8, sig_bits=23, rmode=1)
    dot_native_fp32 = jnp.sum(x.astype(jnp.float32) * y.astype(jnp.float32)).item()
    print(f"dot simulated fp32: {dot_sim_fp32:.6f}")
    print(f"dot custom fp32: {dot_custom_fp32:.6f}")
    print(f"dot native fp32: {dot_native_fp32:.6f}")

    # Test dot with half simulation vs native half
    dot_sim_half = dot(x, y, precision='half')
    dot_custom_half = dot(x, y, exp_bits=5, sig_bits=10, rmode=1)
    dot_native_half = jnp.sum(x.astype(jnp.float16) * y.astype(jnp.float16)).item()
    print(f"dot simulated half: {dot_sim_half:.6f}")
    print(f"dot custom half: {dot_custom_half:.6f}")
    print(f"dot native half: {dot_native_half:.6f}")

    # Test gemm with fp32 simulation vs native fp32
    m, n, k = 3, 3, 2
    A = jax.random.normal(key, (m, k))
    B = jax.random.normal(key, (k, n))
    C = jax.random.normal(key, (m, n))
    alpha, beta = 1.0, 0.5
    gemm_sim_fp32 = gemm(alpha, A, B, beta, C, 'N', 'N', 'fp32')
    gemm_custom_fp32 = gemm(alpha, A, B, beta, C, 'N', 'N', exp_bits=8, sig_bits=23, rmode=1)
    opA_native = A.astype(jnp.float32)
    opB_native = B.astype(jnp.float32)
    gemm_native_fp32 = alpha * jnp.dot(opA_native, opB_native) + beta * C.astype(jnp.float32)
    print(f"gemm simulated fp32:\n{gemm_sim_fp32}")
    print(f"gemm custom fp32:\n{gemm_custom_fp32}")
    print(f"gemm native fp32:\n{gemm_native_fp32}")