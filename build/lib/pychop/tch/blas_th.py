import torch
import numpy as np
import pychop
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Set pychop backend and device
pychop.backend('torch')
device = torch.device('cpu')

# Precision configurations
precision_configs = {
    'q52': {'exp_bits': 5, 'sig_bits': 2, 'rmode': 1},
    'q43': {'exp_bits': 4, 'sig_bits': 3, 'rmode': 1},
    'bf16': {'exp_bits': 8, 'sig_bits': 7, 'rmode': 1},
    'half': {'exp_bits': 5, 'sig_bits': 10, 'rmode': 1},
    'tf32': {'exp_bits': 8, 'sig_bits': 10, 'rmode': 1},
    'fp32': {'exp_bits': 8, 'sig_bits': 23, 'rmode': 1},
    'fp64': {'exp_bits': 11, 'sig_bits': 52, 'rmode': 1}
}

precision_fallback = ['q52', 'q43', 'bf16', 'half', 'tf32', 'fp32', 'fp64']

def chop(x, precision_idx=0):
    """Recursive chop function"""
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float64 if not np.iscomplexobj(x) else torch.complex128, device=device)
    if precision_idx >= len(precision_fallback):
        return x
    precision = precision_fallback[precision_idx]
    if precision == 'fp64':
        return x
    ch = pychop.LightChop(**precision_configs[precision])
    result = ch(x)
    if not torch.any(torch.isnan(result)) and not torch.any(torch.isinf(result)):
        return result.to(torch.float64 if not torch.is_complex(x) else torch.complex128).to(device)
    logging.debug(f"Chop: Precision {precision} failed, escalating to {precision_fallback[precision_idx + 1]}")
    return chop(x, precision_idx + 1)

def rounding(x, precision):
    """Round tensor to specified precision"""
    return chop(x, precision_idx=precision_fallback.index(precision))

def mixed_precision_op(op, x, precision, y=None):
    """Mixed-precision operation"""
    x = rounding(x, precision)
    if y is None:
        unrounded = op(x)
    else:
        y = rounding(y, precision)
        unrounded = op(x, y)
    if precision == 'fp64':
        return unrounded.to(device)
    result = chop(unrounded, precision_idx=precision_fallback.index(precision))
    return result.to(device)

# Level 1 BLAS: Vector-Vector Operations
def axpy(alpha, x, y, precision='fp64'):
    """y = alpha * x + y"""
    x = torch.as_tensor(x, dtype=torch.float64 if not torch.is_complex(x) else torch.complex128, device=device)
    y = torch.as_tensor(y, dtype=x.dtype, device=device)
    if x.shape != y.shape or x.dim() != 1:
        raise ValueError("x and y must be 1D vectors of same length")
    alpha = torch.tensor(alpha, dtype=x.dtype, device=device)
    return mixed_precision_op(lambda a, b: rounding(alpha, precision) * a + b, x, precision, y)

def scal(alpha, x, precision='fp64'):
    """x = alpha * x"""
    x = torch.as_tensor(x, dtype=torch.float64 if not torch.is_complex(x) else torch.complex128, device=device)
    if x.dim() != 1:
        raise ValueError("x must be a 1D vector")
    alpha = torch.tensor(alpha, dtype=x.dtype, device=device)
    return mixed_precision_op(lambda a: rounding(alpha, precision) * a, x, precision)

def copy(x, y, precision='fp64'):
    """y = x"""
    x = torch.as_tensor(x, dtype=torch.float64 if not torch.is_complex(x) else torch.complex128, device=device)
    y = torch.as_tensor(y, dtype=x.dtype, device=device)
    if x.shape != y.shape or x.dim() != 1:
        raise ValueError("x and y must be 1D vectors of same length")
    return mixed_precision_op(lambda a: a, x, precision)

def swap(x, y, precision='fp64'):
    """x <-> y"""
    x = torch.as_tensor(x, dtype=torch.float64 if not torch.is_complex(x) else torch.complex128, device=device)
    y = torch.as_tensor(y, dtype=x.dtype, device=device)
    if x.shape != y.shape or x.dim() != 1:
        raise ValueError("x and y must be 1D vectors of same length")
    x_rounded = rounding(x, precision)
    y_rounded = rounding(y, precision)
    return y_rounded, x_rounded

def dot(x, y, precision='fp64'):
    """x . y"""
    x = torch.as_tensor(x, dtype=torch.float64 if not torch.is_complex(x) else torch.complex128, device=device)
    y = torch.as_tensor(y, dtype=x.dtype, device=device)
    if x.shape != y.shape or x.dim() != 1:
        raise ValueError("x and y must be 1D vectors of same length")
    return mixed_precision_op(lambda a, b: torch.sum(a * b), x, precision, y).item()

def dotc(x, y, precision='fp64'):
    """x . conj(y) (complex)"""
    x = torch.as_tensor(x, dtype=torch.complex128, device=device)
    y = torch.as_tensor(y, dtype=torch.complex128, device=device)
    if x.shape != y.shape or x.dim() != 1:
        raise ValueError("x and y must be 1D vectors of same length")
    return mixed_precision_op(lambda a, b: torch.sum(a * torch.conj(b)), x, precision, y).item()

def dotu(x, y, precision='fp64'):
    """x . y (complex, no conjugate)"""
    x = torch.as_tensor(x, dtype=torch.complex128, device=device)
    y = torch.as_tensor(y, dtype=torch.complex128, device=device)
    if x.shape != y.shape or x.dim() != 1:
        raise ValueError("x and y must be 1D vectors of same length")
    return mixed_precision_op(lambda a, b: torch.sum(a * b), x, precision, y).item()

def nrm2(x, precision='fp64'):
    """Euclidean norm of x"""
    x = torch.as_tensor(x, dtype=torch.float64 if not torch.is_complex(x) else torch.complex128, device=device)
    if x.dim() != 1:
        raise ValueError("x must be a 1D vector")
    return mixed_precision_op(lambda a: torch.sqrt(torch.sum(torch.abs(a)**2)), x, precision).item()

def asum(x, precision='fp64'):
    """Sum of absolute values of x"""
    x = torch.as_tensor(x, dtype=torch.float64 if not torch.is_complex(x) else torch.complex128, device=device)
    if x.dim() != 1:
        raise ValueError("x must be a 1D vector")
    return mixed_precision_op(lambda a: torch.sum(torch.abs(a)), x, precision).item()

def rot(x, y, c, s, precision='fp64'):
    """Apply Givens rotation: x' = c*x + s*y, y' = -s*x + c*y"""
    x = torch.as_tensor(x, dtype=torch.float64 if not torch.is_complex(x) else torch.complex128, device=device)
    y = torch.as_tensor(y, dtype=x.dtype, device=device)
    if x.shape != y.shape or x.dim() != 1:
        raise ValueError("x and y must be 1D vectors of same length")
    c = torch.tensor(c, dtype=torch.float64, device=device)
    s = torch.tensor(s, dtype=x.dtype, device=device)
    def rot_op(a, b): return (rounding(c, precision) * a + rounding(s, precision) * b,
                              -rounding(s, precision) * a + rounding(c, precision) * b)
    x_new, y_new = rot_op(rounding(x, precision), rounding(y, precision))
    return chop(x_new, precision_fallback.index(precision)), chop(y_new, precision_fallback.index(precision))

def rotg(a, b, precision='fp64'):
    """Generate Givens rotation: compute c, s, r, z for rotation"""
    a = torch.tensor(a, dtype=torch.float64, device=device)
    b = torch.tensor(b, dtype=torch.float64, device=device)
    def rotg_op(a, b):
        r = torch.sqrt(a**2 + b**2)
        c = a / r if r != 0 else 1.0
        s = b / r if r != 0 else 0.0
        z = s if c != 0 else 1.0
        return r, c, s, z
    r, c, s, z = rotg_op(rounding(a, precision), rounding(b, precision))
    return (chop(r, precision_fallback.index(precision)).item(),
            chop(c, precision_fallback.index(precision)).item(),
            chop(s, precision_fallback.index(precision)).item(),
            chop(z, precision_fallback.index(precision)).item())

def rotm(x, y, param, precision='fp64'):
    """Apply modified rotation (Hessenberg)"""
    x = torch.as_tensor(x, dtype=torch.float64, device=device)
    y = torch.as_tensor(y, dtype=torch.float64, device=device)
    param = torch.as_tensor(param, dtype=torch.float64, device=device)  # [flag, h11, h21, h12, h22]
    if x.shape != y.shape or x.dim() != 1 or param.shape != (5,):
        raise ValueError("x, y must be 1D vectors of same length, param must be 5-element vector")
    def rotm_op(a, b):
        flag = param[0]
        if flag == -1:
            h11, h21, h12, h22 = param[1:5]
            x_new = h11 * a + h12 * b
            y_new = h21 * a + h22 * b
        elif flag == 0:
            x_new, y_new = a, b
        elif flag == 1:
            h11, h12 = param[1], param[3]
            x_new = h11 * a + h12 * b
            y_new = b
        else:  # flag == 2
            h21, h22 = param[2], param[4]
            x_new = a
            y_new = h21 * a + h22 * b
        return x_new, y_new
    x_new, y_new = rotm_op(rounding(x, precision), rounding(y, precision))
    return chop(x_new, precision_fallback.index(precision)), chop(y_new, precision_fallback.index(precision))

def rotmg(d1, d2, x1, y1, precision='fp64'):
    """Generate modified rotation parameters"""
    d1 = torch.tensor(d1, dtype=torch.float64, device=device)
    d2 = torch.tensor(d2, dtype=torch.float64, device=device)
    x1 = torch.tensor(x1, dtype=torch.float64, device=device)
    y1 = torch.tensor(y1, dtype=torch.float64, device=device)
    def rotmg_op(d1, d2, x1, y1):
        param = torch.zeros(5, dtype=torch.float64, device=device)
        if d1 == 0 or d2 == 0 or x1 == 0:
            param[0] = -1
            return param
        param[0] = -1  # Flag for full matrix
        param[1] = 1.0  # h11
        param[2] = 0.0  # h21
        param[3] = 0.0  # h12
        param[4] = 1.0  # h22
        return param
    param = rotmg_op(rounding(d1, precision), rounding(d2, precision), rounding(x1, precision), rounding(y1, precision))
    return chop(param, precision_fallback.index(precision))

def iamax(x, precision='fp64'):
    """Index of maximum absolute value"""
    x = torch.as_tensor(x, dtype=torch.float64 if not torch.is_complex(x) else torch.complex128, device=device)
    if x.dim() != 1:
        raise ValueError("x must be a 1D vector")
    x_rounded = rounding(x, precision)
    return torch.argmax(torch.abs(x_rounded)).item()

def iamin(x, precision='fp64'):
    """Index of minimum absolute value"""
    x = torch.as_tensor(x, dtype=torch.float64 if not torch.is_complex(x) else torch.complex128, device=device)
    if x.dim() != 1:
        raise ValueError("x must be a 1D vector")
    x_rounded = rounding(x, precision)
    return torch.argmin(torch.abs(x_rounded)).item()

# Level 2 BLAS: Matrix-Vector Operations
def gemv(alpha, A, x, beta, y, trans='N', precision='fp64'):
    """y = alpha * op(A) * x + beta * y, op(A) = A or A^T"""
    A = torch.as_tensor(A, dtype=torch.float64 if not torch.is_complex(A) else torch.complex128, device=device)
    x = torch.as_tensor(x, dtype=A.dtype, device=device)
    y = torch.as_tensor(y, dtype=A.dtype, device=device)
    if A.dim() != 2 or x.dim() != 1 or y.dim() != 1:
        raise ValueError("A must be 2D, x and y must be 1D")
    m, n = A.shape
    if trans == 'N' and (n != x.shape[0] or m != y.shape[0]):
        raise ValueError("Incompatible dimensions for A*x")
    if trans == 'T' and (m != x.shape[0] or n != y.shape[0]):
        raise ValueError("Incompatible dimensions for A^T*x")
    alpha = torch.tensor(alpha, dtype=A.dtype, device=device)
    beta = torch.tensor(beta, dtype=A.dtype, device=device)
    def gemv_op(A, x): return rounding(alpha, precision) * (torch.matmul(A, x) if trans == 'N' else torch.matmul(A.T, x)) + rounding(beta, precision) * y
    return mixed_precision_op(gemv_op, A, precision, x)

def gbmv(alpha, A, x, beta, y, kl, ku, trans='N', precision='fp64'):
    """y = alpha * op(A) * x + beta * y for band matrix A"""
    A = torch.as_tensor(A, dtype=torch.float64, device=device)
    x = torch.as_tensor(x, dtype=A.dtype, device=device)
    y = torch.as_tensor(y, dtype=A.dtype, device=device)
    if A.dim() != 2 or x.dim() != 1 or y.dim() != 1:
        raise ValueError("A must be 2D, x and y must be 1D")
    m, n = A.shape
    if trans == 'N' and (n != x.shape[0] or m != y.shape[0]):
        raise ValueError("Incompatible dimensions")
    return gemv(alpha, A, x, beta, y, trans, precision)

def symv(alpha, A, x, beta, y, uplo='U', precision='fp64'):
    """y = alpha * A * x + beta * y, A symmetric"""
    A = torch.as_tensor(A, dtype=torch.float64, device=device)
    x = torch.as_tensor(x, dtype=A.dtype, device=device)
    y = torch.as_tensor(y, dtype=A.dtype, device=device)
    if A.dim() != 2 or x.dim() != 1 or y.dim() != 1 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square 2D, x and y must be 1D")
    return gemv(alpha, A, x, beta, y, 'N', precision)

def sbmv(alpha, A, x, beta, y, k, uplo='U', precision='fp64'):
    """y = alpha * A * x + beta * y, A symmetric band"""
    A = torch.as_tensor(A, dtype=torch.float64, device=device)
    x = torch.as_tensor(x, dtype=A.dtype, device=device)
    y = torch.as_tensor(y, dtype=A.dtype, device=device)
    if A.dim() != 2 or x.dim() != 1 or y.dim() != 1:
        raise ValueError("A must be 2D, x and y must be 1D")
    return gemv(alpha, A, x, beta, y, 'N', precision)

def hemv(alpha, A, x, beta, y, uplo='U', precision='fp64'):
    """y = alpha * A * x + beta * y, A Hermitian"""
    A = torch.as_tensor(A, dtype=torch.complex128, device=device)
    x = torch.as_tensor(x, dtype=A.dtype, device=device)
    y = torch.as_tensor(y, dtype=A.dtype, device=device)
    if A.dim() != 2 or x.dim() != 1 or y.dim() != 1 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square 2D, x and y must be 1D")
    return gemv(alpha, A, x, beta, y, 'N', precision)

def hbmv(alpha, A, x, beta, y, k, uplo='U', precision='fp64'):
    """y = alpha * A * x + beta * y, A Hermitian band"""
    A = torch.as_tensor(A, dtype=torch.complex128, device=device)
    x = torch.as_tensor(x, dtype=A.dtype, device=device)
    y = torch.as_tensor(y, dtype=A.dtype, device=device)
    return gemv(alpha, A, x, beta, y, 'N', precision)

def spmv(alpha, A, x, beta, y, uplo='U', precision='fp64'):
    """y = alpha * A * x + beta * y, A symmetric packed"""
    A = torch.as_tensor(A, dtype=torch.float64, device=device)
    x = torch.as_tensor(x, dtype=A.dtype, device=device)
    y = torch.as_tensor(y, dtype=A.dtype, device=device)
    n = int((np.sqrt(8 * A.shape[0] + 1) - 1) / 2)
    A_dense = torch.zeros((n, n), dtype=A.dtype, device=device)
    idx = 0
    for i in range(n):
        for j in range(i, n) if uplo == 'U' else range(i + 1):
            A_dense[i, j] = A[idx]
            if i != j:
                A_dense[j, i] = A[idx]
            idx += 1
    return gemv(alpha, A_dense, x, beta, y, 'N', precision)

def hpmv(alpha, A, x, beta, y, uplo='U', precision='fp64'):
    """y = alpha * A * x + beta * y, A Hermitian packed"""
    A = torch.as_tensor(A, dtype=torch.complex128, device=device)
    x = torch.as_tensor(x, dtype=A.dtype, device=device)
    y = torch.as_tensor(y, dtype=A.dtype, device=device)
    n = int((np.sqrt(8 * A.shape[0] + 1) - 1) / 2)
    A_dense = torch.zeros((n, n), dtype=A.dtype, device=device)
    idx = 0
    for i in range(n):
        for j in range(i, n) if uplo == 'U' else range(i + 1):
            A_dense[i, j] = A[idx]
            if i != j:
                A_dense[j, i] = torch.conj(A[idx])
            idx += 1
    return gemv(alpha, A_dense, x, beta, y, 'N', precision)

def trmv(A, x, uplo='U', trans='N', diag='N', precision='fp64'):
    """x = op(A) * x, A triangular"""
    A = torch.as_tensor(A, dtype=torch.float64 if not torch.is_complex(A) else torch.complex128, device=device)
    x = torch.as_tensor(x, dtype=A.dtype, device=device)
    if A.dim() != 2 or x.dim() != 1 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square 2D, x must be 1D")
    def trmv_op(A, x): return torch.matmul(A if trans == 'N' else A.T, x)
    return mixed_precision_op(trmv_op, A, precision, x)

def trsv(A, b, uplo='U', trans='N', diag='N', precision='fp64'):
    """Solve op(A) * x = b, A triangular"""
    A = torch.as_tensor(A, dtype=torch.float64 if not torch.is_complex(A) else torch.complex128, device=device)
    b = torch.as_tensor(b, dtype=A.dtype, device=device)
    if A.dim() != 2 or b.dim() != 1 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square 2D, b must be 1D")
    A_rounded = rounding(A, precision)
    b_rounded = rounding(b, precision)
    x = torch.linalg.solve_triangular(A_rounded if trans == 'N' else A_rounded.T, b_rounded, upper=(uplo == 'U'), unitriangular=(diag == 'U'))
    return chop(x, precision_fallback.index(precision))

def tbmv(A, x, k, uplo='U', trans='N', diag='N', precision='fp64'):
    """x = op(A) * x, A triangular band"""
    A = torch.as_tensor(A, dtype=torch.float64, device=device)
    x = torch.as_tensor(x, dtype=A.dtype, device=device)
    return trmv(A, x, uplo, trans, diag, precision)

def tbsv(A, b, k, uplo='U', trans='N', diag='N', precision='fp64'):
    """Solve op(A) * x = b, A triangular band"""
    A = torch.as_tensor(A, dtype=torch.float64, device=device)
    b = torch.as_tensor(b, dtype=A.dtype, device=device)
    return trsv(A, b, uplo, trans, diag, precision)

def tpmv(A, x, uplo='U', trans='N', diag='N', precision='fp64'):
    """x = op(A) * x, A triangular packed"""
    A = torch.as_tensor(A, dtype=torch.float64 if not torch.is_complex(A) else torch.complex128, device=device)
    x = torch.as_tensor(x, dtype=A.dtype, device=device)
    n = int((np.sqrt(8 * A.shape[0] + 1) - 1) / 2)
    A_dense = torch.zeros((n, n), dtype=A.dtype, device=device)
    idx = 0
    for i in range(n):
        for j in range(i, n) if uplo == 'U' else range(i + 1):
            A_dense[i, j] = A[idx]
            idx += 1
    return trmv(A_dense, x, uplo, trans, diag, precision)

def tpsv(A, b, uplo='U', trans='N', diag='N', precision='fp64'):
    """Solve op(A) * x = b, A triangular packed"""
    A = torch.as_tensor(A, dtype=torch.float64 if not torch.is_complex(A) else torch.complex128, device=device)
    b = torch.as_tensor(b, dtype=A.dtype, device=device)
    n = int((np.sqrt(8 * A.shape[0] + 1) - 1) / 2)
    A_dense = torch.zeros((n, n), dtype=A.dtype, device=device)
    idx = 0
    for i in range(n):
        for j in range(i, n) if uplo == 'U' else range(i + 1):
            A_dense[i, j] = A[idx]
            idx += 1
    return trsv(A_dense, b, uplo, trans, diag, precision)

def ger(alpha, x, y, A, precision='fp64'):
    """A = A + alpha * x * y^T"""
    A = torch.as_tensor(A, dtype=torch.float64 if not torch.is_complex(A) else torch.complex128, device=device)
    x = torch.as_tensor(x, dtype=A.dtype, device=device)
    y = torch.as_tensor(y, dtype=A.dtype, device=device)
    if A.dim() != 2 or x.dim() != 1 or y.dim() != 1 or A.shape[0] != x.shape[0] or A.shape[1] != y.shape[0]:
        raise ValueError("Incompatible dimensions")
    alpha = torch.tensor(alpha, dtype=A.dtype, device=device)
    return mixed_precision_op(lambda a, b: A + rounding(alpha, precision) * torch.outer(a, b), x, precision, y)

def syr(alpha, x, A, uplo='U', precision='fp64'):
    """A = A + alpha * x * x^T, A symmetric"""
    A = torch.as_tensor(A, dtype=torch.float64, device=device)
    x = torch.as_tensor(x, dtype=A.dtype, device=device)
    if A.dim() != 2 or x.dim() != 1 or A.shape[0] != A.shape[1] or A.shape[0] != x.shape[0]:
        raise ValueError("Incompatible dimensions")
    alpha = torch.tensor(alpha, dtype=A.dtype, device=device)
    return mixed_precision_op(lambda a: A + rounding(alpha, precision) * torch.outer(a, a), x, precision)

def spr(alpha, x, A, uplo='U', precision='fp64'):
    """A = A + alpha * x * x^T, A symmetric packed"""
    A = torch.as_tensor(A, dtype=torch.float64, device=device)
    x = torch.as_tensor(x, dtype=A.dtype, device=device)
    n = int((np.sqrt(8 * A.shape[0] + 1) - 1) / 2)
    A_dense = torch.zeros((n, n), dtype=A.dtype, device=device)
    idx = 0
    for i in range(n):
        for j in range(i, n) if uplo == 'U' else range(i + 1):
            A_dense[i, j] = A[idx]
            if i != j:
                A_dense[j, i] = A[idx]
            idx += 1
    A_new = syr(alpha, x, A_dense, uplo, precision)
    A_packed = torch.zeros_like(A)
    idx = 0
    for i in range(n):
        for j in range(i, n) if uplo == 'U' else range(i + 1):
            A_packed[idx] = A_new[i, j]
            idx += 1
    return A_packed

def syr2(alpha, x, y, A, uplo='U', precision='fp64'):
    """A = A + alpha * x * y^T + alpha * y * x^T, A symmetric"""
    A = torch.as_tensor(A, dtype=torch.float64, device=device)
    x = torch.as_tensor(x, dtype=A.dtype, device=device)
    y = torch.as_tensor(y, dtype=A.dtype, device=device)
    if A.dim() != 2 or x.dim() != 1 or y.dim() != 1 or A.shape[0] != A.shape[1] or A.shape[0] != x.shape[0]:
        raise ValueError("Incompatible dimensions")
    alpha = torch.tensor(alpha, dtype=A.dtype, device=device)
    return mixed_precision_op(lambda a, b: A + rounding(alpha, precision) * (torch.outer(a, b) + torch.outer(b, a)), x, precision, y)

def spr2(alpha, x, y, A, uplo='U', precision='fp64'):
    """A = A + alpha * x * y^T + alpha * y * x^T, A symmetric packed"""
    A = torch.as_tensor(A, dtype=torch.float64, device=device)
    x = torch.as_tensor(x, dtype=A.dtype, device=device)
    y = torch.as_tensor(y, dtype=A.dtype, device=device)
    n = int((np.sqrt(8 * A.shape[0] + 1) - 1) / 2)
    A_dense = torch.zeros((n, n), dtype=A.dtype, device=device)
    idx = 0
    for i in range(n):
        for j in range(i, n) if uplo == 'U' else range(i + 1):
            A_dense[i, j] = A[idx]
            if i != j:
                A_dense[j, i] = A[idx]
            idx += 1
    A_new = syr2(alpha, x, y, A_dense, uplo, precision)
    A_packed = torch.zeros_like(A)
    idx = 0
    for i in range(n):
        for j in range(i, n) if uplo == 'U' else range(i + 1):
            A_packed[idx] = A_new[i, j]
            idx += 1
    return A_packed

def her(alpha, x, A, uplo='U', precision='fp64'):
    """A = A + alpha * x * x^H, A Hermitian"""
    A = torch.as_tensor(A, dtype=torch.complex128, device=device)
    x = torch.as_tensor(x, dtype=A.dtype, device=device)
    if A.dim() != 2 or x.dim() != 1 or A.shape[0] != A.shape[1] or A.shape[0] != x.shape[0]:
        raise ValueError("Incompatible dimensions")
    alpha = torch.tensor(alpha, dtype=torch.float64, device=device)
    return mixed_precision_op(lambda a: A + rounding(alpha, precision) * torch.outer(a, torch.conj(a)), x, precision)

def hpr(alpha, x, A, uplo='U', precision='fp64'):
    """A = A + alpha * x * x^H, A Hermitian packed"""
    A = torch.as_tensor(A, dtype=torch.complex128, device=device)
    x = torch.as_tensor(x, dtype=A.dtype, device=device)
    n = int((np.sqrt(8 * A.shape[0] + 1) - 1) / 2)
    A_dense = torch.zeros((n, n), dtype=A.dtype, device=device)
    idx = 0
    for i in range(n):
        for j in range(i, n) if uplo == 'U' else range(i + 1):
            A_dense[i, j] = A[idx]
            if i != j:
                A_dense[j, i] = torch.conj(A[idx])
            idx += 1
    A_new = her(alpha, x, A_dense, uplo, precision)
    A_packed = torch.zeros_like(A)
    idx = 0
    for i in range(n):
        for j in range(i, n) if uplo == 'U' else range(i + 1):
            A_packed[idx] = A_new[i, j]
            idx += 1
    return A_packed

def her2(alpha, x, y, A, uplo='U', precision='fp64'):
    """A = A + alpha * x * y^H + alpha * y * x^H, A Hermitian"""
    A = torch.as_tensor(A, dtype=torch.complex128, device=device)
    x = torch.as_tensor(x, dtype=A.dtype, device=device)
    y = torch.as_tensor(y, dtype=A.dtype, device=device)
    if A.dim() != 2 or x.dim() != 1 or y.dim() != 1 or A.shape[0] != A.shape[1] or A.shape[0] != x.shape[0]:
        raise ValueError("Incompatible dimensions")
    alpha = torch.tensor(alpha, dtype=A.dtype, device=device)
    return mixed_precision_op(lambda a, b: A + rounding(alpha, precision) * (torch.outer(a, torch.conj(b)) + torch.outer(b, torch.conj(a))), x, precision, y)

def hpr2(alpha, x, y, A, uplo='U', precision='fp64'):
    """A = A + alpha * x * y^H + alpha * y * x^H, A Hermitian packed"""
    A = torch.as_tensor(A, dtype=torch.complex128, device=device)
    x = torch.as_tensor(x, dtype=A.dtype, device=device)
    y = torch.as_tensor(y, dtype=A.dtype, device=device)
    n = int((np.sqrt(8 * A.shape[0] + 1) - 1) / 2)
    A_dense = torch.zeros((n, n), dtype=A.dtype, device=device)
    idx = 0
    for i in range(n):
        for j in range(i, n) if uplo == 'U' else range(i + 1):
            A_dense[i, j] = A[idx]
            if i != j:
                A_dense[j, i] = torch.conj(A[idx])
            idx += 1
    A_new = her2(alpha, x, y, A_dense, uplo, precision)
    A_packed = torch.zeros_like(A)
    idx = 0
    for i in range(n):
        for j in range(i, n) if uplo == 'U' else range(i + 1):
            A_packed[idx] = A_new[i, j]
            idx += 1
    return A_packed

# Level 3 BLAS: Matrix-Matrix Operations
def gemm(alpha, A, B, beta, C, transA='N', transB='N', precision='fp64'):
    """C = alpha * op(A) * op(B) + beta * C"""
    A = torch.as_tensor(A, dtype=torch.float64 if not torch.is_complex(A) else torch.complex128, device=device)
    B = torch.as_tensor(B, dtype=A.dtype, device=device)
    C = torch.as_tensor(C, dtype=A.dtype, device=device)
    if A.dim() != 2 or B.dim() != 2 or C.dim() != 2:
        raise ValueError("A, B, C must be 2D")
    alpha = torch.tensor(alpha, dtype=A.dtype, device=device)
    beta = torch.tensor(beta, dtype=A.dtype, device=device)
    opA = A if transA == 'N' else (A.T if transA == 'T' else A.conj().T)
    opB = B if transB == 'N' else (B.T if transB == 'T' else B.conj().T)
    if opA.shape[1] != opB.shape[0] or opA.shape[0] != C.shape[0] or opB.shape[1] != C.shape[1]:
        raise ValueError("Incompatible dimensions")
    return mixed_precision_op(lambda a, b: rounding(alpha, precision) * torch.matmul(a, b) + rounding(beta, precision) * C, opA, precision, opB)

def symm(alpha, A, B, beta, C, side='L', uplo='U', precision='fp64'):
    """C = alpha * A * B + beta * C or alpha * B * A + beta * C, A symmetric"""
    A = torch.as_tensor(A, dtype=torch.float64, device=device)
    B = torch.as_tensor(B, dtype=A.dtype, device=device)
    C = torch.as_tensor(C, dtype=A.dtype, device=device)
    if A.dim() != 2 or B.dim() != 2 or C.dim() != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square 2D, B, C must be 2D")
    alpha = torch.tensor(alpha, dtype=A.dtype, device=device)
    beta = torch.tensor(beta, dtype=A.dtype, device=device)
    def symm_op(a, b): return rounding(alpha, precision) * (torch.matmul(a, b) if side == 'L' else torch.matmul(b, a)) + rounding(beta, precision) * C
    return mixed_precision_op(symm_op, A, precision, B)

def hemm(alpha, A, B, beta, C, side='L', uplo='U', precision='fp64'):
    """C = alpha * A * B + beta * C or alpha * B * A + beta * C, A Hermitian"""
    A = torch.as_tensor(A, dtype=torch.complex128, device=device)
    B = torch.as_tensor(B, dtype=A.dtype, device=device)
    C = torch.as_tensor(C, dtype=A.dtype, device=device)
    if A.dim() != 2 or B.dim() != 2 or C.dim() != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square 2D, B, C must be 2D")
    alpha = torch.tensor(alpha, dtype=A.dtype, device=device)
    beta = torch.tensor(beta, dtype=A.dtype, device=device)
    def hemm_op(a, b): return rounding(alpha, precision) * (torch.matmul(a, b) if side == 'L' else torch.matmul(b, a)) + rounding(beta, precision) * C
    return mixed_precision_op(hemm_op, A, precision, B)

def syrk(alpha, A, beta, C, trans='N', uplo='U', precision='fp64'):
    """C = alpha * A * A^T + beta * C or alpha * A^T * A + beta * C, C symmetric"""
    A = torch.as_tensor(A, dtype=torch.float64, device=device)
    C = torch.as_tensor(C, dtype=A.dtype, device=device)
    if A.dim() != 2 or C.dim() != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("A must be 2D, C must be square 2D")
    alpha = torch.tensor(alpha, dtype=A.dtype, device=device)
    beta = torch.tensor(beta, dtype=A.dtype, device=device)
    def syrk_op(a): return rounding(alpha, precision) * torch.matmul(a, a.T if trans == 'N' else a) + rounding(beta, precision) * C
    return mixed_precision_op(syrk_op, A if trans == 'N' else A.T, precision)

def herk(alpha, A, beta, C, trans='N', uplo='U', precision='fp64'):
    """C = alpha * A * A^H + beta * C or alpha * A^H * A + beta * C, C Hermitian"""
    A = torch.as_tensor(A, dtype=torch.complex128, device=device)
    C = torch.as_tensor(C, dtype=A.dtype, device=device)
    if A.dim() != 2 or C.dim() != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("A must be 2D, C must be square 2D")
    alpha = torch.tensor(alpha, dtype=torch.float64, device=device)
    beta = torch.tensor(beta, dtype=torch.float64, device=device)
    def herk_op(a): return rounding(alpha, precision) * torch.matmul(a, a.conj().T if trans == 'N' else a) + rounding(beta, precision) * C
    return mixed_precision_op(herk_op, A if trans == 'N' else A.conj().T, precision)

def syr2k(alpha, A, B, beta, C, trans='N', uplo='U', precision='fp64'):
    """C = alpha * A * B^T + alpha * B * A^T + beta * C, C symmetric"""
    A = torch.as_tensor(A, dtype=torch.float64, device=device)
    B = torch.as_tensor(B, dtype=A.dtype, device=device)
    C = torch.as_tensor(C, dtype=A.dtype, device=device)
    if A.dim() != 2 or B.dim() != 2 or C.dim() != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("A, B must be 2D, C must be square 2D")
    alpha = torch.tensor(alpha, dtype=A.dtype, device=device)
    beta = torch.tensor(beta, dtype=A.dtype, device=device)
    def syr2k_op(a, b): return rounding(alpha, precision) * (torch.matmul(a, b.T if trans == 'N' else b) + torch.matmul(b, a.T if trans == 'N' else a)) + rounding(beta, precision) * C
    return mixed_precision_op(syr2k_op, A if trans == 'N' else A.T, precision, B if trans == 'N' else B.T)

def her2k(alpha, A, B, beta, C, trans='N', uplo='U', precision='fp64'):
    """C = alpha * A * B^H + alpha * B * A^H + beta * C, C Hermitian"""
    A = torch.as_tensor(A, dtype=torch.complex128, device=device)
    B = torch.as_tensor(B, dtype=A.dtype, device=device)
    C = torch.as_tensor(C, dtype=A.dtype, device=device)
    if A.dim() != 2 or B.dim() != 2 or C.dim() != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("A, B must be 2D, C must be square 2D")
    alpha = torch.tensor(alpha, dtype=A.dtype, device=device)
    beta = torch.tensor(beta, dtype=torch.float64, device=device)
    def her2k_op(a, b): return rounding(alpha, precision) * (torch.matmul(a, b.conj().T if trans == 'N' else b) + torch.matmul(b, a.conj().T if trans == 'N' else a)) + rounding(beta, precision) * C
    return mixed_precision_op(her2k_op, A if trans == 'N' else A.conj().T, precision, B if trans == 'N' else B.conj().T)

def trmm(alpha, A, B, side='L', uplo='U', transA='N', diag='N', precision='fp64'):
    """B = alpha * op(A) * B or alpha * B * op(A), A triangular"""
    A = torch.as_tensor(A, dtype=torch.float64 if not torch.is_complex(A) else torch.complex128, device=device)
    B = torch.as_tensor(B, dtype=A.dtype, device=device)
    if A.dim() != 2 or B.dim() != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square 2D, B must be 2D")
    alpha = torch.tensor(alpha, dtype=A.dtype, device=device)
    opA = A if transA == 'N' else (A.T if transA == 'T' else A.conj().T)
    def trmm_op(a, b): return rounding(alpha, precision) * (torch.matmul(a, b) if side == 'L' else torch.matmul(b, a))
    return mixed_precision_op(trmm_op, opA, precision, B)

def trsm(alpha, A, B, side='L', uplo='U', transA='N', diag='N', precision='fp64'):
    """Solve op(A) * X = alpha * B or X * op(A) = alpha * B, A triangular"""
    A = torch.as_tensor(A, dtype=torch.float64 if not torch.is_complex(A) else torch.complex128, device=device)
    B = torch.as_tensor(B, dtype=A.dtype, device=device)
    if A.dim() != 2 or B.dim() != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square 2D, B must be 2D")
    alpha = torch.tensor(alpha, dtype=A.dtype, device=device)
    opA = A if transA == 'N' else (A.T if transA == 'T' else A.conj().T)
    A_rounded = rounding(opA, precision)
    B_rounded = rounding(rounding(alpha, precision) * B, precision)
    X = torch.linalg.solve_triangular(A_rounded, B_rounded, upper=(uplo == 'U'), unitriangular=(diag == 'U'), left=(side == 'L'))
    return chop(X, precision_fallback.index(precision))

if __name__ == "__main__":
    # Set random seed for reproducibility
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