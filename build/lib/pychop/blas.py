from pychop import LightChop
import torch
import pychop
pychop.backend('torch')


precision_configs = {
    'q52': {'exp_bits': 5, 'sig_bits': 2, 'rmode': 1},
    'q43': {'exp_bits': 4, 'sig_bits': 3, 'rmode': 1},
    'bf16': {'exp_bits': 8, 'sig_bits': 7, 'rmode': 1},
    'half': {'exp_bits': 5, 'sig_bits': 10, 'rmode': 1},
    'tf32': {'exp_bits': 8, 'sig_bits': 10, 'rmode': 1},
    'fp32': {'exp_bits': 8, 'sig_bits': 23, 'rmode': 1},
    'fp64': {'exp_bits': 11, 'sig_bits': 52, 'rmode': 1}
}

precision_fallback = ['q52', 'q43', 'bf16', 'half', 'tf32', 'fp32', 'fp64'] # Precision fallback order

def chop(x, precision_idx=0):
    """Recursive chop function"""
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float64, device=device)
    if precision_idx >= len(precision_fallback):
        return x
    precision = precision_fallback[precision_idx]
    if precision == 'fp64':
        return x
    ch = LightChop(**precision_configs[precision])
    result = ch(x)
    if not torch.any(torch.isnan(result)) and not torch.any(torch.isinf(result)):
        return result.to(torch.float64).to(device)
    logging.debug(f"Chop: Precision {precision} failed, escalating to {precision_fallback[precision_idx + 1]}")
    return chop(x, precision_idx + 1)

def rounding(x, precision):
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


def round_sparse_matrix(A, precision):
    """Round sparse matrix to specified precision"""
    if precision == 'fp64':
        return A
    A_coo = A.tocoo()
    data = torch.tensor(A_coo.data, dtype=torch.float64, device=device)
    ch = LightChop(**precision_configs[precision])
    rounded_data = ch(data)
    if torch.any(torch.isnan(rounded_data)) or torch.any(torch.isinf(rounded_data)):
        logging.warning(f"Rounding sparse matrix to {precision} failed; using fp64")
        return A
    return csc_matrix((rounded_data.cpu().numpy(), (A_coo.row, A_coo.col)), shape=A.shape)



if __name__ == "__main__":
  import numpy as np
  
  A = np.random.randn(100, 100)
  B = np.random.randn(100, 100)
  C = A + B
  print("C:", C)
  print("C (fp32):",mixed_precision_op(lambda x, y: x+y, A, 'fp32', B))
