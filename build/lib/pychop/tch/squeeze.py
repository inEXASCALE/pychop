import torch

def calc_float_max(exp_bits, sig_bits):
    """
    Calculate the maximum representable value for a floating-point format.
    
    Parameters
    ----------
    exp_bits (int): Number of exponent bits
    sig_bits (int): Number of fraction (mantissa) bits
    
    Returns
    -------
    float: Maximum representable value
    """
    bias = 2**(exp_bits - 1) - 1
    max_exponent = (2**exp_bits - 2) - bias
    max_mantissa = 2 - 2**(-sig_bits)
    max_value = max_mantissa * (2**max_exponent)
    return max_value

def two_sided_diagonal_scaling(A, tol=1e-10):
    """
    Two-sided diagonal scaling for a matrix A.
    
    Parameters
    ----------
    A (torch.Tensor): Input matrix
    tol (float): Tolerance for zero division (default: 1e-10)
    
    Returns
    -------
    tuple: (R, S) diagonal scaling vectors
    """
    m, n = A.shape
    
    # Compute row sums and scale
    R = torch.sum(torch.abs(A), dim=1)
    R = torch.where(R < tol, torch.tensor(1.0, device=A.device, dtype=A.dtype), 1 / R)
    
    # Apply row scaling using broadcasting
    A_scaled = A * R[:, None]
    
    # Compute column sums and scale
    S = torch.sum(torch.abs(A_scaled), dim=0)
    S = torch.where(S < tol, torch.tensor(1.0, device=A.device, dtype=A.dtype), 1 / S)
    
    return R, S

def two_sided_diagonal_scaling_sym(A, tol=1e-6, max_iter=100):
    """
    Symmetry-preserving two-sided diagonal scaling for a symmetric matrix A.
    
    Parameters
    ----------
    A (torch.Tensor): Input symmetric matrix
    tol (float): Convergence tolerance (default: 1e-6)
    max_iter (int): Maximum number of iterations (default: 100)
    
    Returns
    -------
    tuple: (R, S) diagonal scaling vectors
    """
    m, n = A.shape
    R = torch.ones(m, device=A.device, dtype=A.dtype)
    S = torch.ones(n, device=A.device, dtype=A.dtype)
    
    for _ in range(max_iter):
        # Compute row and column sums efficiently
        row_sums = torch.sqrt(torch.sum(torch.abs(A), dim=1))
        col_sums = torch.sqrt(torch.sum(torch.abs(A), dim=0))
        
        # Avoid division by zero
        row_sums = torch.where(row_sums == 0, torch.tensor(1.0, device=A.device, dtype=A.dtype), row_sums)
        col_sums = torch.where(col_sums == 0, torch.tensor(1.0, device=A.device, dtype=A.dtype), col_sums)
        
        # Compute scaling factors
        R_new = 1 / row_sums
        S_new = 1 / col_sums
        
        # Update matrix using vectorized operations
        A = (A * R_new[:, None]) * S_new[None, :]
        
        # Update cumulative scaling vectors
        R *= R_new
        S *= S_new
        
        # Check convergence
        if torch.all(torch.abs(R_new - 1) <= tol) and torch.all(torch.abs(S_new - 1) <= tol):
            break
    
    return R, S

def squeeze_fp16(A, theta=1.0):
    """
    Squeeze a matrix to fp16 format with diagonal scaling.
    
    Parameters
    ----------
    A (torch.Tensor): Input matrix
    theta (float): Scaling factor (default: 1.0)
    
    Returns
    -------
    tuple: (rounded_A, params) where rounded_A is the scaled matrix in fp16, and params contains scaling factors
    """
    params = {}
    exp_bits = 5
    sig_bits = 10
    
    params["R"], params["S"] = two_sided_diagonal_scaling(A)
    xmax = calc_float_max(exp_bits, sig_bits)
    
    # Create diagonal matrices and perform scaling
    R_diag = torch.diag(params["R"])
    S_diag = torch.diag(params["S"])
    A_tilde = R_diag @ A @ S_diag
    alpha = torch.max(torch.abs(A_tilde))
    params["mu"] = theta * xmax / alpha
    rounded_A = (params["mu"] * A_tilde).to(torch.float16)
    
    return rounded_A, params

def squeeze_sym_fp16(A, tol=0.1, theta=1.0):
    """
    Symmetry-preserving squeeze to fp16 format for a symmetric matrix.
    
    Parameters
    ----------
    A (torch.Tensor): Input symmetric matrix
    tol (float): Convergence tolerance (default: 0.1)
    theta (float): Scaling factor (default: 1.0)
    
    Returns
    -------
    tuple: (rounded_A, params) where rounded_A is the scaled matrix in fp16, and params contains scaling factors
    """
    params = {}
    exp_bits = 5
    sig_bits = 10
    
    params["R"], params["S"] = two_sided_diagonal_scaling_sym(A, tol=tol)
    xmax = calc_float_max(exp_bits, sig_bits)
    
    # Create diagonal matrices and perform scaling
    R_diag = torch.diag(params["R"])
    S_diag = torch.diag(params["S"])
    A_tilde = R_diag @ A @ S_diag
    alpha = torch.max(torch.abs(A_tilde))
    params["mu"] = theta * xmax / alpha
    rounded_A = (params["mu"] * A_tilde).to(torch.float16)
    
    return rounded_A, params

def desqueeze(rounded_A, params):
    """
    Desqueeze a matrix from fp16 format using scaling parameters.
    
    Parameters
    ----------
    rounded_A (torch.Tensor): Scaled matrix in fp16
    params (dict): Dictionary containing scaling factors R, S, and mu
    
    Returns
    -------
    torch.Tensor: Descaled matrix
    """
    # Convert rounded_A to float32 for numerical stability
    rounded_A = rounded_A.to(torch.float32)
    R_inv = torch.diag(1 / params["R"].to(torch.float32))
    S_inv = torch.diag(1 / params["S"].to(torch.float32))
    mu = params["mu"].to(torch.float32)
    return R_inv @ (rounded_A / mu) @ S_inv
