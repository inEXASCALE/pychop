import numpy as np

def calc_float_max(exp_bits, sig_bits):
    """
    Calculate the maximum representable value for a floating-point format.
    
    Parameters
    ----------
    exp_bits (int): 
        Number of exponent bits
    sig_bits (int): Number of fraction (mantissa) bits

    
    Returns
    ----------
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
        A (np.ndarray)
            Input matrix
            
        tol (float)
            Tolerance for zero division (default: 1e-10)
    
    Returns
    ----------
        tuple: (R, S) diagonal scaling vectors
    """
    m, n = A.shape
    
    # Compute row sums and scale
    R = np.sum(np.abs(A), axis=1)
    R = np.where(R < tol, 1, 1 / R)
    
    # Apply row scaling using broadcasting
    A_scaled = A * R[:, np.newaxis]
    
    # Compute column sums and scale
    S = np.sum(np.abs(A_scaled), axis=0)
    S = np.where(S < tol, 1, 1 / S)
    
    return R, S




def two_sided_diagonal_scaling_sym(A, tol=1e-6, max_iter=100):
    """
    Symmetry-preserving two-sided diagonal scaling for a symmetric matrix A.
    
    Parameters
    ----------
        A (np.ndarray)
            Input symmetric matrix
        
        tol (float)
            Convergence tolerance (default: 1e-6)
            
        max_iter (int)
            Maximum number of iterations (default: 100)
    
    
    Returns
    ----------
        tuple: (R, S) diagonal scaling vectors
    """
    m, n = A.shape

    # Initialize scaling vectors
    R = np.ones(m)
    S = np.ones(n)
    
    # Pre-allocate diagonal matrices to avoid repeated creation
    for _ in range(max_iter):
        # Compute row and column sums efficiently
        row_sums = np.sqrt(np.sum(np.abs(A), axis=1))
        col_sums = np.sqrt(np.sum(np.abs(A), axis=0))
        
        # Avoid division by zero
        row_sums = np.where(row_sums == 0, 1, row_sums)
        col_sums = np.where(col_sums == 0, 1, col_sums)
        
        # Compute scaling factors
        R_new = 1 / row_sums
        S_new = 1 / col_sums
        
        # Update matrix using vectorized operations
        A = (A * R_new[:, np.newaxis]) * S_new[np.newaxis, :]
        
        # Update cumulative scaling vectors
        R *= R_new
        S *= S_new
        
        # Check convergence
        if np.all(np.abs(R_new - 1) <= tol) and np.all(np.abs(S_new - 1) <= tol):
            break
    
    return R, S



def squeeze_fp16(A, theta=0.8):
    params = {}
    
    exp_bits=5
    sig_bits=10
    
    params["R"], params["S"] = two_sided_diagonal_scaling(A)
    xmax = calc_float_max(exp_bits, sig_bits)
    
    A_tilde = np.diag(params["R"]) @ A @ np.diag(params["S"])
    alpha = np.max(np.abs(A_tilde))
    params["mu"] = theta * xmax / alpha
    rounded_A = (params["mu"] * A_tilde).astype(np.float16)
    return rounded_A, params



def squeeze_sym_fp16(A, tol=0.1, theta=0.8):
    params = {}

    exp_bits=5
    sig_bits=10
    params["R"], params["S"] = two_sided_diagonal_scaling_sym(A,tol=tol)
    xmax = calc_float_max(exp_bits, sig_bits)
    
    A_tilde = np.diag(params["R"]) @ A @ np.diag(params["S"])
    alpha = np.max(np.abs(A_tilde))
    params["mu"] = theta * xmax / alpha
    rounded_A = (params["mu"] * A_tilde).astype(np.float16)
    return rounded_A, params


def desqueeze(rounded_A, params):
    return np.diag(1/params["R"]) @(rounded_A / params["mu"]) @ np.diag(1/params["S"])


if __name__ == "__main__":

    A = np.random.randn(3, 3)
    print("A:", A)
    A_rounded, params = squeeze_fp16(A)
    A_recon = desqueeze(A_rounded, params)
    print("A_recon:", A_recon)

    A_rounded, params = squeeze_sym_fp16(A)
    A_recon = desqueeze(A_rounded, params)
    print("A_recon (sym):", A_recon)

    from scipy.sparse.linalg import gmres
    A = np.random.randn(3, 3)
    b = np.random.randn(3)
    x1, exitCode = gmres(A, b, rtol=1e-5)
    print("x1:", x1)
    
    A_rounded, params = squeeze_fp16(A)
    
    x2, exitCode = gmres(A_rounded,  params["mu"]* np.diag(params["R"]) @ b, rtol=1e-5)
    x2 = params["S"] * x2
    print("x2 (1):", x2)

    A_rounded, params = squeeze_fp16(A)
    
    x2, exitCode = gmres(A_rounded,  np.diag(params["R"]) @ b, rtol=1e-5)
    x2 = params["mu"]* params["S"] * x2
    print("x2 (2):", x2)
