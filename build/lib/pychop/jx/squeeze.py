import jax
import jax.numpy as jnp

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
    A (jnp.ndarray): Input matrix
    tol (float): Tolerance for zero division (default: 1e-10)
    
    Returns
    -------
    tuple: (R, S) diagonal scaling vectors
    """
    m, n = A.shape
    
    # Compute row sums and scale
    R = jnp.sum(jnp.abs(A), axis=1)
    R = jnp.where(R < tol, 1.0, 1 / R)
    
    # Apply row scaling using broadcasting
    A_scaled = A * R[:, None]
    
    # Compute column sums and scale
    S = jnp.sum(jnp.abs(A_scaled), axis=0)
    S = jnp.where(S < tol, 1.0, 1 / S)
    
    return R, S

def two_sided_diagonal_scaling_sym(A, tol=1e-6, max_iter=100):
    """
    Symmetry-preserving two-sided diagonal scaling for a symmetric matrix A.
    
    Parameters
    ----------
    A (jnp.ndarray): Input symmetric matrix
    tol (float): Convergence tolerance (default: 1e-6)
    max_iter (int): Maximum number of iterations (default: 100)
    
    Returns
    -------
    tuple: (R, S) diagonal scaling vectors
    """
    m, n = A.shape
    R = jnp.ones(m)
    S = jnp.ones(n)
    
    def body(state, _):
        A, R, S = state
        # Compute row and column sums efficiently
        row_sums = jnp.sqrt(jnp.sum(jnp.abs(A), axis=1))
        col_sums = jnp.sqrt(jnp.sum(jnp.abs(A), axis=0))
        
        # Avoid division by zero
        row_sums = jnp.where(row_sums == 0, 1.0, row_sums)
        col_sums = jnp.where(col_sums == 0, 1.0, col_sums)
        
        # Compute scaling factors
        R_new = 1 / row_sums
        S_new = 1 / col_sums
        
        # Update matrix using vectorized operations
        A_new = (A * R_new[:, None]) * S_new[None, :]
        
        # Update cumulative scaling vectors
        R_updated = R * R_new
        S_updated = S * S_new
        
        return (A_new, R_updated, S_updated), (R_new, S_new)
    
    def cond(state, _):
        A, R, S = state
        R_new, S_new = _
        return jnp.any(jnp.abs(R_new - 1) > tol) | jnp.any(jnp.abs(S_new - 1) > tol)
    
    # Use jax.lax.scan for iterative updates
    (A_final, R_final, S_final), _ = jax.lax.scan(
        lambda state, x: (body(state, x)[0], body(state, x)[1]),
        (A, R, S),
        None,
        length=max_iter
    )
    
    return R_final, S_final

def squeeze_fp16(A, theta=1.0):
    """
    Squeeze a matrix to fp16 format with diagonal scaling.
    
    Parameters
    ----------
    A (jnp.ndarray): Input matrix
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
    R_diag = jnp.diag(params["R"])
    S_diag = jnp.diag(params["S"])
    A_tilde = R_diag @ A @ S_diag
    alpha = jnp.max(jnp.max(A_tilde))
    params["mu"] = theta * xmax / alpha
    rounded_A = (params["mu"] * A_tilde).astype(jnp.float16)
    
    return rounded_A, params

def squeeze_sym_fp16(A, tol=0.1, theta=1.0):
    """
    Symmetry-preserving squeeze to fp16 format for a symmetric matrix.
    
    Parameters
    ----------
    A (jnp.ndarray): Input symmetric matrix
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
    R_diag = jnp.diag(params["R"])
    S_diag = jnp.diag(params["S"])
    A_tilde = R_diag @ A @ S_diag
    alpha = jnp.max(jnp.max(A_tilde))
    params["mu"] = theta * xmax / alpha
    rounded_A = (params["mu"] * A_tilde).astype(jnp.float16)
    
    return rounded_A, params

def desqueeze(rounded_A, params):
    """
    Desqueeze a matrix from fp16 format using scaling parameters.
    
    Parameters
    ----------
    rounded_A (jnp.ndarray): Scaled matrix in fp16
    params (dict): Dictionary containing scaling factors R, S, and mu
    
    Returns
    -------
    jnp.ndarray: Descaled matrix
    """
    # Convert to float32 for numerical stability
    rounded_A = rounded_A.astype(jnp.float32)
    R_inv = jnp.diag(1 / params["R"].astype(jnp.float32))
    S_inv = jnp.diag(1 / params["S"].astype(jnp.float32))
    mu = params["mu"].astype(jnp.float32)
    return R_inv @ (rounded_A / mu) @ S_inv

if __name__ == "__main__":
    # Create a sample matrix
    A = jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32)
    print("Original matrix A:")
    print(A)

    # Apply squeeze_fp16
    A_rounded, params = squeeze_fp16(A)
    print("\nSqueezed matrix (fp16):")
    print(A_rounded)

    # Reconstruct the matrix using desqueeze
    A_recon = desqueeze(A_rounded, params)
    print("\nReconstructed matrix:")
    print(A_recon)
