__all__ = ['detect_array_type', 'to_numpy_array', 'to_torch_tensor', 'to_jax_array']
import warnings
import numpy as np
import importlib.util


# Optional dependencies
has_pandas = False
try:
    import pandas as pd
    has_pandas = True
except ImportError:
    pass

has_torch = False
TorchTensor = None
if importlib.util.find_spec("torch") is not None:
    try:
        import torch
        TorchTensor = torch.Tensor
        has_torch = True
    except ImportError:
        pass

has_jax = False
JaxArray = None
jnp = None

if importlib.util.find_spec("jax") is not None:
    try:
        import jax
        from jax import Array as JaxArray
        import jax.numpy as jnp
        has_jax = True
        has_jnp = True
    except ImportError:
        pass


def detect_array_type(arr) -> str:
    """
    Detect the backend type of an array-like object.

    Parameters
    ----------
    arr : object
        Input array-like object to inspect.

    Returns
    -------
    str
        One of the following strings:
        - 'numpy'   : NumPy ndarray or Pandas DataFrame/Series
        - 'torch'   : PyTorch Tensor
        - 'jax'     : JAX Array
        - 'list'    : Python list or tuple (potentially convertible to array)
        - 'unknown' : Any other type

    Examples
    --------
    >>> detect_array_type(np.zeros(3))
    'numpy'
    >>> detect_array_type([[1, 2], [3, 4]])
    'list'
    >>> detect_array_type(torch.tensor([1.0]))
    'torch'
    """
    if isinstance(arr, (list, tuple)):
        return 'list'
    
    if has_pandas and isinstance(arr, (pd.DataFrame, pd.Series)):
        return 'numpy'
    
    if isinstance(arr, np.ndarray):
        return 'numpy'
    
    if has_torch and isinstance(arr, TorchTensor):
        return 'torch'
    
    if has_jax and isinstance(arr, JaxArray):
        return 'jax'
    
    return 'unknown'


def _try_convert_list_to_numpy(arr) -> np.ndarray:
    """
    Internal helper: attempt to convert a regular Python list/tuple to NumPy ndarray.

    Parameters
    ----------
    arr : list or tuple
        Input list or tuple.

    Returns
    -------
    np.ndarray
        Converted NumPy array.

    Raises
    ------
    ValueError
        If the list/tuple is irregular or contains incompatible types.
    """
    try:
        np_arr = np.array(arr)
        warnings.warn(
            "Input is a Python list/tuple; conversion to array involves data copy.",
            UserWarning
        )
        return np_arr
    except Exception as e:
        raise ValueError(f"Irregular or incompatible list/tuple cannot be converted: {e}")


def to_numpy_safe(arr) -> np.ndarray:
    """
    Internal helper: safely convert NumPy/Pandas to ndarray (zero-copy when possible).

    Parameters
    ----------
    arr : np.ndarray or pandas object
        Input object (must be NumPy ndarray or Pandas DataFrame/Series).

    Returns
    -------
    np.ndarray
        NumPy array view/copy of the input.
    """
    if has_pandas and isinstance(arr, (pd.DataFrame, pd.Series)):
        return arr.to_numpy(copy=False)
    
    if isinstance(arr, np.ndarray):
        return arr
    
    raise TypeError(f"Unsupported type for safe numpy conversion: {type(arr)}")


def to_numpy_array(arr) -> np.ndarray:
    """
    Convert array-like object to NumPy ndarray, with warnings for unsafe conversions.

    Parameters
    ----------
    arr : object
        Input array-like object.

    Returns
    -------
    np.ndarray
        Converted NumPy array.

    Raises
    ------
    ImportError
        If required frameworks (torch/jax) are not available.
    TypeError
        If the input type cannot be converted.
    ValueError
        If a list/tuple is irregular.

    Notes
    -----
    - Zero-copy/view for NumPy and Pandas.
    - Copies from device for PyTorch/JAX if needed.
    - List/tuple conversion emits a warning about copying.

    Examples
    --------
    >>> to_numpy_array([[1, 2], [3, 4]])
    array([[1, 2],
           [3, 4]])
    """
    arr_type = detect_array_type(arr)
    
    if arr_type == 'numpy':
        return to_numpy_safe(arr)
    
    if arr_type == 'list':
        return _try_convert_list_to_numpy(arr)
    
    if arr_type == 'torch':
        if not has_torch:
            raise ImportError("PyTorch is required for this conversion.")
        if arr.device.type != 'cpu':
            warnings.warn(f"Copying tensor from {arr.device} to CPU.", UserWarning)
        return arr.cpu().numpy()
    
    if arr_type == 'jax':
        if not has_jax:
            raise ImportError("JAX is required for this conversion.")
        warnings.warn("Converting JAX array to NumPy may involve device-to-host copy.", UserWarning)
        return np.asarray(arr)
    
    raise TypeError(f"Cannot convert type '{arr_type}' to NumPy ndarray.")


def to_torch_tensor(arr) -> torch.Tensor:
    """
    Convert array-like object to PyTorch Tensor, with warnings for unsafe conversions.

    Parameters
    ----------
    arr : object
        Input array-like object.

    Returns
    -------
    torch.Tensor
        Converted PyTorch tensor (preserves device if already torch).

    Raises
    ------
    ImportError
        If PyTorch is not available.
    TypeError
        If the input type cannot be converted.
    ValueError
        If a list/tuple is irregular.

    Notes
    -----
    - Zero-copy from CPU NumPy/Pandas via shared memory.
    - Copies via NumPy intermediate for JAX or list/tuple.

    Examples
    --------
    >>> to_torch_tensor([[1.0, 2.0], [3.0, 4.0]])
    tensor([[1., 2.],
            [3., 4.]])
    """
    if not has_torch:
        raise ImportError("PyTorch is not available.")
    
    arr_type = detect_array_type(arr)
    
    if arr_type == 'torch':
        return arr
    
    if arr_type == 'numpy':
        np_arr = to_numpy_safe(arr)
        return torch.from_numpy(np_arr)
    
    if arr_type == 'list':
        np_arr = _try_convert_list_to_numpy(arr)
        return torch.from_numpy(np_arr)
    
    if arr_type == 'jax':
        warnings.warn("Converting JAX to PyTorch involves copy via NumPy intermediate.", UserWarning)
        np_arr = np.asarray(arr)
        return torch.from_numpy(np_arr)
    
    raise TypeError(f"Cannot convert type '{arr_type}' to PyTorch tensor.")


def to_jax_array(arr) -> jax.Array:
    """
    Convert array-like object to JAX Array, with warnings for unsafe conversions.

    Parameters
    ----------
    arr : object
        Input array-like object.

    Returns
    -------
    jax.Array
        Converted JAX array (preserves device if already JAX).

    Raises
    ------
    ImportError
        If JAX is not available.
    TypeError
        If the input type cannot be converted.
    ValueError
        If a list/tuple is irregular.

    Notes
    -----
    - Copies to default JAX device for non-JAX inputs.
    - List/tuple and cross-framework conversions involve copying.

    Examples
    --------
    >>> to_jax_array([[1, 2], [3, 4]])
    Array([[1, 2],
           [3, 4]], dtype=int32)
    """
    if not has_jax:
        raise ImportError("JAX is not available.")
    
    arr_type = detect_array_type(arr)
    
    if arr_type == 'jax':
        return arr
    
    if arr_type in ('numpy', 'list'):
        if arr_type == 'list':
            np_arr = _try_convert_list_to_numpy(arr)
        else:
            np_arr = to_numpy_safe(arr)
        warnings.warn("Converting to JAX array involves data copy to default JAX device.", UserWarning)
        return jnp.array(np_arr)
    
    if arr_type == 'torch':
        warnings.warn("Converting PyTorch to JAX involves copy via NumPy intermediate.", UserWarning)
        if arr.device.type != 'cpu':
            warnings.warn(f"Additional copy from {arr.device} to CPU.", UserWarning)
        np_arr = arr.cpu().numpy()
        return jnp.array(np_arr)
    
    raise TypeError(f"Cannot convert type '{arr_type}' to JAX array.")


if __name__ == "__main__":
    warnings.filterwarnings("always", category=UserWarning)

    print("=== Array Type Detection & Conversion Tests ===\n")

    # 1. NumPy array
    print("1. Testing NumPy ndarray")
    np_arr = np.array([[1, 2], [3, 4]])
    print(f"detect_array_type: {detect_array_type(np_arr)}")
    print(f"to_numpy_array: shape {to_numpy_array(np_arr).shape}, dtype {to_numpy_array(np_arr).dtype}")
    if has_torch:
        print(f"to_torch_tensor: {to_torch_tensor(np_arr)}")
    if has_jax:
        print(f"to_jax_array: {to_jax_array(np_arr)}")
    print()

    # 2. Pandas DataFrame
    if has_pandas:
        print("2. Testing Pandas DataFrame")
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        print(f"detect_array_type: {detect_array_type(df)}")
        print(f"to_numpy_array: shape {to_numpy_array(df).shape}, dtype {to_numpy_array(df).dtype}")
        if has_torch:
            print(f"to_torch_tensor: {to_torch_tensor(df)}")
        if has_jax:
            print(f"to_jax_array: {to_jax_array(df)}")
        print()

    # 3. PyTorch Tensor (CPU)
    if has_torch:
        print("3. Testing PyTorch Tensor (CPU)")
        torch_cpu = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        print(f"detect_array_type: {detect_array_type(torch_cpu)}")
        print(f"to_numpy_array: {to_numpy_array(torch_cpu)}")
        print(f"to_torch_tensor: {to_torch_tensor(torch_cpu)} (device: {to_torch_tensor(torch_cpu).device})")
        if has_jax:
            print(f"to_jax_array: {to_jax_array(torch_cpu)}")
        print()

        # 4. PyTorch Tensor (GPU, if available)
        if torch.cuda.is_available():
            print("4. Testing PyTorch Tensor (GPU)")
            torch_gpu = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device="cuda")
            print(f"Original device: {torch_gpu.device}")
            print(f"detect_array_type: {detect_array_type(torch_gpu)}")
            print(f"to_numpy_array: {to_numpy_array(torch_gpu)}  # should warn about copy")
            print(f"to_torch_tensor: device remains {to_torch_tensor(torch_gpu).device}")
            if has_jax:
                print(f"to_jax_array: {to_jax_array(torch_gpu)}  # should warn about extra copy")
            print()

    # 5. JAX Array
    if has_jax:
        print("5. Testing JAX Array")
        jax_arr = jnp.array([[9.0, 10.0], [11.0, 12.0]])
        print(f"detect_array_type: {detect_array_type(jax_arr)}")
        print(f"to_numpy_array: {to_numpy_array(jax_arr)}  # may warn about copy")
        if has_torch:
            print(f"to_torch_tensor: {to_torch_tensor(jax_arr)}  # should warn about copy")
        print(f"to_jax_array: {to_jax_array(jax_arr)} (device: {jax_arr.devices()})")
        print()

    # 6. Unknown type test
    print("6. Testing unknown type (list)")
    list_arr = [[1, 2], [3, 4]]
    try:
        print(f"detect_array_type: {detect_array_type(list_arr)}")
        print(f"to_numpy_array: {to_numpy_array(list_arr)}")
        print(f"to_torch_tensor: {to_torch_tensor(list_arr)} (device: {to_torch_tensor(list_arr).device})")
    except TypeError as e:
        print(f"to_numpy_array correctly raised: {e}")
    print()

    print("=== All tests completed ===")
    if not (has_torch or has_jax):
        print("Note: torch and jax not installed, some tests skipped.")