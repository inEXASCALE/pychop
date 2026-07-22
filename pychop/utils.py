"""Array backend detection and conversion helpers used by pychop front ends."""

from __future__ import annotations
__all__ = ['detect_array_type', 'to_numpy_array', 'to_torch_tensor', 'to_jax_array', 'to_tensorflow_tensor']
import warnings
import numpy as np
import importlib.util
import sys


# Optional dependencies
has_pandas = importlib.util.find_spec("pandas") is not None
has_torch = importlib.util.find_spec("torch") is not None
has_jax = importlib.util.find_spec("jax") is not None
has_tensorflow = importlib.util.find_spec("tensorflow") is not None

pd = None
torch = None
jax = None
jnp = None
tf = None


def _type_module(arr) -> str:
    """Return the module path for an object's concrete type."""
    return type(arr).__module__


def _is_pandas_object(arr) -> bool:
    """Check for pandas objects without importing pandas during pychop import."""
    module = _type_module(arr)
    if not module.startswith("pandas."):
        return False

    pandas_module = sys.modules.get("pandas")
    if pandas_module is None:
        return True

    return isinstance(arr, (pandas_module.DataFrame, pandas_module.Series))


def _import_pandas():
    """Import pandas only when a conversion actually needs it."""
    global pd, has_pandas
    if pd is None:
        if not has_pandas:
            raise ImportError("Pandas is not available.")
        import pandas as _pd
        pd = _pd
    return pd


def _import_torch():
    """Import PyTorch only when a conversion actually needs it."""
    global torch, has_torch
    if torch is None:
        if not has_torch:
            raise ImportError("PyTorch is not available.")
        import torch as _torch
        torch = _torch
    return torch


def _import_jax():
    """Import JAX only when a conversion actually needs it."""
    global jax, jnp, has_jax
    if jax is None or jnp is None:
        if not has_jax:
            raise ImportError("JAX is not available.")
        import jax as _jax
        import jax.numpy as _jnp
        jax = _jax
        jnp = _jnp
    return jax, jnp


def _import_tensorflow():
    """Import TensorFlow only when a conversion actually needs it."""
    global tf, has_tensorflow
    if tf is None:
        if not has_tensorflow:
            raise ImportError("TensorFlow is not available.")
        import tensorflow as _tf
        tf = _tf
    return tf


def detect_array_type(arr, verbose=False) -> str:
    """
    Detect the backend type of an array-like object from the provided arguments.

    This function inspects all positional and keyword arguments in order,
    and returns the backend type of the first detected array-like object.
    It is designed for dispatch scenarios where the function is called as
    detect_array_type(*args, **kwargs), but typically only one primary
    array-like input is present.

    Parameters
    ----------
    X : object
        Array-like object.

    verbose : bool, optional
        If True, prints the type of each argument during detection (default: False).

    Returns
    -------
    str
        One of:
        - 'numpy'   : NumPy ndarray or Pandas DataFrame/Series
        - 'torch'   : PyTorch Tensor
        - 'jax'     : JAX Array
        - 'tensorflow' : TensorFlow Tensor
        - 'list'    : Python list or tuple
        - 'unknown' : No array-like object found or unrecognized type

    Examples
    --------
    >>> detect_array_type(np.zeros(3))
    'numpy'
    >>> detect_array_type([[1, 2], [3, 4]])
    'list'
    >>> detect_array_type(torch.tensor([1.0]))
    'torch'
    >>> detect_array_type(np.zeros(3), x=1.5, flag=True)
    'numpy'  # ignores non-array arguments
    >>> detect_array_type(1, 2, 3)
    'unknown'
    """
    
    if isinstance(arr, (list, tuple)):
        if verbose:
            print("Detected type: list/tuple")
        return 'list'
    
    if has_pandas and _is_pandas_object(arr):
        if verbose:
            print("Detected type: pandas DataFrame/Series")
        return 'numpy'
    
    if isinstance(arr, np.ndarray):
        if verbose:
            print("Detected type: numpy ndarray")
        return 'numpy'
    
    module = _type_module(arr)

    if has_torch and module.startswith("torch"):
        if verbose:
            print("Detected type: torch Tensor")
        return 'torch'
    
    if has_jax and "jax" in module:
        if verbose:
            print("Detected type: jax Array")
        return 'jax'

    if has_tensorflow and "tensorflow" in module:
        if verbose:
            print("Detected type: tensorflow Tensor")
        return 'tensorflow'
    
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
    if has_pandas and _is_pandas_object(arr):
        pandas_module = _import_pandas()
        if not isinstance(arr, (pandas_module.DataFrame, pandas_module.Series)):
            raise TypeError(f"Unsupported pandas object for safe numpy conversion: {type(arr)}")
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
        if hasattr(arr, "detach"):
            arr = arr.detach()
        return arr.cpu().numpy()
    
    if arr_type == 'jax':
        if not has_jax:
            raise ImportError("JAX is required for this conversion.")
        warnings.warn("Converting JAX array to NumPy may involve device-to-host copy.", UserWarning)
        return np.asarray(arr)

    if arr_type == 'tensorflow':
        if not has_tensorflow:
            raise ImportError("TensorFlow is required for this conversion.")
        warnings.warn("Converting TensorFlow tensor to NumPy may involve device-to-host copy.", UserWarning)
        return arr.numpy()
    
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
    torch_module = _import_torch()
    
    arr_type = detect_array_type(arr)
    
    if arr_type == 'torch':
        return arr
    
    if arr_type == 'numpy':
        np_arr = to_numpy_safe(arr)
        return torch_module.from_numpy(np_arr)
    
    if arr_type == 'list':
        np_arr = _try_convert_list_to_numpy(arr)
        return torch_module.from_numpy(np_arr)
    
    if arr_type == 'jax':
        warnings.warn("Converting JAX to PyTorch involves copy via NumPy intermediate.", UserWarning)
        np_arr = np.asarray(arr)
        return torch_module.from_numpy(np_arr)

    if arr_type == 'tensorflow':
        warnings.warn("Converting TensorFlow to PyTorch involves copy via NumPy intermediate.", UserWarning)
        return torch_module.from_numpy(arr.numpy())
    
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
    _, jnp_module = _import_jax()
    
    arr_type = detect_array_type(arr)
    
    if arr_type == 'jax':
        return arr
    
    if arr_type in ('numpy', 'list'):
        if arr_type == 'list':
            np_arr = _try_convert_list_to_numpy(arr)
        else:
            np_arr = to_numpy_safe(arr)
        warnings.warn("Converting to JAX array involves data copy to default JAX device.", UserWarning)
        return jnp_module.array(np_arr)
    
    if arr_type == 'torch':
        warnings.warn("Converting PyTorch to JAX involves copy via NumPy intermediate.", UserWarning)
        if arr.device.type != 'cpu':
            warnings.warn(f"Additional copy from {arr.device} to CPU.", UserWarning)
        if hasattr(arr, "detach"):
            arr = arr.detach()
        np_arr = arr.cpu().numpy()
        return jnp_module.array(np_arr)

    if arr_type == 'tensorflow':
        warnings.warn("Converting TensorFlow to JAX involves copy via NumPy intermediate.", UserWarning)
        return jnp_module.array(arr.numpy())
    
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

def to_tensorflow_tensor(arr):
    """
    Convert array-like object to TensorFlow tensor.
    """
    if not has_tensorflow:
        raise ImportError("TensorFlow is not available.")
    tf_module = _import_tensorflow()

    arr_type = detect_array_type(arr)

    if arr_type == 'tensorflow':
        return arr

    if arr_type == 'numpy':
        return tf_module.convert_to_tensor(to_numpy_safe(arr))

    if arr_type == 'list':
        return tf_module.convert_to_tensor(_try_convert_list_to_numpy(arr))

    if arr_type == 'torch':
        warnings.warn("Converting PyTorch to TensorFlow involves copy via NumPy intermediate.", UserWarning)
        if hasattr(arr, "detach"):
            arr = arr.detach()
        return tf_module.convert_to_tensor(arr.cpu().numpy())

    if arr_type == 'jax':
        warnings.warn("Converting JAX to TensorFlow involves copy via NumPy intermediate.", UserWarning)
        return tf_module.convert_to_tensor(np.asarray(arr))

    raise TypeError(f"Cannot convert type '{arr_type}' to TensorFlow tensor.")
