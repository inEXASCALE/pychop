Examples
=====================================================

To import the package:

.. code:: python

    from pychop import chop

or 

.. code:: python

    from torch_chop import chop
    


+----------------------------------+----------------------------------+
|           format | description          |
+==================================+==================================+
| 'q43', 'fp8-e4m3'         | NVIDIA quarter precision (4 exponent bits, 3 significand (mantissa) bits) |
+----------------------------------+----------------------------------+
| 'q52', 'fp8-e5m2'         | NVIDIA quarter precision (5 exponent bits, 2 significand bits) |
+----------------------------------+----------------------------------+
|  'b', 'bfloat16'          | bfloat16 |
+----------------------------------+----------------------------------+
|  'h', 'half', 'fp16'      | IEEE half precision (the default) |
+----------------------------------+----------------------------------+
|  's', 'single', 'fp32'    | IEEE single precision |
+----------------------------------+----------------------------------+
|  'd', 'double', 'fp64'    | IEEE double precision |
+----------------------------------+----------------------------------+
|  'c', 'custom'            | custom format |
+----------------------------------+----------------------------------+

.. code:: python
