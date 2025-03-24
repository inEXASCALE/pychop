Support in Matlab
=====================================================

MATLAB provides built-in support for calling Python libraries through its Python Interface. This allows users to use Python functions, classes, and modules directly from MATLAB, making it easy to integrate Python-based scientific computing, machine learning, and deep learning libraries into MATLAB workflows. MATLAB interacts with Python by adding the \texttt{py.} prefix, which allows MATLAB to call the needed Python library seamlessly. One can also execute Python statements in the Python interpreter directly from MATLAB using the \texttt{pyrun} or \texttt{pyrunfile} functions. For detail, we refer the users to \url{https://www.mathworks.com/help/matlab/call-python-libraries.html}

If you use Python virtual environments, ensure MATLAB detects it:

.. code:: matlab

    >> pe = pyenv('Version', 'C:\Users\YourUser\Anaconda3\envs\your_env\python.exe');

This allows MATLAB to use Python packages installed in the virtual environment.

.. code:: matlab

    >> pe = 
    PythonEnvironment with properties:

            Version: "3.10"
        Executable: "/software/python/anaconda3/bin/python3"
            Library: "/software/python/anaconda3/lib/libpython3.10.so"
                Home: "/software/python/anaconda3"
            Status: NotLoaded
        ExecutionMode: InProcess



You can call Python’s built-in functions using ``py.<module>.<function>``.

.. code:: matlab

    >> result = py.math.sqrt(16);
    >> disp(result)

    >> version = py.sys.version;
    >> disp(version)



MATLAB automatically converts data types when calling Python functions. However, sometimes explicit conversion is required.

+----------------+-------------+
| MATLAB Type    | Python Type |
+================+=============+
| double         | float       |
+----------------+-------------+
| char (string)  | str         |
+----------------+-------------+
| cell array     | list        |
+----------------+-------------+
| struct         | dict        |
+----------------+-------------+
| logical        | bool        |
+----------------+-------------+


To use an external library like NumPy or SciPy, ensure it's installed in your Python environment.

Calling NumPy’s array and Computing the Sum

.. code:: matlab

    >> numpy = py.importlib.import_module('numpy'); % Import the numpy module
    >> a = numpy.array([1, 2, 3, 4, 5]);  % Create a numpy array
    >> s = numpy.sum(a);  % Compute sum
    >> disp(double(s));   % Convert to MATLAB double



Check if your PyTorch is loaded properly:

.. code:: matlab

    >> torch = py.importlib.import_module('torch');   
    >> torch.cuda.is_available()
