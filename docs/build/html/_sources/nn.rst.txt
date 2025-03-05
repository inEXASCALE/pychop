Quantization aware training
=====================================================

``pychop`` provides easy-to-use API for quantization aware training.  

Simply load the module via:

.. code:: python

    from pychop import QuantLayer

``QuantLayer`` enables the quantization components of ``quant``, ``chop``, and ``fixed_point`` to be integrated into neural network training, 
which is often referred to as quantization-aware training.


.. admonition:: Note

    The QuantLayer only support backend of Torch, so as to successfully run this functionality, please use

    .. code:: python

        pychop.backend('torch') 



The usage of QuantLayer simply extended by the ``quant``, ``chop``, and ``fixed_point``, therefore, we need to first load the corresponding modules via:

.. code:: python

    from pychop import quant
    from pychop import chop
    from pychop import fpoint


The quantization-aware training simply perform by plugging the ``QuantLayer`` into neural network building. We illustrate its usage in fully connected layer training:


.. code:: python
    
    class NeuralNet(nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super(NeuralNet, self).__init__()
            self.quant1 = QuantLayer(fpoint(4, 4)) 
            self.quant2 = QuantLayer(chop('h'))
            self.quant3 = QuantLayer(quant())
            self.fc1 = nn.Linear(input_size, hidden_size) 
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size, num_classes)  
        
        def forward(self, x):
            out = self.quant1(self.fc1(x))
            out = self.quant2(self.relu(out))
            out = self.quant3(self.fc2(out))
            return out

