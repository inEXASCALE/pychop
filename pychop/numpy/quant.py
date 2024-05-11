import numpy as np
import warnings 


class quant():
    """
    Parameters
    ----------
    bits : int, default=8
        The bitwidth of integer format, the larger it is, the wider range the quantized value can be.
        
    sign : bool, default=1
        Whether or not to quantize the value to symmetric integer range.
    
    zpoint : bool, default=1
        Whether or not to set the zero point to zero. If `zpoint=0`, then the quantized range must be symmetric.
        
    rd_func : function, default=None
        The rounding function used for the quantization. The default is round to nearest.
        
    clip_range : list, default=None
        The clipping function for the quantization.
        
    Methods
    ----------
    quant(x):
        Method that quantize ``x`` to the user-specific arithmetic format.
        
    """
    def __init__(self, bits=8, sign=1, zpoint=1, rd_func=None, clip_range=None, epsilon=1e-12):
        self.bits = bits
        self.sign = sign
        self.zpoint = zpoint
        self.epsilon = epsilon 
        self.rd_func = rd_func
        self.clip_range = clip_range
        
        if bits in {8, 16, 32, 64}:
            if bits == 8:
                self.intType = np.int8
                
            elif bits == 16:
                self.intType = np.int16
                
            elif bits == 32:
                self.intType = np.int32
                
            elif bits == 64:
                self.intType = np.int64
                
        else:
            warnings.warn("Current int type not support this bitwidth, use int64 to simulate.")
            self.intType = np.int64
        
        if self.sign == 1:
            if self.zpoint == 1:
                self.alpha_q = -2**(self.bits - 1)
                self.beta_q = 2**(self.bits - 1) - 1
            else:
                self.beta_q = 2**(self.bits - 1) - 1
                self.alpha_q = -self.beta_q
                
        else:
            if self.zpoint == 0:
                self.beta_q = 2**(self.bits - 1) - 1
                self.alpha_q = -self.beta_q
            else:
                raise ValueError('Please set `zpoint` to 1.')
        
        if self.rd_func is None:
            self.rd_func = lambda x: np.round(x, decimals=0)
            
        if self.clip_range is None:
            self.clip_func = lambda x: np.clip(x, a_min=self.alpha_q, a_max=self.beta_q)
        else:
            self.clip_func = lambda x: np.clip(x, a_min=self.clip_range[0], a_max=self.beta_q[self.clip_range[1]])
            
            
    def __call__(self, x):
        x_min = np.min(x)
        x_max = np.max(x)
        
        if self.sign != 1:
            abs_max = max(abs(x_min), abs(x_max))
            x_min = -abs_max
            x_max = abs_max
            
        self.scaling, self.zpoint = self.compute_scaling(x_min, x_max, self.alpha_q, self.beta_q)
        
        try:
            x_q = self.quantization(x, self.scaling, self.zpoint)
        except:
            self.scaling, self.zpoint = self.compute_scaling(x_min, x_max+self.epsilon, self.alpha_q, self.beta_q)
            x_q = self.quantization(x, self.scaling, self.zpoint)
            
        return x_q
        
    def dequant(self, x_q):
        return self.dequantization(x_q, self.scaling, self.zpoint)
        
    def quantization(self, x, s, z):
        x_q = self.rd_func((x - z)/s)
        x_q = self.intType(self.clip_func(x_q))
        return x_q

    def dequantization(self, x_q, s, z):
        x_q = x_q.astype(np.float32)
        x = s * x_q + z
        return x

    def compute_scaling(self, alpha, beta, alpha_q, beta_q):
        s = (beta - alpha) / (beta_q - alpha_q)
        z = (alpha * beta_q - beta * alpha_q) / (beta_q - alpha_q)

        return s, z
    
