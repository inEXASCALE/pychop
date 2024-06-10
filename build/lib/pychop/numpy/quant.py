import numpy as np

class quantizer():
    def __init__(self, bits, epsilon=1e-12):
        self.bits = bits
        self.alpha_q = -2**(self.bits - 1)
        self.beta_q = 2**(self.bits - 1) - 1
        self.epsilon = epsilon 
        
    def quantize(self, x):
        x_min = np.min(x)
        x_max = np.max(x) + self.epsilon
        self.s, self.z = self.compute_scaling(x_min, x_max, 
                                              self.alpha_q, self.beta_q)
        
        return self.quantization(x, self.s, self.z)
        
        
    def dequantize(self, x_q):
        return self.dequantization(x_q, self.s, self.z)
        
        
    def quantization(self, x, s, z):
        x_q = np.round(1 / s * x + z, decimals=0)
        x_q = np.clip(x_q, a_min=self.alpha_q, a_max=self.beta_q)

        return x_q

    def dequantization(self, x_q, s, z):
        x_q = x_q.astype(np.int32)
        x = s * (x_q - z)
        x = x.astype(np.float32)
        return x

    def compute_scaling(self, alpha, beta, alpha_q, beta_q):
        s = (beta - alpha) / (beta_q - alpha_q)
        z = int((beta * alpha_q - alpha * beta_q) / (beta - alpha))

        return s, z
