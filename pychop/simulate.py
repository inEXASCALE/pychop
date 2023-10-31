import numpy as np

class simulate():
    def __init__(self, beta, t, emin, emax, sign=False, subnormal=1, rmode=1):
        self.beta = beta
        self.t = t
        self.emin = emin
        self.emax = emax
        self.sign = sign
        self.subnormal = subnormal
        
        if rmode not in {1, 2, 3, 4, 5, 6}:
            raise ValueError("Please enter valid value.")
        
        self.rmode = rmode

        if self.rmode == 2:
            self._rounding = np.frompyfunc(self._round_to_plus_inf, 1, 1)

        elif self.rmode == 3:
            self._rounding = np.frompyfunc(self._round_to_minus_inf, 1, 1)

        elif self.rmode == 4:
            self._rounding = np.frompyfunc(self._round_to_zero, 1, 1)
        
        elif self.rmode == 5:
            self._rounding = np.frompyfunc(self._round_to_stochastic_distance, 1, 1)

        elif self.rmode == 6:
            self._rounding = np.frompyfunc(self._round_to_stochastic_uniform, 1, 1)

        else:
            self._rounding = np.frompyfunc(self._round_to_nearest, 1, 1)

        

    def generate(self):
        m_max = self.beta**self.t - 1
        
        if self.subnormal:
            m_min = 1
        else:
            m_min = self.beta**(self.t - 1)

        i = 1
        n = (self.emax - self.emin + 1) * (m_max - m_min + 1)

        if self.sign:
            self.fp_numbers = np.zeros(2*n+1)
            for e in np.arange(self.emin, self.emax+1):
                for m in np.arange(m_min, m_max+1):
                    self.fp_numbers[n+i] = m*self.beta**int(e - self.t)
                    self.fp_numbers[n-i] = -m*self.beta**int(e - self.t)
                    i = i + 1
        else:
            self.fp_numbers = np.zeros(n+1)
            for e in np.arange(self.emin, self.emax+1):
                for m in np.arange(m_min, m_max+1):
                    self.fp_numbers[i] = m*self.beta**int(e - self.t)
                    i = i + 1
                    
            
        self.underflow_bound = min(np.abs(self.fp_numbers))
        self.overflow_bound = max(np.abs(self.fp_numbers))
        
        return self.fp_numbers
    
    
    def rounding(self, x):
        if hasattr(x, "__len__"):
            x_copy = x.copy()
            id_underflow = np.abs(x) < self.underflow_bound
            id_overflow = np.abs(x) > self.overflow_bound
            x_copy = self._rounding(x_copy)
            x_copy[id_underflow] = 0
            x_copy[id_overflow] = np.inf
            return x_copy
        
        else:
            if np.abs(x) < self.underflow_bound:
                return 0
            
            if np.abs(x) > self.overflow_bound:
                return np.inf
            
            return self._rounding(x)
        

    def _round_to_nearest(self, x):
        # Round to nearest using round to even last bit to break ties
        return self.fp_numbers[np.argmin(np.abs(self.fp_numbers - x))]
    

    def _round_to_plus_inf(self, x):
        # Round towards plus infinity
        return min(self.fp_numbers[self.fp_numbers >= x])
    
    def _round_to_minus_inf(self, x):
        # Round towards minus infinity
        return max(self.fp_numbers[self.fp_numbers <= x])
    
    def _round_to_zero(self, x):
        # Round towards zero
        if x >= 0:
            return min(self.fp_numbers[self.fp_numbers >= x])
        else:
            return max(self.fp_numbers[self.fp_numbers <= x])
    
    def _round_to_stochastic_distance(self, x):
        # round to the next larger or next smaller floating-point number 
        # with probability proportional to the distance to those floating-point numbers
        distances = np.argsort(np.abs(self.fp_numbers - x))[:2]
        proba = np.random.uniform(0, self.fp_numbers[distances[0]] + self.fp_numbers[distances[1]])
        if proba >= self.fp_numbers[distances[0]]:
            return self.fp_numbers[distances[1]]
        else:
            return self.fp_numbers[distances[0]]

    def _round_to_stochastic_uniform(self, x):
        # round to the next larger or next smaller floating-point number with equal probability
        distances = np.argsort(np.abs(self.fp_numbers - x))[:2]
        proba = np.random.uniform(0, 1)
        if proba >= 0.5:
            return self.fp_numbers[distances[1]]
        else:
            return self.fp_numbers[distances[0]]
