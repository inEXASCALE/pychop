import numpy as np

class Simulate():
    """Simulate a customised floating point system with rounding methods
   
    Only for demonstration and a small number of floating point numbers.
    For large number of floating point numbers, please use the Chop class.

    Parameters
    ----------
    base : int
        The base (or radix) in the floating point number system.
    
    t : int 
        The precision in the floating point number system.
    
    emin, emax : int
        The exponent range, with emin < e < emax.
        If emin is omited, emin = 1 - emax which conform to IEEE 754 standard.
        
    sign : boolean, default=False
        Whether or not give sign to the floating point numbers.
        If ``sign=False``, then the generated floating point numbers are nonnegative.

    subnormal : boolean, default=False
        Whether or not to include subnormal numbers. 
        If subnormal numbers are not included, the floating point numbers are normalized. 

    rmode : int, default=1
        Ways to round the values in the floating point system.
        There are 6 options for rounding: 
            1. Round to nearest using round to even last bit to break ties (the default).
            2. Round towards plus infinity (round up).
            3. Round towards minus infinity (round down).
            4. Round towards zero.
            5. Stochastic rounding - round to the next larger or next smaller floating-point number 
                with probability proportional to the distance to those floating-point numbers.
            6. Stochastic rounding - round to the next larger or next smaller floating-point number 
                with equal probability.

    Methods
    ----------
    generate():
        Generate the floating point numbers given user specified parameters.

    rounding(x):
        Round the values ``x`` in terms of the predefined rounding mode.
        
    
    """

    def __init__(self, base, t, emax, emin=None, sign=False, subnormal=False, rmode=1):
        self.base = base
        self.t = t
        self.emax = emax
        if emin is None:
            self.emin = 1 - self.emax # using IEEE 754 assumption by default 
        else:
            self.emin = emin
            
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
        
        self.__fit__ = False
        

    def generate(self):
        m_max = self.base**self.t - 1
        
        if self.subnormal:
            m_min = 1
        else:
            m_min = self.base**(self.t - 1)

        i = 1
        n = (self.emax - self.emin + 1) * (m_max - m_min + 1)

        if self.sign:
            self.fp_numbers = np.zeros(2*n+1)
            for e in np.arange(self.emin, self.emax+1):
                for m in np.arange(m_min, m_max+1):
                    self.fp_numbers[n+i] = m*self.base**int(e - self.t)
                    self.fp_numbers[n-i] = -m*self.base**int(e - self.t)
                    i = i + 1
        else:
            self.fp_numbers = np.zeros(n+1)
            for e in np.arange(self.emin, self.emax+1):
                for m in np.arange(m_min, m_max+1):
                    self.fp_numbers[i] = m*self.base**int(e - self.t)
                    i = i + 1
                    
            
        self.underflow_bound = min(np.abs(self.fp_numbers))
        self.overflow_bound = max(np.abs(self.fp_numbers))
        
        self.__fit__ = True
        return self.fp_numbers
    
    
    def rounding(self, x):
        """Simulate a customised floating point system with rounding methods

        Parameters
        ----------
        x : flaot or numpy.ndarray
            The values to be rounded.

        """
        
        if self.__fit__ == False:
            self.generate()
            
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
