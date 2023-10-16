import numpy as np

class simulateFTP():
    def __init__(self, beta, t, emin, emax, subnormal=0):
        self.beta = beta
        self.t = t
        self.emin = emin
        self.emax = emax
        self.subnormal = subnormal
        
    def generate(self, sign):
        m_max = self.beta**self.t - 1
        
        if not self.subnormal:
            m_min = self.beta**(self.t - 1)
            i = 0
            n = (self.emax - self.emin + 1) * (m_max - m_min + 1)
            
            if not sign:
                numbers = np.empty(n)
            else:
                numbers = np.empty(n*2)
            
            for e in np.arange(self.emin, self.emax+1):
                for m in np.arange(m_min, m_max+1):
                    numbers[i] = m*self.beta**int(e - self.t)
                    i = i + 1
                
        else:
            m_min = self.beta**(self.t - 1)
            i = 0
            n = (self.emax + 1 - self.emin) * (m_max - m_min + 1)
            
            if not sign:
                numbers = np.empty(n)
            else:
                numbers = np.empty(n*2)
                
            for e in np.arange(self.emin, self.emax+1, dtype=int):
                for m in np.arange(m_min, m_max+1, dtype=int):
                    numbers[n+i] = m*self.beta**int(e - self.t)
                    numbers[n-i-1] = -m*self.beta**int(e - self.t)
                    i = i + 1
        
        self.floating_numbers = numbers
        
        nonnegatives = numbers[numbers > 0]
        self.underflow_bound = min(nonnegatives)
        self.overflow_bound = max(nonnegatives)
        
        return self.floating_numbers
    
    
    def rounding(self, x, rmode=0): # e.g., round to nearest
        if x < self.underflow_bound:
            return 0
        
        if x > self.overflow_bound:
            return np.inf
        elif x < -self.overflow_bound:
            return -np.inf
        
        match rmode:
            case 1:
                return self.floating_numbers[np.argmin(np.abs(self.floating_numbers - x))]
            case 2:
                return min(self.floating_numbers[self.floating_numbers >= x])
            case 3:
                return max(self.floating_numbers[self.floating_numbers <= x])
    
            case 4:
                if x >= 0:
                    return min(self.floating_numbers[self.floating_numbers >= x])
                else:
                    return max(self.floating_numbers[self.floating_numbers <= x])
            case 5:
                distances = np.argsort(np.abs(self.floating_numbers - x))[:2]
                proba = np.random.uniform(0, self.floating_numbers[distances[0]] + self.floating_numbers[distances[1]])
                if proba >= self.floating_numbers[distances[0]]:
                    return self.floating_numbers[distances[1]]
                else:
                    return self.floating_numbers[distances[0]]
            case 6:
                distances = np.argsort(np.abs(self.floating_numbers - x))[:2]
                proba = np.random.uniform(0, 1)
                if proba >= 0.5:
                    return self.floating_numbers[distances[1]]
                else:
                    return self.floating_numbers[distances[0]]
            case _:
                raise ValueError("Please enter valid value.")
