# GHOST - Physical Problem Definition

from abc import ABC, abstractmethod
import numpy as np

class PhysicalFlux(ABC):
    
    def __init__(self, d, N_e):
        self.d = d
        self.N_e = N_e
        
    @abstractmethod
    def __call__(self, u,x):
        pass
        
class VariableAdvectionPhysicalFlux(PhysicalFlux):
    
    def __init__(self, a):
        super().__init__(len(a), 1)
        self.a = a
    
    def __call__(self, u,x):
        return [self.a(x)[i]*u for i in range(0, self.d)]


class ConstantAdvectionPhysicalFlux(PhysicalFlux):
    
    def __init__(self,a):
        super().__init__(len(a), 1)
        self.a = a
        
    def __call__(self, u,x):
        return [self.a[i]*u for i in range(0, self.d)]

 
class NumericalFlux(ABC):
    def __init__(self, d, N_e):
        raise NotImplementedError
    
    @abstractmethod
    def __call__(self, u_m, u_p, x, n):
        pass


class ConstantAdvectionNumericalFlux(NumericalFlux):
    
    def __init__(self, a, alpha):
        super().__init__(len(a), 1)
        self.a = a
        self.alpha = alpha
        
    def __call__(self, u_m, u_p, x, n):
        a_dot_n = np.dot(self.a, n)
        return 0.5*a_dot_n*(u_m + u_p) - 0.5*self.alpha*np.abs(a_dot_n)*(u_p - u_m)
