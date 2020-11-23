# GHOST - Physical Problem Definition

from abc import ABC, abstractmethod
import numpy as np

class ConservationLaw(ABC):
    
    def __init__(self, d, N_eq):
        self.d = d
        self.N_eq = N_eq
        
    @abstractmethod
    def build_physical_flux(self):
        pass
    
    @abstractmethod
    def build_numerical_flux(self):
        pass
        

class ConstantAdvection(ConservationLaw):
    
    def __init__(self, a, alpha):
        
        self.a = a
        self.alpha = alpha
        super().__init__(len(self.a), 1)
        
        
    def build_physical_flux(self):
        
        def f(self, u,x):
            return [self.a[i]*u 
                for i in range(0, self.d)]
        
        return lambda u,x: f(self,u,x)
        
    
    def build_numerical_flux(self):
        
        def f_star(self, u_m, u_p, x, n):
              a_dot_n = n @ self.a
              return 0.5*a_dot_n*(u_m + u_p) - 0.5*self.alpha*np.abs(a_dot_n)*(u_p - u_m)
  
        return lambda u_m, u_p, x, n: f_star(self, u_m, u_p, x, n)

            
class Euler(ConservationLaw):
    
    def __init__(self, d, gamma=1.4, numerical_flux="roe"):
        
        self.gamma = gamma
        self.numerical_flux = numerical_flux
        super().__init__(d, d+2)
        
        
    def primitive_to_conservative(self,q):
        
        v = q[1:self.N_eq-1]
        # q = [rho, v_1, ... v_d, p]
        return np.concatenate(([q[0]], q[0]*v, 
                              [q[self.N_eq-1]/(self.gamma-1)+ 
                              0.5*q[0]*(np.linalg.norm(v))**2]))
    
    
    def conservative_to_primitive(self,u):
        
        # u = [rho, rho*v_1, ..., rho*v_d, e]
        v = u[1:self.N_eq-1]/u[0]
        return np.concatenate(([u[0]], v, 
                                [(self.gamma-1)*(u[self.N_eq-1] 
                                    - 0.5*u[0]*np.linalg.norm(v)**2)]))
       
        
    def build_physical_flux(self):
        
        def f(self, u,x):
            q = self.conservative_to_primative(u)
            
            f_momentum = q[0]*np.outer(q[1:self.N_eq], q[1:self.N_eq]) \
                + q[self.N_eq-1]*np.eye(self.N_eq)
            
            return [np.concatenate(u[1+m], f_momentum[:,m],
                                   q[1+m]*(u[self.N_eq-1] + q[self.N_eq - 1]))  
                for m in range(0, self.d)]
        
        return lambda u,x: f(self,u,x)


    def build_numerical_flux(self):
        
        def f_star(self, u_m, u_p, x, n):
            
            return None
            
        return lambda u_m, u_p, x, n: f_star(self, u_m, u_p, x, n)
