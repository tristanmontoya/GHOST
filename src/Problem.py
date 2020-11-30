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
        
        def f(self, u):
            return [self.a[i]*u 
                for i in range(0, self.d)]
        
        return lambda u,x: f(self,u)
        
    
    def build_numerical_flux(self):
        
        def f_star(self, u_m, u_p, x, n):
              a_dot_n = n.T @ self.a
              return 0.5*a_dot_n*(u_m + u_p) \
                  - 0.5*self.alpha*np.abs(a_dot_n)*(u_p - u_m)
  
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
       
    
    def roe_flux(self,u_m, u_p):
        
        # solve a locally 1d riemann problem for 2d normal/tangential
        # problem (set y-momentum to zero for truly 1d)
        # will have to modify for 3D extension
        
        # difference in conserved variables
        du = u_p - u_m
        
        # get velocities and square magnitudes
        v_m = u_m[1:3]/u_m[0]
        Vs_m = v_m[0]**2 + v_m[1]**2
        v_p = u_p[1:3]/u_p[0]
        Vs_p = v_p[0]**2 + v_p[1]**2
        
        # compute parameter vectors
        z_m = np.sqrt(u_m[0])*np.array([1.0, 
                                        v_m[0],
                                        v_m[1],
                                        (u_m[3] + (self.gamma-1)*(
                                            u_m[3] 
                                            - 0.5*u_m[0]*Vs_m))
                                        /u_m[0]])
        z_p = np.sqrt(u_p[0])*np.array([1.0, 
                                        v_p[0],
                                        v_p[1],
                                        (u_p[3] + (self.gamma-1)*(
                                            u_p[3] 
                                            - 0.5*u_p[0]*Vs_p))
                                        /u_p[0]])
        
        #pressure
        p_m = (self.gamma-1)*(u_m[3] - 0.5*u_m[0]*Vs_m)
        p_p = (self.gamma-1)*(u_p[3] - 0.5*u_p[0]*Vs_p)
        
        # compute roe average
        avg = (z_m + z_p)/(z_m[0] + z_p[0])
        v = avg[1:3]
        Vs = v[0]**2 + v[1]**2
        h = avg[3]
        a =np.sqrt((self.gamma-1)*(h-0.5*Vs))
        
        # right eigenvector matrix (Toro p. 125)
        X = np.array([[1., 1., 0, 1.],
                      [v[0] - a, v[0], 0, v[0]+a],
                      [v[1], v[1], 1, v[1]],
                       [h - v[0]*a, 0.5*Vs, v[1], h + v[0]*a]])
        
        # eigenvalues
        eigval = np.array([v[0]-a, v[0], v[0], v[0]+a])
        
        # wave strengths
        alpha_2 = (self.gamma-1)/(a**2)*(du[0]*(h - v[0]**2) + v[0]*du[1] 
                                         - (du[3] -(du[2] - v[1]*du[0])*v[1]))
        alpha_1 = 1.0/(2.0*a)*(du[0]*(v[0]+a) - du[1] - a*alpha_2)
        alpha = np.array([alpha_1,
                          alpha_2,
                          du[2] - v[1]*du[0],
                          du[0] - (alpha_1 + alpha_2)])
        
        # fluxes
        f_m = np.array([u_m[1], 
                        u_m[0]*v_m[0]**2 + p_m, 
                        u_m[0]*v_m[0]*v_m[1],
                        v_m[0]*(u_m[3]+ p_m)])
     
        f_p = np.array([u_p[1], 
                        u_p[0]*v_p[0]**2 + p_p, 
                        u_p[0]*v_p[0]*v_p[1],
                        v_p[0]*(u_p[3]+ p_p)])
        
        return 0.5*(f_m + f_p) - 0.5*X @ (np.abs(eigval)*alpha)
         
       
    def flux_tensor(self, u):
            
        # gets the full flux tensor (N_e by d array)
        
        # get primitive vars
        q = self.conservative_to_primitive(u) 
        
        return np.vstack(([[u[1:self.N_eq-1]], 
                         q[0]*np.outer(q[1:self.N_eq-1], q[1:self.N_eq-1]) 
                         + q[self.N_eq-1]*np.eye(self.d),
                         [q[1:self.N_eq-1]*(u[self.N_eq-1] + q[self.N_eq - 1])]]))
    
    
    def build_physical_flux(self):
        
        def f(self, u):
            
            N_omega = u.shape[1]
            nodal_flux = np.zeros([N_omega, self.N_eq, self.d])
            
            # u size N_eq x N_omega
            for i in range(0, N_omega):
                nodal_flux[i,:,:] = self.flux_tensor(u[:,i])
                
            return [nodal_flux[:,:,m].T for m in range(0, self.d)]
        
        return lambda u,x: f(self, u)


    def build_numerical_flux(self):
        
        if self.numerical_flux == "central":
            
            def f_star(self, u_m, u_p, n):
                
                return (self.flux_tensor(u_m) + self.flux_tensor(u_p)) @ n
         
        
        elif self.numerical_flux == "roe":
            
            if self.d == 1:
                
                def f_star(self, u_m, u_p, n):
                    
                    fn_2d = self.roe_flux(np.array([u_m[0],
                                                    u_m[1]*n[0], 
                                                    0.0, 
                                                    u_m[2]]), 
                                          np.array([u_p[0],
                                                    u_p[1]*n[0], 
                                                    0.0, 
                                                    u_p[2]]))
                    
                    return np.array([fn_2d[0], fn_2d[1]*n[0], fn_2d[3]])
                
            elif self.d == 2:      
                
                def f_star(self, u_m, u_p, n):
                    
                    fn_2d = self.roe_flux(np.array([u_m[0], 
                                                    u_m[1]*n[0] + u_m[2]*n[1], 
                                                    -u_m[1]*n[1] + u_m[2]*n[0],
                                                    u_m[3]]),
                                          np.array([u_p[0],
                                                    u_p[1]*n[0] + u_p[2]*n[1], 
                                                    -u_p[1]*n[1] + u_p[2]*n[0],
                                                    u_p[3]]))
                    
                    return np.array([fn_2d[0], 
                                     fn_2d[1]*n[0] - fn_2d[2]*n[1],
                                     fn_2d[1]*n[1] + fn_2d[2]*n[0], 
                                     fn_2d[3]])
                    
            else: 
                raise NotImplementedError
                
        else:
            
            raise NotImplementedError
            
        return lambda u_m, u_p, x, n: np.array(
            [f_star(self, u_m[:,i], u_p[:,i], n[:,i])
                      for i in range(0,u_m.shape[1])]).T

