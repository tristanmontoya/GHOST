# GHOST - Time integrators

import numpy as np
from math import floor
import pickle
import time


class TimeIntegrator:
    
    def __init__(self, residual, dt, discretization_type="rk44"):
        
        self.dt_target = dt
        self.type = discretization_type
        self.R = residual
    
    
    @staticmethod
    def calculate_time_step(spatial_discretization, wave_speed, beta, h=None):
        
        if h is None:
            
            h = np.amin(
                spatial_discretization.mesh.extent)/(
                    spatial_discretization.mesh.N_el ** (
                        1.0/spatial_discretization.d))
                    
        return beta/(2*max(spatial_discretization.p) + 1.0)*h/wave_speed
        
    
    def run(self, u_0, T, results_path,
            write_interval, 
            print_interval, 
            restart=True,
            prefix=""):
        
        if restart:
            
            if os.path.isfile(results_path+"times.dat"):
            
                times = None
                dt = None
                N_t = None
                N_write = None
                self.is_done = False
                
                times = pickle.load(open(results_path+"times.dat", "rb"))
                dt = pickle.load(open(results_path+"time_step_size.dat", "rb" ))
                N_t = pickle.load(open(results_path+"number_of_steps.dat", "rb" ))
         
                u = np.copy(u_0)
                n_0 = times[-1][0]
                t = times[-1][1]
                
                screen = open(results_path + "screen.txt", "a")
                print(prefix, "restarting from time step ", n_0, 
                     ", t=", t, file=screen)
                screen.close()
            
            else:
                
                screen = open(results_path + "screen.txt", "w")
                print(prefix, "No previous file found for restart. Starting new run.",
                      file=screen)
                screen.close()
                restart=False
    
        else:
            
            # calculate number of steps to take and actual time step
            N_t = floor(T/self.dt_target) 
            dt = T/N_t
            self.is_done = False
        
            u = np.copy(u_0)
            n_0 = 0
            t = 0
            times = [[n_0,t]]
               
            pickle.dump(dt, open(results_path+"time_step_size.dat", "wb" ))
            pickle.dump(N_t, open(results_path+"number_of_steps.dat", "wb" ))
        
        # interval between prints and writes to file
        if print_interval is None:
            N_print = N_t
        else:
            N_print = floor(print_interval/dt)
        if write_interval is None:
            N_write = N_t
        else:
            N_write = floor(write_interval/dt)    
              
        screen = open(results_path + "screen.txt", "w")
        print(prefix, " dt = ", dt, file=screen)
        print(prefix, "writing every ", 
              N_write, " time steps, total ", N_t, file=screen)
        screen.close()
        start = time.time()
        
        for n in range(n_0,N_t):
            
            u = np.copy(self.time_step(u,t,dt)) # update solution

            t = t + dt
            if ((n+1) % N_print == 0) or (n+1 == N_t):
                screen = open(results_path + "screen.txt", "a")
                print(prefix, "time step: ", n+1, "t: ", t, "wall time: ", 
                      time.time()-start, file=screen)
                screen.close()
                
            if ((n+1) % N_write == 0) or (n+1 == N_t):
                screen = open(results_path + "screen.txt", "a")
                print(prefix, "writing time step ", n+1, "t = ", t, file=screen)
                screen.close()
                times.append([n+1,t])
                
                pickle.dump(u, open(results_path+"res_" +
                                    str(n+1) + ".dat", "wb" ))
                
                pickle.dump(times, open(
                   results_path+"times.dat", "wb" ))
                
            if np.isnan(np.sum(np.array([[np.sum(u[k][e]) 
                                          for e in range(0, u[k].shape[0])] 
                                         for k in range(0,len(u))]))):
                
                pickle.dump(times, open(
                   results_path+"times.dat", "wb" ))
                return None
        
        pickle.dump(times, open(
                   results_path+"times.dat", "wb" ))
        
        
        screen = open(results_path + "screen.txt", "a")
        print(prefix, "Simulation complete.",file=screen)
        screen.close()
        
        self.is_done = True
        
        return u
    

    def time_step(self, u, t, dt):
        
        if self.type == "rk44":
            # Notation adapted from Lomax, Pulliam, and Zingg
        
            r_u = self.R(u,t)

            u_hat_nphalf = [np.array([u[k][e,:] + 0.5 * dt * r_u[k][e,:] 
                            for e in range(0, u[k].shape[0])])
                            for k in range(0, len(u))]
            
            r_u_hat_nphalf = self.R(u_hat_nphalf, t + 0.5*dt)

            u_tilde_nphalf = [np.array([u[k][e,:] 
                                        + 0.5 * dt * r_u_hat_nphalf[k][e,:]
                            for e in range(0, u[k].shape[0])])
                            for k in range(0, len(u))]
                    
            r_u_tilde_nphalf = self.R(u_tilde_nphalf, t + 0.5*dt)

            u_bar_np1 = [np.array([u[k][e,:] + dt * r_u_tilde_nphalf[k][e,:] 
                          for e in range(0, u[k].shape[0])])
                          for k in range(0, len(u))]
                         
            r_u_bar_np1 = self.R(u_bar_np1, t + 1.0*dt)

            return [np.array([u[k][e,:] + (1. / 6.) * dt * (
                r_u[k][e,:] + 2. * (r_u_hat_nphalf[k][e,:] 
                                    + r_u_tilde_nphalf[k][e,:]) 
                + r_u_bar_np1[k][e,:])
                        for e in range(0, u[k].shape[0])])
                        for k in range(0, len(u))]
            
        elif self.type == "explicit_euler":
            
            r = self.R(u,t)
            return [np.array([u[k][e,:] + dt *r[k][e,:] 
                              for e in range(u[k].shape[0])])
                    for k in range(0, len(u))]
        
        else:
            raise NotImplementedError