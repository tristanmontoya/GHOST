# GHOST - Time integrators

import numpy as np
from math import floor
import time
import json
import os

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
            
            if os.path.isfile(results_path+"data.json"):
            
                times = None
                dt = None
                N_t = None
                N_write = None
                self.is_done = False
                
                with open(results_path+"data.json") as file:
                   data = json.load(file)

                times = data["write_times"]
                dt = data["time_step_size"]
                N_t = data["number_of_steps"]
         
                u = np.copy(u_0)
                n_0 = times[-1][0]
                t = times[-1][1]
                
                with open(results_path + "screen.txt", "a") as screen:
                    print(prefix, "restarting from time step ", n_0, 
                        ", t=", t, file=screen)
            
            else:
                
                with open(results_path + "screen.txt", "w") as screen:
                    print(prefix, 
                        "No previous file found for restart. Starting new run.",
                        file=screen)
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
            
            with open(results_path+"data.json", "w") as file:
                data = {"time_step_size": dt, "number_of_steps": N_t,
                    "write_times": times }
                json.dump(data,file)
        
        # interval between prints and writes to file
        if print_interval is None:
            N_print = N_t
        else:
            N_print = floor(print_interval/dt)
        if write_interval is None:
            N_write = N_t
        else:
            N_write = floor(write_interval/dt)    
              
        with open(results_path + "screen.txt", "w") as screen:
            print(prefix, "dt = ", dt, file=screen)
            print(prefix, "writing every ", 
                N_write, " time steps, total ", N_t, file=screen)

        start = time.time()
        
        for n in range(n_0,N_t):
            
            u = np.copy(self.time_step(u,t,dt)) # update solution

            t = t + dt
            if ((n+1) % N_print == 0) or (n+1 == N_t):

                with open(results_path + "screen.txt", "a") as screen:
                    print(prefix, "time step: ", n+1, "t: ", t, "wall time: ", 
                        time.time()-start, file=screen)
                
            if ((n+1) % N_write == 0) or (n+1 == N_t):

                with open(results_path + "screen.txt", "a") as screen:
                    print(prefix, "writing time step ", n+1, "t = ", t, file=screen)
                    
                times.append([n+1,t])
                
                with open(results_path+"res_" + str(n+1)
                    + ".json", "w") as file:
                    json.dump([[u[k][e].tolist() 
                        for e in range(0, u[k].shape[0])]
                        for k in range (0, len(u))], file)
                
                with open(results_path+"data.json", "w") as file:
                    json.dump(data, file)
                
            if np.isnan(np.sum(np.array([[np.sum(u[k][e]) 
                                          for e in range(0, u[k].shape[0])] 
                                         for k in range(0,len(u))]))):
                
                with open(results_path+"data.json", "w") as file:
                    json.dump(data, file)
                return None
        
        with open(results_path+"data.json", "w") as file:
                    json.dump(data, file)
        
        with open(results_path + "screen.txt", "a") as screen:
            print(prefix, "simulation complete!",file=screen)
        
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
                              for e in range(0, u[k].shape[0])])
                    for k in range(0, len(u))]
        
        else:
            raise NotImplementedError