# GHOST - Test Problem Drivers

import os
import pickle
import numpy as np
from Solver import Solver
from Mesh import Mesh1D, Mesh2D
import meshzoo
import meshio


def grid_refine(project_title="advection_grid_refine_test",
                d=1,
                n_refine=4,
                p=2,
                problem="constant_advection",
                integration_type="quadrature",
                solution_representation="modal",
                numerical_flux="upwind",
                map_type="isoparametric",
                form="both",
                correction="dg",
                results_path=None):
    
    # simulation parameters dictionary
    params = {}
    params["project_title"] = project_title
    params["problem"] = problem
    params["numerical_flux"] = numerical_flux
    params["integration_type"] = integration_type
    params["solution_representation"] = solution_representation
    params["solution_degree"] = p
    params["correction"] = correction
    params["time_integrator"] = "rk44"
    params["time_step_scale"] = 0.1
    
    if results_path is None:
        results_path = "../results/" + project_title + "/"
    
    if map_type == "affine":
        p_geo = 1
    elif map_type == "isoparametric":
        p_geo = p
        
    if d == 1:
        return grid_refine_1d(params, form,n_refine,p_geo, 
                              4*p+1, results_path)
    elif d == 2:
        return grid_refine_2d(params, form,n_refine,p_geo, 
                              4*p, results_path)
    else:
        raise NotImplementedError


def grid_refine_1d(params, form, n_refine, p_geo, error_quadrature_degree, results_path):

    #start with 4 elements 
    M = 5
    
    if params["integration_type"] == "quadrature":
        params["volume_quadrature_degree"] = 2*params["solution_degree"] + 1
        
    elif params["integration_type"] == "collocation":
        if p_geo < params["solution_degree"]:
            params["volume_collocation_degree"] = params["solution_degree"]
        else:
            params["volume_collocation_degree"] = 2*params["solution_degree"]
    else:
        raise NotImplementedError
        
    if params["problem"] == "constant_advection":
        params["initial_condition"] = "sine"
        params["wavelength"] = np.array([1.0])
        params["wave_speed"] = np.ones(1)
        params["final_time"] = 1.0
        N_eq = 1
        
    elif params["problem"] == "compressible_euler":
        params["initial_condition"] = "entropy_wave"
        params["specific_heat_ratio"] = 1.4
        params["final_time"] = 1.0
        N_eq = 3
        
    else:
        raise NotImplementedError
        
    if form == "strong" or form == "both":
        params_strong = params.copy()
        params_strong["form"] = "strong"
        dI_strong = np.zeros((N_eq,n_refine))
        dE_strong = np.zeros((N_eq,n_refine))
        error_strong = np.zeros((N_eq,n_refine))
        rates_strong = np.zeros((N_eq,n_refine))

    if form == "weak" or form == "both":
        params_weak = params.copy()
        params_weak["form"] = "strong"
        dI_weak = np.zeros((N_eq,n_refine))
        dE_weak = np.zeros((N_eq,n_refine))
        error_weak = np.zeros((N_eq,n_refine))
        rates_weak = np.zeros((N_eq,n_refine))
        
    if form == "both":
        diff_strong_weak = np.zeros((N_eq,n_refine))
        
    M_list = np.zeros(n_refine)
    
    for n in range(0, n_refine):
        
        mesh = Mesh1D(params["project_title"] + "_M" + str(M),
                           0.0, 1.0, M)
        left = [1.0, 0.0]
        right = [1.0, 1.0]
        mesh.add_bc_on_hyperplanes([left,right],[1,2])
        mesh.make_periodic([1,2])
        
        mesh.map_mesh(f_map=Mesh1D.grid_transformation(
            warp_factor=0.2), p_geo=p_geo)
       
        M_list[n] = M
        
        if form == "strong" or form == "both":
            solver_strong = Solver(params_strong, mesh)
            solver_strong.run(results_path + "/strong/M" + str(M) + "/")
            solver_strong.post_process(process_visualization=False, 
                                       error_quadrature_degree=error_quadrature_degree)
            
            dI_strong[:,n] = solver_strong.I_f - solver_strong.I_0
            dE_strong[:,n] = solver_strong.E_f - solver_strong.E_0
            error_strong[:,n] = solver_strong.calculate_error(norm="L2")
            
            if n == 0:
                rates_strong[:,n] = np.zeros(N_eq)
            else:
                 rates_strong[:,n] = np.array(
                            [(np.log(error_strong[e,n]) - np.log(error_strong[e,n-1]))
                             /(np.log(1.0/M_list[n])-np.log(1.0/M_list[n-1]))
                            for e in range(0,N_eq)])
            
            print("M: ", M, ", strong form")
            print("cons. error: ", dI_strong[:,n])
            print("energy diff: ", dE_strong[:,n])
            print("L2 error: ", error_strong[:,n], 
                  " rate: ", rates_strong[:,n])
        
            
        if form == "weak" or form == "both":
            
            solver_weak = Solver(params_weak, mesh)
            solver_weak.run(results_path + "/weak/M" + str(M) + "/")
            solver_weak.post_process(process_visualization=False,
                                     error_quadrature_degree=error_quadrature_degree)
            
            dI_weak[:,n] = solver_weak.I_f - solver_weak.I_0
            dE_weak[:,n] = solver_weak.E_f - solver_weak.E_0
            error_weak[:,n] = solver_weak.calculate_error(norm="L2")
            
            if n == 0:
                rates_weak[:,n] = np.zeros(N_eq)
            else:
                 rates_weak[:,n] = np.array(
                            [(np.log(error_weak[e,n]) - np.log(error_weak[e,n-1]))/
                             (np.log(1.0/M_list[n])-np.log(1.0/M_list[n-1]))
                            for e in range(0,N_eq)])
            
            print("M: ", M, ", weak form")
            print("cons. error: ", dI_weak[:,n])
            print("energy diff: ", dE_weak[:,n])
            print("L2 error: ", error_weak[:,n], 
                  " rate: ", rates_weak[:,n])
        
        if form == "both":
            diff_strong_weak[:,n] = solver_strong.calculate_difference(solver_weak)
            print("L2 difference (strong vs. weak): ", diff_strong_weak[:,n])
        
        M = M*2
 
    # save results
    np.save(results_path + "/M_list", M_list)
    if form == "strong":
        
        np.save(results_path + "/strong/dI", dI_strong)
        np.save(results_path + "/strong/dE", dE_strong)
        np.save(results_path + "/strong/error", error_strong)
        np.save(results_path + "/strong/rates", rates_strong)
        
        return M_list, dI_strong, dE_strong, error_strong, rates_strong
        
    if form == "weak":
         
        np.save(results_path + "/weak/dI", dI_weak)
        np.save(results_path + "/weak/dE", dE_weak)
        np.save(results_path + "/weak/error", error_weak)
        np.save(results_path + "/weak/rates", rates_weak)
        
        return M_list, dI_weak, dE_weak, error_weak, rates_weak
        
    if form == "both":
        
        np.save(results_path + "/strong/dI", dI_strong)
        np.save(results_path + "/strong/dE", dE_strong)
        np.save(results_path + "/strong/error", error_strong)
        np.save(results_path + "/strong/rates", rates_strong)
        np.save(results_path + "/weak/dI", dI_weak)
        np.save(results_path + "/weak/dE", dE_weak)
        np.save(results_path + "/weak/error", error_weak)
        np.save(results_path + "/weak/rates", rates_weak)
        np.save(results_path + "/diff_strong_weak", diff_strong_weak)
    
        return M_list, diff_strong_weak, dI_strong, dI_weak, dE_strong, dE_weak,\
            error_strong, rates_strong, error_weak, rates_weak
    

def grid_refine_2d(params, form, n_refine, p_geo, error_quadrature_degree, results_path):

    #start with 4 elements 
    M = 5
    
    if params["integration_type"] == "quadrature":
        params["volume_quadrature_degree"] = 2*params["solution_degree"]
        params["facet_quadrature_degree"] = 2*params["solution_degree"] + 1
        
    elif params["integration_type"] == "collocation":
        if p_geo < params["solution_degree"]:
            params["volume_collocation_degree"] = params["solution_degree"]
            params["facet_collocation_degree"] = params["solution_degree"]
        else:
            params["volume_collocation_degree"] = 2*params["solution_degree"]
            params["facet_collocation_degree"] = 2*params["solution_degree"]
    else:
        raise NotImplementedError
        
    if params["problem"] == "constant_advection":
        theta = np.pi/4
        a = np.sqrt(2)
        params["initial_condition"] = "sine"
        params["wavelength"] = np.ones(2)
        params["wave_speed"] = a*np.array([np.sin(theta),np.cos(theta)])
        params["final_time"] = 1.0
        N_eq = 1
        L = 1.0
        
    elif params["problem"] == "compressible_euler":
        params["specific_heat_ratio"] = 1.4
        params["initial_condition"] = "isentropic_vortex"
        params["initial_vortex_centre"] = np.array([5.0,5.0])
        params["background_temperature"] = 1.0
        params["background_velocity"] =np.array([1.0,1.0])
        params["final_time"] = 10.0
        N_eq = 4
        L = 10.0
        
    else:
        raise NotImplementedError
        
    if form == "strong" or form == "both":
           params_strong = params.copy()
           params_strong["form"] = "strong"
           dI_strong = np.zeros((N_eq,n_refine))
           dE_strong = np.zeros((N_eq,n_refine))
           error_strong = np.zeros((N_eq,n_refine))
           rates_strong = np.zeros((N_eq,n_refine))
       
    if form == "weak" or form == "both":
        params_weak = params.copy()
        params_weak["form"] = "strong"
        dI_weak = np.zeros((N_eq,n_refine))
        dE_weak = np.zeros((N_eq,n_refine))
        error_weak = np.zeros((N_eq,n_refine))
        rates_weak = np.zeros((N_eq,n_refine))

    if form == "both":
        diff_strong_weak = np.zeros((N_eq,n_refine))
    
    M_list = np.zeros(n_refine)
    for n in range(0, n_refine):
        
       # read in mesh in GMSH format (here fastest to just generate)
        points, elements = meshzoo.rectangle(
                xmin=0.0, xmax=L,
                ymin=0.0, ymax=L,
                nx=M+1, ny=M+1
                )
        
        if not os.path.exists("../mesh/" + params["project_title"] + "/"):
            os.makedirs("../mesh/" + params["project_title"] + "/")
            
        meshio.write("../mesh/" + params["project_title"] + "/M" + str(M) + ".msh",
                     meshio.Mesh(points, {"triangle": elements}))
        
        mesh = Mesh2D(params["project_title"] + "_M" + str(M),
                      "../mesh/" + params["project_title"] + "/M" + str(M) + ".msh")
        
        # set up periodic boundary conditions
        left = np.array([1.0,0.0,0.0]) 
        right = np.array([1.0,0.0,L])
        bottom = np.array([0.0,1.0,0.0])
        top = np.array([0.0,1.0,L])
        
        mesh.add_bc_on_hyperplanes([left,right,bottom,top],[1,2,3,4])
        mesh.make_periodic((1,2),[1]) # left-right periodic (bcs parallel to axis 1)
        mesh.make_periodic((3,4),[0]) # top-bottom periodic (axis 0)
        
        #curvilinear transformation used in Del Rey Fernandez et al. (2017)
        mesh.map_mesh(f_map=Mesh2D.grid_transformation(warp_factor=0.2, L=L),
                      p_geo=p_geo)
        
        M_list[n] = M
        
        if form == "strong" or form == "both":
            solver_strong = Solver(params_strong, mesh)
            solver_strong.run(results_path + "/strong/M" + str(M) + "/")
            solver_strong.post_process(process_visualization=False, 
                                       error_quadrature_degree=error_quadrature_degree)
            
            dI_strong[:,n] = solver_strong.I_f - solver_strong.I_0
            dE_strong[:,n] = solver_strong.E_f - solver_strong.E_0
            error_strong[:,n] = solver_strong.calculate_error(norm="L2")
            
            if n == 0:
                rates_strong[:,n] = np.zeros(N_eq)
            else:
                 rates_strong[:,n] = np.array(
                            [(np.log(error_strong[e,n]) - np.log(error_strong[e,n-1]))/(np.log(1.0/M_list[n])-np.log(1.0/M_list[n-1]))
                            for e in range(0,N_eq)])
                 
            print("M: ", M, ", strong form")
            print("cons. error: ", dI_strong[:,n])
            print("energy diff: ", dE_strong[:,n])
            
            print("L2 error: ", error_strong[:,n], 
                  " rate: ", rates_strong[:,n])
            
        if form == "weak" or form == "both":
            
            solver_weak = Solver(params_weak, mesh)
            solver_weak.run(results_path + "/weak/M" + str(M) + "/")
            solver_weak.post_process(process_visualization=False,
                                     error_quadrature_degree=error_quadrature_degree)
            
            dI_weak[:,n] = solver_weak.I_f - solver_weak.I_0
            dE_weak[:,n] = solver_weak.E_f - solver_weak.E_0
            error_weak[:,n] = solver_weak.calculate_error(norm="L2")
            
            if n == 0:
                rates_weak[:,n] = np.zeros(N_eq)
            else:
                rates_weak[:,n] = np.array(
                    [(np.log(error_weak[e,n]) - np.log(error_weak[e,n-1]))/(np.log(1.0/M_list[n])-np.log(1.0/M_list[n-1]))
                    for e in range(0,N_eq)])
            
            print("M: ", M, ", weak form")
            print("cons. error: ", dI_weak[:,n])
            print("energy diff: ", dE_weak[:,n])
            print("L2 error: ", error_weak[:,n], 
                  " rate: ", rates_weak[:,n])
        
        if form == "both":
            diff_strong_weak[:,n] = solver_strong.calculate_difference(solver_weak)
            print("L2 difference (strong vs. weak): ", diff_strong_weak[:,n])
            
        M = M*2
 
    # save results
    np.save(results_path + "/M_list", M_list)
    if form == "strong":
        
        np.save(results_path + "/strong/dI", dI_strong)
        np.save(results_path + "/strong/dE", dE_strong)
        np.save(results_path + "/strong/error", error_strong)
        np.save(results_path + "/strong/rates", rates_strong)
        
        return M_list, dI_strong, dE_strong, error_strong, rates_strong
        
    if form == "weak":
         
        np.save(results_path + "/weak/dI", dI_weak)
        np.save(results_path + "/weak/dE", dE_weak)
        np.save(results_path + "/weak/error", error_weak)
        np.save(results_path + "/weak/rates", rates_weak)
        
        return M_list, dI_weak, dE_weak, error_weak, rates_weak
        
    if form == "both":
        
        np.save(results_path + "/strong/dI", dI_strong)
        np.save(results_path + "/strong/dE", dE_strong)
        np.save(results_path + "/strong/error", error_strong)
        np.save(results_path + "/strong/rates", rates_strong)
        np.save(results_path + "/weak/dI", dI_weak)
        np.save(results_path + "/weak/dE", dE_weak)
        np.save(results_path + "/weak/error", error_weak)
        np.save(results_path + "/weak/rates", rates_weak)
        np.save(results_path + "/diff_strong_weak", diff_strong_weak)
    
        return M_list, diff_strong_weak, dI_strong, dI_weak, dE_strong, dE_weak,\
            error_strong, rates_strong, error_weak, rates_weak
             
def euler_driver(mach_number=0.4, p=2, M=11, L=10.0,
                 p_geo=2, c="c_dg", discretization_type=1, 
                 form="strong", suffix=None, run=True):
    
    if c== "c_dg":
        c_desc = "0"
    elif c == "c_+":
        c_desc = "p"
    else:
        raise ValueError
    
    descriptor = "m" + "{:.1f}".format(mach_number).replace(".","") + "p" + str(p) \
       + "c"  + c_desc + "t" + str(discretization_type) + "_" + form 
       
    if suffix is not None:
        descriptor = descriptor + suffix
    
    project_title = "euler_" + descriptor
    # GHOST - Euler Test (2D)
    print("running solver", project_title)
    
    # read in mesh in GMSH format (here fastest to just generate)
    points, elements = meshzoo.rectangle(
            xmin=0.0, xmax=L,
            ymin=0.0, ymax=L,
            nx=M, ny=M)
        
    if not os.path.exists("../mesh/" +  project_title + "/"):
        os.makedirs("../mesh/" +  project_title + "/")
    
    meshio.write("../mesh/" + project_title + "/M" + str(M) + ".msh",
                 meshio.Mesh(points, {"triangle": elements}))
    
    mesh = Mesh2D(project_title + "_M" + str(M),
                  "../mesh/" + project_title + "/M" + str(M) + ".msh")
    
    # set up periodic boundary conditions
    left = np.array([1.0,0.0,0.0]) 
    right = np.array([1.0,0.0,10.0])
    bottom = np.array([0.0,1.0,0.0])
    top = np.array([0.0,1.0,10.0])
    mesh.add_bc_on_hyperplanes([left,right,bottom,top],[1,2,3,4])
    mesh.make_periodic((1,2),[1]) # left-right periodic (bcs parallel to axis 1)
    mesh.make_periodic((3,4),[0]) # top-bottom periodic (axis 0)
    
    #curvilinear transformation used in Del Rey Fernandez et al. (2017)
    mesh.map_mesh(f_map=Mesh2D.grid_transformation(warp_factor=0.2, L=10.0),
                  p_geo=p_geo)
    
    
    # solver parameters
    params = {"project_title": project_title,
             "problem": "compressible_euler",
             "specific_heat_ratio": 1.4,
             "numerical_flux": "roe",
             "initial_condition": "isentropic_vortex",
             "vortex_type": "spiegel",
             "initial_vortex_centre": np.array([0.5*L,0.5*L]),
             "mach_number": mach_number,
             "angle": np.pi/4.,
             "form": form,
             "correction": c,
             "solution_degree": p,
             "time_integrator": "rk44",
             "final_time": L/(mach_number/np.sqrt(2)),
             "time_step_scale": 0.005}
    
    if discretization_type == 1:
         params["facet_rule"] = "lg"
         params["integration_type"] = "quadrature"
         params["volume_quadrature_degree"] = 2*p
         params["facet_quadrature_degree"] = 2*p+1
         params["solution_representation"] = "modal"
    
    elif discretization_type == 2:
         params["integration_type"] = "collocation"
         params["volume_collocation_degree"] = p
         params["facet_collocation_degree"] = p
         params["solution_representation"] = "nodal"
    
    elif discretization_type == 3:
         params["facet_rule"] = "lgl"
         params["integration_type"] = "quadrature"
         params["volume_quadrature_degree"] = 2*p
         params["facet_quadrature_degree"] = 2*p-1
         params["solution_representation"] = "modal"
    
    else:
        raise ValueError
    
    solver = Solver(params,mesh)
    
    if run:
        solver.run(write_interval=params["final_time"]/10, 
                    print_interval=params["final_time"]/1000)
        
        solver.post_process(error_quadrature_degree=4*p)
        l2_error = solver.calculate_error() 
        for e in range(0,4):
                  print("{:.3e}".format((solver.I_f - solver.I_0)[e]), "& ", 
                  "{:.3e}".format(l2_error[e]), " \\\\")
                  
        
        pickle.dump(solver.I_f - solver.I_0, open("../results/"+project_title+"/conservation_error.dat", "wb" ))
        pickle.dump(l2_error, open("../results/"+project_title+"/solution_error.dat", "wb" ))
        
    return solver

    