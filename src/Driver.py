# GHOST - Test Problem Drivers

import os
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
                form="weak",
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
    params["form"] = form
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
        return grid_refine_1d(params,n_refine,p_geo, 21, results_path)
    elif d == 2:
        return grid_refine_2d(params,n_refine,p_geo, 21, results_path)
    else:
        raise NotImplementedError


def grid_refine_1d(params, n_refine, p_geo, error_quad_degree, results_path):

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
        
    M_list = np.zeros(n_refine)
    error = np.zeros((N_eq,n_refine))
    rates = np.zeros((N_eq,n_refine))
    
    for n in range(0, n_refine):
        
        mesh = Mesh1D(params["project_title"] + "_M" + str(M),
                           0.0, 1.0, M)
        left = [1.0, 0.0]
        right = [1.0, 1.0]
        mesh.add_bc_on_hyperplanes([left,right],[1,2])
        mesh.make_periodic([1,2])
        
        mesh.map_mesh(f_map=Mesh1D.grid_transformation(
            warp_factor=0.2), p_geo=p_geo)
        
        solver = Solver(params, mesh)
        solver.run(results_path + "M" + str(M) + "/")
        solver.post_process()
        
        # update results array
        M_list[n] = M
        error[:,n] = solver.calculate_error(norm="L2")
        
        if n == 0:
            rates[:,n] = np.zeros(N_eq)
        else:
            rates[:,n] = np.array([np.polyfit(np.log(1.0/M_list[n-1:n+1]),
                                            np.log(error[e,n-1:n+1]), 1)[0] 
                                   for e in range(0,N_eq)])
            
        print("M: ", M, " errors: ", error[:,n], " rates: ", rates[:,n])
        
        M = M*2
 
    np.save(results_path + "M_list", M_list)
    np.save(results_path + "error", error)
    np.save(results_path + "rates", rates)
       
    return M_list, error, rates
    

def grid_refine_2d(params, n_refine, p_geo, error_quad_degree, results_path):

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
        
    M_list = np.zeros(n_refine)
    error = np.zeros((N_eq,n_refine))
    rates = np.zeros((N_eq,n_refine))
    
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
        
        solver = Solver(params, mesh)
        solver.run(results_path + "M" + str(M) + "/")
        solver.post_process()
        
        # update results array
        M_list[n] = M
        error[:,n] = solver.calculate_error(norm="L2")
        
        if n == 0:
            rates[:,n] = np.zeros(N_eq)
        else:
            rates[:,n] = np.array([np.polyfit(np.log(1.0/M_list[n-1:n+1]),
                                            np.log(error[e,n-1:n+1]), 1)[0] 
                                   for e in range(0,N_eq)])
            
        print("M: ", M, " errors: ", error[:,n], " rates: ", rates[:,n])
        M = M*2
 
    np.save(results_path + "M_list", M_list)
    np.save(results_path + "error", error)
    np.save(results_path + "rates", rates)
       
    return M_list, error, rates
