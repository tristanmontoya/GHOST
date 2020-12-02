# GHOST - Drivers

import numpy as np
import Solver, Mesh


def grid_refine(problem_title="test",
                d=1,
                n_refine=4,
                p=2,
                problem="constant_advection",
                integration_type="quadrature",
                solution_representation="modal",
                map_type="affine",
                form="weak",
                correction_type="dg",
                results_path="../results/"):
    
    # simulation parameters dictionary
    params = {}
    params["problem_title"] = problem_title
    params["problem"] = problem
    params["integration_type"] = integration_type
    params["solution_representation"] = solution_representation
    params["solution_degree"] = p
    params["form"] = form
    params["correction_type"] = correction_type
    params["time_integrator"] = "rk44"
    params["time_step_scale"] = 0.1
    
    if map_type == "affine":
        p_geo = 1
    elif map_type == "isoparametric":
        p_geo = p
        
    if d == 1:
        grid_refine_1d(params,n_refine,p_geo)
        
    else:
        raise NotImplementedError


def grid_refine_1d(params,n_refine,p_geo):

    #start with 4 elements 
    M = 5
    
    # make array of solvers
    solvers = []
    
    if params["integration_type"] == "quadrature":
        params["volume_quadrature_degree"] = 2*params["solution_degree"] + 1
        
    if params["problem"] == "constant_advection":
        params["initial_condition"] = "sine"
        params["wavelength"] = np.array([1.0])
        params["wave_speed"] = np.ones(1)
        params["final_time"] = 1.0
    
    for n in range(0, n_refine):
        
        mesh = Mesh.Mesh1D(params["problem_title_M"] + str(M), 0.0, 1.0, M)
        left = [1.0, 0.0]
        right = [1.0, 1.0]
        mesh.add_bc_on_hyperplanes([left,right],[1,2])
        mesh.make_periodic([1,2])
        
        mesh.map_mesh(f_map=Mesh.Mesh1D.grid_transformation(
            warp_factor=0.2), p_geo=p_geo)
        
        solvers.append[Solver.Solver(params, mesh)]
        
        M = M*2
    
    
    
        
        
        