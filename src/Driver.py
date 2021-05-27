# GHOST - Test Problem Drivers

import os
import pickle
import numpy as np
from Solver import Solver
from Mesh import Mesh2D
from math import floor
import meshzoo
import meshio


def advection_driver(a=np.sqrt(2), theta=np.pi/4, p=2, M=5, L=1.0,
                 p_geo=1, c="c_dg", discretization_type=1, 
                 upwind_parameter = 0.0,
                 form="strong", suffix=None, run=True, new_mesh=True):
    
    if c== "c_dg":
        c_desc = "0"
    elif c == "c_+":
        c_desc = "p"
    else:
        raise ValueError
    
    descriptor = "p" + str(p) + "b" +str(int(round(upwind_parameter))) \
    + "c"  + c_desc + "t" + str(discretization_type) + "_" + form 
       
    if suffix is not None:
        descriptor = descriptor + suffix
    
    project_title = "advection_" + descriptor
    print(project_title)
   
    if new_mesh:
        points, elements = meshzoo.rectangle_tri((0.0,0.0),(L,L), n=M+1, 
                    variant="zigzag")
    
    
        if not os.path.exists("../mesh/" +  project_title + "/"):
            os.makedirs("../mesh/" +  project_title + "/")
        
        meshio.write("../mesh/" + project_title + "/M" + str(M) + ".msh",
                     meshio.Mesh(points, {"triangle": elements}))
        
    mesh = Mesh2D(project_title + "_M" + str(M),
                  "../mesh/" + project_title + "/M" + str(M) + ".msh")
    
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
    
    # solver parameters
    params = {"project_title": project_title,
             "problem": "constant_advection",
             "initial_condition": "sine",
             "wavelength": np.ones(2),
             "wave_speed": a*np.array([np.sin(theta),np.cos(theta)]),
             "upwind_parameter": upwind_parameter,
             "form": form,
             "correction": c,
             "solution_degree": p,
             "time_integrator": "rk44",
             "final_time": L/(a*np.cos(theta)),
             "time_step_scale": 0.0025}
    
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
    
    solver = Solver(params,mesh,L/M)
    
    if run:
        solver.run(write_interval=params["final_time"]/10, 
                    print_interval=params["final_time"]/1000)
        
        solver.post_process(error_quadrature_degree=4*p)
        l2_error = solver.calculate_error() 
        print("{:.3e}".format((solver.I_f[0] - solver.I_0[0])), "& ",
              "{:.3e}".format((solver.E_f[0] - solver.E_0[0])), "& "
              "{:.3e}".format(l2_error[0]), " \\\\")
                  
        
        pickle.dump(solver.I_f - solver.I_0, open("../results/"+project_title+"/conservation_error.dat", "wb" ))
        pickle.dump(solver.E_f - solver.E_0, open("../results/"+project_title+"/energy_error.dat", "wb" ))
        pickle.dump(l2_error, open("../results/"+project_title+"/solution_error.dat", "wb" ))
        
    return solver
          

def euler_driver(mach_number=0.4, theta=np.pi/4, p=2, M=10, L=10.0,
                 p_geo=2, c="c_dg", discretization_type=1, 
                 form="strong", suffix=None, run=True, new_mesh=True):
    
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
    print(project_title)
    
    if new_mesh:
        points, elements = meshzoo.rectangle_tri((0.0,0.0),(L,L), n=M+1, 
                    variant="zigzag")
    
    
        if not os.path.exists("../mesh/" +  project_title + "/"):
            os.makedirs("../mesh/" +  project_title + "/")
        
        meshio.write("../mesh/" + project_title + "/M" + str(M) + ".msh",
                     meshio.Mesh(points, {"triangle": elements}))
        
    mesh = Mesh2D(project_title + "_M" + str(M),
                  "../mesh/" + project_title + "/M" + str(M) + ".msh")
    
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
    
    
    # solver parameters
    params = {"project_title": project_title,
             "problem": "compressible_euler",
             "specific_heat_ratio": 1.4,
             "numerical_flux": "roe",
             "initial_condition": "isentropic_vortex",
             "vortex_type": "spiegel",
             "initial_vortex_centre": np.array([0.5*L,0.5*L]),
             "mach_number": mach_number,
             "angle": theta,
             "form": form,
             "correction": c,
             "solution_degree": p,
             "time_integrator": "rk44",
             "final_time": L/(mach_number/np.sqrt(2)),
             "time_step_scale": 0.0025}
    
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
    
    solver = Solver(params,mesh,L/M)
    
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


def write_output_advection(a=np.sqrt(2), p=2, p_geo=1, M=8, L=1.0, correction_type="c_dg",
                             upwind_parameter = 1, discretization_type=1, 
                                headings_disc=False, headings_corr=False):
    
    discretization_text = {1: "Quadrature I", 2: "Collocation", 3: "Quadrature II"}
    correction_text = {"c_dg": "$c_{\\mathrm{DG}}$", "c_+": "$c_+$"}

    strong = advection_driver(a=a, p=p, M=M, L=L,
                                p_geo=p_geo, c=correction_type, 
                                discretization_type=discretization_type,
                                upwind_parameter=float(upwind_parameter),
                                form="strong", run=False, new_mesh=False)
    weak = advection_driver(a=np.sqrt(2), 
                            p=p, M=M, L=L,
                            p_geo=p_geo, c=correction_type, 
                            discretization_type=discretization_type,
                            upwind_parameter=float(upwind_parameter),
                            form="weak", run=False, new_mesh=False)
    
    strong.load_solution(time_step=floor(strong.T/strong.time_integrator.dt_target))
    strong.post_process(error_quadrature_degree=4*p)
    weak.load_solution(time_step=floor(weak.T/weak.time_integrator.dt_target))
    weak.post_process(error_quadrature_degree=4*p)

    diff = strong.calculate_difference(weak)
    strong_cons_error = pickle.load(open("../results/" + strong.params["project_title"] + "/conservation_error.dat", "rb"))
    weak_cons_error = pickle.load(open("../results/" + weak.params["project_title"] + "/conservation_error.dat", "rb"))
    strong_energy_error = pickle.load(open("../results/" + strong.params["project_title"] + "/energy_error.dat", "rb"))
    weak_energy_error = pickle.load(open("../results/" + weak.params["project_title"] + "/energy_error.dat", "rb"))   
 #   strong_sol_error = pickle.load(open("../results/" + strong.params["project_title"] + "/solution_error.dat", "rb"))
 #   weak_sol_error = pickle.load(open("../results/" + weak.params["project_title"] + "/solution_error.dat", "rb"))
    
    line = ""
    if headings_disc:
        line = line + discretization_text[discretization_type]
    line = line + " & "
    
    if headings_corr:
        line = line + correction_text[correction_type]
    
    line = (line + " & " + str(upwind_parameter) + " & " 
        + "{:.3e}".format(diff[0]) + " & " 
        + "{:.3e}".format(strong_cons_error[0]) + " & "
        + "{:.3e}".format(weak_cons_error[0]) + " & "
        + "{:.3e}".format(strong_energy_error[0]) + " & "
        + "{:.3e}".format(weak_energy_error[0]) + "\\\\ \n")
    
    return line

def write_output_euler(mach_number=0.4, theta=np.pi/4, p=2, p_geo=2, M=8, L=1.0, correction_type="c_dg",
                             upwind_parameter = 1, discretization_type=1, 
                                headings_disc=False, headings_corr=False):
    
    discretization_text = {1: "Quadrature I", 2: "Collocation", 3: "Quadrature II"}
    correction_text = {"c_dg": "$c_{\\mathrm{DG}}$", "c_+": "$c_+$"}
    equation_text = {0: "$\\rho$", 1: "$\\rho V_1$", 2: "$\\rho V_2$", 3: "$E$" }

    strong = euler_driver(mach_number=mach_number, 
                            p=p, M=M, L=L,
                            p_geo=p_geo, c=correction_type, 
                            discretization_type=discretization_type,
                            form="strong", run=False, new_mesh=False)
    weak = euler_driver(mach_number=mach_number, 
                            p=p, M=M, L=L,
                            p_geo=p_geo, c=correction_type, 
                            discretization_type=discretization_type,
                            form="weak", run=False, new_mesh=False)
    
    strong.load_solution(time_step=floor(strong.T/strong.time_integrator.dt_target))
    strong.post_process(error_quadrature_degree=4*p)
    weak.load_solution(time_step=floor(weak.T/weak.time_integrator.dt_target))
    weak.post_process(error_quadrature_degree=4*p)

    diff = strong.calculate_difference(weak)
    strong_cons_error = pickle.load(open("../results/" + strong.params["project_title"] + "/conservation_error.dat", "rb"))
    weak_cons_error = pickle.load(open("../results/" + weak.params["project_title"] + "/conservation_error.dat", "rb"))
 #   strong_sol_error = pickle.load(open("../results/" + strong.params["project_title"] + "/solution_error.dat", "rb"))
 #   weak_sol_error = pickle.load(open("../results/" + weak.params["project_title"] + "/solution_error.dat", "rb"))
    
    line = ""
    if headings_disc:
        line = line + discretization_text[discretization_type]
    line = line + " & "
    
    if headings_corr:
        line = line + correction_text[correction_type]
        
    line = (line + " & " + equation_text[0] + " & " 
        + "{:.3e}".format(diff[0]) + " & " 
        + "{:.3e}".format(strong_cons_error[0]) + " & "
        + "{:.3e}".format(weak_cons_error[0]) + "\\\\ \n")
    
    for e in range(1,4):
        line = (line + " & & " + equation_text[e] + " & " 
            + "{:.3e}".format(diff[e]) + " & " 
            + "{:.3e}".format(strong_cons_error[e]) + " & "
            + "{:.3e}".format(weak_cons_error[e]) + "\\\\ \n")
    
    return line
