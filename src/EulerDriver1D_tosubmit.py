# Tristan Montoya - Driver for the 1D Euler Equations

import numpy as np
from analyticalSolution import *
import matplotlib.pyplot as plt
from problem import *
from spatialDiscretization import *
from temporalDiscretization import *

def quasi1D_driver(M, C, k_2, k_4, S_star, p_01, T_01, gamma, R, figtitle, x_shock=100, runChecks=False, useDiagonalForm=False):
    np.set_printoptions(suppress=True,linewidth=np.nan,threshold=np.nan)
    res = 1000

    #define problem
    q1D = Problem(problemType=0, L=10., gamma=gamma, R=R)

    #solve analytically first
    x = np.linspace(0., 10., res)
    Ma, T, p, Q = quasi1D(x, S_star, p_01, T_01, gamma, R, x_shock)
    np.save("../results/"+figtitle+"_exact.npy", np.array([x, Ma, p]))

    #set initial condtion to inlet
    rho_inlet = Q[0]/sectionCalc(0.)
    rhou_inlet = Q[1]/sectionCalc(0.)
    e_inlet = Q[2]/sectionCalc(0.)
    q1D.setUinformInitialCondition(rho_inlet, rhou_inlet, e_inlet)

    #extract boundary conditions from analytical solution, apply as Riemann invariants for subsonic inlet and exit
    Q_in = np.array([Q[0], Q[1], Q[2]])
    Q_out = np.array([Q[res*3-3], Q[res*3-2], Q[res*3-1]])

    R_in = q1D.flowVariablesToRiemann(Q_in, 0.)
    R_out = q1D.flowVariablesToRiemann(Q_out, 10.)
    q1D.setBCs_subsonicRiemann(R_in[0], R_in[2], R_out[1])

    #set up spatial discretization
    fdScheme = SpatialDiscretization(q1D, M, k_2, k_4)
    #fdScheme.setInitialConditionOnMesh(Q[3:(res-1)*3])

    #if option selected, just run checks, don't actually solve
    if runChecks == True:
        fdScheme.runChecks(fdScheme.Q_0, Q_in, Q_out)
        return

    #run iterations and save to file
    timeMarch = TemporalDiscretization(figtitle, fdScheme, method=0, C=C, isUnsteady= False,
                                       useLocalTimeStep=True, ref_u = 300, ref_a = 315, t_f = 1.0, rel_tol = 1.e-13,
                                       useDiagonalForm=useDiagonalForm)
    timeMarch.runSolver()

def shockTube_driver(M, C, k_2, k_4, x_0, p_L, p_R, rho_L, rho_R, t_f, gamma, R, figtitle, useDiagonalForm=False, method=1):
    np.set_printoptions(suppress=True,linewidth=np.nan,threshold=np.nan)
    res = 1000

    #define problem
    shockTubeProblem = Problem(problemType=1, L=10., gamma=gamma, R=R)

    #solve analytically first
    x = np.linspace(0., 10., res)
    Ma, rho, p = shockTube(x, x_0, p_L, p_R, rho_L, rho_R, t_f, gamma)
    np.save("../results/"+figtitle+"_exact.npy", np.array([x, Ma, rho]))

    #set initial condtion
    shockTubeProblem.setShockTubeInitialCondition(p_L, p_R, rho_L, rho_R, x_0)

    #set boundary condition
    shockTubeProblem.setBCs_allDirichlet(shockTubeProblem.shockTubeInitialCondition(0.),
                                         shockTubeProblem.shockTubeInitialCondition(10.))

    #set up spatial discretization
    fdScheme = SpatialDiscretization(shockTubeProblem, M, k_2, k_4)

    #run time marching and save to file
    timeMarch = TemporalDiscretization(figtitle, fdScheme, method=method, C=C, isUnsteady= True, ref_u = 300,
                                       ref_a = 315, t_f = t_f, useDiagonalForm=useDiagonalForm)
    timeMarch.runSolver()

def createPlotsQuasi1D(figtitle):
    exact = np.load("../results/" + figtitle + "_exact.npy")
    results = np.load("../results/" + figtitle + "_results.npy")
    resHistory = np.load("../results/" + figtitle + "_resHistory.npy")

    # generate plots
    mach = plt.figure()
    plt.grid()
    plt.plot(exact[0,:], exact[1,:], '-k', label="Exact Solution")
    plt.plot(results[9, :], results[8, :], 'xr', label="Numerical Solution")
    plt.xlim([0, 10])
    plt.xlabel("$x$ (m)")
    plt.ylabel("Mach Number")
    plt.legend()
    plt.show()
    mach.savefig("../plots/mach_" + figtitle + ".pdf", bbox_inches='tight')

    pressure = plt.figure()
    plt.grid()
    plt.plot(exact[0,:], exact[2,:]/1000., '-k', label="Exact Solution")
    plt.plot(results[9, :], results[6, :] / 1000., 'xr', label="Numerical Solution")
    plt.xlim([0, 10])
    plt.xlabel("$x$ (m)")
    plt.ylabel("Pressure (kPa)")
    plt.ticklabel_format(style='sci', axis='y')
    plt.legend()
    plt.show()
    pressure.savefig("../plots/pressure_" + figtitle + ".pdf", bbox_inches='tight')

    resPlot = plt.figure()
    plt.grid()
    plt.semilogy(resHistory[0, :], resHistory[1, :]/resHistory[1, 0], '-k', label="Block Form")
    plt.xlabel("Iteration")
    plt.xlim(xmin = 0)
    plt.ylim(1.e-12, 1.e0)
    plt.ylabel("Relative Residual Norm")
    plt.legend()
    plt.show()
    resPlot.savefig("../plots/resHistory_" + figtitle + ".pdf", bbox_inches='tight')

    epsilon2 = plt.figure()
    plt.grid()
    plt.plot(results[9, :], results[10, :], '-g', label="Epsilon 2")
    plt.xlim([0, 10])
    plt.xlabel("$x$ (m)")
    plt.ylabel("Dissipation Coefficient (Second-Difference)")
    plt.ticklabel_format(style='sci', axis='y')
    plt.legend()
    plt.show()
    epsilon2.savefig("../plots/epsilon2_" + figtitle + ".pdf", bbox_inches='tight')

    epsilon4 = plt.figure()
    plt.grid()
    plt.plot(results[9, :], results[11, :], '-g', label="Epsilon 4")
    plt.xlim([0, 10])
    plt.xlabel("$x$ (m)")
    plt.ylabel("Dissipation Coefficient (Fourth-Difference)")
    plt.ticklabel_format(style='sci', axis='y')
    plt.legend()
    plt.show()
    epsilon4.savefig("../plots/epsilon4_" + figtitle + ".pdf", bbox_inches='tight')

def createPlotsShockTube(figtitle):
    exact = np.load("../results/" + figtitle + "_exact.npy")
    results = np.load("../results/" + figtitle + "_results.npy")

    mach = plt.figure()
    plt.grid()
    plt.plot(exact[0,:], exact[1,:], '-k', label="Exact Solution")
    plt.plot(results[9, :], results[8, :], 'xr', label="Numerical Solution")
    plt.xlim([0, 10])
    plt.xlabel("$x$ (m)")
    plt.ylabel("Mach Number")
    plt.legend()
    plt.show()
    mach.savefig("../plots/mach_" + figtitle + ".pdf", bbox_inches='tight')

    density = plt.figure()
    plt.grid()
    plt.plot(exact[0,:], exact[2,:], '-k', label="Exact Solution")
    plt.plot(results[9, :], results[3, :], 'xr', label="Numerical Solution")
    plt.xlim([0, 10])
    plt.xlabel("$x$ (m)")
    plt.ylabel("Density")
    plt.legend()
    plt.show()
    density.savefig("../plots/density_" + figtitle + ".pdf", bbox_inches='tight')

    epsilon2 = plt.figure()
    plt.grid()
    plt.plot(results[9, :], results[10, :], '-g', label="Epsilon 2")
    plt.xlim([0, 10])
    plt.xlabel("$x$ (m)")
    plt.ylabel("Dissipation Coefficient (Second-Difference)")
    plt.ticklabel_format(style='sci', axis='y')
    plt.legend()
    plt.show()
    epsilon2.savefig("../plots/epsilon2_" + figtitle + ".pdf", bbox_inches='tight')

    epsilon4 = plt.figure()
    plt.grid()
    plt.plot(results[9, :], results[11, :], '-g', label="Epsilon 4")
    plt.xlim([0, 10])
    plt.xlabel("$x$ (m)")
    plt.ylabel("Dissipation Coefficient (Fourth-Difference)")
    plt.ticklabel_format(style='sci', axis='y')
    plt.legend()
    plt.show()
    epsilon4.savefig("../plots/epsilon4_" + figtitle + ".pdf", bbox_inches='tight')

#example usage of quasi1D_driver and shockTube_driver
quasi1D_driver(99, 120., 0.05, 0.02, 1.0, 1.e5, 300., 1.4, 287, "transonic_block_M99_C120_k2_0_05_k4_0_0_2", x_shock=7., useDiagonalForm=True)
createPlotsQuasi1D("transonic_diag_M99_C120_k2_0_05_k4_0_0_2")

# shockTube_driver(399, 1., 0.5, 0.02, 5.0, 1.e5, 1.e4, 1., 0.125, 6.1e-3, 1.4, 287., "shocktube_block_ie", useDiagonalForm=False, method=0)
# createPlotsShockTube("shocktube_block_ie")
