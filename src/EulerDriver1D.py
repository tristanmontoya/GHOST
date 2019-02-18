# Tristan Montoya - Driver for the 1D Euler Equations

import numpy as np
from analyticalSolution import *
import matplotlib.pyplot as plt
from problem import *
from spatialDiscretization import *
from temporalDiscretization import *

def quasi1D_driver(res, S_star, p_01, T_01, gamma, R, figtitle, x_shock=100, runChecks=False):
    np.set_printoptions(suppress=True,linewidth=np.nan,threshold=np.nan)

    #solve analytically first
    x = np.linspace(0., 10., res)
    Ma, T, p, Q = quasi1D(x, S_star, p_01, T_01, gamma, R, x_shock)
    np.save("../results/"+figtitle+"_exact.npy", np.array([x, Ma, p]))

    #extract initial inlet conditions
    rho_inlet = Q[0]/sectionCalc(0.)
    rhou_inlet = Q[1]/sectionCalc(0.)
    e_inlet = Q[2]/sectionCalc(0.)

    #extract boundary conditions
    Q_in = np.array([Q[0], Q[1], Q[2]])
    Q_out = np.array([Q[res*3-3], Q[res*3-2], Q[res*3-1]])

    #set up problem physics
    q1D = Problem(problemType=0, L=10., gamma=gamma, R=R)
    q1D.setBCs_allDirichlet(Q_in, Q_out)
    q1D.setUinformInitialCondition(rho_inlet, rhou_inlet, e_inlet)

    #set up spatial scheme
    M = res-2 #number of interior nodes

    #set up spatial discretization
    fdScheme = SpatialDiscretization(q1D, M, 0.0, 0.02)
    #fdScheme.setInitialConditionOnMesh(Q[3:(res-1)*3])

    #just run checks, don't actually solve
    if runChecks == True:
        fdScheme.runChecks(fdScheme.Q_0, Q_in, Q_out)
        return

    #run iterations and save to file
    timeMarch = TemporalDiscretization(figtitle, fdScheme, method=0, C=40., isUnsteady= False,
                                       useLocalTimeStep=False, ref_u = 300, ref_a = 315, t_f = 1.0, rel_tol = 1.e-12)
    timeMarch.timeStepMatrix()
    timeMarch.implicitEuler()

def shockTube_driver(res, x_0, p_L, p_R, rho_L, rho_R, t_f, gamma, figtitle):

    x = np.linspace(0., 10., res)
    M, rho, p = shockTube(x, x_0, p_L, p_R, rho_L, rho_R, t_f, gamma)

    mach = plt.figure()
    plt.grid()
    plt.plot(x, M, '-r')
    plt.xlim([0,10])
    plt.ylim([0,1])
    plt.xlabel("$x$ (m)")
    plt.ylabel("Mach Number")
    plt.show()
    mach.savefig("../plots/mach_"+figtitle+".pdf", bbox_inches='tight')

    density = plt.figure()
    plt.grid()
    plt.plot(x, rho, '-g')
    plt.xlim([0,10])
    plt.xlabel("$x$ (m)")
    plt.ylabel("Density (kg/m$^3$)")
    plt.ticklabel_format(style='sci', axis='y')
    plt.show()
    density.savefig("../plots/density_"+figtitle+".pdf", bbox_inches='tight')

    return

def createPlots(figtitle):
    exact = np.load("../results/" + figtitle + "_exact.npy")
    results = np.load("../results/" + figtitle + "_results.npy")
    resHistory = np.load("../results/" + figtitle + "_resHistory.npy")

    # generate plots
    mach = plt.figure()
    plt.grid()
    plt.plot(exact[0,:], exact[1,:], '-r', label="Exact Solution")
    plt.plot(results[9, :], results[8, :], 'xk', label="Numerical Solution")
    plt.xlim([0, 10])
    plt.xlabel("$x$ (m)")
    plt.ylabel("Mach Number")
    plt.legend()
    plt.show()
    mach.savefig("../plots/mach_" + figtitle + ".pdf", bbox_inches='tight')

    pressure = plt.figure()
    plt.grid()
    plt.plot(exact[0,:], exact[2,:]/1000., '-b', label="Exact Solution")
    plt.plot(results[9, :], results[6, :] / 1000., 'xk', label="Numerical Solution")
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
    plt.ylabel("Relative Residual Norm")
    plt.legend()
    plt.show()
    resPlot.savefig("../plots/resHistory_" + figtitle + ".pdf", bbox_inches='tight')


#quasi1D_driver(8, 0.8, 1.e5, 300., 1.4, 287, "subsonic_test1", runChecks=True)

quasi1D_driver(101, 0.8, 1.e5, 300., 1.4, 287, "subsonic", runChecks=False)
createPlots("subsonic")
#quasi1D_driver(1000, 1.0, 1.e5, 300., 1.4, 287, "transonic", x_shock=7.)
#shockTube_driver(1000, 5., 1.e5, 1.e4, 1., 0.125, 6.1e-3, 1.4, "shocktube")
