# Tristan Montoya - Driver for the 1D Euler Equations

import numpy as np
from analyticalSolution import *
import matplotlib.pyplot as plt
from problem import *
from spatialDiscretization import *

def quasi1D_driver(res, S_star, p_01, T_01, gamma, R, figtitle, x_shock=100):

    #solve analytically first
    x = np.linspace(0., 10., res)
    Ma, T, p, Q1, Q2, Q3 = quasi1D(x, S_star, p_01, T_01, gamma, R, x_shock)

    #extract BCs
    p_exit = p[res-1]
    rho_inlet = Q1[res-1]/sectionCalc(0.)
    rhou_inlet = Q2[res-1]/sectionCalc(0.)

    #set up problem physics
    q1D = Problem(problemType=0, L=10., gamma=gamma, R=R)
    q1D.setSubsonicInlet(rho_inlet, rhou_inlet)
    q1D.setSubsonicExit(p_exit)

    #test fluxes at outlet
    Q_j = np.array([Q1[res-1], Q2[res-1], Q3[res-1]])
    print("Testing flux calculation at exit...")
    print("E: ", q1D.E_j(Q_j))
    print("AQ: ", q1D.A_j(Q_j) @ Q_j)

    #set up spatial scheme
    M = 10 #number of interior nodes

    Q_0 = np.ones(30)
    fdScheme = SpatialDiscretization(q1D, M)
    residual = fdScheme.buildFlowResidual(Q_0)
    print("residual: ", residual)

    print(fdScheme.mesh)

    #plots
    mach = plt.figure()
    plt.grid()
    plt.plot(x, Ma, '-r')
    plt.xlim([0,10])
    plt.xlabel("$x$ (m)")
    plt.ylabel("Mach Number")
    plt.show()
    mach.savefig("../plots/mach_"+figtitle+".pdf", bbox_inches='tight')

    pressure = plt.figure()
    plt.grid()
    plt.plot(x, p/1000., '-b')
    plt.xlim([0,10])
    plt.xlabel("$x$ (m)")
    plt.ylabel("Pressure (kPa)")
    plt.ticklabel_format(style='sci', axis='y')
    plt.show()
    pressure.savefig("../plots/pressure_"+figtitle+".pdf", bbox_inches='tight')

    return

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

quasi1D_driver(1000, 0.8, 1.e5, 300., 1.4, 287, "subsonic")
#quasi1D_driver(1000, 1.0, 1.e5, 300., 1.4, 287, "transonic", x_shock=7.)
#shockTube_driver(1000, 5., 1.e5, 1.e4, 1., 0.125, 6.1e-3, 1.4, "shocktube")
