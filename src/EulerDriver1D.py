# Tristan Montoya - Analytical Solution to the Euler Equations

import numpy as np
from analyticalSolution import *
import matplotlib.pyplot as plt
from problem import *

def quasi1D_driver(res, S_star, p_01, T_01, gamma, R, figtitle, x_shock=100):

    x = np.linspace(0., 10., res)
    M, T, p, Q1, Q2, Q3 = quasi1D(x, S_star, p_01, T_01, gamma, R, x_shock)


    myProblem = Problem(0)
    print("Area(10): ",myProblem.S(10.))
    myProblem.setSubsonicInlet(1.0, 2.0)
    myProblem.setSubsonicExit(1.5)

    Q_j = [Q1[res-1], Q2[res-1], Q3[res-1]]

    w_plus, w_minus, X_inv_plus, X_inv_minus = myProblem.eigsA_j(Q_j)
    X_inv = X_inv_plus + X_inv_minus

    print("Wplus, Wminus: ", w_plus, w_minus)

    print("w1 from pulliam (should be second w here): ", (gamma - 1.)/gamma*Q_j[0])

    print('W', X_inv @ Q_j)
    print("E: ", myProblem.E_j(Q_j))
    print("AQ: ", myProblem.A_j(Q_j) @ Q_j)


    mach = plt.figure()
    plt.grid()
    plt.plot(x, M, '-r')
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
