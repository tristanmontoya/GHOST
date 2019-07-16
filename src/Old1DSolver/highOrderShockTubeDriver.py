# Tristan Montoya - Driver for the 1D Euler Shock-Tube Problem

import numpy as np
from analyticalSolution import *
import matplotlib.pyplot as plt
from problem import *
from element import *
from spatialDiscHighOrder import *
from tvdRK import *

def explicitHighOrderShockTubeDriver(label, x_0, p_L, p_R, rho_L, rho_R, t_f, C, gamma, R,
                                     elementType, limiterType, timeMarching, minmodfun, p, gridType, K):
    np.set_printoptions(suppress=True, linewidth=np.nan, threshold=np.nan)

    # define problem
    problem = Problem(problemType=1, L=10., gamma=gamma, R=R, fluxFunction='roe-ef')

    # set initial condtion to inlet
    problem.setShockTubeInitialCondition(p_L, p_R, rho_L, rho_R, x_0)

    # extract boundary conditions from analytical solution, apply for weak enforcement
    Q_in = np.array([rho_L, 0.0, p_L / (gamma - 1.)])
    Q_out = np.array([rho_R, 0.0, p_R / (gamma - 1.)])
    problem.setBCs_allDirichlet(Q_in, Q_out)

    # reference element
    refElement = Element(elementType, p, gridType)
    hoScheme = SpatialDiscHighOrder(problem, refElement, K, limiterType=limiterType, minmodfun=minmodfun)

    figtitle = "shocktube_" + label + "_" + elementType + "_"+ gridType + "_p" \
               + str(p) + "_K" + str(K) + "_" + minmodfun

    # run iterations and save to file
    timeMarch = explicitSolver(figtitle, hoScheme, C, t_f, method=timeMarching, ref_u = 300, ref_a = 315)
    timeMarch.runSolver()
    u_f = timeMarch.Q
    return u_f, hoScheme, figtitle

def createPlotsShockTube(figtitle, K, N):
    exact = np.load("../results/shocktube_diag_exact.npy")
    results = np.load("../results/" + figtitle + "_results.npy")
    isLimited = np.load("../results/" + figtitle + "_isLimited.npy")
    mach = plt.figure()
    plt.grid()
    plt.plot(exact[0,:], exact[1,:], '-k', label="Exact Solution")
    for i in range(0,K):
        x_loc = results[9, i*N:(i+1)*N]
        print('x: ', x_loc, 'y: ',results[8, i*N:(i+1)*N] )
        z = np.polyfit(results[9, i*N:(i+1)*N], results[8, i*N:(i+1)*N], N-1)
        f = np.poly1d(z)

        # calculate new x's and y's
        x_new = np.linspace(x_loc[0],x_loc[-1], 30)
        y_new = f(x_new)
        if isLimited[i] == 1:
            plt.plot(x_new, y_new, '-r')
        else:
            plt.plot(x_new, y_new, '-r')
    plt.xlim([0, 10])
    plt.xlabel("$x$ (m)")
    plt.ylabel("Mach Number")
    plt.legend()
    plt.show()
    mach.savefig("../plots/mach_" + figtitle + ".pdf", bbox_inches='tight')

    rho = plt.figure()
    plt.grid()
    plt.plot(exact[0, :], exact[2, :], '-k', label="Exact Solution")
    for i in range(0, K):
        x_loc = results[9, i * N:(i + 1) * N]
        print('x: ', x_loc, 'y: ', results[3, i * N:(i + 1) * N])
        z = np.polyfit(results[9, i * N:(i + 1) * N], results[3, i * N:(i + 1) * N], N - 1)
        f = np.poly1d(z)

        # calculate new x's and y's
        x_new = np.linspace(x_loc[0], x_loc[-1], 30)
        y_new = f(x_new)

        plt.plot(x_new, y_new, '-', lw = 2.5)
    plt.xlim([6, 9])
    plt.ylim([0.,0.5])
    plt.xlabel("$x$ (m)")
    plt.ylabel("Density")
    plt.legend()
    plt.show()
    rho.savefig("../plots/density_" + figtitle + ".pdf", bbox_inches='tight')

def createSimplePlotsShockTube(figtitle):
    exact = np.load("../results/shocktube_diag_exact.npy")
    results = np.load("../results/" + figtitle + "_results.npy")
    isLimited = np.load("../results/" + figtitle + "_isLimited.npy")
    mach = plt.figure()
    plt.grid()
    plt.plot(exact[0,:], exact[1,:], '-k', label="Exact Solution")
    plt.plot(results[9,:], results[8,:], '-r', label="Third-Order Limited RKDG")
    plt.xlim([0, 10])
    plt.xlabel("$x$ (m)")
    plt.ylabel("Mach Number")
    plt.legend()
    plt.show()
    mach.savefig("../plots/mach_" + figtitle + ".pdf", bbox_inches='tight')

    rho = plt.figure()
    plt.grid()
    plt.plot(exact[0, :], exact[2, :], '-k', label="Exact Solution")
    plt.plot(results[9,:], results[3,:], '-b', label="Third-Order Limited RKDG")
    plt.xlim([0, 10])
    plt.xlabel("$x$ (m)")
    plt.ylabel("Density (kg/m$^3$)")
    plt.legend()
    plt.show()
    rho.savefig("../plots/density_" + figtitle + ".pdf", bbox_inches='tight')

u_f, hoScheme, figtitle = explicitHighOrderShockTubeDriver('final', 5.0, 1.e5, 1.e4, 1., 0.125, 6.1e-3, 0.5, 1.4, 287.,
                                                        'dg_diag', 'cs', 'SSPRK3', 'tvb', 2, 'lgl', 100)
createPlotsShockTube('shocktube_final_dg_diag_lgl_p2_K100_tvb', 100, 3)
createSimplePlotsShockTube('shocktube_final_dg_diag_lgl_p2_K100_tvb')