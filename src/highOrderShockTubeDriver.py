
import numpy as np
from analyticalSolution import *
import matplotlib.pyplot as plt
from problem import *
from spatialDiscretization import *
from element import *
from spatialDiscHighOrder import *
from tvdRK import *



def explicitHighOrderShockTubeDriver(label, x_0, p_L, p_R, rho_L, rho_R, t_f, C, gamma, R, elementType, p, gridType, K):
    np.set_printoptions(suppress=True, linewidth=np.nan, threshold=np.nan)

    # define problem
    problem = Problem(problemType=1, L=10., gamma=gamma, R=R, fluxFunction='lf')

    # set initial condtion to inlet
    problem.setShockTubeInitialCondition(p_L, p_R, rho_L, rho_R, x_0)

    # extract boundary conditions from analytical solution, apply for weak enforcement
    Q_in = np.array([rho_L, 0.0, p_L / (gamma - 1.)])
    Q_out = np.array([rho_R, 0.0, p_R / (gamma - 1.)])
    problem.setBCs_allDirichlet(Q_in, Q_out)

    # reference element
    refElement = Element(elementType, p, gridType)
    hoScheme = SpatialDiscHighOrder(problem, refElement, K)

    figtitle = "shocktube_" + label + "_" + elementType + "_"+ gridType + "_p" + str(p) + "_K" + str(K)

    # run iterations and save to file
    timeMarch = explicitSolver(figtitle, hoScheme, C, t_f, method='explicit_euler', ref_u = 300, ref_a = 315)
    timeMarch.runSolver()
    u_f = timeMarch.Q
    return u_f, hoScheme, figtitle

def createPlotsShockTube(figtitle, K, N):
    exact = np.load("../results/shocktube_diag_exact.npy")
    results = np.load("../results/" + figtitle + "_results.npy")

    mach = plt.figure()
    plt.grid()
    plt.plot(exact[0,:], exact[1,:], '-k', label="Exact Solution")
    for i in range(0,K):
        plt.plot(results[9, i*N:(i+1)*N], results[8, i*N:(i+1)*N], '-')
    plt.xlim([0, 10])
    plt.xlabel("$x$ (m)")
    plt.ylabel("Mach Number")
    plt.legend()
    plt.show()
    mach.savefig("../plots/mach_" + figtitle + ".pdf", bbox_inches='tight')

u_f, hoScheme, figtitle = explicitHighOrderShockTubeDriver('test', 5.0, 1.e5, 1.e4, 1., 0.125, 6.1e-3, 0.05, 1.4, 287., 'dg_diag', 2, 'lgl', 100)
createPlotsShockTube('shocktube_test_dg_diag_lgl_p2_K100', 100, 3)