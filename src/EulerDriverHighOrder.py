# Tristan Montoya - Driver for the 1D Euler Equations

import numpy as np
from analyticalSolution import *
import matplotlib.pyplot as plt
from problem import *
from spatialDiscretization import *
from element import *
from spatialDiscHighOrder import *
from implicitSolver import *

def implicitHighOrderQuasi1DDriver(label, S_star, p_01, T_01, gamma, R, elementType, p, gridType, K):
    np.set_printoptions(suppress=True, linewidth=np.nan, threshold=np.nan)

    # define problem
    q1D = Problem(problemType=0, L=10., gamma=gamma, R=R)

    # get analytical bcs
    x = np.array([0,10.0])
    Mab, Tb, presb, Qb = quasi1D(x, S_star, p_01, T_01, gamma, R)

    # testing eigenvalues and roe flux
    # X, Xinv, Lambda = q1D.eigsA_j(Q[0:3], 0.)
    # Xroe, Xinvroe, Lambdaroe = q1D.eigsA_roe(Q[0:3], Q[0:3], 0.0)
    # print("lambda= ", Lambda)
    # print("A(Q_L) = ", q1D.A_j(Q[0:3]))
    # print("A(Q_L), eigs = ", X @ Lambda @ Xinv)
    # print("eigsA", np.linalg.eig(q1D.A_j(Q[0:3])))
    # print("F(Q_L), AQ_L:", q1D.E_j(Q[0:3]), q1D.A_j(Q[0:3]) @ Q[0:3])
    # print("A(Q_L),roe = ", Xroe @ Lambdaroe @ Xinvroe)
    # print("Froe(Q_L, Q_L) = ", q1D.numericalFlux(Q[0:3], Q[0:3], 0.0))

    # set initial condtion to inlet
    rho_inlet = Qb[0] / sectionCalc(0.)
    rhou_inlet = Qb[1] / sectionCalc(0.)
    e_inlet = Qb[2] / sectionCalc(0.)
    q1D.setUinformInitialCondition(rho_inlet, rhou_inlet, e_inlet)

    # extract boundary conditions from analytical solution, apply for weak enforcement
    Q_in = np.array([Qb[0], Qb[1], Qb[2]])
    Q_out = np.array([Qb[3], Qb[4], Qb[5]])

    print(Qb[0], Qb[3])
    q1D.setBCs_allDirichlet(Q_in, Q_out)

    # reference element
    refElement = Element(elementType, p, gridType)
    hoScheme = SpatialDiscHighOrder(q1D, refElement, K)

    # Ma, T, pres, Q = quasi1D(hoScheme.mesh, S_star, p_01, T_01, gamma, R)
    # np.save("../results/" + figtitle + "_exact.npy", np.array([hoScheme.mesh, Ma, pres, Q]))

    #hoScheme.u_0_interp = Q

    figtitle = "q1d_subsonic_" + label + "_" + elementType + "_"+ gridType + "_p" + str(p) + "_K" + str(K)

    # run iterations and save to file
    timeMarch = implicitSolver(figtitle, hoScheme, method=0, C=100, isUnsteady=False,
                               useLocalTimeStep=True, max_its=1000000, rel_tol=1.e-12)
    res = timeMarch.runSolver()
    u_f = timeMarch.Q
    return res, u_f, hoScheme, figtitle

def implicitHighOrderQuasi1D_element_refinement(label, S_star, p_01, T_01, gamma, R, elementType, p, gridType, K_0, n_grids):
    np.set_printoptions(suppress=True, linewidth=np.nan, threshold=np.nan)

    # define problem
    q1D = Problem(problemType=0, L=10., gamma=gamma, R=R)

    # get analytical bcs
    x = np.array([0,10.0])
    Mab, Tb, presb, Qb = quasi1D(x, S_star, p_01, T_01, gamma, R)

    # set initial condtion to inlet
    rho_inlet = Qb[0] / sectionCalc(0.)
    rhou_inlet = Qb[1] / sectionCalc(0.)
    e_inlet = Qb[2] / sectionCalc(0.)
    q1D.setUinformInitialCondition(rho_inlet, rhou_inlet, e_inlet)

    # extract boundary conditions from analytical solution, apply for weak enforcement
    Q_in = np.array([Qb[0], Qb[1], Qb[2]])
    Q_out = np.array([Qb[3], Qb[4], Qb[5]])

    q1D.setBCs_allDirichlet(Q_in, Q_out)

    # reference element
    refElement = Element(elementType, p, gridType)

    DOF = np.zeros(n_grids)
    errornorms = np.zeros(n_grids)
    K = K_0
    for i in range(0,n_grids):
        hoScheme = SpatialDiscHighOrder(q1D, refElement, K)
        DOF[i] = hoScheme.M
        Ma, T, pres, u_exact = quasi1D(hoScheme.mesh, S_star, p_01, T_01, gamma, R)
        figtitle = "q1d_subsonic_" + label + "_" + elementType + "_"+ gridType + "_p" + str(p) + "_K" + str(K)

        # run iterations and save to file
        timeMarch = implicitSolver(figtitle, hoScheme, method=0, C=50, isUnsteady=False,
                                   useLocalTimeStep=True, max_its=1000000, rel_tol=1.e-12)
        res = timeMarch.runSolver()
        u_f = timeMarch.Q
        error = hoScheme.calculateError(u_f[0::3], u_exact[0::3])
        print("Grid Level: ", i, " K: ", K, " DOF:", DOF)
        print("Error norm: ", error)
        errornorms[i] = error
        K = K*2

    reftitle = "figtitle" + "_elem_refine.npy"
    np.save("../results/"+reftitle, np.array([DOF, errornorms]))
    return DOF, errornorms, reftitle


def createPlotsQuasi1D(figtitle):
    exact = np.load("../results/subsonic_block_M199_C40_k20_k4_0_02_exact.npy")
    results = np.load("../results/" + figtitle + "_results.npy")
    resHistory = np.load("../results/" + figtitle + "_resHistory.npy")

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
    plt.semilogy(resHistory[0, :], resHistory[1, :]/resHistory[1, 0], '-k')
    plt.xlabel("Iteration")
    plt.xlim(xmin = 0, xmax = resHistory[0, -1])
    plt.ylim(1.e-12, 1.e0)
    plt.ylabel("Relative Residual Norm")
    plt.legend()
    plt.show()
    resPlot.savefig("../plots/resHistory_" + figtitle + ".pdf", bbox_inches='tight')

def createResPlots(names, labels,max_its):
    n_plots = len(names)

    resPlot = plt.figure()
    plt.grid()
    plt.xlabel("Iteration")
    plt.xlim(xmin = 0, xmax = max_its)
    plt.ylim(1.e-12, 1.e0)
    plt.ylabel("Relative Residual Norm (Mass)")
    nametotal = ""

    for i in range(0, n_plots):
        resHistory = np.load("../results/" + names[i] + "_resHistory.npy")
        plt.semilogy(resHistory[0, :], resHistory[1, :]/resHistory[1, 0], '-', label=labels[i])
        nametotal = nametotal + "_" + names[i]

    plt.legend()
    plt.show()
    resPlot.savefig("../plots/resHistory_" + nametotal + ".pdf", bbox_inches='tight')

def gridConvPlot(names, labels):
    n_plots = len(names)

    resPlot = plt.figure()
    plt.grid()
    plt.xlabel("DOF")
    plt.ylabel("$L^2 (\Omega)$ Error in $\mathcal{U}_1$")
    nametotal = ""

    for i in range(0, n_plots):
        results = np.load("../results/" + names[i])
        plt.loglog(results[0, :], results[1, :], '-x', label=labels[i])
        nametotal = nametotal + "_" + names[i]

    plt.legend()
    plt.show()
    resPlot.savefig("../plots/gridconv_" + nametotal + ".pdf", bbox_inches='tight')


# R, u_f, hoScheme, title = implicitHighOrderQuasi1DDriver("test", 0.8, 1.e5, 300., 1.4, 287, "dg_dense", 5, "lg", 2)
# print(title)
# createPlotsQuasi1D(title)
# R_final = hoScheme.localResidualExplicitForm(u_f,0)

DOF, errornorms, title = implicitHighOrderQuasi1D_element_refinement("test", 0.8, 1.e5, 300., 1.4, 287, "dg_dense", 2, "lg", 2, 6)
gridConvPlot([title], ["Diagonal DG on LG"])

# u = ho.u_0_interp
# R_mat = ho.localResidualInterior(u, 2)
# R_exp = ho.localResidualExplicitForm(u, 2)
# jac = dRdQ.toarray()
# print("R1_Matrix: ", R_mat)
# print("Explicit Form: ", R_exp)
# print("Full residual: ", np.reshape(ho.flowResidual(u),[ho.K,ho.Np*ho.n_eq]))