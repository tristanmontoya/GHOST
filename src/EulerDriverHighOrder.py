# Tristan Montoya - Driver for the 1D Euler Equations

import numpy as np
from analyticalSolution import *
import matplotlib.pyplot as plt
from problem import *
from spatialDiscretization import *
from element import *
from spatialDiscHighOrder import *
from implicitSolver import *
import itertools

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

    reftitle = figtitle + "_elem_refine.npy"
    np.save("../results/"+reftitle, np.array([DOF, errornorms]))
    return DOF, errornorms, reftitle

def implicitHighOrderQuasi1D_p_refinement(label, S_star, p_01, T_01, gamma, R, elementType, K, gridType, p_0, n_grids):
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


    DOF = np.zeros(n_grids)
    errornorms = np.zeros(n_grids)
    p = p_0
    for i in range(0,n_grids):
        refElement = Element(elementType, p, gridType)
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
        print("Grid Level: ", i, " p: ", p, " DOF:", DOF)
        print("Error norm: ", error)
        errornorms[i] = error
        p = p+1

    reftitle = figtitle + "_p_refine.npy"
    np.save("../results/"+reftitle, np.array([DOF, errornorms]))
    return DOF, errornorms, reftitle

def implicitHighOrderQuasi1D_fd_refinement(label, S_star, p_01, T_01, gamma, R, elementType, p, gridType, N_0, n_grids):
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


    DOF = np.zeros(n_grids)
    errornorms = np.zeros(n_grids)
    N = N_0
    K = 2
    for i in range(0,n_grids):
        # reference element
        refElement = Element(elementType, p, gridType="uniform", Np=N)
        print("ref np", refElement.Np)
        hoScheme = SpatialDiscHighOrder(q1D, refElement, K)
        DOF[i] = hoScheme.M
        Ma, T, pres, u_exact = quasi1D(hoScheme.mesh, S_star, p_01, T_01, gamma, R)
        figtitle = "q1d_subsonic_" + label + "_" + elementType + "_"+ gridType + "_p" + str(p) + "_K" + str(K)

        # run iterations and save to file
        timeMarch = implicitSolver(figtitle, hoScheme, method=0, C=10, isUnsteady=False,
                                   useLocalTimeStep=True, max_its=1000000, rel_tol=1.e-12)
        res = timeMarch.runSolver()
        u_f = timeMarch.Q
        error = hoScheme.calculateError(u_f[0::3], u_exact[0::3])
        print("Grid Level: ", i, " K: ", K, " DOF:", DOF)
        print("Error norm: ", error)
        errornorms[i] = error
        N = N*2

    reftitle = figtitle + "_fd_refine.npy"
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

def gridConvPlot(title, names, labels):
    n_plots = len(names)
    plt.rcParams.update({'font.size': 14})
    resPlot = plt.figure(figsize=(9,6))
    plt.grid()
    plt.xlabel("Degrees of Freedom")
    plt.ylabel("Error in $\mathcal{U}_1$")
    plt.ylim([1.e-12, 1.e-1])
    plt.xlim([4, 500])
    nametotal = ""
    marker = itertools.cycle(('x', '+', '.', '*'))
    for i in range(0, n_plots):
        results = np.load("../results/" + names[i])
        convrate=np.polyfit(np.log(results[0, -3:-1]), np.log(results[1, -3:-1]), 1)
        leg = labels[i] + (", $r$ = %.2f" % (-1.0*convrate[0]))
        if i > n_plots - 3:
            plt.loglog(results[0, :], results[1, :], marker = 'o', linestyle= '-', markersize=5, linewidth=3.0, label=labels[i])
        else:
            plt.loglog(results[0, :], results[1, :], marker='o', linestyle='-', markersize=5, linewidth=3.0,
                       label=leg)
        nametotal = nametotal + "_" + names[i]

    plt.legend(fontsize='small')
    plt.show()
    resPlot.savefig("../plots/gridconv_" + title + ".pdf", bbox_inches='tight')

def gridConvPlotPref(title, names, labels):
    n_plots = len(names)
    plt.rcParams.update({'font.size': 13})
    resPlot = plt.figure(figsize=(8,6))
    plt.grid()
    plt.xlabel("Degrees of Freedom")
    plt.ylabel("Error in $\mathcal{U}_1$")
    # plt.ylim([1.e-12, 1.e-1])
    # plt.xlim([4, 500])
    nametotal = ""
    marker = itertools.cycle(('x', '+', '.', '*'))
    for i in range(0, n_plots):
        results = np.load("../results/" + names[i])
        convrate=np.polyfit(np.log(results[0, -3:-1]), np.log(results[1, -3:-1]), 1)
        leg = labels[i] + (", $r$ = %.2f" % (-1.0*convrate[0]))
        plt.semilogy(results[0, :], results[1, :], marker = 'o', linestyle= '-', markersize=5, linewidth=3.0, label=leg)
        nametotal = nametotal + "_" + names[i]

    plt.legend()
    plt.show()
    resPlot.savefig("../plots/gridconv_" + title + ".pdf", bbox_inches='tight')



R, u_f, hoScheme, title = implicitHighOrderQuasi1DDriver("test", 0.8, 1.e5, 300., 1.4, 287, "dg_dense", 5, "lg", 2)
# print(title)
createPlotsQuasi1D(title)
# R_final = hoScheme.localResidualExplicitForm(u_f,0)

# DOF, errornorms, title = implicitHighOrderQuasi1D_element_refinement("test", 0.8, 1.e5, 300., 1.4, 287, "dg_diag", 6, "lg", 2, 5)

#DOF, errornorms, title = implicitHighOrderQuasi1D_element_refinement("test", 0.8, 1.e5, 300., 1.4, 287, "csbp", 2, "uniform", 2, 5)
#DOF, errornorms, title = implicitHighOrderQuasi1D_element_refinement("test", 0.8, 1.e5, 300., 1.4, 287, "csbp", 3, "uniform", 2, 5)
#
# DOF, errornorms, title = implicitHighOrderQuasi1D_fd_refinement("test", 0.8, 1.e5, 300., 1.4, 287, "csbp", 2, "uniform", 20, 2)
# DOF, errornorms, title = implicitHighOrderQuasi1D_fd_refinement("test", 0.8, 1.e5, 300., 1.4, 287, "csbp", 3, "uniform", 20, 2)
# DOF, errornorms, title = implicitHighOrderQuasi1D_fd_refinement("test", 0.8, 1.e5, 300., 1.4, 287, "csbp", 4, "uniform", 20, 2)

# DOF, errornorms, title = DOF, errornorms, title = implicitHighOrderQuasi1D_p_refinement("test", 0.8, 1.e5, 300., 1.4, 287, "dg_diag", 2, "lgl", 2, 29)
# DOF, errornorms, title = DOF, errornorms, title = implicitHighOrderQuasi1D_p_refinement("test", 0.8, 1.e5, 300., 1.4, 287, "dg_diag", 2, "lg", 2, 29)
#
# # print(title)
# gridConvPlotPref("pref", ["q1d_subsonic_test_dg_diag_lgl_p20_K2_p_refine.npy", "q1d_subsonic_test_dg_diag_lg_p20_K2_p_refine.npy"], ["DG-LGL", "DG-LG"])



#gridConvPlot("csbp_href", ["q1d_subsonic_test_csbp_uniform_p2_K32_elem_refine.npy", "q1d_subsonic_test_csbp_uniform_p3_K32_elem_refine.npy"], ["CSBP, $p=2$", "CSBP, $p=3$"])
# gridConvPlot("csbp_fdref", ["q1d_subsonic_test_csbp_uniform_p2_K2_fd_refine.npy",
#                             "q1d_subsonic_test_csbp_uniform_p3_K2_fd_refine.npy",
#                             "q1d_subsonic_test_csbp_uniform_p4_K2_fd_refine.npy",],
#                             ["CSBP, $p=2$", "CSBP, $p=3$", "CSBP, $p=4$"])

# gridConvPlot("href_plot", ["q1d_subsonic_test_dg_diag_lgl_p2_K64_elem_refine.npy",
#                 "q1d_subsonic_test_dg_diag_lg_p2_K64_elem_refine.npy",
#                "q1d_subsonic_test_dg_diag_lgl_p4_K64_elem_refine.npy",
#               "q1d_subsonic_test_dg_diag_lg_p4_K64_elem_refine.npy",
#               "q1d_subsonic_test_dg_diag_lgl_p6_K32_elem_refine.npy",
#               "q1d_subsonic_test_dg_diag_lg_p6_K32_elem_refine.npy",
#                 "q1d_subsonic_test_csbp_uniform_p2_K32_elem_refine.npy",
#             "q1d_subsonic_test_csbp_uniform_p3_K32_elem_refine.npy", "q1d_subsonic_test_dg_diag_lgl_p20_K2_p_refine.npy", "q1d_subsonic_test_dg_diag_lg_p20_K2_p_refine.npy"],
#              ["DG-LGL, $p=2$","DG-LG, $p=2$", "DG-LGL, $p=4$","DG-LG, $p=4$","DG-LGL, $p=6$","DG-LG, $p=6$","CSBP, $p=2$", "CSBP, $p=3$", "DG-LGL, $p$-refinement", "DG-LG, $p$-refinement"])

# u = ho.u_0_interp
# R_mat = ho.localResidualInterior(u, 2)
# R_exp = ho.localResidualExplicitForm(u, 2)
# jac = dRdQ.toarray()
# print("R1_Matrix: ", R_mat)
# print("Explicit Form: ", R_exp)
# print("Full residual: ", np.reshape(ho.flowResidual(u),[ho.K,ho.Np*ho.n_eq]))