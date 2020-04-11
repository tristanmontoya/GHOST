import modepy as mp
import quadpy as qp
import numpy as np
from matplotlib import pyplot as plt
p=3


def plot_tri(p):
    v = np.array([[-1.0, 1.0, -1.0, -1.0],[-1.0,-1.0,1.0,-1.0]])
    x_lg = mp.LegendreGaussQuadrature(p).nodes
    xlg_stretch = np.array([np.sqrt(2.0)*x_lg, np.zeros(len(x_lg))] )
    xp = mp.warp_and_blend_nodes(2,p)
    xv = mp.XiaoGimbutasSimplexQuadrature(2*p+1,2).nodes
    rot_mat = np.array([[1.0/np.sqrt(2.0), 1.0/np.sqrt(2.0)],
                       [-1.0/np.sqrt(2.0), 1.0/np.sqrt(2.0)]])
    xlg_diag = rot_mat @ xlg_stretch


    mshpltt = plt.figure()

    plt.plot(v[0,:], v[1,:], '-k')
    plt.plot(x_lg, -1.0*np.ones(len(x_lg)), 's',color='black',fillstyle='none',markersize=10)
    plt.plot(-1.0*np.ones(len(x_lg)), x_lg, 's', color='black',fillstyle='none',markersize=10)
    plt.plot(xlg_diag[0,:],xlg_diag[1,:], 's',color='black',fillstyle='none',markersize=10)

    plt.plot(xv[0,:], xv[1,:], 'o', color='black',fillstyle='none',markersize=10)
    plt.plot(xp[0,:], xp[1,:], 's',color='black', markersize=10)

    ax = plt.axes()
    ax.set_aspect('equal')
    plt.axis('off')
    mshpltt.savefig("../plots/tri_" + str(p) +".pdf", bbox_inches=0, pad_inches=0)
    plt.show()


def plot_quad(p):
    n_1 = p+1
    n_2 = p+1
    rule_1 = 'lgl'
    rule_2 = 'lg'
    x_lg = mp.LegendreGaussQuadrature(p).nodes
    xp = np.zeros([2,n_1**2])
    xv = np.zeros([2,n_2**2])
    v = np.array([[-1.0, 1.0, 1.0, -1.0, -1.0],[-1.0, -1.0,1.0,1.0, -1.0]])

    for i in range(0, n_1):
        if rule_1 == 'lg':
            xp[0, i * (n_1):(i + 1) * (n_1)] = qp.line_segment.GaussLegendre(n_1).points[i]
        elif rule_1 == 'lgl':
            xp[0, i * (n_1):(i + 1) * (n_1)] = qp.line_segment.GaussLobatto(n_1).points[i]
        elif rule_1 == 'lgr':
            xp[0, i * (n_1):(i + 1) * (n_1)] = qp.line_segment.GaussRadau(n_1).points[i]

    for j in range(0, n_1):
        if rule_1 == 'lg':
            xp[1, j * (n_1):(j + 1) * (n_1)] = qp.line_segment.GaussLegendre(n_2).points
        elif rule_1 == 'lgl':
            xp[1, j * (n_1):(j + 1) * (n_1)] = qp.line_segment.GaussLobatto(n_2).points
        elif rule_1 == 'lgr':
            xp[1, j * (n_1):(j + 1) * (n_1)] = qp.line_segment.GaussRadau(n_2).points

    for i in range(0, n_2):
        if rule_2 == 'lg':
            xv[0, i * (n_2):(i + 1) * (n_2)] = qp.line_segment.GaussLegendre(n_1).points[i]
        elif rule_2 == 'lgl':
            xv[0, i * (n_2):(i + 1) * (n_2)] = qp.line_segment.GaussLobatto(n_1).points[i]
        elif rule_2 == 'lgr':
            xv[0, i * (n_2):(i + 1) * (n_2)] = qp.line_segment.GaussRadau(n_1).points[i]

    for j in range(0, n_2):
        if rule_2 == 'lg':
            xv[1, j * (n_2):(j + 1) * (n_2)] = qp.line_segment.GaussLegendre(n_2).points
        elif rule_2 == 'lgl':
            xv[1, j * (n_2):(j + 1) * (n_2)] = qp.line_segment.GaussLobatto(n_2).points
        elif rule_2 == 'lgr':
            xv[1, j * (n_2):(j + 1) * (n_2)] = qp.line_segment.GaussRadau(n_2).points

    mshpltt = plt.figure()
    plt.plot(v[0,:], v[1,:], '-k')

    plt.plot(x_lg, -1.0*np.ones(len(x_lg)), 's',color='black',fillstyle='none',markersize=10)
    plt.plot(x_lg, 1.0*np.ones(len(x_lg)), 's',color='black',fillstyle='none',markersize=10)
    plt.plot(-1.0*np.ones(len(x_lg)), x_lg, 's', color='black',fillstyle='none',markersize=10)
    plt.plot(1.0*np.ones(len(x_lg)), x_lg, 's', color='black',fillstyle='none',markersize=10)

    plt.plot(xp[0,:], xp[1,:], 's', color='black', markersize=10)
    plt.plot(xv[0, :], xv[1, :], 'o', color='black', fillstyle='none', markersize=10)

    ax = plt.axes()
    ax.set_aspect('equal')
    plt.axis('off')
    mshpltt.savefig("../plots/quad_" + str(p) +".pdf", bbox_inches=0, pad_inches=0)
    plt.show()

plot_tri(3)
plot_quad(3)