import quadpy as qp
import numpy as np
from matplotlib import pyplot as plt


def duffy(xi_1, xi_2):

    return (1 + xi_1)*(1-xi_2)/2.0 - 1, xi_2


def make_tensor_mesh(n_1, rule_1, n_2, rule_2):
    N = n_1*n_2
    x = np.zeros([N, 2])
    v = np.zeros([5,2])
    v[0,:] = [-1.0,-1.0]
    v[1,:] = [1.0,-1.0]
    v[2,:] = [1.0,1.0]
    v[3,:] = [-1.0,1.0]
    v[4,:] = [-1.0, -1.0]

    for i in range(0,n_1):
        if rule_1 == 'lg':
            x[i*(n_2):(i+1)*(n_2),0] = qp.line_segment.GaussLegendre(n_1).points[i]
        elif rule_1 == 'lgl':
            x[i*(n_2):(i+1)*(n_2),0] = qp.line_segment.GaussLobatto(n_1).points[i]
        elif rule_1 == 'lgr':
            x[i * (n_2):(i + 1) * (n_2), 0] = qp.line_segment.GaussRadau(n_1).points[i]
    for j in range(0,n_1):
        if rule_2 == 'lg':
            x[j*(n_2):(j+1)*(n_2),1] = qp.line_segment.GaussLegendre(n_2).points
        elif rule_2 == 'lgl':
            x[j*(n_2):(j+1)*(n_2),1] = qp.line_segment.GaussLobatto(n_2).points
        elif rule_2 == 'lgr':
            x[j * (n_2):(j + 1) * (n_2), 1] = qp.line_segment.GaussRadau(n_2).points

    title = rule_1 + ' (N = ' + str(n_1) + ') x ' + rule_2 + ' (N = ' + str(n_2) + ')'
    mshplt = plt.figure()
    #plt.title(title)
    plt.plot(v[:,0],v[:,1], '-k')
    for j in range(0,n_1):
        plt.plot(x[j*(n_2):(j+1)*(n_2),0], x[j*(n_2):(j+1)*(n_2),1], '-k')
    for i in range(0,n_2):
        plt.plot(x[i::n_1,0], x[i::n_1,1], '-k')
    plt.plot(x[:,0],x[:,1],'s',color="k",fillstyle='none',markersize=5)
    ax = plt.axes()
    ax.set_aspect('equal')
    plt.axis('off')
    mshplt.savefig("./mesh_square.pdf", bbox_inches=0, pad_inches=0)
    plt.show()

    xtri = duffy(x[:,0],x[:,1])
    vtri = duffy(v[:,0],v[:,1])

    mshpltt = plt.figure()
    plt.plot(vtri[0],vtri[1], '-k')
    for j in range(0,n_1):
        plt.plot(xtri[0][j*(n_2):(j+1)*(n_2)], xtri[1][j*(n_2):(j+1)*(n_2)], '-k')
    for i in range(0,n_2):
        plt.plot(xtri[0][i::n_1], xtri[1][i::n_1], '-k')
    plt.plot(xtri[0],xtri[1],'s',color="k",fillstyle='none',markersize=5)
    ax = plt.axes()
    ax.set_aspect('equal')
    plt.axis('off')
    mshpltt.savefig("./mesh_tri.pdf", bbox_inches=0, pad_inches=0)
    plt.show()

make_tensor_mesh(4, 'lgl', 4, 'lgr')