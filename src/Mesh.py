# GHOST - Mesh Data Structure (Unstructured)

import numpy as np
import quadpy as qp
from Operator import DenseLinearOperator, DiagonalOperator
import matplotlib.pyplot as plt
from collections import namedtuple

Mesh = namedtuple('Mesh', 'name d K Nf_total Nf FtoE EtoF f_side Nv N_gamma xv xbar x_gamma n_gamma n_gathered')


# -1 to 1
def nonsingular_map_1d(p, x):
    return 2.0/(2.0**p - 1) * ((0.5*(x + 3.0))**p - 1.0) + 1.0


def make_mesh_1d(name, x_L, x_R, K, N, nodes='lg', spacing='uniform',
                        indexing='ordered', periodic=True, transform=None):

    if periodic:
        Nf_total = K
    else:
        Nf_total = K + 1

    Nf = 2*np.ones(K,dtype=int)
    N_gamma = [[1, 1] for k in range(0,K)]
    n_gamma = [[np.array([[-1.0]]),np.array([[1.0]])]for k in range(0,K)]

    Nv = [N for k in range(0, K)]

    # generate vertices
    if spacing == 'uniform':
        v = np.linspace(x_L, x_R, K+1).reshape([K+1, 1])
    else:
        raise NotImplementedError

    # vertex (facet) to element connectivity maps
    FtoE = np.zeros((K, 2), dtype=int)
    FtoE[:, 0] = np.arange(0, K)  # left endpoints
    FtoE[:, 1] = np.arange(1, K+1)  # right endpoints

    if indexing == 'random':  # simulates an unstructured grid
        np.random.shuffle(FtoE[:, 0])
        FtoE[:, 1] = FtoE[:, 0] + 1

    if nodes=='lg':
        xi = qp.line_segment.GaussLegendre(N).points
    elif nodes == 'lgl':
        xi = qp.line_segment.GaussLobatto(N).points
    else:
        raise NotImplementedError

    h = [(v[FtoE[k, 1], 0] - v[FtoE[k, 0], 0]) for k in range(0, K)]

    xbar = 0.5 * (v[FtoE[:, 0], 0] + v[FtoE[:, 1], 0]).reshape((K,1))
    detJv = [DiagonalOperator(np.ones(Nv[k])*h[k]/2.0)
             for k in range(0, K)]
    xv = [detJv[k](xi).reshape(Nv[k],1) + np.ones((Nv[k], 1))*xbar[k, 0]
          for k in range(0, K)]

    # these are overlapped
    x_gamma = [[np.array([xbar[k] - np.array([h[k]])/2.0]),
                np.array([xbar[k] + np.array([h[k]])/2.0])] for k in range(0, K)]

    if transform is not None:
        xbar = transform(xbar)
        xv = [transform(xv[i]) for i in range(0, K)]
        x_gamma = [ [transform(x_gamma[i][0]), transform(x_gamma[i][1])] for i in range(0, K)]

    if periodic:
        endpoint = np.where(FtoE[:, 1] == K)
        FtoE[endpoint, 1] = 0

    EtoF = [np.where(FtoE == f) for f in range(0, Nf_total)]

    n_gathered = [(n_gamma[EtoF[f][0][0]][EtoF[f][1][0]],
                   n_gamma[EtoF[f][0][1]][EtoF[f][1][1]])
                  for f in range(0, Nf_total)]

    f_side_0 = [1 if (EtoF[FtoE[k,0]][0][0] == k) else -1 for k in range(0, K)]
    f_side_1 = [1 if (EtoF[FtoE[k, 1]][0][0] == k) else -1 for k in range(0, K)]
    f_side = [[f_side_0[k], f_side_1[k]] for k in range(0, K)]

    mesh = Mesh(name=name, d=1, Nf_total=Nf_total, Nf=Nf, K=K, FtoE=FtoE,
                EtoF=EtoF, f_side=f_side, Nv=Nv, N_gamma=N_gamma, xbar=xbar,
                xv=xv, x_gamma=x_gamma, n_gamma=n_gamma,
                n_gathered=n_gathered)

    return mesh


def eval_grid_function(mesh, f):
    return [f(mesh.xv[k]) for k in range(0, mesh.K)]


def eval_facet_function(mesh, f):
    return [[f(mesh.x_gamma[k][gamma]) for gamma in range(0,mesh.Nf[k])] for k in range(0, mesh.K)]


def plot_mesh(mesh, fontsize=8):

    if mesh.d == 1:
        mins = [np.amin(mesh.xv[k]) for k in range(0, mesh.K)]
        maxes = [np.amax(mesh.xv[k]) for k in range(0, mesh.K)]
        x_L = min(mins)
        x_R = max(maxes)
        L = x_R - x_L
        meshplt = plt.figure()
        ax = plt.axes()
        plt.xlim([x_L - 0.1 * L, x_R + 0.1 * L])
        plt.ylim([-0.1 * L, 0.1 * L])
        ax.get_xaxis().set_visible(False)  # this removes the ticks and numbers for x axis
        ax.get_yaxis().set_visible(False)  # this removes the ticks and numbers for y axis
        ax.set_aspect('equal')
        plt.axis('off')

        color = iter(plt.cm.rainbow(np.linspace(0, 1, mesh.K)))
        for k in range(0, mesh.K):
            ax.plot(mesh.xv[k][:,0], np.zeros(mesh.Nv[k]), '-o', markersize=fontsize/4, color=next(color))
            plt.text(mesh.xbar[k,0], 0.05 * L, str(k)+ "\n" +str(mesh.FtoE[k]), color='black',
                     fontsize=fontsize, ha='center')

            for gamma in range(0, mesh.Nf[k]):
                ax.plot(mesh.x_gamma[k][gamma][:,0], np.zeros(mesh.N_gamma[k][gamma]),
                        '-x', markersize=fontsize/4, color='black')

        plt.show()
        meshplt.savefig("./" + mesh.name + ".pdf", bbox_inches=0, pad_inches=0)


def gather_field(mesh, u_gamma):

    if mesh.d == 1:
        return [(u_gamma[mesh.EtoF[f][0][0]][mesh.EtoF[f][1][0]],
                 u_gamma[mesh.EtoF[f][0][1]][mesh.EtoF[f][1][1]])
                for f in range(0, mesh.Nf_total)]
    else:
        raise NotImplementedError


def plot_on_volume_nodes(mesh, u, plotname, fontsize=8):

    if mesh.d == 1:

        meshplt = plt.figure()
        ax = plt.axes()
        color = iter(plt.cm.rainbow(np.linspace(0, 1, mesh.K)))
        for k in range(0, mesh.K):
            ax.plot(mesh.xv[k][:,0], u[k], '-o', markersize=fontsize/4, color=next(color))

        plt.show()
        meshplt.savefig("./" + mesh.name + "_" + plotname + ".pdf", bbox_inches=0, pad_inches=0)


def plot_on_volume_and_facet_nodes(mesh, u_v, u_gamma, plotname, fontsize=8):

    if mesh.d == 1:
        meshplt = plt.figure()
        ax = plt.axes()
        color = iter(plt.cm.rainbow(np.linspace(0, 1, mesh.K)))
        for k in range(0, mesh.K):
            col = next(color)
            ax.plot(mesh.xv[k][:,0], u_v[k], '-o', markersize=fontsize/4, color=col)
            for gamma in range(0, mesh.Nf[k]):
                ax.plot(mesh.x_gamma[k][gamma][:,0], u_gamma[k][gamma],
                        '-x', markersize=fontsize/4, color=col)

        plt.show()
        meshplt.savefig("./" + mesh.name + "_" + plotname + ".pdf", bbox_inches=0, pad_inches=0)