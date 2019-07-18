# GHOST - 1D Affine Mesh (Line Segments)

import numpy as np
import matplotlib.pyplot as plt
from AffineMesh import AffineMesh


class AffineMesh1D(AffineMesh):

    def __init__(self, x_L, x_R, K, spacing='uniform', indexing='ordered'):

        Nv = K + 1
        v = np.linspace(x_L, x_R, Nv).reshape([Nv,1])
        VtoE = np.zeros((K, 2), dtype=int)

        VtoE[:, 0] = np.arange(0, Nv-1)  # left endpoints
        VtoE[:, 1] = np.arange(1, Nv)  # right endpoints

        if indexing == 'random':  # simulates an unstructured grid
            np.random.shuffle(VtoE[:, 0])
            VtoE[:, 1] = VtoE[:, 0] + 1

        AffineMesh.__init__(self, 1, v, VtoE, type='simplex')

    def computeCentroids(self):

        self.xbar = 0.5*(self.v[self.VtoE[:, 0], 0] + self.v[self.VtoE[:, 1],0]).reshape(self.K, 1)


    def computeMapping(self):

        self.J = 0.5*(self.v[self.VtoE[:, 1 ], 0] - self.v[self.VtoE[:, 0], 0]).reshape(self.K, 1, 1)
        self.detJ = np.copy(self.J.reshape(self.K, 1))  # copy because in general these are not the same
        self.s = np.copy(self.xbar)

    def plotMesh(self, figtitle, fontsize=12):
        x_L = np.amin(self.v)
        x_R = np.amax(self.v)
        L = x_R - x_L
        meshplt = plt.figure()
        ax = plt.axes()
        ax.plot(self.v[:,0], np.zeros(self.Nv), '-ok')
        plt.xlim([x_L-0.1*L, x_R+0.1*L])
        plt.ylim([-0.1*L, 0.1*L])
        ax.get_xaxis().set_visible(False) # this removes the ticks and numbers for x axis
        ax.get_yaxis().set_visible(False) # this removes the ticks and numbers for y axis
        ax.set_aspect('equal')
        plt.axis('off')

        for k in range(0,self.K):
            plt.text(self.xbar[k,0], 0.02*L, str(k), color='red', fontsize=fontsize)

        for i in range(0,self.Nv):
            plt.text(self.v[i,0], -0.05*L, str(i), color='black', fontsize=fontsize)

        plt.show()
        meshplt.savefig("./" + figtitle + ".pdf", bbox_inches=0, pad_inches=0)
