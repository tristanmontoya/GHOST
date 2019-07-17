# GHOST - 1D Affine Mesh (Line Segments)

import numpy as np
import matplotlib.pyplot as plt
from AffineMesh import AffineMesh


class AffineMesh1D(AffineMesh):

    def __init__(self, x_L, x_R, K, spacing='uniform', indexing='ordered'):
        Nv = K + 1
        v = np.linspace(x_L,x_R,Nv).reshape([Nv,1])
        VtoE = np.zeros((K,2), dtype=int)

        if indexing == 'ordered':
            VtoE[:,0] = np.arange(0,Nv-1)  # left endpoints
            VtoE[:,1] = np.arange(1,Nv)  # right endpoints

        AffineMesh.__init__(self, 1, v, VtoE, type='simplex')

    def plotMesh(self):
        meshplt = plt.figure()
        plt.plot(self.v[:,0], np.zeros(self.Nv), '-o')
        plt.show()

