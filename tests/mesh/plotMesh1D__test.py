import numpy as np
from AffineMesh1D import AffineMesh1D

def plotMesh1D_test():
    mesh = AffineMesh1D(0.0,1.0,10, spacing = 'uniform', indexing='random')
    mesh.plot_mesh('meshtest_1d')
    return mesh

mesh = plotMesh1D_test()
