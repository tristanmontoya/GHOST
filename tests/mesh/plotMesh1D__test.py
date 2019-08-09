import numpy as np
import mesh.AffineMesh1D as mesh


def plotMesh1D_test():
    msh = mesh.AffineMesh1D(0.0,1.0,10, spacing = 'uniform', indexing='random')
    msh.plot_mesh('meshtest_1d')
    return msh

msh = plotMesh1D_test()
