import numpy as np
from AffineMesh1D import AffineMesh1D
import matplotlib.pyplot as plt

def plotMesh1D_test():
    mesh = AffineMesh1D(0.0,1.0,10)
    mesh.plotMesh()

plotMesh1D_test()
