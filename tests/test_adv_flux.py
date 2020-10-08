from Problem import ConstantAdvectionPhysicalFlux

import numpy as np

d = 1
a = [np.sqrt(d) for i in range(0,d)]

f = ConstantAdvectionPhysicalFlux(d, a)
