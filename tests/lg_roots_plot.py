import modepy as mp
import scipy.special as sp
import numpy as np
from matplotlib import pyplot as plt
p=8
x_lg = mp.LegendreGaussQuadrature(p).nodes
L = sp.legendre(p+1)
endpts = np.array([-1.0, 1.0])
xa = np.linspace(-1.0, 1.0, 1000)
mshpltt = plt.figure(figsize=(8,2))
plt.plot(endpts, np.zeros(2), "-k")

plt.plot(xa, L(xa), "-r", label="$\mathcal{L}_6$, $(\mathcal{L}_6, \mathcal{L}_6)_{{\omega}_\Omega} = 0$")

plt.plot(x_lg, np.zeros(len(x_lg)), 'o', color='k', fillstyle='none',
         markersize=10, label="$N_\Omega = 6$ Legendre-Gauss Quadrature")
ax = plt.axes()
#plt.legend()
#ax.set_aspect('equal')
plt.axis('off')
mshpltt.savefig("../plots/lg_plot" + str(p) + ".pdf", bbox_inches=0, pad_inches=0)
plt.show()