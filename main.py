import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from rom.dmd import DMD

from typing import List

plt.rcParams['pcolor.shading'] = 'auto'

x = np.linspace(-10.0, 10, 400)
t = np.linspace(0, 4.0 * np.pi, 200)
dt = t[1] - t[0]
X, T = np.meshgrid(x, t)

f1 = np.exp(2.3j * T) / np.cosh(X + 3)
f2 = 2.0 * np.tanh(X) / np.cosh(X) * np.exp(2.8j * T)
f = f1 + f2

dmd = DMD(2, True).fit(f, t)
dmd.plot_singular_values()
f_dmd = dmd.reconstructed_data

res = plt.subplots(2, 2)
fig: Figure = res[0]
ax: List[List[Axes]] = res[1]

for i in range(len(ax)):
    for j in range(len(ax[i])):
        ax_ij = ax[i][j]
        if i == j == 0:
            cs = ax_ij.pcolormesh(X, T, f1.real)
        elif i == 0 and j == 1:
            cs = ax_ij.pcolormesh(X, T, f2.real)
        elif i == 1 and j == 0:
            cs = ax_ij.pcolormesh(X, T, f.real)
        elif i == j == 1:
            cs = ax_ij.pcolormesh(X, T, f_dmd.real)
        fig.colorbar(cs, ax=ax_ij)
plt.tight_layout()
plt.show()

dmd.plot_modes(x, [0, 1])
dmd.plot_dynamics([0, 1])

