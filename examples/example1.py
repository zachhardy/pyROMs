"""
This example is derived from the PyDMD tutorial 1.
"""
import sys

import numpy as np
import matplotlib.pyplot as plt

from rom.dmd import DMD

plt.rcParams['pcolor.shading'] = 'auto'


# =================================== Signals
def f1(x, t):
    return 1.0 / np.cosh(x + 3) * np.exp(2.3j * t)


def f2(x, t):
    return 2.0 / np.cosh(x) * np.tanh(x) * np.exp(2.8j * t)


# =================================== Define the meshgrid
grid = np.linspace(-5, 5, 65)
times = np.linspace(0, 4 * np.pi, 129)
xgrid, tgrid = np.meshgrid(grid, times)

# =================================== Generate the data
X1 = f1(xgrid, tgrid)
X2 = f2(xgrid, tgrid)
X = X1 + X2

# =================================== Plot the data
fig = plt.figure(figsize=(12, 6))
titles = [r'$f_1(x, t)$', r'$f_2(x, t)$', r'$f~(x, t)$']
data = [X1, X2, X]
for n, title, d in zip(range(131, 134), titles, data):
    plt.subplot(n)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("t")
    plt.pcolormesh(xgrid, tgrid, d.real, cmap='jet')
    plt.colorbar()
plt.tight_layout()
plt.show()

# =================================== Fit a DMD model
dmd = DMD(svd_rank=2)
dmd.fit(X)
dmd.plot_singular_values(logscale=True)
dmd.plot_error_decay(normalized=False)
dmd.plot_eigs()
X_dmd = dmd.reconstructed_data.real

# =================================== Plot DMD results
fig = plt.figure(figsize=(12, 6))
titles = [r'$X$', r'$X_{{DMD}}$', '$| X - X_{{DMD}} |$']
data = [X, X_dmd, abs((X - X_dmd).real)]
for n, title, d in zip(range(131, 134), titles, data):
    plt.subplot(n)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("t")
    plt.pcolormesh(xgrid, tgrid, d.real, cmap='jet')
    plt.colorbar()
plt.tight_layout()
plt.show()
