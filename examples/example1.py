"""
This example is derived from the PyDMD tutorial 1.
"""

import numpy as np
from numpy.linalg import norm

import matplotlib.pyplot as plt
plt.rcParams['pcolor.shading'] = 'auto'

from matplotlib.figure import Figure
from matplotlib.axes import Axes
from typing import List

from rom.dmd import DMD


# =================================== Signals
def f1(x, t):
    return 1.0 / np.cosh(x + 3) * np.exp(2.3j * t)

def f2(x, t):
    return 2.0 / np.cosh(x) * np.tanh(x) * np.exp(2.8j * t)

# =================================== Define the meshgrid
x = np.linspace(-5, 5, 65)
t = np.linspace(0, 4*np.pi, 129)
xgrid, tgrid = np.meshgrid(x, t)

# =================================== Generate the data
X1 = f1(xgrid, tgrid)
X2 = f2(xgrid, tgrid)
X = X1 + X2

# =================================== Plot the data
fig = plt.figure(figsize=(15, 6))
titles = [r'$f_1(x, t)$', r'$f_2(x, t)$', r'$f~(x, t)$']
data = [X1, X2, X]
for n, title, d in zip(range(131, 134), titles, data):
    plt.subplot(n)
    plt.title(title)
    plt.pcolormesh(xgrid, tgrid, d.real, cmap='jet')
    plt.colorbar()
plt.show()

# =================================== Fit a DMD model
dmd = DMD(svd_rank=2).fit(X, t)
dmd.plot_singular_values(logscale=False)
X_dmd = dmd.reconstructed_data.real

# =================================== Plot DMD results
fig = plt.figure(figsize=(15, 6))
titles = [r'$X$', r'$X_{{DMD}}$', '$| X - X_{{DMD}} |$']
data = [X, X_dmd, abs((X - X_dmd).real)]
for n, title, d in zip(range(131, 134), titles, data):
    plt.subplot(n)
    plt.title(title)
    plt.pcolormesh(xgrid, tgrid, d.real, cmap='jet')
    plt.colorbar()
plt.show()




