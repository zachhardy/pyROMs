"""
This example is derived from the PyDMD tutorial 1.
"""
import os.path

import numpy as np
import matplotlib.pyplot as plt

from numpy.linalg import norm

from rom.dmd import DMD
from rom.pod import POD

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
X1 = f1(xgrid, tgrid).T
X2 = f2(xgrid, tgrid).T
X = X1 + X2

# =================================== Plot the data
fig = plt.figure(figsize=(12, 6))
titles = [r"$f_1(x, t)$", r"$f_2(x, t)$", r"$f~(x, t)$"]
data = [X1, X2, X]
for n, title, d in zip(range(131, 134), titles, data):
    plt.subplot(n)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("t")
    plt.pcolormesh(xgrid, tgrid, d.real.T, cmap='jet')
    plt.colorbar()
plt.tight_layout()

# =================================== Fit a DMD model
dmd = DMD(svd_rank=2)
dmd.fit(X)
X_dmd = dmd.reconstructed_data.real

# =================================== Fit a POD model
pod = POD(svd_rank=2)
pod.fit(X)
X_pod = pod.reconstructed_data.real

# =================================== Plot DMD results
fig = plt.figure(figsize=(12, 6))
titles = [r"$X$", r"$X_{{DMD}}$", "$| X - X_{{DMD}} |$"]
data = [X, X_dmd, abs((X - X_dmd).real)]
for n, title, d in zip(range(131, 134), titles, data):
    plt.subplot(n)
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('t')
    plt.pcolormesh(xgrid, tgrid, d.real.T, cmap='jet')
    plt.colorbar()
plt.tight_layout()

# =================================== Plot POD results
fig = plt.figure(figsize=(12, 6))
titles = [r"$X$", r"$X_{{POD}}$", "$| X - X_{{POD}} |$"]
data = [X, X_dmd, abs((X - X_dmd).real)]
for n, title, d in zip(range(131, 134), titles, data):
    plt.subplot(n)
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('t')
    plt.pcolormesh(xgrid, tgrid, d.real.T, cmap='jet')
    plt.colorbar()
plt.tight_layout()

# =================================== Plot DMD modes
fig = plt.figure(2, figsize=(8, 6))
f = [f1, f2]
for i, n in enumerate(range(121, 123)):
    signal = f[i](grid, 0.0).real
    signal /= norm(signal)

    if i == 0:
        dmd_mode = dmd.modes.real[:, 1]
        dmd_mode /= norm(dmd_mode)
    else:
        dmd_mode = dmd.modes.real[:, 0]
        dmd_mode /= norm(dmd_mode)

    pod_mode = pod.modes.real[:, i]
    pod_mode /= norm(pod_mode)

    plt.subplot(n)
    plt.title(f'Signal {i}')
    plt.xlabel('x')
    plt.plot(grid, signal, '-ob', label='Signal')
    plt.plot(grid, dmd_mode, '-*r', label='DMD')
    plt.plot(grid, pod_mode, '-^g', label='POD')
    plt.legend()
    plt.grid(True)
plt.tight_layout()

path = os.path.dirname(os.path.realpath(__file__))
plt.savefig(path + '/test1_dmd_pod_modes.pdf')
plt.show()
