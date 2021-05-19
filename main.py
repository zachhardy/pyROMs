import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from typing import List
from rom.dmd import DMD

plt.rcParams['pcolor.shading'] = 'auto'

# =================================== Signals
def f1(x, t):
    return 1.0 / np.cosh(x + 3) * np.exp(2.3j * t)

def f2(x, t):
    return 2.0 / np.cosh(x) * np.tanh(x) * np.exp(2.8j * t)

# =================================== Define the meshgrid
x = np.linspace(-5, 5, 400)
t = np.linspace(0, 4*np.pi, 100)
xgrid, tgrid = np.meshgrid(x, t)

# =================================== Generate the data
X1 = f1(xgrid, tgrid)
X2 = f2(xgrid, tgrid)
X = X1 + X2

# =================================== Plot the data
# fig = plt.figure(figsize=(17, 6))
# titles = [r'$f_1(x, t)$', r'$f_2(x, t)$', r'$f~(x, t)$']
# data = [X1, X2, X]
# for n, title, d in zip(range(131, 134), titles, data):
#     plt.subplot(n)
#     plt.pcolormesh(xgrid, tgrid, d.real, cmap='jet')
#     plt.title(title)
# plt.colorbar()
# plt.show()

# =================================== Fit a DMD model
dmd = DMD(svd_rank=2).fit(X)
dmd.plot_singular_values()
dmd.plot_1D_profiles(x=x)
dmd.plot_dynamics(t=t)
dmd.plot_1D_profiles_and_dynamics(x=x, t=t)
dmd.plot_timestep_errors()
dmd.plot_error_decay()



# =================================== Plot modes and dynamics
# fig = plt.figure(figsize=(6, 4))
# plt.title('Modes')
# for m, mode in enumerate(dmd.modes.T):
#     plt.plot(x, mode.real, label=f'Mode {m}')
# plt.legend()
# plt.show()
#
# fig = plt.figure(figsize=(6, 4))
# plt.title('Dynamics')
# for d, dynamic in enumerate(dmd.dynamics):
#     plt.plot(t, dynamic.real, label=f'Dynamic {d}')
# plt.legend()
# plt.show()
#
# fig = plt.figure(figsize=(17, 6))
# modes, dynamics = dmd.modes.T, dmd.dynamics
# for n, mode, dynamic in zip(range(131, 134), modes, dynamics):
#     plt.subplot(n)
#     plt.title(f'Mode {n - 131}')
#     data = mode.reshape(-1, 1) @ dynamic.reshape(1, -1)
#     plt.pcolormesh(xgrid, tgrid, data.real.T, cmap='jet')
#
# plt.subplot(133)
# plt.title('Reconstructed Data')
# plt.pcolormesh(xgrid, tgrid, dmd.reconstructed_data.real, cmap='jet')
# plt.colorbar()
# plt.show()
#
# # =================================== Plot error
# error = (X - dmd.reconstructed_data).real
# fig = plt.figure(figsize=(6, 4))
# plt.title('DMD Reconstruction Error')
# plt.pcolormesh(xgrid, tgrid, error, cmap='jet')
# plt.colorbar()
# plt.show()
#
