import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from typing import List
plt.rcParams['pcolor.shading'] = 'auto'

from rom.dmd import DMD

def f1(x, t):
    return 1.0 / np.cosh(x + 3) * np.exp(2.3j * t)

def f2(x, t):
    return 2.0 / np.cosh(x) * np.tanh(x) * np.exp(2.8j * t)


x = np.linspace(-10.0, 10.0, 400)
t = np.linspace(0, 4.0 * np.pi, 200)
xgrid, tgrid = np.meshgrid(x, t)

X1 = f1(xgrid, tgrid)
X2 = f2(xgrid, tgrid)
X = X1 + X2

tinfo = {'t0': t[0], 'tf': t[-1], 'dt': t[1] - t[0]}

dmd = DMD(svd_rank=2)
dmd.fit(X, tinfo)

dmd.plot_singular_values()
dmd.plot_1D_profiles_and_dynamics(x=x, t=t)

