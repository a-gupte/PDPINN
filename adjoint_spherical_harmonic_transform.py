import matplotlib.pyplot as plt
import numpy as np

import pyshtools
from pyshtools.shio import shread
from pyshtools.expand import MakeGridDH
from pyshtools.expand import SHExpandDH
from pyshtools.spectralanalysis import spectrum

from pyshtools.backends.ducc0_wrapper import _synthesize_DH
from pyshtools.backends.ducc0_wrapper import *

def _adjoint_analyze_DH(map, lmax):
    alm = ducc0.sht.experimental.adjoint_synthesis_2d(
        map=map.reshape((1, map.shape[0], map.shape[1])),
        spin=0,
        lmax=lmax,
        geometry="DH",
        nthreads=nthreads,
    )
    return alm[0]

def SHadjoint(grid, norm=1, sampling=1, csphase=1, lmax_calc=None):
    if grid.shape[1] != sampling * grid.shape[0]:
        raise RuntimeError("grid resolution mismatch")
    if lmax_calc is None:
        lmax_calc = grid.shape[0] // 2 - 1
    if lmax_calc > (grid.shape[0] // 2 - 1):
        raise RuntimeError("lmax_calc too high")
    alm = _adjoint_analyze_DH(grid, lmax_calc)
    return _extract_alm(alm, lmax_calc, norm, csphase)

clm = np.array(
    [[[0, 0, 0],
     [0, 0, 0],
     [0, 0, 100]],
     [[0, 0, 0],
     [0, 0, 0],
     [0, 0, 0]]
    ]
)
print(clm.shape)

function_grid = MakeGridDH(clm, sampling=2)
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.imshow(function_grid, extent=(0, 360, -90, 90), cmap='viridis')
ax.set(xlabel='longitude', ylabel='latitude')
plt.show()

adjoint_coeffs = SHadjoint(function_grid, sampling=2)
print(coeffs)

# coeffs = SHExpandDH(function_grid, sampling=2)
# print(coeffs)
# nl = coeffs.shape[1]
# ls = np.arange(nl)
# print(ls)

# power_per_l = spectrum(coeffs)
# print(power_per_l)
# fig, ax = plt.subplots(1, 1, figsize=(len(ls), 5))
# ax.plot(ls, power_per_l, 'bo')
# ax.plot(ls, power_per_l, 'k-')
# plt.xticks(range(len(ls)))
# # ax.set_yscale('log')
# # ax.set_xscale('log')
# ax.grid()
# plt.show()

# power_per_l = spectrum(coeffs)
# print(power_per_l)
# fig, ax = plt.subplots(1, 1, figsize=(len(ls), 5))
# ax.plot(ls, power_per_l, 'bo')
# ax.plot(ls, power_per_l, 'k-')
# plt.xticks(range(len(ls)))
# # ax.set_yscale('log')
# # ax.set_xscale('log')
# ax.grid()
# plt.show()