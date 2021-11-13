import matplotlib.pyplot as plt
import numpy as np

import pyshtools
from pyshtools.shio import shread
from pyshtools.expand import MakeGridDH
# from pyshtools.expand import SHExpandDH
from my_ducc0_wrapper import SHExpandDH
from pyshtools.spectralanalysis import spectrum
from my_backends.ducc0_wrapper import MakeGridDH_adjoint_analysis

clm = np.array(
    [[[0, 0, 0],
     [0, 27, 0],
     [0, 0, 36]],
     
     [[0, 0, 0],
     [0, 0, 0],
     [0, 32, 0]]
    ]
)
print(clm.shape)

function_grid = MakeGridDH(clm, sampling=2)
print(function_grid.shape)
# fig, ax = plt.subplots(1, 1, figsize=(10, 10))
# ax.imshow(function_grid, extent=(0, 360, -90, 90), cmap='viridis')
# ax.set(xlabel='longitude', ylabel='latitude')
# plt.show()

clm_infer = SHExpandDH(function_grid, sampling=2)
print(clm_infer)

adjoint = MakeGridDH_adjoint_analysis(clm_infer, sampling=2)
# adjoint = SHExpandDH(clm_infer, sampling=2, flag=True)
print(adjoint.shape)

# nl = coeffs.shape[1]
# ls = np.arange(nl)
# print(ls)

# power_per_l = spectrum(coeffs)
# print(power_per_l)
# print(type(power_per_l))
# fig, ax = plt.subplots(1, 1, figsize=(len(ls), 5))
# ax.plot(ls, power_per_l, 'bo')
# ax.plot(ls, power_per_l, 'k-')
# plt.xticks(range(len(ls)))
# # ax.set_yscale('log')
# # ax.set_xscale('log')
# ax.grid()
# plt.show()

import numpy as np

# x is a 2D map
# y is spherical harmonic coefficients

# note that this is probably not right because we want some entries of y to be 0
N = 3
x = np.random.uniform(0, 1, (2*N, 4*N))
# y = np.random.uniform(0, 1, (2, N, N))
y = clm

lhs = SHExpandDH(x, sampling=2)
lhs = np.sum(y*lhs)

rhs = MakeGridDH_adjoint_analysis(y, sampling=2)
rhs = np.sum(rhs*x)

print(rhs, lhs, lhs/rhs)
# the dot product test for the adjoint is not working
    # could this be a normalization issue? 
    # then the ratio of lhs and rhs should be the same over different randomizations, which it is!

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