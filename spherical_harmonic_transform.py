import matplotlib.pyplot as plt
import numpy as np

import pyshtools
from pyshtools.shio import shread
from pyshtools.expand import MakeGridDH
from pyshtools.expand import SHExpandDH
from pyshtools.spectralanalysis import spectrum

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

# function_grid = MakeGridDH(clm, sampling=2)
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.imshow(topo, extent=(0, 360, -90, 90), cmap='viridis')
ax.set(xlabel='longitude', ylabel='latitude')
plt.show()

coeffs = SHExpandDH(function_grid, sampling=2)
print(coeffs)
nl = coeffs.shape[1]
ls = np.arange(nl)
print(ls)

power_per_l = spectrum(coeffs)
print(power_per_l)
fig, ax = plt.subplots(1, 1, figsize=(len(ls), 5))
ax.plot(ls, power_per_l, 'bo')
ax.plot(ls, power_per_l, 'k-')
plt.xticks(range(len(ls)))
# ax.set_yscale('log')
# ax.set_xscale('log')
ax.grid()
plt.show()

power_per_l = spectrum(coeffs)
print(power_per_l)
fig, ax = plt.subplots(1, 1, figsize=(len(ls), 5))
ax.plot(ls, power_per_l, 'bo')
ax.plot(ls, power_per_l, 'k-')
plt.xticks(range(len(ls)))
# ax.set_yscale('log')
# ax.set_xscale('log')
ax.grid()
plt.show()