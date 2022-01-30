import matplotlib.pyplot as plt
import numpy as np

from math import sqrt
from math import pi

import pyshtools
from pyshtools.shio import shread
from pyshtools.expand import MakeGridDH
from pyshtools.expand import SHExpandDH
from pyshtools.spectralanalysis import spectrum

# from spherical_harmonics import *

def spherical_to_cartesian(theta, phi):
    # x = r sin(theta) cos(phi)
    # y = r sin(theta) sin(phi)
    # z = r cos(theta)
    return np.sin(theta) *  np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)

def Y_2_0(theta, phi):
    x, y, z = spherical_to_cartesian(theta, phi)
    return sqrt(5.0/16/pi) * (-x**2 - y**2 + 2*z**2)

def Y_2_2(theta, phi):
    x, y, z = spherical_to_cartesian(theta, phi)
    return sqrt(15.0/16/pi) * (x**2 - y**2)

def Y_3_0(theta, phi):
    x, y, z = spherical_to_cartesian(theta, phi)
    return sqrt(7.0/16/pi) * z * (2*z**2 - 3*x**2 - 3*y**2)

def Y_4_0(theta, phi):
    x, y, z = spherical_to_cartesian(theta, phi)
    return 3/16 * sqrt(1.0/pi) * (35*z**4 - 30*z**2)

def Y_4_1(theta, phi):
    x, y, z = spherical_to_cartesian(theta, phi)
    return 3/4 * sqrt(5.0/(2*pi)) * (x*z *7*z**2)

def Y_4_2(theta, phi):
    x, y, z = spherical_to_cartesian(theta, phi)
    return 3/8 * sqrt(5.0/(pi)) * (x**2 - y**2) * 7*z**2


# clm = np.array(
#     [[[0, 0, 0, 0, 0],
#      [0, 0, 0, 0, 0],
#      [1, 0, 0, 0, 0],
#      [0, 0, 0, 0, 0],
#      [1, 0, 0, 0, 0]
#      ],
#      [[0, 0, 0, 0, 0],
#      [0, 0, 0, 0, 0],
#      [0, 0, 0, 0, 0],
#      [0, 0, 0, 0, 0],
#      [0, 0, 0, 0, 0]
#      ]
#     ]
# )
clm = np.array(
    [[[1, 1, 1],
     [1, 1, 1],
     [1, 1, 1]
     ],
     [[1, 1, 1],
     [1, 1, 1],
     [1, 1, 1]
     ]
    ]
)
print(clm)

function_grid = MakeGridDH(clm, sampling=2)
# fig, ax = plt.subplots(1, 1, figsize=(10, 10))
# ax.imshow(function_grid, extent=(0, 360, -90, 90), cmap='viridis')
# ax.set(xlabel='longitude', ylabel='latitude')
# plt.show()

# N = 100

# def true_solution_freq_2_4(theta, phi):
#     return 10 * Y_2_0(theta, phi) + 10 * Y_4_0(theta, phi)  

# azimuth = np.linspace(0, np.pi, N)
# polar = np.linspace(0, 2 * np.pi, 2*N)
# azimuth, polar = np.meshgrid(azimuth, polar)
# location = np.concatenate((azimuth.reshape(-1, 1), polar.reshape(-1, 1)), axis=1)
# value = true_solution_freq_2_4(location[:, 0:1], location[:, 1:])
# function_grid = value.reshape((N, 2*N))

# fig, ax = plt.subplots(1, 1, figsize=(10, 10))
# ax.imshow(function_grid, extent=(0, 360, -90, 90), cmap='viridis')
# ax.set(xlabel='longitude', ylabel='latitude')
# plt.show()

coeffs = SHExpandDH(function_grid, sampling=2)

# print('coeffs', coeffs)

# diag = np.diag([(1 + i*(i+1))**s for i in range(coeffs.shape[1])])
diag = np.diag([i+1 for i in range(coeffs.shape[1])])
# coeffs = np.array([diag @ mat for mat in coeffs])
coeffs = diag @ coeffs

coeffs = np.round_(coeffs, decimals = 2)

print(coeffs)

nl = coeffs.shape[1]
ls = np.arange(nl)
# print(ls)

power_per_l = spectrum(coeffs)

# print(np.round_(power_per_l, decimals=2))
# print(type(power_per_l))
# fig, ax = plt.subplots(1, 1, figsize=(len(ls), 5))
# ax.plot(ls, power_per_l, 'bo')
# ax.plot(ls, power_per_l, 'k-')
# plt.xticks(range(len(ls)))
# ax.set_yscale('log')
# ax.set_xscale('log')
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