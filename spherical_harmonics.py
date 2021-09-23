import torch
import abc
from math import pi
from math import sqrt

## return the real spherical harmonics in terms of theta, phi for spherical harmonic for l=0, 1, ... and m = -l, -l+1, ..., l-1, l
def spherical_to_cartesian(theta, phi):
    # x = r sin(theta) cos(phi)
    # y = r sin(theta) sin(phi)
    # z = r cos(theta)
    return torch.sin(theta) *  torch.cos(phi), torch.sin(theta) * torch.sin(phi), torch.cos(theta)

def Y_0_0(theta, phi):
    return 1.0/2/sqrt(pi)

def Y_1__1(theta, phi):
    x, y, z = spherical_to_cartesian(theta, phi)
    return sqrt(3.0/4/pi) * y

def Y_1_0(theta, phi):
    x, y, z = spherical_to_cartesian(theta, phi)
    return sqrt(3/4.0/pi) * z

def Y_1_1(theta, phi):
    x, y, z = spherical_to_cartesian(theta, phi)
    return sqrt(3/4.0/pi) * x

def Y_2__2(theta, phi):
    x, y, z = spherical_to_cartesian(theta, phi)
    return sqrt(15.0/4/pi) * x*y

def Y_2__1(theta, phi):
    x, y, z = spherical_to_cartesian(theta, phi)
    return sqrt(15.0/4/pi) * z*y

def Y_2_0(theta, phi):
    x, y, z = spherical_to_cartesian(theta, phi)
    return sqrt(5.0/16/pi) * (-x**2 - y**2 + 2*z**2)

def Y_2_1(theta, phi):
    x, y, z = spherical_to_cartesian(theta, phi)
    return sqrt(15.0/4/pi) * z * x

def Y_2_2(theta, phi):
    x, y, z = spherical_to_cartesian(theta, phi)
    return sqrt(15.0/4/pi) * (x**2 - y**2)

def Y_3_0(theta, phi):
    x, y, z = spherical_to_cartesian(theta, phi)
    return sqrt(7.0/16/pi) * z * (2*z**2 - 3*x**2 - 3*y**2)

def true_solution(theta, phi):
    return - Y_2_2(theta, phi)

def rhs_function(theta, phi):
    return 6 * Y_2_2(theta, phi)
