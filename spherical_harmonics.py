import torch
import abc
from math import pi
from math import sqrt
import numpy as np

## return the real spherical harmonics in terms of theta, phi for spherical harmonic for l=0, 1, ... and m = -l, -l+1, ..., l-1, l
# theta = data[:, :1] azimuthal angle
# phi = data[:, 1:] polar angle

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

def Y_8_0(theta, phi):
    x, y, z = spherical_to_cartesian(theta, phi)
    return 1.0/256 * sqrt(17.0/pi) * (6435 * z ** 8 - 12012 * z ** 6 + 6930 * z ** 4 - 1260 * z ** 2 + 35)

#---------------------- PAPER --------------------------
def rhs_function_paper(theta, phi):
    m = 7
    return (-(m + 1) * (m + 2) * torch.cos(theta) * (torch.sin(theta) ** (m)) * torch.cos(m * phi - 0.0 * m))- (-(m) * (m + 1) * torch.cos(theta) * (torch.sin(theta) ** (m - 1)) * torch.cos((m - 1) * phi - 0.0 * m))
    
def true_solution_paper(theta, phi):
    m = 7    
    return (torch.cos(theta) * (torch.sin(theta) ** m) * torch.cos(m * phi)) - (torch.cos(theta) * (torch.sin(theta) ** (m - 1)) * torch.cos((m - 1) * phi))

#---------------------- ONLY LOW FREQ --------------------------
def rhs_function_low_freq(theta, phi):
    return 6 * Y_2_2(theta, phi)

def true_solution_low_freq(theta, phi):
    return - Y_2_2(theta, phi)

#---------------------- FREQs 2 and 4 --------------------------
def rhs_function_freq_2_4(theta, phi):
    return 6 * Y_2_2(theta, phi) + 20 * Y_4_2(theta, phi)

def true_solution_freq_2_4(theta, phi):
    return - Y_2_2(theta, phi) - Y_4_2(theta, phi)    

#---------------------- LOW FREQ + HIGH FREQ --------------------------
def rhs_function_low_high_freq(theta, phi):
    m = 7
    n = 2
    return (-(m + 1) * (m + 2) * torch.cos(theta) * (torch.sin(theta) ** (m)) * torch.cos(m * phi - 0.0 * m))- (-(n) * (n + 1) * torch.cos(theta) * (torch.sin(theta) ** (n - 1)) * torch.cos((n - 1) * phi - 0.0 * n))

def true_solution_low_high_freq(theta, phi):
    m = 7
    n = 2
    return (torch.cos(theta) * (torch.sin(theta) ** m) * torch.cos(m * phi)) - (torch.cos(theta) * (torch.sin(theta) ** (n - 1)) * torch.cos((n - 1) * phi))

#---------------------- FREQs 2 and 8 -------------------------- 
def rhs_function_freq_2_8(theta, phi):
    return 72 * Y_8_0(theta, phi) + 6 * Y_2_2(theta, phi)

def true_solution_freq_2_8(theta, phi):
    return - Y_8_0(theta, phi) - Y_2_2(theta, phi)

#-----------------------

def rhs_function_freq_2_8_03_03(theta, phi):
    return 0.3 * 72 * Y_8_0(theta, phi) + 0.3 * 6 * Y_2_2(theta, phi)

def true_solution_freq_2_8_03_03(theta, phi):
    return - 0.3 * Y_8_0(theta, phi) - 0.3 * Y_2_2(theta, phi)

#-----------------------

def rhs_function_freq_2_8_baseline(theta, phi):
    return 0.6 * 72 * Y_8_0(theta, phi) + 6 * Y_2_2(theta, phi)

def true_solution_freq_2_8_baseline(theta, phi):
    return - 0.6 * Y_8_0(theta, phi) - Y_2_2(theta, phi)

#---------------------- MODIFY CODE BELOW TO CHANGE rhs_function & true_solution -------------------------- 
n1 = 8
m1 = 0

n2 = 2
m2 = 2

def rhs_function(theta, phi):
    return rhs_function_freq_2_4(theta, phi)
    # return - n1 * (n1 + 1) * sph_har(theta, phi, n1, m1) - n2 * (n2 + 1) * sph_har(theta, phi, n2, m2)

def true_solution(theta, phi):
    return true_solution_freq_2_4(theta, phi)
    # theta = theta.clip(0, pi - 0.0001)
    # phi = phi.clip(0, 2 * pi - 0.0001)
    # return sph_har(theta, phi, n1, m1) + sph_har(theta, phi, n2, m2)

