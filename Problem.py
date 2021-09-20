import torch
import abc
from math import pi
from math import sqrt


class Problem(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def __init__(self):
        super().__init__()
        self.ground_truth = None

    @abc.abstractmethod
    def __str__(self):
        pass

    @abc.abstractmethod
    def pde(self, xx, yy):
        pass

    @abc.abstractmethod
    def bound_condition(self, xx, yy):
        pass

    @abc.abstractmethod
    def init_condition(self, xx, yy):
        pass

    def set_groud_truth(self):
        pass


class Problem_1d_bodong(Problem):
    def __init__(self):
        super().__init__()
        self.groud_truth = None
        self.set_groud_truth()

    def __str__(self):
        return "dy_xx = -0.49 * sin(0.7 * x) - 2.25 * cos(1.5 * x)"

    def pde(self, xx, yy):
        dy_x = torch.autograd.grad(sum(yy[:, :]), xx, retain_graph=True, create_graph=True)[0]
        dy_xx = torch.autograd.grad(sum(dy_x[:, :]), xx, retain_graph=True, create_graph=True)[0]
        return -0.49 * torch.sin(0.7 * xx) - 2.25 * torch.cos(1.5 * xx), dy_xx

    def bound_condition(self, xx, yy):
        return self.ground_truth(xx), yy

    def init_condition(self, xx, yy):
        return torch.tensor(0.), torch.tensor(0.)

    def set_groud_truth(self):
        def fun(x):
            return torch.sin(0.7 * x) + torch.cos(1.5 * x) - 0.1 * x

        self.ground_truth = fun
        return fun


class Problem_1d_diffussion(Problem):
    def __init__(self):
        super().__init__()
        self.groud_truth = None
        self.set_groud_truth()

    def __str__(self):
        return "dy_xx = -0.49 * sin(0.7 * x) - 2.25 * cos(1.5 * x)"

    def pde(self, xx, yy):
        dy_x = torch.autograd.grad(sum(yy[:, :]), xx, retain_graph=True, create_graph=True)[0]
        dy_x, dy_t = dy_x[:, 0:1], dy_x[:, 1:]
        dy_xx = torch.autograd.grad(sum(dy_x[:, :]), xx, retain_graph=True, create_graph=True)[0][:, 0:1]
        return (
            -dy_t + dy_xx,
            xx[:, 1:]
            * (0.49 * torch.sin(0.7 * xx[:, 0:1]) + 2.25 * torch.cos(1.5 * xx[:, 0:1])) - torch.sin(
                0.7 * xx[:, 0:1]) - 3 * torch.cos(1.5 * xx[:, 0:1]) + 0.1 * xx[:, 0:1]
        )

    def bound_condition(self, xx, yy):
        return self.ground_truth(xx), yy

    def init_condition(self, xx, yy):
        return self.groud_truth(xx), yy

    def set_groud_truth(self):
        def fun(x):
            return (torch.sin(0.7 * x[:, 0:1]) + torch.cos(1.5 * x[:, 0:1]) - 0.1 * x[:, 0:1]) * x[:, 1:]

        self.ground_truth = fun
        return fun


class Problem_2d_Poisson(Problem):
    def __init__(self):
        super().__init__()
        self.groud_truth = None
        self.set_groud_truth()

    def __str__(self):
        return "dy_xx+d_y_yy =  -0.25 * torch.sin(0.5 * xx[:, 0:1]) - 0.49 * torch.sin(0.7 * xx[:, 1:])"

    def pde(self, xx, yy):
        dy_x = torch.autograd.grad(sum(yy[:, :]), xx, retain_graph=True, create_graph=True)[0]
        dy_x, dy_y = dy_x[:, 0:1], dy_x[:, 1:]
        dy_xx = torch.autograd.grad(sum(dy_x[:, :]), xx, retain_graph=True, create_graph=True)[0][:, 0:1]
        dy_yy = torch.autograd.grad(sum(dy_x[:, :]), xx, retain_graph=True, create_graph=True)[0][:, 1:]
        return (
            dy_yy + dy_xx,
            # -0.01*pi**2 * torch.sin(0.1 * xx[:, 0:1]*pi) - 0.04*pi**2 * torch.sin(0.2 * xx[:, 1:]*pi)
            # torch.zeros_like(dy_xx)
            -torch.sin((xx[:, 1:] + 10) / 20 * pi)
            * (0.49 * torch.sin(0.7 * xx[:, 0:1]) + 2.25 * torch.cos(1.5 * xx[:, 0:1]))
            - (torch.sin(0.7 * xx[:, 0:1]) + torch.cos(1.5 * xx[:, 0:1]) - 0.1 * xx[:, 0:1]) * torch.sin(
                (xx[:, 1:] + 10) / 20 * pi) * pi ** 2 / 400
        )

    def bound_condition(self, xx, yy):
        return self.ground_truth(xx), yy

    def init_condition(self, xx, yy):
        return torch.tensor([0.]), torch.tensor([0.])

    def set_groud_truth(self):
        def fun(x):
            return (torch.sin(0.7 * x[:, 0:1]) + torch.cos(1.5 * x[:, 0:1]) - 0.1 * x[:, 0:1]) * torch.sin(
                (x[:, 1:] + 10) / 20 * pi)

        self.ground_truth = fun
        return fun

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

class Problem_Sphere_Poisson(Problem):
    def __init__(self):
        super().__init__()
        self.groud_truth = None
        self.m = 7
        self.set_groud_truth()

    def __str__(self):
        return "dy_xx+d_y_yy =  -0.25 * torch.sin(0.5 * xx[:, 0:1]) - 0.49 * torch.sin(0.7 * xx[:, 1:])"

    def pde(self, x, y):
        # m = self.m
        # theta, phi = thetaphi[:, 0:1], thetaphi[:, 1:]
        # dy_dthetaphi = torch.autograd.grad(sum(y[:, :]), thetaphi, retain_graph=True, create_graph=True)[0]
        # dy_dtheta, dy_dphi = dy_dthetaphi[:, 0:1], dy_dthetaphi[:, 1:]
        # sintheta = torch.sin(theta)
        # dsintheta_dy_dtheta2 = torch.autograd.grad(sum(dy_dtheta[:, :] * sintheta), thetaphi, retain_graph=True, create_graph=True)[0][:, 0:1]
        # dy_dphi2 = torch.autograd.grad(sum(dy_dphi[:, :]), thetaphi, retain_graph=True, create_graph=True)[0][:, 1:]

        # lhs = dsintheta_dy_dtheta2/sintheta - dy_dphi2/sintheta**2
        # rhs = (-(m + 1) * (m + 2) * torch.cos(theta) * (torch.sin(theta) ** (m)) * torch.cos(\
        #     m * phi - 0.0 * m) - 2 * torch.cos(theta))\
        #     - (-(m) * (m + 1) * torch.cos(theta) * (torch.sin(theta) ** (m - 1)) * torch.cos(\
        #         (m - 1) * phi - 0.0 * m) - 2 * torch.cos(theta))
        # return lhs, rhs
        
        m = self.m
        dy_x = torch.autograd.grad(sum(y[:, :]), x, retain_graph=True, create_graph=True)[0]
        dy_x, dy_t = dy_x[:, 0:1], dy_x[:, 1:]
        sinx = torch.sin(x[:, 0:1])
        dy_xx = torch.autograd.grad(sum(dy_x[:, :] * sinx), x, retain_graph=True, create_graph=True)[0][:, 0:1]
        dy_tt = torch.autograd.grad(sum(dy_t[:, :]), x, retain_graph=True, create_graph=True)[0][:, 1:]
        return (
            dy_xx / sinx
            + dy_tt / sinx ** 2,
            (-(m + 1) * (m + 2) * torch.cos(x[:, :1]) * (torch.sin(x[:, :1]) ** (m)) * torch.cos(
                m * x[:, 1:] - 0.0 * m) - 2 * torch.cos(x[:, :1]))
            - (-(m) * (m + 1) * torch.cos(x[:, :1]) * (torch.sin(x[:, :1]) ** (m - 1)) * torch.cos(
                (m - 1) * x[:, 1:] - 0.0 * m) - 2 * torch.cos(x[:, :1]))
        )

    def bound_condition(self, xx, yy):
        return self.ground_truth(xx), yy

    def init_condition(self, xx, yy):
        return torch.tensor([0.]), torch.tensor([0.])

    def set_groud_truth(self):
        m = self.m

        def fun(x):
            return (torch.cos(x[:, :1]) * (torch.sin(x[:, :1]) ** m) * torch.cos(
                m * x[:, 1:]) + torch.cos(x[:, :1])) \
                   - (torch.cos(x[:, :1]) * (torch.sin(x[:, :1]) ** (m - 1)) * torch.cos(
                (m - 1) * x[:, 1:]) + torch.cos(x[:, :1]))

        self.ground_truth = fun
        return fun
