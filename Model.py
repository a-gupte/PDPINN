import abc
import torch
from Problem import Problem
import matplotlib.pyplot as plt
import numpy as np
from pyshtools.spectralanalysis import spectrum
import my_shcoeffs
from my_shcoeffs import SHCoeffs
import my_backends 
from my_backends.ducc0_wrapper import *

N = 20

class Hs_loss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, target):
        
        u = input - target
        u = u.reshape([20, 40])
        n = u.shape[0]
        s = 0
        
        if not isinstance(u, np.ndarray):
            u = u.detach().numpy()
        
        clm = SHExpandDH(u, sampling=2)
        power_per_l = spectrum(clm)
        result = 0
        for i in range(len(power_per_l)):
            result += (1 + i*(i+1))**s * power_per_l[i]
        ctx.input = input 
        ctx.clm = clm
        ctx.result = result
        ctx.target = target
        ctx.s = s
        return torch.tensor(result)

    @staticmethod
    def backward(ctx, grad_output):
        target = ctx.target
        s = ctx.s
        clm = ctx.clm
        
        diag = np.diag([(1 + i*(i+1))**s for i in range(clm.shape[1])])
        clm_concat = np.hstack((clm[1][:,::-1], clm[0]))
        
#         x = SHCoeffs.from_array(clm)
#         grad_input = 2 * x.expand_adjoint_analysis()
        
        ## use makegrid instead
        grad_input = 2 * MakeGridDH_adjoint_analysis(clm, sampling = 2)   
        grad_input = grad_input.reshape([-1,1])
        return torch.tensor(grad_input), None


class Model(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self, problem: Problem, net: torch.nn.Module, maxiter: int = 1000):
        self.problem = problem
        self.net = net
        self.maxiter = maxiter

        # the following should be implemented
        self.opt = None
        self.pde_loss_f = None
        self.bc_loss_f = None
        self.init_loss_f = None
        self.loss_history = []

        self.set_optimizer()
        self.set_pde_loss_f()
        self.set_bc_loss_f()
        self.set_init_loss_f()

    def set_optimizer(self):
        self.opt = torch.optim.Adam(self.net.parameters(), lr=0.008)

    def set_pde_loss_f(self):
        self.pde_loss_f = torch.nn.MSELoss()
#         self.pde_loss_f = Hs_loss.apply

    def set_bc_loss_f(self):
        self.bc_loss_f = torch.nn.MSELoss()
#         self.bc_loss_f = Hs_loss.apply

    def set_init_loss_f(self):
        self.init_loss_f = torch.nn.MSELoss()
        # self.init_loss_f = Hs_loss.apply

    def train_epoch(self):
        pass

    def train(self):
        problem = self.problem
        net = self.net
        opt = self.opt
        _, axe = plt.subplots(1, 10, figsize=(50, 5))
        maxiter = self.maxiter

        for iter in range(maxiter):
            net.zero_grad()

            coor_inner = self.inner_sample().detach().requires_grad_(True)
            infer_value_inner = net(coor_inner)
            truth_inner, predict_inner = problem.pde(coor_inner, infer_value_inner)
            self.pde_loss = self.pde_loss_f(predict_inner, truth_inner)

            bc_samples = self.bc_sample()
            if bc_samples is None:
                self.bc_loss = torch.tensor(0.)
            else:
                coor_bc = bc_samples.detach().requires_grad_(True)
                infer_value_bc = net(coor_bc)
                truth_bc, predict_bc = problem.bound_condition(coor_bc, infer_value_bc)
                self.bc_loss = self.bc_loss_f(predict_bc, truth_bc)

            init_samples = self.init_sample()
            if init_samples is None:
                self.init_loss = torch.tensor(0.)
            else:
                coor_init = init_samples.detach().requires_grad_(True)
                infer_value_init = net(coor_init)
                truth_init, predict_init = problem.bound_condition(coor_init, infer_value_init)
                self.init_loss = self.bc_loss_f(predict_init, truth_init)
            self.predict_error_value = self.predict_error()
            self.total_loss = self.pde_loss #+ self.bc_loss + self.init_loss
            self.add_loss_history()
            self.total_loss.backward()
            opt.step()
            opt.zero_grad()
            if iter % (maxiter // 100) == 0:
                print("iteration {}: loss = {}".format(iter, self.total_loss))
            if iter > 0 and (iter % (maxiter // 10) == 0):
                # torch.save(net, 'cb_net_{}_order7_03121711'.format(iter))
                self.plot(net, axe[iter // (maxiter // 10)])
        if self.problem.ground_truth:
            self.plot(self.problem.ground_truth, axe[0])
        # coor = sphere_surface_sample(5000)
        # plot_sphere(net, coor)
        plt.show()

    def predict_error(self):
        return None

    def add_loss_history(self, ):
        
        self.loss_history.append(
            [self.total_loss.item(), self.pde_loss.item(), self.bc_loss.item(), self.init_loss.item()])

    def post_process(self):
        plt.plot(self.loss_history)
        plt.yscale('log')
        plt.legend(('total loss', 'pde loss', 'BC loss', 'IC loss'))
        plt.show()

    def __str__(self):
        pass

    def save(self, filename):
        torch.save(self.net.state_dict(), filename)

    def load(self, filename):
        self.net.load_state_dict(torch.load(filename))
        self.net.eval()

    @abc.abstractmethod
    def inner_sample(self, sampling_method):
        pass

    @abc.abstractmethod
    def bc_sample(self):
        pass

    @abc.abstractmethod
    def init_sample(self):
        pass

    @abc.abstractmethod
    def plot(self, net, ax):
        pass
