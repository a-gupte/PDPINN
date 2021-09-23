import abc
import torch
from Problem import Problem
import matplotlib.pyplot as plt
import numpy as np

# class HsLoss(torch.nn.Module):
#     def __init__(self, weight=None, size_average=True):
#         super(HsLoss, self).__init__()
 
#     def forward(self, y_pred, y_true):        
#         u = y_pred - y_true 
#         n = u.shape[0]
#         s = 1
        
#         dft_matrix = np.fft.fft(np.eye(n))
#         inverse_dft_matrix = np.fft.ifft(np.eye(n))
#         hs_weight_matrix = np.diag([(1 + i ** 2)**(s/2) for i in range(n)])
#         result = np.matmul(inverse_dft_matrix, np.matmul(hs_weight_matrix, dft_matrix))
#         hermitian_adjoint = np.matrix(result)
#         P = np.array(np.matmul(hermitian_adjoint.getH(), result))
    
#         ## # SciPy's L-BFGS-B Fortran implementation requires and returns float64
#         P = torch.tensor(P, requires_grad=False)
#         P = P.to(torch.float32)
#         hs_loss = 1.0/n * torch.matmul(torch.transpose(u,0,1), torch.matmul(P, u))
#         return hs_loss

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
        self.opt = torch.optim.Adam(self.net.parameters(), lr=0.001)

    def set_pde_loss_f(self):
        self.pde_loss_f = torch.nn.MSELoss()
#         self.pde_loss_f = HsLoss()

    def set_bc_loss_f(self):
        self.bc_loss_f = torch.nn.MSELoss()
#         self.bc_loss_f = HsLoss()

    def set_init_loss_f(self):
        self.init_loss_f = torch.nn.MSELoss()
#         self.init_loss_f = HsLoss()

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
            self.total_loss = self.pde_loss + self.bc_loss + self.init_loss
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
    def inner_sample(self):
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
