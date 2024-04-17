from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os
import gc
import torch
import torchvision
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import torch.nn.init as init
from torchvision import datasets, transforms
from torchvision import transforms
from Utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class MAID(nn.Module):
    def __init__(self,theta, x0, lower_level_obj, upper_level_obj,LL_solver,Lg,epsilon = 1e-1, delta = 1e-1, eta = 1e-4, rho = 0.9, tau = 0.5, max_iter = 100, beta = 0.1,save_postfix = "",psnr_log =False, jac = None, jact = None) -> None:
        super().__init__()
        self.theta = theta
        self.x = x0
        self.lower_level_obj = lower_level_obj
        self.upper_level_obj = upper_level_obj
        self.epsilon = epsilon
        self.beta = beta
        self.delta = delta
        self.eta = eta
        self.rho = rho
        self.tau = tau
        self.mu = None
        self.max_iter = max_iter
        self.LL_solver = LL_solver
        self.LL_jacobian = jac
        self.LL_jacobiantrans = jact
        self.LL_iter = 2000
        self.Lg = Lg
        self.LAinv = 0
        self.LB = 0
        self.lip_cal = None
        self.iter= 0
        self.ll_calls = 0
        self.CG_calls = 0
        self.save_postfix = save_postfix
        self.fixed_step = False
        self.fixed_eps = False
        self.max_ll_reached = False
        self.bt_flag = False
        self.psnr_log = psnr_log
        self.ll_budget = 1e6
        self.max_bt_loop = 3
        self.nu = 1.05
        self.loss = []
        self.state = {
            "loss": [],
            "LL_calls": [],
            "eps": [],
            "delta": [],
            "error": [],
            "step_size": [],
            "grads": [],
            "CG_calls": [],
            "params": [],
            "x": None
        }
    def call_LL_solver(self, *args, **kwargs):
        # Call the function passed to the class instance
        return self.LL_solver(self,*args, **kwargs)
    def jacobian(self, *args, **kwargs):
        return self.LL_jacobian(self,*args, **kwargs)
    def jacobiantrans(self, *args, **kwargs):
        return self.LL_jacobiantrans(self,*args, **kwargs)
    def grad_lower(self, x):
        x.to(device).requires_grad_(True)
        out = self.lower_level_obj(x,self.theta).to(device)
        return torch.autograd.grad(outputs=out, inputs=x, grad_outputs=None, create_graph=False, retain_graph=False, only_inputs=True, allow_unused=True)[0]
    def grad_upper(self, x_tilda):
        upper_grad =  (x_tilda - self.upper_level_obj.x_true)/x_tilda.shape[0]
        x_tilda.detach().cpu()
        return upper_grad.detach()
    def hessian(self, x,d):
        x.to(device).requires_grad_(True)
        out = self.lower_level_obj(x,self.theta).to(device)
        grad_x = torch.autograd.grad(outputs=out, inputs=x, grad_outputs=None, create_graph=True, retain_graph=True, only_inputs=True, allow_unused=True)[0] #first get grad using autograd
        hvp = torch.autograd.grad(outputs=grad_x, inputs=x, grad_outputs=d, create_graph=False, retain_graph=False, only_inputs=True, allow_unused=False)[0]
        return hvp.detach()
    def CG(self, x, b, tol):
        x = x.to(device)  # Move x tensor to GPU
        b = b.to(device)  # Move b tensor to GPU
        r = b
        p = r
        rsold = (torch.linalg.norm(r)**2).double()
        solution = torch.zeros(x.shape, device=device)
        if torch.linalg.norm(r) <= tol:
            solution = x.clone().detach()
        k = 0
        while torch.linalg.norm(r) > tol and k < 2000:
            k += 1
            Ap = self.hessian(x, p)
            alpha = rsold.double() / torch.dot(p.flatten().double(), Ap.flatten().double())
            solution = solution + alpha * p
            r = r - alpha * Ap
            rsnew = torch.linalg.norm(r)**2
            if torch.sqrt(rsnew) < tol:
                self.CG_calls += k
                return solution
            p = r + (rsnew / rsold) * p
            rsold = rsnew
        self.CG_calls += k
        return solution.detach()

    def power_method_jac(self, x, theta, jac):
        # one iteration of power method for jacobian
        v = self.jacobiantrans(x,theta,jac)
        # v = v/torch.linalg.norm(v)
        return torch.max(torch.abs(v))
        # return (torch.abs(torch.dot(v.flatten(),x.flatten()))/torch.linalg.norm(x)**2).detach()
    def power_method_hes(self, hess,x):
        # one iteration of power method for inverse of hessian
        return torch.max(torch.abs(hess/torch.linalg.norm(hess)))
        # return (torch.abs(torch.dot(hess.flatten(),x.flatten()))/torch.linalg.norm(x)**2).detach()

    def LB_approx(self, hess, jac , x, theta):
        perturb = torch.randn(x.shape).to(device)
        rhs = hess
        jac_perturb = -self.jacobian(perturb,theta, rhs)
        return torch.linalg.norm(jac_perturb - jac)/torch.linalg.norm(rhs)
    
    def LA_inv_approx(self, hess, x, theta):
        perturb = torch.randn(x.shape).to(device)
        rhs = self.grad_upper(x)
        hess_perturb = self.CG(perturb, rhs, self.delta)
        return torch.linalg.norm(hess_perturb - hess)/torch.linalg.norm(rhs)
        
    def bound(self, x, theta,jac,hess, epsilon):
        mu = self.mu
        Bx = self.power_method_jac(x,theta,jac)
        if self.LB is None or self.LAinv is None:
            self.LB = self.LB_approx(hess, jac, x, theta)
            self.LAinv = self.LA_inv_approx(hess, x, theta)
        else:
            self.LB = max(self.LB, self.LB_approx(hess, jac, x, theta))
            self.LAinv = max(self.LAinv, self.LA_inv_approx(hess, x, theta))
        c = Bx/mu+(self.LAinv)*torch.linalg.norm(self.grad_upper(x))*Bx+ self.LB*torch.linalg.norm(self.grad_upper(x))/mu
        return c*epsilon+ (Bx/mu)*self.delta + (self.LB/mu)*epsilon**2

    def inexact_gradient(self, x, theta, epsilon, delta, tau):
        iter = 0
        while True:
            x_tilda = self.call_LL_solver(theta, x, epsilon, self.LL_iter)
            q = self.CG(x_tilda, self.grad_upper(x_tilda), self.delta)
            p = -self.jacobian(x_tilda, theta, q.float())
            omega = self.bound(x_tilda, theta, p , q,  epsilon)
            print("Bound : ", omega/ torch.linalg.norm(p))
            if (omega/ torch.linalg.norm(p))> 1-self.eta and (not self.fixed_eps):
                self.epsilon = tau * epsilon
                self.delta = tau * self.delta
                epsilon = self.epsilon
                delta = self.delta
            else:
                break
            iter += 1
            print("delta", delta, "epsilon", epsilon)
        self.x = x_tilda
        self.state['error'].append(omega)
        self.state['grads'].append(torch.linalg.norm(p))
        return p , epsilon, delta
    def backtrack(self, x, theta, p , epsilon, beta, rho, tau, max_iter = 10):
        for i in range(max_iter):
            g_old = self.loss[-1]
            theta_new = theta - (beta * rho**i) *(p)
            x_new = self.call_LL_solver(theta_new, x, epsilon, self.LL_iter)
            g = self.upper_level_obj(x_new)
            if g + torch.linalg.norm(self.grad_upper(x_new))* epsilon + self.Lg * epsilon**2/2 - g_old + torch.linalg.norm(self.grad_upper(x))* epsilon \
                <= -beta * rho**i * self.eta*(self.eta) * torch.linalg.norm(p)**2:
                if self.psnr_log:
                    print ('backtrack success', "iter = ", i, "loss = ", g, "psnr = ", psnr(x_new, self.upper_level_obj.x_true))
                else:
                    print ('backtrack success', "iter = ", i, "loss = ", g)
                self.state['step_size'].append(beta * rho**i)
                if not self.fixed_step:
                    self.beta = beta * rho**i
                self.loss.append(g.detach().cpu())
                self.bt_flag = True
                self.state['LL_calls'].append(self.ll_calls)
                self.state["CG_calls"].append(self.CG_calls)

                return x_new.detach(), theta_new.detach()
            if self.ll_calls > self.ll_budget:
                print("LL calls exceeded")
                self.max_ll_reached = True
                self.loss.append(self.loss[-1])
                self.state['LL_calls'].append(self.ll_calls)
                self.state["CG_calls"].append(self.CG_calls)
                self.bt_flag = False
                return x.detach(), theta.detach()
        print ('backtrack failed',"loss = ", g , "loss_old = ", g_old, "epsilon = ", epsilon, "delta = ", self.delta, "ll = ", self.ll_calls)
        self.bt_flag = False
        self.state['eps'].append(epsilon)
        self.state['delta'].append(self.delta)
        if self.fixed_eps:
            self.loss.append(self.loss[-1])
            self.state['LL_calls'].append(self.ll_calls)
            self.state["CG_calls"].append(self.CG_calls)
        return x.detach() , theta.detach()

    def main(self):
        epsilon = self.epsilon
        self.loss.append(self.upper_level_obj(self.x))
        print("Initial loss = ", self.loss[-1])
        self.state['LL_calls'].append(0)
        self.state["CG_calls"].append(self.CG_calls)
        self.state['eps'].append(epsilon)
        self.state['delta'].append(self.delta)
        x = self.x.clone().detach()
        for k in range(self.max_iter):
            print("iter = ", k)
            if self.max_ll_reached:
                    break
            if  not self.fixed_eps:
                for j in range(self.max_bt_loop):
                    p , epsilon, delta = self.inexact_gradient(self.x.requires_grad_(True), self.theta.requires_grad_(True), epsilon, self.delta, self.tau)
                    self.x, self.theta = self.backtrack(self.x.requires_grad_(True), self.theta.requires_grad_(True), p , epsilon, self.beta, self.rho, self.tau,max_iter= j+ 10)
                    if self.bt_flag:
                        epsilon *= self.nu
                        self.beta *= 10/9
                        self.delta *= self.nu
                        break
                    if self.max_ll_reached:
                        break
                    epsilon = self.tau *epsilon
                    self.delta = self.tau * self.delta
                self.epsilon = epsilon
            else:
                for j in range(self.max_bt_loop):
                    p , epsilon, delta = self.inexact_gradient(self.x.requires_grad_(True), self.theta.requires_grad_(True), epsilon, self.delta, self.tau)
                    self.x, self.theta = self.backtrack(self.x.requires_grad_(True), self.theta.requires_grad_(True), p , epsilon, self.beta, self.rho, self.tau,max_iter= j+ 5)
                    if self.bt_flag:
                        self.beta *= 10/9
                        break
                    if self.max_ll_reached:
                        break
            epsilon = self.epsilon
            self.state['eps'].append(epsilon)
            self.state['delta'].append(self.delta)
            if k % 5 == 0 and k > 0:
                self.state['params'] = self.theta
                self.state['loss'] = self.loss
                self.state['x'] = self.x.detach().cpu()
                torch.save(self.state, f"state_dict_MAID_{self.save_postfix}.pt")
        return self.theta

