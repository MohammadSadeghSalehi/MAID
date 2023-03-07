import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import time
from torchvision import datasets, transforms
from torch import nn, optim
from torch.autograd import Variable
from torch.autograd.functional import hessian
from torch.autograd.functional import jacobian
import gc
    
torch.manual_seed(0)
torch.random.manual_seed(0)
class HOAG:
    def __init__(self, A, x, y, max_UpperLevel_iter=200, max_LowerLevel_iter=100, eps = 1e-1, eps_decay = 0.9, acc_type = 'dynamic', backtrack = True, step_size = 1e-1):
        # A is the forward operator
        #b is the ground truth
        # y is the noisy image
        # x0 is the initial guess for low-level problem
        # L is the lipschitz constant
        # mu is the strong-convexity parameter
        self.A = A
        self.x = x
        self.y = y
        self.max_UL_iter = max_UpperLevel_iter
        self.max_LL_iter = max_LowerLevel_iter
        self.eps = eps
        self.eps_decay = eps_decay
        self.backtrack = backtrack
        self.step_size = step_size         # For fixed step size wihout backtracking
        self.theta = -torch.FloatTensor([0,0,-3])
        self.loss = []
        self.acc_type = acc_type
        self.LL_calls = 0
        self.LL_calls_list = []
        self.lip_cal = None
    def phi(self, x,y, theta):
        # The lower level objective function
        # For 1D signal, use TV rather than TV2D
        return 0.5*torch.linalg.norm(x-y)**2 +torch.exp(theta[0]) * self.TV2D(x, torch.exp(theta[1])) + 0.5 *torch.exp(theta[2]) * torch.linalg.norm(x)**2
    def TV(self,x, nu):
        return (torch.sum(torch.sqrt((x[1:]-x[:-1])**2+ nu**2)))
    def TV2D(self,x, nu):
        x = torch.reshape(x,(int(np.sqrt(x.shape[0])),int(np.sqrt(x.shape[0]))))
        tv = 0
        tv += (torch.sum(torch.sqrt(torch.abs(x[:,1:]-x[:,:-1])**2+ nu**2)))
        tv += (torch.sum(torch.sqrt(torch.abs(x[1:,:]-x[:-1,:])**2+ nu**2)))
        return tv
    def gradPhi(self, x,y, theta):
        out = self.phi(x,y, theta)
        grad_x = torch.autograd.grad(outputs=out, inputs=x, grad_outputs=None, create_graph=False, retain_graph=False, only_inputs=True, allow_unused=True)[0]
        return grad_x
    def hessianPhi(self, x,y, theta,d):
        x.requires_grad_(True)
        out = self.phi(x,y,theta)
        grad_x = torch.autograd.grad(outputs=out, inputs=x, grad_outputs=None, create_graph=True, retain_graph=True, only_inputs=True, allow_unused=True)[0] #first get grad using autograd
        hvp = torch.autograd.grad(outputs=grad_x, inputs=x, grad_outputs=d, create_graph=True, retain_graph=True, only_inputs=True, allow_unused=True)[0]
        return hvp.detach()

    def jacobianPhi(self, x,y,theta,d):
        x.requires_grad_(True)
        theta.requires_grad_(True)
        out = self.phi(x,y,theta)
        grad_x = torch.autograd.grad(outputs=out, inputs=x, grad_outputs=None, create_graph=True, retain_graph=True, only_inputs=True, allow_unused=True)[0] #first get grad using autograd
        gradvp = torch.dot(grad_x,d)
        jvp = torch.autograd.grad(outputs=gradvp, inputs=theta, grad_outputs=torch.ones(gradvp.shape).requires_grad_(True), create_graph=True, retain_graph=True, only_inputs=True, allow_unused=True)[0]
        return jvp.detach()
    def jacobiantransPhi(self, x,y,theta,d):
        # transpose of jacobian
        x.requires_grad_(True)
        theta.requires_grad_(True)
        out = self.phi(x,y,theta)
        grad_x = torch.autograd.grad(outputs=out, inputs=theta, grad_outputs=None, create_graph=True, retain_graph=True, only_inputs=True, allow_unused=True)[0] #first get grad using autograd
        gradvp = torch.tensordot(grad_x,d, dims=1)
        jvp = torch.autograd.grad(outputs=gradvp, inputs=x, grad_outputs=torch.ones(gradvp.shape).requires_grad_(True), create_graph=True, retain_graph=True, only_inputs=True, allow_unused=True)[0]
        return jvp
    def CG(self,x,y,b,theta,tol):
        r = b
        p = r
        rsold = float(torch.linalg.norm(r)**2)
        solution = torch.zeros(x.shape)
        while torch.linalg.norm(r)>tol:
            Ap = self.hessianPhi(x,y,theta,p)
            alpha = rsold/torch.tensordot(p,Ap, dims=1)
            solution = solution + alpha*p
            r = r - alpha*Ap
            rsnew = torch.linalg.norm(r)**2
            if torch.sqrt(rsnew) < tol:
                return solution
            p = r + (rsnew/rsold)*p
            rsold = rsnew
        return solution
    def FISTA(self,theta, x, y, tol):
        t = 0
        for k in range(self.max_LL_iter):
            x_old = x
            L = 1+ (torch.exp(theta[0])/torch.exp(theta[1]))* 8+torch.exp(theta[2])
            mu = 1+torch.exp(theta[2])
            tau = 1/L
            q = tau*mu
            t_old = t
            t = (1-q*t**2+torch.sqrt((1-q*t**2)**2+4*t**2))/2
            beta = ((t_old-1)*(1-t*q))/(t*(1-q))
            z = (x + beta * (x-x_old))
            p = torch.autograd.grad(outputs=self.phi(z,y,theta), inputs=z, grad_outputs=None, create_graph=False, retain_graph=False, only_inputs=True, allow_unused=True)[0]
            self.LL_calls += 1
            if torch.linalg.norm(p)/mu < tol:
                x = z -  tau*p
                return x
            x = z -  tau*p
        return x
    def backtracking(self,p_k, x, y,x_true, theta, rho = 0.3, alpha = 0.1, beta = 0.9,gamma = 0.9, max_iter = 100):
        for k in range(max_iter):
            x_old = x
            g_theta_k_old = self.loss[-1]
            theta_new = theta - rho*p_k
            x_new = self.LL( x, y,theta_new, eps=self.eps)
            x = x_new
            g_theta_k = self.UL(x_new,x_true)
            if g_theta_k - 2*(torch.linalg.norm(x_new - x_true)) *self.eps - 2*self.eps**2 <= g_theta_k_old + 2*(torch.linalg.norm(x_old - x_true)) *self.eps + 2* self.eps**2 - alpha*rho*torch.linalg.norm(p_k)**2 and \
                g_theta_k_old - 2*(torch.linalg.norm(x_old - x_true)) *self.eps - 2*self.eps**2 <= g_theta_k + 2*(torch.linalg.norm(x_new - x_true)) *self.eps + 2* self.eps**2 + gamma *rho*torch.linalg.norm(p_k)**2:
                print("backtracking iteration = ", k, "backtracking step size = ", rho)
                print ("g = ", g_theta_k, "g_old = ", g_theta_k_old, "norm_P_k = ", torch.linalg.norm(p_k))
                self.loss.append(g_theta_k)
                return theta_new , x_new
            rho = beta*rho
        self.loss.append(g_theta_k)
        print("backtracking rho = ", rho, "Reached max iterations")
        return theta_new , x_new
    def power_method_jac(self, x, y, theta, jac):
        # one iteration of power method for jacobian
        v = self.jacobiantransPhi(x,y,theta,jac)
        v = v/torch.linalg.norm(v)
        return torch.max(torch.abs(v))
    def power_method_hes(self, hess):
        # one iteration of power method for inverse of hessian
        return torch.max(torch.abs(hess/torch.linalg.norm(hess)))
    def bound(self, x,y, theta,jac,hess):
        mu = 1+torch.exp(theta[2])
        Bx = torch.linalg.norm(jac)/torch.linalg.norm(hess)
        if self.lip_cal is None:
            self.LB = self.power_method_jac(x,y,theta,jac)
            self.LAinv = self.power_method_hes(hess)
            self.lip_cal = True
        c = Bx/mu+(self.LAinv)*torch.linalg.norm(2*(x-y))*Bx+ self.LB*torch.linalg.norm(2*(x-y))/mu
        return c*self.eps+ Bx/mu*self.eps + (self.LB/mu)*self.eps**2
    def LL(self, x0,y, theta, eps):
        for i in range (x0.shape[0]):
            x0[i,:] = self.FISTA(theta, x0[i,:].requires_grad_(True), y[i,:], tol = eps)
        return x0

    def UL(self, x_hat, x):
        return torch.mean(torch.linalg.norm(x_hat-x,dim = 1)**2)
    def p_k(self, x_hat, x, y, theta):
        # calculate the inexact gradient of the UL (p_k) and the bound of error C = \|e_k\|
        p_k = []
        C = 0
        for i in range (x.shape[0]):
            q = self.CG(x_hat[i,:],y[i,:],2*(x_hat[i,:]- x[i,:]),theta,self.eps)
            p_k.append(-self.jacobianPhi(x_hat[i,:],y[i,:],theta,q))
            if self.acc_type == 'dynamic':
                C+= self.bound(x_hat[i,:],x[i,:],theta,-p_k[-1],q)/(x.shape[0])
        p_k = torch.mean(torch.stack(p_k),dim=0)
        if self.acc_type == 'dynamic':
            while C/torch.linalg.norm(p_k) > 0.9:
                # self.update_accuracy()
                self.eps = self.eps * self.eps_decay
                p_k , C = self.p_k(x_hat, x, y, theta)
                C = C* torch.linalg.norm(p_k)
            return p_k, C/torch.linalg.norm(p_k)
        elif self.acc_type == 'fixed_acc_backtrack':
            return p_k, "N/A"
        elif self.acc_type == 'HOAG':
            return p_k, "N/A"
        # For fixed accuracy without backtracking, we do the following
        self.update_accuracy()
        return p_k, "N/A"
    def update_accuracy(self):
        #max accuracy is 1e-6
        if self.eps > 1e-6:
            self.eps = self.eps * self.eps_decay
        return
        
    def solver(self,x, y):
        x = self.x
        # Initialize x0
        x0 = torch.randn(x.shape)
        theta = self.theta
        x_hat = self.LL(x0,y,theta,self.eps)
        self.loss.append(self.UL(x_hat,x))
        self.LL_calls_list.append(self.LL_calls)
        for k in range(self.max_UL_iter):
            p_k, C = self.p_k(x_hat, x,y , theta)
            print("bound" , C, "eps ", self.eps)
            print(f'iter = {k}',f"p = {p_k}", f'loss = {self.UL(x_hat,x)}')
            # Backtracking 
            if self.backtrack == True:
                theta, x_hat = self.backtracking(p_k, x_hat, y,x, theta)
            # Adaptive step size of Pedregosa 
            elif self.acc_type == 'HOAG':
                if k == 0: 
                    step_size = 1/torch.linalg.norm(p_k) 
                    L_lambda = torch.linalg.norm(p_k)/p_k.shape[0]
                x_old = x_hat
                incr = torch.linalg.norm(step_size * p_k)
                C = 0.25
                factor_L_lambda = 1.0
                theta_new = theta - step_size*p_k
                old_epsilon_tol = self.eps
                self.update_accuracy()
                epsilon_tol = self.eps
                x_new = self.LL(x_hat,y,theta_new,self.eps)
                g_func_old = self.loss[-1]
                g_func = self.UL(x_new,x)
                if g_func <= g_func_old + C * epsilon_tol + \
                            old_epsilon_tol * (C + factor_L_lambda) * incr - factor_L_lambda * (L_lambda) * incr * incr:
                    L_lambda *= 0.95
                    theta = theta - step_size * p_k
                    self.loss.append(g_func)
                elif g_func >= 1.2 * g_func_old:
                    # decrease step size
                    L_lambda *= 2
                    print('!!step size rejected!!', g_func, g_func_old)
                    # tighten tolerance
                    self.eps *= 0.5
                    self.loss.append(g_func_old)
                else:
                    theta = theta - step_size * theta
                    self.loss.append(g_func)
            # Fixed step size without backtracking
            else:
                theta = theta - self.step_size*p_k
                self.update_accuracy()
                x_hat = self.LL(x_hat,y,theta,self.eps)
                self.loss.append(self.UL(x_hat,x))
            self.theta = theta
            self.LL_calls_list.append(self.LL_calls)
        return theta
    




    
