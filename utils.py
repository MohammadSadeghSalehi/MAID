import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import numpy as np
import os
from torchvision.transforms import ToTensor, Lambda
from Dynamic_HOAG import HOAG

# Plotting curves of loss
def plot_fun(x, y, xlabel, ylabel,label,color,marker,linestyle):
    plt.plot(x, y, label = label, color = color, marker = marker, linestyle = linestyle, markersize = 2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.yscale('log')
    plt.xscale('log')

colors = ['r','b','g','y']
markers = ['o','x','*', 's']
linetypes = ['-',':','-.','--']

one_dim = True
visualize = False # Shows noisy image and denoised image if True

if one_dim:
    #1D image generating 
    N = 256
    d = 5     # number of data points
    X = torch.zeros(d,N)
    for i in range(d):
        c = N/4 +torch.rand(1)*3*N/4
        r = N/8 +torch.rand(1)*N/4
        for j in range(N):
            X[i,j] = 1 if torch.abs(j-c)< r else 0
    NoisyData = torch.zeros(d,N)
    for i in range(d):
        NoisyData[i,:] = X[i,:] + torch.randn(N)*0.1
    if visualize:
        model = HOAG(torch.eye(X.shape[1]),X,NoisyData , max_UpperLevel_iter = 5, max_LowerLevel_iter=100, eps = 1e-1,acc_type='dynamic', backtrack = False, eps_decay = 0.9, step_size = 0.01)
        model.fixed_tol = False
        theta = model.solver( X, NoisyData)
        loss_fixed_tol = torch.stack(model.loss).detach().numpy()
        LL_cost_dynamic = model.LL_calls_list
        print("Epsilon   ", model.eps)
        x = model.FISTA(theta, X[1,:], NoisyData[1,:],tol = 1e-7)
        plt.plot(x.detach().numpy())
        plt.plot(X[1,:].detach().numpy())
        plt.show()

if not one_dim:
    #MNIST
    batch_size = 3
    trainset    = datasets.MNIST('train', download = True, train = True, transform=ToTensor(),target_transform=Lambda(lambda X: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(X), value=1)))
    trainset = torch.utils.data.Subset(trainset, list(range(0,1000)))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size =batch_size, shuffle = True)
    dataiter = iter(trainloader)
    X, y = dataiter.next()
    X = X.squeeze()
    X = X.reshape(X.shape[0],X.shape[1]*X.shape[2])
    # Add noise to the data
    noise = (torch.randn(X.shape,) * 0.1)
    Y = X + noise
    NoisyData = Y
    if visualize:
        plt.imshow(X[1,:].reshape(28,28).cpu().numpy(),cmap = "gray")
        plt.show()

        print("MSE: ", torch.linalg.norm((X[1,:].reshape(28,28))-(Y[1,:].reshape(28,28)))**2)
        plt.imshow(Y[1,:].reshape(28,28).cpu().numpy(),cmap = "gray")
        plt.show()
        model = HOAG(torch.eye(X.shape[1]),X,Y , max_UpperLevel_iter = 200, max_LowerLevel_iter=300, eps = 1e-1,acc_type='dynamic', backtrack = True, eps_decay = 0.9, step_size = 0.01)
        model.solver( X, Y)
        denoised = model.FISTA(model.theta, X[1,:], Y[1,:], tol = 1e-8)
        print("PSNR: ", 10*torch.log10(1/(torch.mean((X[1,:]-denoised)**2))))
        plt.imshow(denoised.reshape(28,28).detach().numpy(),cmap = "gray")
        plt.show()

# Run over different accuracy types and step_size update types
for index, acc in enumerate(['dynamic','HOAG_bactrack','fixed_acc_backtrack', 'HOAG']):
    model = HOAG(torch.eye(X.shape[1]),X,NoisyData , max_UpperLevel_iter = 100, max_LowerLevel_iter= 300, eps = 9*1e-1,acc_type=acc, backtrack = True, eps_decay = 0.9, step_size = 0.01)
    if acc in ['dynamic', 'fixed_acc_backtrack']:
        model.max_UL_iter = 120
    if acc == 'HOAG':
        model.backtrack = False
    theta = model.solver( X, NoisyData)
    loss = torch.stack(model.loss).detach().numpy()
    LL_cost = model.LL_calls_list
    plot_fun(LL_cost, loss, "LL calls", "Loss", f"{acc}", colors[index], markers[index],linetypes[index])

# Uncomment below to run without backtracking 

# model = HOAG(torch.eye(X.shape[1]),X,NoisyData , max_UpperLevel_iter = 40, max_LowerLevel_iter= 100, eps = 1e-1,acc_type='dynamic', backtrack = False, eps_decay = 0.9, step_size = 0.02)
# model.backtrack = False
# model.acc_type = acc
# theta = model.solver( X, NoisyData)
# loss = torch.stack(model.loss).detach().numpy()
# LL_cost = model.LL_calls_list
# plot_fun(LL_cost, loss, "LL calls", "Loss", 'HOAG_no_backtrack', 'y', 's')

plt.savefig('BactrackingVSHOAG1D.png', dpi=300)
plt.show()