from MAID import *
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.random.manual_seed(29)
np.random.seed(29)

# Define the dimensions
n = 1000
m = 10

# Generate random matrices
A1 = torch.rand(n, m)
A2 = torch.rand(n, m)
A3 = torch.rand(n, m)

# Generate random vectors
x1 = torch.rand(m)
x2 = torch.rand(m)
theta_e = torch.rand(m)
y1 = torch.randn(n)
y2 = torch.randn(n)

# Calculate b1 and b2
b1 = torch.matmul(A1, x1) + 0.01 * y1
b2 = torch.matmul(A2, x2) + torch.matmul(A3, theta_e) + 0.01 * y2

class LowerLevelObjective(nn.Module):
    def __init__(self):
        super(LowerLevelObjective, self).__init__()
        self.A2 = nn.Parameter(A2)
        self.A3 = nn.Parameter(A3)
        self.b2 = nn.Parameter(b2)

    def forward(self, x, theta):
        return 0.5 *torch.linalg.norm(self.A2@x + self.A3 @ theta - self.b2)**2

class UpperLevelObjective(nn.Module):
    def __init__(self, lower_level_obj):
        super(UpperLevelObjective, self).__init__()
        self.A1 = nn.Parameter(A1)
        self.b1 = nn.Parameter(b1)
        self.lower_level_obj = lower_level_obj

    def forward(self, x):
        return 0.5 *torch.linalg.norm(self.A1@ x - self.b1)**2

lower_level_obj = LowerLevelObjective()
upper_level_obj = UpperLevelObjective(lower_level_obj)

def compute_x_star(theta):
    x_star = torch.inverse(A2.T @ A2) @ A2.T @ (b2 - (A3@ theta))
    return x_star

x_star = compute_x_star(torch.ones(m))
Lg =  torch.symeig(A1.T @ A1, eigenvectors=False)[0][-1]
temp = (-A1 @ torch.inverse(A2.T @ A2) @ A2.T @ A3).T @ (A1 @ (-torch.inverse(A2.T @ A2) @A2.T @A3))

optimal = torch.inverse(temp)@ (-(-A1 @ torch.inverse(A2.T @ A2) @ A2.T @ A3).T @ (A1@ torch.inverse(A2.T @ A2) @ A2.T @ (b2) - b1))
temp = A1@torch.inverse(A2.T @ A2) @ A2.T @ (-(A3))
x_star = compute_x_star(optimal)
Lf =  torch.symeig(temp.T @ temp, eigenvectors=False)[0][-1]

def jacobian(self, x,theta,d):
    d = d.to(device)
    x.to(device).requires_grad_(True)
    theta.to(device).requires_grad_(True).to(device)
    out = self.lower_level_obj(x,self.theta).to(device)
    grad_x = torch.autograd.grad(outputs=out, inputs=x, grad_outputs=None, create_graph=True, retain_graph=True, only_inputs=True, allow_unused=True)[0].to(device) #first get grad using autograd
    gradvp = torch.dot(grad_x.flatten(),d.flatten())
    jvp = torch.autograd.grad(outputs=gradvp, inputs=theta, grad_outputs=torch.ones(gradvp.shape).to(device).requires_grad_(True), create_graph=True, retain_graph=True, only_inputs=True, allow_unused=True)[0]
    return jvp.detach()
def jacobiantrans(self, x,theta,d):
    # transpose of jacobian
    x.requires_grad_(True)
    theta.requires_grad_(True)
    out = self.lower_level_obj(x,self.theta)
    grad_x = torch.autograd.grad(outputs=out, inputs=theta, grad_outputs=None, create_graph=True, retain_graph=True, only_inputs=True, allow_unused=True)[0] #first get grad using autograd
    gradvp = torch.tensordot(grad_x,d, dims=1)
    return torch.autograd.grad(outputs=gradvp, inputs=x, grad_outputs=torch.ones(gradvp.shape).requires_grad_(True), create_graph=True, retain_graph=True, only_inputs=True, allow_unused=True)[0]

mu =  torch.linalg.eigh(A2.T @ A2)[0][0]
Lphi =  torch.linalg.eigh(A2.T @ A2)[0][-1]  
def FISTA(self, theta, x, tol, max_iter):
        theta = theta.to(device) 
        x = x.to(device)
        t = 0
        for k in range(max_iter):
            x_old = x
            self.mu = mu
            L =  Lphi
            tau = 1 / L
            q = tau * mu
            t_old = t
            t = (1 - q * t**2 + np.sqrt((1 - q * t**2)**2 + 4 * t**2)) / 2
            beta = ((t_old - 1) * (1 - t * q)) / (t * (1 - q))
            z = (x + beta * (x - x_old)).to(device).requires_grad_(True)
            p = torch.autograd.grad(outputs=self.lower_level_obj(z, theta), inputs=z, grad_outputs=None, create_graph=False, retain_graph=False, only_inputs=True, allow_unused=True)[0]
            if torch.linalg.norm(p) / (2* mu) < tol:
                x = z - tau * p
                self.ll_calls += k
                return x
            x = z - tau * p
        self.ll_calls += k
        return x
    

theta = torch.ones(m, requires_grad=True)
x = torch.zeros(m, requires_grad=True)
x = x2
v = torch.randn(m)
gv = A1@torch.inverse(A2.T @ A2) @ A2.T @ (-(A3))
print(torch.max(torch.real(torch.linalg.eigvals(gv.T @ gv))))

for i in range(int(5e3)):
    x_star = compute_x_star(theta)
    temp = -A1 @ torch.inverse(A2.T @ A2) @ A2.T @ A3
    exact_hyper_gradient = (-A1 @ torch.inverse(A2.T @ A2) @ A2.T @ A3).T @ (A1@x_star - b1) 
    theta = theta - 1/Lf * exact_hyper_gradient
optimal = upper_level_obj(x_star)
x0 = torch.zeros(x2.shape)
x0 = x2

for index, postfix in enumerate([ "0303", "0505", "0101", "0303_fixed", "0505_fixed", "0101_fixed"]):
    if index == 1:
        model = MAID(torch.ones(m), x0, lower_level_obj, upper_level_obj,FISTA, Lg, max_iter=3000, epsilon=1e-5, delta=1e-5,rho = 0.5, tau = 0.5, nu = 1.25, beta=0.01,save_postfix=postfix, jac = jacobian, jact= jacobiantrans).to(device)
    elif index ==0:
        model = MAID(torch.ones(m), x0, lower_level_obj, upper_level_obj,FISTA, Lg, max_iter=3000, epsilon=1e-3, delta=1e-3,rho = 0.5, tau = 0.5, nu = 1.25, beta=0.01,save_postfix=postfix, jac = jacobian, jact= jacobiantrans).to(device)
    elif index ==2:
        model = MAID(torch.ones(m), x0, lower_level_obj, upper_level_obj,FISTA, Lg, max_iter=3000, epsilon=1e-1, delta=1e-1,rho = 0.5, tau = 0.5, nu = 1.25, beta=0.01,save_postfix=postfix, jac = jacobian, jact= jacobiantrans).to(device)
        model.fixed_eps = True
    elif index ==4:
        model = MAID(torch.ones(m), x0, lower_level_obj, upper_level_obj,FISTA, Lg, max_iter=3000, epsilon=1e-5, delta=1e-5,rho = 0.5, tau = 0.5, nu = 1.25, beta=0.01,save_postfix=postfix, jac = jacobian, jact= jacobiantrans).to(device)
        model.fixed_eps = True
    elif index ==3:
        model = MAID(torch.ones(m), x0, lower_level_obj, upper_level_obj,FISTA, Lg, max_iter=3000, epsilon=1e-3, delta=1e-3,rho = 0.5, tau = 0.5, nu = 1.25, beta=0.01,save_postfix=postfix, jac = jacobian, jact= jacobiantrans).to(device)
        model.fixed_eps = True
    elif index ==5:
        model = MAID(torch.ones(m), x0, lower_level_obj, upper_level_obj,FISTA, Lg, max_iter=3000, epsilon=1e-1, delta=1e-1,rho = 0.5, tau = 0.5, nu = 1.25, beta=0.01,save_postfix=postfix, jac = jacobian, jact= jacobiantrans).to(device)
        model.fixed_eps = True
    model.main()