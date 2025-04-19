from MAID import *

torch.manual_seed(23)
np.random.seed(23)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LowerLevelObjective(nn.Module):
    def __init__(self, X, y):
        super(LowerLevelObjective, self).__init__()
        self.b = y.float()
        self.a = X

    def forward(self, x, theta):
        return nn.CrossEntropyLoss()(
            self.a @ x, torch.argmax(self.b, dim=1)
        ) + 0.5 * torch.sum(torch.dot(torch.exp(theta), x.flatten() ** 2))


# Define the upper level objective function
class UpperLevelObjective(nn.Module):
    def __init__(self, X, y):
        super(UpperLevelObjective, self).__init__()
        self.b = y.float()
        self.a = X

    def forward(self, x):
        return nn.CrossEntropyLoss()((self.a @ x), torch.argmax(self.b, dim=1))


def jacobian(self, x, theta, d):
    x.requires_grad_(True)
    theta.requires_grad_(True)
    out = self.lower_level_obj(x, self.theta)
    grad_x = torch.autograd.grad(
        outputs=out,
        inputs=x,
        grad_outputs=None,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
        allow_unused=True,
    )[
        0
    ]  # first get grad using autograd
    gradvp = torch.dot(grad_x.flatten(), d.flatten())
    jvp = torch.autograd.grad(
        outputs=gradvp,
        inputs=theta,
        grad_outputs=torch.ones(gradvp.shape).requires_grad_(True),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
        allow_unused=True,
    )[0]
    return jvp.detach()


def jacobiantrans(self, x, theta, d):
    # transpose of jacobian
    x.to(device).requires_grad_(True)
    theta.to(device).requires_grad_(True)
    out = self.lower_level_obj(x, self.theta).to(device)
    grad_x = torch.autograd.grad(
        outputs=out,
        inputs=theta,
        grad_outputs=None,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
        allow_unused=True,
    )[
        0
    ]  # first get grad using autograd
    gradvp = torch.dot(grad_x.flatten(), d.flatten())
    return torch.autograd.grad(
        outputs=gradvp,
        inputs=x,
        grad_outputs=torch.ones(gradvp.shape).requires_grad_(True),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
        allow_unused=True,
    )[0]


def FISTA(self, theta, x, tol, max_iter):
    theta = theta.to(device)
    x0 = x.clone().to(device)
    t = 0
    for k in range(max_iter):
        x_old = x0
        v = torch.randn_like(x0)
        for _ in range(10):
            Hv = self.hessian(x0, v)
            v = Hv / torch.norm(Hv)
        self.Lphi = torch.dot(Hv.flatten(), v.flatten())
        L = self.Lphi
        self.mu = L - 1e-3
        mu = self.mu
        self.Lphi = L
        tau = 1 / L
        q = tau * mu
        t_old = t
        t = (1 - q * t**2 + torch.sqrt((1 - q * t**2) ** 2 + 4 * t**2)) / 2
        beta = ((t_old - 1) * (1 - t * q)) / (t * (1 - q))
        z = (x0 + beta * (x0 - x_old)).to(device)
        p = torch.autograd.grad(
            outputs=self.lower_level_obj(z, theta),
            inputs=z,
            grad_outputs=None,
            create_graph=False,
            retain_graph=False,
            only_inputs=True,
            allow_unused=True,
        )[0].detach()
        if (
            torch.isnan(self.upper_level_obj(z - tau * p))
            or torch.linalg.norm(p) / mu > 1e5
        ):
            return x.detach()
        if torch.linalg.norm(p) / mu < tol:
            x0 = z - tau * p
            self.ll_calls += k
            return x0.detach()
        x0 = z - tau * p
    self.ll_calls += k
    if torch.isnan(self.upper_level_obj(x0)):
        return x.detach()
    return x0.detach()


class HOAG(nn.Module):
    def __init__(
        self,
        theta,
        x0,
        lower_level_obj,
        upper_level_obj,
        Lg,
        max_iter=100,
        epsilon=1e-2,
        rho=0.9,
        save_postfix="",
        accuracy_type="geometric",
    ) -> None:
        super().__init__()
        self.theta = theta
        self.x = x0
        self.lower_level_obj = lower_level_obj
        self.upper_level_obj = upper_level_obj
        self.epsilon = epsilon
        self.rho = rho
        self.mu = 1 + torch.exp(theta[2])
        self.max_iter = max_iter
        self.max_ll_iter = 10000
        self.Lg = Lg
        self.LAinv = 0
        self.LB = 0
        self.lip_cal = True
        self.loss = []
        self.ll_calls = 0
        self.CG_calls = 0
        self.state = {
            "loss": [],
            "LL_calls": [],
            "eps": [],
            "delta": [],
            "step_size": [],
            "grads": [],
            "CG_calls": [],
            "params": [],
        }
        self.save_postfix = save_postfix
        self.accuracy_type = accuracy_type

    def grad_lower(self, x):
        x.to(device).requires_grad_(True)
        out = self.lower_level_obj(x, self.theta).to(device)
        return torch.autograd.grad(
            outputs=out,
            inputs=x,
            grad_outputs=None,
            create_graph=False,
            retain_graph=False,
            only_inputs=True,
            allow_unused=True,
        )[0]

    def grad_upper(self, x_tilda):
        x_tilda.to(device).requires_grad_(True)
        # x_tilda = x_tilda.clone().detach().to(dtype=torch.float).requires_grad_(True)
        out = self.upper_level_obj(x_tilda).to(device).to(dtype=torch.float)
        grad = torch.autograd.grad(
            outputs=out,
            inputs=x_tilda,
            grad_outputs=None,
            create_graph=False,
            retain_graph=False,
            only_inputs=True,
            allow_unused=True,
        )[0]
        return grad

    def hessian(self, x, d):
        x.to(device).requires_grad_(True)
        out = self.lower_level_obj(x, self.theta).to(device)
        grad_x = torch.autograd.grad(
            outputs=out,
            inputs=x,
            grad_outputs=None,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
            allow_unused=True,
        )[
            0
        ]  # first get grad using autograd
        hvp = torch.autograd.grad(
            outputs=grad_x,
            inputs=x,
            grad_outputs=d,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
            allow_unused=True,
        )[0]
        return hvp.detach()

    def jacobian(self, x, theta, d):
        x.requires_grad_(True)
        theta.requires_grad_(True)
        out = self.lower_level_obj(x, self.theta)
        grad_x = torch.autograd.grad(
            outputs=out,
            inputs=x,
            grad_outputs=None,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
            allow_unused=True,
        )[
            0
        ]  # first get grad using autograd
        gradvp = torch.dot(grad_x.flatten(), d.flatten())
        jvp = torch.autograd.grad(
            outputs=gradvp,
            inputs=theta,
            grad_outputs=torch.ones(gradvp.shape).requires_grad_(True),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
            allow_unused=True,
        )[0]
        return jvp.detach()

    def CG(self, x, b, tol):
        x = x.to(device)  # Move x tensor to GPU
        b = b.to(device)  # Move b tensor to GPU
        r = b
        p = r
        rsold = (torch.linalg.norm(r) ** 2).double()
        solution = torch.zeros(x.shape, device=device)
        if torch.linalg.norm(r) <= tol:
            solution = x.clone().detach()
        k = 0
        while torch.linalg.norm(r) > tol and k < 2000:
            k += 1
            Ap = self.hessian(x, p)
            alpha = rsold.double() / torch.dot(
                p.flatten().double(), Ap.flatten().double()
            )
            solution = solution + alpha * p
            r = r - alpha * Ap
            rsnew = torch.linalg.norm(r) ** 2
            if torch.sqrt(rsnew) < tol:
                self.CG_calls += k
                return solution
            p = r + (rsnew / rsold) * p
            rsold = rsnew
        self.CG_calls += k
        return solution

    def FISTA(self, theta, x, tol, max_iter):
        theta = theta.to(device)
        x = x.to(device)
        t = 0
        for k in range(max_iter):
            x_old = x
            v = torch.randn_like(x)
            for _ in range(10):
                Hv = self.hessian(x, v)
                v = Hv / torch.norm(Hv)
            L = torch.dot(Hv.flatten(), v.flatten())
            self.mu = L - 1e-3
            mu = self.mu
            tau = 1 / L
            q = tau * mu
            t_old = t
            t = (1 - q * t**2 + torch.sqrt((1 - q * t**2) ** 2 + 4 * t**2)) / 2
            beta = ((t_old - 1) * (1 - t * q)) / (t * (1 - q))
            z = (x + beta * (x - x_old)).to(device)
            p = torch.autograd.grad(
                outputs=self.lower_level_obj(z, theta),
                inputs=z,
                grad_outputs=None,
                create_graph=False,
                retain_graph=False,
                only_inputs=True,
                allow_unused=True,
            )[0].detach()
            if torch.linalg.norm(p) / mu < tol:
                x = z - tau * p
                self.ll_calls += k
                return x.detach()
            x = z - tau * p
        self.ll_calls += k
        return x.detach()

    def main(self):
        epsilon = self.epsilon
        theta = self.theta
        loss_val = self.upper_level_obj(self.x)
        if torch.isnan(loss_val):
            self.loss.append(self.loss[-1])
        else:
            self.loss.append(loss_val)
        self.state["CG_calls"].append(self.CG_calls)
        self.state["LL_calls"].append(self.ll_calls)
        g_func_old = torch.tensor(torch.inf)
        for k in range(self.max_iter):
            print("Iteration", k)
            x = self.x
            # inner iterations
            x_new = self.FISTA(theta, x, epsilon, self.max_ll_iter)
            g_func = self.upper_level_obj(x_new)
            self.x = x_new
            q = self.CG(x_new, self.grad_upper(x_new), epsilon)
            p = -self.jacobian(x_new, theta, q.float())
            print("Inner Iterations", self.ll_calls)
            g_func = self.upper_level_obj(self.x)
            self.state["grads"].append(torch.linalg.norm(p))
            # Adaptive step size of Pedregosa
            if k == 0:
                if self.state["grads"][-1] > 1e-3:
                    L_lambda = self.state["grads"][-1] / np.sqrt(p.flatten().shape[0])
                else:
                    L_lambda = 1
                # L_lambda = 1/ (1503.3225)
                step_size = 1 / L_lambda
                print("initial step size", step_size)
                self.state["step_size"].append(step_size)
            incr = torch.linalg.norm(step_size * p)
            C = 0.25
            factor_L_lambda = 1.0
            # theta_new = theta - step_size*p
            old_epsilon = self.epsilon
            if self.accuracy_type == "geometric":
                epsilon *= self.rho
            elif self.accuracy_type == "quadratic":
                if k == 0:
                    epsilon0 = self.epsilon
                elif self.epsilon > epsilon0 * (1 / (k + 1)) ** 2:
                    epsilon = epsilon0 * (1 / (k + 1)) ** 2
            elif self.accuracy_type == "cubic":
                if k == 0:
                    epsilon0 = self.epsilon
                elif self.epsilon > epsilon0 * (1 / (k + 1)) ** 3:
                    epsilon = epsilon0 * (1 / (k + 1)) ** 3
            elif self.accuracy_type == "fixed":
                epsilon = self.epsilon
            if epsilon < 1e-12:
                epsilon = 1e-12
            print("g_func", g_func, g_func_old, "epsilon:", epsilon)
            if (
                g_func
                <= g_func_old
                + C * epsilon
                + old_epsilon * (C + factor_L_lambda) * incr
                - factor_L_lambda * (L_lambda) * incr * incr
            ):
                L_lambda *= 0.95
                step_size = 1 / L_lambda
                theta = theta - step_size * p
                if torch.isnan(g_func) or torch.isnan(g_func_old):
                    self.loss.append(self.loss[-1])
                else:
                    self.loss.append(g_func)
                self.state["CG_calls"].append(self.CG_calls)
                self.state["LL_calls"].append(self.ll_calls)
                g_func_old = g_func
                print("step size increased", g_func, g_func_old)
            elif g_func >= 1.2 * g_func_old:
                # decrease step size
                L_lambda *= 2
                print("!!step size rejected!!", g_func, g_func_old)
                # tighten tolerance
                if self.accuracy_type != "fixed":
                    epsilon = epsilon * 0.5
                step_size = 1 / L_lambda
                if torch.isnan(g_func_old) or torch.isnan(g_func):
                    self.loss.append(self.loss[-1])
                else:
                    self.loss.append(g_func_old)
                self.state["CG_calls"].append(self.CG_calls)
                self.state["LL_calls"].append(self.ll_calls)

            else:
                theta = theta - step_size * p
                if torch.isnan(g_func):
                    self.loss.append(self.loss[-1])
                else:
                    self.loss.append(g_func)
                self.state["CG_calls"].append(self.CG_calls)
                self.state["LL_calls"].append(self.ll_calls)
                g_func_old = self.loss[-1]
            with torch.no_grad():
                print("Theta projection")
                theta[theta < -1000] = -1000
                theta[theta > 1000] = 1000
            with torch.no_grad():
                self.theta = theta
                self.theta.grad = theta.grad
            self.epsilon = epsilon
            if self.ll_calls > 1 * 6e5:
                print("max ll calls reached")
                break
            if epsilon < 1e-12:
                epsilon = 1e-12
            self.state["eps"].append(self.epsilon)
            self.state["step_size"].append(step_size)
            self.state["params"] = self.theta
            # self.state['loss'] = self.loss
            gc.collect()
        self.state["loss"] = self.loss
        self.state["params"] = self.theta
        torch.save(
            self.state,
            os.getcwd() + f"/state_dict_HOAG{self.accuracy_type+self.save_postfix}.pt",
        )
        return self.theta


from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

# Multilogistic regression
input_size = 12 * 12  # Number of features (12x12 subsampled images)
num_classes = 10  # Number of classes in MNIST dataset
batch_size = 60000  # Batch size for training
test_batch_size = 10000  # Batch size for testing

# Load MNIST dataset and create data loaders
transform = transforms.Compose(
    [
        transforms.Resize((12, 12)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.Lambda(lambda x: x / 2 + 0.5),
    ]
)

train_dataset = MNIST(root="./data", train=True, transform=transform, download=True)
test_dataset = MNIST(root="./data", train=False, transform=transform, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=test_batch_size, shuffle=True)

data_iter = iter(train_loader)
X_train, y_train = next(data_iter)
y_train = torch.nn.functional.one_hot(y_train)
X_train = X_train.reshape(-1, input_size)
data_iter = iter(test_loader)
X_test, y_test = next(data_iter)
y_test = torch.nn.functional.one_hot(y_test)
X_test = X_test.reshape(-1, input_size)
lower_level_obj = LowerLevelObjective(X_train.to(device), y_train.to(device))
upper_level_obj = UpperLevelObjective(X_test.to(device), y_test.to(device))
theta = -3 * torch.ones((num_classes, input_size), requires_grad=True).flatten().to(
    device
)
x0 = torch.zeros((input_size, num_classes), requires_grad=True).to(device)
# Computing Lg
x = x0
d = torch.randn_like(x0).to(device)
for _ in range(10):
    out = lower_level_obj(x, theta)
    grad_x = torch.autograd.grad(
        outputs=out,
        inputs=x,
        grad_outputs=None,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
        allow_unused=True,
    )[0]
    hvp = torch.autograd.grad(
        outputs=grad_x,
        inputs=x,
        grad_outputs=d,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
        allow_unused=True,
    )[0]
    d = hvp / torch.norm(hvp)
Lg = torch.dot(d.flatten(), hvp.flatten())

# Running HOAG
for index, acc in enumerate(["geometric", "cubic", "quadratic"]):
    model = HOAG(
        theta,
        x0,
        lower_level_obj,
        upper_level_obj,
        Lg,
        max_iter=500,
        epsilon=1e-1,
        rho=0.9,
        save_postfix="",
        accuracy_type=acc,
    ).to(device)
    model.main()

# Setting initial step size of MAID same as HOAG
if os.path.exists("state_dict_HOAGgeometric.pt"):
    init_step_size = torch.load("state_dict_HOAGgeometric.pt")["step_size"][0]
else:
    init_step_size = 1508.3416

# Running MAID
for index, postfix in enumerate(["0101"]):
    if index == 0:
        init_step_size = init_step_size
        model = MAID(
            theta,
            x0,
            lower_level_obj,
            upper_level_obj,
            FISTA,
            Lg,
            max_iter=500,
            epsilon=1e-1,
            delta=1e-1,
            beta=init_step_size,
            save_postfix=postfix,
            jac=jacobian,
            jact=jacobiantrans,
        ).to(device)
    model.main()
