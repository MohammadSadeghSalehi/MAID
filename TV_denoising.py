from MAID import *
torch.random.manual_seed(19)
np.random.seed(19)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class LowerLevelObjective(nn.Module):
    def __init__(self, y):
        super(LowerLevelObjective, self).__init__()
        self.y = y
        self.index = None
    def TV(self,x, nu):
        return (torch.sum(torch.sqrt((x[1:]-x[:-1])**2+ nu**2)))
    def TV2D(self,x, nu):
        x = torch.reshape(x,(x.shape[0],int(np.sqrt(x.shape[1])),int(np.sqrt(x.shape[1]))))
        tv = 0
        dx = torch.diff(x, dim=1)
        dy = torch.diff(x, dim=2)
        tv += (torch.sum(torch.sqrt(dx**2+ nu**2)))
        tv += (torch.sum(torch.sqrt(dy**2+ nu**2)))
        return tv
    def forward(self, x, theta):
        return (0.5*torch.linalg.norm(x-self.y)**2 +0.5 * torch.exp(theta[0]) * self.TV2D(x, torch.exp(theta[1])))/ self.y.shape[0]


class UpperLevelObjective(nn.Module):
    def __init__(self, lower_level_obj, x_true):
        super(UpperLevelObjective, self).__init__()
        self.x_true = x_true.float()

    def forward(self, x):
        return (0.5 * torch.linalg.norm(x-self.x_true)**2)/ self.x_true.shape[0]
    
def jacobian(self, x,theta,d):
    x.requires_grad_(True)
    theta.requires_grad_(True)
    out = self.lower_level_obj(x,self.theta)
    grad_x = torch.autograd.grad(outputs=out, inputs=x, grad_outputs=None, create_graph=True, retain_graph=True, only_inputs=True, allow_unused=True)[0] #first get grad using autograd
    gradvp = torch.dot(grad_x.flatten(),d.flatten())
    jvp = torch.autograd.grad(outputs=gradvp, inputs=theta, grad_outputs=torch.ones(gradvp.shape).requires_grad_(True), create_graph=True, retain_graph=True, only_inputs=True, allow_unused=True)[0]
    return jvp.detach()
def jacobiantrans(self, x,theta,d):
    # transpose of jacobian
    x.to(device).requires_grad_(True)
    theta.to(device).requires_grad_(True)
    out = self.lower_level_obj(x,self.theta).to(device)
    grad_x = torch.autograd.grad(outputs=out, inputs=theta, grad_outputs=None, create_graph=True, retain_graph=True, only_inputs=True, allow_unused=True)[0] #first get grad using autograd
    gradvp = torch.dot(grad_x.flatten(),d.flatten())
    return torch.autograd.grad(outputs=gradvp, inputs=x, grad_outputs=torch.ones(gradvp.shape).requires_grad_(True), create_graph=True, retain_graph=True, only_inputs=True, allow_unused=True)[0]

def FISTA(self,theta, x, tol, max_iter):
    t = 0
    tau = 0
    for k in range(max_iter):
        x_old = x
        self.mu = (1)
        mu = self.mu
        L = (1+ (torch.exp(theta[0])/torch.exp(theta[1]))* torch.sqrt(torch.tensor(8)))
        tau = 1/L
        q = tau*mu
        t_old = t
        t = (1-q*t**2+torch.sqrt((1-q*t**2)**2+4*t**2))/2
        beta = ((t_old-1)*(1-t*q))/(t*(1-q))
        z = (x + beta * (x-x_old))
        p = torch.autograd.grad(outputs=self.lower_level_obj(z,theta), inputs=z, grad_outputs=None, create_graph=False, retain_graph=False, only_inputs=True, allow_unused=True)[0].detach()
        if (torch.linalg.norm(p)/mu) < tol:
            x = z -  tau*p
            self.ll_calls += k
            return x.detach()
        x = z -  tau*p
    self.ll_calls += k
    return x.detach()




kodak_data_path = f"{os.getcwd()}/Kodak_dataset"

transform = transforms.Compose([
    transforms.Resize((96, 96)),  
    transforms.Grayscale(num_output_channels=1),  # Convert to black and white
    transforms.ToTensor(),  # Convert to PyTorch tensor (values between 0 and 1)
])

kodak_images = []
for filename in os.listdir(kodak_data_path):
    image_path = os.path.join(kodak_data_path, filename)
    img = Image.open(image_path)
    img_tensor = transform(img)
    kodak_images.append(img_tensor)
plt.imshow(kodak_images[-1].squeeze(0).detach().numpy(), cmap='gray', vmin=0, vmax=1)
plt.show()
print(torch.min(kodak_images[-1]), torch.max(kodak_images[-1]))
num_images = len(kodak_images)
kodak_images_tensor = torch.stack(kodak_images)
noisy_image = kodak_images_tensor + 0.1 * torch.randn(kodak_images_tensor.shape)
plt.imshow(noisy_image[-1].squeeze(0).detach().numpy(), cmap='gray', vmin=0, vmax=1)
plt.show()
reshaped_images = kodak_images_tensor.view(num_images, -1)
noisy_image = noisy_image.view(num_images, -1)

image_array = reshaped_images
lower_level_obj = LowerLevelObjective(torch.tensor(noisy_image).float().to(device))
upper_level_obj = UpperLevelObjective(lower_level_obj, torch.tensor(image_array).float().to(device))

theta = torch.tensor([-5.0, -5.0], requires_grad=True)
Lg = 1

x = torch.zeros(image_array.shape).float()
# Move tensors to GPU if available
# device = torch.device("mps")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
lower_level_obj.to(device)
upper_level_obj.to(device)
theta = theta.to(device)
x = x.to(device)




for index, postfix in enumerate( ["10100", "1002", "100100", "1020", "1001", "1010", "0000", "0101", "0303", "0505", "0101_fixed", "0303_fixed", "0505_fixed"]):
    if index == 2:
        model = MAID(theta, x, lower_level_obj, upper_level_obj,FISTA, Lg,max_iter= 1000, epsilon= 1e2, delta = 1e2, beta= 0.01, rho= 0.5,tau= 0.5, save_postfix=postfix, jac = jacobian, jact= jacobiantrans)
        # model.fixed_eps= True
    elif index == 1:
        model = MAID(theta, x, lower_level_obj, upper_level_obj,FISTA, Lg,max_iter= 1000, epsilon= 1e1, delta = 1e-2, beta= 0.01, rho= 0.5,tau= 0.5, save_postfix=postfix, jac = jacobian, jact= jacobiantrans)
    elif index == 0:
        model = MAID(theta, x, lower_level_obj, upper_level_obj,FISTA, Lg,max_iter= 1000, epsilon= 1e1, delta = 1e2, beta= 0.01, rho= 0.5,tau= 0.5, save_postfix=postfix, jac = jacobian, jact= jacobiantrans)
        # model.fixed_eps= True
    elif index == 3:
        model = MAID(theta, x, lower_level_obj, upper_level_obj,FISTA, Lg,max_iter= 1000, epsilon= 1e1, delta = 2e1, beta= 0.01, rho= 0.5,tau= 0.5, save_postfix=postfix, jac = jacobian, jact= jacobiantrans)
        # model.fixed_eps= True
    elif index == 4:
        model = MAID(theta, x, lower_level_obj, upper_level_obj, FISTA,Lg,max_iter= 1000, epsilon= 1e1, delta = 1e-1, beta= 0.01, rho= 0.5,tau= 0.5, save_postfix=postfix, jac = jacobian, jact= jacobiantrans)
    elif index == 5:
        model = MAID(theta, x, lower_level_obj, upper_level_obj, FISTA,Lg,max_iter= 1000, epsilon= 1e1, delta = 1e1, beta= 0.01, rho= 0.5,tau= 0.5, save_postfix=postfix, jac = jacobian, jact= jacobiantrans)

    elif index == 6:
        model = MAID(theta, x, lower_level_obj, upper_level_obj,FISTA, Lg,max_iter= 1000, epsilon= 1e0, delta = 1e0, beta= 0.01, rho= 0.5,tau= 0.5, save_postfix=postfix, jac = jacobian, jact= jacobiantrans)
    elif index == 7:
        model = MAID(theta, x, lower_level_obj, upper_level_obj,FISTA, Lg,max_iter= 1000, epsilon= 1e-1, delta = 1e-1, beta= 0.01, rho= 0.5,tau= 0.5, save_postfix=postfix, jac = jacobian, jact= jacobiantrans)
    elif index == 8:
        model = MAID(theta, x, lower_level_obj, upper_level_obj,FISTA, Lg,max_iter= 1000, epsilon= 1e-3, delta = 1e-3, beta= 0.01, rho= 0.5,tau= 0.5, save_postfix=postfix, jac = jacobian, jact= jacobiantrans)
    elif index == 9:
        model = MAID(theta, x, lower_level_obj, upper_level_obj,FISTA, Lg,max_iter= 1000, epsilon= 1e-5, delta = 1e-5, beta= 0.01, rho= 0.5,tau= 0.5, save_postfix=postfix, jac = jacobian, jact= jacobiantrans)
    elif index == 10:
        model = MAID(theta, x, lower_level_obj, upper_level_obj, FISTA,Lg,max_iter= 1000, epsilon= 1e-1, delta = 1e-1, beta= 0.01, rho= 0.5,tau= 0.5, save_postfix=postfix, jac = jacobian, jact= jacobiantrans)
        model.fixed_eps= True
    elif index == 11:
        model = MAID(theta, x, lower_level_obj, upper_level_obj, FISTA,Lg,max_iter= 1000, epsilon= 1e-3, delta = 1e-3, beta= 0.01, rho= 0.5,tau= 0.5, save_postfix=postfix, jac = jacobian, jact= jacobiantrans)
        model.fixed_eps= True
    elif index == 12:
        model = MAID(theta, x, lower_level_obj, upper_level_obj, FISTA,Lg,max_iter= 1000, epsilon= 1e-5, delta = 1e-5, beta= 0.01, rho= 0.5,tau= 0.5, save_postfix=postfix, jac = jacobian, jact= jacobiantrans)
        model.fixed_eps= True
    model.main()
