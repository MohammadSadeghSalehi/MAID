from MAID import *

torch.random.manual_seed(31)
np.random.seed(31)

num_filters = 48
kernel_size = 7
filter_size = 7

# device = torch.device("mps")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class UpperLevelObjective(nn.Module):
    def __init__(self, lower_level_obj, x_true):
        super(UpperLevelObjective, self).__init__()
        self.x_true = x_true.float()

    def forward(self, x):
        return 0.5 * torch.linalg.norm(x - self.x_true) ** 2 / x.shape[0]


class FOE(nn.Module):
    def __init__(self, num_kernels, kernel_size):
        super(FOE, self).__init__()
        self.num_kernels = num_kernels
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            1, num_kernels, kernel_size, padding=2, bias=False, padding_mode="reflect"
        )
        self.params = nn.Parameter(torch.exp(-3 * torch.ones(2 * num_kernels + 1)))

    def smoothed_norm1(self, x, nu):
        return torch.sum(torch.sqrt(x**2 + nu**2))

    def forward(self, x, theta):
        with torch.no_grad(), torch.autograd.set_detect_anomaly(True):
            self.params.data = theta[: self.num_kernels * 2 + 1]
            self.conv.weight.data = theta[
                self.num_kernels * 2
                + 1 : self.num_kernels * 2
                + 1
                + self.num_kernels * self.kernel_size**2
            ].reshape(self.num_kernels, 1, self.kernel_size, self.kernel_size)
        x = self.conv(x)
        out = sum(
            torch.exp(self.params[i])
            * self.smoothed_norm1(
                x[:, i - 1], torch.exp(self.params[i + self.num_kernels])
            )
            for i in range(1, self.num_kernels + 1)
        )

        return torch.exp(self.params[0]) * out


class LowerLevelObjective_FOE(nn.Module):
    def __init__(self, y):
        super(LowerLevelObjective_FOE, self).__init__()
        self.y = y
        self.FOE = FOE(num_filters, kernel_size)

    def forward(self, x, theta):
        return (
            0.5 * torch.linalg.norm(x - self.y) ** 2
            + 0.5
            * self.FOE(
                torch.reshape(
                    x, (x.shape[0], int(np.sqrt(x.shape[1])), int(np.sqrt(x.shape[1])))
                ).unsqueeze(1),
                theta,
            )
        ) / x.shape[0]


def jacobian(self, x, theta, d):
    x.requires_grad_(True)
    theta.requires_grad_(True)
    grad_x = torch.autograd.grad(
        outputs=self.lower_level_obj(x, self.theta),
        inputs=x,
        grad_outputs=None,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
        allow_unused=True,
    )[0]
    grad_vector_dot = torch.dot(grad_x.flatten(), d.flatten())
    grad_weights = torch.autograd.grad(
        outputs=grad_vector_dot,
        inputs=self.lower_level_obj.FOE.conv.weight,
        grad_outputs=None,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
        allow_unused=True,
    )[0]
    grad_params = torch.autograd.grad(
        outputs=grad_vector_dot,
        inputs=self.lower_level_obj.FOE.params,
        grad_outputs=None,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
        allow_unused=True,
    )[0]
    jvp = torch.cat((grad_params.flatten(), grad_weights.flatten()))
    return jvp.detach()


def jacobiantrans(self, x, theta, d):
    # transpose of jacobian
    x.to(device).requires_grad_(True)
    theta = theta.requires_grad_(True).to(device)
    out = self.lower_level_obj(x, theta).to(device)
    grad_weights = torch.autograd.grad(
        outputs=out,
        inputs=self.lower_level_obj.FOE.conv.weight,
        grad_outputs=None,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
        allow_unused=True,
    )[0]
    grad_params = torch.autograd.grad(
        outputs=out,
        inputs=self.lower_level_obj.FOE.params,
        grad_outputs=None,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
        allow_unused=True,
    )[0]
    cat = torch.cat((grad_params.flatten(), grad_weights.flatten()))
    gradvp = torch.dot(cat.flatten(), d.flatten())
    jtvp = torch.autograd.grad(
        outputs=gradvp,
        inputs=x,
        grad_outputs=torch.ones(gradvp.shape).requires_grad_(True),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
        allow_unused=True,
    )[0]
    return jtvp.detach()


# Define the path to the Kodak dataset folder
kodak_data_path = f"{os.getcwd()}/Kodak_dataset"
transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),  # Resize
        transforms.Grayscale(num_output_channels=1),  # Convert to black and white
        transforms.ToTensor(),  # Convert to PyTorch tensor (values between 0 and 1)
    ]
)

kodak_images = []
for filename in os.listdir(kodak_data_path):
    image_path = os.path.join(kodak_data_path, filename)
    img = Image.open(image_path)
    img_tensor = transform(img)
    kodak_images.append(img_tensor)
plt.imshow(kodak_images[-1].squeeze(0).detach().numpy(), cmap="gray", vmin=0, vmax=1)
plt.show()
print(torch.min(kodak_images[-1]), torch.max(kodak_images[-1]))
num_images = len(kodak_images)
num_images = 20
kodak_images_tensor = torch.stack(kodak_images)
kodak_images_tensor = kodak_images_tensor[5 : num_images + 5]

noisy_image = kodak_images_tensor + 0.1 * torch.randn(kodak_images_tensor.shape)
plt.imshow(noisy_image[-1].squeeze(0).detach().numpy(), cmap="gray", vmin=0, vmax=1)
plt.show()
reshaped_images = kodak_images_tensor.view(num_images, -1)
noisy_image = noisy_image.view(num_images, -1)
image_array = reshaped_images
lower_level_obj = LowerLevelObjective_FOE(torch.tensor(noisy_image).float().to(device))
upper_level_obj = UpperLevelObjective(
    lower_level_obj, torch.tensor(image_array).float().to(device)
)

filters = np.zeros((num_filters, filter_size, filter_size))
rand_filter = torch.randn(filter_size, filter_size)
# initialize filters
for i in range(num_filters):
    rand_filter = init.xavier_normal_(torch.empty(filter_size, filter_size))
    filters[i] = rand_filter - torch.mean(rand_filter)

theta = -3 * torch.ones(
    2 * num_filters + 1 + num_filters * kernel_size**2, requires_grad=True
)
theta[num_filters : 2 * num_filters] = -3.5
for i in range(num_filters):
    theta[
        2 * num_filters
        + 1
        + i * kernel_size**2 : 2 * num_filters
        + 1
        + (i + 1) * kernel_size**2
    ] = torch.tensor(filters[i].flatten())
Lg = 1


x = torch.zeros(image_array.shape)
# device = torch.device("mps")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lower_level_obj.to(device)
lower_level_obj.FOE.to(device)
upper_level_obj.to(device)
theta = theta.to(device)
x = x.to(device)
with torch.no_grad(), torch.autograd.set_detect_anomaly(True):
    lower_level_obj.FOE.params.data = theta[: lower_level_obj.FOE.num_kernels * 2 + 1]
    lower_level_obj.FOE.conv.weight.data = theta[
        lower_level_obj.FOE.num_kernels * 2
        + 1 : lower_level_obj.FOE.num_kernels * 2
        + 1
        + lower_level_obj.FOE.num_kernels * lower_level_obj.FOE.kernel_size**2
    ].reshape(
        lower_level_obj.FOE.num_kernels,
        1,
        lower_level_obj.FOE.kernel_size,
        lower_level_obj.FOE.kernel_size,
    )


def FISTA(self, theta, x, tol, max_iter):
    t = 0
    tau = 0
    for k in range(max_iter):
        x_old = x
        mu = torch.exp(theta[0]) * sum(
            (torch.exp(theta[i]) / (torch.exp(theta[i + num_filters])))
            * torch.linalg.norm(
                theta[
                    num_filters * 2
                    + 1
                    + i * kernel_size**2 : num_filters * 2
                    + 1
                    + (i + 1) * kernel_size**2
                ]
            )
            ** 2
            for i in range(1, num_filters + 1)
        )
        # L = (1+ (torch.exp(theta[0]) * sum((torch.exp(theta[i])/ (torch.exp(theta[i+num_filters]))) * torch.linalg.norm(theta[num_filters*2+1+i * kernel_size**2: num_filters*2+1+(i+1) * kernel_size**2])**2 for i in range(1,num_filters+1))))
        v = torch.rand(x.shape).to(device)
        for _ in range(30):
            Hv = self.hessian(x, v)
            v = Hv / torch.norm(Hv)
        L = torch.dot(Hv.flatten(), v.flatten()) / torch.dot(v.flatten(), v.flatten())
        v.detach().cpu()
        self.mu = mu
        tau = 1 / L
        q = tau * mu
        t_old = t
        t = (1 - q * t**2 + torch.sqrt((1 - q * t**2) ** 2 + 4 * t**2)) / 2
        beta = ((t_old - 1) * (1 - t * q)) / (t * (1 - q))
        z = x + beta * (x - x_old)
        p = torch.autograd.grad(
            outputs=self.lower_level_obj(z, theta),
            inputs=z,
            grad_outputs=None,
            create_graph=False,
            retain_graph=False,
            only_inputs=True,
            allow_unused=False,
        )[0].detach()
        if (torch.linalg.norm(p) / mu) < tol:
            x = z - tau * p
            self.ll_calls += k
            return x.detach()
        x = (z - tau * p).clone().detach()
        p.cpu()
        del p
    self.ll_calls += k
    return x.detach()


for index, postfix in enumerate(
    [
        "0000_FOE",
        "0101_FOE",
        "0303_FOE",
        "0505_FOE",
        "0101_FOE_fixed",
        "0303_FOE_fixed",
        "0505_FOE_fixed",
    ]
):
    if index == 4:
        model = MAID(
            theta,
            x,
            lower_level_obj,
            upper_level_obj,
            FISTA,
            Lg,
            max_iter=1000,
            epsilon=1e-1,
            delta=1e-1,
            beta=0.01,
            rho=0.5,
            tau=0.5,
            save_postfix=postfix,
            psnr_log=True,
            jac=jacobian,
            jact=jacobiantrans,
        )
        model.fixed_eps = True
    elif index == 5:
        model = MAID(
            theta,
            x,
            lower_level_obj,
            upper_level_obj,
            Lg,
            max_iter=1000,
            epsilon=1e-3,
            delta=1e-3,
            beta=0.01,
            rho=0.5,
            tau=0.5,
            save_postfix=postfix,
            psnr_log=True,
            jac=jacobian,
            jact=jacobiantrans,
        )
        model.fixed_eps = True
    elif index == 6:
        model = MAID(
            theta,
            x,
            lower_level_obj,
            upper_level_obj,
            Lg,
            max_iter=1000,
            epsilon=1e-5,
            delta=1e-5,
            beta=0.01,
            rho=0.5,
            tau=0.5,
            save_postfix=postfix,
            psnr_log=True,
            jac=jacobian,
            jact=jacobiantrans,
        )
        model.fixed_eps = True
    elif index == 3:
        model = MAID(
            theta,
            x,
            lower_level_obj,
            upper_level_obj,
            Lg,
            max_iter=1000,
            epsilon=1e-5,
            delta=1e-5,
            beta=0.01,
            rho=0.5,
            tau=0.5,
            save_postfix=postfix,
            psnr_log=True,
            jac=jacobian,
            jact=jacobiantrans,
        )
    elif index == 2:
        model = MAID(
            theta,
            x,
            lower_level_obj,
            upper_level_obj,
            Lg,
            max_iter=1000,
            epsilon=1e-3,
            delta=1e-3,
            beta=0.01,
            rho=0.5,
            tau=0.5,
            save_postfix=postfix,
            psnr_log=True,
            jac=jacobian,
            jact=jacobiantrans,
        )
    elif index == 1:
        model = MAID(
            theta,
            x,
            lower_level_obj,
            upper_level_obj,
            Lg,
            max_iter=1000,
            epsilon=1e-1,
            delta=1e-1,
            beta=0.01,
            rho=0.5,
            tau=0.5,
            save_postfix=postfix,
            psnr_log=True,
            jac=jacobian,
            jact=jacobiantrans,
        )
    elif index == 0:
        model = MAID(
            theta,
            x,
            lower_level_obj,
            upper_level_obj,
            FISTA,
            Lg,
            max_iter=10000,
            epsilon=1e0,
            delta=1e0,
            beta=0.01,
            rho=0.5,
            tau=0.5,
            save_postfix=postfix,
            psnr_log=True,
            jac=jacobian,
            jact=jacobiantrans,
        )
    model.main()
