import torch


def psnr(img1, img2):
    img1 = torch.clip(img1, 0, 1)
    img2 = torch.clip(img2, 0, 1)
    mse = torch.mean((img1 - img2) ** 2)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))
