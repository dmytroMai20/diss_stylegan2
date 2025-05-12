import torch
from generator import Generator
from mappingmlp import MappingMLP
import math
from train import gen_images

import matplotlib.pyplot as plt
import torchvision.utils as vutils
from torchvision.utils import make_grid
from scipy.stats import truncnorm

def load_model(path, dataset, res, device):
    checkpoint = torch.load(f"{path}/stylegan2_{dataset}_{res}.pt", map_location=device)
    generator = Generator(int(math.log2(res)),512).to(device) # keep latent dimensions to 512 in all experiments
    ema = Generator(int(math.log2(res)),512).to(device)
    best = Generator(int(math.log2(res)),512).to(device)
    mapping_net = MappingMLP(512, 8).to(device)
    generator.load_state_dict(checkpoint['generator'])
    mapping_net.load_state_dict(checkpoint['mapping_net'])
    ema.load_state_dict(checkpoint['ema'])
    best.load_state_dict(checkpoint['best_model'])
    return ema,best, generator, mapping_net

def main():
    res = 64
    num_blocks = int(math.log2(res))-1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ema,best, generator, mapping_net = load_model("data","CelebA", res, device)
    #imgs, w = gen_images(32, generator, num_blocks, 0.9, 512, mapping_net, device)
    interpolate_w_and_generate(mapping_net,ema,num_blocks,512,device, 7)
    return 
    imgs = (imgs + 1) / 2
    grid = vutils.make_grid(imgs, nrow=8)
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.show()
    imgs, w = gen_images(32, ema, num_blocks, 0.9, 512, mapping_net, device)
    imgs = (imgs + 1) / 2
    grid = vutils.make_grid(imgs, nrow=8)
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.show()
    imgs, w = gen_images(32, best , num_blocks, 0.9, 512, mapping_net, device)
    imgs = (imgs + 1) / 2
    grid = vutils.make_grid(imgs, nrow=8)
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.show()
    

def truncated_z_sample(batch_size, z_dim, truncation=0.5):
  values = truncnorm.rvs(-2, 2, size=(batch_size, z_dim))
  return truncation * values

@torch.no_grad()
def interpolate_w_and_generate(
    mapping_network,
    synthesis_network,
    num_blocks: int,
    w_dims: int,
    device: torch.device,
    num_steps: int = 5,
):
    """
    Interpolate between two w vectors in W space and generate images.
    
    Args:
        mapping_network: The StyleGAN2 mapping network.
        synthesis_network: The StyleGAN2 synthesis network.
        num_blocks: Number of synthesis blocks (determines depth of network).
        w_dims: Dimensionality of W space.
        device: PyTorch device (e.g., 'cuda' or 'cpu').
        num_steps: Number of interpolation steps (including endpoints).
    """
    # Get two latent vectors z1 and z2
    z1 = torch.randn(1, w_dims).to(device)
    z2 = torch.randn(1, w_dims).to(device)

    # Map them to W space
    w1 = mapping_network(z1)  # shape: (1, w_dims)
    w2 = mapping_network(z2)  # shape: (1, w_dims)

    # Create interpolated ws
    interpolated_ws = []
    for alpha in torch.linspace(0, 1, num_steps):
        w = (1 - alpha) * w1 + alpha * w2  # shape: (1, w_dims)
        # Expand to match per-layer W
        w_expanded = w[None, :, :].expand(num_blocks, -1, -1)  # shape: (num_blocks, 1, w_dims)
        interpolated_ws.append(w_expanded)

    # Generate images from each interpolated w
    images = []
    for w in interpolated_ws:
        img = synthesis_network(w)  # Assumes output shape: (1, 3, H, W)
        images.append(img[0])  # Take the image out of the batch

    # Visualize with torchvision grid and matplotlib
    grid = make_grid(images, nrow=num_steps, normalize=True, scale_each=True)
    plt.figure(figsize=(num_steps * 2, 2))
    plt.axis("off")
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.title("Interpolation in W space")
    plt.savefig(f"interpolation_plot_sgan2.png")
    plt.show()


if __name__=="__main__":
    main()
