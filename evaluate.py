import torch
from generator import Generator
from mappingmlp import MappingMLP
import math
from train import gen_images

import matplotlib.pyplot as plt
import torchvision.utils as vutils


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
    imgs, w = gen_images(32, generator, num_blocks, 0.9, 512, mapping_net, device)
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
    

if __name__=="__main__":
    main()
