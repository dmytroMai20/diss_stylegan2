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
    mapping_net = MappingMLP(512, 8).to(device)
    generator.load_state_dict(checkpoint['generator'])
    mapping_net.load_state_dict(checkpoint['mapping_net'])
    return generator, mapping_net

def main():
    num_blocks = int(math.log2(64))-1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator, mapping_net = load_model("data","CelebA", 64, device)
    imgs, w = gen_images(32, generator, num_blocks, 0.9, 512, mapping_net, device)
    imgs = (imgs + 1) / 2
    grid = vutils.make_grid(imgs, nrow=8)
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.show()
    

if __name__=="__main__":
    main()
