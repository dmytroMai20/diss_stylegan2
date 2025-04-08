import torch
from torch import optim
from generator import Generator
from discriminator import Discriminator
from mappingmlp import MappingMLP
from util import PathLengthPenalty, GradientPenalty
from losses import DiscriminatorLoss, GeneratorLoss
import math
from tqdm import tqdm
from torch import nn
import dataset
import matplotlib.pyplot as plt

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dim_w = 512
    im_size = 64
    mappingnet_layers = 8
    lr = 1e-3
    mapping_lr=1e-5
    mixing_prob = 0.9
    epochs = 30

    grad_pen_interval = 4
    grad_pen_coef = 10.
    path_pen_interval = 32
    path_pen_after = 5000

    img_res = 64

    generator = Generator(int(math.log2(img_res)),dim_w).to(device)
    discriminator = Discriminator(int(math.log2(img_res))).to(device)
    mapping_net = MappingMLP(dim_w, mappingnet_layers).to(device)

    num_blocks = int(math.log2(img_res))-1
    disc_loss = DiscriminatorLoss().to(device)
    gen_loss = GeneratorLoss().to(device)

    path_len_pen = PathLengthPenalty(0.99).to(device)
    r1_pen = GradientPenalty()

    g_optim = optim.Adam(generator.parameters(), lr=lr, betas=(0.0, 0.99))
    d_optim = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.0, 0.99))
    mlp_optim = optim.Adam(mapping_net.parameters(), lr=mapping_lr, betas=(0.0, 0.99))

    # add mlp mapping and optim

    epochs = 3
    batch_size = 32

    data_loader = dataset.get_loader(batch_size,im_size,"STL10")

    num_batches = len(data_loader)

    d_losses, g_losses = [], []

    for epoch in range(epochs):
        for batch_idx, (real_images, _) in enumerate(tqdm(data_loader)):
            d_optim.zero_grad()
            real_images = real_images.to(device)
            #d_optim.zero_grad() 
            fake_images, _ = gen_images(batch_size, generator, num_blocks, mixing_prob, dim_w, mapping_net, device)
            fake_output = discriminator(fake_images.detach())
            # requires.grad if reaches gradient penalty interval (set to 4)
            if (batch_idx+1) % grad_pen_interval == 0:
                real_images.requires_grad_()
            real_output = discriminator(real_images)

            real_loss, fake_loss = disc_loss(real_output, fake_output)
            d_loss = real_loss + fake_loss
            
            if (batch_idx+1) % grad_pen_interval == 0:
                 r1 = r1_pen(real_images, real_output)
                 d_loss = d_loss + 0.5 * grad_pen_coef * r1 * grad_pen_interval
            d_loss.backward()
            nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1)
            d_optim.step()

            d_losses.append(d_loss.item())

            g_optim.zero_grad()
            mlp_optim.zero_grad()

            fake_images, w = gen_images(batch_size, generator, num_blocks, mixing_prob, dim_w, mapping_net, device)

            fake_output = discriminator(fake_images)

            g_loss = gen_loss(fake_output)

            if (batch_idx+1) % path_pen_interval == 0 and batch_idx+(epoch*num_batches) > path_pen_after:
                 path_len_penalty = path_len_pen(w, fake_images)
                 if not torch.isnan(path_len_penalty):
                      g_loss = g_loss + path_len_penalty
            g_loss.backward()
            nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
            nn.utils.clip_grad_norm_(mapping_net.parameters(), max_norm=1.0)

            g_optim.step()
            mlp_optim.step()

            g_losses.append(g_loss.item())
        print(f"epoch {epoch}/{epochs} completed")
    save_model("data", mapping_net, generator, "STL10", str(im_size))

    plt.figure(figsize=(10, 5))
    plt.plot(d_losses, label="Discriminator Loss")
    plt.plot(g_losses, label="Generator Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training Loss Curves")
    plt.savefig("loss_plot.png")
    plt.show()

def get_w(batch_size: int, style_mixing_prob, num_blocks, w_dims, mapping_network,device):
        if torch.rand(()).item() < style_mixing_prob:
            cross_over_point = int(torch.rand(()).item() * num_blocks)
            z2 = torch.randn(batch_size, w_dims).to(device)
            z1 = torch.randn(batch_size, w_dims).to(device)

            w1 = mapping_network(z1)
            w2 = mapping_network(z2)

            w1 = w1[None, :, :].expand(cross_over_point, -1, -1)
            w2 = w2[None, :, :].expand(num_blocks - cross_over_point, -1, -1)
            return torch.cat((w1, w2), dim=0)

        else:

            z = torch.randn(batch_size, w_dims).to(device)

            w = mapping_network(z)

            return w[None, :, :].expand(num_blocks, -1, -1)

def get_noise(batch_size: int, num_blocks: int, device):
        noise = []
        resolution = 4
        for i in range(num_blocks):
            if i == 0:
                n1 = None
            else:
                n1 = torch.randn(batch_size, 1, resolution, resolution, device=device)
            n2 = torch.randn(batch_size, 1, resolution, resolution, device=device)
            noise.append((n1, n2))
            resolution *= 2
        return noise

def gen_images(batch_size, generator, num_blocks, style_mixing_prob, w_dims, mlp, device):
     w = get_w(batch_size, style_mixing_prob, num_blocks, w_dims, mlp, device)
     noise = get_noise(batch_size, num_blocks,device)
     imgs = generator(w, noise)
     return imgs, w

def save_model(path, mapping_net, generator, dataset,res):
     save_path = f"{path}/stylegan2_{dataset}_{res}.pt"
     torch.save({'generator':generator.state_dict(),
                 'mapping_net':mapping_net.state_dict()}, save_path)

if __name__ == "__main__":
    train()
    #data_loader = dataset.get_loader(32,"STL10")
    #print(f"len : {len(data_loader)}")
