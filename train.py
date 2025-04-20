import torch
from torch import optim
from generator import Generator
from discriminator import Discriminator
from mappingmlp import MappingMLP
from ema import EMA
from util import PathLengthPenalty, GradientPenalty
from losses import DiscriminatorLoss, GeneratorLoss
import math
from tqdm import tqdm
from torch import nn
import dataset
import matplotlib.pyplot as plt
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
import torchvision.utils as vutils
import torchvision.transforms as T
import time
import numpy as np

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_name = "CelebA"
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
    ema = EMA(generator).to(device)

    num_blocks = int(math.log2(img_res))-1
    disc_loss = DiscriminatorLoss().to(device)
    gen_loss = GeneratorLoss().to(device)

    path_len_pen = PathLengthPenalty(0.99).to(device)
    r1_pen = GradientPenalty()

    g_optim = optim.Adam(generator.parameters(), lr=lr, betas=(0.0, 0.99))
    d_optim = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.0, 0.99))
    mlp_optim = optim.Adam(mapping_net.parameters(), lr=mapping_lr, betas=(0.0, 0.99))

    # add mlp mapping and optim

    batch_size = 32

    data_loader = dataset.get_loader(batch_size,im_size,dataset_name)

    num_batches = len(data_loader)

    d_losses, g_losses = [], []
    fid_scores = []
    kid_means = []
    kid_stds = []
    best_fid = 1000000
    best_model = generator.state_dict()
    fid_real_imgs = load_real_images(data_loader,device)

    total_imgs_seen = 0

    gpu_mb_alloc = []
    gpu_mb_reserved = []
    times_per_epoch = []
    for epoch in range(epochs):
        start_time = time.time()
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
            ema.update(generator)
            g_losses.append(g_loss.item())
            
            gpu_mb_alloc.append(torch.cuda.memory_allocated() / (1024 ** 2))
            gpu_mb_reserved.append(torch.cuda.memory_reserved() / (1024 ** 2))
            total_imgs_seen+=batch_size
        times_per_epoch.append(time.time()-start_time)
        fid_score = compute_fid(fid_real_imgs,mapping_net,ema.ema_model,device)
        fid_scores.append(fid_score)
        if fid_score >= best_fid:
             best_fid = fid_score
             best_model = ema.ema_model.state_dict()
        kid_mean, kid_std = compute_kid(fid_real_imgs, mapping_net, ema.ema_model, device)
        kid_means.append(kid_mean)
        kid_stds.append(kid_std)
        print(f"epoch {epoch}/{epochs} completed. FID score: {fid_scores[-1]}")
    save_model("data", mapping_net, generator, ema.ema_model,best_model, dataset_name, str(im_size))

    time_per_kimg = ((sum(times_per_epoch)/len(times_per_epoch))/(len(data_loader)*batch_size))*1000

    print(f"Time per 1kimg: {time_per_kimg:.3f}")
    plt.figure(figsize=(10, 5))
    plt.plot(d_losses, label="Discriminator Loss")
    plt.plot(g_losses, label="Generator Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training Loss Curves")
    plt.savefig(f"loss_plot_{dataset_name}_{img_res}.png")
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(gpu_mb_alloc, label="CUDA Memory Allocated (MB)")
    plt.plot(gpu_mb_reserved, label="CUDA Memory Reserved (MB)")
    plt.xlabel("Iteration")
    plt.ylabel("MB")
    plt.legend()
    plt.title("Training Memory Usage")
    plt.savefig(f"memory_plot_{dataset_name}_{img_res}.png")
    plt.show()
    """
    plt.figure(figsize=(10, 5))
    x = range(1,epochs+1)
    plt.plot(x,fid_scores, label="FID scores")
    plt.xlabel("Epoch")
    plt.ylabel("EMA FID")
    plt.legend()
    plt.title("EMA FID curve")
    plt.savefig(f"fid_plot_{dataset_name}_{img_res}.png")
    plt.show()
    
    plt.figure(figsize=(10, 5))
    x = range(1,epochs+1)
    plt.errorbar(x,kid_means,yerr=kid_stds,capsize=5, label="KID scores")
    plt.xlabel("Epoch")
    plt.ylabel("EMA KID")
    plt.legend()
    plt.title("EMA KID curve")
    plt.savefig(f"kid_plot_{dataset_name}_{img_res}.png")
    plt.show()
    """
    # plot FID and KID against time 
    cum_times = np.cumsum(np.array(times_per_epoch))
    fig, ax1 = plt.subplots()   # may need to fix figure size to (10,5) too

    # FID line (left y-axis)
    color = 'tab:blue'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('FID', color=color)
    ax1.plot(cum_times, fid_scores, color=color, label='FID')
    ax1.tick_params(axis='y', labelcolor=color)

    # KID line with error bars (right y-axis)
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('KID', color=color)
    ax2.errorbar(cum_times, kid_means, yerr=kid_stds, color=color, linestyle='--', marker='o', label='KID ± std')
    ax2.tick_params(axis='y', labelcolor=color)

    # Layout and title
    fig.tight_layout()
    plt.title("FID and KID over Training Time")
    plt.savefig(f"fid_vs_kid_plot_{dataset_name}_{str(im_size)}.png")
    plt.show()



    # save FID, KID and times at each epoch to compare to DDPM
    save_metrics("data",cum_times, fid_scores, kid_means, kid_stds, gpu_mb_alloc, gpu_mb_reserved, time_per_kimg, batch_size, dataset_name, str(im_size))
    
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

def save_metrics(path, times, fids, kids_mean, kids_stds, gpu_alloc, gpu_reserved, time_kimg, batch_size, dataset, res):
    save_path = f"{path}/stylegan2_{dataset}_{res}_metrics.pth"
    torch.save({'times':times,
                'fids':fids,
                'kids_mean':kids_mean,
                'kids_stds':kids_stds,
                'gpu_alloc':gpu_alloc,  #gpu alloc and reserved in mb
                'gpu_reserved':gpu_reserved,
                'time_kimg':time_kimg,
                'batch_size':batch_size}, save_path)
     
def save_model(path, mapping_net, generator, ema, best_model, dataset,res):
    save_path = f"{path}/stylegan2_{dataset}_{res}.pt"
    torch.save({'generator':generator.state_dict(),
                 'mapping_net':mapping_net.state_dict(),
                 'ema':ema.state_dict(),
                 'best_model':best_model}, save_path)

transform = T.Compose([
    T.Resize((299, 299)),  # InceptionV3 expects 299x299
    T.Normalize([-1]*3, [2]*3)
])

@torch.no_grad()
def load_real_images(real_dataloader, device, sample_size= 5000): # fit all real images on the device straight away to reduce I/O cost of fitting on device 
    real_images = []
    for imgs, _ in real_dataloader:
        real_images.append(imgs)
        if sum([i.size(0) for i in real_images]) >= sample_size:
            break
    real_images = torch.cat(real_images, dim=0)[:sample_size]
    real_images = transform(real_images).to(device)
    #real_images = transform(real_images)
    return real_images
@torch.no_grad()
def compute_fid(real_imgs, mapping_net, generator, device, res=64, mixing_prob=0.9, dim_w=512,batch_size=32, sample_size=5000): # use EMA for generator
    real_imgs.to(device)
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    num_blocks = int(math.log2(res))-1
    fid.update(real_imgs[:sample_size], real=True)
    num_batches = sample_size // batch_size # there is going to be slight imbalance (5000 real vs 5024 fake) but should not affect results and still yeild in accurate result
    for _ in range(num_batches):
        fake_images, _ = gen_images(batch_size, generator, num_blocks, mixing_prob, dim_w, mapping_net, device)
        fake_images = transform(fake_images)
        fid.update(fake_images, real=False)
        del fake_images
        torch.cuda.empty_cache
    fid_value = fid.compute()
    fid.reset()
    return fid_value.item()

@torch.no_grad()
def compute_kid(real_imgs, mapping_net, generator, device, res=64, mixing_prob=0.9, dim_w=512, batch_size=32, sample_size=500): # smaller sample size for kid
    real_imgs.to(device)
    kid = KernelInceptionDistance(feature=2048,subset_size=50, normalize=True).to(device)
    num_blocks = int(math.log2(res))-1
    kid.update(real_imgs[:sample_size], real=True)
    num_batches = sample_size // batch_size # there is going to be slight imbalance (5000 real vs 5024 fake) but should not affect results and still yeild in accurate result
    for _ in range(num_batches):
        fake_images, _ = gen_images(batch_size, generator, num_blocks, mixing_prob, dim_w, mapping_net, device)
        fake_images = transform(fake_images)
        kid.update(fake_images, real=False)
        del fake_images
        torch.cuda.empty_cache
    kid_values = kid.compute()
    kid.reset()
    return [kid_values[0].item(), kid_values[1].item()]
          
if __name__ == "__main__":
    train()
    #data_loader = dataset.get_loader(32,64,"CelebA")
    #print(f"len : {len(data_loader)}")
