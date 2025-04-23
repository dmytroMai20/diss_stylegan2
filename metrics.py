import torch
import torchvision.transforms as T
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
import math
import os
from torchvision.utils import save_image

transform = T.Compose([
    T.Resize((299, 299)),  # InceptionV3 expects 299x299
    T.Normalize([-1]*3, [2]*3)
])
@torch.no_grad()
def compute_kid(real_imgs, fake_imgs, device, res=64, batch_size=32, sample_size=500): # smaller sample size for kid
    #real_imgs.to(device) # should already be on device and transformed from load_real_images
    #real_imgs = transform(real_imgs)
    fake_imgs = transform(fake_imgs)
    kid = KernelInceptionDistance(feature=2048,subset_size=50, normalize=True).to(device)
    kid.update(real_imgs[:sample_size], real=True)
    kid.update(fake_imgs[:sample_size], real=False)
    #del fake_imgs
    #torch.cuda.empty_cache
    kid_values = kid.compute()
    kid.reset()
    del kid
    return [kid_values[0].item(), kid_values[1].item()]

@torch.no_grad()
def compute_fid(real_imgs, fake_imgs, device, sample_size=500): # smaller sample size for kid
    #real_imgs.to(device) # should already be on device and transformed from load_real_images
    #real_imgs = transform(real_imgs)
    fake_imgs = transform(fake_imgs)
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    fid.update(real_imgs[:sample_size], real=True)
    fid.update(fake_imgs[:sample_size], real=False)
    #del fake_imgs
    #torch.cuda.empty_cache
    fid_value = fid.compute().item()
    del fid
    return fid_value

def save_generated_images(images, epoch,dataset,res,t_max, folder='data'):
    if images.min() < 0:
        images = (images + 1) / 2  # Assuming input is in [-1, 1]

    # Create directory for this epoch
    epoch_dir = os.path.join(folder,f'stylegan_{dataset}_{str(res)}_{str(t_max)}', f'epoch_{str(epoch)}')
    os.makedirs(epoch_dir, exist_ok=True)

    # Save each image
    for i in range(images.size(0)):
        save_path = os.path.join(epoch_dir, f'image_{i:04d}.png')
        save_image(images[i], save_path)
    print(f"Epoch images saved to {epoch_dir}")