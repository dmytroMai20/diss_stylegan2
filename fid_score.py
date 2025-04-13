from torchmetrics.image.fid import FrechetInceptionDistance
import torchvision.utils as vutils
import torchvision.transforms as T
import os
import torch
from train import gen_images
import math

# transform images from [-1, 1] to [0, 1]
transform = T.Compose([
    T.Resize((299, 299)),  # InceptionV3 expects 299x299
    T.Normalize([-1]*3, [2]*3)
])

def load_real_images(real_dataloader, device, sample_size= 5000): # fit all real images on the device straight away to reduce I/O cost of fitting on device 
    real_images = []
    with torch.no_grad():
        for imgs, _ in real_dataloader:
            real_images.append(imgs)
            if sum([i.size(0) for i in real_images]) >= sample_size:
                break
    real_images = torch.cat(real_images, dim=0)[:sample_size]
    real_images = transform(real_images).to(device)
    return real_images
def compute_fid(real_loader, mapping_net, generator, device, res=64, mixing_prob=0.9, dim_w=512,batch_size=32, sample_size=5000): # use EMA for generator
    real_imgs = load_real_images(real_loader, device, sample_size)
    fid = FrechetInceptionDistance(feature=2048).to(device)
    num_blocks = int(math.log2(res))-1
    fid.update(real_imgs[:sample_size], real=True)

    num_batches = sample_size // batch_size # there is going to be slight imbalance (5000 real vs 5024 fake) but should not affect results and still yeild in accurate result
    for _ in range(num_batches):
        with torch.no_grad():
            fake_images, _ = gen_images(batch_size, generator, num_blocks, mixing_prob, dim_w, mapping_net, device)
        fake_images = transform(fake_images)
        fid.update(fake_images, real=False)
    
    return fid.compute().item()
