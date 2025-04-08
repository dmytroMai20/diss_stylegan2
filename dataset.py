import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.datasets import FashionMNIST
from torchvision.datasets import CelebA
from torchvision.datasets import STL10
from torch.utils.data import DataLoader

def get_loader(batch_s,res, data="FashionMNIST"):

    # Define transformations
    if data == "FashionMNIST":
        transform = transforms.Compose([
            transforms.Resize(32),  # Resize to desired resolution
            transforms.CenterCrop(32),  # Crop to maintain face alignment
            transforms.ToTensor(),  # Convert to tensor
            transforms.Normalize([0.5], [0.5])  # Normalize between -1 and 1
        ])

        # Download and load the dataset
        dataset = FashionMNIST(root="./data", train=True, download=True, transform=transform)

        # Create a DataLoader
        dataloader = DataLoader(dataset, batch_size=batch_s, shuffle=True, num_workers=1)
        return dataloader
    elif data == "CelebA":
        transform = transforms.Compose([
        transforms.Resize(128),  # Resize to 128x128 (common for GANs)
        transforms.CenterCrop(128),  # Crop to maintain face alignment
        transforms.ToTensor(),  # Convert to tensor
        #transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalize RGB channels
        ])

        # Download and load the dataset
        dataset = datasets.ImageFolder(root="./data/celeba", transform=transform)

        # Create a DataLoader
        dataloader = DataLoader(dataset, batch_size=batch_s, shuffle=True, num_workers=1, pin_memory=True)
        return dataloader
    elif data == "STL10":
        transform = transforms.Compose([
        transforms.Resize(res),  # Resize to 128x128 (common for GANs)
        #transforms.CenterCrop(64),  # Crop to maintain face alignment
        transforms.ToTensor(),  # Convert to tensor
        #transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalize RGB channels
        ])

        # Download and load the dataset
        dataset = STL10(root="./data", split="unlabeled", download=True, transform=transform)
        # Create a DataLoader
        dataloader = DataLoader(dataset, batch_size=batch_s, shuffle=True, num_workers=1, pin_memory=True)
        return dataloader
    else:
        return "error"