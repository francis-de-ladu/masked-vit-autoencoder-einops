import os

import torch
import torchvision
from torch.utils.data import DataLoader, random_split
from torchvision import transforms


def load_dataset(data_path='../../data', batch_size=64, valid_size=5000, normalize=False):
    if normalize is True:
        transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    else:
        transform = transforms.ToTensor()

    train = torchvision.datasets.CIFAR10(
        data_path, train=True, download=True, transform=transform)
    train, valid = random_split(train, [len(train) - valid_size, valid_size])

    test = torchvision.datasets.CIFAR10(
        data_path, train=False, download=True, transform=transform)

    kwargs = dict(
        batch_size=batch_size,
        pin_memory=True,
        num_workers=os.cpu_count() - 2
    )
    train_loader = DataLoader(train, shuffle=True, **kwargs)
    valid_loader = DataLoader(valid, shuffle=False, **kwargs)
    test_loader = DataLoader(test, shuffle=False, **kwargs)

    return train_loader, valid_loader, test_loader


def add_noise(orig, noise_factor):
    noisy = orig + torch.randn_like(orig) * noise_factor
    noisy = torch.clip(noisy, 0., 1.)
    return noisy
