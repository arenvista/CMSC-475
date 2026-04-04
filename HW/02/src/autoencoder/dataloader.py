import os
from typing import Any, Callable, List, Optional

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision import datasets

from dcgan.augment import Augmentor
from torch.utils.data import DataLoader

class CIFAR10DataModule:
    def __init__(self, batch_size: int = 32):
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # 3 channels for CIFAR
            transforms.Normalize((0.5, ), (0.5, )), # Normailize to [-1,1]
        ])

        train_set: datasets.CIFAR10 = datasets.CIFAR10(root="./data", train=True, download=True, transform=self.transform)
        self.train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)

        test_set: datasets.CIFAR10 = datasets.CIFAR10(root="./data", train=False, download=True, transform=self.transform)
        self.test_loader: DataLoader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)
