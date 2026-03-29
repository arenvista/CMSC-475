import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class DiscriminatorLayerManager(nn.Module): 
    def __init__(self) -> None:
        super(DiscriminatorLayerManager, self).__init__()
        self.layers_conv: nn.ModuleList = nn.ModuleList() 
        self.layers_norm: nn.ModuleList = nn.ModuleList() 

    def add_conv(self, in_ch: int, out_ch: int, kernel_size: int = 3) -> None:
        layer_conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size // 2)
        self.layers_conv.append(layer_conv)

    def add_norm(self, num_features: int) -> None:
        layer_norm = nn.BatchNorm2d(num_features) 
        self.layers_norm.append(layer_norm)

    def forward(self, z: Tensor) -> Tensor:
        for i in range(len(self.layers_conv) - 1):
            z = self.layers_conv[i](z)
            # Safely apply normalization if a corresponding norm layer exists
            if i < len(self.layers_norm):
                z = self.layers_norm[i](z)
            z = F.relu(z)
        
        if len(self.layers_conv) > 0:
            z = self.layers_conv[-1](z)
        
        return z

class Discriminator(nn.Module):
    def __init__(self, in_channels: int = 3, dims_conv: int = 64) -> None:
        super(Discriminator, self).__init__()
        
        self.stem: nn.Conv2d = nn.Conv2d(in_channels, dims_conv, kernel_size=3, padding=1)
        self.feature_extractor: DiscriminatorLayerManager = DiscriminatorLayerManager()
        
        self.feature_extractor.add_conv(dims_conv, dims_conv)
        self.feature_extractor.add_norm(dims_conv) 
        self.feature_extractor.add_conv(dims_conv, dims_conv * 2)

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.stem(x))
        x = self.feature_extractor(x)
        return x
