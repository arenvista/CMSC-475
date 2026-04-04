import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class GeneratorLayerManager(nn.Module): 
    def __init__(self) -> None:
        super(GeneratorLayerManager, self).__init__()
        self.layers_conv: nn.ModuleList = nn.ModuleList() 
        self.layers_norm: nn.ModuleList = nn.ModuleList() 

    def add_conv(self, in_ch: int, out_ch: int, kernel_size: int = 3) -> None:
        layer_conv = nn.ConvTranspose2d(
            in_ch, out_ch, kernel_size, padding=kernel_size // 2
        )
        self.layers_conv.append(layer_conv)

    def add_norm(self, num_features: int) -> None:
        layer_norm = nn.BatchNorm2d(num_features) 
        self.layers_norm.append(layer_norm)

    def forward(self, z: Tensor) -> Tensor:
        for i in range(len(self.layers_conv) - 1):
            z = self.layers_conv[i](z)
            
            if i < len(self.layers_norm):
                z = self.layers_norm[i](z)
                
            z = torch.tanh(z)
            
        if len(self.layers_conv) > 0:
            z = self.layers_conv[-1](z)
            
        return z

class Generator(nn.Module):
    def __init__(self, latent_dim: int = 100, out_channels: int = 3, dims_conv: int = 64, num_internal_layers=4, num_fcl_layers=1) -> None:
        print("Loading Generator")
        super(Generator, self).__init__()
        self.dims_conv = dims_conv 
        self.out_channels = out_channels
        self.num_internal_layers = num_internal_layers
        self.num_fcl_layers = num_fcl_layers

        self.stem: nn.ConvTranspose2d = nn.ConvTranspose2d(
            latent_dim, dims_conv, kernel_size=3, padding=1
        )
        
        self.feature_extractor: GeneratorLayerManager = GeneratorLayerManager()

        self.add_internal_layer()
        self.add_fcl()

    def add_internal_layer(self):
        for _ in range(self.num_internal_layers):
            self.feature_extractor.add_conv(self.dims_conv, self.dims_conv)
            self.feature_extractor.add_norm(self.dims_conv) 

    def add_fcl(self):
        self.feature_extractor.add_conv(self.dims_conv, self.out_channels)

    def forward(self, x: Tensor) -> Tensor:
        x = torch.tanh(self.stem(x))
        x = self.feature_extractor(x)
        return torch.tanh(x)
