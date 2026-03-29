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
        # ConvTranspose2d is used for upsampling/deconvolution
        layer_conv = nn.ConvTranspose2d(
            in_ch, out_ch, kernel_size, padding=kernel_size // 2
        )
        self.layers_conv.append(layer_conv)

    def add_norm(self, num_features: int) -> None:
        layer_norm = nn.BatchNorm2d(num_features) 
        self.layers_norm.append(layer_norm)

    def forward(self, z: Tensor) -> Tensor:
        # Loop through all but the last layer
        for i in range(len(self.layers_conv) - 1):
            z = self.layers_conv[i](z)
            
            # Use current layer index to access corresponding normalization
            if i < len(self.layers_norm):
                z = self.layers_norm[i](z)
                
            # tanh activation (F.tanh is technically legacy)
            z = torch.tanh(z)
            
        # The final layer (the output) doesn't typically get activated
        if len(self.layers_conv) > 0:
            z = self.layers_conv[-1](z)
            
        return z

class Generator(nn.Module):
    def __init__(self, in_channels: int = 3, dims_conv: int = 64) -> None:
        super(Generator, self).__init__()
        
        # Upsampling stem
        self.stem: nn.ConvTranspose2d = nn.ConvTranspose2d(
            in_channels, dims_conv, kernel_size=3, padding=1
        )
        
        self.feature_extractor: GeneratorLayerManager = GeneratorLayerManager()
        
        # Layer 1: ConvTranspose -> Norm -> Tanh
        self.feature_extractor.add_conv(dims_conv, dims_conv)
        self.feature_extractor.add_norm(dims_conv) 
        
        # Layer 2: Final output convolution (e.g. mapping to final channel count)
        self.feature_extractor.add_conv(dims_conv, dims_conv * 2)

    def forward(self, x: Tensor) -> Tensor:
        # Applying tanh after the stem
        x = torch.tanh(self.stem(x))
        x = self.feature_extractor(x)
        return x
