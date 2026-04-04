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
    def __init__(self, in_channels: int = 3, dims_conv: int = 64, num_internal_layers: int = 4, num_fcl_layers: int = 1) -> None:
        print("Loading Discriminator")
        super(Discriminator, self).__init__()
        
        self.stem: nn.Conv2d = nn.Conv2d(in_channels, dims_conv, kernel_size=3, padding=1)
        self.feature_extractor: DiscriminatorLayerManager = DiscriminatorLayerManager()
        
        self.dims_conv = dims_conv 
        self.in_channels = in_channels
        self.num_internal_layers = num_internal_layers
        self.num_fcl_layers = num_fcl_layers

        # Keep the internal feature layers
        self.add_internal_layer()
        
        # REMOVED: self.add_fcl() 
        # We don't want to go back to 3 channels (RGB) in the Discriminator.

        # Shrink the feature maps down to 1x1
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # FIXED: Match the input features (64) and set output to 1 for binary classification
        self.classifier = nn.Linear(dims_conv, 1)

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.stem(x))
        x = self.feature_extractor(x)  # Now shape is [batch_size, 64, 224, 224]
        
        x = self.pool(x)               # Shape becomes [batch_size, 64, 1, 1]
        x = torch.flatten(x, 1)        # Shape becomes [batch_size, 64]
        
        x = self.classifier(x)         # Shape becomes [batch_size, 1]
        x = torch.sigmoid(x)           
        
        return x

    def add_internal_layer(self):
        for _ in range(self.num_internal_layers):
            self.feature_extractor.add_conv(self.dims_conv, self.dims_conv)
            self.feature_extractor.add_norm(self.dims_conv) 

    # We can keep the method, but don't call it in __init__ for the Discriminator
    def add_fcl(self):
        self.feature_extractor.add_conv(self.dims_conv, self.in_channels)
