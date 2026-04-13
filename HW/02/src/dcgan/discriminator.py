import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from dcgan.layermgr import *

class Discriminator(nn.Module):
    def __init__(self, in_channels: int = 3, dims_conv: int = 64, num_internal_layers: int = 4, num_fcl_layers: int = 1) -> None:
        print("Loading Discriminator")
        super(Discriminator, self).__init__()
        
        self.feature_extractor: LayerMgr = LayerMgr(ModelType.DISCRIMINATOR)
        
        self.dims_conv = dims_conv 
        self.in_channels = in_channels
        self.num_internal_layers = num_internal_layers
        self.num_fcl_layers = num_fcl_layers

        self.add_internal_layer()

        self.classifier = nn.Linear(dims_conv, 1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.feature_extractor(x)  # Now shape is [batch_size, 64, 224, 224]
        x = torch.flatten(x,1)
        x = self.classifier(x)         # Shape becomes [batch_size, 1]
        x = torch.sigmoid(x)           
        return x

    def add_internal_layer(self):
        # 64x64 -> 32x32
        self.feature_extractor.add_conv(self.in_channels, self.dims_conv, 4, 2, 1)
        self.feature_extractor.add_norm(self.dims_conv) 

        # 32x32 -> 16x16
        self.feature_extractor.add_conv(self.dims_conv, self.dims_conv * 2, 4, 2, 1)
        self.feature_extractor.add_norm(self.dims_conv * 2) 

        # 16x16 -> 8x8
        self.feature_extractor.add_conv(self.dims_conv * 2, self.dims_conv * 4, 4, 2, 1)
        self.feature_extractor.add_norm(self.dims_conv * 4)

        # 8x8 -> 4x4
        self.feature_extractor.add_conv(self.dims_conv * 4, self.dims_conv * 8, 4, 2, 1)
        self.feature_extractor.add_norm(self.dims_conv * 8)

        # 4x4 -> 1x1 (Final Conv)
        self.feature_extractor.add_conv(self.dims_conv * 8, self.dims_conv, 4, 1, 0)

    def add_fcl(self):
        self.feature_extractor.add_conv(self.dims_conv, self.in_channels)
