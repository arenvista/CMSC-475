import torch
import torch.nn as nn
from torch import Tensor
import torchvision.transforms as T
from dcgan.layermgr import *

class Generator(nn.Module):
    def __init__(self, latent_dim: int = 100, out_channels: int = 3, dims_conv: int = 64, num_internal_layers=4, num_fcl_layers=1) -> None:
        print("Loading Generator")
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.dims_conv = dims_conv 
        self.out_channels = out_channels
        self.num_internal_layers = num_internal_layers
        self.num_fcl_layers = num_fcl_layers

        self.feature_extractor: LayerMgr = LayerMgr(ModelType.GENERATOR)

        self.add_internal_layer()
        self.add_fcl()

    def add_internal_layer(self):
        # 1x1 => 4x4
        self.feature_extractor.add_conv(self.latent_dim, self.dims_conv * 8, 4, 1, 0)
        self.feature_extractor.add_norm(self.dims_conv * 8) 

        # 4x4 => 8x8
        self.feature_extractor.add_conv(self.dims_conv * 8, self.dims_conv * 4, 4, 2, 1)
        self.feature_extractor.add_norm(self.dims_conv * 4) 

        # 8x8 => 16x16
        self.feature_extractor.add_conv(self.dims_conv * 4, self.dims_conv * 2, 4, 2, 1)
        self.feature_extractor.add_norm(self.dims_conv * 2) 

        # 16x16 => 32x32
        self.feature_extractor.add_conv(self.dims_conv * 2, self.dims_conv, 4, 2, 1)
        self.feature_extractor.add_norm(self.dims_conv) 

    def add_fcl(self):
        # 32x32 => 64x64
        self.feature_extractor.add_conv(self.dims_conv, self.out_channels, 4, 2, 1)

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() == 2: x = x.view(-1, self.latent_dim,1,1)
        x = self.feature_extractor(x)
        return torch.tanh(x)
