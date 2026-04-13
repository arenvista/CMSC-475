import torch
import torch.nn as nn
from torch import Tensor
from enum import Enum

class ModelType(Enum):
    DISCRIMINATOR=1,
    GENERATOR=2,

class LayerMgr(nn.Module): 
    def __init__(self, modeltype: ModelType) -> None:
        super(LayerMgr, self).__init__()
        self.layers_conv: nn.ModuleList = nn.ModuleList() 
        self.layers_norm: nn.ModuleList = nn.ModuleList() 
        self.modeltype = modeltype

    def add_conv(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1, padding: int = 0) -> None:
        layer_conv = None
        if self.modeltype == ModelType.GENERATOR:
            layer_conv = nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding)
        elif self.modeltype == ModelType.DISCRIMINATOR:
            layer_conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding)
        else: raise ValueError("Failed to add conv layer")
        self.layers_conv.append(layer_conv)

    def add_norm(self, num_features: int) -> None:
        layer_norm = nn.BatchNorm2d(num_features) 
        self.layers_norm.append(layer_norm)

    def forward(self, z: Tensor) -> Tensor:
        for i in range(len(self.layers_conv) - 1):
            z = self.layers_conv[i](z)
            
            if i < len(self.layers_norm):
                z = self.layers_norm[i](z)
                
            z = torch.relu(z)
            
        if len(self.layers_conv) > 0:
            z = self.layers_conv[-1](z)
            
        return z
