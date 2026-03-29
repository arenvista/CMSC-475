# src/dcgan/__init__.py

# Expose the primary classes and enums at the package level 
# so they can be imported directly from 'botc'.
from .dataloader import CustomDataset
from .augment import Augmentor
from .discriminator import Discriminator
from .generator import Generator
from .trainer import Trainer
