import sys
from pathlib import Path
import argparse
import os

# from dcgan.generator import Generator
# from dcgan.discriminator import Discriminator
# from dcgan.dataloader import CustomDataset
# from dcgan.trainer import Trainer

from autoencoder.dataloader import CIFAR10DataModule
from autoencoder.autoencoder import TrainerAutoencoder
from autoencoder.autoencoder import Autoencoder


# def main_gan():
#     trainer = Trainer()
#     trainer.training_loop(5)

def main_autoecoder():
    data_mgr = CIFAR10DataModule()
    train_loader = data_mgr.train_loader
    test_loader = data_mgr.test_loader
    model = Autoencoder()
    trainer = TrainerAutoencoder(model)
    trainer.train(epochs=10, train_loader=train_loader, test_loader=test_loader)

if __name__ == "__main__":
    main_autoecoder()
