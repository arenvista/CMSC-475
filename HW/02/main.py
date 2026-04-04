import sys
from pathlib import Path
import argparse
import os

from dcgan.generator import Generator
from dcgan.discriminator import Discriminator
from dcgan.dataloader import CustomDataset
from dcgan.trainer import Trainer

from autoencoder.dataloader import CIFAR10DataModule
from autoencoder.autoencoder import TrainerAutoencoder
from autoencoder.autoencoder import Autoencoder

from kmeans import kmeans



def main_gan():
    trainer = Trainer()
    trainer.training_loop(5)

def main_autoecoder():
    data_mgr = CIFAR10DataModule()
    train_loader = data_mgr.train_loader
    test_loader = data_mgr.test_loader
    model = Autoencoder()
    trainer = TrainerAutoencoder(model)
    trainer.train(epochs=50, train_loader=train_loader, test_loader=test_loader)

def kmeans_start():
    kmeans()

def main():
    # 1. Create the parser
    parser = argparse.ArgumentParser(
        prog="FileProcessor",
        description="A simple script to process text files.",
        epilog="Thanks for using FileProcessor!"
    )

    parser.add_argument(
        "-g", "--gan", 
        action="store_true", 
        help="Enable verbose."
    )
    parser.add_argument(
        "-a", "--auto", 
        action="store_true", 
        help="Enable autoencoder."
    )
    parser.add_argument(
        "-k", "--kmean", 
        action="store_true", 
        help="Enable kmeans."
    )


    # 3. Parse the arguments
    args = parser.parse_args()
    if args.gan:
        main_gan()
    if args.auto:
        main_autoecoder()
    if args.kmean:
        kmeans()

if __name__ == "__main__":
    main()
