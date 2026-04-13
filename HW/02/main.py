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

from view_loss import *

def main_gan(opts=None):
    if opts == "delux":
        print("Running GAN w/ Delux")
        trainer = Trainer(augment_mode="delux")
        trainer.training_loop(200)
    elif opts == "simple":
        print("Running GAN w/ Simple")
        trainer = Trainer()
        trainer.training_loop(200)
    elif opts == "generate":
        print("Select weight for generator.")
        wght_path = fuzzy_find_file("./data/gan/weights")
        print(f"Selected {wght_path}")
        trainer = Trainer()
        trainer.generate_img()
    elif opts is not None:
        raise ValueError("Not a valid opts")

def main_autoecoder():
    print("Running Autoencoder")
    data_mgr = CIFAR10DataModule()
    train_loader = data_mgr.train_loader
    test_loader = data_mgr.test_loader
    model = Autoencoder()
    trainer = TrainerAutoencoder(model)
    trainer.train(epochs=50, train_loader=train_loader, test_loader=test_loader)

# testing; update this later
def latent(wght_pth):
    print("Calling latent")
    data_mgr = CIFAR10DataModule()
    test_loader = data_mgr.test_loader
    model = Autoencoder(wght_pth)

def main():
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
    parser.add_argument(
        "-l", "--latent", 
        action="store_true", 
        help="Enable latent."
    )
    parser.add_argument(
        "-o", "--opts",
        help="pass an option as a string"
    )


    args = parser.parse_args()
    opts = args.opts
    if args.gan:
        main_gan(opts)
    if args.auto:
        main_autoecoder()
    if args.latent:
        latent("data/wght.pth")

if __name__ == "__main__":
    main()
