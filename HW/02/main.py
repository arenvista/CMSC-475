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
from autoencoder.autoencoder import Autoencoder, calculate_class_distances, plot_latent_distances
from kmeans import kmeans
from autoencoder.autoencoder import cluster_and_plot_cifar



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

def latent(wght_pth):
    print("Calling latent")
    data_mgr = CIFAR10DataModule()
    test_loader = data_mgr.test_loader
    model = Autoencoder(wght_pth)
    distances = calculate_class_distances(model, test_loader)
    # plot_latent_distances(distances)
    cluster_and_plot_cifar(test_loader, model)


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
    parser.add_argument(
        "-l", "--latent", 
        action="store_true", 
        help="Enable latent."
    )


    # 3. Parse the arguments
    args = parser.parse_args()
    if args.gan:
        main_gan()
    if args.auto:
        main_autoecoder()
    if args.kmean:
        kmeans()
    if args.latent:
        latent("data/wght.pth")

if __name__ == "__main__":
    main()
