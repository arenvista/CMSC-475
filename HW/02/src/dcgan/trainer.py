from __future__ import annotations
from PIL import Image
from torchvision.utils import save_image
import pandas as pd
import os
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import random
from dcgan.generator import Generator
from dcgan.discriminator import Discriminator
from dcgan.dataloader import CustomDataset
from dcgan.dataloader import Augmentor
from datetime import datetime

from dataclasses import asdict, dataclass
from pathlib import Path
@dataclass
class LossEntry:
    epoch: int = 0
    itter: int = 0
    loss_current: float = 0
    loss_average: float = 0
    running_loss: float = 0
    @classmethod
    def save_to_csv(cls, entry: LossEntry, filename: str):
        data_dicts = [asdict(entry)] 
        df = pd.DataFrame(data_dicts)
        file_exists = os.path.exists(filename)
        df.to_csv(filename, mode="a", header=not file_exists, index=False)

class Trainer():
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu", augment_mode = None):
        print("Loading Trainer")
        self.device = device
        self.latent_dim = 100
        
        # Models
        self.generator = Generator().to(self.device)
        self.discriminator = Discriminator().to(self.device)
        
        # Optimizers
        self.optim_g = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optim_d = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        
        # Loss & Data
        self.loss_func = nn.BCELoss()
        augmentor = Augmentor(64)
        transform = None
        if augment_mode != "delux": transform = augmentor.simple_transform()
        if augment_mode == "delux": transform = augmentor.delux_transform()
        self.loader = DataLoader(CustomDataset("data/grump", transform), batch_size=64, shuffle=True)


    def generate_img(self, batch_size=10):
        random.seed(10)
        noise = self.sample_noise(batch_size)
        fake_imgs = self.generator(noise)
        save_image(fake_imgs, "grid.png", nrow=5, normalize=True, value_range=(-1, 1))

    def sample_noise(self, batch_size):
        return torch.randn(batch_size, self.latent_dim, 1, 1).to(self.device)

    def train_discriminator(self, real_imgs):
        batch_size = real_imgs.size(0)
        self.optim_d.zero_grad()

        labels_real = torch.ones(batch_size, 1).to(self.device)
        output_real = self.discriminator.forward(real_imgs)
        loss_real = self.loss_func(output_real, labels_real)

        noise = self.sample_noise(batch_size)
        fake_imgs = self.generator(noise)
        labels_fake = torch.zeros(batch_size, 1).to(self.device)
        output_fake = self.discriminator(fake_imgs.detach())
        loss_fake = self.loss_func(output_fake, labels_fake)

        # Backprop
        d_loss = loss_real + loss_fake
        d_loss.backward()
        self.optim_d.step()
        return d_loss.item()

    def train_generator(self, batch_size):
        self.optim_g.zero_grad()
        
        noise = self.sample_noise(batch_size)
        fake_imgs = self.generator(noise)
        
        labels = torch.ones(batch_size, 1).to(self.device)
        output = self.discriminator(fake_imgs)
        
        g_loss = self.loss_func(output, labels)
        g_loss.backward()
        self.optim_g.step()
        return g_loss.item()

    def training_loop(self, epochs):
        running_loss_g = 0
        running_loss_d = 0
        total_itter = 0
        now = datetime.now()
        safe_time_str = now.strftime("%m-%d %H:%M:%S")
        
        for epoch in tqdm(range(epochs), desc="Epochs"):  
            for i, real_imgs in enumerate(self.loader):  
                total_itter += 1
                real_imgs = real_imgs.to(self.device)
                batch_size = real_imgs.size(0)

                d_loss = self.train_discriminator(real_imgs)
                g_loss = self.train_generator(batch_size)

                running_loss_d += d_loss
                running_loss_g += g_loss
                
                if i % 100 == 0:
                    # print(f"Epoch [{epoch}/{epochs}] Batch {i} | D Loss: {d_loss:.4f} G Loss: {g_loss:.4f}")
                    entry = LossEntry(
                        epoch=epoch,
                        itter=total_itter,
                        loss_current=d_loss,
                        loss_average=(running_loss_d / total_itter),
                        running_loss=running_loss_d
                    )
                    LossEntry.save_to_csv(entry, f"data/gan/training_log_discrim_{safe_time_str}.csv")
                    entry = LossEntry(
                        epoch=epoch,
                        itter=total_itter,
                        loss_current=g_loss, 
                        loss_average=(running_loss_g / total_itter),
                        running_loss=running_loss_g
                    )
                    LossEntry.save_to_csv(entry, f"data/gan/training_log_generator_{safe_time_str}.csv")


                if total_itter % 200 == 0:
                    torch.save(self.generator.state_dict(), f"data/gan/weights/generator_weights_{safe_time_str}_{total_itter}.pth")
                    torch.save(self.discriminator.state_dict(), f"data/gan/weights/discriminator_weights_{safe_time_str}_{total_itter}.pth")
