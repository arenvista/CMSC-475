import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
# Assuming these are your local files
from dcgan.generator import Generator
from dcgan.discriminator import Discriminator
from dcgan.dataloader import CustomDataset

class Trainer():
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
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
        self.loss_func = nn.BCELoss() # Standard for GANs
        self.loader = DataLoader(CustomDataset("data/grump"), batch_size=64, shuffle=True)

    def sample_noise(self, batch_size):
        return torch.randn(batch_size, self.latent_dim, 1, 1).to(self.device)

    def train_discriminator(self, real_imgs):
        batch_size = real_imgs.size(0)
        self.optim_d.zero_grad()

        labels_real = torch.ones(batch_size, 1).to(self.device)
        print("finding outputs discrim")
        output_real = self.discriminator.forward(real_imgs)
        print("finding loss")
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
        
        # The Generator wants the Discriminator to think these are REAL (1.0)
        labels = torch.ones(batch_size, 1).to(self.device)
        output = self.discriminator(fake_imgs)
        
        g_loss = self.loss_func(output, labels)
        g_loss.backward()
        self.optim_g.step()
        return g_loss.item()

    def training_loop(self, epochs):
        for epoch in range(epochs):
            for i, real_imgs in enumerate(self.loader):
                real_imgs = real_imgs.to(self.device)
                batch_size = real_imgs.size(0)

                d_loss = self.train_discriminator(real_imgs)
                g_loss = self.train_generator(batch_size)

                if i % 100 == 0:
                    print(f"Epoch [{epoch}/{epochs}] Batch {i} | D Loss: {d_loss:.4f} G Loss: {g_loss:.4f}")
