from __future__ import annotations

import datetime
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

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

class LayerManager(nn.Module):
    def __init__(self) -> None:
        super(LayerManager, self).__init__()
        self.layers_conv: nn.ModuleList = nn.ModuleList()
        self.layers_norm: nn.ModuleList = nn.ModuleList()

    def add_conv(self, in_ch: int, out_ch: int, kernel_size: int = 3, transpose: bool = False) -> None:
        if transpose:
            layer_conv = nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride=2, padding=1, output_padding=1)
        else:
            layer_conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size // 2)
        self.layers_conv.append(layer_conv)

    def add_norm(self, num_features: int) -> None:
        self.layers_norm.append(nn.BatchNorm2d(num_features))

    def forward(self, x: Tensor) -> Tensor:
        for conv, norm in zip(self.layers_conv, self.layers_norm):
            x = F.tanh(norm(conv(x)))
        return x

class Encoder(nn.Module):
    def __init__(self, latent_dim: int = 100, in_channels: int = 3, dims_conv: int = 64, num_layers=4) -> None:
        super(Encoder, self).__init__()
        self.stem = nn.Conv2d(in_channels, dims_conv, kernel_size=3, padding=1)
        self.feature_extractor = LayerManager()

        # Add internal layers
        for _ in range(num_layers):
            self.feature_extractor.add_conv(dims_conv, dims_conv)
            self.feature_extractor.add_norm(dims_conv)

        # Final layer to reach latent space (using a simple convolution for spatial compression)
        self.bottleneck = nn.Conv2d(dims_conv, latent_dim, kernel_size=3, stride=2, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        x = F.tanh(self.stem(x))
        x = self.feature_extractor(x)
        x = self.bottleneck(x)
        return x

class Decoder(nn.Module):
    def __init__(self, latent_dim: int = 100, out_channels: int = 3, dims_conv: int = 64, num_layers=4) -> None:
        super(Decoder, self).__init__()
        # Use ConvTranspose to start upsampling from latent space
        self.stem = nn.ConvTranspose2d(latent_dim, dims_conv, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.feature_extractor = LayerManager()

        for _ in range(num_layers):
            self.feature_extractor.add_conv(dims_conv, dims_conv)
            self.feature_extractor.add_norm(dims_conv)

        self.head = nn.Conv2d(dims_conv, out_channels, kernel_size=3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        x = F.tanh(self.stem(x))
        x = self.feature_extractor(x)
        x = torch.tanh(self.head(x)) 
        return x

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x: Tensor) -> Tensor:
        z = self.encoder(x)
        z_hat = self.decoder(z)
        return z_hat


class TrainerAutoencoder():
    def __init__(self, model):
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: torch.nn.Module = model.to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer: torch.optim.Adam = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.total_itter: int = 0
        now = datetime.datetime.now()
        safe_time_str = now.strftime("%m-%d %H:%M:%S")
        csv_dir = "data/loss/"
        csv_dir_path = Path(csv_dir)
        csv_dir_path.mkdir(parents=True, exist_ok=True)
        self.csv_filename: str = csv_dir + safe_time_str + ".csv" 
        self.img_dir = "data/imgs/" + safe_time_str + "/"
        img_dir_path = Path(self.img_dir)
        img_dir_path.mkdir(parents=True, exist_ok=True)
        self.weights_path = "data/" + safe_time_str + ".pth"
        self.running_loss = 0

    def train(self, epochs, train_loader, test_loader=None):
        for epoch in range(epochs):
            self.model.train()
            # The decorator makes this call run the full loader loop
            self.train_one_epoch(epoch, epochs, train_loader, test_loader)

    def train_one_epoch(self, epoch, epochs, train_loader, test_loader=None):
        total_batches = len(train_loader)
        running_loss = 0.0

        for i, (data, target) in enumerate(train_loader):
            data = data.to(self.device) 

            loss_val = self.train_one_itter(data)
            self.running_loss += loss_val

            # Terminal Animation
            percent = 100 * (i + 1) / total_batches
            bar = '█' * int(percent / 5) + '-' * (20 - int(percent / 5))

            self.total_itter += 1
            move_up = "\033[3F" if self.total_itter > 1 else "\r" 

            sys.stdout.write(
                f"{move_up}Epoch [{epoch+1}/{epochs}] |{bar}| {percent:.1f}% [Itterations => {self.total_itter}]\n"
                    f"Current Loss: {loss_val:.4f} \n"
                    f"Running Loss: {running_loss:.4f} \n"
                    f"Average Loss: {(running_loss/self.total_itter):.4f}"
            )

            entry = LossEntry(
                epoch=epoch,
                itter=self.total_itter,
                loss_current=loss_val,
                loss_average=(running_loss/(self.total_itter)),
                running_loss=self.running_loss
            )
            LOG_INTERVAL = 200
            if self.total_itter%LOG_INTERVAL==0:
                LossEntry.save_to_csv(entry, self.csv_filename)
                if test_loader:
                    self.visualize_results(test_loader, True)
                torch.save(self.model.state_dict(), self.weights_path)
            sys.stdout.flush()

        avg_loss = running_loss / self.total_itter
        return avg_loss


    def train_one_itter(self, data): 
        """Processes one batch of data."""
        self.optimizer.zero_grad()
        outputs = self.model(data)

        # Reconstruction loss (input vs output)
        loss = self.criterion(outputs, data)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def visualize_results(self, test_loader, save_img=False, display_plt=False):
        if not test_loader: return
        self.model.eval()
        with torch.no_grad():
            data, _ = next(iter(test_loader))
            data = data.to(self.device)
            recon = self.model(data)

        fig, ax = plt.subplots(2, 7, figsize=(15, 4))
        for i in range(7):
            orig = (data[i].cpu().permute(1, 2, 0) * 0.5 + 0.5).clamp(0, 1)
            res = (recon[i].cpu().permute(1, 2, 0) * 0.5 + 0.5).clamp(0, 1)

            # orig = (data[i].cpu().numpy().transpose(((1,2,0))))
            # res = (recon[i].cpu().numpy().transpose(((1,2,0))))

            ax[0, i].imshow(orig)
            ax[1, i].imshow(res)
            ax[0, i].set_title("Original")
            ax[1, i].set_title("Reconstructed")
            ax[0, i].axis('off')
            ax[1, i].axis('off')
        if display_plt: 
            plt.show()
        if save_img:
            img_path = self.img_dir + f"I{self.total_itter}" + ".png"
            plt.savefig(img_path)

