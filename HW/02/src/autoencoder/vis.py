from __future__ import annotations

import datetime
import os
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch import Tensor

def calculate_class_distances(model: nn.Module, test_loader, device=None, img_path="test.png"):
    """Calculates average L2 distances of latent vectors to class centroids and saves a reconstruction plot."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    model.eval()
    encoder = model.encoder
    z_by_class = defaultdict(list)
    
    print("Extracting latent vectors...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            z = encoder(images) 
            z = torch.flatten(z, start_dim=1).cpu()
            for i, label in enumerate(labels.tolist()):
                z_by_class[label].append(z[i])

    results = {}
    for c, z_list in z_by_class.items():
        Z_c = torch.stack(z_list) # Shape: (n, latent_dim)
        z_c_mean = torch.mean(Z_c, dim=0)
        # https://docs.pytorch.org/docs/stable/generated/torch.norm.html => Bless Dr. Sousedik
        distances = torch.norm(Z_c - z_c_mean, p=2, dim=1)
        results[c] = torch.mean(distances).item()

    print("Generating reconstruction plot...")
    with torch.no_grad():
        data, _ = next(iter(test_loader))
        data = data.to(device)
        recon = model(data)
        
    num_images = min(10, data.shape[0])
    fig, ax = plt.subplots(2, num_images, figsize=(1.5 * num_images, 4))
    
    if num_images == 1:
        ax = ax[:, None]
        
    for i in range(num_images):
        orig = (data[i].cpu().permute(1, 2, 0) * 0.5 + 0.5).clamp(0, 1)
        res = (recon[i].cpu().permute(1, 2, 0) * 0.5 + 0.5).clamp(0, 1)

        ax[0, i].imshow(orig.numpy())
        ax[1, i].imshow(res.numpy())
        ax[0, i].set_title(f"Ori {i}")
        ax[1, i].set_title(f"Rec {i}")
        ax[0, i].axis('off')
        ax[1, i].axis('off')
        
    plt.tight_layout()
    plt.savefig(img_path)
    plt.close(fig)

    min_class = min(results, key=results.get)
    max_class = max(results, key=results.get)
    
    print(f"\n{'Class':<10} | {'Mean L2 Distance (dist_c)':<25}")
    print("-" * 40)
    for c in sorted(results.keys()):
        print(f"{c:<10} | {results[c]:.4f}")
    
    print("\n--- Summary ---")
    print(f"Least distance: Class {min_class} ({results[min_class]:.4f})")
    print(f"Highest distance: Class {max_class} ({results[max_class]:.4f})")
    
    return results


def cluster_and_plot_cifar(train_loader, model: nn.Module) -> pd.DataFrame:
    """Runs KMeans, PCA, and TSNE on latent encodings and visualizes the results."""
    all_images = []
    all_labels = []

    # 1. Encode and Flatten
    for images, labels in train_loader:
        with torch.no_grad():
            encoded = model.encoder(images)
        flattened = encoded.view(encoded.size(0), -1).numpy()
        all_images.append(flattened)
        all_labels.append(labels.numpy())

    X = np.concatenate(all_images, axis=0)
    true_labels = np.concatenate(all_labels, axis=0)

    num_clusters = 10
    kmeans_model = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto')
    cluster_ids = kmeans_model.fit_predict(X)
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(X)

    df_pca = pd.DataFrame({
        'PCA1': pca_results[:, 0],
        'PCA2': pca_results[:, 1],
        'Cluster': cluster_ids.astype(str),
        'True Label': true_labels.astype(str)
    })
    
    plot_label_cluster_focus(df_pca)

    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    sns.scatterplot(data=df_pca, x='PCA1', y='PCA2', hue='Cluster', palette='tab10', alpha=0.6, s=10)
    plt.title('CIFAR-10: K-Means Clusters (PCA Projection)')

    plt.subplot(1, 2, 2)
    sns.scatterplot(data=df_pca, x='PCA1', y='PCA2', hue='True Label', palette='tab10', alpha=0.6, s=10)
    plt.title('CIFAR-10: True Labels (PCA Projection)')

    plt.tight_layout()
    plt.show()

    #TSNE 
    tsne = TSNE(n_components=2)
    tsne_results = tsne.fit_transform(X)

    df_tsne = pd.DataFrame({
        'tsne1': tsne_results[:, 0],
        'tsne2': tsne_results[:, 1],
        'Cluster': cluster_ids.astype(str),
        'True Label': true_labels.astype(str)
    })

    # TSNE Plots
    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    sns.scatterplot(data=df_tsne, x='tsne1', y='tsne2', hue='Cluster', palette='tab10', alpha=0.6, s=10)
    plt.title('CIFAR-10: K-Means Clusters (t-SNE Projection)')

    plt.subplot(1, 2, 2)
    sns.scatterplot(data=df_tsne, x='tsne1', y='tsne2', hue='True Label', palette='tab10', alpha=0.6, s=10)
    plt.title('CIFAR-10: True Labels (t-SNE Projection)')

    plt.tight_layout()
    plt.show()
    
    return df_pca


def plot_label_cluster_focus(df: pd.DataFrame):
    catppuccin_frappe = [
        "#ca9ee6", "#8caaee", "#85c1dc", "#99d1db", "#81c8be", 
        "#a6d189", "#e5c890", "#ef9f76", "#f2d5cf", "#eebebe", 
        "#e78284", "#ea999c", "#f4b8e4"
    ]

    unique_clusters = sorted(df['True Label'].unique())

    for itter, target_lbl in enumerate(df["True Label"].unique()):
        plt.figure(figsize=(16, 6))
        
        color_idx = itter % len(catppuccin_frappe)
        palette = {c: catppuccin_frappe[color_idx] if c == target_lbl else "#D3D3D3" for c in unique_clusters}

        df['is_target'] = (df['True Label'] == target_lbl)

        plt.subplot(1, 2, 1)
        sns.scatterplot(
            data=df.sort_values('is_target'),
            x='PCA1', 
            y='PCA2', 
            hue='True Label', 
            palette=palette,
            alpha=0.6, s=10
        )
        plt.title(f'CIFAR-10: Cluster {target_lbl} (PCA Projection)')

        is_lbl = df["True Label"] == target_lbl
        res = is_lbl.groupby(df["Cluster"]).sum().astype(int).reset_index(name="Count")
        
        plt.subplot(1, 2, 2)
        sns.barplot(data=res, x='Cluster', y='Count', hue='Cluster', palette='tab10', legend=False)
        plt.title(f'CIFAR-10: {target_lbl} (Cluster Distribution)')

        plt.tight_layout()
        plt.savefig(f"joint_cluster_{target_lbl}.png")
        plt.close()
