import os
import torch
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
from torchattacks import PGD
from utils import get_imagenet_data, get_accuracy, save_comparison_grid, get_pred


def get_device() -> str:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device}")
    return device


def load_model(device: str) -> torch.nn.Module:
    print("[Model] Loading pretrained ResNet18...")
    model: torch.nn.Module = models.resnet18(pretrained=True).to(device).eval()
    print("[Model] Loaded and set to eval mode")
    return model


def load_data(device: str) -> tuple[torch.Tensor, torch.Tensor]:
    print("[Data] Loading ImageNet samples...")
    images, labels = get_imagenet_data()
    images = images.to(device)
    labels = labels.to(device)
    print(f"[Data] Loaded images with shape={tuple(images.shape)}, labels shape={tuple(labels.shape)}")
    return images, labels


def imshow(img_tensor: torch.Tensor, title: str = "", save_path: str = "output/adv_display.png") -> None:
    """Display a single CHW float tensor as a matplotlib image, matching notebook style."""
    img = img_tensor.clone().detach().cpu()
    # Convert [C, H, W] float in [0,1] → [H, W, C] for matplotlib
    img_np = img.permute(1, 2, 0).numpy()
    img_np = np.clip(img_np, 0, 1)

    plt.figure()
    plt.title(title)
    plt.imshow(img_np)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()
    print(f"[Display] Saved to {save_path}")


def attack_targeted(
    model: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    print("[Attack] Initializing targeted PGD attack...")
    atk: PGD = PGD(model, eps=8/255, alpha=2/255, steps=10, random_start=True)

    atk.set_normalization_used(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # Target: shift each label by +1 mod 1000
    target_map_function = lambda imgs, lbls: (lbls + 1) % 1000
    atk.set_mode_targeted_by_function(target_map_function=target_map_function)
    print(f"[Attack] Configured: {atk}")

    print("[Attack] Generating targeted adversarial images...")
    adv_images: torch.Tensor = atk(images, labels)

    idx: int = 0
    target_label: int = int(((labels[idx] + 1) % 1000).item())

    pre: torch.Tensor = get_pred(model, adv_images[idx:idx+1], images.device)
    print(f"[Attack] True label: {labels[idx].item()}, Target label: {target_label}, Prediction: {pre.item()}")

    # --- Notebook cell [9] style: single image display with True/Pre title ---
    imshow(
        adv_images[idx],
        title="True:%d, Pre:%d" % (int(labels[idx].item()), int(pre.item())),
        save_path="output/adv_display.png",
    )

    # Also save the 3-panel comparison grid
    print("[Attack] Saving comparison grid to output/ ...")
    save_comparison_grid(
        clean_images=images[idx:idx+1].cpu(),
        adv_images=adv_images[idx:idx+1].cpu(),
        output_dir="output",
        prefix="targeted",
        true_labels=labels[idx:idx+1].cpu(),
        pred_labels=pre.cpu(),
    )
    print("[Attack] Save complete")

    return adv_images


def main() -> None:
    print("[Main] Starting pipeline...")
    device: str = get_device()
    model: torch.nn.Module = load_model(device)

    images: torch.Tensor
    labels: torch.Tensor
    images, labels = load_data(device)

    print("[Main] Computing clean accuracy...")
    acc: float = get_accuracy(model, [(images, labels)], device)
    print('Acc: %2.2f %%' % (acc))

    print("[Main] Launching targeted attack...")
    attack_targeted(model, images, labels)


if __name__ == "__main__":
    main()
