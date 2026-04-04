import os
from typing import Any, Callable, List, Optional

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms

from dcgan.augment import Augmentor


class CustomDataset(Dataset):
    print("Loading Data")
    default_transform: Callable[[Any], Tensor] = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    def __init__(
        self, main_dir: str, transform: Optional[Callable[[Any], Tensor]] = None
    ) -> None:
        self.total_imgs: List[str] = [
            os.path.join(main_dir, f)
            for f in os.listdir(main_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        self.transform = transform if transform is not None else self.default_transform

    def __len__(self) -> int:
        return len(self.total_imgs)

    def __getitem__(self, idx: int) -> Tensor:
        img_path = self.total_imgs[idx]
        image = Image.open(img_path).convert("RGB")

        image_tensor: Tensor = self.transform(image)

        return image_tensor


# Initialize the dataset
dataset = CustomDataset(main_dir="data/grump/")

# Fetch one item
sample_tensor: Tensor = dataset[0]
print(f"Output Type: {type(sample_tensor)}")
print(f"Output Shape: {sample_tensor.shape}")
