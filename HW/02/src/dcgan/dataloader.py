import os
from PIL import Image
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, main_dir, transform=None):
        # Create a list of full file paths for every item in the directory
        self.total_imgs = [os.path.join(main_dir, file) for file in os.listdir(main_dir)]
        # use the main_dir argument to locate and load your images
        self.transform = transform

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        # Get the path for the specific image using the index
        img_path = self.total_imgs[idx]
        # Open the image and ensure it has 3 channels (RGB)
        image = Image.open(img_path).convert("RGB")
        # Apply any provided transformations (like resizing or ToTensor)
        if self.transform is not None:
            image = self.transform(image)
            
        return image
