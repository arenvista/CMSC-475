import os
import torch
import urllib.request
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image, make_grid

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

def get_imagenet_data():
    url = 'https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg'
    filename = 'dog.jpg'
    
    if not os.path.exists(filename):
        urllib.request.urlretrieve(url, filename)
    img = Image.open(filename)
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    
    img_tensor = transform(img).unsqueeze(0)
    label_tensor = torch.tensor([258])
    return img_tensor, label_tensor

def get_pred(model, images, device):
    model.eval()
    normalize = transforms.Normalize(mean=MEAN, std=STD)
    images_norm = torch.stack([normalize(img) for img in images]).to(device)
    
    with torch.no_grad():
        outputs = model(images_norm)
        _, predicted = torch.max(outputs.data, 1)
        
    return predicted

def get_accuracy(model, data, device):
    correct = 0
    total = 0
    model.eval()
    normalize = transforms.Normalize(mean=MEAN, std=STD)
    
    with torch.no_grad():
        for images, labels in data:
            images, labels = images.to(device), labels.to(device)
            images_norm = torch.stack([normalize(img) for img in images])
            
            outputs = model(images_norm)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    return 100.0 * correct / total

def save_comparison_grid(clean_images, adv_images, output_dir="output", prefix="adv", true_labels=None, pred_labels=None):
    """
    Saves a side-by-side grid: [Clean Image | Perturbation (Amplified) | Adversarial Image]
    """
    os.makedirs(output_dir, exist_ok=True)
    
    clean_images = clean_images.clone().detach().cpu()
    adv_images = adv_images.clone().detach().cpu()
    
    # Calculate noise/perturbation
    noise = adv_images - clean_images
    
    # Scale noise to [0, 1] for visualization purposes so it isn't just a black square
    noise_min, noise_max = noise.min(), noise.max()
    if noise_max - noise_min > 0:
        noise_vis = (noise - noise_min) / (noise_max - noise_min)
    else:
        noise_vis = noise
        
    for idx in range(clean_images.size(0)):
        # Stack them side by side
        grid_tensors = torch.stack([
            clean_images[idx], 
            noise_vis[idx], 
            adv_images[idx]
        ])
        
        # Create a single grid image
        grid = make_grid(grid_tensors, nrow=3, padding=2, normalize=False)
        
        # Format filename
        true_val = int(true_labels[idx].item()) if true_labels is not None else "X"
        pred_val = int(pred_labels[idx].item()) if pred_labels is not None else "X"
        
        filename = f"{prefix}_true{true_val}_pred{pred_val}_{idx:04d}.png"
        filepath = os.path.join(output_dir, filename)
        
        save_image(grid, filepath)
