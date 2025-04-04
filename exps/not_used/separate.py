import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

def compute_dataset_stats(dataset):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Ensure consistent size
        transforms.ToTensor()           # Convert to tensor
    ])
    dataset.transform = transform  # Apply transform to dataset
    
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    n_samples = 0
    
    for images, _ in loader:
        batch_samples = images.size(0)  # Number of images in batch
        images = images.view(batch_samples, images.size(1), -1)  # Reshape to [batch, channels, pixels]
        mean += images.mean(2).sum(0)   # Sum mean across pixels and batch
        std += images.std(2).sum(0)     # Sum std across pixels and batch
        n_samples += batch_samples
    
    mean /= n_samples  # Average over all samples
    std /= n_samples   # Average over all samples
    return mean, std

# Load raw dataset
raw_dataset = datasets.OxfordIIITPet(
    root='./data',
    split='trainval',
    download=True,
    transform=None  # Will be set in compute_dataset_stats
)

mean, std = compute_dataset_stats(raw_dataset)
print(f"Dataset mean: {mean}, std: {std}")