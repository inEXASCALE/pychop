import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score, f1_score
import os

# Set random seed and device
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Combined Loss
class CombinedLoss(nn.Module):
    def __init__(self, weight=0.7, smooth=1e-6):
        super(CombinedLoss, self).__init__()
        self.weight = weight
        self.dice = DiceLoss(smooth=smooth)
        self.ce = nn.BCELoss()
    
    def forward(self, pred, target):
        return self.weight * self.dice(pred, target) + (1 - self.weight) * self.ce(pred, target)

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = pred.contiguous()
        target = target.contiguous()
        intersection = (pred * target).sum(dim=2).sum(dim=2)
        dice = (2. * intersection + self.smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + self.smooth)
        return 1 - dice.mean()

# Optimized U-Net
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        
        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
        
        self.enc1 = conv_block(in_channels, 64)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        self.middle = conv_block(256, 512)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = conv_block(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = conv_block(128, 64)
        
        self.final = nn.Conv2d(64, out_channels, 1)
        self.pool = nn.MaxPool2d(2)
        
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        m = self.middle(self.pool(e3))
        
        d3 = self.upconv3(m)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        return torch.sigmoid(self.final(d1))

def calculate_metrics(pred, target, threshold=0.5):
    pred = (pred > threshold).flatten()
    target = (target > 0.5).flatten()
    iou = jaccard_score(target, pred)
    dice = f1_score(target, pred)
    return iou, dice

def visualize_results(images, masks, predictions, num_images=3, save_dir='results'):
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(num_images, 3, figsize=(18, 6*num_images))
    
    for i in range(num_images):
        # Handle image dimensions
        img = images[i].squeeze(0).permute(1, 2, 0).numpy() if images[i].dim() == 4 else images[i].permute(1, 2, 0).numpy()
        mask = masks[i].squeeze().numpy()  # Shape: (256, 256)
        pred = predictions[i].squeeze().numpy()  # Shape: (256, 256)
        
        # Instead of just flipping horizontally, let's try to match the original orientation
        # Remove np.fliplr() and test if the mask aligns correctly
        # If needed, you can uncomment one of these transformations:
        # mask_corrected = np.fliplr(mask)  # Horizontal flip
        # mask_corrected = np.flipud(mask)  # Vertical flip
        # mask_corrected = np.rot90(mask, k=1)  # Rotate 90 degrees
        mask_corrected = mask  # No transformation, original mask
        
        # Debug prints
        print(f"Image {i} - Denormalized image range: min={img.min():.4f}, max={img.max():.4f}")
        print(f"Image {i} - Original mask shape: {mask.shape}, unique values: {np.unique(mask)}")
        print(f"Image {i} - Corrected mask shape: {mask_corrected.shape}, unique values: {np.unique(mask_corrected)}")
        print(f"Image {i} - Pred shape: {pred.shape}, unique values: {np.unique(pred.round(2))}")
        
        # Original Image
        axes[i, 0].imshow(img)
        axes[i, 0].set_title('Original Image', fontsize=12)
        axes[i, 0].axis('off')
        
        # Ground Truth
        gt_colored = np.zeros((mask_corrected.shape[0], mask_corrected.shape[1], 3), dtype=np.uint8)
        gt_mask = (mask_corrected > 0.5)  # Threshold to ensure binary mask
        gt_colored[gt_mask] = [0, 255, 0]  # Green for foreground
        gt_colored[~gt_mask] = [0, 0, 0]   # Black for background
        axes[i, 1].imshow(gt_colored)
        axes[i, 1].set_title('Ground Truth (Green)', fontsize=12)
        axes[i, 1].axis('off')
        
        # Prediction
        pred_colored = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
        pred_mask = (pred > 0.5)
        pred_colored[pred_mask] = [255, 0, 0]  # Red for foreground
        pred_colored[~pred_mask] = [0, 0, 0]   # Black for background
        axes[i, 2].imshow(pred_colored)
        axes[i, 2].set_title('Prediction (Red)', fontsize=12)
        axes[i, 2].axis('off')
        
        # Save individual images
        plt.imsave(f'{save_dir}/image_{i}_original.png', img)
        plt.imsave(f'{save_dir}/image_{i}_groundtruth.png', gt_colored)
        plt.imsave(f'{save_dir}/image_{i}_prediction.png', pred_colored)
    
    plt.tight_layout()
    save_path = f'{save_dir}/segmentation_results.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {save_path}")
    plt.show()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    mean_vals = [0.4821, 0.4465, 0.4070]
    std_vals = [0.2689, 0.2610, 0.2718]
    train_transform = transforms.Compose([
        transforms.Resize((288, 288)),
        transforms.RandomCrop((256, 256)),
        #transforms.RandomHorizontalFlip(),
        #transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_vals, std=std_vals)
    ])
    
    mask_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.Lambda(lambda x: torch.tensor(np.array(x), dtype=torch.float32).unsqueeze(0)),
        transforms.Lambda(lambda x: (x == 1).float())
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_vals, std=std_vals)
    ])
    
    dataset = datasets.OxfordIIITPet(
        root='./data',
        split='trainval',
        target_types='segmentation',
        download=True,
        transform=train_transform,
        target_transform=mask_transform
    )
    
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    test_dataset.dataset.transform = test_transform
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    model = UNet().to(device)
    criterion = CombinedLoss(weight=0.7)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        scheduler.step()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')
    
    model.eval()
    test_images, test_masks, predictions = [], [], []
    ious, dices = [], []
    
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            for images, masks in test_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                
                if len(test_images) < 3:
                    test_images.extend(images.cpu())
                    test_masks.extend(masks.cpu())
                    predictions.extend(outputs.cpu())
                
                for pred, mask in zip(outputs.cpu().numpy(), masks.cpu().numpy()):
                    iou, dice = calculate_metrics(pred, mask)
                    ious.append(iou)
                    dices.append(dice)
                if len(ious) >= 12:
                    break
    
    print(f'Average IoU: {np.mean(ious):.4f} ± {np.std(ious):.4f}')
    print(f'Average Dice: {np.mean(dices):.4f} ± {np.std(dices):.4f}')
    
    mean = torch.tensor(mean_vals).view(1, 3, 1, 1)
    std = torch.tensor(std_vals).view(1, 3, 1, 1)
    test_images = [((img * std) + mean).clamp(0, 1) for img in test_images[:3]]
    
    visualize_results(test_images, test_masks[:3], predictions[:3])

if __name__ == '__main__':
    main()