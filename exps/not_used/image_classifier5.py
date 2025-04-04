import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.models import resnet50, ResNet50_Weights

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Cutout(object):
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = torch.ones((h, w), dtype=torch.float32)
        for _ in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)
            mask[y1:y2, x1:x2] = 0.
        mask = mask.expand_as(img)
        img = img * mask.to(img.device)
        return img

# Mixup function
def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# 1. Load Datasets with Enhanced Augmentation
def load_data(dataset_name):
    if dataset_name == "MNIST":
        transform_train = transforms.Compose([
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            Cutout(n_holes=1, length=8),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)
        num_classes = 10
        input_channels = 1
        input_size = 28
    elif dataset_name == "FashionMNIST":
        transform_train = transforms.Compose([
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),
            Cutout(n_holes=1, length=8),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),
        ])
        trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform_test)
        num_classes = 10
        input_channels = 1
        input_size = 28
    elif dataset_name == "Caltech101":
        transform_train = transforms.Compose([
            transforms.Lambda(lambda img: img.convert('RGB')),
            transforms.Resize(256),
            transforms.RandomCrop(224, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            Cutout(n_holes=1, length=32),
        ])
        transform_test = transforms.Compose([
            transforms.Lambda(lambda img: img.convert('RGB')),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        full_dataset = torchvision.datasets.Caltech101(root='./data', download=True)
        train_size = int(0.7 * len(full_dataset))
        val_size = int(0.15 * len(full_dataset))
        test_size = len(full_dataset) - train_size - val_size
        trainset, valset, testset = torch.utils.data.random_split(full_dataset, [train_size, val_size, test_size])
        trainset.dataset.transform = transform_train
        valset.dataset.transform = transform_test
        testset.dataset.transform = transform_test
        num_classes = 102
        input_channels = 3
        input_size = 224
    elif dataset_name == "OxfordIIITPet":
        transform_train = transforms.Compose([
            transforms.Lambda(lambda img: img.convert('RGB')),
            transforms.Resize(256),
            transforms.RandomCrop(224, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            Cutout(n_holes=1, length=32),
        ])
        transform_test = transforms.Compose([
            transforms.Lambda(lambda img: img.convert('RGB')),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        trainset = torchvision.datasets.OxfordIIITPet(root='./data', split='trainval', download=True, transform=transform_train)
        testset = torchvision.datasets.OxfordIIITPet(root='./data', split='test', download=True, transform=transform_test)
        num_classes = 37
        input_channels = 3
        input_size = 224
    else:
        raise ValueError("Unknown dataset")
    
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)
    valloader = DataLoader(valset, batch_size=64, shuffle=False, num_workers=2) if dataset_name == "Caltech101" else None
    return trainloader, valloader, testloader, num_classes, input_channels, input_size

# 2. Define Enhanced ResNet50 Model
class ResNet(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(ResNet, self).__init__()
        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        # Adapt input channels for grayscale datasets (MNIST/FashionMNIST)
        if input_channels == 1:
            self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Modify the final fully connected layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

# 3. Training Function with Mixed Precision and Scheduler
def train_model(model, trainloader, valloader, epochs=50):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler()
    
    best_val_acc = 0.0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, alpha=1.0)
            
            optimizer.zero_grad()
            with autocast("cuda"):
                outputs = model(inputs)
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
        
        scheduler.step()
        print(f"Epoch {epoch+1}, Train Loss: {running_loss / len(trainloader):.4f}")
        
        if valloader:  # Validation for Caltech101
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for inputs, labels in valloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            val_acc = 100 * correct / total
            print(f"Validation Accuracy: {val_acc:.2f}%")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), f"{dataset}_best_model.pth")

    return best_val_acc if valloader else None

# 4. FP16 Inference
def inference_fp16(model, testloader):
    model.eval()
    model.to(device).half()
    correct, total = 0, 0
    predictions, ground_truth, images = [], [], []
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device).half(), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            predictions.extend(predicted.cpu().numpy())
            ground_truth.extend(labels.cpu().numpy())
            images.extend(inputs.cpu().float().numpy())
    accuracy = 100 * correct / total
    return accuracy, predictions, ground_truth, images

def visualize_predictions(predictions, ground_truth, dataset_name, pdf):
    plt.figure(figsize=(8, 3))  # Reduced from (10, 4)
    colors = ['green' if p == g else 'red' for p, g in zip(predictions[:20], ground_truth[:20])]
    plt.bar(range(20), predictions[:20], color=colors, alpha=0.7, width=0.6)  # Added width parameter
    plt.plot(range(20), ground_truth[:20], 'bo-', label='Ground Truth', markersize=4)  # Smaller markers
    plt.title(f"Predictions vs Ground Truth ({dataset_name})", fontsize=10)  # Smaller font
    plt.xlabel("Sample Index", fontsize=8)
    plt.ylabel("Class Label", fontsize=8)
    plt.legend(fontsize=8)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.tight_layout(pad=0.5)  # Reduced padding
    pdf.savefig(bbox_inches='tight')  # Tight bounding box
    plt.close()

def visualize_images(images, predictions, ground_truth, dataset_name, pdf):
    fig, axes = plt.subplots(2, 10, figsize=(12, 2.5))  # Reduced from (15, 3)
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            img = images[i].transpose(1, 2, 0)
            if img.shape[2] == 1:
                img = img.squeeze(2)
                ax.imshow(img, cmap='gray')
            else:
                img = (img - img.min()) / (img.max() - img.min())
                ax.imshow(img)
            color = 'green' if predictions[i] == ground_truth[i] else 'red'
            ax.set_title(f"Pred:{predictions[i]}, True:{ground_truth[i]}", 
                        color=color, fontsize=6, pad=2)  # Smaller font, reduced padding
            ax.axis('off')
    plt.suptitle(f"Image Predictions ({dataset_name})", y=1.02, fontsize=10)  # Closer title, smaller font
    plt.tight_layout(pad=0.3)  # Reduced padding
    pdf.savefig(bbox_inches='tight')  # Tight bounding box
    plt.close()


# 6. Main Execution
datasets = ["MNIST", "FashionMNIST", "Caltech101", "OxfordIIITPet"]
for dataset in datasets:
    print(f"\nProcessing {dataset}")
    trainloader, valloader, testloader, num_classes, input_channels, input_size = load_data(dataset)
    
    # Initialize and train model
    model = ResNet(input_channels, num_classes).to(device)
    best_val_acc = train_model(model, trainloader, valloader, epochs=5)
    
    # Load best model for Caltech101
    if dataset == "Caltech101" and best_val_acc:
        model.load_state_dict(torch.load(f"{dataset}_best_model.pth"))
    
    # PDF file for saving visualizations
    pdf_filename = f"{dataset}_visualizations.pdf"
    with PdfPages(pdf_filename) as pdf:
        # FP16 Inference
        acc_fp16, preds_fp16, gt_fp16, images_fp16 = inference_fp16(model, testloader)
        print(f"{dataset} FP16 Accuracy: {acc_fp16:.2f}%")
        visualize_predictions(preds_fp16, gt_fp16, f"{dataset} FP16", pdf)
        visualize_images(images_fp16, preds_fp16, gt_fp16, f"{dataset} FP16", pdf)
    
    print(f"Visualizations saved to {pdf_filename}")