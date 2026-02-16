import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.models import resnet50, ResNet50_Weights
import random

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)  

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Cutout(object):
    def __init__(self, n_holes, length):
        torch.manual_seed(42)
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
        trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
        testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)
        valloader = None
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
        trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
        testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)
        valloader = None
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
        generator = torch.Generator().manual_seed(42)
        trainset, valset, testset = torch.utils.data.random_split(full_dataset, [train_size, val_size, test_size], generator=generator)
        trainset.dataset.transform = transform_train
        valset.dataset.transform = transform_test
        testset.dataset.transform = transform_test
        num_classes = 102
        input_channels = 3
        input_size = 224
        trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
        testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)
        valloader = DataLoader(valset, batch_size=64, shuffle=False, num_workers=2)
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
        trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
        testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)
        valloader = None
    else:
        raise ValueError("Unknown dataset")
    
    return trainloader, valloader, testloader, num_classes, input_channels, input_size

class ResNet(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(ResNet, self).__init__()
        torch.manual_seed(42)
        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        if input_channels == 1:
            self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

def train_model(model, trainloader, valloader, testloader, epochs=20):
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
        
        # Validation
        if valloader:
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
                print(f"Saved best model with Val Acc: {best_val_acc:.2f}%")
        
        # Test accuracy for diagnostics
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        test_acc = 100 * correct / total
        print(f"Test Accuracy: {test_acc:.2f}%")
    
    # Save final model for all datasets
    torch.save(model.state_dict(), f"{dataset}_final_model.pth")
    print(f"Saved final model as {dataset}_final_model.pth with Test Acc: {test_acc:.2f}%")
    
    return best_val_acc if valloader else None

# Main Execution for Training
datasets = ["MNIST", "FashionMNIST", "Caltech101", "OxfordIIITPet"]
for dataset in datasets:
    print(f"\nTraining {dataset}")
    trainloader, valloader, testloader, num_classes, input_channels, input_size = load_data(dataset)
    
    model = ResNet(input_channels, num_classes).to(device)
    best_val_acc = train_model(model, trainloader, valloader, testloader, epochs=30)
    
    print(f"Training completed for {dataset}")
    if dataset == "Caltech101" and best_val_acc:
        print(f"Best model saved as {dataset}_best_model.pth with Val Acc: {best_val_acc:.2f}%")