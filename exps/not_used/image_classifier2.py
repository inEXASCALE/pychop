import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

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

def load_data(dataset_name):
    if dataset_name == "MNIST":
        transform_train = transforms.Compose([
            transforms.RandomRotation(10),
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
    elif dataset_name == "FashionMNIST":
        transform_train = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(),
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
    elif dataset_name == "STL10":
        transform_train = transforms.Compose([
            transforms.Resize(96),  # STL-10 native size
            transforms.RandomCrop(96, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            Cutout(n_holes=1, length=16),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(96),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        trainset = torchvision.datasets.STL10(root='./data', split='train', download=True, transform=transform_train)
        testset = torchvision.datasets.STL10(root='./data', split='test', download=True, transform=transform_test)
        num_classes = 10
        input_channels = 3
    else:
        raise ValueError("Unknown dataset")
    
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)
    return trainloader, testloader, num_classes, input_channels

# 2. Define Enhanced CNN Model
class EnhancedCNN(nn.Module):
    def __init__(self, input_channels, num_classes, input_size=28):
        super(EnhancedCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)
        
        # Calculate FC input size based on input resolution
        if input_size == 28:  # MNIST/FashionMNIST
            fc_input_size = 128 * 3 * 3  # 28 → 14 → 7 → 3
        elif input_size == 96:  # STL-10
            fc_input_size = 128 * 12 * 12  # 96 → 48 → 24 → 12
        elif input_size == 64:  # Tiny ImageNet
            fc_input_size = 128 * 8 * 8  # 64 → 32 → 16 → 8
        
        self.fc1 = nn.Linear(fc_input_size, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# 3. Training Function with Mixup
def train_model(model, trainloader, epochs=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            # Apply Mixup
            inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, alpha=1.0)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss / len(trainloader):.4f}")

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

# 5. Visualization Functions
def visualize_predictions(predictions, ground_truth, dataset_name, pdf):
    plt.figure(figsize=(10, 4))
    colors = ['green' if p == g else 'red' for p, g in zip(predictions[:20], ground_truth[:20])]
    plt.bar(range(20), predictions[:20], color=colors, alpha=0.7)
    plt.plot(range(20), ground_truth[:20], 'bo-', label='Ground Truth')
    plt.title(f"Predictions vs Ground Truth ({dataset_name})")
    plt.xlabel("Sample Index")
    plt.ylabel("Class Label")
    plt.legend()
    plt.tight_layout()
    pdf.savefig()
    plt.close()

def visualize_images(images, predictions, ground_truth, dataset_name, pdf):
    fig, axes = plt.subplots(2, 10, figsize=(15, 3))
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
            ax.set_title(f"Pred: {predictions[i]}\nTrue: {ground_truth[i]}", color=color, fontsize=8)
            ax.axis('off')
    plt.suptitle(f"Image Predictions ({dataset_name})", y=1.05)
    plt.tight_layout()
    pdf.savefig()
    plt.close()

# 6. Main Execution
datasets = ["MNIST", "FashionMNIST", "STL10"]
for dataset in datasets:
    print(f"\nProcessing {dataset}")
    trainloader, testloader, num_classes, input_channels = load_data(dataset)
    
    # Set input size based on dataset
    input_size = 28 if dataset in ["MNIST", "FashionMNIST"] else 96 if dataset == "STL10" else 64
    
    # Initialize and train model
    model = EnhancedCNN(input_channels, num_classes, input_size).to(device)
    train_model(model, trainloader, epochs=10)  # Increased epochs for STL-10 and Tiny ImageNet
    
    # PDF file for saving visualizations
    pdf_filename = f"{dataset}_visualizations.pdf"
    with PdfPages(pdf_filename) as pdf:
        # FP16 Inference
        acc_fp16, preds_fp16, gt_fp16, images_fp16 = inference_fp16(model, testloader)
        print(f"{dataset} FP16 Accuracy: {acc_fp16:.2f}%")
        visualize_predictions(preds_fp16, gt_fp16, f"{dataset} FP16", pdf)
        visualize_images(images_fp16, preds_fp16, gt_fp16, f"{dataset} FP16", pdf)
    
    print(f"Visualizations saved to {pdf_filename}")