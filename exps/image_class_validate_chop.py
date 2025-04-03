import sys
# appending a path
sys.path.append('../')
import pychop
from pychop import LightChop
from pychop.layers import post_quantization_ft

pychop.backend("torch")

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from torchvision.models import resnet50, ResNet50_Weights

torch.manual_seed(42)
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

def load_data(dataset_name):
    if dataset_name == "MNIST":
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)
        num_classes = 10
        input_channels = 1
    elif dataset_name == "FashionMNIST":
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),
        ])
        testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform_test)
        num_classes = 10
        input_channels = 1
    elif dataset_name == "Caltech101":
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
        _, _, testset = torch.utils.data.random_split(full_dataset, [train_size, val_size, test_size], generator=generator)
        testset.dataset.transform = transform_test
        num_classes = 102
        input_channels = 3
    elif dataset_name == "OxfordIIITPet":
        transform_test = transforms.Compose([
            transforms.Lambda(lambda img: img.convert('RGB')),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        testset = torchvision.datasets.OxfordIIITPet(root='./data', split='test', download=True, transform=transform_test)
        num_classes = 37
        input_channels = 3
    else:
        raise ValueError("Unknown dataset")
    
    testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)
    return testloader, num_classes, input_channels

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

def inference(model, testloader):
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

def visualize_images(images, predictions, ground_truth, dataset_name, pdf):
    fig, axes = plt.subplots(2, 10, figsize=(12, 2.5))
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
                        color=color, fontsize=6, pad=2)
            ax.axis('off')
    plt.suptitle(f"Image Predictions ({dataset_name})", y=1.02, fontsize=10)
    plt.tight_layout(pad=0.3)
    pdf.savefig(bbox_inches='tight')
    plt.close()

datasets = ["MNIST", "FashionMNIST", "Caltech101", "OxfordIIITPet"]
for dataset in datasets:
    print(f"\nEvaluating {dataset}")
    testloader, num_classes, input_channels = load_data(dataset)
    
    model = ResNet(input_channels, num_classes).to(device)
    model_file = f"{dataset}_best_model.pth" if dataset == "Caltech101" else f"{dataset}_final_model.pth"
    try:
        model.load_state_dict(torch.load(model_file))
        print(f"Loaded {model_file} for {dataset}")
    except FileNotFoundError:
        print(f"Warning: {model_file} not found for {dataset}. Using pre-trained weights.")
    
    pdf_filename = f"{dataset}_visualizations.pdf"
    with PdfPages(pdf_filename) as pdf:
        acc_fp16, preds_fp16, gt_fp16, images_fp16 = inference(model, testloader)
        print(f"{dataset} FP16 Accuracy: {acc_fp16:.2f}%")
        visualize_images(images_fp16, preds_fp16, gt_fp16, f"{dataset} FP16", pdf)
    
    print(f"Visualizations saved to {pdf_filename}")