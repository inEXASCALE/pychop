import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )
    
    def forward(self, x):
        x = self.flatten(x)
        return self.layers(x)

# Quantization functions (unchanged)
def quantize_to_int8(tensor, scale, zero_point):
    q_tensor = torch.round(tensor / scale + zero_point)
    q_tensor = torch.clamp(q_tensor, -128, 127)
    return q_tensor.int()

def dequantize_from_int8(q_tensor, scale, zero_point):
    return scale * (q_tensor.float() - zero_point)

def get_quantization_params(tensor):
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)
    qmin, qmax = -128, 127
    scale = (max_val - min_val) / (qmax - qmin)
    zero_point = qmin - min_val / scale
    return scale, zero_point

# Load data (unchanged)
def load_data(dataset_name):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    if dataset_name.lower() == 'mnist':
        dataset_class = torchvision.datasets.MNIST
    elif dataset_name.lower() == 'fashion_mnist':
        dataset_class = torchvision.datasets.FashionMNIST
    else:
        raise ValueError("Dataset must be 'mnist' or 'fashion_mnist'")
    
    train_dataset = dataset_class(root='./data', train=True, download=True, transform=transform)
    test_dataset = dataset_class(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    return train_loader, test_loader

# Training function (unchanged)
def train_model(model, train_loader, epochs=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}')

# Modified evaluation with visualization
def evaluate_and_visualize(model, test_loader, dataset_name, quantized=False, scale=None, zero_point=None):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    sample_images = []
    sample_preds = []
    sample_labels = []
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            outputs = model(images)
            if quantized:
                outputs = quantize_to_int8(outputs, scale, zero_point)
                outputs = dequantize_from_int8(outputs, scale, zero_point)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Collect samples for image visualization (first batch only)
            if i == 0:
                sample_images = images.cpu().numpy()
                sample_preds = predicted.cpu().numpy()
                sample_labels = labels.cpu().numpy()
    
    accuracy = 100 * correct / total
    
    # Visualization 1: Scatter plot of predictions vs ground truth
    plt.figure(figsize=(10, 8))
    colors = ['green' if p == l else 'red' for p, l in zip(all_preds, all_labels)]
    plt.scatter(range(len(all_labels)), all_labels, c=colors, alpha=0.5, label='Ground Truth')
    plt.scatter(range(len(all_preds)), all_preds, c=colors, alpha=0.5, marker='x', label='Predicted')
    plt.xlabel('Sample Index')
    plt.ylabel('Class Label')
    plt.title(f'{dataset_name} - Predictions vs Ground Truth (Accuracy: {accuracy:.2f}%)')
    plt.legend()
    plt.tight_layout()
    
    pdf_name = f'{dataset_name}_performance{"_int8" if quantized else "_fp32"}.pdf'
    with PdfPages(pdf_name) as pdf:
        pdf.savefig()
    plt.close()
    
    # Visualization 2: Sample images with predicted labels
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        if i < len(sample_images):
            img = sample_images[i][0]  # Remove channel dimension
            ax.imshow(img, cmap='gray')
            ax.set_title(f'Pred: {sample_preds[i]}\nTrue: {sample_labels[i]}', 
                        color='green' if sample_preds[i] == sample_labels[i] else 'red')
            ax.axis('off')
    plt.suptitle(f'{dataset_name} - Sample Predictions {"(INT8)" if quantized else "(FP32)"}')
    plt.tight_layout()
    
    pdf_name = f'{dataset_name}_samples{"_int8" if quantized else "_fp32"}.pdf'
    with PdfPages(pdf_name) as pdf:
        pdf.savefig()
    plt.close()
    
    return accuracy

# Main execution
def main():
    datasets = ['mnist', 'fashion_mnist']
    
    for dataset in datasets:
        print(f"\nProcessing {dataset.upper()} dataset")
        train_loader, test_loader = load_data(dataset)
        
        # Initialize and train model
        model = DNN()
        print("Training model...")
        train_model(model, train_loader)
        
        # Evaluate FP32
        print("Evaluating FP32 model...")
        fp32_accuracy = evaluate_and_visualize(model, test_loader, dataset.upper())
        print(f"FP32 Accuracy: {fp32_accuracy:.2f}%")
        
        # Calibration for INT8
        model.eval()
        with torch.no_grad():
            for images, _ in test_loader:
                outputs = model(images)
                scale, zero_point = get_quantization_params(outputs)
                break
        
        # Evaluate INT8
        print("Evaluating INT8 model...")
        int8_accuracy = evaluate_and_visualize(model, test_loader, dataset.upper(), True, scale, zero_point)
        print(f"INT8 Accuracy: {int8_accuracy:.2f}%")

if __name__ == "__main__":
    main()