import sys
import os
import csv
import copy
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from pychop.layers import post_quantization
sys.path.append('../')
import pychop
from pychop import Chop
pychop.backend("torch")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def fuse_resnet(model):
    """
    Fuse Conv+BN layers in ResNet.
    """
    model.eval() # Fusion must under eval mode
    
    # 1. fuse the first layer of backbone  (Stem)
    try:
        torch.ao.quantization.fuse_modules(model.backbone, [['conv1', 'bn1', 'relu']], inplace=True)
    except Exception as e:
        print(f"Note: Stem fusion skipped or modified: {e}")

    # 2. Fuse Bottleneck module
    for m in model.modules():
        if isinstance(m, torchvision.models.resnet.Bottleneck):
            try:
                torch.ao.quantization.fuse_modules(m, [
                    ['conv1', 'bn1'],
                    ['conv2', 'bn2'],
                    ['conv3', 'bn3'] 
                ], inplace=True)
            except: pass # Skip if already fused or structure differs
            
            if m.downsample is not None:
                try:
                    torch.ao.quantization.fuse_modules(m.downsample, [['0', '1']], inplace=True)
                except: pass
                
    for m in model.modules():
        if isinstance(m, torchvision.models.resnet.BasicBlock):
            try:
                torch.ao.quantization.fuse_modules(m, [
                    ['conv1', 'bn1'], 
                    ['conv2', 'bn2']
                ], inplace=True)
            except: pass
            if m.downsample is not None:
                try:
                    torch.ao.quantization.fuse_modules(m.downsample, [['0', '1']], inplace=True)
                except: pass

    return model

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_FILE = "quantization_results.csv"

os.makedirs("class_images", exist_ok=True)


# --- Model & Data Definitions ---
class ResNet(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(ResNet, self).__init__()
        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        if input_channels == 1:
            self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
    def forward(self, x): return self.backbone(x)


def load_data(dataset_name):
    print(f"Loading {dataset_name}...")
    if dataset_name == "MNIST":
        t = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        d = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=t)
        classes, ch = 10, 1
    elif dataset_name == "FashionMNIST":
        t = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))])
        d = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=t)
        classes, ch = 10, 1
    elif dataset_name == "Caltech101":
        t = transforms.Compose([
            transforms.Lambda(lambda x: x.convert('RGB')), transforms.Resize(256), transforms.CenterCrop(224),
            transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        full = torchvision.datasets.Caltech101(root='./data', download=True)
        generator = torch.Generator().manual_seed(42)
        train_size = int(0.7 * len(full))
        val_size = int(0.15 * len(full))
        test_size = len(full) - train_size - val_size
        _, _, d = torch.utils.data.random_split(full, [train_size, val_size, test_size], generator=generator)
        d.dataset.transform = t 
        classes, ch = 102, 3
    elif dataset_name == "OxfordIIITPet":
        t = transforms.Compose([
            transforms.Lambda(lambda x: x.convert('RGB')), transforms.Resize(256), transforms.CenterCrop(224),
            transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        d = torchvision.datasets.OxfordIIITPet(root='./data', split='test', download=True, transform=t)
        classes, ch = 37, 3
    else: raise ValueError(f"Unknown dataset {dataset_name}")
    
    # num_workers=0 to ensure safety in simple scripts
    return DataLoader(d, batch_size=64, shuffle=False, num_workers=0), classes, ch


# --- Numerical Error Analysis Metrics ---
def compute_mse(t1, t2):
    return torch.mean((t1 - t2) ** 2).item()

def measure_weight_mse(clean_model, quant_model, device):
    """
    Computes Weight MSE only. 
    Activation SQNR is now computed dynamically on logits in the main loop.
    """
    w_mse_list = []
    clean_params = dict(clean_model.named_parameters())
    quant_params = dict(quant_model.named_parameters())
    
    for name, p_c in clean_params.items():
        if name in quant_params:
            p_q = quant_params[name]
            # Only compare if shapes match
            if p_c.shape == p_q.shape:
                w_mse_list.append(compute_mse(p_c.to(device), p_q.to(device)))
    
    avg_w_mse = np.mean(w_mse_list) if w_mse_list else 0.0
    return avg_w_mse


# --- Visualization Function ---
def visualize_images(images, predictions, ground_truth, pred_probs, dataset_name, pdf):
    # Ensure inputs are valid lists/arrays and take top 20
    count = min(len(images), 20)
    if count == 0: return

    images = images[:count]
    predictions = predictions[:count]
    ground_truth = ground_truth[:count]
    pred_probs = pred_probs[:count]
    
    fig, axes = plt.subplots(2, 10, figsize=(12, 2.5))
    for i, ax in enumerate(axes.flat):
        if i < count:
            img = images[i]  # shape: (C, H, W) or (H, W, C) depending on source, usually (C, H, W) from tensor
            
            # Simple normalization for display
            if img.max() != img.min():
                img = (img - img.min()) / (img.max() - img.min())
            
            img = np.transpose(img, (1, 2, 0)) # To (H, W, C)
            
            if img.shape[2] == 1:
                img = img.squeeze(2)
                ax.imshow(img, cmap='gray')
            else:
                ax.imshow(img)
                
            color = 'green' if predictions[i] == ground_truth[i] else 'red'
            ax.set_title(f"P:{predictions[i]} ({pred_probs[i]:.2f})\nT:{ground_truth[i]}", 
                         color=color, fontsize=8)
            ax.axis('off')
        else:
            ax.axis('off') # Hide unused subplots

    plt.tight_layout(pad=0.8)
    pdf.savefig(bbox_inches='tight')
    plt.close()


def run_experiment_and_save():
    float_types = {
        "q43": (4, 3),
        "q52": (5, 2),
        "custom(5, 5)": (5, 5),
        "custom(5, 7)": (5, 7),
        "custom(8, 4)": (8, 4),
        "half": (5, 10),
        "bfloat16": (8, 7),
        "tf32": (8, 10),
        "fp32": (8, 23),
    }
    rounding_modes = [1, 2, 3, 4, 5, 6]
    datasets = ["MNIST", "FashionMNIST", "Caltech101", "OxfordIIITPet"]

    headers = ["Dataset", "Format", "Exp_Bits", "Sig_Bits", "Rounding_Mode", "Accuracy", "Weight_MSE", "Activation_SQNR_dB"]
    
    if not os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'w', newline='') as f:
            csv.writer(f).writerow(headers)
    
    for dataset in datasets:
        print(f"\n--- Processing {dataset} ---")
        testloader, num_classes, in_ch = load_data(dataset)
        
        # 1. Prepare Clean Model
        clean_model = ResNet(in_ch, num_classes)
        model_path = f"{dataset}_best_model.pth" if dataset == "Caltech101" else f"{dataset}_final_model.pth"
        
        try:
            clean_model.load_state_dict(torch.load(model_path, map_location='cpu'))
            print(f"Loaded weights: {model_path}")
        except FileNotFoundError:
            print(f"Warning: Weights not found. Using random init.")

        # Fuse Clean Model (Standard Baseline)
        print("Fusing BN layers...")
        clean_model = fuse_resnet(clean_model)
        clean_model.eval()
        clean_model.to(device) # Move Clean Model to GPU once

        for fmt_name, (exp, sig) in float_types.items():
            for rm in rounding_modes:
                # PDF per configuration
                pdf_filename = f"class_images/{dataset}_{fmt_name}_{rm}_visualizations.jpg"
                
                print(f"Running: {fmt_name} (R{rm})...", end="")
                
                # 2. Prepare Quantized Model
                model_q = copy.deepcopy(clean_model).to("cpu") # Copy on CPU to save memory
                chopper = Chop(exp_bits=exp, sig_bits=sig, rmode=rm)
                model_q = post_quantization(model_q, chopper)
                model_q.to(device)
                
                # 3. Compute Weight MSE (Static Analysis)
                w_mse = measure_weight_mse(clean_model, model_q, device)
                
                # 4. Inference & Logit SQNR (Dynamic Analysis)
                correct = 0
                total = 0
                
                # Accumulators for Global SQNR Calculation
                sum_signal_pow = 0.0
                sum_noise_pow = 0.0
                
                # Lists for visualization (Small subset only)
                vis_predictions, vis_ground_truth, vis_images, vis_pred_probs = [], [], [], []
                collected_vis = False
                
                with torch.no_grad():
                    with PdfPages(pdf_filename) as pdf: # Open PDF here
                        for inputs, labels in testloader:
                            inputs, labels = inputs.to(device), labels.to(device)
                            
                            # Parallel Inference
                            logits_q = model_q(inputs)
                            logits_c = clean_model(inputs)
                            
                            # Calculate Accuracy
                            probs = torch.softmax(logits_q, dim=1)
                            max_probs, predicted = torch.max(probs, 1)
                            total += labels.size(0)
                            correct += (predicted == labels).sum().item()
                            
                            # Calculate Batch Logit SQNR Components
                            # (Sum of Squares for Signal and Noise)
                            noise = logits_c - logits_q
                            sum_signal_pow += torch.sum(logits_c ** 2).item()
                            sum_noise_pow += torch.sum(noise ** 2).item()
                            
                            # Collect Data for Visualization (First Batch Only)
                            if not collected_vis:
                                vis_predictions.extend(predicted.cpu().numpy())
                                vis_ground_truth.extend(labels.cpu().numpy())
                                vis_images.extend(inputs.cpu().float().numpy())
                                vis_pred_probs.extend(max_probs.cpu().numpy())
                                collected_vis = True
                        
                        # End of dataset loop
                        accuracy = 100 * correct / total
                        
                        # Compute Final SQNR
                        if sum_noise_pow == 0:
                            act_sqnr = 100.0 # Perfect match
                        else:
                            act_sqnr = 10 * np.log10(sum_signal_pow / sum_noise_pow)

                        print(f" Acc: {accuracy:.2f}%, Logit SQNR: {act_sqnr:.2f}dB")
                        
                        # Create Visualization Page
                        visualize_images(vis_images, vis_predictions, vis_ground_truth, vis_pred_probs, dataset, pdf)
                
                # 5. Save Results
                with open(RESULTS_FILE, 'a', newline='') as f:
                    csv.writer(f).writerow([
                        dataset, fmt_name, exp, sig, rm, 
                        f"{accuracy:.4f}", f"{w_mse:.8f}", f"{act_sqnr:.4f}"
                    ])

if __name__ == "__main__":
    run_experiment_and_save()
    print(f"\nExperiments completed. Data saved to {RESULTS_FILE}")