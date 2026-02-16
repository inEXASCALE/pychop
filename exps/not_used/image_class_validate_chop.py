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
from pychop import LightChop
pychop.backend("torch")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def fuse_resnet(model):
    """
    Fuse Conv+BN layers in ResNet.
    """
    model.eval() # Fusion must under eval mode
    
    # 1. fuse the first layer of backbone  (Stem)
    # The self.relu only used once in Stem，without internal share in block
    try:
        torch.ao.quantization.fuse_modules(model.backbone, [['conv1', 'bn1', 'relu']], inplace=True)
    except Exception as e:
        print(f"Note: Stem fusion skipped or modified: {e}")

    # 2. Fuse Bottleneck module
    for m in model.modules():
        if isinstance(m, torchvision.models.resnet.Bottleneck):
            # only ['conv', 'bn']
            torch.ao.quantization.fuse_modules(m, [
                ['conv1', 'bn1'],
                ['conv2', 'bn2'],
                ['conv3', 'bn3'] 
            ], inplace=True)
            
            if m.downsample is not None:
                torch.ao.quantization.fuse_modules(m.downsample, [['0', '1']], inplace=True)
                
    for m in model.modules():
        if isinstance(m, torchvision.models.resnet.BasicBlock):
            torch.ao.quantization.fuse_modules(m, [
                ['conv1', 'bn1'], 
                ['conv2', 'bn2']
            ], inplace=True)
            if m.downsample is not None:
                torch.ao.quantization.fuse_modules(m.downsample, [['0', '1']], inplace=True)

    return model

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_FILE = "quantization_results.csv"

os.makedirs("class_images", exist_ok=True)  # 创建文件夹以保存 PDF


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
    
    return DataLoader(d, batch_size=64, shuffle=False, num_workers=0), classes, ch


# --- Numerical Error Analysis Metrics ---
def compute_mse(t1, t2):
    return torch.mean((t1 - t2) ** 2).item()

def compute_sqnr(t_clean, t_quant):
    noise = t_clean - t_quant
    power_clean = torch.sum(t_clean ** 2)
    power_noise = torch.sum(noise ** 2)
    if power_noise == 0: return 100.0 # Max dB
    return (10 * torch.log10(power_clean / power_noise)).item()

def measure_quantization_error(clean_model, quant_model, dataloader, device):
    # 1. Weight Analysis
    w_mse_list = []
    clean_params = dict(clean_model.named_parameters())
    quant_params = dict(quant_model.named_parameters())
    
    for name, p_c in clean_params.items():
        if name in quant_params:
            p_q = quant_params[name]
            if p_c.shape == p_q.shape:
                w_mse_list.append(compute_mse(p_c.to(device), p_q.to(device)))
    avg_w_mse = np.mean(w_mse_list) if w_mse_list else 0

    # 2. Activation Analysis (Forward Hook)
    act_sqnr_list = []
    clean_acts = {}
    
    def get_clean_hook(name):
        def hook(model, input, output):
            clean_acts[name] = output.detach()
        return hook

    def get_quant_hook(name):
        def hook(model, input, output):
            if name in clean_acts:
                sqnr = compute_sqnr(clean_acts[name], output.detach())
                act_sqnr_list.append(sqnr)
        return hook

    hooks = []
    for name, m in clean_model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            hooks.append(m.register_forward_hook(get_clean_hook(name)))
            
    # Run one batch
    inputs, _ = next(iter(dataloader))
    inputs = inputs.to(device)
    with torch.no_grad():
        clean_model(inputs)
        
    for h in hooks: h.remove()
    hooks = []
    
    for name, m in quant_model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            hooks.append(m.register_forward_hook(get_quant_hook(name)))
            
    with torch.no_grad():
        quant_model(inputs)
        
    for h in hooks: h.remove()
    
    avg_act_sqnr = np.mean(act_sqnr_list) if act_sqnr_list else 0
    return avg_w_mse, avg_act_sqnr


# --- 新增：可视化函数 ---
def visualize_images(images, predictions, ground_truth, pred_probs, dataset_name, pdf):
    # 只取前 20 张图像进行可视化
    images = images[:20]
    predictions = predictions[:20]
    ground_truth = ground_truth[:20]
    pred_probs = pred_probs[:20]
    
    fig, axes = plt.subplots(2, 10, figsize=(12, 2.5))
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            img = images[i]  # shape: (C, H, W)
            channels = img.shape[0]
            
            # 根据数据集选择正确的 mean/std 并反归一化
            if "MNIST" in dataset_name or "FashionMNIST" in dataset_name:
                mean = np.array([0.1307 if "MNIST" in dataset_name else 0.2860]).reshape(1, 1, 1)
                std = np.array([0.3081 if "MNIST" in dataset_name else 0.3530]).reshape(1, 1, 1)
            else:
                mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
                std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
            
            img = img * std + mean
            img = np.clip(img, 0, 1)
            img = img.transpose(1, 2, 0)  # to (H, W, C)
            
            if channels == 1:
                img = img.squeeze(2)  # to (H, W)
                ax.imshow(img, cmap='gray')
            else:
                ax.imshow(img)
                
            color = 'green' if predictions[i] == ground_truth[i] else 'red'
            ax.set_title(f"Pred:{predictions[i]} (Prob:{pred_probs[i]:.2f})\nTrue:{ground_truth[i]}", 
                         color=color, fontsize=8)
            ax.axis('off')

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
    rounding_modes = [1, 2, 3, 4]
    datasets = ["MNIST", "FashionMNIST", "Caltech101", "OxfordIIITPet"]

    headers = ["Dataset", "Format", "Exp_Bits", "Sig_Bits", "Rounding_Mode", "Accuracy", "Weight_MSE", "Activation_SQNR_dB"]
    
    if not os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'w', newline='') as f:
            csv.writer(f).writerow(headers)
    
    for dataset in datasets:
        print(f"\n--- Processing {dataset} ---")
        testloader, num_classes, in_ch = load_data(dataset)
        
        clean_model = ResNet(in_ch, num_classes)
        model_path = f"{dataset}_best_model.pth" if dataset == "Caltech101" else f"{dataset}_final_model.pth"
        
        try:
            clean_model.load_state_dict(torch.load(model_path, map_location='cpu'))
            print(f"Loaded weights: {model_path}")
        except FileNotFoundError:
            print(f"Warning: Weights not found. Using random init.")
        except RuntimeError as e:
            print(f"Error loading weights: {e}\nCheck if model architecture matches weights.")

        clean_model.eval()

        print("Fusing BN layers...")
        clean_model = fuse_resnet(clean_model)
        
        clean_model.to(device)

        for fmt_name, (exp, sig) in float_types.items():
            for rm in rounding_modes:
                pdf_filename = f"class_images/{dataset}_{fmt_name}_{rm}_visualizations.pdf"
                with PdfPages(pdf_filename) as pdf:
                    print(f"Running: {fmt_name} (R{rm})...", end="")
                    
                    model_q = copy.deepcopy(clean_model).to("cpu") 
                    
                    chopper = LightChop(exp_bits=exp, sig_bits=sig, rmode=rm)
                    model_q = post_quantization(model_q, chopper)
                    
                    model_q.to(device)
                    w_mse, act_sqnr = measure_quantization_error(clean_model, model_q, testloader, device)
                    
                    correct = 0
                    total = 0
                    predictions, ground_truth, images, pred_probs = [], [], [], []
                    with torch.no_grad():
                        for inputs, labels in testloader:
                            inputs, labels = inputs.to(device), labels.to(device)
                            outputs = model_q(inputs)
                            probabilities = torch.softmax(outputs, dim=1)
                            max_probs, predicted = torch.max(probabilities, 1)
                            
                            total += labels.size(0)
                            correct += (predicted == labels).sum().item()
                            
                            predictions.extend(predicted.cpu().numpy())
                            ground_truth.extend(labels.cpu().numpy())
                            images.extend(inputs.cpu().numpy())
                            pred_probs.extend(max_probs.cpu().numpy())
                    
                    accuracy = 100 * correct / total
                    print(f" Acc: {accuracy:.2f}%, SQNR: {act_sqnr:.2f}dB")
                    
                    # 生成可视化 PDF
                    visualize_images(images, predictions, ground_truth, pred_probs, dataset, pdf)
                
                with open(RESULTS_FILE, 'a', newline='') as f:
                    csv.writer(f).writerow([
                        dataset, fmt_name, exp, sig, rm, 
                        f"{accuracy:.4f}", f"{w_mse:.8f}", f"{act_sqnr:.4f}"
                    ])


if __name__ == "__main__":
    run_experiment_and_save()
    print(f"\nExperiments completed. Data saved to {RESULTS_FILE}")