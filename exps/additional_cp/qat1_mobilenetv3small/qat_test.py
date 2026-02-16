import csv
from utils import *
from tqdm import tqdm
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, ConcatDataset
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from torch.utils.data import DataLoader, ConcatDataset, Dataset

# ------------------ Wrapper for applying different transforms to subsets ------------------
class TransformedSubset(Dataset):
    """Apply a specific transform to a Subset or ConcatDataset."""
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.dataset)
    

# ------------------ Visualization Function (exactly matching the first script) ------------------
def visualize_images(images, predictions, ground_truth, pred_probs, dataset_name, pdf):
    """Visualize the first 20 images in a 2x10 grid and save to PDF."""
    fig, axes = plt.subplots(2, 10, figsize=(12, 2.5))
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            img = images[i].transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
            if img.shape[2] == 1:
                img = img.squeeze(2)
                ax.imshow(img, cmap='gray')
            else:
                img = (img - img.min()) / (img.max() - img.min())
                ax.imshow(img)
            color = 'green' if predictions[i] == ground_truth[i] else 'red'
            ax.set_title(f"Pred:{predictions[i]} (Prob:{np.round(pred_probs[i],2):.2f}) \nTrue:{ground_truth[i]}",
                         color=color, fontsize=8, pad=2)
            ax.axis('off')

    plt.tight_layout(pad=0.8)
    pdf.savefig(bbox_inches='tight')
    plt.close()

# ------------------ Cutout Augmentation ------------------
class Cutout(object):
    """Cutout augmentation that works on Tensor images (C, H, W)."""
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        c, h, w = img.shape
        mask = torch.ones((h, w), dtype=torch.float32, device=img.device)
        for _ in range(self.n_holes):
            y = torch.randint(high=h, size=(1,), device=img.device).item()
            x = torch.randint(high=w, size=(1,), device=img.device).item()
            y1 = max(0, y - self.length // 2)
            y2 = min(h, y + self.length // 2)
            x1 = max(0, x - self.length // 2)
            x2 = min(w, x + self.length // 2)
            mask[y1:y2, x1:x2] = 0.0
        mask = mask.unsqueeze(0).expand_as(img)
        img = img * mask
        return img

# ------------------ Data Loading ------------------
def load_train_test(dataset_name):
    """Load train and test DataLoaders with appropriate transforms."""
    if dataset_name in ["MNIST", "FashionMNIST"]:
        if dataset_name == "MNIST":
            mean, std = (0.1307,), (0.3081,)
        else:
            mean, std = (0.2860,), (0.3530,)
        transform_train = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.ToTensor(),
            Cutout(n_holes=1, length=16),
            transforms.Normalize(mean, std),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        DatasetClass = torchvision.datasets.MNIST if dataset_name == "MNIST" else torchvision.datasets.FashionMNIST
        trainset = DatasetClass(root='./data', train=True, download=True, transform=transform_train)
        testset = DatasetClass(root='./data', train=False, download=True, transform=transform_test)
        num_classes = 10
        input_channels = 1
    else:
        transform_train = transforms.Compose([
            transforms.Lambda(lambda img: img.convert('RGB')),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        transform_test = transforms.Compose([
            transforms.Lambda(lambda img: img.convert('RGB')),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        if dataset_name == "Caltech101":
            # Match PTQ split exactly: 70% train, 15% val, ~15% test
            full_dataset = torchvision.datasets.Caltech101(root='./data', download=True, transform=None)  # No default transform
            L = len(full_dataset)
            train_size = int(0.7 * L)
            val_size = int(0.15 * L)
            test_size = L - train_size - val_size
            generator = torch.Generator().manual_seed(42)
            train_split, val_split, test_split = torch.utils.data.random_split(
                full_dataset, [train_size, val_size, test_size], generator=generator)

            # Apply different transforms via wrapper
            train_concat = ConcatDataset([train_split, val_split])
            trainset = TransformedSubset(train_concat, transform=transform_train)
            testset = TransformedSubset(test_split, transform=transform_test)

            num_classes = 102
            input_channels = 3
        else:  # OxfordIIITPet
            trainset = torchvision.datasets.OxfordIIITPet(root='./data', split='trainval', download=True, transform=transform_train)
            testset = torchvision.datasets.OxfordIIITPet(root='./data', split='test', download=True, transform=transform_test)
            num_classes = 37
            input_channels = 3

    trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)
    return trainloader, testloader, num_classes, input_channels

# ------------------ QAT Training ------------------
def train_qat(model, trainloader, testloader, epochs=20, lr=1e-3):
    """Train the model with quantization-aware training and return best test accuracy."""
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_acc = 0.0

    for epoch in tqdm(range(epochs), desc="Training Epochs"):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, pred = outputs.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()
        acc = 100. * correct / total
        print(f"Epoch {epoch+1:2d}/{epochs} | Loss: {running_loss/len(trainloader):.4f} | Train Acc: {acc:.2f}%")

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, pred = outputs.max(1)
                total += labels.size(0)
                correct += pred.eq(labels).sum().item()
        test_acc = 100. * correct / total
        print(f" | Test Acc: {test_acc:.2f}%")
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), "temp_best_qat.pth")
            print(f" └─ Saved best model (Test Acc: {best_acc:.2f}%)")
        scheduler.step()

    model.load_state_dict(torch.load("temp_best_qat.pth"))
    print(f"Final Best Test Acc: {best_acc:.2f}%")
    return best_acc

# ------------------ Collect First 20 Samples for Visualization (matching first script style) ------------------
def collect_first_20_samples(model, testloader):
    """Run inference on test set and collect data for the first 20 images only."""
    model.eval()
    images, predictions, ground_truth, pred_probs = [], [], [], []
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1)
            max_probs, predicted = torch.max(probabilities, 1)

            for i in range(inputs.size(0)):
                images.append(inputs[i].cpu().numpy())
                predictions.append(predicted[i].cpu().item())
                ground_truth.append(labels[i].cpu().item())
                pred_probs.append(max_probs[i].cpu().item())
                if len(images) >= 20:
                    return images, predictions, ground_truth, pred_probs

    return images, predictions, ground_truth, pred_probs

# ------------------ Main Experiment Loop ------------------
def run_and_save_results(dataset, float_type_dict, rounding_modes, epochs_dict):
    """Run QAT experiments for one dataset, save CSV results and visualizations."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = f"qat_results_{dataset}_{timestamp}.csv"
    headers = ["float_type", "exp_bits", "sig_bits", "rounding_mode", "best_test_acc (%)", "note"]
    results = []

    os.makedirs("qat_models", exist_ok=True)
    os.makedirs("qat_class_images", exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Starting dataset: {dataset}")
    print(f"Results will be saved to: {result_file}")
    print(f"{'='*60}\n")

    trainloader, testloader, num_classes, input_channels = load_train_test(dataset)

    for key, (exp, sig) in float_type_dict.items():
        for rd in rounding_modes:
            print(f"\n>>> {dataset} | {key} | exp={exp}, sig={sig}, rmode={rd}")

            # Fresh model for each config
            model = MobileNetV3Small(input_channels, num_classes).to(device)

            chop = LightChop(exp_bits=exp, sig_bits=sig, rmode=rd)
            replace_for_qat(model, chop, quant_act=True)

            acc = train_qat(model, trainloader, testloader,
                            epochs=epochs_dict.get(dataset, 20), lr=1e-3)

            # Save final QAT model
            save_path = f"qat_models/{dataset}_{key}_e{exp}s{sig}_r{rd}_qat.pth"
            torch.save(model.state_dict(), save_path)
            print(f"Saved model: {save_path}")

            # ------------------ Visualization (exactly matching the first script) ------------------
            vis_images, vis_preds, vis_gt, vis_probs = collect_first_20_samples(model, testloader)
            pdf_filename = f"qat_class_images/{dataset}_{key}_{rd}_visualizations.pdf"
            with PdfPages(pdf_filename) as pdf:
                visualize_images(vis_images, vis_preds, vis_gt, vis_probs,
                                 dataset_name=f"{dataset}_{key}_{rd}", pdf=pdf)
            print(f"Visualization saved: {pdf_filename}")

            results.append({
                "float_type": key,
                "exp_bits": exp,
                "sig_bits": sig,
                "rounding_mode": rd,
                "best_test_acc (%)": f"{acc:.2f}",
                "note": ""
            })

    # Write CSV results
    with open(result_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nAll done! Results saved to: {result_file}")

# ===================== Main =====================
if __name__ == "__main__":
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from pychop import LightChop  # Ensure pychop is imported


    float_type = {
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
    rounding_mode = [1, 2, 3, 4, 5, 6]

    datasets = ["MNIST", "FashionMNIST", "Caltech101", "OxfordIIITPet"]

    #epochs_per_dataset = {
    #    "MNIST": 25,
    #    "FashionMNIST": 25,
    #    "Caltech101": 12,
    #    "OxfordIIITPet": 12,
    #}

    epochs_per_dataset = {
        "MNIST": 10,
        "FashionMNIST": 10,
        "Caltech101": 10,
        "OxfordIIITPet": 10,
    }
    
    for ds in datasets:
        run_and_save_results(
            dataset=ds,
            float_type_dict=float_type,
            rounding_modes=rounding_mode,
            epochs_dict=epochs_per_dataset
        )