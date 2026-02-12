from utils import *
from tqdm import tqdm
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import numpy as np
class Cutout(object):
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (torch.Tensor): Tensor image of shape (C, H, W)
        Returns:
            torch.Tensor: Image with cutout applied
        """
        c, h, w = img.shape

        mask = torch.ones((h, w), dtype=torch.float32, device=img.device)

        for _ in range(self.n_holes):
            # Random center
            y = torch.randint(high=h, size=(1,), device=img.device).item()
            x = torch.randint(high=w, size=(1,), device=img.device).item()

            y1 = max(0, y - self.length // 2)
            y2 = min(h, y + self.length // 2)
            x1 = max(0, x - self.length // 2)
            x2 = min(w, x + self.length // 2)

            mask[y1:y2, x1:x2] = 0.0

        # Expand mask to (C, H, W)
        mask = mask.unsqueeze(0).expand_as(img)
        img = img * mask

        return img
    
def load_train_test(dataset_name):
    if dataset_name in ["MNIST", "FashionMNIST"]:
        if dataset_name == "MNIST":
            mean, std = (0.1307,), (0.3081,)  
        else:
            mean, std = (0.2860,), (0.3530,)
        transform_train = transforms.Compose([
            transforms.RandomCrop(28, padding=4),   # Works on PIL Image
            transforms.ToTensor(),                  # ← MUST be BEFORE Cutout (converts PIL → Tensor)
            Cutout(n_holes=1, length=16),           # Now receives Tensor (C, H, W)
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
    else:  # Caltech101, OxfordIIITPet
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
            full_dataset = torchvision.datasets.Caltech101(root='./data', download=True)
            train_size = int(0.85 * len(full_dataset))
            test_size = len(full_dataset) - train_size
            generator = torch.Generator().manual_seed(42)
            trainset, testset = torch.utils.data.random_split(full_dataset, [train_size, test_size], generator=generator)
            trainset.dataset.transform = transform_train
            testset.dataset.transform = transform_test
            num_classes = 102
        else:  # OxfordIIITPet
            trainset = torchvision.datasets.OxfordIIITPet(root='./data', split='trainval', download=True, transform=transform_train)
            testset = torchvision.datasets.OxfordIIITPet(root='./data', split='test', download=True, transform=transform_test)
            num_classes = 37
        input_channels = 3

    trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)
    return trainloader, testloader, num_classes, input_channels

def train_qat(model, trainloader, testloader, epochs=20, lr=1e-3):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    # 改用 AdamW，更適合微調 + 量化訓練
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    # 或保持 SGD 但 lr 加大
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4, nesterov=True)
    
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
        print(f"          | Test Acc: {test_acc:.2f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), "temp_best_qat.pth")
            print(f"          └─ Saved best model (Test Acc: {best_acc:.2f}%)")

        scheduler.step()

    model.load_state_dict(torch.load("temp_best_qat.pth"))
    print(f"Final Best Test Acc: {best_acc:.2f}%")
    return best_acc


# ==================== 主訓練循環 ====================
float_type = {
    "q43": (4, 3),
    "q52": (5, 2),
    "half": (5, 10),
    "bfloat16": (8, 7),
    "fp32": (8, 23),
}

rounding_mode = [1]  # 先只跑 nearest rounding 測試 baseline

datasets = ["MNIST", "FashionMNIST"]  # 先只跑這兩個

for dataset in datasets:
    print(f"\n{'='*50}")
    print(f"Training QAT for {dataset} with MobileNetV3-Small")
    print(f"{'='*50}\n")
    
    trainloader, testloader, num_classes, input_channels = load_train_test(dataset)

    model = MobileNetV3Small(input_channels, num_classes).to(device)

    # 先跑一次 fp32 baseline（不量化）
    print(">>> Running FP32 Baseline (no quantization)")
    baseline_model = MobileNetV3Small(input_channels, num_classes).to(device)
    train_qat(baseline_model, trainloader, testloader, epochs=20, lr=1e-3)
    print(">>> FP32 Baseline finished\n")

    for key in float_type:
        for rd in rounding_mode:
            bits = float_type[key]
            chop = LightChop(exp_bits=bits[0], sig_bits=bits[1], rmode=rd)

            # 重要：只量化權重
            replace_for_qat(model, chop, quant_act=False)

            print(f"Training {dataset} | {key} | rounding {rd}")
            epochs = 25 if dataset in ["MNIST", "FashionMNIST"] else 12
            train_qat(model, trainloader, testloader, epochs=epochs, lr=1e-3)

            save_path = f"qat_models/{dataset}_{key}_{rd}_mobilenetv3_qat.pth"
            torch.save(model.state_dict(), save_path)
            print(f"Saved: {save_path}\n")