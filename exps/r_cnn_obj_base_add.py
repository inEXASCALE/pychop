import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torch.amp import autocast
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
from torchvision.transforms import functional as F
import os
import requests
import zipfile
import csv
from datetime import datetime

def download_coco_val2017():
    """Download and prepare COCO val2017 dataset."""
    data_dir = "./data/coco_val2017"
    img_dir = os.path.join(data_dir, "val2017")
    ann_file = os.path.join(data_dir, "annotations/instances_val2017.json")
    
    img_url = "http://images.cocodataset.org/zips/val2017.zip"
    ann_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

    if not os.path.exists(img_dir):
        print("Downloading COCO val2017 images (~6GB)...")
        img_zip = os.path.join(data_dir, "val2017.zip")
        os.makedirs(data_dir, exist_ok=True)
        with open(img_zip, "wb") as f:
            f.write(requests.get(img_url).content)
        print("Extracting images...")
        with zipfile.ZipFile(img_zip, "r") as zip_ref:
            zip_ref.extractall(data_dir)
        os.remove(img_zip)

    if not os.path.exists(ann_file):
        print("Downloading COCO annotations (~241MB)...")
        ann_zip = os.path.join(data_dir, "annotations.zip")
        with open(ann_zip, "wb") as f:
            f.write(requests.get(ann_url).content)
        print("Extracting annotations...")
        with zipfile.ZipFile(ann_zip, "r") as zip_ref:
            zip_ref.extractall(data_dir)
        os.remove(ann_zip)

    dataset = torchvision.datasets.CocoDetection(
        root=img_dir, annFile=ann_file, transform=F.to_tensor
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=6, shuffle=False, num_workers=4,
        collate_fn=lambda x: tuple(zip(*x)), pin_memory=True
    )
    return loader, dataset.coco

def quantize_tensor(tensor, bit_width=8):
    """Quantize a tensor to a custom bit-width (simulated)."""
    qmin, qmax = -(2**(bit_width - 1)), 2**(bit_width - 1) - 1
    scale = (tensor.max() - tensor.min()) / (qmax - qmin)
    zero_point = qmin - tensor.min() / scale
    quantized = torch.round((tensor / scale) + zero_point).clamp(qmin, qmax)
    dequantized = (quantized - zero_point) * scale
    return dequantized, scale, zero_point

def get_quantized_faster_rcnn(bit_width=8):
    """Load and quantize Faster R-CNN model weights."""
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    model.eval()
    
    with torch.no_grad():
        for name, param in model.named_parameters():
            quantized_param, _, _ = quantize_tensor(param, bit_width)
            param.copy_(quantized_param)
    
    return model.cuda()

def run_inference(model, loader, use_amp=False, max_images=None, bit_width=8):
    """Run inference with low-precision simulation."""
    latencies = []
    predictions = []
    images_for_viz = []
    outputs_for_viz = []
    image_ids = []

    with torch.no_grad():
        for i, (images, targets) in enumerate(loader):
            if max_images is not None and i * len(images) >= max_images:
                break
            try:
                images = [img.cuda() for img in images]
                start_time = time.time()

                if use_amp:
                    with autocast('cuda'):
                        outputs = model(images)
                else:
                    outputs = model(images)

                for output in outputs:
                    output["boxes"], _, _ = quantize_tensor(output["boxes"], bit_width)
                    output["scores"], _, _ = quantize_tensor(output["scores"], bit_width)
                    output["labels"] = output["labels"].float()

                latency = (time.time() - start_time) * 1000 / len(images)
                latencies.extend([latency] * len(images))

                for output, target in zip(outputs, targets):
                    if not target:
                        continue
                    boxes = output["boxes"].cpu().numpy()
                    scores = output["scores"].cpu().numpy()
                    labels = output["labels"].cpu().numpy()
                    image_id = target[0]["image_id"]
                    image_ids.append(image_id)

                    if i < 2:
                        print(f"Image {image_id}: {len(boxes)} boxes detected")
                        for j in range(min(3, len(boxes))):
                            x, y, x2, y2 = boxes[j]
                            width, height = x2 - x, y2 - y
                            print(f"  Pred Box: [{x:.2f}, {y:.2f}, {width:.2f}, {height:.2f}], Score: {scores[j]:.3f}, Label: {labels[j]}")
                        print(f"  Ground Truth: {len(target)} annotations")
                        for j in range(min(3, len(target))):
                            print(f"    GT Box: {target[j]['bbox']}, Category: {target[j]['category_id']}")

                    for box, score, label in zip(boxes, scores, labels):
                        if score > 0.5 and label > 0:
                            x, y, x2, y2 = box
                            width = x2 - x
                            height = y2 - y
                            if width > 0 and height > 0:
                                predictions.append({
                                    "image_id": int(image_id),
                                    "category_id": int(label),
                                    "bbox": [float(x), float(y), float(width), float(height)],
                                    "score": float(score)
                                })

                    if i == 0 and not images_for_viz:
                        images_for_viz.extend([img.cpu() for img in images])
                        outputs_for_viz.extend(outputs)
            except Exception as e:
                print(f"Error processing batch {i}: {e}")
                continue

    print(f"Total predictions: {len(predictions)} across {len(image_ids)} images")
    return latencies, predictions, images_for_viz, outputs_for_viz, image_ids

def evaluate_coco(predictions, coco_gt, image_ids):
    """Evaluate predictions using COCO metrics."""
    if not predictions:
        print("No predictions generated!")
        return 0.0
    coco_dt = coco_gt.loadRes(predictions)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval.stats[0]

def save_results_to_csv(latencies_fp32, latencies_quant, map_fp32, map_quant, bit_width, filename_prefix="results"):
    """Save inference results to a CSV file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename_prefix}_{timestamp}.csv"
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Mode", "Mean Latency (ms)", "Std Latency (ms)", "mAP@0.5:0.95"])
        writer.writerow(["FP32", f"{np.mean(latencies_fp32):.2f}", f"{np.std(latencies_fp32):.2f}", f"{map_fp32:.3f}"])
        writer.writerow([f"Quantized ({bit_width}-bit)", f"{np.mean(latencies_quant):.2f}", f"{np.std(latencies_quant):.2f}", f"{map_quant:.3f}"])
    print(f"Results saved to {filename}")

def plot_performance(latencies_fp32, latencies_quant, map_fp32, map_quant, bit_width):
    """Plot latency vs. mAP and save to PNG."""
    plt.figure(figsize=(8, 6))
    plt.scatter(np.mean(latencies_fp32), map_fp32, label="FP32", color="blue", s=100)
    plt.scatter(np.mean(latencies_quant), map_quant, label=f"Quantized ({bit_width}-bit)", color="red", s=100)
    plt.errorbar(np.mean(latencies_fp32), map_fp32, xerr=np.std(latencies_fp32), fmt="none", color="blue")
    plt.errorbar(np.mean(latencies_quant), map_quant, xerr=np.std(latencies_quant), fmt="none", color="red")
    plt.xlabel("Latency per Image (ms)")
    plt.ylabel("mAP@0.5:0.95")
    plt.title("Latency vs. Accuracy Trade-off")
    plt.legend()
    plt.grid(True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"performance_tradeoff_{timestamp}.png"
    plt.savefig(filename)
    plt.show()
    print(f"Performance plot saved to {filename}")

def plot_detection(images, outputs, targets, prefix="detection"):
    """Visualize predictions and ground truth for first batch and save to PNGs."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_files = []
    
    for i, (image, output, target) in enumerate(zip(images, outputs, targets)):
        fig, ax = plt.subplots(1, figsize=(12, 9))
        img = image.permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        ax.imshow(img)

        for ann in target:
            x, y, w, h = ann["bbox"]
            rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor="g", facecolor="none", linestyle="--")
            ax.add_patch(rect)
            plt.text(x, y, f"GT {ann['category_id']}", color="white", bbox=dict(facecolor="green", alpha=0.5))

        boxes = output["boxes"].cpu().numpy()
        scores = output["scores"].cpu().numpy()
        labels = output["labels"].cpu().numpy()
        for box, score, label in zip(boxes, scores, labels):
            if score > 0.5 and label > 0:
                x, y, x2, y2 = box
                rect = patches.Rectangle((x, y), x2 - x, y2 - y, linewidth=2, edgecolor="r", facecolor="none")
                ax.add_patch(rect)
                plt.text(x, y, f"Pred {label} ({score:.2f})", color="white", bbox=dict(facecolor="red", alpha=0.5))

        image_id = target[0]["image_id"]
        plt.title(f"Object Detection (Image {image_id}, Red: Pred, Green: GT)")
        plt.axis("off")
        
        filename = f"{prefix}_image_{image_id}_{timestamp}.png"
        plt.savefig(filename)
        plt.close(fig)  # Close figure to free memory
        saved_files.append(filename)
        
        if i >= 1:  # Limit to first two images
            break
    
    print(f"Detection visualizations saved to: {', '.join(saved_files)}")


def plot_detection_all(images, outputs, targets, prefix="detection"):
    """Visualize predictions and ground truth for first 6 images in a single plot."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Limit to 6 images or available images if less than 6
    num_images = min(6, len(images))
    if num_images == 0:
        print("No images available for visualization!")
        return
    
    # Create a 2x3 grid of subplots with adjusted figure size
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
    axes = axes.flatten()  # Flatten the 2D array of axes for easier iteration
    
    for i in range(num_images):
        # Ensure we have valid image, output, and target data
        if i >= len(outputs) or i >= len(targets):
            print(f"Warning: Missing output or target data for image {i}")
            axes[i].axis("off")
            continue
            
        image = images[i]
        output = outputs[i]
        target = targets[i]
        
        # Convert image tensor to numpy array and ensure correct format
        img = image.permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        axes[i].imshow(img)

        # Plot ground truth boxes (green dashed)
        if target:  # Check if target is not empty
            for ann in target:
                x, y, w, h = ann["bbox"]
                rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor="g", 
                                       facecolor="none", linestyle="--")
                axes[i].add_patch(rect)
                axes[i].text(x, y-5, f"GT {ann['category_id']}", color="white", 
                           bbox=dict(facecolor="green", alpha=0.5))

        # Plot predicted boxes (red solid)
        if "boxes" in output:  # Check if output contains boxes
            boxes = output["boxes"].cpu().numpy()
            scores = output["scores"].cpu().numpy()
            labels = output["labels"].cpu().numpy()
            for box, score, label in zip(boxes, scores, labels):
                if score > 0.5 and label > 0:
                    x, y, x2, y2 = box
                    rect = patches.Rectangle((x, y), x2 - x, y2 - y, linewidth=2, 
                                          edgecolor="r", facecolor="none")
                    axes[i].add_patch(rect)
                    axes[i].text(x, y-5, f"Pred {label} ({score:.2f})", color="white", 
                               bbox=dict(facecolor="red", alpha=0.5))

        # Set title with image ID if available
        image_id = target[0]["image_id"] if target and len(target) > 0 else f"Unknown_{i}"
        axes[i].set_title(f"Image {image_id}")
        axes[i].axis("off")

    # Hide any unused subplots
    for i in range(num_images, 6):
        axes[i].axis("off")

    # Add main title
    plt.suptitle("Object Detection (Red: Predictions, Green: Ground Truth)", 
                fontsize=16, y=1.05)
    
    # Save the figure
    filename = f"{prefix}_6images_{timestamp}.png"
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close(fig)  # Close figure to free memory
    
    print(f"Detection visualization saved to: {filename}")


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This script requires a GPU.")

    print("Preparing COCO val2017 dataset...")
    loader, coco_gt = download_coco_val2017()

    # FP32 Model
    model_fp32 = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT).cuda()
    model_fp32.eval()

    # Quantized Model
    bit_width = 8  # Customizable: 8 (INT8), 4 (INT4), etc.
    model_quant = get_quantized_faster_rcnn(bit_width=bit_width)

    max_images = 100  # Set to None to process all images

    # FP32 Inference
    print("Running FP32 inference...")
    latencies_fp32, preds_fp32, img_fp32, out_fp32, ids_fp32 = run_inference(model_fp32, loader, use_amp=False, max_images=max_images)
    map_fp32 = evaluate_coco(preds_fp32, coco_gt, ids_fp32)

    # Quantized Inference
    print(f"Running quantized inference (simulated {bit_width}-bit)...")
    latencies_quant, preds_quant, img_quant, out_quant, ids_quant = run_inference(model_quant, loader, use_amp=False, max_images=max_images, bit_width=bit_width)
    map_quant = evaluate_coco(preds_quant, coco_gt, ids_quant)

    # Print Results
    print(f"FP32 - Latency: {np.mean(latencies_fp32):.2f} ± {np.std(latencies_fp32):.2f} ms, mAP: {map_fp32:.3f}")
    print(f"Quantized ({bit_width}-bit) - Latency: {np.mean(latencies_quant):.2f} ± {np.std(latencies_quant):.2f} ms, mAP: {map_quant:.3f}")

    save_results_to_csv(latencies_fp32, latencies_quant, map_fp32, map_quant, bit_width)

    # Plot and Save Performance Trade-off
    plot_performance(latencies_fp32, latencies_quant, map_fp32, map_quant, bit_width)

    # Save Detection Visualizations
    batch_size = len(img_quant)
    targets = [loader.dataset[i][1] for i in range(batch_size)]
    plot_detection(img_quant, out_quant, targets, prefix=f"detection_quant_{bit_width}bit")
    plot_detection_all(img_quant, out_quant, targets, prefix=f"detection_quant_{bit_width}bit")


if __name__ == "__main__":
    main()