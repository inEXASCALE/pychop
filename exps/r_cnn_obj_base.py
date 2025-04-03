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
        dataset, batch_size=4, shuffle=False, num_workers=4,
        collate_fn=lambda x: tuple(zip(*x)), pin_memory=True
    )
    return loader, dataset.coco

def get_faster_rcnn():
    """Load pre-trained Faster R-CNN model."""
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    model.eval()
    return model.cuda()

def run_inference(model, loader, use_amp=False, max_images=None):
    """Run inference and collect predictions."""
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

def plot_performance(latencies_fp32, latencies_amp, map_fp32, map_amp):
    """Plot latency vs. mAP."""
    plt.figure(figsize=(8, 6))
    plt.scatter(np.mean(latencies_fp32), map_fp32, label="FP32", color="blue", s=100)
    plt.scatter(np.mean(latencies_amp), map_amp, label="AMP", color="red", s=100)
    plt.errorbar(np.mean(latencies_fp32), map_fp32, xerr=np.std(latencies_fp32), fmt="none", color="blue")
    plt.errorbar(np.mean(latencies_amp), map_amp, xerr=np.std(latencies_amp), fmt="none", color="red")
    plt.xlabel("Latency per Image (ms)")
    plt.ylabel("mAP@0.5:0.95")
    plt.title("Latency vs. Accuracy Trade-off")
    plt.legend()
    plt.grid(True)
    plt.savefig("performance_tradeoff.png")
    plt.show()

def plot_detection(images, outputs, targets):
    """Visualize predictions and ground truth for first batch."""
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

        plt.title(f"Object Detection (Image {target[0]['image_id']}, Red: Pred, Green: GT)")
        plt.axis("off")
        plt.savefig(f"detection_example_{i}.png")
        plt.show()
        if i >= 1:  # Limit to first two images
            break

def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This script requires a GPU.")

    print("Preparing COCO val2017 dataset...")
    loader, coco_gt = download_coco_val2017()

    model = get_faster_rcnn()

    max_images = 100  # Set to None to process all images
    print("Running FP32 inference...")
    latencies_fp32, preds_fp32, img_fp32, out_fp32, img_ids_fp32 = run_inference(model, loader, use_amp=False, max_images=max_images)
    map_fp32 = evaluate_coco(preds_fp32, coco_gt, img_ids_fp32)

    print("Running AMP inference...")
    latencies_amp, preds_amp, img_amp, out_amp, img_ids_amp = run_inference(model, loader, use_amp=True, max_images=max_images)
    map_amp = evaluate_coco(preds_amp, coco_gt, img_ids_amp)

    print(f"FP32 - Latency: {np.mean(latencies_fp32):.2f} ± {np.std(latencies_fp32):.2f} ms, mAP: {map_fp32:.3f}")
    print(f"AMP  - Latency: {np.mean(latencies_amp):.2f} ± {np.std(latencies_amp):.2f} ms, mAP: {map_amp:.3f}")

    plot_performance(latencies_fp32, latencies_amp, map_fp32, map_amp)
    
    # Get ground truth for the first batch
    batch_size = len(img_amp)
    targets = [loader.dataset[i][1] for i in range(batch_size)]
    plot_detection(img_amp, out_amp, targets)

if __name__ == "__main__":
    main()