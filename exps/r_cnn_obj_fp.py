import torch

import sys
# appending a path
sys.path.append('../')
import pychop
from pychop import Chopf

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
from pychop.layers import post_quantization


pychop.backend("torch")



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
        dataset, batch_size=8, shuffle=False, num_workers=4,
        collate_fn=lambda x: tuple(zip(*x)), pin_memory=True
    )
    return loader, dataset.coco



def get_quantized_faster_rcnn(chop):
    """Load and quantize Faster R-CNN model weights."""
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    model.eval()
    
    model = post_quantization(model, chop, verbose=True)
    return model.cuda()


def run_inference(model, loader, use_amp=False, max_images=None, quantizer=None):
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
                    if quantizer is not None:
                        output["boxes"] = output["boxes"]
                        output["scores"] = output["scores"]
                        
                    output["labels"] = output["labels"]# .float()

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

# def save_results_to_csv(latencies_fp32, latencies_quant, map_fp32, map_quant, chop, filename_prefix="results"):
#     """Save inference results to a CSV file."""
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     filename = f"{filename_prefix}.csv"
# 
#     with open(filename, "w", newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow(["Mode", "Mean Latency (ms)", "Std Latency (ms)", "mAP@0.5:0.95"])
#         writer.writerow(["FP32", f"{np.mean(latencies_fp32):.2f}", f"{np.std(latencies_fp32):.2f}", f"{map_fp32:.3f}"])
#         writer.writerow([f"Quantized ({chop.num_bits}-bit)", f"{np.mean(latencies_quant):.2f}", f"{np.std(latencies_quant):.2f}", f"{map_quant:.3f}"])
#     print(f"Results saved to {filename}")


# def plot_detection(images, outputs, targets, prefix="detection"):
#     """Visualize predictions and ground truth for first batch and save to PNGs."""
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     saved_files = []
#     
#     for i, (image, output, target) in enumerate(zip(images, outputs, targets)):
#         fig, ax = plt.subplots(1, figsize=(12, 9))
#         img = image.permute(1, 2, 0).numpy()
#         img = np.clip(img, 0, 1)
#         ax.imshow(img)
# 
#         for ann in target:
#             x, y, w, h = ann["bbox"]
#             rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor="g", facecolor="none", linestyle="--")
#             ax.add_patch(rect)
#             plt.text(x, y, f"GT {ann['category_id']}", color="white", bbox=dict(facecolor="green", alpha=0.5))
# 
#         boxes = output["boxes"].cpu().numpy()
#         scores = output["scores"].cpu().numpy()
#         labels = output["labels"].cpu().numpy()
#         for box, score, label in zip(boxes, scores, labels):
#             if score > 0.5 and label > 0:
#                 x, y, x2, y2 = box
#                 rect = patches.Rectangle((x, y), x2 - x, y2 - y, linewidth=2, edgecolor="r", facecolor="none")
#                 ax.add_patch(rect)
#                 plt.text(x, y, f"Pred {label} ({score:.2f})", color="white", bbox=dict(facecolor="red", alpha=0.5))
# 
#         image_id = target[0]["image_id"]
#         plt.title(f"Object Detection (Image {image_id}, Red: Pred, Green: GT)")
#         plt.axis("off")
#         
#         filename = f"obj_images/{prefix}_image_{image_id}.png"
#         plt.savefig(filename)
#         plt.close(fig)  # Close figure to free memory
#         saved_files.append(filename)
#         
#         if i >= 1:  # Limit to first two images
#             break
#     
#     print(f"Detection visualizations saved to: {', '.join(saved_files)}")


def plot_detection_all(images, outputs, targets, prefix="detection"):
    """Visualize predictions and ground truth for first 6 images in a single plot."""
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Limit to 8 images or available images if less than 6
    num_images = min(8, len(images))
    if num_images == 0:
        print("No images available for visualization!")
        return
    
    # Create a 2x5 grid of subplots with adjusted figure size
    fig, axes = plt.subplots(2, 4, figsize=(16, 10), constrained_layout=True)
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
                rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor="darkred", 
                                       facecolor="none", linestyle="--")
                axes[i].add_patch(rect)
                axes[i].text(x, y-5, f"GT {ann['category_id']}", color="black", 
                           bbox=dict(facecolor="tomato", alpha=0.5))

        # Plot predicted boxes (red solid)
        if "boxes" in output:  # Check if output contains boxes
            boxes = output["boxes"].cpu().numpy()
            scores = output["scores"].cpu().numpy()
            labels = output["labels"].cpu().numpy()
            for box, score, label in zip(boxes, scores, labels):
                if score > 0.5 and label > 0:
                    x, y, x2, y2 = box
                    rect = patches.Rectangle((x, y), x2 - x, y2 - y, linewidth=2, 
                                          edgecolor="darkgreen", facecolor="none")
                    axes[i].add_patch(rect)
                    axes[i].text(x, y-5, f"Pred {label} ({score:.2f})", color="black", 
                               bbox=dict(facecolor="yellowgreen", alpha=0.45))

        # Set title with image ID if available
        # image_id = target[0]["image_id"] if target and len(target) > 0 else f"Unknown_{i}"
        # axes[i].set_title(f"Image {image_id}")
        axes[i].axis("off")

    for i in range(num_images, 8):
        axes[i].axis("off")

    # plt.suptitle("Object Detection (Red: Ground Truth, Green: Predictions)", 
    #            fontsize=19, y=1.03)
    
    # Save the figure
    filename = f"obj_images/{prefix}.png"
    plt.savefig(filename, bbox_inches='tight', dpi=400)
    plt.close(fig)  # Close figure to free memory
    
    print(f"Detection visualization saved to: {filename}")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This script requires a GPU.")

    print("Preparing COCO val2017 dataset...")
    loader, coco_gt = download_coco_val2017()

    # FP32 Model
    model_fp32 = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT).cuda()
    model_fp32.eval()


    max_images = 100  

    # FP32 Inference
    print("Running FP32 inference...")
    latencies_fp32, preds_fp32, img_fp32, out_fp32, ids_fp32 = run_inference(model_fp32,
                                    loader, use_amp=False, max_images=max_images, quantizer=None)
    map_fp32 = evaluate_coco(preds_fp32, coco_gt, ids_fp32)


    # save_results_to_csv(latencies_fp32, latencies_quant, map_fp32, map_quant, quantizer_int8)
    filename = f"obj_detect_results_fp.csv"

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Mode", "Rounding", "Mean Latency (ms)", "Std Latency (ms)", "mAP@0.5:0.95"])
        writer.writerow(["FP32 (reference)", "1", f"{np.mean(latencies_fp32):.2f}", f"{np.std(latencies_fp32):.2f}", f"{map_fp32:.3f}"])
        
        # Quantized Model
        for i in [1, 2, 3, 4, 5, 6]:#np.arange(1, 3):
            print("\n\n---------------------------------")
            quantizer_ft52 = Chopf(ibits=5, fbits=2, rmode=i)
            quantizer_ft43 = Chopf(ibits=4, fbits=3, rmode=i)
            quantizer_ft55 = Chopf(ibits=5, fbits=5, rmode=i)
            quantizer_ft57 = Chopf(ibits=5, fbits=7, rmode=i)
            
            quantizer_ft84 = Chopf(ibits=8, fbits=4, rmode=i)
            quantizer_ft87 = Chopf(ibits=8, fbits=7, rmode=i)
            quantizer_tf32 = Chopf(ibits=8, fbits=10, rmode=i)
            quantizer_ft510 = Chopf(ibits=5, fbits=10, rmode=i)
            quantizer_ft823 = Chopf(ibits=8, fbits=23, rmode=i)

            model_quant_ft52 = get_quantized_faster_rcnn(quantizer_ft52)
            model_quant_ft43 = get_quantized_faster_rcnn(quantizer_ft43)
            model_quant_ft55 = get_quantized_faster_rcnn(quantizer_ft55)
            model_quant_ft57 = get_quantized_faster_rcnn(quantizer_ft57)

            model_quant_ft84 = get_quantized_faster_rcnn(quantizer_ft84)
            model_quant_ft87 = get_quantized_faster_rcnn(quantizer_ft87)
            model_quant_tf32 = get_quantized_faster_rcnn(quantizer_tf32)
            model_quant_ft510 = get_quantized_faster_rcnn(quantizer_ft510)
            model_quant_ft823 = get_quantized_faster_rcnn(quantizer_ft823)

            print(f"Running quantized inference (simulated (52)...")
            latencies_quant52, preds_quant52, img_quant52, out_quant52, ids_quant52 = run_inference(model_quant_ft52, 
                                            loader, use_amp=False, max_images=max_images, quantizer=quantizer_ft52)
            
            map_quant52 = evaluate_coco(preds_quant52, coco_gt, ids_quant52)


            print(f"Running quantized inference (simulated (43)...")
            latencies_quant43, preds_quant43, img_quant43, out_quant43, ids_quant43 = run_inference(model_quant_ft43, 
                                            loader, use_amp=False, max_images=max_images, quantizer=quantizer_ft43)
            
            map_quant43 = evaluate_coco(preds_quant43, coco_gt, ids_quant43)

            print(f"Running quantized inference (simulated (55)...")
            latencies_quant55, preds_quant55, img_quant55, out_quant55, ids_quant55 = run_inference(model_quant_ft55, 
                                            loader, use_amp=False, max_images=max_images, quantizer=quantizer_ft55)
            
            map_quant55 = evaluate_coco(preds_quant55, coco_gt, ids_quant55)

            print(f"Running quantized inference (simulated (43)...")
            latencies_quant57, preds_quant57, img_quant57, out_quant57, ids_quant57 = run_inference(model_quant_ft57, 
                                            loader, use_amp=False, max_images=max_images, quantizer=quantizer_ft57)
            
            map_quant57 = evaluate_coco(preds_quant57, coco_gt, ids_quant57)

            print(f"Running quantized inference (simulated (84)...")
            latencies_quant84, preds_quant84, img_quant84, out_quant84, ids_quant84 = run_inference(model_quant_ft84, 
                                            loader, use_amp=False, max_images=max_images, quantizer=quantizer_ft84)
            
            map_quant84 = evaluate_coco(preds_quant84, coco_gt, ids_quant84)

            print(f"Running quantized inference (simulated (tf32)...")
            latencies_quanttf32, preds_quanttf32, img_quanttf32, out_quanttf32, ids_quanttf32 = run_inference(model_quant_tf32, 
                                            loader, use_amp=False, max_images=max_images, quantizer=quantizer_tf32)
            
            map_quanttf32 = evaluate_coco(preds_quanttf32, coco_gt, ids_quanttf32)

            print(f"Running quantized inference (simulated (87)...")
            latencies_quant87, preds_quant87, img_quant87, out_quant87, ids_quant87 = run_inference(model_quant_ft87, 
                                            loader, use_amp=False, max_images=max_images, quantizer=quantizer_ft87)
            
            map_quant87 = evaluate_coco(preds_quant87, coco_gt, ids_quant87)

            print(f"Running quantized inference (simulated (52)...")
            latencies_quant510, preds_quant510, img_quant510, out_quant510, ids_quant510 = run_inference(model_quant_ft510, 
                                            loader, use_amp=False, max_images=max_images, quantizer=quantizer_ft510)
            
            map_quant510 = evaluate_coco(preds_quant510, coco_gt, ids_quant510)

            print(f"Running quantized inference (simulated (823)...")
            latencies_quant823, preds_quant823, img_quant823, out_quant823, ids_quant823 = run_inference(model_quant_ft823, 
                                            loader, use_amp=False, max_images=max_images, quantizer=quantizer_ft823)
            
            map_quant823 = evaluate_coco(preds_quant823, coco_gt, ids_quant823)

            # Print Results
            print(f"FP32 - Latency: {np.mean(latencies_fp32):.2f} ± {np.std(latencies_fp32):.2f} ms, mAP: {map_fp32:.3f}")
            print(f"Quantized (43) - Latency: {np.mean(latencies_quant43):.2f} ± {np.std(latencies_quant43):.2f} ms, mAP: {map_quant43:.3f}")
            print(f"Quantized (52) - Latency: {np.mean(latencies_quant52):.2f} ± {np.std(latencies_quant52):.2f} ms, mAP: {map_quant52:.3f}")
            print(f"Quantized (55) - Latency: {np.mean(latencies_quant55):.2f} ± {np.std(latencies_quant55):.2f} ms, mAP: {map_quant55:.3f}")
            print(f"Quantized (57) - Latency: {np.mean(latencies_quant57):.2f} ± {np.std(latencies_quant57):.2f} ms, mAP: {map_quant57:.3f}")
            
            print(f"Quantized (84) - Latency: {np.mean(latencies_quant84):.2f} ± {np.std(latencies_quant84):.2f} ms, mAP: {map_quant84:.3f}")
            print(f"Quantized (tf32) - Latency: {np.mean(latencies_quanttf32):.2f} ± {np.std(latencies_quanttf32):.2f} ms, mAP: {map_quanttf32:.3f}")
            print(f"Quantized (87) - Latency: {np.mean(latencies_quant87):.2f} ± {np.std(latencies_quant87):.2f} ms, mAP: {map_quant87:.3f}")
            print(f"Quantized (510) - Latency: {np.mean(latencies_quant510):.2f} ± {np.std(latencies_quant510):.2f} ms, mAP: {map_quant510:.3f}")
            print(f"Quantized (823) - Latency: {np.mean(latencies_quant823):.2f} ± {np.std(latencies_quant823):.2f} ms, mAP: {map_quant823:.3f}")
            
            batch_size = len(img_quant43)
            targets = [loader.dataset[i][1] for i in range(batch_size)]
            plot_detection_all(img_quant43, out_quant43, targets, prefix=f"detection_quant_43_fp_r{i}")
            plot_detection_all(img_quant52, out_quant52, targets, prefix=f"detection_quant_52_fp_r{i}")
            plot_detection_all(img_quant55, out_quant55, targets, prefix=f"detection_quant_55_fp_r{i}")
            plot_detection_all(img_quant57, out_quant57, targets, prefix=f"detection_quant_57_fp_r{i}")

            plot_detection_all(img_quant84, out_quant84, targets, prefix=f"detection_quant_84_fp_r{i}")
            plot_detection_all(img_quanttf32, out_quanttf32, targets, prefix=f"detection_quant_tf32_fp_r{i}")
            plot_detection_all(img_quant87, out_quant87, targets, prefix=f"detection_quant_87_fp_r{i}")
            plot_detection_all(img_quant510, out_quant510, targets, prefix=f"detection_quant_510_fp_r{i}")
            plot_detection_all(img_quant823, out_quant823, targets, prefix=f"detection_quant_823_fp_r{i}")

            # writer.writerow([f"Quantized ({quantizer_int4.num_bits}-bit)", f"{np.mean(latencies_quant4):.2f}", f"{np.std(latencies_quant4):.2f}", f"{map_quant4:.3f}"])
            writer.writerow(["q43", f"{i}", f"{np.mean(latencies_quant43):.2f}", f"{np.std(latencies_quant43):.2f}", f"{map_quant43:.3f}"])
            writer.writerow(["q52", f"{i}", f"{np.mean(latencies_quant52):.2f}", f"{np.std(latencies_quant52):.2f}", f"{map_quant52:.3f}"])
            writer.writerow(["Custom(5, 5)", f"{i}", f"{np.mean(latencies_quant55):.2f}", f"{np.std(latencies_quant55):.2f}", f"{map_quant55:.3f}"])
            writer.writerow(["Custom(5, 7)", f"{i}", f"{np.mean(latencies_quant57):.2f}", f"{np.std(latencies_quant57):.2f}", f"{map_quant57:.3f}"])
            
            writer.writerow(["Custom(8, 4)", f"{i}", f"{np.mean(latencies_quant84):.2f}", f"{np.std(latencies_quant84):.2f}", f"{map_quant84:.3f}"])
            writer.writerow(["bfloat16", f"{i}", f"{np.mean(latencies_quant87):.2f}", f"{np.std(latencies_quant87):.2f}", f"{map_quant87:.3f}"])
            writer.writerow(["tf32", f"{i}", f"{np.mean(latencies_quanttf32):.2f}", f"{np.std(latencies_quanttf32):.2f}", f"{map_quanttf32:.3f}"])
            writer.writerow(["half", f"{i}", f"{np.mean(latencies_quant510):.2f}", f"{np.std(latencies_quant510):.2f}", f"{map_quant510:.3f}"])
            writer.writerow(["FP32", f"{i}", f"{np.mean(latencies_quant823):.2f}", f"{np.std(latencies_quant823):.2f}", f"{map_quant823:.3f}"])

        print(f"Results saved to {filename}")
