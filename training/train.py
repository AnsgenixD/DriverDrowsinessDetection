# train.py - Trains the driver drowsiness and distraction classification model

import os
import sys
import time
import argparse
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
from models.custom_cnn import Custom2DCNN_Fusion
from models.mobilenet_v3 import MobileNetV3Small_Fusion
from models.quantization import quantize_model_dynamic, save_quantized_model
from training.dataset import create_synthetic_multimodal_dataset


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for imgs, geoms, labels in loader:
        imgs = imgs.to(device)
        geoms = geoms.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs, geoms)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels).item()
        total += labels.size(0)

    return running_loss / total, (correct / total) * 100.0


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    class_correct = [0] * config.NUM_CLASSES
    class_total = [0] * config.NUM_CLASSES

    with torch.no_grad():
        for imgs, geoms, labels in loader:
            imgs = imgs.to(device)
            geoms = geoms.to(device)
            labels = labels.to(device)

            outputs = model(imgs, geoms)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels).item()
            total += labels.size(0)

            for p, l in zip(preds, labels):
                if p == l:
                    class_correct[l.item()] += 1
                class_total[l.item()] += 1

    overall_acc = (correct / total) * 100.0
    class_accs = {
        config.CLASSES[i]: (class_correct[i] / max(1, class_total[i])) * 100.0
        for i in range(config.NUM_CLASSES)
    }
    return running_loss / total, overall_acc, class_accs


def run_training(
    model_name: str = "custom_cnn",
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 1e-3,
    save_path: str = str(config.DEFAULT_WEIGHTS_PATH),
    samples_per_class: int = 400,
):
    print(f"============================================================")
    print(f"Starting Training: {model_name.upper()} Dual-Branch DMS")
    print(f"============================================================")

    # Device selection
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using training acceleration device: {device}")

    # Build dataset
    print(f"Generating balanced multimodal dataset ({samples_per_class} per class)...")
    train_ds, val_ds = create_synthetic_multimodal_dataset(num_samples_per_class=samples_per_class)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    print(f"Dataset summary: Train={len(train_ds)} samples, Val={len(val_ds)} samples.")

    # Instantiate model
    if model_name == "mobilenet_v3":
        model = MobileNetV3Small_Fusion(num_classes=config.NUM_CLASSES, geom_dim=config.GEOM_FEATURE_DIM)
    else:
        model = Custom2DCNN_Fusion(num_classes=config.NUM_CLASSES, geom_dim=config.GEOM_FEATURE_DIM)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, class_accs = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"   -> [BEST] Saved new champion weights to {save_path} (Val Acc: {val_acc:.2f}%)")

    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed:.1f}s. Best Validation Accuracy: {best_acc:.2f}%\n")

    # Final Per-Class Performance
    print("Per-Class Accuracy Evaluation:")
    for cls_name, acc in class_accs.items():
        print(f"  - {cls_name:<14}: {acc:.2f}%")

    # Quantization Step (Methodology Thread 4)
    print("\n--- Performing Post-Training Static / Dynamic Quantization (INT8) ---")
    model.load_state_dict(torch.load(save_path, map_location="cpu"))
    quantized_model = quantize_model_dynamic(model)
    save_quantized_model(quantized_model, str(config.QUANTIZED_WEIGHTS_PATH))

    print("\nAll model artifacts successfully exported.")
    return best_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DMS Dual-Branch Classifier")
    parser.add_argument("--model", type=str, choices=["custom_cnn", "mobilenet_v3"], default="custom_cnn")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    run_training(model_name=args.model, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
