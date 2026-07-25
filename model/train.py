"""
train.py

Trains the ResNet18 classifier on the chest_xray dataset (real or synthetic).

Usage:
    python train.py --data_dir data/chest_xray --epochs 10 --batch_size 32

Saves:
    saved_models/pneumonia_resnet18.pth   (weights only, for load_model())
    saved_models/training_history.json    (loss/acc curves)
    saved_models/metrics.json             (final test metrics)
"""

import argparse
import json
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix

from model_def import build_model, CLASS_NAMES

IMG_SIZE = 224


def get_transforms(train: bool):
    if train:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out = model(x)
            preds = out.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())

    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary", zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds).tolist()
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1, "confusion_matrix": cm}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/chest_xray")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--out_dir", default="saved_models")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Using device: {device}")

    train_ds = datasets.ImageFolder(os.path.join(args.data_dir, "train"), transform=get_transforms(True))
    val_ds = datasets.ImageFolder(os.path.join(args.data_dir, "val"), transform=get_transforms(False))
    test_ds = datasets.ImageFolder(os.path.join(args.data_dir, "test"), transform=get_transforms(False))

    print(f"[train] Classes found: {train_ds.classes} (must match CLASS_NAMES={CLASS_NAMES})")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = build_model(pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        t0 = time.time()

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)

        train_loss = running_loss / len(train_ds)

        model.eval()
        val_loss, correct = 0.0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
                val_loss += loss.item() * x.size(0)
                correct += (out.argmax(1) == y).sum().item()

        val_loss /= len(val_ds)
        val_acc = correct / len(val_ds)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"[train] Epoch {epoch+1}/{args.epochs} "
              f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
              f"({time.time()-t0:.1f}s)")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.out_dir, "pneumonia_resnet18.pth"))
            print(f"[train] Saved new best model (val_acc={val_acc:.4f})")

    with open(os.path.join(args.out_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # Final test evaluation using the best saved model
    best_model = build_model(pretrained=False).to(device)
    best_model.load_state_dict(torch.load(os.path.join(args.out_dir, "pneumonia_resnet18.pth"), map_location=device))
    metrics = evaluate(best_model, test_loader, device)
    metrics["class_names"] = train_ds.classes
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("[train] Final test metrics:", json.dumps(metrics, indent=2))
    print(f"[train] Model saved to {args.out_dir}/pneumonia_resnet18.pth")


if __name__ == "__main__":
    main()
