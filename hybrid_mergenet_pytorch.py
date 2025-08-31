#!/usr/bin/env python3

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    try:
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass


class HybridMergeNet(nn.Module):
    def __init__(self, num_classes=2):
        super(HybridMergeNet, self).__init__()
        # Stage 1
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.skip1 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn_skip1 = nn.BatchNorm2d(32)
        self.reduce1 = nn.Conv2d(32, 16, 1)
        self.pool1 = nn.MaxPool2d(2)

        # Stage 2
        self.conv2 = nn.Conv2d(48, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.skip2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn_skip2 = nn.BatchNorm2d(64)
        self.reduce2 = nn.Conv2d(64, 32, 1)
        self.pool2 = nn.MaxPool2d(2)

        # Head
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(96, num_classes)

    def forward(self, x):
        # Stage 1
        x = self.relu(self.bn1(self.conv1(x)))
        skip = self.relu(self.bn_skip1(self.skip1(x)))
        x = x + skip
        reduce = self.reduce1(x)
        x = torch.cat([x, reduce], dim=1)
        x = self.pool1(x)

        # Stage 2
        y = self.relu(self.bn2(self.conv2(x)))
        skip2 = self.relu(self.bn_skip2(self.skip2(y)))
        y = y + skip2
        reduce2 = self.reduce2(y)
        y = torch.cat([y, reduce2], dim=1)
        y = self.pool2(y)

        # Head
        y = self.gap(y)
        y = y.view(y.size(0), -1)
        out = self.fc(y)
        return out


def read_classes(classes_file="classes.txt"):
    if classes_file and os.path.isfile(classes_file):
        with open(classes_file, 'r') as f:
            classes = [line.strip() for line in f if line.strip()]
        return classes
    return None


def build_dataloaders(img_dir="images", img_size=224, batch_size=32, num_workers=2, classes_file="classes.txt"):
    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop(img_size, scale=(0.9, 1.0)),
        transforms.ToTensor(),
    ])
    dataset = datasets.ImageFolder(img_dir, transform=tfm)

    # Filter/reorder to match classes.txt if provided
    desired = read_classes(classes_file) if classes_file else None
    if desired:
        desired_set = set(desired)
        new_samples = []
        for path, _y in list(dataset.samples):
            cls_name = os.path.basename(os.path.dirname(path))
            if cls_name in desired_set:
                new_idx = desired.index(cls_name)
                new_samples.append((path, new_idx))
        if not new_samples:
            raise RuntimeError(f"No samples found in '{img_dir}' matching classes from {classes_file}")
        dataset.samples = new_samples
        if hasattr(dataset, 'imgs'):
            dataset.imgs = new_samples
        dataset.classes = desired
        dataset.class_to_idx = {c: i for i, c in enumerate(desired)}
        try:
            dataset.targets = [y for _, y in new_samples]
        except Exception:
            pass

    pin = torch.cuda.is_available()
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)
    return train_loader, val_loader, len(dataset.classes)


def train_pytorch_model(
    epochs=20,
    lr=1e-4,
    img_dir="images",
    img_size=224,
    batch_size=32,
    save_path="final_hybrid_mergenet_pytorch.pth",
    device_override=None,
    classes_file="classes.txt",
):
    dev = device_override or device
    train_loader, val_loader, num_classes = build_dataloaders(img_dir, img_size, batch_size, classes_file=classes_file)
    if num_classes < 2:
        raise RuntimeError("Need at least 2 classes for training.")
    model = HybridMergeNet(num_classes=num_classes).to(dev)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs = inputs.to(dev, non_blocking=True)
            labels = labels.to(dev, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total if total else 0.0
        train_acc = correct / total if total else 0.0

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(dev, non_blocking=True)
                labels = labels.to(dev, non_blocking=True)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
        val_loss = val_loss / val_total if val_total else 0.0
        val_acc = val_correct / val_total if val_total else 0.0

        print(
            f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

    torch.save(model.state_dict(), save_path)
    print(f"Saved weights: {save_path}")
    return model


def export_onnx(model: nn.Module, onnx_out: str = "hybrid_mergenet.onnx", img_size: int = 224):
    model_cpu = model.to('cpu').eval()
    dummy = torch.randn(1, 3, img_size, img_size)
    # Try new exporter first; if it fails, fall back to legacy with static shapes
    try:
        torch.onnx.export(
            model_cpu,
            dummy,
            onnx_out,
            opset_version=12,
            input_names=["input"],
            output_names=["logits"],
            dynamo=True,
        )
        print(f"Exported ONNX (dynamo): {onnx_out}")
        return
    except Exception as e:
        print(f"Dynamo exporter failed, falling back to legacy: {e}")
    # Legacy exporter with static shapes (no dynamic_axes) and constant folding
    torch.onnx.export(
        model_cpu,
        dummy,
        onnx_out,
        opset_version=12,
        input_names=["input"],
        output_names=["logits"],
        do_constant_folding=True,
    )
    print(f"Exported ONNX (legacy): {onnx_out}")


if __name__ == "__main__":
    # No arguments: use defaults and run
    img_dir = 'images'
    img_size = 224
    batch = 32
    epochs = 20
    lr = 1e-4
    pth_out = 'final_hybrid_mergenet_pytorch.pth'
    onnx_out = 'hybrid_mergenet.onnx'
    classes_file = 'classes.txt' if os.path.isfile('classes.txt') else None

    model = train_pytorch_model(
        epochs=epochs,
        lr=lr,
        img_dir=img_dir,
        img_size=img_size,
        batch_size=batch,
        save_path=pth_out,
        classes_file=classes_file,
    )

    export_onnx(model, onnx_out, img_size)