import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 🔹 Define HybridMergeNet Architecture
class HybridMergeNet(nn.Module):
    def __init__(self, num_classes=2):
        super(HybridMergeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.skip1 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn_skip1 = nn.BatchNorm2d(32)

        self.reduce1 = nn.Conv2d(32, 16, 1)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(48, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.skip2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn_skip2 = nn.BatchNorm2d(64)

        self.reduce2 = nn.Conv2d(64, 32, 1)
        self.pool2 = nn.MaxPool2d(2)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(96, num_classes)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        skip = self.relu(self.bn_skip1(self.skip1(x)))
        x = x + skip

        reduce = self.reduce1(x)
        x = torch.cat([x, reduce], dim=1)
        x = self.pool1(x)

        y = self.relu(self.bn2(self.conv2(x)))
        skip2 = self.relu(self.bn_skip2(self.skip2(y)))
        y = y + skip2

        reduce2 = self.reduce2(y)
        y = torch.cat([y, reduce2], dim=1)
        y = self.pool2(y)

        y = self.gap(y)
        y = y.view(y.size(0), -1)
        out = self.fc(y)
        return out

# 🔹 Data Preparation
IMG_DIR = "images"
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.9, 1.0)),
    transforms.ToTensor(),
])

train_dataset = datasets.ImageFolder(IMG_DIR, transform=transform)
val_size = int(0.2 * len(train_dataset))
train_size = len(train_dataset) - val_size
train_ds, val_ds = torch.utils.data.random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# 🔹 Training Loop
model = HybridMergeNet(num_classes=len(train_dataset.classes)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

train_acc_hist, val_acc_hist = [], []
train_loss_hist, val_loss_hist = [], []

for epoch in range(EPOCHS):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    train_loss = running_loss / total
    train_acc = correct / total
    train_loss_hist.append(train_loss)
    train_acc_hist.append(train_acc)

    # Validation
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)
    val_loss = val_loss / val_total
    val_acc = val_correct / val_total
    val_loss_hist.append(val_loss)
    val_acc_hist.append(val_acc)

    print(f"Epoch {epoch+1}/{EPOCHS} - "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

# 🔹 Plot Training History
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, EPOCHS+1), train_acc_hist, 'b-', label='Training Accuracy')
plt.plot(range(1, EPOCHS+1), val_acc_hist, 'r-', label='Validation Accuracy')
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, EPOCHS+1), train_loss_hist, 'b-', label='Training Loss')
plt.plot(range(1, EPOCHS+1), val_loss_hist, 'r-', label='Validation Loss')
plt.title('Training vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 🔹 Save Model
torch.save(model.state_dict(), "final_hybrid_mergenet_pytorch.pth")