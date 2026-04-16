"""
PyTorch 공식 ResNet18 vs 내 구현 ResNet18 비교 실험
동일 조건(CIFAR-10, 같은 하이퍼파라미터)에서 학습하여 성능을 비교한다.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.resnet import ResNet18

BATCH_SIZE = 128
EPOCHS = 30
LEARNING_RATE = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
NUM_CLASSES = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2470, 0.2435, 0.2616]),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2470, 0.2435, 0.2616]),
])

train_dataset = datasets.CIFAR10(root="./data", train=True, download=True,
                                 transform=train_transform)
test_dataset = datasets.CIFAR10(root="./data", train=False, download=True,
                                transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=2, pin_memory=True)


def make_pytorch_resnet18(num_classes):
    """PyTorch 공식 ResNet18을 CIFAR-10용으로 수정"""
    model = models.resnet18(weights=None, num_classes=num_classes)
    # 공식 모델은 ImageNet용(7x7 conv + maxpool)이라 CIFAR-10에는 과한 다운샘플링
    # 그대로 사용해서 차이를 확인한다
    return model


def train_one_epoch(model, optimizer, criterion):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / len(train_loader), 100. * correct / total


def evaluate(model, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return running_loss / len(test_loader), 100. * correct / total


def train_model(model, name):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE,
                          momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    history = {"train_loss": [], "test_loss": [], "train_acc": [], "test_acc": []}
    best_acc = 0.0

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(model, optimizer, criterion)
        test_loss, test_acc = evaluate(model, criterion)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)

        print(f"[{name}] Epoch {epoch+1}/{EPOCHS} | "
              f"Train: {train_acc:.2f}% | Test: {test_acc:.2f}%")

        if test_acc > best_acc:
            best_acc = test_acc

    print(f"[{name}] Best Test Accuracy: {best_acc:.2f}%\n")
    return history, best_acc


def save_comparison_plot(hist_mine, hist_official):
    os.makedirs("experiments", exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(hist_mine["test_loss"], label="My ResNet18", linewidth=2)
    ax1.plot(hist_official["test_loss"], label="PyTorch ResNet18", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Test Loss")
    ax1.set_title("Test Loss Comparison")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(hist_mine["test_acc"], label="My ResNet18", linewidth=2)
    ax2.plot(hist_official["test_acc"], label="PyTorch ResNet18", linewidth=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Test Accuracy (%)")
    ax2.set_title("Test Accuracy Comparison")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("experiments/comparison_result.png", dpi=150)
    plt.close()
    print("Comparison plot saved to experiments/comparison_result.png")


if __name__ == "__main__":
    print("=" * 50)
    print("My ResNet18 (CIFAR-10 optimized)")
    print("=" * 50)
    my_model = ResNet18(num_classes=NUM_CLASSES).to(device)
    hist_mine, best_mine = train_model(my_model, "Mine")

    print("=" * 50)
    print("PyTorch Official ResNet18 (ImageNet default)")
    print("=" * 50)
    official_model = make_pytorch_resnet18(NUM_CLASSES).to(device)
    hist_official, best_official = train_model(official_model, "PyTorch")

    print("=" * 50)
    print(f"My ResNet18 Best Acc:      {best_mine:.2f}%")
    print(f"PyTorch ResNet18 Best Acc: {best_official:.2f}%")
    print("=" * 50)

    save_comparison_plot(hist_mine, hist_official)
