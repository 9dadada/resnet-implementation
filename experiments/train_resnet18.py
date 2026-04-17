"""
ResNet18 학습 스크립트.
CIFAR-10에서 학습한 뒤 epoch별 loss/accuracy를 results/resnet18_history.json에 저장한다.
"""
import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.resnet import ResNet18

BATCH_SIZE = 128
EPOCHS = 30
LEARNING_RATE = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
NUM_CLASSES = 10
MODEL_NAME = "resnet18"

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


def train_one_epoch(model, optimizer, criterion):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
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
    running_loss, correct, total = 0.0, 0, 0
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


def main():
    model = ResNet18(num_classes=NUM_CLASSES).to(device)
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

        print(f"[{MODEL_NAME}] Epoch {epoch+1}/{EPOCHS} | "
              f"Train: {train_acc:.2f}% | Test: {test_acc:.2f}%")
        best_acc = max(best_acc, test_acc)

    print(f"[{MODEL_NAME}] Best Test Accuracy: {best_acc:.2f}%")

    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)
    output_path = os.path.join(results_dir, f"{MODEL_NAME}_history.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"model": MODEL_NAME, "best_acc": best_acc,
                   "epochs": EPOCHS, "history": history}, f, indent=2)
    print(f"History saved to {output_path}")


if __name__ == "__main__":
    main()
