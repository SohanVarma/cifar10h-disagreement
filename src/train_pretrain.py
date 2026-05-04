import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import *
from dataset import set_seed, get_transforms, load_cifar10
from models import CIFARResNet18, count_parameters

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for images, labels in tqdm(loader, desc="Pretraining"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total

def main():
    set_seed(SEED)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_set = load_cifar10(train=True, transform=get_transforms(train=True))
    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )

    model = CIFARResNet18(num_classes=NUM_CLASSES, head="linear").to(device)
    total, trainable = count_parameters(model)
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(PRETRAIN_EPOCHS):
        loss, acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}/{PRETRAIN_EPOCHS} | Loss: {loss:.4f} | Acc: {acc:.4f}")

    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "resnet18_cifar10_pretrained.pt"))
    print("Saved checkpoint: checkpoints/resnet18_cifar10_pretrained.pt")

if __name__ == "__main__":
    main()
