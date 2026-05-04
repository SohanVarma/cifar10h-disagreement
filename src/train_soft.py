import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from config import *
from dataset import set_seed, get_transforms, get_cifar10h_splits
from models import CIFARResNet18, count_parameters
from losses import get_loss
from metrics import evaluate_distribution_metrics

def run_epoch(model, loader, optimizer, criterion, device, train=True):
    model.train() if train else model.eval()

    total_loss = 0
    all_true = []
    all_pred = []

    for images, soft_labels, hard_labels, idx in tqdm(loader, desc="Train" if train else "Eval"):
        images = images.to(device)
        soft_labels = soft_labels.to(device)

        with torch.set_grad_enabled(train):
            logits = model(images)
            loss = criterion(logits, soft_labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        probs = F.softmax(logits, dim=1)

        total_loss += loss.item() * images.size(0)
        all_true.append(soft_labels.detach().cpu())
        all_pred.append(probs.detach().cpu())

    true_probs = torch.cat(all_true).numpy()
    pred_probs = torch.cat(all_pred).numpy()
    metrics, _, _ = evaluate_distribution_metrics(true_probs, pred_probs)

    return total_loss / len(loader.dataset), metrics

def plot_curves(train_losses, val_losses, output_path):
    plt.figure()
    plt.plot(train_losses, label="train loss")
    plt.plot(val_losses, label="validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--loss", choices=["kl", "ce", "jsd", "custom"], default="custom")
    parser.add_argument("--head", choices=["linear", "mlp"], default="mlp")
    parser.add_argument("--use_pretrained", action="store_true")
    args = parser.parse_args()

    set_seed(SEED)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_set, val_set, test_set = get_cifar10h_splits(
        transform_train=get_transforms(train=True),
        transform_eval=get_transforms(train=False)
    )

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = CIFARResNet18(num_classes=NUM_CLASSES, head=args.head).to(device)

    if args.use_pretrained:
        ckpt_path = os.path.join(CHECKPOINT_DIR, "resnet18_cifar10_pretrained.pt")
        if os.path.exists(ckpt_path):
            pretrained = torch.load(ckpt_path, map_location=device)
            model_dict = model.state_dict()
            compatible = {k: v for k, v in pretrained.items() if k in model_dict and v.shape == model_dict[k].shape}
            model_dict.update(compatible)
            model.load_state_dict(model_dict)
            print("Loaded compatible CIFAR-10 pretrained weights.")
        else:
            print("Pretrained checkpoint not found. Training from random initialization.")

    total, trainable = count_parameters(model)
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")

    criterion = get_loss(args.loss)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=4, factor=0.5)

    best_val = float("inf")
    patience = 10
    patience_counter = 0
    train_losses, val_losses = [], []

    best_path = os.path.join(CHECKPOINT_DIR, f"best_soft_{args.loss}_{args.head}.pt")

    for epoch in range(SOFT_EPOCHS):
        train_loss, train_metrics = run_epoch(model, train_loader, optimizer, criterion, device, train=True)
        val_loss, val_metrics = run_epoch(model, val_loader, optimizer, criterion, device, train=False)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        print(
            f"Epoch {epoch+1}/{SOFT_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val KL: {val_metrics['KL_mean']:.4f} | "
            f"Val JSD: {val_metrics['JSD_mean']:.4f} | "
            f"Val Pearson: {val_metrics['Pearson_entropy']:.4f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_path)
            print(f"Saved best checkpoint: {best_path}")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    plot_curves(
        train_losses,
        val_losses,
        os.path.join(RESULTS_DIR, f"loss_curve_{args.loss}_{args.head}.png")
    )

if __name__ == "__main__":
    main()
