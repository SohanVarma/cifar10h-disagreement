import os
import argparse
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import ImageFilter, ImageEnhance

from config import *
from dataset import get_cifar10h_splits, get_transforms
from models import CIFARResNet18
from metrics import entropy_np

class AddGaussianNoise:
    def __init__(self, std):
        self.std = std

    def __call__(self, tensor):
        return torch.clamp(tensor + torch.randn_like(tensor) * self.std, -3, 3)

class PILGaussianBlur:
    def __init__(self, radius):
        self.radius = radius

    def __call__(self, img):
        return img.filter(ImageFilter.GaussianBlur(radius=self.radius))

class PILContrast:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, img):
        return ImageEnhance.Contrast(img).enhance(self.factor)

def make_transform(corruption, severity):
    base = []

    if corruption == "blur":
        base.append(PILGaussianBlur(radius=severity))
    elif corruption == "contrast":
        base.append(PILContrast(factor=max(0.1, 1.0 - 0.15 * severity)))

    base.extend([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])

    if corruption == "noise":
        base.append(AddGaussianNoise(std=0.05 * severity))

    return transforms.Compose(base)

def predict_entropy(model, loader, device):
    model.eval()
    all_pred = []

    with torch.no_grad():
        for images, soft_labels, hard_labels, idx in loader:
            images = images.to(device)
            probs = F.softmax(model(images), dim=1).cpu().numpy()
            all_pred.append(probs)

    pred_probs = np.concatenate(all_pred, axis=0)
    return entropy_np(pred_probs).mean()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--head", choices=["linear", "mlp"], default="mlp")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CIFARResNet18(num_classes=NUM_CLASSES, head=args.head).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    corruptions = ["noise", "blur", "contrast"]
    severities = [0, 1, 2, 3, 4, 5]

    results = {}

    for corruption in corruptions:
        entropy_values = []

        for s in severities:
            transform = get_transforms(train=False) if s == 0 else make_transform(corruption, s)

            _, _, test_set = get_cifar10h_splits(
                transform_train=get_transforms(train=True),
                transform_eval=transform
            )
            loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

            mean_entropy = predict_entropy(model, loader, device)
            entropy_values.append(float(mean_entropy))

        results[corruption] = entropy_values

        plt.figure()
        plt.plot(severities, entropy_values, marker="o")
        plt.xlabel("Corruption Severity")
        plt.ylabel("Mean Predicted Entropy")
        plt.title(f"Entropy Response to {corruption}")
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"robustness_{corruption}.png"))
        plt.close()

    with open(os.path.join(RESULTS_DIR, "robustness_results.json"), "w") as f:
        json.dump(results, f, indent=4)

    print("Saved robustness results to results/")

if __name__ == "__main__":
    main()
