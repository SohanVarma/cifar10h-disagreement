import os
import argparse
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from config import *
from dataset import get_transforms, get_cifar10h_splits
from models import CIFARResNet18
from metrics import evaluate_distribution_metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--head", choices=["linear", "mlp"], default="mlp")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    _, _, test_set = get_cifar10h_splits(
        transform_train=get_transforms(train=True),
        transform_eval=get_transforms(train=False)
    )

    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = CIFARResNet18(num_classes=NUM_CLASSES, head=args.head).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    all_true, all_pred = [], []

    with torch.no_grad():
        for images, soft_labels, hard_labels, idx in test_loader:
            images = images.to(device)
            logits = model(images)
            probs = F.softmax(logits, dim=1).cpu()

            all_true.append(soft_labels)
            all_pred.append(probs)

    true_probs = torch.cat(all_true).numpy()
    pred_probs = torch.cat(all_pred).numpy()

    results, true_entropy, pred_entropy = evaluate_distribution_metrics(true_probs, pred_probs)

    print(json.dumps(results, indent=4))

    with open(os.path.join(RESULTS_DIR, "test_metrics.json"), "w") as f:
        json.dump(results, f, indent=4)

    plt.figure()
    plt.scatter(true_entropy, pred_entropy, alpha=0.5)
    plt.xlabel("True Human Entropy")
    plt.ylabel("Predicted Entropy")
    plt.title("Predicted vs True Entropy")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "predicted_vs_true_entropy.png"))
    plt.close()

    print("Saved results to results/")

if __name__ == "__main__":
    main()
