import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from config import *
from dataset import load_cifar10h_probs, load_cifar10, get_transforms

EPS = 1e-8

def entropy(p):
    return -(p * np.log2(p + EPS)).sum(axis=1)

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    probs = load_cifar10h_probs()
    ent = entropy(probs)

    cifar_test = load_cifar10(train=False, transform=None)
    hard_labels = np.array(cifar_test.targets)

    print("CIFAR-10H shape:", probs.shape)
    print("Entropy min:", ent.min())
    print("Entropy max:", ent.max())
    print("Entropy mean:", ent.mean())

    plt.figure()
    plt.hist(ent, bins=40)
    plt.xlabel("Human Label Entropy")
    plt.ylabel("Number of Images")
    plt.title("Entropy Distribution in CIFAR-10H")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "entropy_histogram.png"))
    plt.close()

    class_entropies = []
    for c in range(NUM_CLASSES):
        class_entropies.append(ent[hard_labels == c].mean())

    plt.figure()
    plt.bar(CLASS_NAMES, class_entropies)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Average Entropy")
    plt.title("Per-Class Average Human Disagreement")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "per_class_entropy.png"))
    plt.close()

    confusion_soft = np.zeros((NUM_CLASSES, NUM_CLASSES))
    for c in range(NUM_CLASSES):
        confusion_soft[c] = probs[hard_labels == c].mean()

    plt.figure(figsize=(8, 6))
    plt.imshow(confusion_soft)
    plt.xticks(range(NUM_CLASSES), CLASS_NAMES, rotation=45, ha="right")
    plt.yticks(range(NUM_CLASSES), CLASS_NAMES)
    plt.colorbar(label="Average Human Probability")
    plt.title("Annotator Distribution Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "annotator_confusion_matrix.png"))
    plt.close()

    sorted_idx = np.argsort(ent)
    low_idx = sorted_idx[:12]
    high_idx = sorted_idx[-12:]

    for name, indices in [("low_entropy_examples", low_idx), ("high_entropy_examples", high_idx)]:
        plt.figure(figsize=(10, 4))
        for i, idx in enumerate(indices):
            img, label = cifar_test[idx]
            plt.subplot(2, 6, i + 1)
            plt.imshow(img)
            plt.axis("off")
            plt.title(f"H={ent[idx]:.2f}")
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"{name}.png"))
        plt.close()

    print("Saved data visualizations to results/")

if __name__ == "__main__":
    main()
