import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms
from config import DATA_DIR, CIFAR10H_LABEL_FILE, SEED, TRAIN_SIZE, VAL_SIZE, TEST_SIZE

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2470, 0.2435, 0.2616)),
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])

def load_cifar10(train=True, transform=None):
    return datasets.CIFAR10(
        root=DATA_DIR,
        train=train,
        download=True,
        transform=transform
    )

def load_cifar10h_probs():
    if not os.path.exists(CIFAR10H_LABEL_FILE):
        raise FileNotFoundError(
            f"CIFAR-10H label file not found at {CIFAR10H_LABEL_FILE}. "
            "Download CIFAR-10H from https://github.com/jcpeterson/cifar-10h "
            "and place the probability .npy file there."
        )

    probs = np.load(CIFAR10H_LABEL_FILE)

    if probs.shape[0] != 10000 or probs.shape[1] != 10:
        raise ValueError(f"Expected CIFAR-10H probabilities of shape (10000, 10), got {probs.shape}")

    row_sums = probs.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-4):
        raise ValueError("Some CIFAR-10H soft-label rows do not sum to 1.")

    return probs.astype(np.float32)

class CIFAR10HSoftDataset(Dataset):
    def __init__(self, transform=None):
        self.cifar_test = load_cifar10(train=False, transform=transform)
        self.soft_labels = load_cifar10h_probs()

    def __len__(self):
        return len(self.cifar_test)

    def __getitem__(self, idx):
        image, hard_label = self.cifar_test[idx]
        soft_label = torch.tensor(self.soft_labels[idx], dtype=torch.float32)
        return image, soft_label, hard_label, idx

def get_cifar10h_splits(transform_train=None, transform_eval=None):
    set_seed(SEED)

    full_train_aug = CIFAR10HSoftDataset(transform=transform_train)
    full_eval = CIFAR10HSoftDataset(transform=transform_eval)

    indices = np.arange(10000)
    rng = np.random.default_rng(SEED)
    rng.shuffle(indices)

    train_idx = indices[:TRAIN_SIZE]
    val_idx = indices[TRAIN_SIZE:TRAIN_SIZE + VAL_SIZE]
    test_idx = indices[TRAIN_SIZE + VAL_SIZE:TRAIN_SIZE + VAL_SIZE + TEST_SIZE]

    train_set = Subset(full_train_aug, train_idx)
    val_set = Subset(full_eval, val_idx)
    test_set = Subset(full_eval, test_idx)

    return train_set, val_set, test_set
