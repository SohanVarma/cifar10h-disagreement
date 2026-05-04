import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(ROOT_DIR, "data")
CIFAR10H_DIR = os.path.join(DATA_DIR, "cifar10h")

# Change this if your downloaded CIFAR-10H file has another name.
CIFAR10H_LABEL_FILE = os.path.join(CIFAR10H_DIR, "cifar10h-probs.npy")

CHECKPOINT_DIR = os.path.join(ROOT_DIR, "checkpoints")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")

NUM_CLASSES = 10
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

SEED = 42
BATCH_SIZE = 128
NUM_WORKERS = 2
PRETRAIN_EPOCHS = 20
SOFT_EPOCHS = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
ENTROPY_LAMBDA = 0.5

TRAIN_SIZE = 6000
VAL_SIZE = 2000
TEST_SIZE = 2000
