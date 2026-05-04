import torch
import torch.nn as nn
from torchvision.models import resnet18

class CIFARResNet18(nn.Module):
    def __init__(self, num_classes=10, head="linear"):
        super().__init__()

        self.backbone = resnet18(weights=None)

        # Adapt ResNet for 32x32 CIFAR images.
        self.backbone.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.backbone.maxpool = nn.Identity()

        in_features = self.backbone.fc.in_features

        if head == "linear":
            self.backbone.fc = nn.Linear(in_features, num_classes)
        elif head == "mlp":
            self.backbone.fc = nn.Sequential(
                nn.Linear(in_features, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes)
            )
        else:
            raise ValueError("head must be 'linear' or 'mlp'")

    def forward(self, x):
        logits = self.backbone(x)
        return logits

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
