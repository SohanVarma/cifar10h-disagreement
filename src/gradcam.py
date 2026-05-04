import os
import argparse
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import cv2

from torch.utils.data import DataLoader
from config import *
from dataset import get_transforms, get_cifar10h_splits
from models import CIFARResNet18
from losses import entropy_from_probs

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, image_tensor, class_idx):
        self.model.zero_grad()
        logits = self.model(image_tensor)
        score = logits[:, class_idx].sum()
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1).squeeze()
        cam = torch.relu(cam)
        cam = cam.cpu().numpy()

        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam

def unnormalize(img_tensor):
    mean = torch.tensor((0.4914, 0.4822, 0.4465)).view(3, 1, 1)
    std = torch.tensor((0.2470, 0.2435, 0.2616)).view(3, 1, 1)
    img = img_tensor.cpu() * std + mean
    img = torch.clamp(img, 0, 1)
    return img.permute(1, 2, 0).numpy()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--head", choices=["linear", "mlp"], default="mlp")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CIFARResNet18(num_classes=NUM_CLASSES, head=args.head).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    _, _, test_set = get_cifar10h_splits(
        transform_train=get_transforms(train=True),
        transform_eval=get_transforms(train=False)
    )

    loader = DataLoader(test_set, batch_size=1, shuffle=False)

    target_layer = model.backbone.layer4[-1]
    cam_generator = GradCAM(model, target_layer)

    saved = 0

    for image, soft_label, hard_label, idx in loader:
        image = image.to(device)
        soft_label = soft_label.to(device)

        logits = model(image)
        probs = F.softmax(logits, dim=1)
        pred_class = probs.argmax(dim=1).item()

        cam = cam_generator.generate(image, pred_class)
        cam = cv2.resize(cam, (32, 32))

        img = unnormalize(image[0])
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
        overlay = 0.5 * img + 0.5 * heatmap

        true_entropy = entropy_from_probs(soft_label).item()
        pred_entropy = entropy_from_probs(probs).item()

        plt.figure(figsize=(8, 3))
        plt.subplot(1, 3, 1)
        plt.imshow(img)
        plt.axis("off")
        plt.title("Image")

        plt.subplot(1, 3, 2)
        plt.imshow(cam)
        plt.axis("off")
        plt.title("Grad-CAM")

        plt.subplot(1, 3, 3)
        plt.imshow(overlay)
        plt.axis("off")
        plt.title(f"T H={true_entropy:.2f}, P H={pred_entropy:.2f}")

        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"gradcam_example_{saved}.png"))
        plt.close()

        saved += 1
        if saved >= 10:
            break

    print("Saved Grad-CAM examples to results/")

if __name__ == "__main__":
    main()
