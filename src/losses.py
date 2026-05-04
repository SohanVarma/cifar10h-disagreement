import torch
import torch.nn as nn
import torch.nn.functional as F
from config import ENTROPY_LAMBDA

EPS = 1e-8

def entropy_from_probs(p):
    return -(p * torch.log2(p + EPS)).sum(dim=1)

class SoftKLLoss(nn.Module):
    def forward(self, logits, target_probs):
        log_probs = F.log_softmax(logits, dim=1)
        return F.kl_div(log_probs, target_probs, reduction="batchmean")

class SoftCrossEntropyLoss(nn.Module):
    def forward(self, logits, target_probs):
        log_probs = F.log_softmax(logits, dim=1)
        return -(target_probs * log_probs).sum(dim=1).mean()

class JensenShannonLoss(nn.Module):
    def forward(self, logits, target_probs):
        pred_probs = F.softmax(logits, dim=1)
        m = 0.5 * (target_probs + pred_probs)

        kl_pm = F.kl_div(torch.log(m + EPS), target_probs, reduction="batchmean")
        kl_qm = F.kl_div(torch.log(m + EPS), pred_probs, reduction="batchmean")

        return 0.5 * (kl_pm + kl_qm)

class CustomKLEntropyLoss(nn.Module):
    def __init__(self, entropy_lambda=ENTROPY_LAMBDA):
        super().__init__()
        self.kl = SoftKLLoss()
        self.entropy_lambda = entropy_lambda

    def forward(self, logits, target_probs):
        pred_probs = F.softmax(logits, dim=1)

        kl_loss = self.kl(logits, target_probs)

        true_entropy = entropy_from_probs(target_probs)
        pred_entropy = entropy_from_probs(pred_probs)

        entropy_loss = F.mse_loss(pred_entropy, true_entropy)

        return kl_loss + self.entropy_lambda * entropy_loss

def get_loss(name):
    if name == "kl":
        return SoftKLLoss()
    if name == "ce":
        return SoftCrossEntropyLoss()
    if name == "jsd":
        return JensenShannonLoss()
    if name == "custom":
        return CustomKLEntropyLoss()
    raise ValueError("loss must be one of: kl, ce, jsd, custom")
