# Predicting Human Annotator Disagreement on CIFAR-10H

This project predicts the full human annotator label distribution for a CIFAR-10 image, instead of predicting only one hard class.

## Objective

Given a CIFAR-10 image, the model outputs a 10-dimensional probability distribution:

```text
[airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck]
```

This distribution should match CIFAR-10H human annotator disagreement labels.

## Full-marks features included

- CIFAR-10 and CIFAR-10H loading
- CIFAR-10H alignment checks
- Entropy analysis
- Per-class entropy plots
- Annotator confusion matrix
- ResNet-18 adapted for CIFAR-10
- Linear head and MLP head
- CIFAR-10 hard-label pretraining
- CIFAR-10H soft-label fine-tuning
- KL loss
- Jensen-Shannon loss
- Custom KL + entropy penalty loss
- Evaluation metrics:
  - KL Divergence
  - Jensen-Shannon Divergence
  - Cosine Similarity
  - Pearson entropy correlation
  - Spearman entropy correlation
  - Precision@100, @200, @500
- Ablation-ready structure
- Robustness checks using Gaussian noise, blur, and contrast reduction
- Grad-CAM helper script
- Report template

## Folder structure

```text
cifar10h_disagreement_project/
│
├── src/
│   ├── config.py
│   ├── dataset.py
│   ├── models.py
│   ├── losses.py
│   ├── metrics.py
│   ├── train_pretrain.py
│   ├── train_soft.py
│   ├── evaluate.py
│   ├── analyze_data.py
│   ├── robustness.py
│   └── gradcam.py
│
├── requirements.txt
├── report_template.md
└── README.md
```

## Setup

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate

pip install -r requirements.txt
```

## Dataset setup

CIFAR-10 is downloaded automatically through torchvision.

Download CIFAR-10H human labels from:

https://github.com/jcpeterson/cifar-10h

Place the CIFAR-10H file inside:

```text
data/cifar10h/
```

The code expects a `.npy` file containing soft label distributions. If the filename differs, update `CIFAR10H_LABEL_FILE` in `src/config.py`.

## Run data analysis

```bash
python src/analyze_data.py
```

## Pretrain on CIFAR-10 hard labels

```bash
python src/train_pretrain.py
```

## Fine-tune on CIFAR-10H soft labels

```bash
python src/train_soft.py --loss kl --head linear
python src/train_soft.py --loss jsd --head linear
python src/train_soft.py --loss custom --head mlp
```

## Evaluate

```bash
python src/evaluate.py --checkpoint checkpoints/best_soft_custom_mlp.pt
```

## Robustness checks

```bash
python src/robustness.py --checkpoint checkpoints/best_soft_custom_mlp.pt
```

## Grad-CAM

```bash
python src/gradcam.py --checkpoint checkpoints/best_soft_custom_mlp.pt
```

## Best model to present

For viva/report, present this as your strongest model:

```text
ResNet-18 adapted for CIFAR-10
CIFAR-10 hard-label pretraining
CIFAR-10H soft-label fine-tuning
MLP prediction head
Custom loss = KL Divergence + Entropy Error Penalty
```
