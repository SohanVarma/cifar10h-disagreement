# Predicting Human Annotator Disagreement on CIFAR-10H

A research-oriented deep learning project that predicts **human annotator disagreement** for CIFAR-10 images using CIFAR-10H soft labels.

Instead of predicting only one hard class label, this project predicts a full 10-class probability distribution representing how humans are likely to disagree when labeling an image.

---

## Project Idea

Standard image classifiers usually answer:

> What is the correct class of this image?

This project answers a more human-centered question:

> How will different humans label this image, and how much will they disagree?

For example, instead of predicting only:

```text
cat
```

this model predicts a distribution such as:

```text
cat: 0.60
 dog: 0.25
 deer: 0.10
 horse: 0.05
```

This means the model is not only learning the most likely class, but also learning the structure of human uncertainty and class confusion.

---

## Motivation

Traditional classification uses hard labels, where each image is assigned exactly one correct class. However, many real-world images are ambiguous. Humans may disagree because of blur, low resolution, object similarity, poor lighting, or genuine class-boundary confusion.

The CIFAR-10H dataset captures this disagreement by collecting many human annotations for each CIFAR-10 test image. The goal of this project is to train a model that matches these human label distributions rather than discarding disagreement as noise.

This makes the project useful for studying:

- uncertainty-aware image classification
- soft-label learning
- human-aligned machine learning
- robust computer vision
- model interpretability

---

## Dataset

### CIFAR-10

CIFAR-10 contains 60,000 RGB images of size 32×32 across 10 classes:

```text
airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
```

The 50,000 CIFAR-10 training images are used for hard-label pretraining.

### CIFAR-10H

CIFAR-10H provides human soft-label distributions for 10,000 CIFAR-10 test images. Each image is associated with a 10-dimensional probability vector representing the fraction of annotators who selected each class.

Example target format:

```text
[airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck]
```

The soft-label vector sums to 1 and is used as the main target for disagreement prediction.

---

## Objective

Given an input CIFAR-10 image `x`, the model predicts:

```text
q(y|x)
```

where `q(y|x)` is a 10-dimensional predicted probability distribution.

The goal is to make `q(y|x)` as close as possible to the true human annotator distribution:

```text
p(y|x)
```

from CIFAR-10H.

This is a distribution-matching problem, not a normal single-label classification problem.

---

## Model Pipeline

```text
CIFAR-10 Image
      ↓
Data preprocessing and augmentation
      ↓
ResNet-18 backbone adapted for 32×32 images
      ↓
Prediction head: Linear / MLP
      ↓
Softmax output
      ↓
10-class human disagreement distribution
```

The strongest model uses:

```text
Backbone: ResNet-18 adapted for CIFAR-10
Initialization: CIFAR-10 hard-label pretraining
Fine-tuning: CIFAR-10H soft labels
Head: MLP prediction head
Loss: KL Divergence + Entropy Error Penalty
```

---

## Key Features

- CIFAR-10 and CIFAR-10H loading
- CIFAR-10H alignment checks
- Soft-label distribution prediction
- Entropy-based disagreement analysis
- Per-class entropy analysis
- Annotator confusion matrix
- ResNet-18 adapted for CIFAR-10 image size
- Linear and MLP prediction heads
- CIFAR-10 hard-label pretraining
- CIFAR-10H soft-label fine-tuning
- KL Divergence loss
- Jensen-Shannon Divergence loss
- Custom KL + entropy penalty loss
- Evaluation using distribution-matching metrics
- Robustness checks using Gaussian noise, blur, and contrast reduction
- Grad-CAM visual explanations
- Failure-case analysis script
- Research-style report with generated result figures

---

## Loss Functions

### 1. KL Divergence

KL Divergence measures how different the predicted distribution is from the true human distribution.

```text
KL(p || q) = Σ p(y) log(p(y) / q(y))
```

Lower KL means the predicted distribution is closer to the human annotator distribution.

### 2. Jensen-Shannon Divergence

Jensen-Shannon Divergence is a symmetric and bounded version of KL Divergence. It is useful for comparing two probability distributions more stably.

### 3. Custom Loss

The best model uses a composite loss:

```text
Custom Loss = KL Divergence + λ × Entropy Error
```

where entropy error penalizes incorrect uncertainty estimation:

```text
Entropy Error = MSE(H(true distribution), H(predicted distribution))
```

This helps the model learn both:

1. which classes humans confuse, and
2. how much humans disagree.

---

## Evaluation Metrics

The project is evaluated using metrics that measure distribution quality and uncertainty prediction, not only classification accuracy.

| Metric | Purpose |
|---|---|
| KL Divergence | Measures mismatch between true and predicted distributions |
| Jensen-Shannon Divergence | Symmetric distribution comparison |
| Cosine Similarity | Measures directional similarity of probability vectors |
| Pearson Correlation | Measures linear correlation between true and predicted entropy |
| Spearman Correlation | Measures ranking quality of disagreement prediction |
| Precision@100, @200, @500 | Checks if the model identifies the most ambiguous images |

---

## Result Artifacts

The `results/` folder contains generated outputs used for analysis and reporting, such as:

```text
entropy_histogram.png
per_class_entropy.png
annotator_confusion_matrix.png
low_entropy_examples.png
high_entropy_examples.png
predicted_vs_true_entropy.png
loss_curve_custom_mlp.png
robustness_noise.png
robustness_blur.png
robustness_contrast.png
gradcam_example_0.png ... gradcam_example_9.png
test_metrics.json
robustness_results.json
```

These files support the final report and presentation by showing dataset behavior, model performance, robustness, and interpretability.

---

## Folder Structure

```text
cifar10h-disagreement/
│
├── data/
│   └── cifar10h/
│       └── cifar10h-probs.npy
│
├── src/
│   ├── config.py
│   ├── dataset.py
│   ├── models.py
│   ├── losses.py
│   ├── metrics.py
│   ├── analyze_data.py
│   ├── train_pretrain.py
│   ├── train_soft.py
│   ├── evaluate.py
│   ├── robustness.py
│   ├── gradcam.py
│   └── failure_analysis.py
│
├── checkpoints/
├── results/
├── requirements.txt
├── report_template.md
└── README.md
```

---

## Setup

Create and activate a virtual environment.

### Mac/Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Dataset Setup

CIFAR-10 is downloaded automatically through `torchvision`.

Download CIFAR-10H human probability labels from:

```text
https://github.com/jcpeterson/cifar-10h
```

Place the probability file inside:

```text
data/cifar10h/
```

Expected file path:

```text
data/cifar10h/cifar10h-probs.npy
```

If the filename is different, update `CIFAR10H_LABEL_FILE` in:

```text
src/config.py
```

---

## How to Run

### 1. Data Analysis

```bash
python3 src/analyze_data.py
```

This generates entropy plots, per-class entropy analysis, annotator confusion matrix, and low/high disagreement image examples.

### 2. CIFAR-10 Hard-Label Pretraining

```bash
python3 src/train_pretrain.py
```

This trains the backbone on standard CIFAR-10 hard labels before soft-label fine-tuning.

### 3. CIFAR-10H Soft-Label Fine-Tuning

```bash
python3 src/train_soft.py --loss kl --head linear
python3 src/train_soft.py --loss jsd --head linear
python3 src/train_soft.py --loss custom --head mlp
```

The custom-loss MLP model is the main model used for final analysis.

### 4. Evaluation

```bash
python3 src/evaluate.py --checkpoint checkpoints/best_soft_custom_mlp.pt
```

This computes KL Divergence, JSD, cosine similarity, entropy correlations, and Precision@K.

### 5. Robustness Checks

```bash
python3 src/robustness.py --checkpoint checkpoints/best_soft_custom_mlp.pt
```

This evaluates how predicted uncertainty changes under Gaussian noise, blur, and contrast reduction.

### 6. Grad-CAM Explainability

```bash
python3 src/gradcam.py --checkpoint checkpoints/best_soft_custom_mlp.pt
```

This generates visual explanations showing which image regions influence model predictions.

### 7. Failure Case Analysis

```bash
python3 src/failure_analysis.py --checkpoint checkpoints/best_soft_custom_mlp.pt
```

This identifies examples where the model badly misestimates human disagreement.

---

## Presentation Summary

This project demonstrates that human disagreement is not simply noise. By learning from CIFAR-10H soft labels, the model predicts both the most likely class and the uncertainty structure behind human decisions.

The main contribution is an uncertainty-aware image classification pipeline that models how humans perceive ambiguous images. The project includes data analysis, model training, loss comparison, robustness testing, Grad-CAM interpretability, and failure-case analysis.

---

## Key Takeaway

A normal classifier predicts one answer.

This project predicts how humans disagree.

That makes the model more informative, interpretable, and better aligned with real human perception.
