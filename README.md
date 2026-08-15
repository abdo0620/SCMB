# Skin Cancer Classification: Malignant vs. Benign

## Overview

Early and accurate cancer detection is one of the most critical factors in improving patient survival rates. When identified in its initial stages, before metastasis to other organs, tumors are typically smaller and more treatable, allowing for less invasive interventions and significantly better long-term outcomes.

This project implements a deep learning pipeline to classify dermatological images as **malignant** or **benign**, using transfer learning on a pretrained convolutional neural network (CNN).

## Background: Benign vs. Malignant Tumors

- **Benign tumors** are noncancerous, generally do not spread to other parts of the body, and are typically not life-threatening. They require regular monitoring and treatment only if they grow and compress adjacent tissue.
- **Malignant tumors** (cancer) can invade surrounding tissue and spread to other organs via the bloodstream or lymphatic system, forming metastases. Early detection is essential to prevent disease progression.

## Approach

The model is built on **MobileNetV2**, a lightweight CNN architecture pretrained on ImageNet, adapted to this binary classification task through transfer learning and fine-tuning:

1. **Feature extraction phase**: The pretrained MobileNetV2 base is frozen, and a custom classification head (Global Average Pooling → Dense(128, ReLU) → Dense(1, Sigmoid)) is trained on top.
2. **Fine-tuning phase**: The upper layers of the base model are unfrozen (from layer 125 onward) and trained with a significantly lower learning rate to refine feature representations specific to dermatological imaging, while BatchNormalization layers are kept frozen to preserve pretrained statistics.
3. **Data augmentation** is applied during training to improve generalization and reduce overfitting.
4. **Early stopping** (patience-based, with best-weight restoration) is used to prevent overfitting and select the optimal model checkpoint.

### Technical Stack
- **Framework**: TensorFlow / Keras
- **Base model**: MobileNetV2 (ImageNet weights)
- **Input size**: 224×224×3
- **Optimizer**: Adam (differentiated learning rates for feature extraction vs. fine-tuning)
- **Loss function**: Binary Crossentropy

## Dataset

[Skin Cancer: Malignant vs. Benign](https://www.kaggle.com/datasets/fanconic/skin-cancer-malignant-vs-benign) (Kaggle, by fanconic)

The dataset consists of dermatoscopic images labeled as malignant or benign, split into training and test sets.

## Results

## Fine-tuning: Layer Unfreezing Ablation

Comparison of validation performance when unfreezing MobileNetV2 from different starting layers (out of ~154 total layers), with BatchNormalization layers kept frozen throughout.

| Layers Unfrozen (from) | Val Accuracy | Val Precision | Val Recall | Val F1 | Val AUC | Val FN | Val FP |
|---|---|---|---|---|---|---|---|
| 125 | 87.86% | 89.66% | 80.89% | 85.05% | 94.95% | 43 | 21 |
| 100 | 89.18% | 85.90% | 89.33% | 87.58% | 95.47% | 24 | 33 |
| 90  | **90.70%** | 87.29% | **91.56%** | **89.37%** | **95.85%** | **19** | 30 |
| 75  | 88.24% | 82.73% | 91.56% | 86.92% | 95.41% | **19** | 43 |

**Best configuration: unfreezing from layer 90 onward**, offering the best balance of accuracy, F1, and AUC while matching the top recall (91.56%) achieved at layer 75 — but with significantly fewer false positives (30 vs. 43).


## Project Structure

```
├── datasets/           # Raw dataset (train/test)
├── notebooks/          # Training history logs
├── weights/             # Saved model weights
├── Model.py             # Model architecture definition
├── dataset.py           # Data loading and augmentation utilities
├── train.py              # Training pipeline
└── README.md
```

## Key Takeaways

This project provided hands-on experience with transfer learning best practices for medical image classification, including the importance of careful BatchNormalization handling during fine-tuning, learning rate scheduling across training phases, and mitigating overfitting through early stopping and data augmentation.

## Future Improvements

- Experiment with alternative architectures (EfficientNet, ResNet)
