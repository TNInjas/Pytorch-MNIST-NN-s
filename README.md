# MNIST Digit Recognition in PyTorch: From a Linear Network to a 99%+ CNN

A hands-on PyTorch project where I classify handwritten digits (MNIST) with three models of increasing complexity, and see how much each architectural step actually buys.

I started this to brush up on PyTorch fundamentals. The goal along the way was simple: **start with a plain fully-connected network, then push towards 99% accuracy.**

## Results

| Model | Architecture | Epochs | Test accuracy |
|---|---|---|---|
| `SimpleNN` | 2 × `nn.Linear` (784 → 128 → 10) | 5 | **97.51%** |
| `SimpleCNN` | 2 conv layers + max-pooling + 2 linear layers | 15 | **98.44%**\* |
| `AdvancedCNN` | 4 conv layers (2 blocks) + max-pooling + dropout + 2 linear layers | 15 | **99.44%** |

\* Measured on the first test batch only (64 images). See [Known limitations](#known-limitations).

The jump from a linear network to a CNN is the main lesson: flattening an image throws away its spatial structure, and convolutions keep it.

## What's inside

1. **Setup**: dependency version check, GPU/CPU detection, MNIST download via `torchvision`.
2. **Tensor practice**: inspecting shapes, `view`, `squeeze`, and computing/applying mean and std normalization on a batch.
3. **`SimpleNN`**: a baseline using only `nn.Linear` layers.
4. **`SimpleCNN`**: two `Conv2d` + `MaxPool2d` blocks.
5. **`AdvancedCNN`**: four `Conv2d` layers in two blocks (64 → 128 channels), `Dropout` at 0.25 / 0.25 / 0.5, and a 256-unit classifier head.
6. **Prediction visualization**: a 4×4 grid of test images with predicted label, confidence, and green/red correctness coloring.

All three models use `CrossEntropyLoss` and the `Adam` optimizer with batch size 64.

## Model details

**AdvancedCNN**

```
Input (1×28×28)
 ├─ Conv(1→64, 3×3, pad 1) → ReLU
 ├─ Conv(64→64, 3×3, pad 1) → ReLU → MaxPool(2) → Dropout(0.25)
 ├─ Conv(64→128, 3×3, pad 1) → ReLU
 ├─ Conv(128→128, 3×3, pad 1) → ReLU → MaxPool(2) → Dropout(0.25)
 ├─ Flatten (128×7×7)
 ├─ Linear(6272→256) → ReLU → Dropout(0.5)
 └─ Linear(256→10)
```

## Getting started

```bash
git clone https://github.com/EMP0RI0M/pytorch-mnist-cnn.git
cd pytorch-mnist-cnn
pip install -r requirements.txt
jupyter notebook notebooks/mnist_classification.ipynb
```

MNIST is downloaded automatically into `data/` on the first run. A GPU is optional but makes training much faster; the notebook falls back to CPU automatically.

## Project structure

```
.
├── notebooks/
│   └── mnist_classification.ipynb
├── requirements.txt
├── .gitignore
└── README.md
```

## Known limitations

- The `SimpleCNN` evaluation cell reads a single batch from the test loader, so its 98.44% comes from 64 images, not the full 10,000. The other two models are evaluated on the full test set.
- The mean/std normalization is explored on one batch but not applied in the training pipeline; the models train on raw `ToTensor()` inputs.
- Random seeds are defined but not applied, so exact numbers will vary slightly between runs.
- The visualization helper was written with AI assistance.

## Ideas for next steps

- Evaluate `SimpleCNN` on the full test set
- Apply MNIST normalization (mean 0.1307, std 0.3081) in the transform pipeline
- Add batch normalization and a learning-rate scheduler
- Plot training loss and accuracy curves
- Add a confusion matrix to see which digits get mixed up
- Try Fashion-MNIST or CIFAR-10 with the same architectures

## Tech stack

Python · PyTorch · torchvision · matplotlib · NumPy

## Author

**Rafi**: [GitHub](https://github.com/EMP0RI0M)
