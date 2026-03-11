# Multi-Layer Perceptron from Scratch using NumPy

**Name:** Rishabh Mishra

**Roll No:** DA25M025

**WandB Report:** [Assignment-1](https://wandb.ai/rishabh-mishra-cer16-iitmaana/mlp-100-sweeps/reports/Assignment-1--VmlldzoxNjEzNTIyNg)

**Project Report [Alternate to Wandb]:** [Assignment-1](https://github.com/r1sha13h/DA6401_Assignment_1/blob/main/PROJECT_REPORT.md)

**GitHub Repository:** [DA6401_Assignment_1](https://github.com/r1sha13h/DA6401_Assignment_1)

---

A complete implementation of a configurable Multi-Layer Perceptron (MLP) for MNIST and Fashion-MNIST classification, built entirely with NumPy. This project demonstrates fundamental neural network concepts including forward/backward propagation, various optimization algorithms, and comprehensive experiment tracking.

## Table of Contents
- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Implementation Details](#implementation-details)
- [Experiments and Results](#experiments-and-results)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

---

## Project Overview

This project implements a fully-featured neural network library from scratch using only NumPy, designed for educational purposes and deep learning fundamentals understanding. The implementation includes:

- **Pure NumPy**: No PyTorch, TensorFlow, or JAX dependencies for core functionality
- **Modular Design**: Clean separation of concerns with reusable components
- **Production-Ready**: Comprehensive error handling, logging, and experiment tracking
- **Educational**: Well-documented code with clear mathematical foundations

### Key Statistics
- **Total Lines of Code**: ~1,484
- **Core Components**: 10 Python modules
- **Supported Optimizers**: 4 (SGD, Momentum, NAG, RMSProp)
- **Activation Functions**: 4 (ReLU, Sigmoid, Tanh, Linear)
- **Loss Functions**: 2 (Cross-Entropy, MSE)

---

## Project Structure

```
DA6401_Assignment_1/
├── src/
│   ├── ann/                          # Neural network core module
│   │   ├── __init__.py              # Package exports
│   │   ├── activations.py           # Activation functions + derivatives (61 lines)
│   │   ├── neural_layer.py          # Single layer implementation (98 lines)
│   │   ├── neural_network.py        # Main MLP class (284 lines)
│   │   ├── objective_functions.py   # Loss functions (101 lines)
│   │   └── optimizers.py            # Optimization algorithms (161 lines)
│   ├── utils/                        # Utility functions
│   │   ├── __init__.py
│   │   └── data_loader.py           # Data loading & preprocessing (74 lines)
│   ├── plots/                        # Experiment visualization outputs
│   │   ├── Exp-2.1/                 # Data exploration plots
│   │   ├── Exp-2.2/                 # Hyperparameter sweep plots (100+ configs)
│   │   ├── Exp-2.3/                 # Optimizer comparison plots
│   │   ├── Exp-2.4/                 # Vanishing gradient analysis
│   │   ├── Exp-2.5/                 # Dead neuron investigation
│   │   ├── Exp-2.6/                 # Loss function comparison
│   │   ├── Exp-2.7/                 # Overfitting analysis
│   │   ├── Exp-2.8/                 # Error analysis
│   │   ├── Exp-2.9/                 # Weight initialization impact
│   │   └── Exp-2.10/                # Transfer learning (Fashion-MNIST)
│   ├── train.py                      # Training script with CLI (389 lines)
│   ├── inference.py                  # Evaluation script (186 lines)
│   ├── experiment.py                 # Comprehensive experiments suite (1071 lines)
│   ├── best_config.json              # Top configurations from sweep
│   ├── best_configs.log              # All 100 sweep configurations ranked
│   ├── overfit_configs.log           # Top 3 overfitting configurations
│   └── best_model.npy                # Best model weights
├── models/                           # Saved model checkpoints
├── requirements.txt                  # Python dependencies
├── README.md                         # This file
└── PROJECT_REPORT.md                 # Comprehensive project report with all experiments
```

---

## Features

### Core Neural Network Components
- **Activation Functions**: 
  - ReLU (Rectified Linear Unit) - Best for hidden layers
  - Sigmoid - Smooth, bounded output
  - Tanh - Zero-centered alternative to sigmoid
  - Linear - For output layer (returns logits)
  
- **Loss Functions**: 
  - Cross-Entropy - Optimal for classification tasks
  - Mean Squared Error (MSE) - Alternative loss function
  
- **Optimizers**: 
  - SGD (Stochastic Gradient Descent) - Simple baseline
  - Momentum - Accelerated convergence with velocity
  - NAG (Nesterov Accelerated Gradient) - Look-ahead momentum
  - RMSProp - Adaptive learning rates
  
- **Weight Initialization**: 
  - Xavier/Glorot - `sqrt(2/(n_in + n_out))` - Best for tanh/sigmoid
  - He - `sqrt(2/n_in)` - Optimized for ReLU
  - Random - Small random values `N(0, 0.01)`
  - Zeros - For debugging symmetry breaking

### Training Features
- **Mini-batch Gradient Descent** with configurable batch sizes
- **Automatic Data Splitting**: 90% train, 10% validation
- **Model Checkpointing**: Saves best model based on validation F1 score
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1 Score, Confusion Matrix
- **Weights & Biases Integration**: Full experiment tracking and visualization
- **L2 Regularization**: Weight decay for preventing overfitting

---

## Installation

### Requirements
- Python 3.7+
- NumPy
- scikit-learn (for data loading and metrics)
- matplotlib (for plotting)
- Weights & Biases (optional, for experiment tracking)

### Setup

```bash
# Clone the repository
git clone https://github.com/r1sha13h/DA6401_Assignment_1.git
cd DA6401_Assignment_1

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# (Optional) Login to Weights & Biases for experiment tracking
wandb login
```

---

## Usage

### Quick Start - Running Experiments

The project includes a comprehensive experiment suite covering all aspects of neural network training. To run specific experiments:

**Run All Experiments:**
```bash
# Run complete experiment suite (Experiments 2.1 through 2.10)
python src/experiment.py --experiment all --use_wandb --wandb_project mlp-100-sweeps
```

**Run Individual Experiments:**
```bash
# Experiment 2.1: Data Exploration
python src/experiment.py --experiment 2.1 --use_wandb --wandb_project mlp-100-sweeps

# Experiment 2.2: Hyperparameter Sweep (100 runs)
python src/experiment.py --experiment 2.2 --use_wandb --wandb_project mlp-100-sweeps --sweep_name mlp-100-sweeps

# Experiment 2.3: Optimizer Comparison
python src/experiment.py --experiment 2.3 --use_wandb --wandb_project mlp-100-sweeps

# Experiment 2.4: Vanishing Gradient Analysis
python src/experiment.py --experiment 2.4 --use_wandb --wandb_project mlp-100-sweeps

# Experiment 2.5: Dead Neuron Investigation
python src/experiment.py --experiment 2.5 --use_wandb --wandb_project mlp-100-sweeps

# Experiment 2.6: Loss Function Comparison
python src/experiment.py --experiment 2.6 --use_wandb --wandb_project mlp-100-sweeps

# Experiment 2.7: Overfitting Analysis
python src/experiment.py --experiment 2.7 --use_wandb --wandb_project mlp-100-sweeps --sweep_name mlp-100-sweeps

# Experiment 2.8: Error Analysis
python src/experiment.py --experiment 2.8 --use_wandb --wandb_project mlp-100-sweeps

# Experiment 2.9: Weight Initialization Impact
python src/experiment.py --experiment 2.9 --use_wandb --wandb_project mlp-100-sweeps

# Experiment 2.10: Transfer Learning (Fashion-MNIST)
python src/experiment.py --experiment 2.10 --use_wandb --wandb_project mlp-100-sweeps
```

**Note:** All experiment plots are automatically saved to `src/plots/Exp-X.X/` directories.

### Training a Model

**Basic Training:**
```bash
python src/train.py -d mnist -e 10 -b 64 -o sgd -lr 0.01
```

**Advanced Training with All Options:**
```bash
python src/train.py \
    --dataset fashion_mnist \
    --epochs 20 \
    --batch_size 128 \
    --loss cross_entropy \
    --optimizer rmsprop \
    --learning_rate 0.001 \
    --weight_decay 0.0001 \
    --num_layers 3 \
    --hidden_size 128 128 64 \
    --activation relu \
    --weight_init xavier \
    --wandb_project my-mlp-project \
    --use_wandb \
    --save_model \
    --model_save_path models/best_model.npy \
    --config_save_path models/best_config.json
```

### Command Line Arguments Reference

| Argument | Short | Type | Description | Choices | Default |
|----------|-------|------|-------------|---------|---------|
| `--dataset` | `-d` | str | Dataset to use | `mnist`, `fashion_mnist` | `mnist` |
| `--epochs` | `-e` | int | Number of training epochs | - | `10` |
| `--batch_size` | `-b` | int | Mini-batch size | - | `64` |
| `--loss` | `-l` | str | Loss function | `cross_entropy`, `mean_squared_error` | `cross_entropy` |
| `--optimizer` | `-o` | str | Optimization algorithm | `sgd`, `momentum`, `nag`, `rmsprop` | `sgd` |
| `--learning_rate` | `-lr` | float | Learning rate | - | `0.01` |
| `--weight_decay` | `-wd` | float | L2 regularization coefficient | - | `0.0` |
| `--num_layers` | `-nhl` | int | Number of hidden layers | - | `2` |
| `--hidden_size` | `-sz` | int+ | Neurons per hidden layer (space-separated) | - | `[128]` |
| `--activation` | `-a` | str | Activation function | `relu`, `sigmoid`, `tanh` | `relu` |
| `--weight_init` | `-w_i` | str | Weight initialization method | `random`, `xavier`, `he`, `zeros` | `xavier` |
| `--wandb_project` | `-w_p` | str | W&B project name | - | `mlp-numpy` |
| `--use_wandb` | - | flag | Enable W&B logging | - | `False` |
| `--save_model` | - | flag | Save best model | - | `False` |
| `--model_save_path` | - | str | Path to save model weights | - | `best_model.npy` |
| `--config_save_path` | - | str | Path to save config | - | `best_config.json` |

### Model Inference

**Using Saved Configuration:**
```bash
python src/inference.py \
    --model_path models/best_model.npy \
    --config models/best_config.json \
    --dataset mnist
```

**Specifying Architecture Manually:**
```bash
python src/inference.py \
    --model_path models/best_model.npy \
    --dataset mnist \
    --num_layers 3 \
    --hidden_size 128 128 64 \
    --activation relu \
    --loss cross_entropy
```

**Output Example:**
```
Loading mnist dataset...
Test samples: 10000

Initializing model...
Architecture: [784, 128, 128, 64, 10]
Activation: relu

Loading model from models/best_model.npy...
Model loaded successfully!

Evaluating on test set...

Test Results:
  Loss: 0.0892
  Accuracy: 0.9734
  F1 Score: 0.9733
  Precision: 0.9735
  Recall: 0.9734

Confusion Matrix:
[[ 972    0    1    0    0    1    3    1    2    0]
 [   0 1126    2    2    0    1    2    0    2    0]
 [   2    0 1018    2    1    0    0    7    2    0]
 ...
]
```

---

## Implementation Details

### Forward Propagation
1. Linear transformation: `Z = XW + b`
2. Apply activation function: `A = f(Z)` where `f` is one of {ReLU, Sigmoid, Tanh}
3. Output layer returns **logits** (no softmax applied)

### Backward Propagation
1. Compute loss gradient at output layer
2. Backpropagate through layers using chain rule
3. Store gradients: `grad_W[0]` = last layer, `grad_W[-1]` = first layer

### Optimization
- **SGD**: `W = W - lr * grad_W`
- **Momentum**: `v = Î²*v - lr*grad, W = W - v`
- **NAG**: Nesterov look-ahead momentum
- **RMSProp**: Adaptive learning rate with squared gradient cache

### Weight Initialization
- **Xavier**: `W ~ Uniform(-sqrt(6/(n_in + n_out)), sqrt(6/(n_in + n_out)))`
- **Random**: `W ~ N(0, 0.01)`
- **Zeros**: `W = 0` (for debugging symmetry breaking)

### Model Saving/Loading

### Saving
Models are automatically saved during training as `.npy` files containing weight dictionaries:
```python
weights = {
    'W0': W0, 'b0': b0,  # First layer
    'W1': W1, 'b1': b1,  # Second layer
    ...
}
```

### Loading
```python
import numpy as np
from ann.neural_network import NeuralNetwork

# Load weights
weights = np.load('best_model.npy', allow_pickle=True).item()

# Initialize model with same architecture
model = NeuralNetwork(args)
model.set_weights(weights)
```

---

## Implementation Details

### Architecture Design

The neural network follows a standard feedforward architecture:
```
Input (784) → Hidden Layer 1 → ... → Hidden Layer N → Output (10)
```

**Key Design Decisions:**
- Output layer uses **linear activation**, returns raw logits (not probabilities)
- Gradients stored in **input-to-output order** for optimizer compatibility
- Softmax computed **internally** in backward pass for numerical stability

### Forward Propagation

For each layer `l`:
1. **Linear Transformation**: `Z[l] = X[l] @ W[l] + b[l]`
2. **Activation**: `A[l] = activation(Z[l])`
3. **Cache**: Store `X[l]`, `Z[l]`, `A[l]` for backward pass

Output layer returns logits directly (no softmax).

### Backward Propagation

1. **Output Layer**: 
   - Compute softmax internally: `probs = softmax(logits)`
   - Cross-entropy: `dA = (probs - y_true) / batch_size`
   - MSE: `dA = 2 * (probs - y_true) / batch_size` (with softmax Jacobian)

2. **Hidden Layers**:
   - `dZ = dA * activation_derivative(Z)`
   - `dW = X.T @ dZ`
   - `db = sum(dZ, axis=0)`
   - `dX = dZ @ W.T`

### Optimization Algorithms

| Algorithm | Update Rule |
|-----------|-------------|
| **SGD** | `W = W - lr * grad_W` |
| **Momentum** | `v = γ*v + lr*grad; W = W - v` |
| **NAG** | `v = γ*v + lr*grad; W = W - (γ*v + lr*grad)` |
| **RMSProp** | `cache = β*cache + (1-β)*grad²; W = W - lr*grad/√(cache+ε)` |

### Weight Initialization

| Method | Formula | Best For |
|--------|---------|----------|
| Xavier | `U(-√(2/(n_in+n_out)), √(2/(n_in+n_out)))` | Tanh, Sigmoid |
| He | `N(0, √(2/n_in))` | ReLU |
| Random | `N(0, 0.01)` | Baseline |

### Model Persistence

**Saving** (1-indexed keys):
```python
weights = model.get_weights()  # {'W1': ..., 'b1': ..., 'W2': ..., 'b2': ...}
np.save('best_model.npy', weights)
```

**Loading** (auto-rebuilds architecture):
```python
weights = np.load('best_model.npy', allow_pickle=True).item()
model.set_weights(weights)
```

---

## Experiments and Results

### Hyperparameter Sweep Results (mlp-100-sweeps)

A comprehensive hyperparameter sweep was conducted with 100 runs using Bayesian optimization. The sweep explored:
- **Optimizers**: SGD, Momentum, NAG, RMSProp
- **Learning Rates**: 0.0001 to 0.1
- **Activations**: ReLU, Sigmoid, Tanh
- **Network Depths**: 1-3 hidden layers
- **Hidden Sizes**: 32, 64, 128, 256 neurons
- **Batch Sizes**: 16, 32, 64
- **Loss Functions**: Cross-Entropy, MSE
- **Weight Initializations**: Xavier, He, Random

**Best Configuration Achieved:**
- **Validation Accuracy**: 98.25%
- **Optimizer**: NAG
- **Learning Rate**: 0.1
- **Activation**: Sigmoid
- **Architecture**: 1 hidden layer with 128 neurons
- **Batch Size**: 16
- **Loss**: Cross-Entropy
- **Initialization**: Xavier

### Expected Performance

| Dataset | Accuracy | F1 Score | Training Time (10 epochs) |
|---------|----------|----------|---------------------------|
| MNIST | 97-98% | 0.97-0.98 | 2-5 min (CPU) |
| Fashion-MNIST | 87-90% | 0.87-0.90 | 2-5 min (CPU) |

### Recommended Configurations

**MNIST - High Accuracy:**
```bash
python src/train.py -d mnist -e 20 -b 64 -o rmsprop -lr 0.001 \
    -nhl 2 -sz 128 64 -a relu -w_i xavier -l cross_entropy
```

**Fashion-MNIST - Balanced:**
```bash
python src/train.py -d fashion_mnist -e 20 -b 128 -o rmsprop -lr 0.001 \
    -wd 0.0001 -nhl 3 -sz 256 128 64 -a relu -w_i xavier -l cross_entropy
```

### Hyperparameter Guidelines

**Learning Rates:**
| Optimizer | Range | Default |
|-----------|-------|---------|
| SGD | 0.01-0.1 | 0.01 |
| Momentum/NAG | 0.001-0.01 | 0.01 |
| RMSProp | 0.0001-0.001 | 0.001 |

**Architecture:**
- MNIST: 2-3 layers, 64-128 neurons
- Fashion-MNIST: 3-4 layers, 128-256 neurons

**Activations:**
- ReLU: Best default (prevents vanishing gradients)
- Tanh: Good for deep networks with Xavier init
- Sigmoid: Avoid (vanishing gradient problem)

### Weights & Biases Integration

**Enable W&B Logging:**
```bash
wandb login
python src/train.py -d mnist -e 20 -b 64 -o rmsprop -lr 0.001 \
    --use_wandb --wandb_project mlp-100-sweeps
```

**Tracked Metrics:**
- Training/Validation Loss & Accuracy
- F1, Precision, Recall
- Gradient Norms per Layer
- Weight Statistics
- Dead Neuron Count

**View Results:**
- Project Dashboard: [mlp-100-sweeps](https://wandb.ai/rishabh-mishra-cer16-iitmaana/mlp-100-sweeps)
- Full Report: [Assignment 1 Report](https://wandb.ai/rishabh-mishra-cer16-iitmaana/mlp-100-sweeps/reports/Assignment-1--VmlldzoxNjEzNTIyNg)

---

## Best Practices

### Training Tips

1. **Start Simple**: 2 layers, 128 neurons, ReLU, Xavier init
2. **Use RMSProp**: Faster convergence than SGD
3. **Cross-Entropy**: Better than MSE for classification
4. **Batch Size 64-128**: Good speed/stability balance
5. **Monitor Validation**: Watch for overfitting

### Troubleshooting

**Loss Not Decreasing:**
- Check learning rate (try 0.001-0.01)
- Verify data normalization ([0, 1])
- Print gradient norms
- Try RMSProp optimizer

**Loss is NaN:**
- Reduce learning rate
- Check softmax numerical stability
- Verify weight initialization
- Add gradient clipping

**Overfitting (train >> val):**
- Add L2 regularization (`--weight_decay 0.0001`)
- Reduce model size
- Implement early stopping

**Underfitting (low train & val):**
- Increase model capacity
- Train more epochs
- Reduce regularization

### Debugging Checklist

**1. Check Gradient Norms:**
```python
for i, layer in enumerate(model.layers):
    print(f"Layer {i}: ||grad_W|| = {np.linalg.norm(layer.grad_W):.6f}")
# < 1e-6 → vanishing, > 100 → exploding
```

**2. Check Weight Statistics:**
```python
for i, layer in enumerate(model.layers):
    print(f"Layer {i}: W mean={layer.W.mean():.4f}, std={layer.W.std():.4f}")
# All zeros → not learning, very large → diverging
```

**3. Check Activations:**
```python
for i, layer in enumerate(model.layers):
    print(f"Layer {i}: A min={layer.A.min():.4f}, max={layer.A.max():.4f}")
# All zeros (ReLU) → dead neurons, saturated → vanishing gradients
```

---

## Project Summary

### Implementation Highlights

**Pure NumPy**: No PyTorch/TensorFlow for core functionality  
**Modular Design**: Clean separation of layers, activations, losses, optimizers  
**Production Features**: Checkpointing, metrics, experiment tracking  
**Educational**: Well-documented with mathematical foundations  
**Extensible**: Easy to add new components  

### Performance Achievements

- **MNIST**: 97-98% accuracy in 10-20 epochs
- **Fashion-MNIST**: 87-90% accuracy in 20 epochs
- **Training Speed**: ~2-5 min per 10 epochs (CPU)
- **Code Quality**: 1,484 lines of clean Python

### Key Learnings

1. **Gradient Flow**: Manual backpropagation implementation
2. **Numerical Stability**: Proper softmax and loss computation
3. **Optimization**: SGD vs Momentum vs NAG vs RMSProp
4. **Initialization**: Xavier vs He vs Random
5. **Regularization**: L2 weight decay

---

## License

MIT License

---

## Acknowledgments

This project was developed as part of a Deep Learning course assignment, implementing fundamental neural network concepts from scratch for hands-on experience with forward/backward propagation, optimization algorithms, and model training.
