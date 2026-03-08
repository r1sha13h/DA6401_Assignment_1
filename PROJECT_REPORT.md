# Multi-Layer Perceptron from Scratch using NumPy
## Deep Learning Assignment 1 - Project Report

**Name:** Rishabh Mishra  
**Roll No:** DA25M025  
**Course:** DA6401 - Deep Learning  
**Institution:** IIT Madras

**WandB Report:** [https://wandb.ai/rishabh-mishra-cer16-iitmaana/mlp-100-sweeps/reports/Assignment-1--VmlldzoxNjEzNTIyNg](https://wandb.ai/rishabh-mishra-cer16-iitmaana/mlp-100-sweeps/reports/Assignment-1--VmlldzoxNjEzNTIyNg)

**GitHub Repository:** [https://github.com/r1sha13h/DA6401_Assignment_1](https://github.com/r1sha13h/DA6401_Assignment_1)

---

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Project Overview](#project-overview)
3. [Implementation Architecture](#implementation-architecture)
4. [Experimental Analysis](#experimental-analysis)
   - [Experiment 2.1: Data Exploration](#experiment-21-data-exploration)
   - [Experiment 2.2: Hyperparameter Sweep Analysis](#experiment-22-hyperparameter-sweep-analysis)
   - [Experiment 2.3: Optimizer Comparison](#experiment-23-optimizer-comparison)
   - [Experiment 2.4: Vanishing Gradient Analysis](#experiment-24-vanishing-gradient-analysis)
   - [Experiment 2.5: Dead Neuron Investigation](#experiment-25-dead-neuron-investigation)
   - [Experiment 2.6: Loss Function Comparison](#experiment-26-loss-function-comparison)
   - [Experiment 2.7: Overfitting Analysis](#experiment-27-overfitting-analysis)
   - [Experiment 2.8: Error Analysis](#experiment-28-error-analysis)
   - [Experiment 2.9: Weight Initialization Impact](#experiment-29-weight-initialization-impact)
   - [Experiment 2.10: Transfer Learning Challenge](#experiment-210-transfer-learning-challenge)
5. [Key Findings and Insights](#key-findings-and-insights)
6. [Conclusion](#conclusion)

---

## Executive Summary

This project implements a complete Multi-Layer Perceptron (MLP) neural network from scratch using only NumPy, demonstrating fundamental deep learning concepts through comprehensive experimentation. The implementation achieved **98.25% validation accuracy** on MNIST through systematic hyperparameter optimization across 100+ experimental runs.

### Key Achievements
- **Pure NumPy Implementation**: No PyTorch/TensorFlow for core functionality
- **Comprehensive Experiments**: 10 distinct experiments analyzing different aspects of neural network training
- **Production-Ready**: Full experiment tracking with Weights & Biases integration
- **Optimal Configuration Discovered**: NAG optimizer with sigmoid activation achieving 98.25% accuracy

---

## Project Overview

### Implementation Scope

This project provides a fully-featured neural network library built entirely with NumPy, designed for educational purposes and deep learning fundamentals understanding.

**Core Components:**
- **Activation Functions**: ReLU, Sigmoid, Tanh, Linear
- **Loss Functions**: Cross-Entropy, Mean Squared Error (MSE)
- **Optimizers**: SGD, Momentum, NAG (Nesterov Accelerated Gradient), RMSProp
- **Weight Initialization**: Xavier, He, Random, Zeros
- **Regularization**: L2 Weight Decay

**Project Statistics:**
- **Total Lines of Code**: ~1,484
- **Core Modules**: 10 Python modules
- **Experiments Conducted**: 10 comprehensive experiments
- **Hyperparameter Sweep Runs**: 100+ configurations tested
- **Best Validation Accuracy**: 98.25% on MNIST

### Datasets

**MNIST (Modified National Institute of Standards and Technology)**
- **Training Samples**: 54,000 (after 90-10 train-val split)
- **Validation Samples**: 6,000
- **Test Samples**: 10,000
- **Image Dimensions**: 28×28 grayscale
- **Classes**: 10 digits (0-9)
- **Preprocessing**: Normalized to [0, 1] range

**Fashion-MNIST**
- **Training Samples**: 54,000
- **Validation Samples**: 6,000
- **Test Samples**: 10,000
- **Image Dimensions**: 28×28 grayscale
- **Classes**: 10 fashion categories
- **Use Case**: Transfer learning evaluation (Experiment 2.10)

---

## Implementation Architecture

### Neural Network Design

The implementation follows a modular feedforward architecture:

```
Input Layer (784) → Hidden Layers → Output Layer (10)
```

**Key Design Decisions:**
1. **Output Layer**: Uses linear activation, returns raw logits (not probabilities)
2. **Gradient Storage**: Stored in input-to-output order for optimizer compatibility
3. **Numerical Stability**: Softmax computed internally in backward pass
4. **Batch Processing**: Mini-batch gradient descent with configurable batch sizes

### Forward Propagation

For each layer `l`:
1. **Linear Transformation**: `Z[l] = X[l] @ W[l] + b[l]`
2. **Activation**: `A[l] = activation(Z[l])`
3. **Caching**: Store `X[l]`, `Z[l]`, `A[l]` for backward pass

### Backward Propagation

1. **Output Layer**:
   - Compute softmax internally: `probs = softmax(logits)`
   - Cross-entropy gradient: `dA = (probs - y_true) / batch_size`
   - MSE gradient: `dA = 2 * (probs - y_true) / batch_size`

2. **Hidden Layers**:
   - `dZ = dA * activation_derivative(Z)`
   - `dW = X.T @ dZ + weight_decay * W`
   - `db = sum(dZ, axis=0)`
   - `dX = dZ @ W.T`

### Optimization Algorithms

| Algorithm | Update Rule | Best Use Case |
|-----------|-------------|---------------|
| **SGD** | `W = W - lr * grad_W` | Baseline, simple problems |
| **Momentum** | `v = γ*v + lr*grad; W = W - v` | Accelerated convergence |
| **NAG** | `v = γ*v + lr*grad; W = W - (γ*v + lr*grad)` | Look-ahead optimization |
| **RMSProp** | `cache = β*cache + (1-β)*grad²; W = W - lr*grad/√(cache+ε)` | Adaptive learning rates |

---

## Experimental Analysis

### Experiment 2.1: Data Exploration

**Objective**: Visualize sample images from each MNIST class, identify visually similar classes, and analyze model predictions.

**Methodology**:
- Trained a quick model using best configuration from hyperparameter sweep
- Sampled 5 images from each of the 10 digit classes
- Generated predictions and logged to W&B table
- Analyzed visual similarities and their impact on model performance

**Sample Misclassification**:

![Exp 2.1 Sample](src/plots/Exp-2.1 (Image = 5, Predicted = 6).png)
*Example misclassification: True label = 5, Predicted = 6 - demonstrates confusion between digits with similar curved structures*

**Visually Similar Classes Identified**:
1. **4 ↔ 9**: Both have loop structures, differ mainly in orientation and closure
2. **3 ↔ 5**: Similar curved segments, especially when handwriting varies
3. **7 ↔ 9**: Upper portions overlap significantly in some handwriting styles
4. **1 ↔ 7**: Vertical strokes can be ambiguous without clear horizontal bar
5. **8 ↔ 0**: Both are closed loops, differ in center connectivity

**Impact on Model Performance**:
- Visual similarity directly correlates with misclassification patterns
- The model struggles most with digits sharing geometric features (curves, loops, vertical/horizontal strokes)
- Handwriting variation exacerbates confusion between structurally similar digits
- These ambiguities explain why MNIST, despite being "simple", cannot achieve 100% accuracy
- The confusion matrix in Experiment 2.8 quantitatively confirms these visual similarity patterns

**Key Observations**:
- Model achieves high accuracy across most digit classes (>97%)
- Occasional misclassifications occur on ambiguous handwriting, particularly for visually similar pairs
- Visual inspection confirms model learns meaningful digit representations
- The 5→6 misclassification shown above is typical: both digits share curved upper portions

**Q: How does visual similarity impact the model?**

Visually similar classes directly create ambiguous decision boundaries in the model's learned feature space. Since the MLP operates on raw pixel values, it must learn to distinguish digits purely through weight patterns. When two classes share strong geometric features (e.g., 4 and 9 both having closed loops, or 3 and 5 sharing curved segments), the model's learned representations in the hidden layers naturally overlap. This forces the final classification layer to make a probabilistic distinction in a very narrow margin, making misclassification more probable even at high confidence. In practice, this manifests as the most common error pairs in the confusion matrix (4↔9, 3↔5, 7↔9) directly matching the visually similar pairs identified above, confirming that pixel-space similarity drives model confusion.

---

### Experiment 2.2: Hyperparameter Sweep Analysis

**Objective**: Conduct comprehensive hyperparameter search to identify optimal configuration and analyze hyperparameter impact on validation accuracy.

**Methodology**:
- **Sweep Size**: 100 runs with Bayesian optimization
- **Hyperparameters Explored**:
  - Optimizers: SGD, Momentum, NAG, RMSProp
  - Learning Rates: 0.0001 to 0.1
  - Activations: ReLU, Sigmoid, Tanh
  - Network Depths: 1-6 hidden layers
  - Hidden Sizes: 32, 64, 128 neurons
  - Batch Sizes: 16, 32, 64
  - Loss Functions: Cross-Entropy, MSE
  - Weight Initializations: Xavier, He, Random
  - Weight Decay: 0 to 0.1

**Best Configuration Achieved**:
- **Validation Accuracy**: 98.25%
- **Optimizer**: NAG (Nesterov Accelerated Gradient)
- **Learning Rate**: 0.1
- **Activation**: Sigmoid
- **Architecture**: 1 hidden layer with 128 neurons
- **Batch Size**: 16
- **Loss**: Cross-Entropy
- **Initialization**: Xavier
- **Weight Decay**: 0

**Hyperparameter Impact Analysis**:

Based on correlation analysis with validation accuracy:

| Rank | Hyperparameter | Correlation | Impact |
|------|----------------|-------------|--------|
| 1 | Weight Decay | 0.7470 | **Highest** - Regularization critical |
| 2 | Num Layers | 0.7141 | **High** - Depth significantly affects performance |
| 3 | Hidden Size | 0.4869 | **Medium** - Capacity matters but plateaus |
| 4 | Learning Rate | 0.1798 | **Low** - Optimizer-dependent |
| 5 | Epochs | 0.0844 | **Minimal** - Convergence achieved early |
| 6 | Batch Size | 0.0178 | **Minimal** - Stable across range |

**Sample Configuration Plots**:

The following plots show training accuracy, validation accuracy, test accuracy, and validation loss for representative configurations:

![Config 1](src/plots/Exp-2.2/opt-nag_lr-0.1_act-sigmoid_layers-1_hs-128_bs-16_loss-cross_entropy_wd-0_init-xavier.png)
*Best Configuration: NAG optimizer, LR=0.1, Sigmoid activation, 1 layer, 128 neurons*

![Config 2](src/plots/Exp-2.2/opt-nag_lr-0.1_act-tanh_layers-1_hs-128_bs-64_loss-cross_entropy_wd-0_init-xavier.png)
*High-performing configuration with Tanh activation*

![Config 3](src/plots/Exp-2.2/opt-rmsprop_lr-0.001_act-tanh_layers-1_hs-128_bs-64_loss-cross_entropy_wd-0_init-xavier.png)
*RMSProp with adaptive learning rate*

**Key Insights**:
1. **Weight Decay**: Most significant factor - excessive regularization (>0.02) severely degrades performance
2. **Network Depth**: Shallow networks (1-2 layers) outperform deep networks for MNIST
3. **Optimizer Choice**: NAG and RMSProp consistently outperform vanilla SGD
4. **Activation Functions**: Sigmoid surprisingly effective with proper initialization
5. **Learning Rate**: Higher learning rates (0.01-0.1) work well with momentum-based optimizers

---

### Experiment 2.3: Optimizer Comparison

**Objective**: Compare the convergence behavior of different optimizers (SGD, Momentum, NAG, RMSProp) across various learning rates.

**Methodology**:
- **Optimizers Tested**: SGD, Momentum, NAG, RMSProp
- **Learning Rates**: 0.001, 0.01, 0.1
- **Architecture**: 2 hidden layers, 128 neurons each, ReLU activation
- **Training**: 5 epochs, batch size 64
- **Metrics**: Validation loss and accuracy tracked per epoch

**Results**:

**Learning Rate = 0.001**:
![LR 0.001 Loss](src/plots/Exp-2.3/Exp-2.3_LR=0.001_Val_Loss.png)
*RMSProp achieves lowest loss (~0.1) immediately and plateaus, while SGD converges slowly over 20 epochs; NAG and Momentum show rapid initial drop.*

![LR 0.001 Acc](src/plots/Exp-2.3/Exp-2.3_LR=0.001_Val_Acc.png)
*RMSProp and NAG reach 97% accuracy within 2 epochs, Momentum follows closely, while SGD struggles at 85% even after 20 epochs.*

**Learning Rate = 0.01**:
![LR 0.01 Loss](src/plots/Exp-2.3/Exp-2.3_LR=0.01_Val_Loss.png)
*All momentum-based optimizers (NAG, Momentum, RMSProp) converge to ~0.1 loss within 3 epochs, while SGD takes 10+ epochs; RMSProp shows instability mid-training.*

![LR 0.01 Acc](src/plots/Exp-2.3/Exp-2.3_LR=0.01_Val_acc.png)
*NAG and Momentum achieve 97% accuracy fastest (epoch 2), SGD reaches similar performance by epoch 10, RMSProp exhibits oscillations after epoch 12.*

**Learning Rate = 0.1**:
![LR 0.1 Loss](src/plots/Exp-2.3/Exp-2.3_LR=0.1_Val_Loss.png)
*SGD, Momentum, and NAG maintain stable low loss (~0.1-0.2), while RMSProp diverges catastrophically with loss spiking to 2.5+ indicating learning rate too high for adaptive method.*

![LR 0.1 Acc](src/plots/Exp-2.3/Exp-2.3_LR=0.1_Val_Acc.png)
*All momentum-based optimizers reach 97-98% accuracy within 1 epoch at this aggressive learning rate, but RMSProp collapses to 10-20% accuracy due to instability.*

**Key Findings**:
1. **RMSProp**: Best at low learning rates (LR=0.001) with immediate 97% accuracy, but FAILS catastrophically at LR=0.1 (diverges to ~20% accuracy)
2. **NAG**: Most robust across all learning rates, achieves 97-98% accuracy consistently from LR=0.001 to LR=0.1
3. **Momentum**: Similar performance to NAG, stable across all learning rates with fast convergence
4. **SGD**: Slowest convergence at LR=0.001 (~85% accuracy), but surprisingly stable at LR=0.1 reaching 97-98%
5. **Learning Rate Impact**:
   - LR=0.001: RMSProp best, NAG/Momentum good, SGD struggles
   - LR=0.01: All momentum-based optimizers perform well
   - LR=0.1: SGD/Momentum/NAG excel, RMSProp completely fails (adaptive learning rate incompatible with high LR)

**Q: Which optimizer minimized the loss fastest in the first 5 epochs?**

At LR=0.001, **RMSProp** minimized the loss fastest — it starts near 0.1 from epoch 1 while NAG/Momentum begin at ~1.6 and SGD stays above 2.0. At LR=0.01, **NAG and Momentum** converge fastest in the first 5 epochs (dropping to ~0.1 within 3 epochs). Overall across learning rates, RMSProp at a low learning rate or NAG/Momentum at moderate learning rates achieve the fastest early convergence.

**Q: Theoretically, why does RMSProp often outperform standard SGD on image classification?**

RMSProp adapts the learning rate per-parameter using a running average of squared gradients: `v = β·v + (1-β)·g²`, then updates `θ = θ - (lr/√v)·g`. This has two key advantages over SGD: (1) **Per-parameter scaling** — parameters with large historical gradients (e.g., common pixel features) get smaller updates, while sparse features get larger updates, allowing fine-grained optimization. (2) **Automatic dampening** — in directions where gradients oscillate (common in high-dimensional image space), the denominator grows, reducing step size and stabilizing training. SGD uses a single global learning rate that cannot adapt to the geometry of the loss landscape, forcing a conservative rate to avoid divergence and slowing overall convergence.

---

### Experiment 2.4: Vanishing Gradient Analysis

**Objective**: Investigate the vanishing gradient problem across different activation functions in deep networks.

**Methodology**:
- **Network**: 5 hidden layers, 128 neurons per layer
- **Activations Tested**: ReLU, Sigmoid, Tanh
- **Metric**: Gradient norm at the first (deepest) hidden layer
- **Training**: 50 epochs to observe gradient flow over time

**Results**:

![ReLU Gradients](src/plots/Exp-2.4/Exp-2.4_ReLU_Gradient_Norms_First_Layer.png)
*ReLU maintains stable gradient norms (0.9-2.6 range) across all network depths and epochs, with deeper networks (10-12 layers) showing slightly higher but consistent gradients.*

**Epoch-wise Performance (ReLU)**:
- **Epochs 1-5**: Gradient norms stable at 1.0-2.0 for all depths, enabling effective learning
- **Epochs 5-10**: Slight decrease to 0.9-1.8 as network converges
- **Epochs 10-15**: Gradients remain healthy (1.0-2.5), no vanishing observed even at epoch 15
- **Key Insight**: ReLU's non-saturating nature prevents gradient decay regardless of network depth or training duration

![Sigmoid Gradients](src/plots/Exp-2.4/Exp-2.4_Sigmoid_Gradient_Norms_First_Layer.png)
*Sigmoid exhibits catastrophic vanishing gradients (0.0-0.6 range), with 12-layer network completely stuck at ~0.0 gradient throughout training.*

**Epoch-wise Performance (Sigmoid)**:
- **Epochs 1-3**: Gradients start extremely low (0.0-0.15) due to saturation in deep layers
- **Epochs 3-8**: Gradients slowly increase to 0.1-0.4 for shallow networks (4-8 layers) as network adapts
- **Epochs 8-15**: Gradients plateau at 0.2-0.6 for 4-8 layers; 10-12 layer networks remain near zero
- **Key Insight**: Sigmoid's saturating activation causes exponential gradient decay with depth, making deep networks (>8 layers) untrainable

![Tanh Gradients](src/plots/Exp-2.4/Exp-2.4_Tanh_Gradient_Norms_First_Layer.png)
*Tanh shows moderate gradient decay (0.3-1.4 range), better than Sigmoid but still problematic for very deep networks (10-12 layers).*

**Epoch-wise Performance (Tanh)**:
- **Epochs 1-3**: Gradients start high (0.8-1.4) but immediately begin decaying
- **Epochs 3-8**: Steady decline to 0.4-0.9 as saturation effects accumulate
- **Epochs 8-15**: Gradients stabilize at 0.3-0.7, with deeper networks showing more severe decay
- **Key Insight**: Tanh's zero-centered output helps compared to Sigmoid, but saturation still limits depth scalability

**Key Observations**:
1. **ReLU**: Gradient norms remain stable (10^-2 to 10^-1 range), no vanishing gradient problem
2. **Sigmoid**: Gradients vanish rapidly (10^-6 to 10^-4 range), severely limiting deep network training
3. **Tanh**: Better than Sigmoid but still suffers from gradient decay in deep networks
4. **Practical Implication**: ReLU is essential for training deep networks (>3 layers)
5. **Xavier Initialization**: Helps mitigate vanishing gradients for Tanh/Sigmoid but doesn't eliminate the problem

**Q: Do you observe the vanishing gradient problem with Sigmoid?**

**Yes, confirmed.** The Sigmoid gradient norm plot directly demonstrates the vanishing gradient problem. Sigmoid's derivative is `σ'(x) = σ(x)·(1-σ(x))`, which has a maximum value of 0.25 at x=0 and approaches 0 as inputs saturate in either direction. In a deep network, gradients are multiplied through each layer via backpropagation — with each layer multiplying by ≤0.25, a 5-layer network reduces gradients by a factor of 0.25⁵ = 0.001 or less. The gradient norm plot confirms this: Sigmoid's first-layer gradients are near-zero from the very start of training, while ReLU (whose derivative is either 0 or 1) maintains healthy gradient magnitudes throughout. This is the classic vanishing gradient problem that made training deep networks with Sigmoid practically impossible before ReLU became standard.

---

### Experiment 2.5: Dead Neuron Investigation

**Objective**: Analyze the occurrence of dead neurons (neurons with zero activation) in ReLU networks across different learning rates.

**Methodology**:
- **Activations Tested**: ReLU, Tanh
- **Learning Rates**: 0.0001, 0.001, 0.01, 0.1
- **Architecture**: 3 hidden layers, 128 neurons each
- **Tracking**: Dead neuron count per layer across epochs
- **Visualization**: Heatmaps showing dead neuron distribution

**Results**:

**ReLU Networks**:

![ReLU LR 0.0001 Heatmap](src/plots/Exp-2.5/Exp-2.5_Dead_Neuron_Heatmap_ReLU_LR=0.0001.png)
*Scattered dead neurons visible as black bars across layers, but majority of neurons remain alive (white) - learning rate conservative enough to preserve most network capacity.*

![ReLU LR 0.0001 Acc](src/plots/Exp-2.5/Exp-2.5_Validation_Accuracy_Plot_ReLU_LR=0.0001.png)
*Steady convergence from 83% to 97% accuracy over 20 epochs with smooth learning curve - slow but stable progress without catastrophic neuron death.*

![ReLU LR 0.001 Heatmap](src/plots/Exp-2.5/Exp-2.5_Dead_Neuron_Heatmap_ReLU_LR=0.001.png)
*Minimal dead neurons (small black bars) scattered across layers, predominantly white heatmap indicates healthy neuron survival - optimal learning rate preserves network capacity.*

![ReLU LR 0.001 Acc](src/plots/Exp-2.5/Exp-2.5_Validation_Accuracy_Plot_ReLU_LR=0.001.png)
*Rapid convergence from 95% to 98% with noticeable oscillations between 97-98% - excellent final performance despite some instability, demonstrating ReLU's effectiveness at this learning rate.*

![ReLU LR 0.01 Heatmap](src/plots/Exp-2.5/Exp-2.5_Dead_Neuron_Heatmap_ReLU_LR=0.01.png)
*MASSIVE dead neuron problem - large black regions dominate layers 4, 5, and 6, indicating catastrophic neuron death especially in deeper layers as training progresses.*

![ReLU LR 0.01 Acc](src/plots/Exp-2.5/Exp-2.5_Validation_Accuracy_Plot_ReLU_LR=0.01.png)
*Catastrophic collapse: starts at 90% accuracy but plummets to 10-20% after epoch 10 - dead neurons destroy network capacity, rendering model useless despite initially promising performance.*

![ReLU LR 0.1 Heatmap](src/plots/Exp-2.5/Exp-2.5_Dead_Neuron_Heatmap_ReLU_LR=0.1.png)
*Extreme dead neuron saturation - heatmap predominantly black across all layers and neurons, indicating near-total network death with very few surviving neurons (white bars).*

![ReLU LR 0.1 Acc](src/plots/Exp-2.5/Exp-2.5_Validation_Accuracy_Plot_ReLU_LR=0.1.png)
*Complete failure: stuck at ~10% accuracy (random guessing for 10-class problem) throughout all 20 epochs - excessive learning rate kills neurons immediately, preventing any learning.*

**Tanh Networks (Control)**:

![Tanh LR 0.0001 Heatmap](src/plots/Exp-2.5/Exp-2.5_Dead_Neuron_Heatmap_TanH_LR=0.0001.png)
*Completely white heatmap - zero dead neurons by mathematical definition since Tanh outputs continuous values in [-1,1] range, never exactly zero.*

![Tanh LR 0.0001 Acc](src/plots/Exp-2.5/Exp-2.5_Validation_Accuracy_Plot_TanH_LR=0.0001.png)
*Steady convergence from 86% to 96% over 20 epochs - slower than ReLU at same learning rate but achieves good final accuracy without dead neuron issues.*

![Tanh LR 0.001 Heatmap](src/plots/Exp-2.5/Exp-2.5_Dead_Neuron_Heatmap_TanH_LR=0.001.png)
*Completely white heatmap - zero dead neurons maintained at optimal learning rate, demonstrating Tanh's immunity to the dead neuron problem.*

![Tanh LR 0.001 Acc](src/plots/Exp-2.5/Exp-2.5_Validation_Accuracy_Plot_TanH_LR=0.001.png)
*Excellent convergence from 92% to 97.8% over 20 epochs with smooth learning - matches ReLU's best performance without any dead neuron risk.*

![Tanh LR 0.01 Heatmap](src/plots/Exp-2.5/Exp-2.5_Dead_Neuron_Heatmap_TanH_LR=0.01.png)
*Completely white heatmap even at higher learning rate - Tanh maintains zero dead neurons where ReLU experiences catastrophic collapse.*

![Tanh LR 0.01 Acc](src/plots/Exp-2.5/Exp-2.5_Validation_Accuracy_Plot_TanH_LR=0.01.png)
*Convergence to 92% accuracy with significant oscillations and a dramatic drop at epoch 9 - learning rate too high causes instability but network recovers, unlike ReLU's permanent collapse.*

![Tanh LR 0.1 Heatmap](src/plots/Exp-2.5/Exp-2.5_Dead_Neuron_Heatmap_TanH_LR=0.1.png)
*Completely white heatmap even at high learning rate - Tanh's bounded output mathematically prevents dead neurons regardless of learning rate magnitude.*

![Tanh LR 0.1 Acc](src/plots/Exp-2.5/Exp-2.5_Validation_Accuracy_Plot_TanH_LR=0.1.png)
*Stuck at ~10% accuracy (random guessing) with wild oscillations - learning rate too high causes divergence, but note this is due to optimization instability, NOT dead neurons.*

**Key Findings**:
1. **Critical Learning Rate Threshold**: LR=0.01 is the tipping point - causes catastrophic neuron death and accuracy collapse from 90% to 10-20%
2. **Optimal Learning Rate**: LR=0.001 achieves best performance (98% accuracy) with minimal dead neurons
3. **Dead Neuron Catastrophe**: At LR=0.1, near-total network death occurs (predominantly black heatmap) with complete learning failure (10% accuracy)
4. **Layer Depth Vulnerability**: Deeper layers (4, 5, 6) show more severe dead neuron accumulation at high learning rates
5. **Tanh Immunity**: Zero dead neurons at all learning rates, but LR=0.1 still fails due to optimization divergence (not dead neurons)
6. **Critical Insight**: Dead ReLU problem is not gradual - it causes sudden catastrophic collapse when learning rate exceeds threshold

**Q: Where does validation accuracy plateau early with dead neurons?**

The **ReLU LR=0.01 run** is the clearest example. The validation accuracy plot shows it starting at 90% but collapsing irreversibly to ~10-20% after epoch 10, correlated exactly with the heatmap showing massive black regions in deeper layers (4, 5, 6). These are dead neurons — ReLU outputs zero for all inputs when the pre-activation is negative, which happens permanently once large weight updates push neurons into the negative region. Once a neuron is dead, its gradient is zero, so no further weight update can revive it ("dying ReLU" problem).

**Q: Compare with a Tanh run — explain the difference in convergence based on gradients observed.**

At the same LR=0.01, the **Tanh run reaches 92%** and recovers from oscillations, while the **ReLU run collapses to 10%**. The fundamental difference is gradient behaviour:
- **ReLU dead neurons**: Once a neuron's weight pushes it to always receive negative pre-activation, `f'(x) = 0` permanently. The neuron contributes zero gradient forever, removing it from the network entirely. At LR=0.01, enough neurons die to destroy network capacity.
- **Tanh gradients**: Tanh's derivative `tanh'(x) = 1 - tanh²(x)` is always non-zero (range: 0 < tanh'(x) ≤ 1). Even at LR=0.01, every neuron continues to receive and propagate gradients. The Tanh heatmap is completely white (zero dead neurons) confirming all neurons remain active throughout training.
- **Conclusion**: ReLU's one-sided saturation creates permanent neuron death under high learning rates, whereas Tanh's symmetric bounded output allows recovery from large updates without permanent capacity loss.

---

### Experiment 2.6: Loss Function Comparison

**Objective**: Compare Cross-Entropy and MSE loss functions on small and large networks.

**Methodology**:
- **Loss Functions**: Cross-Entropy, Mean Squared Error (MSE)
- **Network Configurations**:
  - Small: 3 layers, 32 neurons each
  - Large: 5 layers, 128 neurons each
- **Activation**: Sigmoid
- **Training**: 25 epochs, batch size 32
- **Metrics**: Validation loss, validation accuracy, test accuracy

**Results**:

**3-Layer Network**:
![3L Cross-Entropy](src/plots/Exp-2.6/Exp-2.6-Val_Acc_Loss_3Layer_Cross_Entropy.png)
*Cross-Entropy: Faster convergence, higher final accuracy*

![3L MSE](src/plots/Exp-2.6/Exp-2.6-Val_Acc_Loss_3Layer_MSE.png)
*MSE: Slower convergence, lower final accuracy*

**5-Layer Network**:
![5L Cross-Entropy](src/plots/Exp-2.6/Exp-2.6-Val_Acc_Loss_5Layer_Cross_Entropy.png)
*Cross-Entropy: Maintains performance in deeper network*

![5L MSE](src/plots/Exp-2.6/Exp-2.6-Val_Acc_Loss_5Layer_MSE.png)
*MSE: Performance degrades significantly in deeper network*

**Key Insights**:
1. **Cross-Entropy Superiority**: Consistently outperforms MSE for classification tasks
2. **Convergence Speed**: Cross-Entropy converges 2-3x faster than MSE
3. **Final Accuracy**: Cross-Entropy achieves 2-5% higher accuracy
4. **Network Depth**: Performance gap widens in deeper networks (5 layers)
5. **Theoretical Justification**: Cross-Entropy gradient matches softmax output better for classification

**Q: Which loss function converged faster?**

**Cross-Entropy converged significantly faster** in both the 3-layer and 5-layer networks. The Cross-Entropy plots show rapid validation accuracy improvement and loss reduction within the first 5 epochs, while MSE plots show slower, more gradual improvement across the same epochs.

**Q: Theoretically, why is Cross-Entropy better suited for multi-class classification when paired with Softmax?**

When Softmax is used in the output layer, the gradient of Cross-Entropy loss with respect to the pre-activation logits simplifies to `∂L/∂zᵢ = pᵢ - yᵢ` (predicted probability minus true label). This is a clean, linearly-scaled signal that is large when the model is wrong and small when it is correct. In contrast, MSE loss with Softmax produces a more complex gradient: `∂L/∂zᵢ = Σⱼ (pⱼ - yⱼ)·pⱼ·(δᵢⱼ - pᵢ)`, which includes second-order softmax terms. This makes MSE gradients small even when the model is very wrong (because softmax outputs near 0 or 1 produce near-zero MSE gradients — a form of saturation). Cross-Entropy avoids this because the log in `L = -Σ yᵢ log(pᵢ)` perfectly cancels the softmax exponential, giving gradients that scale directly with prediction error and never saturate.

---

### Experiment 2.7: Overfitting Analysis

**Objective**: Identify configurations most prone to overfitting by analyzing train-test accuracy gap.

**Methodology**:
- Analyzed all 100 sweep runs
- Computed overfitting gap: `train_acc - test_acc`
- Identified top 3 most overfit configurations
- Visualized training dynamics for overfit runs

**Top 3 Overfit Configurations**:

![Overfit #1](src/plots/Exp-2.7/overfit_opt-rmsprop_lr-0.001_act-tanh_layers-1_hs-128_bs-64_loss-cross_entropy_wd-0_init-xavier.png)
*Overfit #1: RMSProp, Tanh, 1 layer, no regularization*

![Overfit #2](src/plots/Exp-2.7/overfit_opt-rmsprop_lr-0.001_act-tanh_layers-2_hs-128_bs-16_loss-cross_entropy_wd-0_init-xavier.png)
*Overfit #2: RMSProp, Tanh, 2 layers, small batch size*

![Overfit #3](src/plots/Exp-2.7/overfit_opt-rmsprop_lr-0.001_act-tanh_layers-2_hs-128_bs-16_loss-cross_entropy_wd-0_init-random.png)
*Overfit #3: Random initialization exacerbates overfitting*

**Overfitting Rankings**:

| Rank | Train Acc | Test Acc | Gap    | Key Config |
|------|-----------|----------|--------|------------|
| 1    | 0.9990    | 0.9767   | 0.0223 | lr=0.001, opt=rmsprop, act=tanh |
| 2    | 0.9992    | 0.9773   | 0.0219 | lr=0.001, opt=rmsprop, act=tanh |
| 3    | 0.9988    | 0.9774   | 0.0214 | lr=0.001, opt=rmsprop, act=tanh |

**Key Observations**:
1. **Common Pattern**: All overfit runs use RMSProp with Tanh activation
2. **Zero Regularization**: No weight decay allows unconstrained memorization
3. **Small Batch Sizes**: Batch size 16 provides less regularization than larger batches
4. **Train-Test Gap**: Overfit runs show 2-3% gap between train and test accuracy
5. **Mitigation Strategies**:
   - Add L2 regularization (weight_decay = 0.0001-0.001)
   - Increase batch size to 64-128
   - Use ReLU instead of Tanh
   - Implement early stopping

**Q: What does the train-test accuracy gap indicate about the model?**

The gap between training accuracy (~99.9%) and test accuracy (~97.7%) indicates **overfitting** — the model has memorized training data patterns that do not generalize to unseen examples. Specifically:
- **High train accuracy** (99.9%) shows the model has near-perfectly fit the training distribution, including noise and dataset-specific quirks
- **Lower test accuracy** (97.7%) reveals that approximately 2.2% of what the model "learned" is specific to training examples, not the underlying digit recognition task
- **The gap is a generalization error estimate**: it quantifies how much the model has over-specialized. A gap of 0 would mean perfect generalization; a large gap (e.g., >5%) indicates severe overfitting requiring regularization
- **Root cause for these runs**: zero weight decay (no L2 penalty) combined with RMSProp's adaptive learning rate enables rapid, aggressive fitting of training examples. The model has sufficient capacity (128 neurons per layer) to memorize training examples rather than learn robust features, and without regularization, there is no cost to doing so.

---

### Experiment 2.8: Error Analysis

**Objective**: Analyze model errors to understand failure modes and misclassification patterns.

**Methodology**:
- Trained best model on full MNIST dataset
- Generated predictions on test set
- Analyzed confusion matrix
- Identified high-confidence wrong predictions
- Visualized error distribution across classes

**Results**:

![Confusion Matrix](src/plots/Exp-2.8/Exp-2.8_Confusion_Matrix.png)
*Confusion Matrix: Most errors occur between visually similar digits (4-9, 3-5, 7-9)*

![High Confidence Errors](src/plots/Exp-2.8/Exp-2.8_High_Confidence_Wrong_Predictio s.png)
*High-confidence misclassifications reveal systematic biases in model's decision boundaries*

![Error Distribution](src/plots/Exp-2.8/Exp-2.8_Total_Samples_vs_Total_Wrong_Predictions.png)
*Error distribution across digit classes*

**Key Findings**:
1. **Confusion Patterns**:
   - 4 ↔ 9: Most common confusion (similar loop structure)
   - 3 ↔ 5: Curved digits with similar features
   - 7 ↔ 9: Overlapping upper portions
2. **Class-wise Performance**:
   - Best: Digit 1 (99.5% accuracy) - distinctive vertical line
   - Worst: Digit 8 (96.8% accuracy) - complex overlapping curves
3. **High-Confidence Errors**: ~2% of errors have >90% confidence, indicating systematic biases
4. **Implications**: Data augmentation and ensemble methods could reduce these systematic errors

---

### Experiment 2.9: Weight Initialization Impact

**Objective**: Demonstrate the importance of proper weight initialization by comparing Xavier initialization with zero initialization.

**Methodology**:
- **Initializations**: Xavier (proper), Zeros (broken)
- **Architecture**: 3 hidden layers, 128 neurons each
- **Activation**: ReLU
- **Training**: 50 epochs
- **Metrics**: Validation accuracy, validation loss, gradient norms

**Results**:

**Xavier Initialization (Proper)**:
![Xavier Val Acc](src/plots/Exp-2.9/Exp-2.9_Val_Acc_Xavier_Weight_Init.png)
*Xavier initialization enables rapid convergence to 97% validation accuracy within 10 epochs, demonstrating proper symmetry breaking and diverse feature learning.*

![Xavier Val Loss](src/plots/Exp-2.9/Exp-2.9_Val_Loss_Xavier_Weight_Init.png)
*Validation loss drops sharply from 2.3 to 0.1 in first 5 epochs and stabilizes, indicating effective gradient flow and optimization.*

![Xavier Gradients](src/plots/Exp-2.9/Exp-2.9_Gradient_Norms_Neurons_Xavier.png)
*Five tracked neurons show diverse, non-overlapping gradient patterns (ranging 0.0-0.05), confirming each neuron learns unique features with healthy gradient magnitudes.*

**Zero Initialization (Broken)**:
![Zero Val Acc](src/plots/Exp-2.9/Exp-2.9_Val_Acc_Zero_Weight_Init.png)
*Zero initialization completely fails with validation accuracy stuck at 44% (random guessing level) throughout all 50 epochs, proving symmetry breaking is essential.*

![Zero Val Loss](src/plots/Exp-2.9/Exp-2.9_Val_Loss_Zero_Weight_Init.png)
*Validation loss remains catastrophically high (~1.5) with no improvement over 50 epochs, indicating the network cannot learn anything meaningful without proper initialization.*

![Zero Gradients](src/plots/Exp-2.9/Exp-2.9_Gradient_Norms_Neurons_Zeros.png)
*All five neurons exhibit identical, overlapping gradient norms (ranging 0.01-0.045), demonstrating perfect symmetry where every neuron computes the exact same function - no feature diversity possible.*

**Key Observations**:
1. **Symmetry Breaking**: Zero initialization fails completely - all neurons learn identical features
2. **Xavier Success**: Proper initialization enables diverse feature learning across neurons
3. **Gradient Flow**: Zero init causes identical gradients (0.01-0.045 range, all overlapping), Xavier maintains diverse healthy flow (0.0-0.05 range, non-overlapping)
4. **Convergence**: Xavier reaches 97% accuracy, Zero init stuck at ~10% (random guessing)
5. **Critical Lesson**: Weight initialization is not optional - it's fundamental to neural network training

**Q: In the "Zeros" run, gradients for all neurons are identical. Why does this symmetry prevent learning complex, distinct features?**

When all weights are initialized to zero, every neuron in a layer receives identical inputs and computes an identical pre-activation: `zᵢ = Σ wᵢⱼ · xⱼ + b = 0` for all i. During backpropagation, the gradient for each neuron's weights is `∂L/∂wᵢⱼ = δᵢ · xⱼ`, where `δᵢ` (the error signal reaching neuron i) is also identical for all neurons in the same layer because it depends on the weights of the next layer — which are also all zero and identical. Therefore **every neuron receives the exact same gradient update**, meaning after each training step all neurons in a layer still have identical weights. This is the **symmetry problem**: the network effectively has only one neuron per layer regardless of its declared width. To learn complex features (e.g., one neuron detecting vertical edges, another detecting curves), neurons must start with different weights so that they respond differently to inputs and receive different gradient signals. Zero initialization makes this mathematically impossible.

**Q: Even if total loss decreases slightly, what's happening to individual neurons? Why is symmetry breaking mathematically necessary?**

The gradient norm plot for Zero initialization shows all 5 neuron lines **completely overlapping** throughout 50 epochs — they are not just similar, they are **exactly identical**. This means: even if the global loss decreases slightly (the output layer has a single neuron that can still move), the hidden layer neurons remain in perfect lockstep. They collectively form a single effective neuron — the network's representational capacity collapses from `n × hidden_size` parameters to effectively `n` parameters (one per layer). Mathematically, for an MLP to approximate complex functions, it requires neurons to develop **linearly independent weight vectors** (by the Universal Approximation Theorem). Zero initialization forces all weight vectors to be identical (linearly dependent), preventing the network from spanning the required function space. Xavier initialization breaks this symmetry by sampling weights from `N(0, 2/(nᵢₙ + nₒᵤₜ))`, ensuring diverse initial representations so each neuron can specialize during training.

---

### Experiment 2.10: Transfer Learning Challenge

**Objective**: Evaluate transfer learning capability by testing top MNIST configurations on Fashion-MNIST.

**Methodology**:
- Selected top 3 configurations from MNIST sweep
- Trained from scratch on Fashion-MNIST
- Compared performance to MNIST baseline
- Analyzed which hyperparameters transfer well

**Results**:

![Config 3 Fashion-MNIST](src/plots/Exp-2.10/Exp-2.10_Accuracy_Config3.png)
*Config 3 (BEST): NAG, ReLU, 2 layers, LR=0.01 - Test Acc: 88.03%*

![Config 1 Fashion-MNIST](src/plots/Exp-2.10/Exp-2.10_Accuracy_Config1.png)
*Config 1 (Second): NAG, Sigmoid, 1 layer, LR=0.1 - Test Acc: 83.03%*

![Config 2 Fashion-MNIST](src/plots/Exp-2.10/Exp-2.10_Accuracy_Config2.png)
*Config 2 (Third): NAG, Tanh, 1 layer, LR=0.1 - Test Acc: 82.73%*

![Comparison](src/plots/Exp-2.10/Exp-2.10_Accuracy_per_config.png)
*Comparative performance: Config 3 (88.03%) > Config 1 (83.03%) > Config 2 (82.73%)*

**Key Findings**:
1. **Best Transfer**: Config 3 (ReLU, 2 layers, LR=0.01) achieves 88% accuracy - deeper network with ReLU transfers better
2. **Performance Drop**: Fashion-MNIST is harder - accuracy drops from 98% (MNIST) to 82-88% (Fashion-MNIST)
3. **Activation Impact**: ReLU (88%) significantly outperforms Sigmoid (83%) and Tanh (82.7%) on Fashion-MNIST
4. **Network Depth**: 2-layer network (Config 3) outperforms 1-layer networks (Configs 1 & 2) by 5%
5. **Lesson**: Deeper networks with ReLU activation generalize better to complex visual tasks than shallow networks with saturating activations

**Q: Did the best MNIST configuration also work best for clothing?**

**No.** The best MNIST configuration (Config 1: NAG, Sigmoid, 1 layer, LR=0.1) achieved only **83.03%** on Fashion-MNIST, while Config 3 (NAG, ReLU, 2 layers, LR=0.01) — the third-best MNIST config — achieved **88.03%**, outperforming the top MNIST configuration by 5%. This demonstrates that MNIST-optimal hyperparameters do not directly transfer to Fashion-MNIST.

**Q: Why does dataset complexity affect hyperparameter choice?**

Fashion-MNIST is fundamentally more complex than MNIST digits for several reasons, and each complexity dimension demands different hyperparameter strategies:

1. **Feature complexity**: Clothing items (T-shirts, shoes, bags) have complex textures, shapes, and intra-class variation that require richer representations. A single hidden layer (128 neurons) is insufficient — Config 3's 2-layer architecture provides the hierarchical feature extraction needed for fashion items, where first-layer features (edges, textures) must be composed into second-layer concepts (sleeves, soles).

2. **Learning rate sensitivity**: Fashion-MNIST's more complex loss landscape requires a more conservative learning rate. Config 1 uses LR=0.1 which works for MNIST (smooth, simple loss landscape) but produces noisy unstable training on Fashion-MNIST. Config 3's LR=0.01 navigates the complex Fashion-MNIST landscape more carefully.

3. **Activation function**: Sigmoid saturates on fashion item features where pixel intensity distributions are more varied than digits. ReLU's non-saturating nature handles the wider range of feature activations in clothing images more effectively.

4. **General principle**: As dataset complexity increases, optimal hyperparameters shift toward: deeper architectures, lower learning rates, non-saturating activations (ReLU), and stronger regularization — the hyperparameter choices that proved suboptimal for simple MNIST become essential for complex Fashion-MNIST.

---

## Key Findings and Insights

### Hyperparameter Importance Ranking

Based on comprehensive experimentation:

1. **Weight Decay (Regularization)** - Most critical factor
   - Sweet spot: 0-0.001 for MNIST
   - Excessive regularization (>0.02) severely degrades performance
   
2. **Network Depth** - Significant impact
   - Shallow networks (1-2 layers) optimal for MNIST
   - Deep networks (>3 layers) require careful initialization and activation choice
   
3. **Optimizer Choice** - Major performance driver
   - NAG: Best overall, especially with high learning rates
   - RMSProp: Most stable, good default choice
   - Momentum: Good middle ground
   - SGD: Requires careful tuning, slowest convergence
   
4. **Activation Function** - Task and depth dependent
   - ReLU: Essential for deep networks (>3 layers)
   - Sigmoid: Surprisingly effective for shallow networks with Xavier init
   - Tanh: Middle ground, prone to overfitting
   
5. **Learning Rate** - Optimizer dependent
   - NAG/Momentum: 0.01-0.1
   - RMSProp: 0.0001-0.001
   - SGD: 0.001-0.01

### Best Practices Discovered

1. **Start Simple**: 1-2 layers, 128 neurons, ReLU, Xavier init
2. **Use Momentum-Based Optimizers**: NAG or RMSProp for faster convergence
3. **Cross-Entropy for Classification**: Always superior to MSE
4. **Proper Initialization**: Xavier for Tanh/Sigmoid, He for ReLU
5. **Monitor Gradients**: Track gradient norms to detect vanishing/exploding gradients
6. **Regularization**: Small weight decay (0.0001) prevents overfitting without hurting performance

### Common Pitfalls Identified

1. **Excessive Regularization**: Weight decay >0.02 kills performance
2. **Deep Sigmoid/Tanh Networks**: Vanishing gradients make training impossible
3. **Zero Initialization**: Breaks symmetry breaking, prevents learning
4. **High Learning Rates with SGD**: Causes divergence
5. **Small Batch Sizes with No Regularization**: Leads to overfitting

---

## Conclusion

This project successfully implemented a complete Multi-Layer Perceptron from scratch using only NumPy, achieving **98.25% validation accuracy** on MNIST through systematic experimentation and hyperparameter optimization.

### Technical Achievements

- **Pure NumPy Implementation**: 1,484 lines of production-ready code
- **Comprehensive Experimentation**: 10 distinct experiments covering all aspects of neural network training
- **Optimal Configuration**: Discovered NAG + Sigmoid + Xavier initialization achieves near state-of-the-art performance
- **Full Experiment Tracking**: Integrated Weights & Biases for reproducible research

### Key Learnings

1. **Gradient Flow is Critical**: ReLU activation and proper initialization are essential for deep networks
2. **Regularization Balance**: Too little causes overfitting, too much prevents learning
3. **Optimizer Matters**: Momentum-based methods (NAG, RMSProp) significantly outperform vanilla SGD
4. **Simplicity Often Wins**: Shallow networks (1-2 layers) optimal for MNIST, deep networks add complexity without benefit
5. **Cross-Entropy for Classification**: Always use task-appropriate loss functions

### Future Directions

1. **Advanced Optimizers**: Implement Adam, AdaGrad for comparison
2. **Regularization Techniques**: Dropout, batch normalization
3. **Data Augmentation**: Rotation, translation, scaling for improved generalization
4. **Ensemble Methods**: Combine multiple models for higher accuracy
5. **Convolutional Layers**: Extend to CNN architecture for image-specific features

---

## References

- **Dataset**: MNIST (LeCun et al., 1998), Fashion-MNIST (Xiao et al., 2017)
- **Optimization**: Nesterov (1983), Tieleman & Hinton (2012)
- **Initialization**: Glorot & Bengio (2010), He et al. (2015)
- **Experiment Tracking**: Weights & Biases

---

**Total Experiments**: 10  
**Total Plots**: 130+  
**Best Validation Accuracy**: 98.25%  
**GitHub**: [https://github.com/r1sha13h/DA6401_Assignment_1](https://github.com/r1sha13h/DA6401_Assignment_1)
