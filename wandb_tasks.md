# Weights & Biases Report

You must submit a **public W&B report link**. Your report must contain experimental evidence and written answers to the following questions.

**All plots should be logged in Weights & Biases.**

## 2.1 Data Exploration and Class Distribution

Log a W&B Table containing **5 sample images from each of the 10 classes** in the dataset. 

- Identify any classes that look visually similar in their raw form
- Explain how this visual similarity might impact your model

## 2.3 The Optimizer Showdown

Compare the **convergence rates of all 4 optimizers** using the same architecture:
- **3 hidden layers, 128 neurons each, ReLU activation**

**Questions:**
- Which optimizer minimized the loss fastest in the first 5 epochs?
- Theoretically, why does RMSProp often outperform standard SGD on image classification?

## 2.4 Vanishing Gradient Analysis

**Fix the optimizer to RMSProp** and compare **Sigmoid vs ReLU** for different network configurations.

**Requirements:**
- Log the **gradient norms for the first hidden layer**
- Do you observe the **vanishing gradient problem with Sigmoid**?
- **Provide a plot** to support your observation

## 2.5 The "Dead Neuron" Investigation

Using **ReLU activation** and a **high learning rate (e.g., 0.1)**:

1. Monitor the **activations of your hidden layers**
2. Find a run where **validation accuracy plateaus early** and analyze the **distribution of your activations** and identify **"dead neurons"** (neurons that output zero for all inputs)
3. **Compare this run with a Tanh run** and explain the **difference in convergence** based on the gradients observed.

## 2.6 Loss Function Comparison

Compare training curves of two models:
- **Mean Squared Error (MSE)**
- **Cross-Entropy**

**Use identical architecture and learning rate for both.**

**Questions:**
- Which loss function converged faster?
- **Theoretically**, why is Cross-Entropy better suited for multi-class classification when paired with Softmax output?

## 2.8 Error Analysis

For your **best-performing model** on the test set:

1. **Plot a Confusion Matrix**
2. **Creative visualization**: Beyond the standard matrix, provide a visualization of your model's failures
3. Create plots using **matplotlib** and **log them into W&B**

## 2.9 Weight Initialization & Symmetry

Compare two training runs:

### 1. Zeros Initialization
- All weights and biases set to **0**

### 2. Xavier Initialization
- Weights sampled from distribution with specific variance

**In your W&B report:**
- Create a **line plot** showing gradients of **5 different neurons** within the **same hidden layer**
- Track over the **first 50 training iterations**

**Answer these questions:**

1. In the "Zeros" run, gradients for all neurons within a layer are **identical** (lines overlap perfectly). Why does this **symmetry** prevent learning complex, distinct features?

2. Even if total loss decreases slightly, what's happening to **individual neurons**? Use gradient plots to explain why **symmetry breaking** is mathematically necessary for MLPs.

## 2.10 The Fashion-MNIST Transfer Challenge

**Note**: All previous experiments used MNIST digits. Now use **Fashion-MNIST**.

**Scenario**: Limited computation budget - **only 3 hyperparameter configurations** allowed.

**Task:**
- Based **strictly on MNIST learnings**, select **3 specific configurations**:
  - Architecture + Optimizer + Activation
- Report the **accuracies obtained**
- Did the **best MNIST configuration** also work best for clothing?
- Justify why **dataset complexity** affects hyperparameter choice

## 2.2 Hyperparameter Sweep

- Perform **W&B Sweep with ≥100 runs**
- Vary hyperparameters systematically
- Use **Parallel Coordinates plot** to analyze
- Identify **most significant hyperparameter** impacting validation accuracy
- Report your **best-performing configuration**

## 2.7 Global Performance Analysis

Create an **overlay plot** showing:
- **Training vs Test Accuracy** across **all hyperparameter search runs**

**Analysis:**
- Identify runs with **high training accuracy but poor test accuracy**
- What does this **gap indicate** about the model?
