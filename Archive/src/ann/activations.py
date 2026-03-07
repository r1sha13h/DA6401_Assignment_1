"""
Activation Functions and Their Derivatives
Implements: ReLU, Sigmoid, Tanh, Softmax and their derivatives
"""
import numpy as np

def relu(x):
    """ReLU activation function"""
    return np.maximum(0, x)

def relu_derivative(x):
    """Derivative of ReLU"""
    return (x > 0).astype(float)

def sigmoid(x):
    """Sigmoid activation function with numerical stability"""
    x = np.clip(x, -500, 500)  # Prevent overflow
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """Derivative of sigmoid"""
    s = sigmoid(x)
    return s * (1 - s)

def tanh(x):
    """Tanh activation function"""
    return np.tanh(x)

def tanh_derivative(x):
    """Derivative of tanh"""
    t = np.tanh(x)
    return 1 - t**2

def softmax(x):
    """Softmax activation function with numerical stability"""
    # Subtract max for numerical stability
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def get_activation(name):
    """Get activation function by name"""
    activations = {
        'relu': relu,
        'sigmoid': sigmoid,
        'tanh': tanh,
        'softmax': softmax
    }
    if name not in activations:
        raise ValueError(f"Unknown activation: {name}")
    return activations[name]

def get_activation_derivative(name):
    """Get activation derivative by name"""
    derivatives = {
        'relu': relu_derivative,
        'sigmoid': sigmoid_derivative,
        'tanh': tanh_derivative
    }
    if name not in derivatives:
        raise ValueError(f"Unknown activation derivative: {name}")
    return derivatives[name]