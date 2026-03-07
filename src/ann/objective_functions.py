"""
Loss/Objective Functions and Their Derivatives
Implements: Cross-Entropy, Mean Squared Error (MSE)
"""
import numpy as np
from .activations import softmax

def cross_entropy_loss(y_true, y_pred_logits):
    """
    Cross-entropy loss with softmax.

    Args:
        y_true: One-hot encoded labels of shape (batch_size, num_classes)
        y_pred_logits: Raw logits of shape (batch_size, num_classes)

    Returns:
        loss: Scalar loss value
    """
    batch_size = y_true.shape[0]

    # Apply softmax to get probabilities
    y_pred_probs = softmax(y_pred_logits)

    # Clip probabilities to avoid log(0)
    y_pred_probs = np.clip(y_pred_probs, 1e-10, 1 - 1e-10)

    # Compute cross-entropy
    loss = -np.sum(y_true * np.log(y_pred_probs)) / batch_size

    return loss

def cross_entropy_derivative(y_true, y_pred_logits):
    """
    Derivative of cross-entropy loss with softmax.
    Combined derivative: softmax + cross-entropy = (y_pred - y_true)

    Args:
        y_true: One-hot encoded labels of shape (batch_size, num_classes)
        y_pred_logits: Raw logits of shape (batch_size, num_classes)

    Returns:
        gradient: Gradient of shape (batch_size, num_classes)
    """
    batch_size = y_true.shape[0]
    y_pred_probs = softmax(y_pred_logits)

    # Simplified gradient for softmax + cross-entropy (averaged over batch)
    gradient = (y_pred_probs - y_true) / batch_size

    return gradient

def mean_squared_error(y_true, y_pred):
    """
    Mean Squared Error loss.

    Args:
        y_true: True labels of shape (batch_size, num_classes)
        y_pred: Predicted values of shape (batch_size, num_classes)

    Returns:
        loss: Scalar loss value
    """
    batch_size = y_true.shape[0]
    loss = np.sum((y_pred - y_true)**2) / (2 * batch_size)
    return loss

def mean_squared_error_derivative(y_true, y_pred):
    """
    Derivative of Mean Squared Error.

    Args:
        y_true: True labels of shape (batch_size, num_classes)
        y_pred: Predicted values of shape (batch_size, num_classes)

    Returns:
        gradient: Gradient of shape (batch_size, num_classes)
    """
    batch_size = y_true.shape[0]
    return (y_pred - y_true) / batch_size

def get_loss_function(name):
    """Get loss function by name"""
    losses = {
        'cross_entropy': cross_entropy_loss,
        'mean_squared_error': mean_squared_error,
        'mse': mean_squared_error
    }
    if name not in losses:
        raise ValueError(f"Unknown loss function: {name}")
    return losses[name]

def get_loss_derivative(name):
    """Get loss derivative by name"""
    derivatives = {
        'cross_entropy': cross_entropy_derivative,
        'mean_squared_error': mean_squared_error_derivative,
        'mse': mean_squared_error_derivative
    }
    if name not in derivatives:
        raise ValueError(f"Unknown loss derivative: {name}")
    return derivatives[name]