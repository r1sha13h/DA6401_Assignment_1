"""
Neural Layer Implementation
Handles weight initialization, forward pass, and gradient computation
"""
import numpy as np
from .activations import get_activation, get_activation_derivative

class NeuralLayer:
    """
    Single neural network layer with weights, biases, and activation function.
    """

    def __init__(self, input_size, output_size, activation='relu', weight_init='xavier'):
        """
        Initialize layer with weights and biases.

        Args:
            input_size: Number of input features
            output_size: Number of output neurons
            activation: Activation function name
            weight_init: Weight initialization method ('random' or 'xavier')
        """
        self.input_size = input_size
        self.output_size = output_size
        self.activation_name = activation

        # Initialize weights
        if weight_init == 'xavier':
            # Xavier/Glorot initialization
            limit = np.sqrt(6 / (input_size + output_size))
            self.W = np.random.uniform(-limit, limit, (input_size, output_size))
        elif weight_init == 'random':
            self.W = np.random.randn(input_size, output_size) * 0.01
        elif weight_init == 'zeros':
            self.W = np.zeros((input_size, output_size))
        else:
            raise ValueError(f"Unknown weight initialization: {weight_init}")

        # Initialize biases to zeros
        self.b = np.zeros((1, output_size))

        # Get activation functions
        self.activation = get_activation(activation)
        if activation != 'softmax':
            self.activation_derivative = get_activation_derivative(activation)

        # Cache for backward pass
        self.X = None
        self.Z = None  # Pre-activation
        self.A = None  # Post-activation

        # Gradients
        self.grad_W = None
        self.grad_b = None

    def forward(self, X):
        """
        Forward pass through the layer.

        Args:
            X: Input data of shape (batch_size, input_size)

        Returns:
            A: Activated output of shape (batch_size, output_size)
        """
        self.X = X
        self.Z = np.dot(X, self.W) + self.b  # Linear transformation
        self.A = self.activation(self.Z)      # Apply activation
        return self.A

    def backward(self, dA, apply_activation_derivative=True):
        """
        Backward pass through the layer.

        Args:
            dA: Gradient of loss with respect to layer output
            apply_activation_derivative: Whether to apply activation derivative

        Returns:
            dX: Gradient with respect to input
        """
        batch_size = self.X.shape[0]

        if apply_activation_derivative and self.activation_name != 'softmax':
            # Apply activation derivative
            dZ = dA * self.activation_derivative(self.Z)
        else:
            # For softmax with cross-entropy, gradient is already computed
            dZ = dA

        # Compute gradients
        self.grad_W = np.dot(self.X.T, dZ)
        self.grad_b = np.sum(dZ, axis=0, keepdims=True)

        # Gradient with respect to input
        dX = np.dot(dZ, self.W.T)

        return dX