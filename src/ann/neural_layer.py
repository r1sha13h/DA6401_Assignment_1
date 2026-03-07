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
            limit = np.sqrt(2.0 / (input_size + output_size))
            self.W = np.random.uniform(-limit, limit, (input_size, output_size))
        elif weight_init == 'he':
            self.W = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        elif weight_init == 'random':
            self.W = np.random.randn(input_size, output_size) * 0.01
        elif weight_init == 'zeros':
            self.W = np.zeros((input_size, output_size))
        else:
            # Fallback to small random
            self.W = np.random.randn(input_size, output_size) * 0.01

        # Initialize biases to zeros
        self.b = np.zeros((1, output_size))

        # Get activation functions
        self.activation = get_activation(activation)
        if activation not in ('softmax',):
            self.activation_derivative = get_activation_derivative(activation)
        else:
            self.activation_derivative = None

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

    def backward(self, dA):
        """
        Backward pass through the layer.

        Args:
            dA: Gradient of loss with respect to layer output

        Returns:
            (dX, dW, db): Gradients with respect to input, weights, biases
        """
        # Apply activation derivative
        if self.activation_derivative is not None:
            dZ = dA * self.activation_derivative(self.Z)
        else:
            dZ = dA

        # Compute gradients
        dW = np.dot(self.X.T, dZ)
        db = np.sum(dZ, axis=0, keepdims=True)
        self.grad_W = dW
        self.grad_b = db

        # Gradient with respect to input
        dX = np.dot(dZ, self.W.T)

        return dX, dW, db