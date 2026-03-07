"""
Optimization Algorithms
Implements: SGD, Momentum, Adam, Nadam, etc.
"""
import numpy as np

class Optimizer:
    """Base optimizer class"""

    def __init__(self, learning_rate=0.01, weight_decay=0.0):
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def update(self, layers, grad_W, grad_b):
        """Update weights - to be implemented by subclasses"""
        raise NotImplementedError

class SGD(Optimizer):
    """Stochastic Gradient Descent"""

    def __init__(self, learning_rate=0.01, weight_decay=0.0):
        super().__init__(learning_rate, weight_decay)

    def update(self, layers, grad_W, grad_b):
        """
        Update weights using gradient descent.
        grad_W and grad_b are in reverse order (last layer first)
        """
        num_layers = len(layers)

        for i, layer in enumerate(layers):
            # grad_W and grad_b are in forward order: grad_W[0] for first layer
            grad_idx = i

            # Add L2 regularization gradient
            grad_W_reg = grad_W[grad_idx] + self.weight_decay * layer.W

            # Update weights and biases
            layer.W -= self.learning_rate * grad_W_reg
            layer.b -= self.learning_rate * grad_b[grad_idx]

class Momentum(Optimizer):
    """Momentum optimizer"""

    def __init__(self, learning_rate=0.01, weight_decay=0.0, momentum=0.9):
        super().__init__(learning_rate, weight_decay)
        self.momentum = momentum
        self.velocity_W = []
        self.velocity_b = []
        self.initialized = False

    def update(self, layers, grad_W, grad_b):
        """Update weights using momentum"""
        num_layers = len(layers)

        # Initialize velocities on first call
        if not self.initialized:
            self.velocity_W = [np.zeros_like(layer.W) for layer in layers]
            self.velocity_b = [np.zeros_like(layer.b) for layer in layers]
            self.initialized = True

        for i, layer in enumerate(layers):
            grad_idx = i

            # Add L2 regularization
            grad_W_reg = grad_W[grad_idx] + self.weight_decay * layer.W

            # Update velocities
            self.velocity_W[i] = self.momentum * self.velocity_W[i] + self.learning_rate * grad_W_reg
            self.velocity_b[i] = self.momentum * self.velocity_b[i] + self.learning_rate * grad_b[grad_idx]

            # Update weights
            layer.W -= self.velocity_W[i]
            layer.b -= self.velocity_b[i]

class NAG(Optimizer):
    """Nesterov Accelerated Gradient"""

    def __init__(self, learning_rate=0.01, weight_decay=0.0, momentum=0.9):
        super().__init__(learning_rate, weight_decay)
        self.momentum = momentum
        self.velocity_W = []
        self.velocity_b = []
        self.initialized = False

    def update(self, layers, grad_W, grad_b):
        """Update weights using Nesterov momentum"""
        num_layers = len(layers)

        # Initialize velocities on first call
        if not self.initialized:
            self.velocity_W = [np.zeros_like(layer.W) for layer in layers]
            self.velocity_b = [np.zeros_like(layer.b) for layer in layers]
            self.initialized = True

        for i, layer in enumerate(layers):
            grad_idx = i

            # Add L2 regularization
            grad_W_reg = grad_W[grad_idx] + self.weight_decay * layer.W

            # Save old velocities
            velocity_W_prev = self.velocity_W[i].copy()
            velocity_b_prev = self.velocity_b[i].copy()

            # Update velocities
            self.velocity_W[i] = self.momentum * self.velocity_W[i] + self.learning_rate * grad_W_reg
            self.velocity_b[i] = self.momentum * self.velocity_b[i] + self.learning_rate * grad_b[grad_idx]

            # Nesterov update: -momentum * v_old + (1 + momentum) * v_new
            layer.W -= -self.momentum * velocity_W_prev + (1 + self.momentum) * self.velocity_W[i]
            layer.b -= -self.momentum * velocity_b_prev + (1 + self.momentum) * self.velocity_b[i]

class RMSProp(Optimizer):
    """RMSProp optimizer"""

    def __init__(self, learning_rate=0.01, weight_decay=0.0, beta=0.9, epsilon=1e-8):
        super().__init__(learning_rate, weight_decay)
        self.beta = beta
        self.epsilon = epsilon
        self.cache_W = []
        self.cache_b = []
        self.initialized = False

    def update(self, layers, grad_W, grad_b):
        """Update weights using RMSProp"""
        num_layers = len(layers)

        # Initialize cache on first call
        if not self.initialized:
            self.cache_W = [np.zeros_like(layer.W) for layer in layers]
            self.cache_b = [np.zeros_like(layer.b) for layer in layers]
            self.initialized = True

        for i, layer in enumerate(layers):
            grad_idx = i

            # Add L2 regularization
            grad_W_reg = grad_W[grad_idx] + self.weight_decay * layer.W

            # Update cache with squared gradients
            self.cache_W[i] = self.beta * self.cache_W[i] + (1 - self.beta) * grad_W_reg**2
            self.cache_b[i] = self.beta * self.cache_b[i] + (1 - self.beta) * grad_b[grad_idx]**2

            # Update weights
            layer.W -= self.learning_rate * grad_W_reg / (np.sqrt(self.cache_W[i]) + self.epsilon)
            layer.b -= self.learning_rate * grad_b[grad_idx] / (np.sqrt(self.cache_b[i]) + self.epsilon)

def get_optimizer(name, learning_rate=0.01, weight_decay=0.0):
    """Get optimizer by name"""
    optimizers = {
        'sgd': SGD,
        'momentum': Momentum,
        'nag': NAG,
        'rmsprop': RMSProp
    }

    if name not in optimizers:
        raise ValueError(f"Unknown optimizer: {name}")

    return optimizers[name](learning_rate=learning_rate, weight_decay=weight_decay)