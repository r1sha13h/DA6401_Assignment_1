"""
Main Neural Network Model class
Handles forward and backward propagation loops
"""
import numpy as np
from .neural_layer import NeuralLayer
from .objective_functions import get_loss_function, get_loss_derivative
from .optimizers import get_optimizer
from .activations import softmax

class NeuralNetwork:
    """
    Main model class that orchestrates the neural network training and inference.
    """

    def __init__(self, cli_args):
        """
        Initialize neural network from command-line arguments.

        Args:
            cli_args: Parsed arguments containing network configuration
        """
        self.cli_args = cli_args

        # Network architecture
        self.input_size = 784  # MNIST/Fashion-MNIST: 28x28 = 784
        self.output_size = 10  # 10 classes
        raw_hidden_size = getattr(cli_args, 'hidden_size', [64])
        provided_sizes = raw_hidden_size if isinstance(raw_hidden_size, list) else [raw_hidden_size]
        self.num_layers = getattr(cli_args, 'num_layers', len(provided_sizes))
        # Broadcast logic
        if len(provided_sizes) == 1 and self.num_layers > 1:
            self.hidden_sizes = provided_sizes * self.num_layers
        elif len(provided_sizes) > self.num_layers:
            self.hidden_sizes = provided_sizes[:self.num_layers]
        else:
            self.hidden_sizes = provided_sizes
        self.activation_str = getattr(cli_args, 'activation', 'relu')
        self.weight_init = getattr(cli_args, 'weight_init', 'random')

        # Training parameters
        self.learning_rate = getattr(cli_args, 'learning_rate', 0.01)
        self.weight_decay = getattr(cli_args, 'weight_decay', 0.0)
        self.loss_name = getattr(cli_args, 'loss', 'cross_entropy')

        # Build network layers
        self.layers = []
        self._build_network()

        # Loss function
        self.loss_fn = get_loss_function(self.loss_name)

        # Optimizer
        self.optimizer = get_optimizer(
            getattr(cli_args, 'optimizer', 'sgd'),
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay
        )

        # Gradient storage
        self.grad_W = None
        self.grad_b = None

        # Validation histories
        self.val_loss_history = []
        self.val_acc_history = []
        self.grad_norms_first_history = []
        self.train_acc_history = []
        self.grad_history_epoch = []
        self.iter_count = 0
        self.dead_neurons_history = []

    def _build_network(self):
        """Builds architecture: hidden layers with activation, output layer with linear."""
        self.layers = []
        current_input_size = self.input_size

        # Hidden layers
        for i in range(self.num_layers):
            layer_size = self.hidden_sizes[i]
            self.layers.append(NeuralLayer(current_input_size, layer_size,
                                           self.activation_str, self.weight_init))
            current_input_size = layer_size

        # Output layer with linear activation
        self.layers.append(NeuralLayer(current_input_size, self.output_size,
                                       'linear', self.weight_init))

    def forward(self, X):
        """
        Forward propagation through all layers.
        Returns logits (no softmax applied).

        Args:
            X: Input data of shape (batch_size, input_size)

        Returns:
            logits: Raw output (no softmax) of shape (batch_size, output_size)
        """
        self.layer_outputs = [X.copy()]
        A = X

        # Forward through hidden layers
        for layer in self.layers[:-1]:
            A = layer.forward(A)
            self.layer_outputs.append(A.copy())

        # Output layer (linear activation → returns logits directly)
        logits = self.layers[-1].forward(A)
        self.layer_outputs.append(logits.copy())

        return logits

    def backward(self, y_true, logits):
        """
        Backward propagation through all layers.
        Computes softmax internally and derives gradient from loss type.

        Args:
            y_true: True labels (one-hot or integer), shape (batch_size, num_classes) or (batch_size,)
            logits: Raw logits from forward pass, shape (batch_size, num_classes)

        Returns:
            grad_W: List of weight gradients for each layer
            grad_b: List of bias gradients for each layer
        """
        grads = []
        m = y_true.shape[0]

        # Compute softmax probabilities
        if np.allclose(np.sum(logits, axis=1), 1.0):
            probabilities = logits
        else:
            exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        # One-hot encoding if needed
        if y_true.ndim == 1 or (y_true.ndim == 2 and y_true.shape[1] == 1):
            y_one_hot = np.zeros_like(probabilities)
            y_one_hot[np.arange(m), y_true.flatten().astype(int)] = 1
            y_true_used = y_one_hot
        else:
            y_true_used = y_true

        # Compute initial gradient based on loss type
        if self.loss_name == 'cross_entropy':
            dA_to_pass = (probabilities - y_true_used) / m
        elif self.loss_name in ('mse', 'mean_squared_error'):
            # Full MSE derivative through softmax Jacobian
            dA = 2 * (probabilities - y_true_used) / m
            dA_to_pass = probabilities * (dA - np.sum(dA * probabilities, axis=1, keepdims=True))

        # Output layer backward (linear activation, derivative = 1)
        original_deriv = self.layers[-1].activation_derivative
        self.layers[-1].activation_derivative = lambda Z: 1.0
        dX, dW, db = self.layers[-1].backward(dA_to_pass)
        grads.append((dW, db))
        self.layers[-1].activation_derivative = original_deriv

        # Propagate through hidden layers
        for layer in reversed(self.layers[:-1]):
            dX, dW, db = layer.backward(dX)
            grads.append((dW, db))

        # Reverse to get input-to-output order
        grads = grads[::-1]

        # Store gradients in object arrays (for optimizer compatibility)
        grad_w = [g[0] for g in grads]
        grad_b = [g[1] for g in grads]
        self.grad_W = np.empty(len(grad_w), dtype=object)
        self.grad_b = np.empty(len(grad_b), dtype=object)
        for i, (gw, gb) in enumerate(zip(grad_w, grad_b)):
            self.grad_W[i] = gw
            self.grad_b[i] = gb

        return self.grad_W, self.grad_b

    def train(self, X_train, y_onehot_train, epochs=1, batch_size=32, val_X=None, val_y=None):
        """
        Train the model for specified epochs.
        """
        from utils.data_loader import create_batches
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(X_train.shape[0])
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_onehot_train[indices]

            # Collect grad norms for this epoch
            epoch_grad_norms = []

            # Train on batches
            for X_batch, y_batch in create_batches(X_train_shuffled, y_train_shuffled, batch_size):
                logits = self.forward(X_batch)
                loss = self.loss_fn(y_batch, logits)
                self.backward(y_batch, logits)
                self.optimizer.update(self.layers, self.grad_W, self.grad_b)
                epoch_grad_norms.append(np.linalg.norm(self.grad_W[0]))

                self.iter_count += 1

            print(f"Epoch {epoch+1}/{epochs} completed")

            # Collect grad norms for first hidden layer neurons at end of epoch
            n_neurons = min(5, self.grad_W[0].shape[1])
            norms = [np.linalg.norm(self.grad_W[0][:, j]) for j in range(n_neurons)]
            self.grad_history_epoch.append(norms)

            # Average grad norm for the epoch
            self.grad_norms_first_history.append(np.mean(epoch_grad_norms))

            # Validate if val data provided
            if val_X is not None and val_y is not None:
                val_logits = self.forward(val_X)
                val_loss = self.loss_fn(val_y, val_logits)
                val_pred = self.predict(val_X)
                val_acc = np.mean(val_pred == np.argmax(val_y, axis=1))
                self.val_loss_history.append(val_loss)
                self.val_acc_history.append(val_acc)

                # Check dead neurons
                dead_list = []
                for i, out in enumerate(self.layer_outputs):
                    if i > 0 and i < len(self.layer_outputs) - 1:  # hidden layers only
                        layer_dead = np.zeros(out.shape[1])
                        for j in range(out.shape[1]):
                            if np.all(out[:, j] == 0):
                                layer_dead[j] = 1  # dead
                        dead_list.append(layer_dead)
                self.dead_neurons_history.append(dead_list)

            # Compute train accuracy on subset
            y_train_labels = np.argmax(y_onehot_train, axis=1)
            train_pred = self.predict(X_train[:1000])
            train_acc = np.mean(train_pred == y_train_labels[:1000])
            self.train_acc_history.append(train_acc)

    def predict(self, X):
        """
        Predict class labels for input data.
        """
        logits = self.forward(X)
        return np.argmax(logits, axis=1)

    def train_step(self, X_batch, y_batch):
        """
        Perform one training step on a batch.
        """
        logits = self.forward(X_batch)
        loss = self.loss_fn(y_batch, logits)
        self.backward(y_batch, logits)
        self.optimizer.update(self.layers, self.grad_W, self.grad_b)

    def evaluate(self, X, y):
        """
        Evaluate accuracy on given data.
        """
        predictions = self.predict(X)
        acc = np.mean(predictions == y)
        return acc

    def get_weights(self):
        d = {}
        for i, layer in enumerate(self.layers):
            d[f"W{i+1}"] = layer.W.copy()
            d[f"b{i+1}"] = layer.b.copy()
        return d

    def set_weights(self, weights_dict):
        w_keys = sorted([k for k in weights_dict.keys() if k.startswith('W')])
        b_keys = sorted([k for k in weights_dict.keys() if k.startswith('b')])
        # Rebuild layers if architecture doesn't match
        if len(w_keys) != len(self.layers) or self.layers[-1].W.shape[1] != weights_dict[w_keys[-1]].shape[1]:
            self.layers = []
            for i in range(len(w_keys)):
                in_size = weights_dict[w_keys[i]].shape[0]
                out_size = weights_dict[w_keys[i]].shape[1]
                act = 'linear' if i == len(w_keys) - 1 else self.activation_str
                self.layers.append(NeuralLayer(in_size, out_size, act, 'random'))
        # Load the weights
        for i, (wk, bk) in enumerate(zip(w_keys, b_keys)):
            self.layers[i].W = weights_dict[wk].copy()
            self.layers[i].b = weights_dict[bk].copy()