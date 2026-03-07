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
        self.hidden_sizes = cli_args.hidden_size
        self.num_layers = cli_args.num_layers
        self.activation = cli_args.activation
        self.weight_init = cli_args.weight_init

        # Training parameters
        self.learning_rate = cli_args.learning_rate
        self.weight_decay = cli_args.weight_decay
        self.loss_name = cli_args.loss

        # Build network layers
        self.layers = []
        layer_sizes = [self.input_size] + self.hidden_sizes + [self.output_size]

        for i in range(len(layer_sizes) - 1):
            # Use specified activation for hidden layers, no activation for output
            if i < len(layer_sizes) - 2:
                activation = self.activation
            else:
                activation = 'softmax'  # Output layer (but we return logits)

            layer = NeuralLayer(
                input_size=layer_sizes[i],
                output_size=layer_sizes[i + 1],
                activation=activation,
                weight_init=self.weight_init
            )
            self.layers.append(layer)

        # Loss function
        self.loss_fn = get_loss_function(self.loss_name)
        self.loss_derivative = get_loss_derivative(self.loss_name)

        # Optimizer
        self.optimizer = get_optimizer(
            cli_args.optimizer,
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

    def forward(self, X):
        """
        Forward propagation through all layers.
        Returns logits (no softmax applied)
        X is shape (b, D_in) and output is shape (b, D_out).
        b is batch size, D_in is input dimension, D_out is output dimension.

        Args:
            X: Input data of shape (batch_size, input_size)

        Returns:
            logits: Raw output (no softmax) of shape (batch_size, output_size)
        """
        self.layer_outputs = [X.copy()]
        activation = X

        # Forward through all layers except the last
        for layer in self.layers[:-1]:
            activation = layer.forward(activation)
            self.layer_outputs.append(activation.copy())

        # Last layer: compute pre-activation but return logits (no softmax)
        last_layer = self.layers[-1]
        last_layer.X = activation
        last_layer.Z = np.dot(activation, last_layer.W) + last_layer.b
        logits = last_layer.Z  # Return raw logits

        # Still compute activated output for internal use if needed
        last_layer.A = softmax(last_layer.Z)

        self.layer_outputs.append(logits.copy())

        return logits

    def backward(self, dA):
        """
        Backward propagation through all layers.
        Args:
            dA: Upstream gradient from loss function (batch_size, num_classes)
        Returns:
            grad_W: List of weight gradients for each layer
            grad_b: List of bias gradients for each layer
        """
        # Backprop through layers in reverse
        grad_W_list = []
        grad_b_list = []

        # Start with output layer
        dX = self.layers[-1].backward(dA, apply_activation_derivative=False)
        grad_W_list.append(self.layers[-1].grad_W)
        grad_b_list.append(self.layers[-1].grad_b)

        # Backprop through hidden layers
        for layer in reversed(self.layers[:-1]):
            dX = layer.backward(dX, apply_activation_derivative=True)
            grad_W_list.append(layer.grad_W)
            grad_b_list.append(layer.grad_b)

        # Reverse the lists so that index 0 = last layer
        grad_W_list.reverse()
        grad_b_list.reverse()

        # Store in object arrays
        self.grad_W = np.empty(len(grad_W_list), dtype=object)
        self.grad_b = np.empty(len(grad_b_list), dtype=object)
        for i, (gw, gb) in enumerate(zip(grad_W_list, grad_b_list)):
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
                grad = self.loss_derivative(y_batch, logits)
                self.backward(grad)
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
        grad = self.loss_derivative(y_batch, logits)
        self.backward(grad)
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
            d[f"W{i}"] = layer.W.copy()
            d[f"b{i}"] = layer.b.copy()
        return d

    def set_weights(self, weight_dict):
        for i, layer in enumerate(self.layers):
            w_key = f"W{i}"
            b_key = f"b{i}"
            if w_key in weight_dict:
                layer.W = weight_dict[w_key].copy()
            if b_key in weight_dict:
                layer.b = weight_dict[b_key].copy()