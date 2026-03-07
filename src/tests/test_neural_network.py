# Unit and Integration Tests for Neural Network

import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ann.neural_layer import NeuralLayer
from ann.neural_network import NeuralNetwork
from utils.data_loader import one_hot_encode, create_batches

def test_neural_layer():
    """Unit test for NeuralLayer forward and backward pass"""
    print("Testing NeuralLayer...")

    # Test forward pass shape
    layer = NeuralLayer(784, 128, 'relu', 'xavier')
    X = np.random.randn(32, 784)
    output = layer.forward(X)
    assert output.shape == (32, 128), f"Expected (32, 128), got {output.shape}"

    # Test gradient computation
    dA = np.random.randn(32, 128)
    dX = layer.backward(dA)
    assert dX.shape == (32, 784), f"Expected (32, 784), got {dX.shape}"
    assert layer.grad_W.shape == (784, 128), f"Expected (784, 128), got {layer.grad_W.shape}"
    assert layer.grad_b.shape == (1, 128), f"Expected (1, 128), got {layer.grad_b.shape}"

    print("NeuralLayer tests passed!")

def test_neural_network():
    """Integration test for NeuralNetwork training"""
    print("Testing NeuralNetwork...")

    # Create dummy args
    class Args:
        def __init__(self):
            self.num_layers = 2
            self.hidden_size = [128, 64]
            self.activation = 'relu'
            self.weight_init = 'xavier'
            self.loss = 'cross_entropy'
            self.optimizer = 'sgd'
            self.learning_rate = 0.01
            self.weight_decay = 0.0

    args = Args()

    # Test training on small dataset
    X_train = np.random.randn(100, 784)
    y_train = np.random.randint(0, 10, 100)
    y_train_onehot = one_hot_encode(y_train)

    model = NeuralNetwork(args)
    initial_acc = model.evaluate(X_train, y_train)

    # Train for 5 epochs
    for _ in range(5):
        for X_batch, y_batch in create_batches(X_train, y_train_onehot, 32):
            model.train_step(X_batch, y_batch)

    final_acc = model.evaluate(X_train, y_train)
    assert final_acc > initial_acc, f"Accuracy did not improve: {initial_acc} -> {final_acc}"

    print("NeuralNetwork tests passed!")

if __name__ == "__main__":
    test_neural_layer()
    test_neural_network()
    print("All tests passed!")
