"""
Inference Script
Evaluate trained models on test sets
"""

import argparse
import json
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
import sys
import os

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ann.neural_network import NeuralNetwork
from utils.data_loader import load_data, one_hot_encode

def parse_arguments():
    """
    Parse command-line arguments for inference.
    """
    parser = argparse.ArgumentParser(description='Run inference on test set')

    # Model and data
    parser.add_argument('--model_path', type=str, default='best_model.npy',
                        help='Path to saved model weights')
    parser.add_argument('-d', '--dataset', type=str, default='mnist',
                        choices=['mnist', 'fashion_mnist'],
                        help='Dataset to evaluate on')
    parser.add_argument('-b', '--batch_size', type=int, default=64,
                        help='Batch size for inference')

    # Network architecture (must match training)
    parser.add_argument('-nhl', '--num_layers', type=int, default=2,
                        help='Number of hidden layers')
    parser.add_argument('-sz', '--hidden_size', type=int, nargs='+', default=[128, 64],
                        help='Hidden layer sizes')
    parser.add_argument('-a', '--activation', type=str, default='relu',
                        choices=['relu', 'sigmoid', 'tanh'],
                        help='Activation function')
    parser.add_argument('-l', '--loss', type=str, default='cross_entropy',
                        choices=['cross_entropy', 'mean_squared_error', 'mse'],
                        help='Loss function')
    parser.add_argument('-o', '--optimizer', type=str, default='sgd',
                        help='Optimizer (for initialization only)')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.01,
                        help='Learning rate (for initialization only)')
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.0,
                        help='Weight decay (for initialization only)')
    parser.add_argument('-w_i', '--weight_init', type=str, default='xavier',
                        help='Weight initialization (overridden by loaded weights)')
    parser.add_argument('-w_p', '--wandb_project', type=str, default='mlp-numpy',
                        help='Weights & Biases project name (not used in inference)')

    # Config file (optional - overrides individual arguments)
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config JSON file (overrides other arguments)')

    args = parser.parse_args()

    # Load config if provided
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
        # Override args with config values
        for key, value in config.items():
            if hasattr(args, key):
                setattr(args, key, value)

    # Validate hidden_size matches num_layers
    if len(args.hidden_size) == 1 and args.num_layers > 1:
        args.hidden_size = args.hidden_size * args.num_layers
    elif len(args.hidden_size) != args.num_layers:
        if len(args.hidden_size) < args.num_layers:
            args.hidden_size.extend([args.hidden_size[-1]] * (args.num_layers - len(args.hidden_size)))
        else:
            args.hidden_size = args.hidden_size[:args.num_layers]

    return args

def load_model(model_path):
    """
    Load trained model from disk.

    Args:
        model_path: Path to .npy file containing model weights

    Returns:
        weights: Dictionary of model weights
    """
    data = np.load(model_path, allow_pickle=True).item()
    return data

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model on test data.

    Args:
        model: Trained neural network model
        X_test: Test input data
        y_test: Test labels (integers)

    Returns:
        results: Dictionary containing logits, loss, accuracy, f1, precision, recall
    """
    # Get predictions
    logits = model.forward(X_test)
    predictions = model.predict(X_test)

    # Compute metrics
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average='weighted')
    precision = precision_score(y_test, predictions, average='weighted', zero_division=0)
    recall = recall_score(y_test, predictions, average='weighted', zero_division=0)

    # Compute loss
    y_test_onehot = one_hot_encode(y_test)
    loss = model.loss_fn(y_test_onehot, logits)

    results = {
        'logits': logits,
        'loss': float(loss),
        'accuracy': float(accuracy),
        'f1': float(f1),
        'precision': float(precision),
        'recall': float(recall)
    }

    return results

def main():
    """
    Main inference function.

    Returns:
        Dictionary containing logits, loss, accuracy, f1, precision, recall
    """
    args = parse_arguments()

    print(f"Loading {args.dataset} dataset...")
    (X_train, y_train), (X_test, y_test) = load_data(args.dataset)

    print(f"Test samples: {X_test.shape[0]}")

    # Initialize model with same architecture as training
    print("\nInitializing model...")
    print(f"Architecture: {[784] + args.hidden_size + [10]}")
    print(f"Activation: {args.activation}")

    model = NeuralNetwork(args)

    # Load trained weights
    print(f"\nLoading model from {args.model_path}...")
    weights = load_model(args.model_path)
    model.set_weights(weights)
    print("Model loaded successfully!")

    # Evaluate on test set
    print("\nEvaluating on test set...")
    results = evaluate_model(model, X_test, y_test)

    print(f"\nTest Results:")
    print(f"  Loss: {results['loss']:.4f}")
    print(f"  Accuracy: {results['accuracy']:.4f}")
    print(f"  F1 Score: {results['f1']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall: {results['recall']:.4f}")

    # Compute and display confusion matrix
    predictions = model.predict(X_test)
    cm = confusion_matrix(y_test, predictions)
    print("\nConfusion Matrix:")
    print(cm)

    print("\nEvaluation complete!")

    return results

if __name__ == '__main__':
    main()