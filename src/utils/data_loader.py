"""
Data Loading and Preprocessing
Handles MNIST and Fashion-MNIST datasets
"""
import numpy as np
from tensorflow import keras

def load_data(dataset_name='mnist'):
    """
    Load MNIST or Fashion-MNIST dataset.

    Args:
        dataset_name: 'mnist' or 'fashion_mnist'

    Returns:
        (X_train, y_train), (X_test, y_test): Preprocessed data
    """
    if dataset_name == 'mnist':
        (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    elif dataset_name == 'fashion_mnist':
        (X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Flatten images: 28x28 -> 784
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    # Normalize to [0, 1]
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    return (X_train, y_train), (X_test, y_test)

def one_hot_encode(y, num_classes=10):
    """
    Convert integer labels to one-hot encoding.

    Args:
        y: Integer labels of shape (n,)
        num_classes: Number of classes

    Returns:
        one_hot: One-hot encoded labels of shape (n, num_classes)
    """
    n = y.shape[0]
    one_hot = np.zeros((n, num_classes))
    one_hot[np.arange(n), y] = 1
    return one_hot

def create_batches(X, y, batch_size=32, shuffle=True):
    """
    Create batches from data.

    Args:
        X: Input data
        y: Labels
        batch_size: Size of each batch
        shuffle: Whether to shuffle data

    Yields:
        (X_batch, y_batch): Batches of data
    """
    n_samples = X.shape[0]
    indices = np.arange(n_samples)

    if shuffle:
        np.random.shuffle(indices)

    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_indices = indices[start_idx:end_idx]

        yield X[batch_indices], y[batch_indices]