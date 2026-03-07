"""
Data Loading and Preprocessing
Handles MNIST and Fashion-MNIST datasets
"""
import numpy as np
from sklearn.datasets import fetch_openml

def load_data(dataset_name='mnist'):
    """
    Load MNIST or Fashion-MNIST dataset.

    Args:
        dataset_name: 'mnist' or 'fashion_mnist'

    Returns:
        (X_train, y_train), (X_test, y_test): Preprocessed data
    """
    if dataset_name == 'mnist':
        data = fetch_openml('mnist_784', version=1, as_frame=False, parser='liac-arff')
    elif dataset_name == 'fashion_mnist':
        data = fetch_openml('Fashion-MNIST', version=1, as_frame=False, parser='liac-arff')
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    X, y = data.data, data.target.astype(int)

    # Normalize to [0, 1]
    X = X.astype('float32') / 255.0

    # Split into train (60000) and test (10000) — standard MNIST split
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]

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