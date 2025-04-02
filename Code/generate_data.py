import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np


def get_dataloaders(num_samples, input_dim, output_dim,
                    batch_size, test_split=0.2, seed=42):
    """
    Generates training and test DataLoaders.
    
    Parameters:
        num_samples (int): Total number of samples.
        input_dim (int): Dimensionality of the input.
        output_dim (int): Dimensionality of the output.
        batch_size (int): Batch size for the DataLoaders.
        test_split (float): Fraction of data to be used as test data.
        seed (int): Random seed for reproducibility.

    Returns:
        train_loader, test_loader (DataLoader, DataLoader):
            Training and test data loaders.
    """
    np.random.seed(seed)

    # Sample uniformly from [0, 1]^input_dim.
    X = np.random.uniform(0, 1, (num_samples, input_dim)).astype(np.float32)
    X_tensor = torch.from_numpy(X)

    # Compute f(x) = sum(sin(x_i)) for each sample.
    # f returns a scalar value, so add an extra dimension to match output_dim.
    y = np.sum(np.sin(X), axis=1, keepdims=True).astype(np.float32)
    y_tensor = torch.from_numpy(y)

    # Create the full dataset.
    dataset = TensorDataset(X_tensor, y_tensor)

    # Split dataset into training and test.
    test_size = int(num_samples * test_split)
    train_size = num_samples - test_size
    train_dataset, test_dataset = random_split(dataset,
                                               [train_size, test_size])

    # Create DataLoaders.
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True)
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=False)

    return train_loader, test_loader
