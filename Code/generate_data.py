import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np


def f(X, seed=42):
    """
    Computes the function:
    f(x) = sum_{i=1}^{m} [ a_i prod_{j ∈ S_i^1} x_j
        + b_i sin(prod_{k ∈ S_i^2} x_k )
        ]
    where S_i^1 and S_i^2 are random subsets of {1, ..., d}.
    This function should recreate the function f(x) used in the paper
    Liu (2024) "Characterizing ResNet’s Universal Approximation Capability".

    Parameters:
        X (np.ndarray): Input array of shape (num_samples, input_dim)
                        with values in [0,1].
        m (int): Number of terms in the summation.
        seed (int): Random seed for reproducibility.

    Returns:
        np.ndarray: Output array of shape (num_samples, 1).
    """
    np.random.seed(seed)
    d = X.shape[1]  # Set d to the number of features
    m = d // 10

    num_samples = X.shape[0]
    output = np.zeros((num_samples, 1))

    for _ in range(m):
        # Randomly sample indices for S_i^1 and S_i^2 with replacement
        S_i1 = np.random.choice(d, size=np.random.randint(1, np.sqrt(d)),
                                replace=True)
        S_i2 = np.random.choice(d, size=np.random.randint(1, np.sqrt(d)),
                                replace=True)

        # Generate random coefficients a_i and b_i
        a_i = np.random.uniform(0, 1)
        b_i = np.random.uniform(0, 0.1)

        # Compute the respective terms
        prod_x_S1 = np.prod(X[:, S_i1], axis=1, keepdims=True)
        prod_x_S2 = np.prod(X[:, S_i2], axis=1, keepdims=True)

        # Accumulate the result
        output += a_i * prod_x_S1 + b_i * np.sin(prod_x_S2)

    return output


def get_dataloaders(num_samples, input_dim, output_dim, batch_size,
                    d=100, test_split=0.1, seed=42):
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

    # Compute f(x) using the external function.
    y = f(X, d).astype(np.float32)
    y_tensor = torch.from_numpy(y)

    # Create the full dataset.
    dataset = TensorDataset(X_tensor, y_tensor)

    # Split dataset into training and test.
    test_size = int(num_samples * test_split)
    train_size = num_samples - test_size
    # Create a generator with a fixed seed for reproducibility.
    generator = torch.Generator().manual_seed(seed)
    # Split dataset into training and test using the generator.
    train_dataset, test_dataset = random_split(
        dataset,
        [train_size, test_size],
        generator=generator,
    )

    # Create DataLoaders.
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              generator=generator)
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=False)

    return train_loader, test_loader


# Example usage
if __name__ == "__main__":
    # Parameters
    num_samples = 1000
    input_dim = 10
    output_dim = 1
    batch_size = 32

    # Generate DataLoaders
    train_loader, test_loader = get_dataloaders(
        num_samples=num_samples,
        input_dim=input_dim,
        output_dim=output_dim,
        batch_size=batch_size
    )

    # Print the shape of the first batch to verify.
    for batch_x, batch_y in train_loader:
        print(batch_x)
        print(batch_y)
        break
