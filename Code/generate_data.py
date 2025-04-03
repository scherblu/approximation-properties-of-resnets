import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np


def f_generate_parameters(d, m):
    """
    Generates the parameters a, b, S1, and S2 for the function f(x).

    Parameters:
        d (int): Dimensionality of the input.
        m (int): Number of terms in the summation.
        seed (int): Random seed for reproducibility.

    Returns:
        list: A list of tuples (a_i, b_i, S_i1, S_i2) for each term.
    """
    parameters = []

    for _ in range(m):
        # Randomly sample indices for S_i^1 and S_i^2 with replacement
        S_i1 = np.random.choice(d, size=np.random.randint(1, np.sqrt(d)),
                                replace=True)
        S_i2 = np.random.choice(d, size=np.random.randint(1, np.sqrt(d)),
                                replace=True)

        # Generate random coefficients a_i and b_i
        a_i = np.random.uniform(0, 1)
        b_i = np.random.uniform(0, 0.1)

        # Append the parameters as a tuple
        parameters.append((a_i, b_i, S_i1, S_i2))

    return parameters


def f_eval(X, f_parameters):
    """
    Evaluates the function f(x) based on the given parameters.

    Parameters:
        X (np.ndarray): Input array of shape (num_samples, input_dim)
                        with values in [0,1].
        f_parameters (list): list of tuples (a_i, b_i, S_i1, S_i2)

    Returns:
        np.ndarray: Output array of shape (num_samples, 1).
    """
    num_samples = X.shape[0]
    output = np.zeros((num_samples, 1))

    for a_i, b_i, S_i1, S_i2 in f_parameters:
        # Compute the respective terms
        prod_x_S1 = np.prod(X[:, S_i1], axis=1, keepdims=True)
        prod_x_S2 = np.prod(X[:, S_i2], axis=1, keepdims=True)

        # Accumulate the result
        output += a_i * prod_x_S1 + b_i * np.sin(prod_x_S2)

    return output


def get_X(num_samples, input_dim, seed=42):
    """
    Generates a random input tensor X.

    Parameters:
        num_samples (int): Number of samples.
        input_dim (int): Dimensionality of the input.
        d (int): Dimensionality for the function f.
        seed (int): Random seed for reproducibility.

    Returns:
        torch.Tensor: Input tensor of shape (num_samples, input_dim).
    """
    np.random.seed(seed)
    X = np.random.uniform(0, 1, (num_samples, input_dim)).astype(np.float32)
    return X


def get_dataloaders(num_samples, input_dim, f_parameters, batch_size,
                    test_split=0.1, seed=42):
    """
    Generates training and test DataLoaders.

    Parameters:
        num_samples (int): Total number of samples.
        input_dim (int): Dimensionality of the input.
        f_parameters (list): Parameters for the function f(x).
        batch_size (int): Batch size for the DataLoaders.
        test_split (float): Fraction of data to be used as test data.
        seed (int): Random seed for reproducibility.

    Returns:
        train_loader, test_loader (DataLoader, DataLoader):
            Training and test data loaders.
    """
    np.random.seed(seed)

    # Sample uniformly from [0, 1]^input_dim.
    X = get_X(num_samples, input_dim, seed=seed)
    X_tensor = torch.from_numpy(X)

    # Compute f(x) using the external function.
    y = f_eval(X, f_parameters).astype(np.float32)
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
    f_parameters = f_generate_parameters(input_dim, 5)
    # Generate DataLoaders
    train_loader, test_loader = get_dataloaders(
        num_samples,
        input_dim,
        f_parameters,
        batch_size,
    )

    # Print the shape of the first batch to verify.
    for batch_x, batch_y in train_loader:
        print(batch_x)
        print(batch_y)
        break
