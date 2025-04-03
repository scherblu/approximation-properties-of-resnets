import torch
from torch import nn
import torch.optim as optim
from tqdm import tqdm
from test import test_model


def train_model(
        model,
        train_loader,
        test_loader,
        epochs,
        learning_rate=1e-3,
        device='cpu',
        verbose=True
        ):
    """
    Trains the given model using the provided training DataLoader.

    Parameters:
        model (nn.Module): The neural network model.
        train_loader (DataLoader): DataLoader for the training data.
        test_loader (DataLoader): DataLoader for the test data.
        epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for Adam optimizer.
        device (str): Device ('cpu' or 'cuda').
        verbose (bool): If False, suppress all prints and tqdm progress bars.

    Returns:
        model (nn.Module): The trained model.
        loss_over_epochs (list): List of average test losses over epochs.
    """
    torch.manual_seed(42)
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    loss_over_epochs = []
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0

        # Set up tqdm progress bar for batches in the current epoch.
        progress_bar = tqdm(enumerate(train_loader),
                            total=len(train_loader),
                            desc=f"Epoch {epoch+1}/{epochs}", ncols=80,
                            disable=not verbose)

        for batch_idx, (batch_x, batch_y) in progress_bar:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if verbose:
                progress_bar.set_postfix({
                                'Batch': batch_idx + 1,
                                'Avg Loss': running_loss / (batch_idx + 1)
                })
        # Test the model after each epoch.
        model.eval()
        av_test_loss = test_model(model, test_loader, verbose=False)
        model.train()
        loss_over_epochs.append(av_test_loss)

    return model, loss_over_epochs
