from torch import nn, no_grad
from tqdm import tqdm


def test_model(model, test_loader, device='cpu', verbose=True):
    """
    Tests the model on the provided test DataLoader.

    Parameters:
        model (nn.Module): The trained neural network model.
        test_loader (DataLoader): DataLoader for the test data.
        device (str): Device to run the evaluation ('cpu' or 'cuda').
        verbose (bool): If True, prints the test loss.

    Returns:
        avg_loss (float): The average MSE loss on the test set.
    """
    model.to(device)
    criterion = nn.MSELoss()
    model.eval()

    loss_list = []
    total_loss = 0.0
    total_samples = 0

    # No gradient calculation is needed for evaluation.
    with no_grad():
        # tqdm is used to show progress in a bar if verbose is True.
        data_iterator = (tqdm(test_loader, desc="Testing", ncols=80) if verbose
                         else test_loader)
        for batch_x, batch_y in data_iterator:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss_list.append(loss.item() * batch_x.size(0))
            total_samples += batch_x.size(0)

    total_loss = sum(loss_list)
    avg_loss = total_loss / total_samples
    if verbose:
        print(f"Test MSE Loss: {avg_loss:.6f}")
    return avg_loss

# Example usage:
# Assuming you have a trained model and a test_loader from get_dataloaders:
# test_loss = test_model(trained_model, test_loader, device='cpu')
