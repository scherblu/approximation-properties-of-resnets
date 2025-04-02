from torch import nn, no_grad
from tqdm import tqdm


def test_model(model, test_loader, device='cpu'):
    """
    Tests the model on the provided test DataLoader.

    Parameters:
        model (nn.Module): The trained neural network model.
        test_loader (DataLoader): DataLoader for the test data.
        device (str): Device to run the evaluation ('cpu' or 'cuda').

    Returns:
        avg_loss (float): The average MSE loss on the test set.
    """
    model.to(device)
    criterion = nn.MSELoss()
    model.eval()

    total_loss = 0.0
    total_samples = 0

    # No gradient calculation is needed for evaluation.
    with no_grad():
        for batch_x, batch_y in tqdm(test_loader, desc="Testing", ncols=80):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item() * batch_x.size(0)
            total_samples += batch_x.size(0)

    avg_loss = total_loss / total_samples
    print(f"Test MSE Loss: {avg_loss:.6f}")
    return avg_loss

# Example usage:
# Assuming you have a trained model and a test_loader from get_dataloaders:
# test_loss = test_model(trained_model, test_loader, device='cpu')
