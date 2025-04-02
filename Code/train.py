from torch import nn
import torch.optim as optim
from tqdm import tqdm


def train_model(model, train_loader, epochs, learning_rate=1e-3, device='cpu'):
    """
    Trains the given model using the provided training DataLoader.
    
    Parameters:
        model (nn.Module): The neural network model.
        train_loader (DataLoader): DataLoader for the training data.
        epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for Adam optimizer.
        device (str): Device ('cpu' or 'cuda').
        
    Returns:
        model (nn.Module): The trained model.
    """
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        
        # Set up tqdm progress bar for batches in the current epoch.
        progress_bar = tqdm(enumerate(train_loader),
                            total=len(train_loader),
                            desc=f"Epoch {epoch+1}/{epochs}", ncols=80)
        
        for batch_idx, (batch_x, batch_y) in progress_bar:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            progress_bar.set_postfix({'Batch': batch_idx + 1,
                                      'Loss': loss.item()})
        
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} completed."
              + f"Average Loss: {epoch_loss:.6f}")
    
    return model
