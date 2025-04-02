from generate_data import get_dataloaders
from model import ANN  # , ResNet
from train import train_model
from test import test_model
# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Parameters
input_dim = 10
output_dim = 1
width = 10
depth = 5
batch_size = 32
epochs = 5
learning_rate = 1e-3

# Generate DataLoaders
train_loader, test_loader = get_dataloaders(
    num_samples=1000,
    input_dim=input_dim,
    output_dim=output_dim,
    batch_size=batch_size
    )

# Initialize model
model = ANN(input_dim=input_dim,
            output_dim=output_dim,
            width=width,
            depth=depth)

# Train the model
trained_model = train_model(model,
                            train_loader,
                            epochs=epochs,
                            learning_rate=learning_rate)

# Test the model on test_loader
test_loss = test_model(trained_model, test_loader, device='cpu')
