# import torch
from generate_data import get_dataloaders
from model import ANN, ResNet
from train import train_model
from test import test_model

Device = 'cpu'
#  Device = torch.device("mps")
# print(f"Using device: {Device}")


# Parameters
input_dim = 100
output_dim = 1
width = input_dim + 1
depth = input_dim // 10
batch_size = 32
epochs = 100
learning_rate = 1e-3

# Generate DataLoaders
train_loader, test_loader = get_dataloaders(
    num_samples=1000 * input_dim,
    input_dim=input_dim,
    output_dim=output_dim,
    batch_size=batch_size
    )

# Initialize model
ann_model = ANN(
    input_dim=input_dim,
    output_dim=output_dim,
    width=width,
    depth=depth
)

resnet_model = ResNet(
    input_dim=input_dim,
    output_dim=output_dim,
    width=width,
    depth=depth
)

ann_loss_list = []
resnet_loss_list = []

number_of_runs = 1
verbose = True
# Train and test ANN model
for _ in range(number_of_runs):
    trained_ann_model = train_model(ann_model,
                                    train_loader,
                                    epochs=epochs,
                                    learning_rate=learning_rate,
                                    device=Device,
                                    verbose=verbose)

    # Test the ANN model on test_loader
    ann_test_loss = test_model(trained_ann_model,
                               test_loader,
                               device=Device,
                               verbose=verbose)
    ann_loss_list.append(ann_test_loss)

# Train and test ResNet model
for _ in range(number_of_runs):
    trained_resnet_model = train_model(resnet_model,
                                       train_loader,
                                       epochs=epochs,
                                       learning_rate=learning_rate,
                                       device=Device,
                                       verbose=verbose)

    # Test the ResNet model on test_loader
    resnet_test_loss = test_model(trained_resnet_model,
                                  test_loader,
                                  device=Device,
                                  verbose=verbose)
    resnet_loss_list.append(resnet_test_loss)

print("Average test Loss")
print(f"ANN: {sum(ann_loss_list) / len(ann_loss_list)}")
print(f"ResNet: {sum(resnet_loss_list) / len(resnet_loss_list)}")
