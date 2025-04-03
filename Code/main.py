# import torch
import numpy as np
import matplotlib.pyplot as plt
from generate_data import f_generate_parameters, get_dataloaders
from model import ANN, ResNet
from train import train_model


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

# Generate parameters for the function
m = input_dim // 10
function_num = 30
f_parameter_list = [f_generate_parameters(input_dim, m)
                    for _ in range(function_num)]
# Generate parameters for the function
# Generate DataLoaders
loaders = []
for f_parameters in f_parameter_list:
    # Generate DataLoaders
    train_loader, test_loader = get_dataloaders(
        num_samples=1000 * input_dim,  # 1000 times the input_dim
        input_dim=input_dim,
        f_parameters=f_parameters,
        batch_size=batch_size
    )
    loaders.append((train_loader, test_loader))

# Initialize models
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

# Train and test ANN model
ann_loss_list = []
resnet_loss_list = []
# initialize as numpy arrays with the same shape as the number of epochs
# to store the loss over epochs
ann_loss_list_over_epochs = np.zeros(epochs)
resnet_loss_list_over_epochs = np.zeros(epochs)

verbose = False
for i, (train_loader, test_loader) in enumerate(loaders):
    trained_ann_model, ann_loss_over_epochs = train_model(
        ann_model,
        train_loader,
        test_loader,
        epochs=epochs,
        learning_rate=learning_rate,
        device=Device,
        verbose=verbose
    )
    ann_loss_list.append(ann_loss_over_epochs[-1])
    ann_loss_list_over_epochs += np.array(ann_loss_over_epochs)

    # Train and test ResNet model
    trained_resnet_model, resnet_loss_over_epochs = train_model(
        resnet_model,
        train_loader,
        test_loader,
        epochs=epochs,
        learning_rate=learning_rate,
        device=Device,
        verbose=verbose
    )
    resnet_loss_list.append(resnet_loss_over_epochs[-1])
    resnet_loss_list_over_epochs += np.array(resnet_loss_over_epochs)

# save the models
# torch.save(trained_ann_model.state_dict(),
#            "./_stored_models/ann_model.pth")
# torch.save(trained_resnet_model.state_dict(),
#            "./_stored_models/resnet_model.pth")

ann_loss_list_over_epochs /= len(loaders)
resnet_loss_list_over_epochs /= len(loaders)
print("Average test Loss")
print("ANN:", sum(ann_loss_list) / len(ann_loss_list))
print(ann_loss_list)
print("ResNet:", sum(resnet_loss_list) / len(resnet_loss_list))
print(resnet_loss_list)

# Plot loss over epochs for ANN and ResNet
plt.figure(figsize=(10, 6))
epochs_range = list(range(1, epochs + 1))

# Assuming the loss lists are collected over epochs
# for the last training iteration
plt.plot(epochs_range, ann_loss_list_over_epochs, label="ANN Loss")
plt.plot(epochs_range, resnet_loss_list_over_epochs, label="ResNet Loss")

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss over Epochs for ANN and ResNet")
plt.legend()
plt.grid(True)
# save the plot as svg
plt.savefig("./_stored_graphs/loss_over_epochs.svg")
plt.show()
