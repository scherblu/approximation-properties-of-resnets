from torch import nn
import torch.nn.functional as F
import torchsummary


class ANN(nn.Module):
    def __init__(self, input_dim, output_dim, width, depth):
        """
        Initializes the ANN.

        Parameters:
            input_dim (int): Dimension of the input.
            output_dim (int): Dimension of the output.
            width (int): Number of neurons in each hidden layer.
            depth (int): Number of hidden layers.
        """
        super(ANN, self).__init__()

        # Create a list of layers using ModuleList.
        self.layers = nn.ModuleList()

        # Input layer: from input_dim to width.
        self.layers.append(nn.Linear(input_dim, width))

        # Hidden layers: each from width to width.
        for _ in range(depth - 1):
            self.layers.append(nn.Linear(width, width))

        # Output layer: from width to output_dim.
        self.out = nn.Linear(width, output_dim)

    def forward(self, x):
        """
        Forward pass through the ANN.
        Parameters:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim).
        """
        # Pass through each hidden layer with ReLU activation.
        for layer in self.layers:
            x = F.relu(layer(x))
        # Compute the final output.
        x = self.out(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, width):
        """
        Initializes a residual block.
        Parameters:
            width (int): Number of neurons in the block.
        """
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(width, width)
        self.linear2 = nn.Linear(width, width)

    def forward(self, x):
        identity = x
        out = F.relu(self.linear1(x))
        out = self.linear2(out)
        out += identity  # identity skip connection
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, input_dim, output_dim, width, depth):
        """
        Initializes the ResNet.

        Parameters:
            input_dim (int): Dimension of the input.
            output_dim (int): Dimension of the output.
            width (int): Number of neurons in each hidden layer.
            depth (int): Number of layers (adjusted to match ANN parameters).
        """
        super(ResNet, self).__init__()

        # Initial layer: from input_dim to width.
        self.input_layer = nn.Linear(input_dim, width)

        # Adjust the number of residual blocks to match the param. count of ANN
        num_blocks = max(1, (depth - 1) // 2)
        self.blocks = nn.ModuleList(
            [ResidualBlock(width) for _ in range(num_blocks)]
        )

        # Additional linear layers to match the parameter count.
        self.linear_layers = nn.ModuleList(
            [nn.Linear(width, width) for _ in range(depth - 3 - num_blocks)]
        )

        # Output layer: from width to output_dim.
        self.output_layer = nn.Linear(width, output_dim)

    def forward(self, x):
        """
        Forward pass through the ResNet.
        Parameters:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim).
        """
        # Project input to the hidden width.
        x = F.relu(self.input_layer(x))

        # Pass through the residual blocks.
        for block in self.blocks:
            x = block(x)

        # Pass through additional linear layers.
        for layer in self.linear_layers:
            x = F.relu(layer(x))

        # Map to the output dimension.
        x = self.output_layer(x)
        return x


if __name__ == "__main__":
    # Example usage for a small ANN and ResNet model:
    model = ANN(input_dim=10, output_dim=2, width=11, depth=5)
    print(model)
    # print model summary
    torchsummary.summary(model, (10,), device="cpu")

    model = ResNet(input_dim=10, output_dim=2, width=11, depth=5)
    print(model)
    torchsummary.summary(model, (10,), device="cpu")
