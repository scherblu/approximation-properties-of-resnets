import numpy as np
import matplotlib.pyplot as plt


def plot_function(func, x_min=-2, x_max=2, num_points=10000):
    x = np.linspace(x_min, x_max, num_points)
    y = func(x)
    plt.figure(figsize=(1.5, 1.5))
    plt.plot(x, y)
    plt.xlim(x_min, x_max)
    plt.ylim(-1, 2)
    plt.grid(True)
    plt.tight_layout()
    # Save with tight bounding box to minimize blank space
    path = f"plot_functions/_stored_plots/{func.__name__}.png"
    plt.savefig(path, dpi=200, bbox_inches='tight', pad_inches=0)
    plt.show()


if __name__ == "__main__":
    # Define activation functions
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def tanh(x):
        return np.tanh(x)

    def relu(x):
        return np.maximum(0, x)

    def leaky_relu(x, alpha=0.1):
        return np.where(x > 0, x, alpha * x)

    def softplus(x):
        return np.log(1 + np.exp(x))

    def swish(x):
        return x * sigmoid(x)

    def gelu(x):
        """Gaussian Error Linear Unit (GELU) activation function."""
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi)
                                      * (x + 0.044715 * np.power(x, 3))))

    def squareplus(x):
        return (x + np.sqrt(x**2 + 1)) / 2

    def lalu(x):
        "Laplace Linear Unit (LaLU) activation function."
        return x * np.where(x > 0, 1 - 0.5 * np.exp(-x), 0.5 * np.exp(x))

    def calu(x):
        """Cauchy Linear Unit (CaLU) activation function."""
        return x * (np.arctan(x) / np.pi + 0.5)

    def elu(x):
        """Exponential Linear Unit (ELU) activation function."""
        alpha = 1.0
        return np.where(x > 0, x, alpha * (np.exp(x) - 1))

    def selu(x):
        """Scaled Exponential Linear Unit (SELU) activation function."""
        scale = 1.0507009873554805
        alpha = 1.6732632423543772
        return scale * np.where(x > 0, x, alpha * (np.exp(x) - 1))

    def elish(x):
        """Exponential Linear Unit (ELiSH) activation function."""
        return np.where(x > 0, x * sigmoid(x), (np.exp(x) - 1) * sigmoid(x))

    # Plot each activation function
    plot_function(sigmoid)
    plot_function(tanh)
    plot_function(relu)
    plot_function(leaky_relu)
    plot_function(softplus)
    plot_function(swish)
    plot_function(gelu)
    plot_function(squareplus)
    plot_function(lalu)
    plot_function(calu)
    # plot_function(elu)
    plot_function(selu)
    plot_function(elish)
