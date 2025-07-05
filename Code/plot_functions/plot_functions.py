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


def trapezoid(x, h=1.5, delta=0.3):
    """Vectorized trapezoid activation function."""
    x = np.asarray(x)  # ensure input is a NumPy array

    y = np.zeros_like(x, dtype=float)

    # Flat top region
    mask_middle = (x >= -1 + delta) & (x <= 1 - delta)
    y[mask_middle] = h

    # Rising edge
    mask_rising = (x > -1) & (x < -1 + delta)
    y[mask_rising] = h / delta * (x[mask_rising] - (-1))

    # Falling edge
    mask_falling = (x > 1 - delta) & (x < 1)
    y[mask_falling] = h / delta * (1 - x[mask_falling])

    # Values outside [-1, 1] already set to 0

    return y


if __name__ == "__main__":
    plot_function(trapezoid, x_min=-2, x_max=2)
