import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt

class CustomDataset(Dataset):
    def __init__(self, priors, posteriors, new_xs, new_ys, y_stars):
        self.priors = priors
        self.posteriors = posteriors
        self.new_xs = new_xs
        self.new_ys = new_ys
        self.y_stars = y_stars

    def __len__(self):
        return len(self.priors)

    def __getitem__(self, idx):
        return (self.priors[idx], self.posteriors[idx], self.new_xs[idx], self.new_ys[idx], self.y_stars[idx])

def get_the_best_past_y_and_x(new_ys, new_xs, t, n, initial_value, dim, grid_size = 100, no_observation = 0):
    aranged_y = torch.reshape(new_ys, (n, t, 1))
    aranged_x = torch.reshape(new_xs, (n, t, dim))

    maxs_y, argmaxs_y  = torch.cummax(aranged_y, dim=1)
    tiled_argmaxs_y = torch.tile(argmaxs_y, (1, 1, dim))
    maxs_x = torch.gather(aranged_x, 1, tiled_argmaxs_y)
    flattened_maxs_x = torch.zeros((n, t, 1))
    for i in range(dim):
        flattened_maxs_x = flattened_maxs_x + maxs_x[:, :, i].unsqueeze(-1) * grid_size ** (dim - i - 1)
    one_hot_x = torch.zeros((n, t, grid_size ** dim)).scatter_(2, flattened_maxs_x.long(), 1)
    one_hot_obs = (torch.ones((n, t, grid_size ** dim)) * no_observation).scatter_(2, flattened_maxs_x.long(), maxs_y)

    one_hot_x_reshaped = one_hot_x.view(n, t, *[grid_size for _ in range(dim)])
    one_hot_x_shifted = torch.zeros_like(one_hot_x_reshaped)
    one_hot_x_shifted[:, 1:] = one_hot_x_reshaped[:, :-1]
    best_past_x = torch.reshape(one_hot_x_shifted, (n*t, *[grid_size for _ in range(dim)]))

    one_hot_obs_reshaped = one_hot_obs.view(n, t, *[grid_size for _ in range(dim)])
    one_hot_obs_shifted = torch.zeros_like(one_hot_obs_reshaped)
    one_hot_obs_shifted[:, 1:] = one_hot_obs_reshaped[:, :-1]
    best_past_obs = torch.reshape(one_hot_obs_shifted, (n*t, *[grid_size for _ in range(dim)]))


    shifted_y = torch.ones_like(maxs_y) * initial_value
    shifted_y[:, 1:] = maxs_y[:, :-1]
    best_past_y = torch.reshape(shifted_y, (n*t, 1))


    return best_past_y, best_past_x, best_past_obs, torch.tile(shifted_y.view(n*t, *[1 for _ in range(dim)]), (1, *[grid_size for _ in range(dim)]))

def bootstrap_confidence_interval(data, num_bootstrap_samples=1000, alpha=0.05):
    """
    Compute the bootstrap confidence interval for the input data.
    
    Parameters:
        data (np.ndarray): 1D array of data.
        num_bootstrap_samples (int): Number of bootstrap samples.
        alpha (float): Significance level for the confidence interval.
        
    Returns:
        tuple: Lower and upper bounds of the confidence interval.
    """
    bootstrap_samples = np.random.choice(data, size=(num_bootstrap_samples, len(data)), replace=True)
    bootstrap_medians = np.median(bootstrap_samples, axis=1)
    lower = np.percentile(bootstrap_medians, 100 * alpha / 2)
    upper = np.percentile(bootstrap_medians, 100 * (1 - alpha / 2))
    return lower, upper

def get_conf_interval(datas):
    data_regrets = datas.numpy()
    lower_regrets = np.empty((int(datas.shape[1])))
    upper_regrets = np.empty((int(datas.shape[1])))
    for t in range(int(datas.shape[1])):
        lower_regret, upper_regret = bootstrap_confidence_interval(data_regrets[:, t])
        lower_regrets[t] = lower_regret
        upper_regrets[t] = upper_regret
    return lower_regrets, upper_regrets

def plot_tensor_heatmap(tensor):
    """
    Plots a 2D tensor as a color grid/heatmap.
    
    Parameters:
    tensor (torch.Tensor): A 2D tensor to visualize
    """
    # Convert to numpy for matplotlib
    if isinstance(tensor, torch.Tensor):
        tensor_np = tensor.cpu().detach().numpy()
    else:
        tensor_np = np.array(tensor)
    
    # Create a figure with a single subplot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot the heatmap
    im = ax.imshow(tensor_np, cmap='Blues')
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Value Magnitude", rotation=-90, va="bottom")
    
    # Set ticks and labels
    ax.set_xticks(np.arange(tensor_np.shape[1]))
    ax.set_yticks(np.arange(tensor_np.shape[0]))
    ax.set_xticklabels([f"{i}" for i in range(tensor_np.shape[1])])
    ax.set_yticklabels([f"{i}" for i in range(tensor_np.shape[0])])
    
    # Loop over data dimensions and create text annotations
    for i in range(tensor_np.shape[0]):
        for j in range(tensor_np.shape[1]):
            text = ax.text(j, i, f"{tensor_np[i, j]:.2f}",
                          ha="center", va="center", 
                          color="white" if tensor_np[i, j] > tensor_np.max()/2 else "black")
    
    ax.set_title("Tensor Heatmap Visualization")
    fig.tight_layout()
    
    return fig

def integer_to_grid_coordinates(max_val_estimate_x, grid_size, dim):
    """
    Converts an integer to a tensor of shape (1, dim) by repeatedly dividing by grid_size.
    
    Parameters:
    max_val_estimate_x (int): The integer to convert
    grid_size (int): The size of the grid (number of divisions in each dimension)
    dim (int): The number of dimensions in the resulting tensor
    
    Returns:
    torch.Tensor: A tensor of shape (1, dim) containing the coordinates
    
    Example:
    >>> integer_to_grid_coordinates(120, 11, 2)
    tensor([[10, 10]])
    """
    coordinates = []
    remaining_value = max_val_estimate_x
    
    for i in range(dim):
        # For each dimension, extract the coordinate by taking the remainder
        # when dividing by grid_size
        coordinate = remaining_value % grid_size
        coordinates.append(coordinate)
        
        # Update the remaining value by integer division
        remaining_value = remaining_value // grid_size
    
    # Reverse the coordinates to match the expected output
    # (This depends on how you want to order your dimensions)
    coordinates.reverse()
    
    # Convert to torch tensor of shape (1, dim)
    return torch.tensor([coordinates])

def find_closest_grid_point(tensor, domain_size, grid_size):
    """
    Find the closest grid point to a given tensor and its index in the flattened grid.
    
    Parameters:
    - tensor: A tensor of shape (1, dim) with values bounded between 0 and domain_size
    - domain_size: The upper bound of the domain in each dimension
    - grid_size: Number of grid points in each dimension
    
    Returns:
    - closest_grid_point: The closest grid point to the input tensor
    - index: The index of the closest grid point in the flattened grid tensor of shape (grid_size ** dim, dim)
    """
    # Extract the dimension from the tensor shape
    dim = tensor.shape[1]
    
    # Calculate the grid spacing
    grid_spacing = domain_size / (grid_size - 1)
    
    # Initialize the closest grid point and indices
    closest_grid_point = np.zeros((1, dim))
    closest_indices = np.zeros(dim, dtype=int)
    
    # For each dimension, find the closest grid point
    for i in range(dim):
        # Get the value in the current dimension
        value = tensor[0, i]
        
        # Calculate the index of the closest grid point in this dimension
        closest_index = round(value.item() / grid_spacing)
        
        # Ensure the index is within bounds
        closest_index = max(0, min(grid_size - 1, closest_index))
        
        # Convert the index back to the actual coordinate
        closest_coordinate = closest_index * grid_spacing
        
        # Store the coordinate in the result
        closest_grid_point[0, i] = closest_coordinate
        
        # Store the index for this dimension
        closest_indices[i] = closest_index
    
    # Calculate the flat index in the grid tensor
    # The grid points are ordered in ascending order starting from the first coordinate
    flat_index = 0
    for i in range(dim):
        # For each dimension, we multiply the index by the number of points in all previous dimensions
        # and add it to the flat index
        flat_index += closest_indices[i] * (grid_size ** (dim - 1 - i))
    
    return torch.tensor(closest_grid_point), flat_index
