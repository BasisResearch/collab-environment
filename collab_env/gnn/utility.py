import torch
import numpy as np
from scipy.interpolate import UnivariateSpline
from torch.utils.data import random_split
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader


def handle_discrete_data(position, input_differentiation):
    if "spline" in input_differentiation:
        p_smooth, v_smooth, a_smooth, v_function = spline_data(position)
    else:  # "finite differencing" is the default
        p_smooth, v_smooth, a_smooth, v_function = finite_diff_data(position)
    return p_smooth, v_smooth, a_smooth, v_function


def spline_data(position):
    """
    p_smooth, v_smooth, a_smooth are np arrays
    v_function is function
    """
    B, F, N, dim = np.shape(position)
    time = np.arange(F)
    p_smooth = np.zeros_like(position)
    v_smooth = np.zeros_like(position)
    a_smooth = np.zeros_like(position)
    v_function = {}

    for b in range(B):
        v_function[b] = {}
        for n in range(N):
            v_function[b][n] = {}
            for d in range(dim):
                (
                    p_smooth[b, :, n, d],
                    v_smooth[b, :, n, d],
                    a_smooth[b, :, n, d],
                    v_function[b][n][d],
                ) = fit_spline_to_data(time, position[b, :, n, d])

    return p_smooth, v_smooth, a_smooth, v_function


def v_function_2_vminushalf(v_function, frame):
    if v_function is None:
        return None
        
    B = len(v_function.keys())
    N = len(v_function[0].keys())
    D = len(v_function[0][0])

    vminushalf = np.zeros((B, N, D))

    b_ind = 0
    for b in v_function.keys():
        v_function_b = v_function[b]

        for n in range(N):
            for d in range(D):
                vminushalf[b_ind, n, d] = v_function_b[n][d](frame - 1 / 2)

        b_ind += 1

    return vminushalf


def finite_diff(time, position, query=None):
    if query is None:
        query = time

    velocity_fit = np.zeros(position.shape)

    velocity_fit[1:] = torch.diff(position).detach().cpu().numpy()
    spline_velocity = UnivariateSpline(time, velocity_fit, s=0)

    return spline_velocity


def finite_diff_data(position):
    """Like upgrade_data, but no smoothing, just upgrade_data_finite_diff"""
    B, F, N, dim = np.shape(position)
    time = np.arange(F)

    v = torch.zeros_like(position)
    a = torch.zeros_like(position)
    v_function = {}

    v[:, 1:, :, :] = torch.diff(position, axis=1)
    a[:, :-1, :, :] = torch.diff(v, axis=1)

    return position, v, a, None


def fit_spline_to_data(time, position, query=None):
    if query is None:
        query = time
    # 1. Fit a spline to the position data
    # The 's' parameter controls the smoothing. A smaller 's' makes the spline fit closer to the data points.
    # For interpolation (passing exactly through the points), you can use s=0.
    # For noisy data, a higher 's' value will provide a smoother spline.
    spline_position = UnivariateSpline(time, position, s=0)

    # 2. Calculate the first derivative (velocity)
    # The 'derivative()' method with n=1 calculates the first derivative.
    spline_velocity = spline_position.derivative(n=1)

    # 3. Calculate the second derivative (acceleration)
    # The 'derivative()' method with n=2 calculates the second derivative.
    spline_acceleration = spline_position.derivative(n=2)

    # 4. Evaluate the spline and its derivatives at desired time points
    # position_fit_t = np.linspace(query[0],query[-1],len(query) * 10)
    position_fit = spline_position(query)

    velocity_fit = spline_velocity(query)
    acceleration_fit = spline_acceleration(query)

    return position_fit, velocity_fit, acceleration_fit, spline_velocity

def dataset2testloader(dataset, batch_size = 1, return_train = 0, device = None):
    # have to use seed = 2025
    # split data into training set and test set
    test_size = int(len(dataset) / 2)
    train_size = len(dataset) - test_size
    
    # Create generator on the appropriate device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    generator = torch.Generator(device=device).manual_seed(2025)
    
    train_dataset, test_dataset = random_split(
        dataset, [train_size, test_size], generator=generator
    )

    # right now we assume the batch_size = 1, because our real dataset are of different lengths.
    # But we can expand to minibatches - except fpr a few specific functions, every function is written with minibatches in mind.
    test_loader = DataLoader(test_dataset,
                             batch_size = batch_size, shuffle=False)

    if return_train:
        train_loader = DataLoader(train_dataset,
                             batch_size = batch_size, shuffle=False)
        return test_loader, train_loader
    
    return test_loader

