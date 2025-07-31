import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

def upgrade_data(position):
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
                (p_smooth[b,:,n,d],
                v_smooth[b,:,n,d], a_smooth[b,:,n,d], v_function[b][n][d]) = fit_spline_to_data(
                    time, position[b,:,n,d])

    return p_smooth, v_smooth, a_smooth, v_function



def v_function_2_vminushalf(v_function_b, frame):

    N = len(v_function_b.keys())
    D = len(v_function_b[0])
    vminushalf = np.zeros((1,N,D))
    for n in range(N):
        for d in range(D):
            vminushalf[0,n,d] = v_function_b[n][d](frame - 1/2)
    return vminushalf

def finite_difference_data(time, position, query = None):
    if query is None:
        query = time
    velocity_fit = np.zeros_like(position)
    acceleration_fit = np.zeros_like(position)

    velocity_fit[1:] = np.diff(position)
    acceleration_fit[2:] = np.diff(velocity_fit[1:])
    spline_velocity = UnivariateSpline(time, velocity_fit, s=0)

    return position, velocity_fit, acceleration_fit, spline_velocity


def upgrade_data_finite_diff(position):
    """Like upgrade_data, but no smoothing, just upgrade_data_finite_diff"""
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
                (p_smooth[b,:,n,d],
                v_smooth[b,:,n,d], a_smooth[b,:,n,d], v_function[b][n][d]) = finite_difference_data(
                    time, position[b,:,n,d])

    return p_smooth, v_smooth, a_smooth, v_function

def fit_spline_to_data(time, position, query = None):
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
    #position_fit_t = np.linspace(query[0],query[-1],len(query) * 10)
    position_fit = spline_position(query)

    velocity_fit = spline_velocity(query)
    acceleration_fit = spline_acceleration(query)

    return position_fit, velocity_fit, acceleration_fit, spline_velocity
