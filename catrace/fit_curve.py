import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize_scalar

def bi_exponential(x, a1, b1, b2, c, t0):
    """
    Double exponential function
    Args:
        x: time
        a1: amplitude of first exponential
        b1: decay rate of first exponential
        b2: decay rate of second exponential
        c: offset
        t0: time offset
    """
    return a1 * (1 - np.exp(-b1 * (x-t0))) * np.exp(-b2 * (x-t0)) + c

def compute_biexp_peak_time(params):
    """
    Compute the peak time of the double exponential function.
    Args:
        params: parameters of the double exponential function (a1, b1, b2, c, t0)
    Returns:
        peak_time: time at which the function reaches its maximum
    """
    a1, b1, b2, c, t0 = params
    
    # Define the function to minimize (negative of bi_exponential to find the maximum)
    def neg_bi_exponential(x):
        return -bi_exponential(x, a1, b1, b2, c, t0)
    
    # Use a scalar minimization method to find the peak time
    # The initial guess could be near t0, and the bounds can be adjusted as needed
    result = minimize_scalar(neg_bi_exponential, bounds=(t0, t0 + 100), method='bounded')
    
    # The peak time will be the result of the minimization
    peak_time = result.x
    
    return peak_time

def fit_bi_exponential(x_data, y_data, initial_guess=None, maxfev=10000):
    """
    Fit a double exponential function to the data
    Args:
        x_data: time
        y_data: spike probability
        initial_guess: initial guess for the parameters
    Returns:
        params: parameters of the double exponential function
    """
    if initial_guess is None:
        initial_guess = [4, 0.1, 0.1, 0.1, 0]

    params, _ = curve_fit(bi_exponential, x_data, y_data, p0=initial_guess, maxfev=maxfev)

    return params


# Function to fit double exponential and return parameters for each odor
def fit_bi_exponential_for_each_odor(df):
    params_dict = {}

    # Group by odor
    for odor, group in df.groupby('odor', observed=True, sort=False):
        # Extract the time and spike_prob data
        x_data = group['time'].values - group['time'].values[0]
        y_data = group['spike_prob'].values

        a1 = 4
        b1 = 0.1
        b2 = 0.1
        c = 0.1
        t0 = 0
        initial_guess = [a1, b1, b2, c, t0]

        # Fit the double exponential function to the data
        try:
            params, _ = curve_fit(bi_exponential, x_data, y_data, p0=initial_guess, maxfev=10000)
            params = dict(zip(['a1', 'b1', 'b2', 'c', 't0'], params))
            # Store the parameters for the odor
            params_dict[odor] = params

        except RuntimeError:
            print(f"Could not fit double exponential for odor: {odor}")
            params_dict[odor] = np.nan


    return params_dict

def plot_fits(x_data, y_data, params, ax):
    ax.plot(x_data, y_data, 'o')
    ax.plot(x_data, bi_exponential(x_data, **params))
