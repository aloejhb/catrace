import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

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

# Function to fit double exponential and return parameters for each odor
def fit_double_exponential_for_each_odor(df):
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
