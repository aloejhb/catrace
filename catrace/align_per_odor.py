import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
from .fit_curve import fit_bi_exponential, compute_biexp_peak_time

# Define the Gaussian function
def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))

def fit_gaussian_to_odor(time_points, activity_values, stddev_bounds=(1, 5)):
    # Initial guess for the parameters: amplitude, mean, stddev
    # amplitude: maximum value of the activity
    # mean: time where the maximum value occurs
    # stddev
    initial_guess = [activity_values.max(), time_points[np.argmax(activity_values)], 2]
    
    # Define bounds for the parameters: (amplitude, mean, stddev)
    # amplitude: [0, np.inf] (positive and unbounded)
    # mean: [min(time_points), max(time_points)] (within the range of time points)
    # stddev: [stddev_bounds[0], stddev_bounds[1]] (within the specified bounds)
    lower_bounds = [0, min(time_points), stddev_bounds[0]]
    upper_bounds = [np.inf, max(time_points), stddev_bounds[1]]

    # Fit the Gaussian function to the data
    try:
        params, _ = curve_fit(gaussian, time_points, activity_values, p0=initial_guess, bounds=(lower_bounds, upper_bounds))
        return params
    except RuntimeError:
        return None


def find_peak_times(odor_avg, window, second_window_size=None, method='gaussian', fit_params={}):
    odor_avg_original = odor_avg.copy()
    odor_avg = odor_avg_original.loc[:, window[0]:window[1]]
    # Convert columns to float (time points)
    time_points = np.array(odor_avg.columns, dtype=float)
    
    if method == 'gaussian':
        fit_params['stddev_bounds'] = fit_params.get('stddev_bounds', (1, 5))

    # Fit for each odor and find the peak time
    results = {}
    for odor in odor_avg.index:
        activity_values = odor_avg.loc[odor].values
        peak_time = fit_peak_times(time_points, activity_values, method, fit_params)
        if second_window_size:
            # set second window around peak time
            second_window = (int(peak_time - second_window_size/2), int(peak_time + second_window_size/2))
            # update time points and activity values
            second_odor_avg = odor_avg_original.loc[:, second_window[0]:second_window[1]]
            second_time_points = np.array(second_odor_avg.columns, dtype=float)
            second_activity_values = second_odor_avg.loc[odor].values
            # fit again
            peak_time = fit_peak_times(second_time_points, second_activity_values, method, fit_params)
        results[odor] = peak_time

    # Convert the results to a Series
    peak_times = pd.Series(results, name='Peak Time').astype(int)
    return peak_times


def fit_peak_times(time_points, activity_values, method='gaussian', fit_params={}):
    pktime = np.nan
    if method == 'gaussian':
        params = fit_gaussian_to_odor(time_points, activity_values, **fit_params)
        if params is not None:
            amplitude, mean, stddev = params
            pktime = mean
    elif method == 'biexp':
        time_offset = time_points[0]
        params = fit_bi_exponential(time_points-time_offset, activity_values, **fit_params)
        if params is not None:
            pktime = compute_biexp_peak_time(params) + time_offset
    else:
        raise ValueError("Invalid method. Choose 'gaussian' or 'biexp'.")
    return pktime


def align_odors(dff, delays):
    # Initialize an empty list to collect the results
    aligned_data = []
    # delays has index odor and value for delay as a dataframe
    # Iterate over each odor and its corresponding shift from the pandas series delays
    for odor, shift in delays.items():
        # Extract data for the current odor
        odor_data = dff.xs(odor, level='odor', drop_level=False)

        # Initialize a list to collect the shifted trials for the current odor
        shifted_trials = []

        # Group by 'trial' and shift each trial
        for trial, trial_data in odor_data.groupby(level='trial'):
            shifted_trial = trial_data.shift(-shift)
            shifted_trials.append(shifted_trial)

        # Combine the shifted trials back together
        shifted_data = pd.concat(shifted_trials)

        # Collect the shifted data
        aligned_data.append(shifted_data)

    # Concatenate all aligned data into a single DataFrame
    dff_odors_aligned = pd.concat(aligned_data).sort_index()

    return dff_odors_aligned

