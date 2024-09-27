import numpy as np
import pandas as pd

def spike_timing_to_rate(data, dt, num_neurons, num_steps, debug=False):
    dt_in_steps = dt * 10
    df = pd.DataFrame(data, columns=['step', 'neuron'])
    # Convert starting from step 1 to step from step 0 (matlab convention -> python convention)
    df['step'] = df['step'] - 1
    # Bin the time points
    df['time'] = (df['step'] // (dt_in_steps)).astype(int)
    # Count spikes per neuron per bin
    result = df.pivot_table(index='neuron', columns='time', aggfunc='size', fill_value=0)
    print(result.shape)
    num_times = np.floor(num_steps/ (dt_in_steps)).astype(int)
    # Ensure all neurons are represented in the result
    all_neurons = pd.DataFrame(0.0, index=np.arange(1, num_neurons + 1), columns=range(0, num_times))
    print(all_neurons.shape)
    # Update the full DataFrame with the existing counts
    all_neurons.update(result)

    # Normalize by dt to get rate (assuming dt is in milliseconds or needs conversion to seconds)
    all_neurons /= (dt / 1000)

    all_neurons = all_neurons.T
    if debug:
        return result, all_neurons
    return all_neurons


def specify_odor_and_trial(dff, odors, num_trials):
    # Number of odors
    num_odors = len(odors)

    # Calculate the number of time points per trial
    num_time_points = len(dff) // (num_odors * num_trials)

    # Create MultiIndex
    # Generating the odor, trial, and time indices
    trial_labels = np.repeat(np.arange(num_trials), num_odors * num_time_points)
    odor_labels = np.tile(np.repeat(odors, num_time_points), num_trials)
    time_labels = np.tile(np.arange(num_time_points), num_odors * num_trials)

    # Setting the MultiIndex
    dff.index = pd.MultiIndex.from_arrays([trial_labels, odor_labels, time_labels], names=['trial', 'odor', 'time'])
    dff = dff.reorder_levels(['odor', 'trial', 'time'])
    dff = dff.reindex(odors, level='odor')
    return dff


def simulation_mat_to_df(spike_timing, dt, num_neurons, num_steps, num_trials, odors):
    dff = spike_timing_to_rate(spike_timing, dt, num_neurons, num_steps)
    dff = specify_odor_and_trial(dff, odors, num_trials)
    return dff
