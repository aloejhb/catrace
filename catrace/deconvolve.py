import os
import sys
import numpy as np
import scipy.io as sio
import ruamel.yaml as yaml

from cascade2p import cascade


def compute_noise_levels(traces_df, sampling_rate,
                         baseline_window=None):
    if baseline_window:
        time_indices = traces.index.get_level_values('time')
        baseline = traces[time_indices >= baseline_window[0] &
                          time_indices <= baseline_window[1]]
    else:
        baseline = traces

    nl_dfs = []
    for (odor, trial), subset in baseline.groupby(level=('odor', 'trial')):
        noise_levels = utils.calculate_noise_levels(subset.T, sampling_rate)
        nl_df = pd.DataFrame(noise_levels.reshape(1, -1), columns=subset.columns)
        nl_df.index = pd.MultiIndex.from_tuples([(odor, trial)],
                                                names=['odor', 'trial'])
        nl_dfs.append(nl_df)

    noise_level_df = pd.concat(nl_dfs)
    return noise_level_df


def deconvolve_experiment(traces_df, model_name, model_folder):
    # Compute noise level for each neuron
    noise_levels = compute_noise_levels(traces_df)

    sp_dfs = []
    # Loop through each odor and trial combination
    for (odor, trial), subset in result.groupby(level=('odor', 'trial')):
        traces = subset.to_numpy().T
        # Predict spike prob
        spike_prob = cascade.predict(model_name, traces,
                                     trace_noise_levels=noise_levels,
                                     model_folder=model_folder,
                                     verbosity=False)
        import pdb; pdb.set_trace()
        sp_df = pd.DataFrame(spike_prob.T, index=subset.index, columns=subset.columns)
        sp_dfs.append(sp_df)

    # Concatenate and return the spike probabilities
    spike_prob_df = pd.concat(sp_dfs)
    return spike_prob_df
