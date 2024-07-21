import os
import sys
import numpy as np
import pandas as pd
import scipy.io as sio
import ruamel.yaml as yaml

from cascade2p import cascade, utils
from catrace.process_time_trace import restack_as_pattern
from dataclasses import dataclass
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class DeconvolveConfig:
    model_folder: str
    model_name: str
    baseline_window: list[int, int] # in terms of frames
    sampling_rate: float


def compute_noise_levels(trace_df, sampling_rate,
                         baseline_window=None):
    if baseline_window is not None:
        time_indices = trace_df.index.get_level_values('time')
        baseline = trace_df[(time_indices >= baseline_window[0]) &
                            (time_indices <= baseline_window[1])]
    else:
        baseline = trace_df

    nl_dfs = []
    for (odor, trial), subset in baseline.groupby(level=('odor', 'trial')):
        noise_levels = utils.calculate_noise_levels(subset.T, sampling_rate)
        nl_df = pd.DataFrame(noise_levels.reshape(1, -1), columns=subset.columns)
        nl_df.index = pd.MultiIndex.from_tuples([(odor, trial)],
                                                names=['odor', 'trial'])
        nl_dfs.append(nl_df)

    noise_level_df = pd.concat(nl_dfs)
    mean_noise_levels = noise_level_df.mean(axis=0).to_numpy()
    return mean_noise_levels


def deconvolve_experiment(trace_df, model_name, model_folder,
                          sampling_rate, baseline_window, restack=False):
    if restack:
        trace_df = restack_as_pattern(trace_df)
    # Compute noise level for each neuron
    noise_levels = compute_noise_levels(trace_df, sampling_rate, baseline_window)

    sp_dfs = []
    # Loop through each odor and trial combination
    for (odor, trial), subset in trace_df.groupby(level=('odor', 'trial')):
        traces = subset.to_numpy().T
        # Predict spike prob
        spike_prob = cascade.predict(model_name, traces,
                                     trace_noise_levels=noise_levels,
                                     model_folder=model_folder,
                                     verbosity=False)
        sp_df = pd.DataFrame(spike_prob.T, index=subset.index, columns=subset.columns)
        sp_dfs.append(sp_df)

    # Concatenate and return the spike probabilities
    spike_prob_df = pd.concat(sp_dfs)
    return spike_prob_df
