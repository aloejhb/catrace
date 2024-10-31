from .similarity import cosine_distance, pattern_correlation, cosine_distance_to_template, pattern_correlation_to_template
from .exp_collection import read_df

from matplotlib.colors import Normalize




def compute_trial_similarity_over_time(trial_traces, bin_size=None, similarity_method='pattern_correlation'):
    # Bin the trial traces if bin_size is specified
    if bin_size is not None:
        trial_traces = trial_traces.groupby(trial_traces.index // bin_size).mean()
        frame_range = np.array(frame_range) // bin_size

    # Compute the similarity matrix based on the chosen method
    if similarity_method == 'cosine':
        trial_similarity = cosine_distance(trial_traces.to_numpy())
    elif similarity_method == 'pattern_correlation':
        trial_similarity = pattern_correlation(trial_traces.to_numpy())
    else:
        raise ValueError("Invalid similarity_method. Choose 'cosine' or 'pattern_correlation'.")

    return trial_similarity


from matplotlib.colors import PowerNorm


def plot_correlation_over_time_subplots(xvec, trial_traces_mean, mat, similarity_method='pattern_correlation', clim=(0, 1),
                                        cmap='magma', power_norm=1.2, figsize=(5, 5)):
    """
    Plots the correlation over time with a line plot above a similarity matrix plot.
    Both plots share the same x-axis and have the same widths.

    Parameters:
    - xvec: array, the x-axis vector
    - trial_traces_mean: Series, the mean spike probability over time
    - mat: array, the similarity matrix
    - similarity_method: str, the similarity method used, either 'cosine' or 'pattern_correlation' (for labeling)
    - clim: tuple, the color limits for the similarity matrix (default is (0, 1))
    - cmap: str, the colormap for the similarity matrix (default is 'magma')
    - power_norm: float, the power normalization factor for the similarity matrix (default is 1.2)
    - figsize: tuple, the figure size (default is (5, 5))

    Returns:
    - fig, (ax_line, ax_mat): The figure and axes objects of the plot
    """
    # Create subplots with shared x-axis
    fig, (ax_line, ax_mat) = plt.subplots(2, 1, figsize=figsize, sharex=True,
                                          gridspec_kw={'height_ratios': [1, 4]}, constrained_layout=True)

    # Line plot on the top
    ax_line.plot(xvec, trial_traces_mean)
    ax_line.set_ylabel('Mean spike probability', fontsize=7)
    ax_line.tick_params(labelbottom=False)

    # Matshow plot below
    img = ax_mat.imshow(mat, interpolation='none', cmap=cmap, norm=PowerNorm(power_norm),
                        clim=clim, aspect='auto')

    # Set labels
    ax_mat.set_xlabel('Frames')
    ax_mat.set_ylabel('Frames')

    # Add colorbar without affecting the width
    cbar = fig.colorbar(img, ax=ax_mat, pad=0.02, fraction=0.046)
    clabel = 'Cosine Distance' if similarity_method == 'cosine' else 'Pattern Correlation'
    cbar.set_label(clabel, fontsize=7)

    return fig, (ax_line, ax_mat)

from dataclasses import dataclass
from dataclasses_json import dataclass_json

@dataclass_json
@dataclass
class SimToTemplateParams:
    bin_size: int = None
    similarity_method: str = 'pattern_correlation'
    template_frame_range: tuple = None


def compute_trial_similarity_over_time_to_template(trial_traces, bin_size=None, similarity_method='pattern_correlation',
                                                   template=None, template_frame_range=None):
    # Bin the trial traces if bin_size is specified
    if bin_size is not None:
        trial_traces = trial_traces.groupby(trial_traces.index // bin_size).mean()

    times = trial_traces.index.get_level_values('time')
    idx = (times >= template_frame_range[0]) & (times <= template_frame_range[1])
    template = trial_traces.loc[idx, :].mean()

    # Compute the similarity matrix based on the chosen method
    if similarity_method == 'cosine':
        trial_similarity = cosine_distance_to_template(trial_traces.to_numpy(), template)
    elif similarity_method == 'pattern_correlation':
        trial_similarity = pattern_correlation_to_template(trial_traces.to_numpy(), template)
    else:
        raise ValueError("Invalid similarity_method. Choose 'cosine' or 'pattern_correlation'.")

    time_idx = trial_traces.index
    # Convert trial_similarity to a DataFrame with the row and column index to be the time index
    trial_similarity = pd.DataFrame(trial_similarity, index=time_idx)
    return trial_similarity


import pandas as pd
import numpy as np

def compute_fish_similarity_over_time(dsconfig, fish_id, odor_trial, bin_size, similarity_method):
    dff = read_df(dsconfig.processed_trace_dir, fish_id)
    trial_traces = dff.xs(odor_trial, level=('odor', 'trial'))
    trial_similarity = compute_trial_similarity_over_time(trial_traces, bin_size=bin_size, similarity_method=similarity_method)
    time_idx = trial_traces.index
    # Convert trial_similarity to a DataFrame with the row and column index to be the time index
    trial_similarity = pd.DataFrame(trial_similarity, index=time_idx, columns=time_idx)
    return trial_similarity


def get_slicing_from_trial_similarity(trial_similarity, slicing_frame):
    return trial_similarity[slicing_frame, :]


def compute_population_trial_similarity_over_time(dsconfig, odor_trial, exp_list, bin_size, similarity_method):
    trial_similarities = []
    for exp_name, condition in exp_list:
        trial_similarity = compute_fish_similarity_over_time(dsconfig, exp_name, odor_trial, bin_size, similarity_method)
        trial_similarities.append(trial_similarity)

    simdf = pd.concat(trial_similarities, keys=exp_list, names=['fish_id', 'condition'])
    return simdf

def compute_population_multitrial_similarity_over_time(dsconfig, odor_trials, exp_list, bin_size, similarity_method):
    simdfs = []
    for odor_trial in odor_trials:
        simdf = compute_population_trial_similarity_over_time(dsconfig, odor_trial, exp_list, bin_size, similarity_method)
        simdfs.append(simdf)
    multi_trial_simdf = pd.concat(simdfs, keys=odor_trials, names=['odor', 'trial'])
    return multi_trial_simdf



# Plot the average trial_similarity for each condition
# Plot matrix per condition
from catrace.run.run_distance import get_mat_lists
from catrace.exp_collection import mean_mat_over_cond
from catrace.visualize import plot_conds_mat

from matplotlib.colors import PowerNorm

def select_frame_range_mat(avg_simdf, frame_range):
    times = avg_simdf.index.get_level_values('time')
    idx = (times >= frame_range[0]) & (times <= frame_range[1])
    avg_simdf = avg_simdf.loc[idx]
    times_column = avg_simdf.columns.get_level_values('time')
    idx = (times_column >= frame_range[0]) & (times_column <= frame_range[1])
    avg_simdf = avg_simdf.loc[:, idx]
    return avg_simdf


@dataclass_json
@dataclass
class PlotTrialSimilarityParams:
    cmap: str = 'magma'
    clim: tuple = None
    color_norm: Normalize = None
    frame_rate: float = None
    figsize: tuple = (5, 5)
    label_fontsize: int = 7
    tick_fontsize: int = 6


def plot_trial_similarity_mat(df,
                              ax=None,
                              cmap='magma',
                              clim=None,
                              color_norm: Normalize = None,
                              frame_rate: float = None,
                              figsize=(5, 5),
                              label_fontsize=7,
                              tick_fontsize=6,
                              ):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    im = ax.imshow(df.to_numpy(), cmap=cmap, norm=color_norm, interpolation='none', clim=clim)

    if frame_rate is not None:
        # Convert frame to time in seconds# Convert x axis from frame to time
        interval_in_sec = 1
        xticks = np.arange(0, df.shape[1], interval_in_sec * frame_rate)
        ax.set_xticks(xticks)
        ax.set_yticks(xticks)
        xticklabels = xticks / frame_rate
        ax.set_xticklabels(xticklabels, fontsize=tick_fontsize)
        ax.set_yticklabels(xticklabels, fontsize=tick_fontsize)

    if frame_rate is None:
        xlabel = 'Frames'
    else:
        xlabel = 'Time (s)'
    ax.set_xlabel(xlabel, fontsize=label_fontsize)
    return fig

@dataclass_json
@dataclass
class PlotTrialSimilarityPerCondParams:
    ncol: int = 2
    row_height: float = 2
    col_width: float = 2
    colorbar_fontsize: int = 6
    plot_params: PlotTrialSimilarityParams = PlotTrialSimilarityParams()


def plot_matrix_per_condition(avg_simdf, conditions, frame_range=None, params: PlotTrialSimilarityPerCondParams = PlotTrialSimilarityPerCondParams()):
    if frame_range is not None:
        avg_simdf = select_frame_range_mat(avg_simdf, frame_range)

    simdf_list, exp_cond_list = get_mat_lists(avg_simdf)
    avg_mats = mean_mat_over_cond(simdf_list, exp_cond_list, conditions)
    # Average each avg_mats so that only time is left for index
    avg_mats = {cond: mat.groupby('time').mean() for cond, mat in avg_mats.items()}

    if params.plot_params.clim is None:
        cmin = min([mat.min().min() for mat in avg_mats.values()])
        cmax = max([mat.max().max() for mat in avg_mats.values()])
        clim = (cmin, cmax)
        params.plot_params.clim = clim
    # Pop the plot_params from params

    params_dict = params.to_dict()
    plot_params = params_dict.pop('plot_params')
    fig, axs = plot_conds_mat(avg_mats, conditions, plot_trial_similarity_mat, **params_dict, **plot_params)
    return fig, axs


import pandas as pd
import numpy as np

def compute_fish_similarity_over_time_to_template(dsconfig, fish_id, odor_trial, sim_params: SimToTemplateParams):
    dff = read_df(dsconfig.processed_trace_dir, fish_id)
    trial_traces = dff.xs(odor_trial, level=('odor', 'trial'))
    sim_to_template = compute_trial_similarity_over_time_to_template(trial_traces, **sim_params.to_dict())
    #time_idx = trial_traces.index
    # Convert trial_similarity to a DataFrame with the row and column index to be the time index
    #trial_similarity = pd.DataFrame(trial_similarity, index=time_idx, columns=time_idx)
    return sim_to_template


def compute_population_trial_similarity_over_time_to_template(dsconfig, odor_trial, exp_list, sim_params):
    sim_to_templates = []
    for exp_name, condition in exp_list:
        sim_to_template = compute_fish_similarity_over_time_to_template(dsconfig, exp_name, odor_trial, sim_params)
        sim_to_templates.append(sim_to_template.T)

    simtempdf = pd.concat(sim_to_templates, keys=exp_list, names=['fish_id', 'condition'])
    return simtempdf


def compute_population_multitrial_similarity_over_time_to_template(dsconfig, odor_trials, exp_list, sim_params):
    simtempdfs = []
    for odor_trial in odor_trials:
        simtempdf = compute_population_trial_similarity_over_time_to_template(dsconfig, odor_trial, exp_list, sim_params)
        simtempdfs.append(simtempdf)
    multiodor_simtempdfs = pd.concat(simtempdfs, keys=odor_trials, names=['odor', 'trial'])
    return multiodor_simtempdfs


# Plot slicing per condition
from catrace.visualize import plot_conds
from catrace.plot_trace import plot_mean_with_std

def select_frame_range(simdf_slicing, frame_range):
    times = simdf_slicing.index.get_level_values('time')
    idx = (times >= frame_range[0]) & (times <= frame_range[1])
    simdf_slicing = simdf_slicing.loc[idx]
    return simdf_slicing


def plot_trial_similarity_slicing(simdf_slicing, metric, ax=None, color='blue', line_label=None, frame_range=None, frame_rate=None, **kwargs):
    if line_label is None:
        line_label = metric
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    if frame_range is not None:
        simdf_slicing = select_frame_range(simdf_slicing, frame_range)


    if frame_rate is not None:
        start_time_with_zero = True
    else:
        start_time_with_zero = False
    plot_mean_with_std(simdf_slicing, ax=ax, label=line_label, color=color, frame_rate=frame_rate, start_time_with_zero=start_time_with_zero, **kwargs)

import matplotlib.pyplot as plt
def plot_slicing_per_condition(simtempdf, metric, colors, frame_range=None, frame_rate=None, ax=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    for condition, group in simtempdf.groupby('condition'):
        plot_trial_similarity_slicing(group.T, metric, ax=ax, color=colors[condition], line_label=condition, frame_range=frame_range, frame_rate=frame_rate, **kwargs)
        ax.legend()
    ax.set_ylabel(metric)
    return fig
