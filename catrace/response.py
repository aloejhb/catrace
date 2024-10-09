import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from dataclasses import dataclass
from dataclasses_json import dataclass_json

from .dataset import DatasetConfig, get_odors_by_key
from .exp_collection import concatenate_df_from_db
from .process_time_trace import (select_odors_and_sort,
                                mean_pattern_in_time_window,
                                average_over_trials)
from .visualize import plot_measure_by_cond, PlotBoxplotByCondParams


def group_and_flatten_responses(resp):
    resp_grouped = resp.T.groupby(level='condition', sort=False)
    respli = []
    for cond, group in resp_grouped:
        values = np.ravel(group.values)
        for resp in values:
            respli.append((cond, resp))

    resp_df = pd.DataFrame(respli, columns=['condition', 'response'])
    resp_df.set_index('condition', inplace=True)
    return resp_df


def normalize_responses(resp_df):
    # resp_df has one level index 'condtion'
    # select the 'naive' condition and normalize all responses to it
    baseline = resp_df.xs('naive').response.median()
    resp_df['normalized_response'] = (resp_df.response - baseline)/ baseline * 100
    return resp_df


def compute_response(dff, time_window, odors):
    mean_pattern = mean_pattern_in_time_window(dff, time_window)
    mean_pattern = select_odors_and_sort(mean_pattern, odors)
    mean_pattern = average_over_trials(mean_pattern)
    return mean_pattern


def select_top_neurons_per_cond(mean_pattern, top_ratio):
    max_resp = mean_pattern.max(axis=0)
    top_indices_list = []
    for condition, group in max_resp.groupby(level=0):
        top_indices = group.nlargest(int(len(group)*top_ratio)).index
        top_indices_list.append(top_indices)

    selected_indices = pd.Index([])
    for indices in top_indices_list:
        selected_indices = selected_indices.union(indices)

    mean_pattern_top = mean_pattern.loc[:, selected_indices]
    return mean_pattern_top


def plot_hist_by_cond(resp_df, value_name,
                      binwidth=None,
                      log_scale=False,
                      figsize=(2, 2),
                      label_fontsize=7,
                      tick_label_fontsize=6,
):
    fig, ax = plt.subplots(figsize=figsize)
    # Plot histplot with defined bin size
    sns.histplot(ax=ax, data=resp_df, hue='condition', x=value_name, element="poly", stat="density", common_norm=False, fill=False, binwidth=binwidth)
    if log_scale:
        ax.set_yscale('log')
    ax.set_xlabel(value_name, fontsize=label_fontsize)
    ax.set_ylabel('density', fontsize=label_fontsize)
    ax.tick_params(axis='both', which='major', labelsize=tick_label_fontsize)
    # Retrieve the automatically generated seaborn legend and reposition it
    legend = ax.get_legend()
    if legend:
        legend.set_bbox_to_anchor((1, 0.5))  # Move it outside the plot
        legend.set_loc('center left')  # Set location
        # Adjust font size for legend entries
        for text in legend.get_texts():
            text.set_fontsize(tick_label_fontsize)
        # Adjust font size for legend title
        legend.get_title().set_fontsize(tick_label_fontsize)

    fig.tight_layout()
    return fig, ax


def compute_response_flattened(dff, time_window, odors, top_ratio=None):
    time_window = np.array(time_window)
    mean_pattern = compute_response(dff, time_window, odors)
    if top_ratio is not None:
        mean_pattern = select_top_neurons_per_cond(mean_pattern, top_ratio)
    resp = group_and_flatten_responses(mean_pattern)
    return resp



def load_all_dff(dsconfig):
    in_dir = dsconfig.processed_trace_dir
    exp_list = dsconfig.exp_list
        
    dff = concatenate_df_from_db(in_dir, exp_list)
    dff = select_odors_and_sort(dff, dsconfig.odors_stimuli)
    return dff


@dataclass_json
@dataclass
class RunResponseParams:
    dff: pd.DataFrame
    dsconfig: DatasetConfig
    time_window: list[int]
    odor_key: str
    top_ratio: float = None
    boxplot_params: PlotBoxplotByCondParams = PlotBoxplotByCondParams()
    histplot_params: dict = None


def run_response(params: RunResponseParams):
    odors = get_odors_by_key(params.dsconfig, params.odor_key)
    resp = compute_response_flattened(params.dff, params.time_window, odors, params.top_ratio)
    # hline_y should be the mean of the naive condition
    hline_y = resp.xs('naive').response.mean()
    boxplot_params = params.boxplot_params
    boxplot_params.hline_y = hline_y
    fig_box, ax, test_results = plot_measure_by_cond(resp, 'response', params=params.boxplot_params)
    if params.histplot_params is None:
        params.histplot_params = {}
    fig_hist, ax = plot_hist_by_cond(resp, 'response', **params.histplot_params)
    return resp, test_results, fig_box, fig_hist


