import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from os.path import join as pjoin
from dataclasses import dataclass
from dataclasses_json import dataclass_json

from ..dataset import DatasetConfig
from ..utils import load_config
from ..exp_collection import (process_data_db_decorator, read_df, plot_explist_with_cond, mean_mat_over_cond)
from ..similarity import (compute_similarity_mat, cosine_distance,
                          pattern_correlation, plot_similarity_mat, 
                          extract_upper_triangle_similarities)
from ..process_time_trace import select_odors_and_sort
from ..visualize import plot_conds_mat

from .run_utils import plot_avg_trace_with_window


@dataclass_json
@dataclass
class ComputeSimilarityParams:
    in_dir: str
    exp_list: list[tuple[str, str]]
    metric: str
    time_window: list[int]
    odors: list[str]
    overwrite_computation: bool


def compute_similarity(params: ComputeSimilarityParams):
    in_dir = params.in_dir
    metric = params.metric
    if metric == 'cosine':
        similarity_func = cosine_distance
    elif metric == 'pattern_correlation':
        similarity_func = pattern_correlation

    time_window = np.array(params.time_window)

    out_dir = pjoin(in_dir, metric, f'window_{time_window[0]}to{time_window[1]}')

    def select_odors_and_compute_similarity(dff, odors, **kwargs):
        dff = select_odors_and_sort(dff, odors)
        dff = compute_similarity_mat(dff, **kwargs)
        return dff

    if not os.path.exists(out_dir) or params.overwrite_computation:
        os.makedirs(out_dir, exist_ok=True)
        func = select_odors_and_compute_similarity
        compute_pcr_explist = process_data_db_decorator(func, 
                                                        params.exp_list,
                                                        out_dir, in_dir)
        compute_pcr_explist(odors=params.odors,
                            time_window=params.time_window,
                            frame_rate=1,
                            similarity_func=similarity_func)
    return out_dir


def read_mats_from_dir(in_dir, exp_list):
    simdf_list = [read_df(in_dir, exp[0]) for exp in exp_list]
    exp_cond_list = [exp[1] for exp in exp_list]
    return simdf_list, exp_cond_list


def plot_matrix_per_fish(simdf_list, exp_cond_list):
    cmin = min([mat.min().min() for mat in simdf_list])
    cmax = max([mat.max().max() for mat in simdf_list])
    fig, axes = plot_explist_with_cond(simdf_list, exp_cond_list, plot_similarity_mat, clim=(cmin, cmax), cmap='turbo')
    ax = axes[-1, 0]
    img = ax.get_children()[0]
    fig.colorbar(img, ax=ax)
    return fig, axes


def plot_matrix_per_cond(simdf_list, exp_cond_list, conditions):
    avg_mats = mean_mat_over_cond(simdf_list, exp_cond_list, conditions)
    cmin = min([mat.min().min() for mat in avg_mats.values()])
    cmax = max([mat.max().max() for mat in avg_mats.values()])
    fig, axes = plot_conds_mat(avg_mats, conditions, plot_similarity_mat, clim=(cmin, cmax), cmap='turbo', ncol=2, ylabel_fontsize=12)
    return fig, axes


def extract_cross_trial_similarity(simdf_list, exp_list):
    final_df_list = []
    for idx, simdf in enumerate(simdf_list):
        final_df = extract_upper_triangle_similarities(simdf)
        final_df_list.append(final_df)

    cross_trial_df = pd.concat(final_df_list, keys=exp_list, names=['fish_id', 'condition'])
    cross_trial_df.index = cross_trial_df.index.droplevel(-1)
    return cross_trial_df


def save_cross_trial_similarity(cross_trial_df, out_dir):
    cross_trial_path = pjoin(out_dir, f'cross_trial_similarity.pkl')
    cross_trial_df.to_pickle(cross_trial_path)
    return cross_trial_path


@dataclass_json
@dataclass
class RunPatternSimilarityParams:
    config_file: str
    assembly_name: str
    time_window: list[int]
    metric: str
    overwrite_computation: bool = False
    do_plot_per_fish: bool = False
    do_plot_per_condition: bool = False
    do_save_cross_trial: bool = False

def run_pattern_similarity(params: RunPatternSimilarityParams):
    dsconfig= load_config(params.config_file, DatasetConfig)
    exp_list = dsconfig.exp_list
    trace_dir = dsconfig.processed_trace_dir
    select_neuron_dir = pjoin(trace_dir, params.assembly_name)
    time_window = np.array(params.time_window)
    in_dir = select_neuron_dir
    
    fig_avg_trace, ax = plot_avg_trace_with_window(in_dir, exp_list[0][0], time_window)

    sim_params = ComputeSimilarityParams(in_dir=in_dir,
                                         exp_list=exp_list,
                                         metric=params.metric, 
                                         time_window=params.
                                         time_window,
                                         odors=dsconfig.odors_stimuli,
                                         overwrite_computation=params.overwrite_computation)
    sim_dir = compute_similarity(sim_params)
    
    simdf_list, exp_cond_list = read_mats_from_dir(sim_dir, exp_list)

    if params.do_plot_per_fish:
        plot_matrix_per_fish(simdf_list, exp_cond_list)

    if params.do_plot_per_condition:
        plot_matrix_per_cond(simdf_list, exp_cond_list, dsconfig.conditions)

    if params.do_save_cross_trial:
        cross_trial_df = extract_cross_trial_similarity(simdf_list, exp_list)
        cross_trial_path = save_cross_trial_similarity(cross_trial_df, sim_dir)
        return sim_dir, cross_trial_path
    
    return sim_dir



