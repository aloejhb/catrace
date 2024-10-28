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
from ..stats import pool_training_conditions
from ..similarity import (compute_similarity_mat, cosine_distance,
                          pattern_correlation, plot_similarity_mat, 
                          extract_upper_triangle_similarities,
                          compute_diff_to_naive,
                          plot_mean_delta_mat, PlotMeanDeltaMatParams,
                          average_mat_over_trials)
from ..process_time_trace import select_odors_and_sort
from ..visualize import plot_conds_mat, PlotPerCondMatParams, plot_measure_multi_odor_cond, PlotBoxplotMultiOdorCondParams

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


def plot_matrix_per_cond(simdf_list, exp_cond_list, conditions, params: PlotPerCondMatParams):
    avg_mats = mean_mat_over_cond(simdf_list, exp_cond_list, conditions)
    cmin = min([mat.min().min() for mat in avg_mats.values()])
    cmax = max([mat.max().max() for mat in avg_mats.values()])
    if params.clim is None:
        params.clim = (cmin, cmax)
    fig, axes = plot_conds_mat(avg_mats, conditions, plot_similarity_mat, **params.to_dict())
    return fig


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


def concatenate_simdfs(simdf_lists, exp_list):
    all_simdf = pd.concat(simdf_lists, keys=exp_list, names=['fish_id', 'condition'])
    return all_simdf


from .run_utils import get_vs_tuple, get_group_vs_group


def pool_odor_pair(group1, group2, selected_conditions, all_simdf, measure_name, naive_name='naive', deduplicate=False):

    subsimdf = get_group_vs_group(all_simdf, group1, group2, measure_name=measure_name, deduplicate=deduplicate)
    # Select conditions
    subsimdf = subsimdf[subsimdf.index.get_level_values('condition').isin(selected_conditions)]

    # Map naive to naive, others to trained
    condition_map = {cond: 'trained' if cond != naive_name else 'naive' for cond in selected_conditions}
    pooled_subsimdf = pool_training_conditions(subsimdf, condition_map)

    return pooled_subsimdf


def compute_multi_vs(vsnames, dsconfig, conditions, all_simdf, metric, naive_name):
    vsdict = {vsname: get_vs_tuple(dsconfig, vsname) for vsname in vsnames}

    subsimdfs = {}
    for vsname, (group1, group2) in vsdict.items():
        try:
            pooled_subsimdf = pool_odor_pair(group1, group2, conditions, all_simdf, metric, naive_name=naive_name)
        except Exception as err:
            print(f'Error in {vsname}')
            raise err
        subsimdfs[vsname] = pooled_subsimdf

    vsdff = pd.concat(subsimdfs.values(), keys=subsimdfs.keys(),
                      names=['vsname'])
    return vsdff


from typing import Union

@dataclass_json
@dataclass
class PlotPatternSimilarityParams:
    per_cond: PlotPerCondMatParams = PlotPerCondMatParams()
    mean_delta: Union[PlotMeanDeltaMatParams, dict] = None
    vs_measure: PlotBoxplotMultiOdorCondParams = PlotBoxplotMultiOdorCondParams()

    # if mean_delta is a dict, it will be converted to PlotMeanDeltaMatParams
    def __post_init__(self):
        if isinstance(self.mean_delta, dict):
            self.mean_delta = PlotMeanDeltaMatParams(**self.mean_delta)


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
    do_reorder_cs: bool = False
    odor_orders: list = None
    naive_name: str = 'naive'
    plot_params: PlotPatternSimilarityParams = PlotPatternSimilarityParams()
    vsnames: list[str] = None


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
        fig_per_cond = plot_matrix_per_cond(simdf_list, exp_cond_list, dsconfig.conditions, params=params.plot_params.per_cond)

    print('Plotting delta matrix...')
    if params.do_reorder_cs:
        mean_delta_mat = compute_diff_to_naive(simdf_list, exp_cond_list,params.do_reorder_cs, params.odor_orders, dsconfig.odors_aa, naive_name=params.naive_name)
    else:
        mean_delta_mat = compute_diff_to_naive(simdf_list, exp_cond_list, do_reorder_cs=params.do_reorder_cs, naive_name=params.naive_name)
    print(params.plot_params.mean_delta)
    fig_delta, ax = plot_mean_delta_mat(mean_delta_mat, params.plot_params.mean_delta)

    avgsimdf_list = [average_mat_over_trials(simdf) for simdf in simdf_list]
    all_simdf = concatenate_simdfs(avgsimdf_list, exp_list)

    output_figs = {}
    output_figs['fig_per_cond'] = fig_per_cond
    output_figs['fig_delta'] = fig_delta

    if params.vsnames is not None:
        vsdff = compute_multi_vs(params.vsnames, dsconfig, dsconfig.conditions, all_simdf, params.metric, naive_name=params.naive_name)
        fig_multi_vs, ax, test_results = plot_measure_multi_odor_cond(vsdff, params.metric, odor_name='vsname', condition_name='condition', params=params.plot_params.vs_measure)
        output_figs['fig_multi_vs'] = fig_multi_vs

    if params.do_save_cross_trial:
        cross_trial_df = extract_cross_trial_similarity(simdf_list, exp_list)
        cross_trial_path = save_cross_trial_similarity(cross_trial_df, sim_dir)
        return sim_dir, output_figs, test_results, cross_trial_path
    
    return sim_dir, output_figs, test_results


