import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from os.path import join as pjoin
from dataclasses import dataclass
from dataclasses_json import dataclass_json

from ..dataset import load_dataset_config
from ..exp_collection import (read_df, plot_explist_with_cond, concatenate_df_from_db)
from ..process_time_trace import select_odors_and_sort
from ..plot_trace import plot_trace_avg


@dataclass_json
@dataclass
class RunAverageTimecourseParams:
    config_file: str
    assembly_name: str
    do_plot_per_fish: bool = False
    figsize: tuple = (2, 3)
    label_fontsize: int = 7
    legend_fontsize: int = 6
    linewidth: int = 1
    cut_time: float = 5
    dff: pd.DataFrame = None
    odor_colors: dict = None
    alpha: float = 1

def run_average_timecourse(params: RunAverageTimecourseParams):
    dsconfig= load_dataset_config(params.config_file)
    exp_list = dsconfig.exp_list
    trace_dir = dsconfig.processed_trace_dir
    select_neuron_dir = pjoin(trace_dir, params.assembly_name)
    in_dir = select_neuron_dir
    

    if params.do_plot_per_fish:
        dffs = [read_df(in_dir, exp[0]) for exp in exp_list]
        exp_cond_list = [exp[1] for exp in exp_list]
        fig_per_fish, axs = plot_explist_with_cond(dffs, exp_cond_list, plot_trace_avg,
                                           frame_rate=1, yname='spike_rate',
                                           sharex=True, sharey=True)

    if params.dff is None:
        dff = concatenate_df_from_db(in_dir, exp_list)
        dff = select_odors_and_sort(dff, dsconfig.odors_stimuli)
    else:
        dff = params.dff

    # Get figsize, label_fontsize, legend_fontsize, linewidth from params into a dictionary
    params_dict = params.to_dict()
    sub_params = {k: params_dict[k] for k in ['figsize', 'label_fontsize', 'legend_fontsize', 'linewidth', 'odor_colors']}

    fig_frame, ax = plot_trace_avg(dff, frame_rate=None, cut_time=0, show_legend=True,
                                   **sub_params)

    fig_time, ax = plot_trace_avg(dff, frame_rate=dsconfig.frame_rate, cut_time=params.cut_time, show_legend=True,
                   **sub_params)

    if params.do_plot_per_fish:
        return fig_per_fish, fig_frame, fig_time

    return fig_frame, fig_time, dff