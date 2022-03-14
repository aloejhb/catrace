import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pymongo
import gridfs
from sklearn import decomposition

from . import pattern_correlation as pcr
from . import process_time_trace as ptt
from . import dataio
from . import frame_time


def plot_exp_pattern_correlation(dfovf_cut, odor_list, frame_rate,
                                 time_window=[5.3,7.3], ax=None):
    pat = ptt.mean_pattern_in_time_window(dfovf_cut, time_window, frame_rate)

    pat = pat.reset_index()
    pat['odor'] = pd.Categorical(pat.odor, ordered=True,
                                 categories=odor_list)
    pat = pat.sort_values('odor')

    if ax is None:
        fig, ax = plt.subplots()
    im = pcr.plot_pattern_correlation(pat, ax, clim=(0, 1))


def load_dfovf(exp_name, region_name, data_root_dir):
    plane_nb_list = np.array([1,2,3,4]) - 1
    num_trial = 3
    odor_list = ['phe', 'trp', 'arg', 'tdca', 'tca', 'gca', 'acsf', 'spont']
    frame_rate = 30/4
    # Load data
    exp_subdir = os.path.join(exp_name, region_name)
    tracedf = dataio.load_trace_file(data_root_dir, exp_subdir, plane_nb_list, num_trial, odor_list)

    # Cut first X second to exclude PMT off period
    cut_time = 5+2
    cut_win = frame_time.convert_sec_to_frame([cut_time, 40], frame_rate)
    tracedf = ptt.cut_tracedf(tracedf, cut_win[0], 0, cut_win[1])

    # Calculate dF/F
    fzero_twindow = np.array([7, 9])
    dfovf = ptt.compute_dfovf(tracedf, fzero_twindow, frame_rate, intensity_offset=-10)

    # Detect or set response onset
    xwindow = ((np.array([15, 20])-cut_time)*frame_rate).astype('int')
    onset_param = dict(thresh=0.013, sigma=3, xwindow=xwindow)
    dfovf_avg, y, dy = ptt.detect_tracedf_onset(dfovf, onset_param, debug=True)
    dfovf_avg['onset'] = 70
    # dfovf_avg['onset'] = dfovf_avg['onset'].astype('int32')


    # Align time traces to onset
    pre_time = 4
    post_time = 18
    dfovf_cut = ptt.align_tracedf(dfovf, dfovf_avg['onset'],
                                  pre_time, post_time, frame_rate)
    # dfovf_restack = ptt.restack_as_pattern(dfovf_cut)
    return dfovf_cut

def plot_explist(data_list, plot_func, sharex, sharey, *args, **kwargs):
        ncol = 5
        nrow = int(np.ceil(len(data_list) / ncol))
        figsize=[17, 3.4*nrow]
        fig, axes = plt.subplots(nrow, ncol, sharex=sharex,
                                 sharey=sharey, figsize=figsize)
        for idx, data in enumerate(data_list):
            ax = axes.flatten()[idx]
            plot_func(data, *args, **kwargs, ax=ax)
        plt.tight_layout()
        return fig


def plot_explist_decorator(plot_func, csplus_dict, data_dict, sharex=False, sharey=False):
    def plot_explist_wrapper(csplus, region, *args, **kwargs):
        csexp_list = csplus_dict[csplus]
        data_list = [data_dict[region][exp] for exp in csexp_list]
        return plot_explist(data_list, plot_func, sharex, sharey, *args, **kwargs)
    return plot_explist_wrapper



def get_data_dict_decorator(exp_list, region_list, dfovf_dict, data_func):
    def get_data_dict(*args, **kwargs):
        data_dict = dict()
        for region in region_list:
            data_dict[region] = dict()
            for exp in exp_list:
                exp_name = exp[0]
                print(exp_name, region)
                dfovf = dfovf_dict[region][exp_name]
                data_dict[region][exp_name] = data_func(dfovf,
                                                        *args, **kwargs)
        return data_dict
    return get_data_dict


def process_data_dict_decorator(data_func, exp_list, region_list,
                                db_dir, in_collect_name):
    def get_data_dict(*args, **kwargs):
        data_dict = dict()
        for exp in exp_list:
            exp_name = exp[0]
            data_dict[exp_name] = dict()
            for region in region_list:
                print(exp_name, region)
                df = read_df(in_collect_name, exp_name, region, db_dir)
                data_dict[exp_name][region] = data_func(df, *args, **kwargs)
        return data_dict
    return get_data_dict


def process_data_db_decorator(data_func, exp_list, region_list,
                              out_collect_name, db_dir, in_collect_name=None):
    def process_data_db(*args, **kwargs):
        for exp in exp_list:
            exp_name = exp[0]
            for region in region_list:
                print(exp_name, region)
                if in_collect_name:
                    df = read_df(in_collect_name, exp_name, region, db_dir)
                    outdf = data_func(df, *args, **kwargs)
                else:
                    outdf = data_func(exp_name, region, *args, **kwargs)
                update_df(outdf, out_collect_name, exp_name, region, db_dir)
    return process_data_db


def read_df(collect_name, exp_name, region, db_dir):
    filename = get_filename(exp_name, region, 'pkl')
    df_file = os.path.join(db_dir, collect_name, filename)
    df = pd.read_pickle(df_file)
    return df

def update_df(df, collect_name, exp_name, region, db_dir):
    collect_dir = os.path.join(db_dir, collect_name)
    if not os.path.exists(collect_dir):
        os.mkdir(collect_dir)
    filename = get_filename(exp_name, region, 'pkl')
    df_file = os.path.join(collect_dir, filename)
    df.to_pickle(df_file)


def get_filename(exp_name, region, ext):
    return '{}_{}.{}'.format(exp_name, region, ext)


def concatenate_df():
    # all_dfovf_select = pd.concat(plist, keys=dfovf_select_dict[region].keys(), names=['fish_id'])
    pass
