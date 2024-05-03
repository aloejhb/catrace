import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import pickle
import re
from sklearn import decomposition
from functools import partial

from . import pattern_correlation as pcr
from . import process_time_trace as ptt
from . import dataio
from . import frame_time


def plot_exp_pattern_correlation(dfovf_cut, odor_list, frame_rate,
                                 time_window=[5.3,7.3], ax=None):
    pat = ptt.mean_pattern_in_time_window(dfovf_cut, time_window, frame_rate)
    pat = pat.reindex(odor_list, level='odor').reset_index()
    # pat['odor'] = pd.Categorical(pat.odor, ordered=True,
    #                              categories=odor_list)
    # pat = pat.sort_values('odor')
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
    cut_time = 12
    cut_win = frame_time.convert_sec_to_frame([cut_time, 40], frame_rate)
    tracedf = ptt.cut_tracedf(tracedf, cut_win)

    # Calculate dF/F
    fzero_twindow = np.array([0.5, 2.5])
    dfovf = ptt.compute_dfovf(tracedf, fzero_twindow, frame_rate, intensity_offset=-10)

    # # Detect or set response onset
    # xwindow = ((np.array([15, 20])-cut_time)*frame_rate).astype('int')
    # onset_param = dict(thresh=0.013, sigma=3, xwindow=xwindow)
    # dfovf_avg, y, dy = ptt.detect_tracedf_onset(dfovf, onset_param, debug=True)
    # dfovf_avg['onset'] = 70
    # # dfovf_avg['onset'] = dfovf_avg['onset'].astype('int32')


    # # Align time traces to onset
    # pre_time = 4
    # post_time = 18
    # dfovf_cut = ptt.align_tracedf(dfovf, dfovf_avg['onset'],
    #                               pre_time, post_time, frame_rate)
    # dfovf_restack = ptt.restack_as_pattern(dfovf_cut)
    return dfovf

def plot_explist(data_list, plot_func, sharex=False,
                 sharey=False, *args, **kwargs):
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


def plot_explist_with_cond(data_list, exp_cond_list, plot_func, sharex=False,
                           sharey=False, *args, **kwargs):
    total_exp = len(data_list)
    cond_list = list(dict.fromkeys(exp_cond_list))
    cond_count = [exp_cond_list.count(cond) for cond in cond_list]
    cond_cumsum = np.cumsum(cond_count)

    ncol = 5
    nrow_list = [int(np.ceil(ct/ncol)) for ct in cond_count]
    total_nrow = sum(nrow_list)

    axidx_list = [_get_axidx(k, cond_cumsum, nrow_list, ncol) for k in range(total_exp)]

    figsize=[10, 2*total_nrow]
    fig, axes = plt.subplots(total_nrow, ncol, sharex=sharex,
                             sharey=sharey, figsize=figsize)
    for idx, data in enumerate(data_list):
        axidx = axidx_list[idx]
        ax = axes.flatten()[axidx]
        plot_func(data, *args, **kwargs, ax=ax)
    plt.tight_layout()
    return fig, axes

def _get_axidx(k, cond_cumsum, nrow_list, ncol):
    ncond = sum(cond_cumsum<=k)
    if ncond == 0:
        shift = 0
        nrow_shift = 0
    else:
        shift = cond_cumsum[ncond-1]
        nrow_shift = np.sum(nrow_list[:ncond])
    return nrow_shift*ncol + k - shift


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
                              out_collect_name, in_collect_name=None, db_dir=''):
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


def process_data_model_decorator(data_func, exp_list, region_list,
                                out_collect_name, db_dir, in_collect_name=None):
    model_out_dir = os.path.join(db_dir, out_collect_name, 'models')
    if not os.path.exists(model_out_dir):
        os.mkdir(model_out_dir)
    def process_data_model(*args, **kwargs):
        for exp in exp_list:
            exp_name = exp[0]
            for region in region_list:
                print(exp_name, region)
                if in_collect_name:
                    df = read_df(in_collect_name, exp_name, region, db_dir)
                    outdf, model = data_func(df, *args, **kwargs)
                else:
                    outdf, model = data_func(exp_name, region, *args, **kwargs)
                update_df(outdf, out_collect_name, exp_name, region, db_dir)
                update_df(model, model_out_dir, exp_name, region, '')
    return process_data_model


def process_data_db_decorator_parallel(data_func, exp_list, region_list,
                                       out_collect_name, db_dir, in_collect_name=None,
                                       parallelism=1):

    def process_data_db(*args, **kwargs):
        func = partial(data_func, *args, **kwargs)
        def _process_data(exp_name, region):
            if in_collect_name:
                df = read_df(in_collect_name, exp_name, region, db_dir)
                outdf = func(df)
            else:
                outdf = func(exp_name, region)
            update_df(outdf, out_collect_name, exp_name, region, db_dir)

        exp_names = [e[0] for e in exp_list]
        exps = itertools.product(exp_names, region_list)
        with Pool(processes=parallelism) as pool:
            pool.map(_process_data, exps)

    return process_data_db


def process_data_db_decorator_dict(data_func, exp_list, region_list,
                                   db_dir, in_collect_name=None):
    def process_data_db(*args, **kwargs):
        result = dict()
        for exp in exp_list:
            exp_name = exp[0]
            result[exp_name] = dict()
            for region in region_list:
                print(exp_name, region)
                if in_collect_name:
                    df = read_df(in_collect_name, exp_name, region, db_dir)
                    outdf = data_func(df, *args, **kwargs)
                else:
                    outdf = data_func(exp_name, region, *args, **kwargs)
                result[exp_name][region] = outdf
        return result
    return process_data_db


def process_dataframe_decorator(data_func, level=['fish_id', 'cond'], axis=1):
    # if 'region' in df.columns.names:
    #     level = ['region', 'fish_id', 'cond']

    def process_dataframe(df, **kwargs):
        out_dataframe = df.groupby(level=level, axis=axis).apply(data_func, **kwargs)
        return out_dataframe

    return process_dataframe


def read_df(collect_name, exp_name, region, db_dir=''):
    """
    Read the data frame of a single experiment and brain region
    Args:
        collect_name: str, the name of the data collection, for example 'dfovf'
        exp_name: str, the name of the experiment
        region: str, region name
        db_dir: str, the root directory containing all data collections
    Returns:
        df: pandas.DataFrame, the data frame containing required data
    """
    print(exp_name, region)
    filename = get_filename(exp_name, region, 'pkl')
    df_file = os.path.join(db_dir, collect_name, filename)
    df = pd.read_pickle(df_file)
    return df

def update_df(df, collect_name, exp_name, region, db_dir=''):
    collect_dir = os.path.join(db_dir, collect_name)
    if not os.path.exists(collect_dir):
        os.mkdir(collect_dir)
    filename = get_filename(exp_name, region, 'pkl')
    df_file = os.path.join(collect_dir, filename)
    # df.to_pickle(df_file)
    pickle.dump(df, open(df_file, 'wb'))


def get_filename(exp_name, region, ext):
    return '{}_{}.{}'.format(exp_name, region, ext)


def concatenate_df_from_db(exp_list, region_list, in_collect_name, db_dir, axis=1):
    expkey_list = get_expkey_list(exp_list, region_list)
    df_list = [read_df(in_collect_name, expkey[0], expkey[1], db_dir)\
               for expkey in expkey_list]
    if in_collect_name == 'dfovf':
        df_list = [ptt.restack_as_pattern(df) for df in df_list]
    all_df = pd.concat(df_list, axis=axis, keys=expkey_list, names=['fish_id', 'region', 'cond'])
    return all_df


def get_expkey_list(exp_list, region_list):
    prod = itertools.product(exp_list, region_list)
    expkey_list = [(key[0][0], key[1], key[0][1]) for key in prod]
    return expkey_list


def read_data_dict(db_dir, collect_name, exp_list, region_list):
    data_dict = dict()
    for exp in exp_list:
        exp_name = exp[0]
        data_dict[exp_name] = dict()
        for region in region_list:
            print(exp_name, region)
            df = read_df(collect_name, exp_name, region, db_dir)
            data_dict[exp_name][region] = df
    return data_dict


def sort_exp_list(exp_list, cond_list):
    cond_list = ['phe-arg', 'arg-phe', 'phe-trp', 'naive']
    exp_list.sort(key=lambda y: cond_list.index(y[1]))
    return exp_list


# For each experiment including both OB and Dp
def read_np(collect_name, exp_name, file_name, db_dir):
    """
    Read the numpy data of a single experiment
    Args:
        collect_name: str, the name of the data collection, for example 'dfovf'
        exp_name: str, the name of the experiment
        file_name: str, the file name of the numpy data file
        db_dir: str, the root directory containing all data collections
    Returns:
        result: numpy.array, the data frame containing required data
    """
    np_file = os.path.join(db_dir, collect_name, exp_name, file_name)
    result = np.load(np_file)
    return result


def concatenate_np_from_db(exp_list, in_collect_name, file_name, db_dir, axis=1):
    arr_list = [read_np(in_collect_name, exp_name, file_name, db_dir)\
                for exp_name, cond in exp_list]

    df_list = [pd.DataFrame(arr).rename_axis('latent', axis=1) for arr in arr_list]
    all_df = pd.concat(df_list, axis=axis, keys=exp_list, names=['fish_id', 'cond'])
    all_df.index = all_df.index.rename('time_index')
    return all_df


def concatenate_df_from_dict(df_dict, exp_list, region_list, axis=1):
    expkey_list = get_expkey_list(exp_list, region_list)
    df_list = [df_dict[expkey[0]][expkey[1]] for expkey in expkey_list]
    all_df = pd.concat(df_list, axis, keys=expkey_list,
                       names=('fish_id', 'region', 'cond'))
    return all_df


def mean_mat_over_cond(mat_list, exp_cond_list, cond_list):
    avg_mats = dict()
    for cond in cond_list:
        filtered_dfs = [df for df, con in zip(mat_list, exp_cond_list) if con == cond]
        average_df = pd.DataFrame().reindex_like(filtered_dfs[0])

        for df in filtered_dfs:
            average_df = average_df.add(df, fill_value=0)

        average_df = average_df / len(filtered_dfs)
        avg_mats[cond] = average_df
    return avg_mats

def mean_mat(mat_list):
    average_df = pd.DataFrame().reindex_like(mat_list[0])

    for df in mat_list:
            average_df = average_df.add(df, fill_value=0)

    average_df = average_df / len(mat_list)
    return average_df