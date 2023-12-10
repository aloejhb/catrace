import os
import sys
import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
from scipy.stats import sem
from importlib import reload
from scipy.ndimage import gaussian_filter1d

from .dataio import load_trace_file
from .trace_dataframe import get_colname
from .process_time_trace import mean_pattern_in_time_window


def load_trace_from_planes(data_root_dir, exp_name, plane_num_list, cell_list=[]):
    trace = []
    for i, plane_num in enumerate(plane_num_list):
        trace_dict = dataio.load_trace_file(data_root_dir, exp_name, plane_num)
        trc = trace_dict['df_trace']
        if cell_list:
            trc = trc[:, :cell_list[i], :]
        trace.append(trc)
    trace = np.concatenate(trace, axis=1)
    return trace


def convert_trace_to_pattern(trace, time_window, frame_rate):
    """trace: LxNxM matrix. L is the number of trials. N is the number of neurons. M is the number of time points"""
    frame_window = np.array(np.array(time_window) * frame_rate).astype('int')
    pattern = np.nanmean(trace[:, :, frame_window[0]:frame_window[1]], axis=2)
    return pattern


def calc_pattern_correlation(pattern):
    corrmat = np.corrcoef(pattern)
    return corrmat


def calc_correlation_tvec(trace):
    corrmat_tvec = [np.corrcoef(trace[:, :, i]) for i in range(trace.shape[2])]
    corrmat_tvec = np.stack(corrmat_tvec)
    return corrmat_tvec


def get_same_odor_idxpair(odor_num, n_trial):
    idxpair = itertools.combinations(range(odor_num*n_trial, (odor_num+1)*n_trial),2)
    return idxpair

# def get_all_same_odor_idxpair(n_trial, n_odor):
    # idxpair = [get_same_odor_idxpair(n_trial, i) for i in range(n_odor)]
    # return itertools.chain.from_iterable(idxpair)

def get_paired_odor_idxpair(odor_pair, n_trial):
    idx1 = range(odor_pair[0]*n_trial, (odor_pair[0]+1)*n_trial)
    idx2 = range(odor_pair[1]*n_trial, (odor_pair[1]+1)*n_trial)
    idxpair = itertools.product(idx1, idx2)
    return idxpair


def get_avgcorr(corrmat_tvec, idxpair, sigma):
    def _get_tvec(corrmat_tvec, idxp, sigma):
        tvec = corrmat_tvec[:, idxp[0], idxp[1]]
        if sigma:
            tvec = gaussian_filter1d(tvec, sigma)
        return tvec
    corr_tvec_array = [_get_tvec(corrmat_tvec, idxp, sigma)
                       for idxp in idxpair]
    corr_tvec_array = np.stack(corr_tvec_array)
    avgcorr_tvec = np.mean(corr_tvec_array, axis=0)
    return avgcorr_tvec


def get_same_odor_avgcorr(corrmat_tvec, odor_num, n_trial, sigma=1):
    idxpair = get_same_odor_idxpair(odor_num, n_trial)
    avgcorr_tvec = get_avgcorr(corrmat_tvec, idxpair, sigma)
    return avgcorr_tvec


def get_paired_odor_avgcorr(corrmat_tvec, odor_pair, n_tiral, sigma=1):
    idxpair = get_paired_odor_idxpair(odor_pair, n_tiral)
    avgcorr_tvec = get_avgcorr(corrmat_tvec, idxpair, sigma)
    return avgcorr_tvec


def plot_avg_trace(trace, onset_list=[]):
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    [plt.plot(np.mean(x, axis=0)) for x in trace]
    if onset_list:
        [plt.vlines(x, 0, 1, color=colors[i % len(colors)]) for i, x in enumerate(onset_list)]


def plot_pattern_correlation_old(pattern, ax, clim=None, title='', perc=0):
    if perc:
        cellidx = filter_cell_based_on_response(pattern, perc)
        pattern = pattern[:, cellidx]

    cat_color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

    odor_list = pattern.odor
    # cldf = pd.DataFrame(data=dict(color=cat_color_list, odor=odor_list.unique()))
    # idxdf = pattern.index.to_frame()
    # idxdf = idxdf.reset_index(drop=True)
    # idxdf = idxdf.merge(cldf)
    # color_list = idxdf.loc[:,'color']
    color_list = []

    corrmat = calc_pattern_correlation(pattern.drop(columns=['odor', 'trial']))
    im = ax.imshow(corrmat, cmap='RdBu_r')
    tick_pos = np.arange(corrmat.shape[0])
    if len(odor_list):
        # ax.set_xticks(tick_pos)
        # ax.set_xticklabels(odor_list, rotation='vertical')
        ax.set_yticks(tick_pos)
        ax.set_yticklabels(odor_list)
        ax.yaxis.set_tick_params(length=0)
        if len(color_list):
            for xtick, color in zip(ax.get_yticklabels(), color_list):
                xtick.set_color(color)
        ax.set_xticks([])

    if clim:
        im.set_clim(clim)
    if title:
        ax.set_title(title)
    # plt.colorbar(im)
    return im


def plot_avg_std(xvec, yavg, std, ax=None, line_label=None, **kwargs):
    if ax:
        plt.sca(ax)
    if line_label:
        plt.plot(xvec, yavg, label=line_label, **kwargs)
    else:
        plt.plot(xvec, yavg, **kwargs)
    plt.fill_between(xvec, yavg-std, yavg+std, alpha=0.4, **kwargs)


def calc_decorrelation(corrmat_tvec, odor_range, ax, frame_rate=1, colors=['blue', 'orange'], labels=['same', 'diff']):
    n_trial = 3
    n_odor = 8


    sigma = 0

    samecorr = [get_same_odor_avgcorr(corrmat_tvec, od, n_trial, sigma)
                for od in odor_range]
    # samecorr = np.stack(samecorr)

    diffcorr = [get_paired_odor_avgcorr(corrmat_tvec, odp, n_trial, sigma)
                for odp in itertools.combinations(odor_range, 2)]
    # diffcorr = np.stack(diffcorr)

    sigma2 = 0
    # [plt.plot(gaussian_filter1d(x,sigma2), color='blue') for x in samecorr]
    # [plt.plot(gaussian_filter1d(x,sigma2), color='orange') for x in diffcorr]

    sigma3 = 0
    samecorr_avg = np.mean(gaussian_filter1d(samecorr, sigma3), axis=0)
    diffcorr_avg = np.mean(gaussian_filter1d(diffcorr, sigma3), axis=0)
    samecorr_std = np.std(gaussian_filter1d(samecorr, sigma3), axis=0)
    diffcorr_std = np.std(gaussian_filter1d(diffcorr, sigma3), axis=0)
    xvec = np.arange(len(samecorr_avg)) / frame_rate

    plot_avg_std(xvec, samecorr_avg, samecorr_std, ax, color=colors[0],
                 line_label=labels[0])
    plot_avg_std(xvec, diffcorr_avg, diffcorr_std, ax, color=colors[1],
                 line_label=labels[1])


def calc_cross_odor_group_corr(corrmat_tvec, aa_range, bb_range, n_trial):
    corr = [get_paired_odor_avgcorr(corrmat_tvec, odp, n_trial)
            for odp in itertools.product(aa_range, bb_range)]
    corr_avg = np.mean(corr, axis=0)
    corr_std = np.std(corr, axis=0)
    return corr_avg, corr_std


def plot_df_histogram(trace, time_window, frame_rate, **kwargs):
    pattern = convert_trace_to_pattern(trace, time_window, frame_rate)
    plt.hist(pattern.flatten(), **kwargs)


def analyze_ob_data(data_root_dir, ob_exp, plane_num):
    trace_dict = dataio.load_trace_file(data_root_dir, ob_exp, plane_num)
    trace = trace_dict['df_trace']

    ## Detect onset
    thresh = 0.09
    xwindow = [85, 125]
    trc = np.mean(trace[16,:,:], axis=0)
    xx = detect_onset(trc, thresh, xwindow)
    # plt.plot(trc)
    # plt.plot(np.gradient(trc))
    # plt.vlines(xx, 0, 1)
    # plt.show()

    if plane_num == 3:
        onset_list = detect_trace_onset(trace, thresh, xwindow)
        thresh2 = 0.05
        onset_list[8] = detect_onset(np.mean(trace[8,:,:], axis=0), thresh2, xwindow)
        onset_list[16] = detect_onset(np.mean(trace[16,:,:], axis=0), thresh2, xwindow)
        onset_list[18:24] = [102] * 6
    else:
        onset_list = [101, 102, 103, 99, 114, 102, 102, 102, 116, 102, 102, 104, 104, 102, 104, 114, 111, 111, 102, 102, 102, 102, 102, 102]
    plot_avg_trace(trace, onset_list)

    trace_aligned = align_trace(trace, onset_list, pre_time, post_time, frame_rate)
    # trace_aligned = gaussian_filter1d(trace_aligned, 0.5, axis=2)

    # time_window = np.array([5, 10])
    # plot_df_histogram(trace_aligned, time_window, frame_rate, bins=20)
    # plt.figure()
    # plot_df_histogram(trace_aligned[:,:80,:], time_window, frame_rate, bins=20)

    aa_range = range(3)
    bb_range = range(3,6)
    cc_range = range(6)

    calc_decorrelation(trace_aligned[:,:70], aa_range)
    plt.figure()
    calc_decorrelation(trace_aligned[:,:70], bb_range)

    ## Pattern correlation
    time_window = np.array([5, 10])
    corrmat = calc_pattern_correlation(trace_aligned, time_window, frame_rate)
    # plt.imshow(corrmat)
    # plt.show()
    #    for plane_num in range(2, 5):


    ###### Donnot delete this part before making it a usable function thank you!
    # ## Pattern decorrelation
    # spike_frame_rate = 100
    # spike_array_list = []
    # roi_number = 50


    # bin_factor = 1
    # for plane_num in range(4, 5):
    #     # for plane_num in range(2, 3):
    #     spk = dataio.load_spike_file(data_root_dir, ob_exp, plane_num)
    #     spk = np.swapaxes(spk, 1, 2)
    #     spk = spk[:, :roi_number, :]
    #     maxidx = int(np.floor(spk.shape[2]/bin_factor)*bin_factor)
    #     spk = spk[:,:,:maxidx].reshape(spk.shape[0], spk.shape[1], int(maxidx/bin_factor), bin_factor).mean(axis=3)
    #     spike_array_list.append(spk)

    # spike_array = np.concatenate(spike_array_list, axis=1)

    # cut_time = 8
    # spike_onset_list = ((np.array(onset_list) / frame_rate - cut_time)* spike_frame_rate/bin_factor).astype('int')
    # spike_pre_nframe = int(5 * spike_frame_rate/bin_factor)
    # spike_post_nframe = int(10 * spike_frame_rate/bin_factor)
    # spike_array_aligned = [spike_array[i, :, onset-spike_pre_nframe:onset+spike_post_nframe]
    #                        for i,onset in enumerate(spike_onset_list)]
    # spike_array_aligned = np.stack(spike_array_aligned)


    # calc_decorrelation(spike_array_aligned[:,:60,:], aa_range)
    # plt.show()

    # # To get accurate spike frame rate
    # trace_dict = dataio.load_trace_file(data_root_dir, ob_exp, plane_num)
    # tracex, fsx = preprocess(trace_dict['df_trace'][0], frame_rate)
    # print(fsx)

def get_dp_onset_list():
    onset_param = dict(thresh=0.1, sigma=1, xwindow=[85, 125])
    onset_list = detect_trace_onset(trace, **onset_param)
    # trc = np.mean(trace[15,:,:], axis=0)
    # xx = detect_onset(trc, thresh, xwindow, sigma=sigma, plotfig=True)
    onset_list[15] = detect_onset(np.mean(trace[15, :, :], axis=0), 0.06, xwindow, sigma=sigma)
    onset_list[21:24] = [100] * 3
    print(onset_list)
    # plot_avg_trace(trace, onset_list)


def filter_cell_based_on_response(pattern, perc):
    max_response = np.max(abs(pattern), axis=0)
    thresh = np.percentile(max_response, perc)
    cellidx =  max_response>= thresh
    return cellidx


def plot_region_avg(trace, odor_idx, frame_rate, sigma, region_text_dict={}, color_dict={}):
    for i, region_name in enumerate(trace.keys()):
        trc = [trace[region_name][oid, :, :] for oid in odor_idx]
        trc = np.concatenate(trc)
        if sigma:
            trc = gaussian_filter1d(trc, sigma, axis=1)
        avg_trace = np.mean(trc, axis=0)
        sem_trace = sem(trc, axis=0)
        xvec = np.arange(len(avg_trace)) / frame_rate

        if len(region_text_dict):
            region_text = region_text_dict[region_name]
        else:
            region_text = region_name
        if len(color_dict):
            color = color_dict[region_name]
            kwargs = dict(color=color)
        plot_avg_std(xvec, avg_trace, sem_trace, line_label=region_text, **kwargs)


def plot_decorrelation(trace, ax, sigma0=0, aa_range = range(3), bb_range=range(3,6), num_trial=3, frame_rate=7.5, perc=0, time_window=[5, 10]):
    if perc:
        pattern = convert_trace_to_pattern(trace, time_window, frame_rate)
        cellidx = filter_cell_based_on_response(pattern, perc)
        print(trace.shape)
        trace = trace[:, cellidx, :]
        print(trace.shape)

    fig, ax = plt.subplots(1, 2, figsize=[10, 5], sharex=True, sharey=True)
    corrmat_tvec = calc_correlation_tvec(gaussian_filter1d(trace, sigma0, axis=2))
    cc_corr_avg, cc_corr_std = calc_cross_odor_group_corr(corrmat_tvec, aa_range, bb_range, num_trial)
    xvec = np.arange(len(cc_corr_avg)) / frame_rate
    cc_color = 'gray'
    cc_label = 'a.a. vs b.a'

    aa_colors = ['blue', 'orange']
    aa_labels = ['a.a. same', 'a.a. diff']
    calc_decorrelation(corrmat_tvec, aa_range, ax[0], frame_rate,
                       aa_colors, aa_labels)
    plot_avg_std(xvec, cc_corr_avg, cc_corr_std, ax[0],
                 color=cc_color, line_label=cc_label)

    bb_colors = ['purple', 'green']
    bb_labels = ['b.a. same', 'b.a. diff']
    calc_decorrelation(corrmat_tvec, bb_range, ax[1], frame_rate,
                       bb_colors, bb_labels)
    plot_avg_std(xvec, cc_corr_avg, cc_corr_std, ax[1],
                 color=cc_color, line_label=cc_label)

    for i in range(2):
        ax[i].set_xlabel('Time (s)')
        ax[i].xaxis.set_tick_params(labelsize=18)
    ax[0].set_ylabel('Corr. coef.')


def compute_pattern_correlation(dfovf, time_window, frame_rate):
    """Compute pattern correlation of from time traces of neurons"""
    pattern = mean_pattern_in_time_window(dfovf, time_window, frame_rate)
    corrmat = np.corrcoef(pattern.to_numpy())
    corrmat = pd.DataFrame(corrmat, index=pattern.index, columns=pattern.index)
    return corrmat


def plot_pattern_correlation(df, ax=None, clim=None, title=''):
    """
    Plot patthern correlation matrix heatmap

    Args:
        **df**: pandas.DataFrame. Square matrix of pattern correlation.
        Row index levels: odor, trial. Column index levels: odor, trial.
        **ax**: plot Axis object. Axis to plot the matrix heatmap.
        **clim**: List. Color limit of the heatmap. Default ``None``.
        **title**: str. Title of the plot. Default ``''``.

    Returns:
        Image object.
    """
    im = ax.imshow(df.to_numpy(), cmap='RdBu_r')

    color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    odor_list = df.index.unique(level='odor')
    color_dict = dict(zip(odor_list, color_list[:len(odor_list)]))
    y_labels = [label[0] for label in df.index]
    tick_pos = np.arange(df.shape[0])
    ax.yaxis.set_tick_params(length=0)
    ax.set_yticks(tick_pos)
    ax.set_yticklabels(y_labels)
    ax.set_xticks([])

    for i, ytick in enumerate(ax.get_yticklabels()):
        ytick.set_color(color_dict[ytick.get_text()])

    if clim:
        im.set_clim(clim)
    if title:
        ax.set_title(title)
    return im
