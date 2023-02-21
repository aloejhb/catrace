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


def plot_pattern_correlation(pattern, ax, clim=None, title='', perc=0):
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


if __name__ == '__main__':
    data_root_dir = '/home/hubo/Projects/Ca_imaging/results/'
    dp_exp = '2019-09-03-OBfastZ'
    ob_exp = '2019-09-03-OBFastZ2'

    outdir = os.path.join(data_root_dir, ob_exp, 'analysis')
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    frame_rate = 30/4
    n_trial = 3
    pre_time = 5
    post_time = 10
    trace = {}
    trace_aligned = {}
    time_window_dict = dict(ob_outer=[5, 10], ob_deep=[5, 10], dp=[4.8, 6.5])


    # get odor list
    trc_dict = dataio.load_trace_file(data_root_dir, ob_exp, 4)
    odor_list = trc_dict['odor_list']
    trc_dict = dataio.load_trace_file(data_root_dir, dp_exp, 1)
    if not (odor_list == trc_dict['odor_list']).all():
        raise Exception('OB and Dp odor lists are different')

    # cell_cut = dict(ob_outer=80, ob_deep=275, dp=348)


    ## OB data
    trace['ob_outer'] = load_trace_from_planes(data_root_dir, ob_exp, [4], [70])
    trace['ob_deep'] = load_trace_from_planes(data_root_dir, ob_exp, [2])
    ob_onset_list = [101, 102, 103, 99, 114, 102, 102, 102, 116, 102, 102, 104, 104, 102, 104, 114, 111, 111, 102, 102, 102, 102, 102, 102]

    trace_aligned['ob_outer'] = align_trace(trace['ob_outer'], ob_onset_list,
                                            pre_time, post_time, frame_rate)
    trace_aligned['ob_deep'] = align_trace(trace['ob_deep'], ob_onset_list,
                                           pre_time, post_time, frame_rate)


    ## Dp data
    plane_num_list = [2]
    # cell_list = [100, 120]
    trace['dp'] = load_trace_from_planes(data_root_dir, dp_exp, plane_num_list)

    ## Detect onset
    dp_onset_list = [94, 101, 100, 96, 100, 99, 101, 101, 100, 102, 96, 90, 93, 100, 96, 96, 97, 95, 97, 97, 86, 100, 100, 100]
    if not len(dp_onset_list):
        dp_onset_list = get_df_onset_list(dp_trace)

    trace_aligned['dp'] = align_trace(trace['dp'], dp_onset_list,
                                      pre_time, post_time, frame_rate)

    ## Plot sample average trace
    sigma = 0.1
    ala_range = range(3,6)
    tca_range = range(9,12)
    region_text_dict = dict(ob_outer='OB outer', ob_deep='OB deep', dp='Dp')
    color_dict = dict(ob_outer='#2ca02c', ob_deep='#1f77b4', dp='#ff7f0e')
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=[10, 5], sharex=True, sharey=True)
    plt.sca(ax[0])
    plot_region_avg(trace_aligned, ala_range, frame_rate, sigma, region_text_dict, color_dict)
    ax[0].set_xlabel('Time (s)')
    ax[0].set_ylabel('dF/F')
    ax[0].yaxis.set_tick_params(labelsize=18)
    ax[0].xaxis.set_tick_params(labelsize=18)
    plt.sca(ax[1])
    plot_region_avg(trace_aligned, tca_range, frame_rate, sigma, region_text_dict, color_dict)
    ax[1].set_xlabel('Time (s)')
    ax[1].xaxis.set_tick_params(labelsize=18)
    plt.legend(framealpha=0.5, fontsize=12)
    plt.tight_layout()
    fig_file = os.path.join(outdir, 'sample_avg.svg')
    plt.savefig(fig_file)
    plt.show()

    perc = 0
    # region_name = 'ob_outer'

    ## Calculate response pattern
    mytrace = {}
    pattern = {}
    for region_name in trace_aligned.keys():
        time_window = time_window_dict[region_name]
        trc = trace_aligned[region_name]
        pat = convert_trace_to_pattern(trc, time_window, frame_rate)
        if perc:
            cellidx = filter_cell_based_on_response(pattern, perc)
            trc = trc[:, cellidx, :]
            pat = convert_trace_to_pattern(trc, time_wrindow, frame_rate)
        mytrace[region_name] = trc
        pattern[region_name] = pat

    ## Plot PCA
    # fig, ax = plt.subplots(nrows=3, ncols=1, figsize=[5, 12])
    # for i, region_name in enumerate(trace_aligned.keys()):
    #     nrnpca.plot_response_pca(pattern[region_name], odor_list[:6], n_trial, ax[i], ellipsekwargs=dict(linewidth=2))
    #     ax[i].set_xlabel('PC1')
    #     ax[i].set_ylabel('PC2')
    # ax[0].legend(framealpha=0.5, fontsize=16)
    # plt.tight_layout()

    # fig_file = os.path.join(outdir, 'pca.svg')
    # plt.savefig(fig_file)
    # plt.show()

    ## Plot correlation matrix
    # fig, ax = plt.subplots(nrows=3, ncols=1, figsize=[5, 12])
    # rep_odor_list = np.repeat(odor_list, 3)
    # color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    # rep_color_list = np.repeat(color_list, 3)
    # for i, region_name in enumerate(trace_aligned.keys()):
    #     im = plot_pattern_correlation(pattern[region_name], ax[i],
    #                                   odor_list=rep_odor_list, color_list=rep_color_list,
    #                                   clim=(-0.05, 1))
    #     plt.sca(ax[i])
    #     cbar = plt.colorbar(im)
    #     cbar.ax.yaxis.set_tick_params(labelsize=18)

    # plt.tight_layout()

    # fig_file = os.path.join(outdir, 'corrmat.svg')
    # plt.savefig(fig_file)
    # plt.show()
    #

    aa_range = range(3)
    bb_range = range(3, 6)
    sigma0 = 0.5

    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=[10, 12], sharex=True, sharey='row')
    for i, region_name in enumerate(mytrace.keys()):
        corrmat_tvec = calc_correlation_tvec(gaussian_filter1d(mytrace[region_name], sigma0, axis=2))
        cc_corr_avg, cc_corr_std = calc_cross_odor_group_corr(corrmat_tvec, aa_range, bb_range, n_trial)
        xvec = np.arange(len(cc_corr_avg)) / frame_rate
        cc_color = 'gray'
        cc_label = 'a.a. vs b.a'

        aa_colors = ['blue', 'orange']
        aa_labels = ['a.a. same', 'a.a. diff']
        calc_decorrelation(corrmat_tvec, aa_range, ax[i, 0], frame_rate,
                           aa_colors, aa_labels)
        plot_avg_std(xvec, cc_corr_avg, cc_corr_std, ax[i, 0],
                     color=cc_color, line_label=cc_label)

        bb_colors = ['purple', 'green']
        bb_labels = ['b.a. same', 'b.a. diff']
        calc_decorrelation(corrmat_tvec, bb_range, ax[i, 1], frame_rate,
                           bb_colors, bb_labels)
        plot_avg_std(xvec, cc_corr_avg, cc_corr_std, ax[i, 1],
                     color=cc_color, line_label=cc_label)
    for i in range(2):
        ax[2, i].legend(framealpha=0.5, fontsize=12)
        ax[2, i].set_xlabel('Time (s)')
        ax[2, i].xaxis.set_tick_params(labelsize=18)
    for i in range(3):
        ax[i, 0].set_ylabel('Corr. coef.')
        ax[i, 0].yaxis.set_tick_params(labelsize=18)
    ax[1, 0].set_yticks([0, 0.2, 0.4, 0.6])
    ax[2, 0].set_yticks([0, 0.2, 0.4, 0.6])
    plt.tight_layout()
    fig_file = os.path.join(outdir, 'corr_tvec2.svg')
    plt.savefig(fig_file)
    plt.show()

    # # plt.plot(np.mean(trace_aligned[:9, :, :], axis=(0, 1)))
    # # plt.plot(np.mean(trace_aligned[10:18, :, :], axis=(0, 1)))



    # cc_range = range(6)

    # plt.show()
