import os
import sys
import pandas as pd
import numpy as np
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import sem
from importlib import reload
from scipy.ndimage import gaussian_filter1d

from .dataio import load_trace_file
from .process_time_trace import mean_pattern_in_time_window


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



def plot_pattern_correlation(df, ax=None, clim=None, title='', cmap='turbo', show_legend=False):
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
    im = ax.imshow(df.to_numpy(), cmap=cmap)#'RdBu_r'

    color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    odor_list = df.index.unique(level='odor')
    color_dict = dict(zip(odor_list, color_list[:len(odor_list)]))
    y_labels = [label for label in df.index.get_level_values('odor')]
    tick_pos = np.arange(df.shape[0])
    ax.yaxis.set_tick_params(length=0)
    ax.set_yticks(tick_pos)
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_xticks([])

    for i, ytick in enumerate(ax.get_yticklabels()):
        ytick.set_color(color_dict[ytick.get_text()])

    if clim:
        im.set_clim(clim)
    if title:
        ax.set_title(title)
    if show_legend:
        ax.colorbar()
    return im
