import os
import sys
import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
from scipy.stats import sem
from importlib import reload
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import pdist, squareform

from .dataio import load_trace_file
from .process_time_trace import mean_pattern_in_time_window

def cosine_distance(mat):
    # Compute the pairwise cosine distances between trials
    distances = pdist(mat, metric='cosine')
    # Convert the condensed distance matrix to a square matrix
    sim_mat = squareform(distances)
    return sim_mat


def pattern_correlation(mat):
    # Compute the pairwise correlation between trials
    corr_mat = np.corrcoef(mat)
    return corr_mat


def compute_similarity_mat(dfovf, time_window, frame_rate, similarity_func):
    """
    Compute of similarity matrix from response patterns of neurons
        Args:
            dfovf
            time_window
            frame_rate
            similarity_func: np.corrcoef or scipy.spatial.distance.cosine
    """
    pattern = mean_pattern_in_time_window(dfovf, time_window, frame_rate)
    pattern_mat = pattern.to_numpy()
    sim_mat = similarity_func(pattern_mat)
    sim_mat = pd.DataFrame(sim_mat, index=pattern.index, columns=pattern.index)
    return sim_mat

def compute_similarity_mat_timecourse(dfovf, bin_size, frame_rate, similarity_func):
    import pdb; pdb.set_trace()
    # TODO
    # mats = []

    # for i in range(0, dfovf.shape[1], bin_size):
    #     mat = compute_similarity_mat(dfovf[:, i:i+bin_size], time_window, frame_rate, similarity_func)
    #     mats.append(mat)


def plot_similarity_mat(df, ax=None, clim=None, cmap='RdBu_r', ylabel_fontsize=8, title=''):
    """
    Plot similarity matrix heatmap

    Args:
        **df**: pandas.DataFrame. Square matrix of pattern correlation.
        Row index levels: odor, trial. Column index levels: odor, trial.
        **ax**: plot Axis object. Axis to plot the matrix heatmap.
        **clim**: List. Color limit of the heatmap. Default ``None``.
        **title**: str. Title of the plot. Default ``''``.

    Returns:
        Image object.
    """
    im = ax.imshow(df.to_numpy(), cmap=cmap)

    color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    odor_list = df.index.unique(level='odor')
    color_dict = dict(zip(odor_list, color_list[:len(odor_list)]))
    y_labels = [label for label in df.index.get_level_values('odor')]
    tick_pos = np.arange(df.shape[0])
    ax.yaxis.set_tick_params(length=0)
    ax.set_yticks(tick_pos)
    ax.set_yticklabels(y_labels, fontsize=ylabel_fontsize)
    ax.set_xticks([])

    for i, ytick in enumerate(ax.get_yticklabels()):
        ytick.set_color(color_dict[ytick.get_text()])

    if clim:
        im.set_clim(clim)
    if title:
        ax.set_title(title)
    return im


def select_odors_mat(matdf, odors):
    if matdf.index.names == ['odor', 'trial']:
        smat = matdf.loc[(odors, slice(None)), (odors, slice(None))]
    else:
        smat = matdf.loc[odors, odors]
    return smat


def compute_aavsba(simdf, aa_odors, ba_odors):
    if simdf.index.names == ['odor', 'trial']:
        aavsba = simdf.loc[(aa_odors, slice(None)), (ba_odors, slice(None))].mean().mean()
    else:
        aavsba = simdf.loc[aa_odors, ba_odors].mean().mean()

    return aavsba
