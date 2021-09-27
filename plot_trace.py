import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d

from .trace_dataframe import get_colname, concatenate_planes


def get_plot_idx(i, n_trial, n_odor):
    idx = (i % n_trial) * n_odor + np.floor(i/n_trial) + 1
    return idx


def plot_trace(trace, n_trial, n_odor):
    for i in range(len(trace)):
        plt.subplot(n_trial, n_odor, get_plot_idx(i, n_trial, n_odor))
        plt.imshow(trace[i,:,:], aspect='auto')


def plot_trace_heatmap(trace, ax):
    im = ax.imshow(trace, aspect='auto')
    return im


def plot_tracedf_heatmap(tracedf, num_trial, odor_list, climit, cut_window=None):
    num_odor = len(odor_list)
    fig, axes = plt.subplots(num_trial, num_odor, sharex=True,
                             sharey=True, figsize=[10, 4])
    for name, group in tracedf.groupby(level=['odor', 'trial']):
        if cut_window is not None:
            trace = group.iloc[:, cut_window[0]:cut_window[1]]
        else:
            trace = group
        odor_nb = odor_list.index(name[0])
        trial_nb = name[1]
        ax = axes[trial_nb, odor_nb]
        im = plot_trace_heatmap(trace, ax)
        im.set_clim(climit)
        if not trial_nb:
            ax.set_title(name[0])
    plt.colorbar(im)
    return fig


def plot_trace_avg(trace, odor_list, frame_rate, ax=None):
    if ax is None:
        new_ax_flag = 1
        fig, ax = plt.subplots()
    else:
        new_ax_flag = 0
    odor_avg = trace.groupby(level=['odor']).mean()
    odor_avg = odor_avg.reindex(odor_list)
    xvec = np.arange(len(odor_avg.columns)) / frame_rate
    odor_avg.columns = xvec
    odor_avg.transpose().plot(ax=ax)
    if new_ax_flag:
        plt.xlabel('Time (s)')
        plt.ylabel('dF/F')
        plt.legend()
        return fig
    else:
        ax.legend().remove()



# plot trial avg
# xx = [plt.plot(r) for ind,r in dfovf_cut.groupby(['odor','trial']).mean().iterrows()]
# xx = [plt.plot(gaussian_filter1d(r,3)) for ind,r in dfovf_cut.groupby(['odor','trial']).mean().iterrows()]
