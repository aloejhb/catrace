import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d


def get_plot_idx(i, n_trial, n_odor):
    idx = (i % n_trial) * n_odor + np.floor(i/n_trial) + 1
    return idx


def plot_trace(trace, n_trial, n_odor):
    for i in range(len(trace)):
        plt.subplot(n_trial, n_odor, get_plot_idx(i, n_trial, n_odor))
        plt.imshow(trace[i,:,:], aspect='auto')


def plot_trace_heatmap(trace, ax):
    im = ax.matshow(trace, aspect='auto')
    return im


def plot_tracedf_heatmap(tracedf, num_trial, odor_list, climit, figsize=(10,4), cut_window=None):
    num_odor = len(odor_list)
    fig, axes = plt.subplots(num_trial, num_odor, sharex=True,
                             sharey=True, figsize=figsize)
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
        ax.set_xticks(range(0, trace.shape[1], 50))
        ax.xaxis.tick_bottom()
        if not trial_nb:
            ax.set_title(name[0])
    plt.colorbar(im)
    return fig


def plot_trace_avg(trace, frame_rate, odor_list=None, ax=None):
    """
    Plot averaged time trace for each odor

    Args:
        trace (pandas.DataFrame): the dataframe that contains value of dF/F of
                                  all neurons at different time point.
                                  The shape is as follows:
                                      row multi-indices: odor, trial, neuron
                                      column multi-indices: time
        frame_rate (float): the frame rate (how many time points per second) of
                            the time course.
        odor_list (list of strings, optional): Default is None. If not None, the
                            odors will be sorted according to the order in the list.
        ax (axis object): Default is None. If ax is not provided, it will create
                          a new figure. Otherwise it will plot on the given axis.
    """
    if ax is None:
        new_ax_flag = 1
        fig, ax = plt.subplots()
    else:
        new_ax_flag = 0
    trace = trace.stack(level=['plane','neuron']).unstack(level='time')
    odor_avg = trace.groupby(level=['odor']).mean()
    if odor_list:
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


def plot_average_time_trace(dff):
    plt.plot(dff.groupby('time').mean().mean(axis=1))
