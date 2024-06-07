import matplotlib.pyplot as plt
import seaborn as sns
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


def plot_trace_avg(trace, frame_rate, odor_list=None, ax=None, show_legend=False, yname='spike_prob'):
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
    odor_avg = trace.groupby(level=['odor', 'time']).mean().T.mean()
    # if odor_list:
    #     odor_avg = odor_avg.reindex(odor_list)
    tvec = odor_avg.index.get_level_values('time')
    xvec = tvec / frame_rate
    odor_avg = odor_avg.reset_index()
    odor_avg = odor_avg.rename(columns={0: yname})
    sns.lineplot(odor_avg, x='time', y=yname, hue='odor', ax=ax)
    if not show_legend:
        ax.legend().remove()
    else:
        ax.legend(loc=2, prop={'size': 6})

def plot_average_time_trace(dff):
    plt.plot(dff.groupby('time').mean().mean(axis=1))