import matplotlib.pyplot as plt
import numpy as np
from .trace_dataframe import get_colname


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


def concatenate_planes(tracedf, plane_nb_list):
    trace_list = [None] * len(tracedf)
    for i in range(len(tracedf)):
        trace = [tracedf[get_colname('dfovf', k)][i] for k in plane_nb_list]
        trace = np.concatenate(trace)
        trace_list[i] = trace
    return trace_list


def plot_tracedf_heatmap(tracedf, num_trial, num_odor, plane_nb_list, cut_window, climit):
    fig, axes = plt.subplots(num_trial, num_odor)
    for i in range(len(tracedf)):
        trace = [tracedf[get_colname('dfovf', k)][i] for k in plane_nb_list]
        trace = np.concatenate(trace)[:, cut_window[0]:cut_window[1]]
        odor_nb = tracedf['odor_code'][i]
        trial_nb = i % num_trial
        ax = axes[trial_nb, odor_nb]
        im = plot_trace_heatmap(trace, ax)
        im.set_clim(climit)
        if not trial_nb:
            ax.set_title(tracedf['odor'][i])


def plot_trace_avg(tracedf, plane_nb_list):
    trace_list = concatenate_planes(tracedf, plane_nb_list)
    trace_avg = [np.mean(x, axis=0) for x in trace_list]
    newdf = tracedf[['odor']]
    newdf['trace_avg'] = trace_avg
    grouped = newdf.groupby(['odor'])
    for name, group in grouped:
        plt.plot(np.stack(group['trace_avg']).mean(axis=0))


