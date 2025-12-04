import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.patches as patches

from scipy.ndimage.filters import gaussian_filter1d


def get_plot_idx(i, n_trial, n_odor):
    idx = (i % n_trial) * n_odor + np.floor(i / n_trial) + 1
    return idx


def plot_trace(trace, n_trial, n_odor):
    for i in range(len(trace)):
        plt.subplot(n_trial, n_odor, get_plot_idx(i, n_trial, n_odor))
        plt.imshow(trace[i, :, :], aspect='auto')


def plot_trace_heatmap(trace, ax):
    im = ax.matshow(trace, aspect='auto', interpolation='none')
    return im


def plot_tracedf_heatmap(
    tracedf, num_trial, odor_list, climit, figsize=(10, 4), cut_window=None
):
    num_odor = len(odor_list)
    fig, axes = plt.subplots(
        num_trial, num_odor, sharex=True, sharey=True, figsize=figsize
    )
    for name, group in tracedf.groupby(level=['odor', 'trial']):
        if cut_window is not None:
            trace = group.iloc[:, cut_window[0] : cut_window[1]]
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


def plot_trace_avg(
    trace,
    frame_rate=None,
    cut_time=0,
    ax=None,
    show_legend=False,
    yname='spike_prob',
    linewidth=2,
    figsize=(10, 4),
    legend_fontsize=6,
    label_fontsize=7,
    tick_label_fontsize=6,
    odor_colors=None,
    alpha=1,
):
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
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    odor_avg = (
        trace.groupby(level=['odor', 'time'], sort=False, observed=False)
        .mean()
        .T.mean()
    )
    xvec = odor_avg.index.get_level_values('time')
    if frame_rate is not None:
        xvec = xvec / frame_rate
    xvec = xvec - cut_time
    odor_avg = odor_avg.reset_index()
    odor_avg = odor_avg.rename(columns={0: yname})
    odor_avg['time'] = xvec
    if odor_colors is None:
        palette = sns.color_palette(
            'husl', n_colors=len(odor_avg['odor'].unique())
        )
    else:
        palette = dict(zip(odor_avg['odor'].unique(), odor_colors))
    sns.lineplot(
        odor_avg,
        x='time',
        y=yname,
        hue='odor',
        ax=ax,
        linewidth=linewidth,
        palette=palette,
        alpha=alpha,
    )
    if not show_legend:
        ax.legend().remove()
    else:
        # Put legend outside the plot
        ax.legend(
            loc='center left',
            bbox_to_anchor=(1, 0.5),
            prop={'size': legend_fontsize},
        )

    ax.tick_params(axis='both', labelsize=tick_label_fontsize)
    xlabel = 'Time (s)' if frame_rate is not None else 'Frame'
    ax.set_xlabel(xlabel, fontsize=label_fontsize)
    ax.set_ylabel(r'Deconvolved $\Delta F/F$', fontsize=label_fontsize)
    sns.despine(ax=ax)

    fig.tight_layout()

    return fig, ax


def plot_average_time_trace(dff):
    plt.plot(dff.groupby('time').mean().mean(axis=1))


def plot_mean_with_std(
    time_traces_df,
    frame_rate=1,
    time_level_name='time',
    ax=None,
    color='blue',
    label='Mean',
    err_type='std',
    linewidth=1,
    start_time_with_zero=False,
    std_alpha=0.3,
):
    """
    Plots the mean trace with a shaded area representing the standard deviation.

    Parameters:
    - time_traces_df: pd.DataFrame
        A DataFrame where the rows correspond to time points and the columns are different trials or observations.
    - ax: matplotlib.axes.Axes, optional
        The axes on which to plot. If not provided, a new figure and axes will be created.
    - color: str, optional
        The color of the mean trace and the shaded area representing the standard deviation.
    - label: str, optional
        The label for the mean trace.
    """
    # Calculate mean and standard deviation across time
    mean_trace = time_traces_df.mean(axis=1)
    if err_type == 'std':
        err_trace = time_traces_df.std(axis=1)
    elif err_type == 'sem':
        err_trace = time_traces_df.sem(axis=1)
    else:
        raise ValueError(f'err_type must be either "std" or "sem". Got: {err}')

    # Create the plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.get_figure()

    tvec = mean_trace.index.get_level_values(time_level_name)
    xvec = tvec / frame_rate
    if start_time_with_zero:
        xvec = xvec - xvec[0]

    # Convert xvec to float
    xvec = np.array(xvec).astype(float)
    # Plot mean trace
    sns.lineplot(
        x=xvec,
        y=mean_trace,
        label=label,
        ax=ax,
        color=color,
        linewidth=linewidth,
    )

    # Plot standard deviation as shaded area, line width = 0 to remove line
    ax.fill_between(
        xvec,
        mean_trace - err_trace,
        mean_trace + err_trace,
        color=color,
        alpha=std_alpha,
        linewidth=0,
    )
    # , label=f'{label} ± std')

    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.legend()

    return fig, ax


def add_time_range_rectangle(ax, time_range, color='gray', alpha=0.6, text=None, fontsize=6, text_offset=0.02):
    # Retrieve the current y-limits of the axes
    ylim = ax.get_ylim()
    
    # Create a rectangle that spans from ylim[0] to ylim[1]
    rect = patches.Rectangle(
        (time_range[0], ylim[0]),
        width=time_range[1] - time_range[0],
        height=ylim[1] - ylim[0],
        color=color,
        alpha=alpha,
        linewidth=0
    )
    ax.add_patch(rect)
    
    if text is not None:
        # Place the text at the center of the rectangle
        ax.text(
            x=time_range[0] + (time_range[1] - time_range[0]) / 2,
            y=ylim[1] + text_offset,
            s=text,
            ha='center',
            va='center',
            color=color,
            fontsize=fontsize
        )


def plot_example_trace_heatmap(edff, frame_rate, figsize=(7, 5), vmin=0, vmax=1, cmap='viridis', gamma=None, ax=None):
    if cmap == 'clut2b':
        cmap = load_colormap('clut2b', remove_top_white=True)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        plot_colorbar = True
    else:
        fig = ax.get_figure()
        plot_colorbar = False
    # Power normalize the heatmap
    from matplotlib.colors import PowerNorm
    if gamma is not None: # For example gamma=0.5
        norm = PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax)
        img = ax.matshow(edff, aspect='auto', cmap=cmap, norm=norm)
    else:
        img = ax.matshow(edff, aspect='auto', cmap=cmap)

    # Convert x axis from frame to time
    xticks = np.arange(0, edff.shape[1], 2.5 * frame_rate)
    # append 2*frame_rate to xticks
    xticks = np.append(xticks, 2 * frame_rate)
    ax.set_xticks(xticks)
    xticklabels = xticks / frame_rate
    ax.set_xticklabels(xticklabels)

    # Put the x tick label at the bottom
    ax.xaxis.set_ticks_position('bottom')
    # remove x tick label at the top
    ax.tick_params(axis='x', which='both', bottom=True, top=False)

    # Remove y ticks
    ax.set_yticks([])
    ax.set_yticklabels([])

    if plot_colorbar:
        colorbar = fig.colorbar(img)
        return fig, ax, img, colorbar
    return fig, ax, img