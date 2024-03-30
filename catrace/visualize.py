import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_pattern_heatmap(pattern, climit=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    if climit:
        im = ax.imshow(pattern.T, aspect='auto', interpolation='none',
                       vmin=climit[0], vmax=climit[1])
    else:
        im = ax.imshow(pattern.T, aspect='auto', interpolation='none')
    return im


def plot_conds_mat(dfs, cond_list, plot_func, sharex=False,
                   sharey=False, *args, **kwargs):
    """
    Plot matrices for each training condtion.
    so far only for 4 conditions
    """
    ncol = 2
    nrow = 2
    figsize=[8, 3.5*nrow]
    fig, axes = plt.subplots(nrow, ncol, sharex=sharex,
                             sharey=sharey, figsize=figsize)
    for idx, name in enumerate(cond_list):
        group = dfs[name]
        ax = axes.flatten()[idx]
        plot_func(group, *args, **kwargs, ax=ax)
        ax.set_title(name)
        img = ax.get_children()[0]
        fig.colorbar(img, ax=ax)

    plt.tight_layout()
    return fig, axes


def plot_conds(df, plot_func, sharex=False,
               sharey=False, *args, **kwargs):
    """
    Plot for each training condtion.
    so far only for 4 conditions
    """
    ncol = 2
    nrow = 2
    figsize=[8, 3.5*nrow]
    fig, axes = plt.subplots(nrow, ncol, sharex=sharex,
                             sharey=sharey, figsize=figsize)
    for idx, (name, group) in enumerate(df.groupby(level='cond', axis=1)):
        ax = axes.flatten()[idx]
        plot_func(group, *args, **kwargs, ax=ax)
        ax.set_title(name)
    plt.tight_layout()
    return fig, axes


def plot_response_by_cond(df, yname, plot_type='box', naive_comparisons=None, hline_y=1.0, stat_y_max=0.5, stat_y_offset=0.06):
    """
    Plot responses and statistical test results
    """
    current_palette = sns.color_palette()
    conds = df['cond'].unique()

    palette = dict(zip(conds, current_palette))
    #{"phe-arg": current_palette[0], "arg-phe": current_palette[1], 'phe-trp':current_palette[2], 'naive':current_palette[3]}
    fig, ax = plt.subplots(figsize=(10, 6))
    if plot_type == 'box':
        sns.boxplot(ax=ax, data=df, x='cond', y=yname, palette=palette, gap=.1, showfliers=False, fill=False, whis=(1,99))
    else:
        sns.violinplot(ax=ax, data=df, x='cond', y=yname, palette=palette, gap=.1, fill=False)

    ax.set_xlabel('Condition')

    if naive_comparisons:
        ax.axhline(hline_y, linestyle='--', color='black')
        significance_levels = {0.001: '***', 0.01: '**', 0.05: '*'}

        num_conds = df['cond'].nunique()

        xmin = 0
        xmax = num_conds - 1

        y_max = df[yname].max() * stat_y_max
        y_offset = (df[yname].max() - df[yname].min()) * stat_y_offset  # Slight offset above the violin

        for i, cond in enumerate(naive_comparisons.index):
            if cond != 'naive':
                p_value = naive_comparisons.loc[cond]
                for sig_level, marker in significance_levels.items():
                    if p_value < sig_level:
                        # Place the text annotation above the violin
                        x_pos = np.where(df['cond'].unique() == cond)[0]
                        ax.text(x_pos, y_max+y_offset, marker, ha='center', va='bottom', color='black')
                        break  # Found the significant level, no need to check further
        ax.hlines(y=y_max, xmin=xmin, xmax=xmax, color='black')
        ax.vlines(x=xmax, ymin=y_max-y_offset*0.7, ymax=y_max, color='black')

        # Adjust y-axis limit to account for the space needed by annotations
        y_lim = ax.get_ylim()
        ax.set_ylim(y_lim[0], y_lim[1] + (y_lim[1] - y_lim[0]) * 0.2)  # Increase the upper limit to avoid overlap
