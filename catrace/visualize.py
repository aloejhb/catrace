import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

def load_colormap(name):
    current_folder = os.path.dirname(os.path.abspath(__file__))
    if name == 'clut2b':
        colormap_data =  np.load(os.path.join(current_folder, '../colormap/clut2b.npy'))
    else:
        raise ValueError('Unknown colormap name')
    colormap = LinearSegmentedColormap.from_list('clut2b', colormap_data)
    return colormap


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
                   sharey=False, ncol=2, row_height=3.5, col_width=4, *args, **kwargs):
    """
    Plot matrices for each training condtion.
    """
    nrow = np.ceil(len(cond_list) /ncol).astype(int)
    print(nrow)
    figsize=[col_width*ncol, row_height*nrow]
    fig, axes = plt.subplots(nrow, ncol, sharex=sharex,
                             sharey=sharey, figsize=figsize)
    vmin = min([df.to_numpy().min() for df in  dfs.values()])
    vmax = max([df.to_numpy().max() for df in  dfs.values()])
    for idx, name in enumerate(cond_list):
        group = dfs[name]
        ax = axes.flatten()[idx]
        plot_func(group, *args, **kwargs, ax=ax)
        ax.set_title(name)
        img = ax.get_children()[0]
        fig.colorbar(img, ax=ax)

    plt.tight_layout()
    return fig, axes


def plot_conds(dfs, cond_list, plot_func, sharex=False,
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
    for idx, name in enumerate(cond_list):
        group = dfs[name]
        ax = axes.flatten()[idx]
        plot_func(group, *args, **kwargs, ax=ax)
        ax.set_title(name)
    plt.tight_layout()
    return fig, axes


def plot_response_by_cond(df, yname, plot_type='box', naive_comparisons=None, hline_y=None, stat_y_max=0.5, stat_y_offset=0.06):
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
    elif plot_type == 'violin':
        sns.violinplot(ax=ax, data=df, x='cond', y=yname, palette=palette, gap=.1, fill=False)
    elif plot_type == 'strip':
        sns.stripplot(ax=ax, data=df, x='cond', y=yname, palette=palette, jitter=True, dodge=True)
    else:
        raise ValueError('plot_type must be one of "box", "violin", or "strip"')

    ax.set_xlabel('Condition')

    if hline_y is not None:
        ax.axhline(hline_y, linestyle='--', color='black')
    if naive_comparisons is not None:
        significance_levels = {0.001: '***', 0.01: '**', 0.05: '*'}

        num_conds = df['cond'].nunique()

        xmin = 0
        xmax = num_conds - 1

        y_max = df[yname].max() * stat_y_max
        y_offset = (df[yname].max() - df[yname].min()) * stat_y_offset  # Slight offset above the violin

        starred = False
        for i, cond in enumerate(naive_comparisons.index):
            if cond != 'naive':
                p_value = naive_comparisons.loc[cond]
                for sig_level, marker in significance_levels.items():
                    if p_value < sig_level:
                        if not starred:
                            starred = True
                        # Place the text annotation above the violin
                        x_pos = np.where(df['cond'].unique() == cond)[0]
                        ax.text(x_pos, y_max+y_offset, marker, ha='center', va='bottom', color='black')
                        break  # Found the significant level, no need to check further
        if starred:
            ax.hlines(y=y_max, xmin=xmin, xmax=xmax, color='black')
            ax.vlines(x=xmax, ymin=y_max-y_offset*0.7, ymax=y_max, color='black')

        # Adjust y-axis limit to account for the space needed by annotations
        y_lim = ax.get_ylim()
        ax.set_ylim(y_lim[0], y_lim[1] + (y_lim[1] - y_lim[0]) * 0.2)  # Increase the upper limit to avoid overlap
    return ax

def plot_avgdf(avgdf, ax=None):
    ax.plot(avgdf)


def plot_boxplot_with_significance(datadf, xname, yname,
                                   ylabel,
                                   test_results,
                                   test_type='single', ref_key=None,
                                   figsize=(5,3),
                                   ylim=None,
                                   hline_y=0,
                                   show_ns=True,
                                   pvalue_marker_xoffset=0.01,
                                   box_color='green'):
    """
    Plot boxplot with significance annotations

    Args:
        datadf: DataFrame with columns 'odor' and `yname`.
        test_results: Dictionary with keys as odor names and values as test results.
        yname: Name of the column in `datadf` to plot.
    """
    fig, ax = plt.subplots(figsize=figsize)

    sns.stripplot(ax=ax, x=xname, y=yname, data=datadf, color='black', jitter=True, size=4, alpha=0.4, zorder=1)
    sns.boxplot(ax=ax, data=datadf, x=xname, y=yname, saturation=0.5,
                width=0.45, zorder=2,
                showfliers=False, showcaps=False,
                medianprops=dict(color=box_color, alpha=0.95, linewidth=4),
                boxprops=dict(color=box_color, alpha=0.95, fill=False, linewidth=4),
                whiskerprops=dict(color=box_color, linewidth=4, alpha=0.7))
    if hline_y is not None:
        ax.axhline(hline_y, linestyle='--', color='0.2', alpha=0.85)

    # Calculate means
    means = datadf.groupby(xname, sort=False)[yname].mean()
    # Add mean to the boxplot
    for i, mean in enumerate(means):
        ax.plot([i-0.2, i+0.2], [mean, mean], color='red', linewidth=3, zorder=3)

    ax.set_xlabel('')
    label_fontsize = 24
    # Adjusting the font size and rotation of x-axis tick labels
    ax.tick_params(axis='x', labelsize=label_fontsize)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=label_fontsize)

    ax.set_ylabel(ylabel, fontsize=label_fontsize)
    if ylim:
        ax.set_ylim(ylim)

    current_ylim = ax.get_ylim()
    ylevel = 1.02*current_ylim[1]
    plot_pvalue_marker(ax, ylevel, test_results, test_type, ref_key=ref_key,
                       show_ns=show_ns, pvalue_marker_xoffset=pvalue_marker_xoffset)

    # Removing the top and right spines
    sns.despine(ax=ax)
    return fig, ax

def pvalue_to_marker(p_value, pvalue_marker_xoffset=0.01):
    significance_levels = {0.001: '***', 0.01: '**', 0.05: '*'}
    for sig_level, marker in significance_levels.items():
        if p_value < sig_level:
            xoffset = len(marker) * pvalue_marker_xoffset
            return marker, xoffset
    return 'n.s.', 4*pvalue_marker_xoffset


def plot_pvalue_marker(ax, ylevel, test_results, test_type, ref_key=None, show_ns=True, **kwargs):
    # Getting the positions and labels
    xticks = ax.get_xticks()
    xlabels = [label.get_text() for label in ax.get_xticklabels()]
    # Mapping the labels to their positions
    xpos_dict = dict(zip(xlabels, xticks))

    if test_type == 'single':
        for key, val in test_results.items():
            marker, xoffset = pvalue_to_marker(val['p-value'], **kwargs)
            if marker != 'n.s.' or show_ns:
                ax.text(xpos_dict[key]-xoffset, ylevel, marker, fontsize=14)
    elif test_type == 'one_reference':
        if ref_key is None:
            raise ValueError('ref_key must be specified when test_type is "one_reference"')
        for key, val in test_results.items():
            if key != ref_key:
                marker, xoffset = pvalue_to_marker(val['p'], **kwargs)
                if marker != 'n.s.' or show_ns:
                    ax.text(xpos_dict[key]-xoffset, ylevel, marker, fontsize=14)
    elif test_type == 'pairwise':
        for key, val in test_results.items():
            marker, xoffset = pvalue_to_marker(val['p-value'], **kwargs)
            if marker != 'n.s.' or show_ns:
                xstart = xpos_dict[key[0]]
                xend = xpos_dict[key[1]]
                xmid = (xstart + xend) / 2
                ax.text(xmid-xoffset, ylevel, marker, fontsize=14)
                ax.hlines(y=ylevel-0.02, xmin=xstart, xmax=xend, color='black')
    else:
        raise ValueError('test_type must be one of "single" or "one_reference"')


def plot_boxplot_with_significance_multi_cond(datadf, yname, test_results, box_color='green'):
    """
    Plot boxplot with significance annotations

    Args:
        datadf: DataFrame with a multi-level index ['cond', 'fish_id'] and columns 'odor' and `yname`.
        test_results: Dictionary with keys as tuples (cond1, cond2, odor) and values as test results.
        yname: Name of the column in `datadf` to plot.
    """
    # Reset index if needed to make 'cond' and 'fish_id' regular columns for plotting
    datadf = datadf.reset_index()

    fig, ax = plt.subplots(figsize=(10, 5))  # Adjusted size for better visibility

    # Plotting stripplot and boxplot with hue
    # sns.stripplot(ax=ax, x='odor', y=yname, hue='cond', data=datadf, jitter=True, dodge=True, size=2, alpha=0.2, zorder=1)
    sns.boxplot(ax=ax, x='odor', y=yname, hue='cond', data=datadf, saturation=0.5,
                width=0.45, zorder=2, dodge=True,
                showfliers=False, showcaps=False,
                medianprops=dict(linewidth=2))
                #boxprops=dict(fill=False, linewidth=2),
                # whiskerprops=dict(linewidth=2))

                #medianprops=dict(color=box_color, linewidth=2),
                #boxprops=dict(color=box_color, fill=False, linewidth=2),
                #whiskerprops=dict(color=box_color, linewidth=2))

    ax.axhline(0, linestyle='--', color='0.2', alpha=0.7)
    ax.set_ylabel(yname)

    # Handling annotations for significance
    # current_ylim = ax.get_ylim()
    # ymax = 1.02 * current_ylim[1]
    # for odor, results in test_results.items():
    #     for comparison, result in results['Dunn_naive'].items():
    #         if comparison != 'naive':  # Ignore naive-naive comparison
    #             cond1, cond2 = 'naive', comparison
    #             odor_pos = datadf['odor'].unique().tolist().index(odor)
    #             cond_pos = datadf['cond'].unique().tolist().index(cond2)
    #             position = odor_pos + (cond_pos) * 0.2
    #             p_value = result
    #             ax.text(position, ymax, pvalue_to_marker(p_value))

    return fig, ax