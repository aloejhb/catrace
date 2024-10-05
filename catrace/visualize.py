import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from .stats import apply_test_pair, apply_test_by_cond

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

from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_conds_mat(dfs, cond_list, plot_func, sharex=False,
                   sharey=False, ncol=2, row_height=3.5, col_width=4,
                   title_fontsize=16, colorbar_fontsize=12,
                   **kwargs):
    """
    Plot matrices for each training condtion.
    """
    nrow = np.ceil(len(cond_list) /ncol).astype(int)
    figsize=[col_width*ncol, row_height*nrow]
    fig, axes = plt.subplots(nrow, ncol, sharex=sharex,
                             sharey=sharey, figsize=figsize)
    vmin = min([df.to_numpy().min() for df in  dfs.values()])
    vmax = max([df.to_numpy().max() for df in  dfs.values()])
    for idx, name in enumerate(cond_list):
        group = dfs[name]
        ax = axes.flatten()[idx]
        plot_func(group, **kwargs, ax=ax)
        ax.set_title(name, fontsize=title_fontsize)
        img = ax.get_children()[0]
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)  # 'size' controls width, 'pad' controls spacing
        cbar = fig.colorbar(img, cax=cax)
        cbar.ax.tick_params(labelsize=colorbar_fontsize)
        # cbar = fig.colorbar(img, ax=ax)
        # cbar.ax.tick_params(labelsize=colorbar_fontsize)

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
    conditions = df['condition'].unique()

    palette = dict(zip(conditions, current_palette))
    #{"phe-arg": current_palette[0], "arg-phe": current_palette[1], 'phe-trp':current_palette[2], 'naive':current_palette[3]}
    fig, ax = plt.subplots(figsize=(10, 6))
    if plot_type == 'box':
        sns.boxplot(ax=ax, data=df, x='condition', y=yname, palette=palette, gap=.1, showfliers=False, fill=False, whis=(1,99))
    elif plot_type == 'violin':
        sns.violinplot(ax=ax, data=df, x='condition', y=yname, palette=palette, gap=.1, fill=False)
    elif plot_type == 'strip':
        sns.stripplot(ax=ax, data=df, x='condition', y=yname, palette=palette, jitter=True, dodge=True)
    else:
        raise ValueError('plot_type must be one of "box", "violin", or "strip"')

    ax.set_xlabel('Condition')

    if hline_y is not None:
        ax.axhline(hline_y, linestyle='--', color='black')
    if naive_comparisons is not None:
        significance_levels = {0.001: '***', 0.01: '**', 0.05: '*'}

        num_conditions = df['condition'].nunique()

        xmin = 0
        xmax = num_conditions - 1

        y_max = df[yname].max() * stat_y_max
        y_offset = (df[yname].max() - df[yname].min()) * stat_y_offset  # Slight offset above the violin

        starred = False
        for i, condition in enumerate(naive_comparisons.index):
            if condition != 'naive':
                p_value = naive_comparisons.loc[condition]
                for sig_level, marker in significance_levels.items():
                    if p_value < sig_level:
                        if not starred:
                            starred = True
                        # Place the text annotation above the violin
                        x_pos = np.where(df['condition'].unique() == condition)[0]
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


from dataclasses import dataclass
from dataclasses_json import dataclass_json
from typing import Union
@dataclass_json
@dataclass
class PlotBoxplotParams:
    figsize: tuple = (4, 4)
    label_fontsize: float = 7
    y_tick_label_fontsize: float = 7
    ylevel_scale: float = 1.1
    pvalue_marker_xoffset: float = 0.034
    do_plot_strip: bool = True
    strip_size: float = 1
    box_width: float=0.45
    box_linewidth: float=1.5
    mean_marker_size: float=1
    pvalue_marker_fontsize: float=7
    box_color: str = 'tab:blue'
    box_colors: list[str] = None
    mean_marker_color: str = 'tab:red'
    hline_y: float = None
    pvalue_bar_linewidth: float = 1


def plot_boxplot_with_significance(datadf, xname, yname,
                                   ylabel,
                                   test_results,
                                   test_type='single', ref_key=None,
                                   ax=None,
                                   figsize=(5,3),
                                   ylim=None,
                                   hline_y=0,
                                   show_ns=True,
                                   pvalue_marker_xoffset=0.01,
                                   box_color='tab:blue',
                                   box_colors=None,
                                   label_fontsize=24,
                                   y_tick_label_fontsize=18,
                                   ylevel_scale=1.1,
                                   mean_marker_color='tab:red',
                                   strip_size=4,
                                   do_plot_strip=True,
                                   x_order=None,
                                   box_width=0.45,
                                   box_linewidth=4,
                                   mean_marker_size=5,
                                   pvalue_marker_fontsize=24,
                                   pvalue_bar_linewidth=1):
    """
    Plot boxplot with significance annotations
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    if do_plot_strip:
        sns.stripplot(ax=ax, x=xname, y=yname, data=datadf, color='black', jitter=True, size=strip_size, alpha=0.4, zorder=1, order=x_order)
    sns.boxplot(ax=ax, data=datadf, x=xname, y=yname, saturation=0.5,
                width=box_width, zorder=2,
                showfliers=False, showcaps=False,
                fill=False,
                color=box_color,
                colors=box_colors,
                medianprops=dict(alpha=0.95, linewidth=box_linewidth),
                boxprops=dict(alpha=0.95, linewidth=box_linewidth),
                whiskerprops=dict(linewidth=box_linewidth, alpha=0.7), order=x_order)
    if hline_y is not None:
        ax.axhline(hline_y, linestyle='--', color='0.2', alpha=0.85)

    # Calculate means
    cond_name = xname
    mean_points = datadf.groupby([cond_name], as_index=False, sort=False, observed=True)[yname].mean()
    sns.pointplot(ax=ax, x=cond_name, y=yname, data=mean_points, 
                  markers='D', linestyle='none', zorder=3, markersize=mean_marker_size, color=mean_marker_color, order=x_order)

    ax.set_xlabel('')
    # Adjusting the font size and rotation of x-axis tick labels
    ticks = ax.get_xticks()
    ax.set_xticks(ticks) 
    ax.tick_params(axis='x', labelsize=label_fontsize)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=label_fontsize)

    ax.tick_params(axis='y', labelsize=y_tick_label_fontsize)
    ax.set_ylabel(ylabel, fontsize=label_fontsize)
    if ylim:
        ax.set_ylim(ylim)

    # Get maximum y value from datadf
    max_y = datadf[yname].max()
    # current_ylim = ax.get_ylim()
    ylevel = ylevel_scale*max_y #current_ylim[1]
    plot_pvalue_marker(ax, ylevel, test_results, test_type, ref_key=ref_key,
                       show_ns=show_ns,
                       linewidth=pvalue_bar_linewidth,
                       pvalue_marker_xoffset=pvalue_marker_xoffset,
                       fontsize=pvalue_marker_fontsize)

    # Removing the top and right spines
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig, ax

def pvalue_to_marker(p_value, pvalue_marker_xoffset=0.01, fontsize=24):
    significance_levels = {0.001: '***', 0.01: '**', 0.05: '*'}
    for sig_level, marker in significance_levels.items():
        if p_value < sig_level:
            xoffset = len(marker) * pvalue_marker_xoffset*fontsize/14*1.7
            return marker, xoffset
    return 'n.s.', 4*pvalue_marker_xoffset*fontsize/14


def plot_pvalue_marker(ax, ylevel, test_results, test_type, ref_key=None, show_ns=True, fontsize=24, linewidth=1, **kwargs):
    # Getting the positions and labels
    xticks = ax.get_xticks()
    xlabels = [label.get_text() for label in ax.get_xticklabels()]
    # Mapping the labels to their positions
    xpos_dict = dict(zip(xlabels, xticks))

    if test_type == 'single':
        for key, val in test_results.items():
            marker, xoffset = pvalue_to_marker(val['p_value'], fontsize=fontsize, **kwargs)
            if marker != 'n.s.' or show_ns:
                text = ax.text(xpos_dict[key]-xoffset, ylevel, marker, fontsize=14)
                text.set_gid('pvalue_text')
    elif test_type == 'one_reference':
        if ref_key is None:
            raise ValueError('ref_key must be specified when test_type is "one_reference"')
        for key, val in test_results.items():
            if key != ref_key:
                marker, xoffset = pvalue_to_marker(val['p'], fontsize=fontsize, **kwargs)
                if marker != 'n.s.' or show_ns:
                    text = ax.text(xpos_dict[key]-xoffset, ylevel, marker, fontsize=fontsize)
                    text.set_gid('pvalue_text')
    elif test_type == 'pairwise':
        for key, val in test_results.items():
            marker, xoffset = pvalue_to_marker(val['p_value'], fontsize=fontsize, **kwargs)
            if marker != 'n.s.' or show_ns:
                xstart = xpos_dict[key[0]]
                xend = xpos_dict[key[1]]
                xmid = (xstart + xend) / 2
                text = ax.text(xmid-xoffset, ylevel, marker, fontsize=fontsize)
                text.set_gid('pvalue_text')
                line = ax.hlines(y=ylevel*0.97, xmin=xstart, xmax=xend, color='black', linewidth=linewidth)
                line.set_gid('pvalue_line')
    else:
        raise ValueError('test_type must be one of "single" or "one_reference"')

def _get_darker_color(color: str):
    return sns.set_hls_values(color, l=0.4)


@dataclass_json
@dataclass
class PlotBoxplotMultiOdorCondParams:
    figsize: tuple = (3.6, 2)
    ylim: tuple = None
    label_fontsize: float = 7
    legend_fontsize: float = 6
    show_ns: bool = False
    hline_y: float = None
    box_width: float = 0.45
    box_linewidth: float = 1.5
    strip_size: float = 1
    strip_jitter: float = 0.15
    box_hue_separation_scaler: float = 1.4
    strip_hue_separation_scaler: float = 0.8
    mean_dodge: float = 0.32
    mean_marker_size: float = 1.5
    pvalue_marker_fontsize: float = 7

def plot_boxplot_with_significance_multi_odor_cond(datadf, yname,
                                                   test_results=None,
                                                   odor_name='odor',
                                                   condition_name='condition',
                                                   ax=None,
                                                   ylabel=None,
                                                   figsize=(10, 5),
                                                   ylim=None,
                                                   label_fontsize = 24,
                                                   legend_fontsize=16,
                                                   show_ns=False,
                                                   hline_y=None,
                                                   box_width=0.45,
                                                   box_linewidth=1,
                                                   strip_size=1,
                                                   strip_jitter=0.2,
                                                   box_hue_separation_scaler=1.0,
                                                   strip_hue_separation_scaler=1.0,
                                                   mean_dodge=0.4,
                                                   mean_marker_size=2,
                                                   pvalue_marker_fontsize=7):
    #### IMPORTANT ####
    # This function requires seaborn version from Bo's fork aloejhb
    ###################
    # Reset index if needed to make condition_name and 'fish_id' regular columns for plotting
    datadf = datadf.reset_index()

    if ylabel is None:
        ylabel = yname

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)  # Adjusted size for better visibility
    else:
        fig = ax.get_figure()
    nconds = datadf[condition_name].nunique()

    # So far assuming only two conditions
    assert nconds == 2, 'Only two conditions are supported'

    hue_colors = ['tab:blue', 'tab:orange']
    strip_hue_colors = ['lightgray', 'lightgray']
    mean_hue_colors = [_get_darker_color(color) for color in hue_colors]
    # Plotting stripplot and boxplot with hue
    sns.stripplot(ax=ax, x=odor_name, y=yname, hue=condition_name, data=datadf,
                  jitter=strip_jitter, dodge=True, size=strip_size, alpha=0.8, zorder=1,
                  palette=strip_hue_colors,
                  hue_separation_scaler=strip_hue_separation_scaler)
    sns.boxplot(ax=ax, x=odor_name, y=yname, hue=condition_name, data=datadf,
                saturation=0.5, width=box_width,
                zorder=2, dodge=True,
                medianprops=dict(alpha=0.95, linewidth=box_linewidth),
                boxprops=dict(alpha=0.95, linewidth=box_linewidth),
                whiskerprops=dict(linewidth=box_linewidth, alpha=0.7), 
                showfliers=False, showcaps=False, fill=False, palette=hue_colors,
                hue_separation_scaler=box_hue_separation_scaler)

    # Add mean points
    mean_points = datadf.groupby([odor_name, condition_name], as_index=False, sort=False)[yname].mean()
    sns.pointplot(ax=ax, x=odor_name, y=yname, hue=condition_name, data=mean_points, 
                  dodge=mean_dodge, markers='D', linestyle='none', zorder=3, markersize=mean_marker_size, palette=mean_hue_colors)

    # Adjust the legend to show only one set of hue labels
    handles, labels = ax.get_legend_handles_labels()
    # legend = ax.legend(handles[nconds*2:], labels[nconds*2:], ncol=4)#, loc='lower right')
    # Put legend outside the plot on the right
    legend = ax.legend(handles=handles[nconds*2:], labels=labels[nconds*2:], ncol=1, loc='upper left', bbox_to_anchor=(1, 1))
    # Set the fontsize of the legend
    for text in legend.get_texts():
        text.set_fontsize(legend_fontsize)

    if hline_y is not None:
        ax.axhline(hline_y, linestyle='--', color='0.2', alpha=0.7)

    ax.set_ylabel(ylabel, fontsize=label_fontsize)

    ax.set_xlabel('')
    ax.tick_params(axis='x', labelsize=label_fontsize)
    ax.tick_params(axis='y', labelsize=label_fontsize)

    if ylim:
        ax.set_ylim(ylim)

    sns.despine(ax=ax)

    if test_results is not None:
        plot_pvalue_marker_multi_odor_two_cond(ax, test_results, 
                                               datadf,
                                               condition_name=condition_name,
                                               hue_separation_scaler=strip_hue_separation_scaler,
                                               show_ns=show_ns,
                                               fontsize=pvalue_marker_fontsize)

    fig.tight_layout()

    return fig, ax

def plot_pvalue_marker_multi_odor_two_cond(ax, test_results, datadf, condition_name='condition', hue_separation_scaler=1.0, show_ns=False,
                                           fontsize=7):
    current_ylim = ax.get_ylim()
    ymax = 1.02 * current_ylim[1]
    # Getting the positions and labels
    xticks = ax.get_xticks()
    xlabels = [label.get_text() for label in ax.get_xticklabels()]
    # Mapping the labels to their positions
    xpos_dict = dict(zip(xlabels, xticks))

    def _get_hue_offset(hue_idx, n_hues, hue_separation_scaler, width=0.8):
        width = width / n_hues
        full_width = width * n_hues
        offset = width * hue_idx + width/2 - full_width/2
        offset *= hue_separation_scaler
        return offset
        
    conditions = datadf[condition_name].unique().tolist()
    for odor, results in test_results.items():
        comparison = list(results.keys())[0]
        result = results[comparison]
        odor_pos = xpos_dict[odor]
        cond_idxs = [conditions.index(cond) for cond in comparison]
        hue_offsets = [_get_hue_offset(cond_idx, len(conditions), hue_separation_scaler) for cond_idx in cond_idxs]
        hue_poses = [odor_pos + hue_offset for hue_offset in hue_offsets]
        # Draw a horizontal line between the two conditions
        ax.hlines(y=ymax*0.97, xmin=hue_poses[0], xmax=hue_poses[1], color='black')
        pvalue = result['p_value']
        marker, xoffset = pvalue_to_marker(pvalue, pvalue_marker_xoffset=0.02, fontsize=fontsize)
        if marker != 'n.s.' or show_ns:
            xmid = (hue_poses[0] + hue_poses[1]) / 2
            ax.text(xmid-xoffset, ymax, marker, fontsize=fontsize)


def not_used_pvalue_marker_multi_odor_multi_cond(ax, datadf, yname, test_results, odor_name='odor', condition_name='condition', show_ns=False):
    # Handling annotations for significance
    current_ylim = ax.get_ylim()
    ymax = 1.02 * current_ylim[1]
    for odor, results in test_results.items():
        for comparison, result in results['Dunn_naive'].items():
            if comparison != 'naive':  # Ignore naive-naive comparison
                cond1, cond2 = 'naive', comparison
                odor_pos = datadf[odor_name].unique().tolist().index(odor)
                cond_pos = datadf[condition_name].unique().tolist().index(cond2)
                position = odor_pos + (cond_pos - nconds/2+0.5) * 0.205
                p_value = result
                marker, xoffset = pvalue_to_marker(p_value)
                if marker !='n.s.' or show_ns:
                    fontsize = 14
                    ax.text(position-xoffset*fontsize*0.05, ymax, marker, fontsize=fontsize)


def plot_measure_multi_odor_cond(mdff, measure_name, odor_name='odor',
                                 condition_name='condition',
                                 test_type='mannwhitneyu',
                                 ax=None, params=PlotBoxplotMultiOdorCondParams()):
    sub_mean_madff = mdff[[measure_name]]

    test_results = {}
    for odor, subdf in sub_mean_madff.groupby(odor_name):
        test_results[odor] = apply_test_pair(subdf, test_type=test_type)
    
    fig, ax = plot_boxplot_with_significance_multi_odor_cond(mdff, measure_name, test_results=test_results, odor_name=odor_name, condition_name=condition_name, **params.to_dict())
    return fig, ax, test_results

def plot_measure(mdff, measure_name,
                 name_to_label=None,
                 test_type='mannwhitneyu',
                 condition_name='condition',
                 ax=None, params=PlotBoxplotParams()):
    sub_mean_madff = mdff[[measure_name]]

    test_results = apply_test_pair(sub_mean_madff, test_type=test_type)
    xname = condition_name
    yname = measure_name
    if name_to_label is not None:
        ylabel = name_to_label[measure_name]
    else:
        ylabel = yname
    datadf = sub_mean_madff.reset_index()
    fig, ax = plot_boxplot_with_significance(datadf, xname, yname, ylabel,
                                    test_results, test_type='pairwise',
                                    ref_key='naive',
                                    ax=ax,
                                    **params.to_dict())
                                    # pvalue_marker_xoffset=0.034,

    plt.tight_layout()
    return fig, ax, test_results


def plot_all_measures(mdff, measure_names=None, name_to_label=None, test_type='mannwhitneyu', **kwargs):
    if measure_names is None:
        measure_names = mdff.columns

    fig, axs = plt.subplots(1, len(measure_names), figsize=(5*len(measure_names), 7))
    test_results_list = []
    for measure_name, ax in zip(measure_names, axs.flatten()):
        _, ax, test_results = plot_measure(mdff, measure_name, 
                                           name_to_label=name_to_label, 
                                           test_type=test_type,
                                           ax=ax, **kwargs)
        test_results_list.append(test_results)
    return fig, axs, test_results_list



def plot_boxplot_with_significance_by_cond(datadf, yname, ylabel, test_results,
                                           ax=None,
                                           figsize=(10, 5),
                                           ylim=None,
                                           label_fontsize=24,
                                           show_zero_line=False,
                                           show_ns=False,
                                           box_color='#1f77b4',
                                           tick_label_fontsize=16,
                                           ax_label_fontsize=20,
                                           star_fontsize=16):
    datadf  = datadf.reset_index()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = 'dummy'

    cond_name = 'condition'

    sns.stripplot(ax=ax, x=cond_name, y=yname, data=datadf, color='black', jitter=True, size=4, alpha=0.4, zorder=1)
    sns.boxplot(ax=ax, x=cond_name, y=yname, data=datadf, saturation=0.5,
                zorder=2, showfliers=False, showcaps=False,
                medianprops=dict(color=box_color, alpha=0.95, linewidth=3),
                boxprops=dict(edgecolor=box_color, alpha=0.95, fill=False, linewidth=3),
                whiskerprops=dict(color=box_color, linewidth=3, alpha=0.7))
    

    mean_points = datadf.groupby([cond_name], as_index=False, sort=False, observed=True)[yname].mean()
    sns.pointplot(ax=ax, x=cond_name, y=yname, data=mean_points, 
                  markers='D', linestyle='none', zorder=3, markersize=5, color='#d62728')

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[datadf[cond_name].nunique():], labels[datadf[cond_name].nunique():], ncol=4, loc='lower right')

    if show_zero_line:
        ax.axhline(0, linestyle='--', color='0.2', alpha=0.7)
    ax.set_ylabel(ylabel, fontsize=ax_label_fontsize)

    ax.set_xlabel('')
    ax.tick_params(axis='x', labelsize=tick_label_fontsize)
    ax.tick_params(axis='y', labelsize=tick_label_fontsize)

    if ylim:
        ax.set_ylim(ylim)

    ticks = ax.get_xticks()
    ax.set_xticks(ticks) 
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=tick_label_fontsize)

    xtick_labels = ax.get_xticklabels()
    xtick_labels = [label.get_text() for label in xtick_labels]
    current_ylim = ax.get_ylim()
    ymax = 1.02 * current_ylim[1]
    if test_results is not None:
        for cond, p_value in test_results['Dunn_naive'].items():
            if cond != 'naive':
                cond_pos = xtick_labels.index(cond)
                position = cond_pos
                marker, xoffset = pvalue_to_marker(p_value)
                if marker != 'n.s.' or show_ns:
                    ax.text(position - xoffset * star_fontsize * 0.05, ymax, marker, fontsize=star_fontsize)
    sns.despine(ax=ax)

    return fig, ax



def plot_measure_by_cond(measure_name, mdff, name_to_label=None,
                         figsize=(10, 5),
                         test_type='kruskal', **kwargs):
    cond_name = 'condition'
    submadf_by_cond = mdff[[measure_name]]
    # naive_mean = submadf_by_cond.xs('naive', level=cond_name).mean()
    # delta = (submadf_by_cond - naive_mean) / naive_mean * 100
    delta = submadf_by_cond
    yname = measure_name
    if name_to_label is None:
        ylabel = yname
    else:
        ylabel = name_to_label[yname]

    results = apply_test_by_cond(delta, measure_name, test_type=test_type)

    datadf = delta.reset_index()
    datadf.rename(columns={0: yname}, inplace=True)

    fig, ax = plot_boxplot_with_significance_by_cond(datadf, yname, ylabel, results, figsize=figsize, show_zero_line=False,
                                                     show_ns=False, **kwargs)
    return fig, ax, results


def plot_all_measure_by_cond(mdff, measure_names=None, name_to_label=None, test_type='kruskal'):
    if measure_names is None:
        measure_names = mdff.columns

    ncol = 3
    nrow = np.ceil(len(measure_names) / ncol).astype(int)
    fig, axs = plt.subplots(nrow, ncol, figsize=(4*ncol, 4*nrow))
    test_results_list = []
    for measure_name, ax in zip(measure_names, axs.flatten()):
        _, ax, test_results = plot_measure_by_cond(measure_name, mdff,
                                                    name_to_label=name_to_label,
                                                    test_type=test_type,
                                                    ax=ax)
        test_results_list.append(test_results)
    plt.tight_layout()
    return fig, axs, test_results_list


def move_pvalue_indicator(ax, line_new_y, text_new_y=None):
    for collection in ax.collections:
        if collection.get_gid() == 'pvalue_line':
            xstart, xend = collection.get_segments()[0][0][0], collection.get_segments()[0][1][0]
            new_segments = [
                [(xstart, line_new_y), (xend, line_new_y)]
            ]
            collection.set_segments(new_segments)

    if text_new_y is None:
        text_new_y = line_new_y*1.02
    for text_obj in ax.texts:
        if text_obj.get_gid() == 'pvalue_text':
            text_obj.set_position((text_obj.get_position()[0], text_new_y))


def set_yticks_interval(ax, tick_interval):
    ax.yaxis.set_major_locator(plt.MultipleLocator(tick_interval))