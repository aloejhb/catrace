import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import replace
from catrace.visualize import PlotBoxplotParams
from catrace.for_paper import save_figure_for_paper, save_stats_json
from catrace.stats import sort_conditions
from catrace.visualize import plot_measure, plot_all_measures


def get_group_vs_group(dff, odor1_group, odor2_group):
    # Get  tuples of (odor1, odor2) that are in the group
    odor_tuples = [(odor1, odor2) for odor1 in odor1_group for odor2 in odor2_group]
    # Remove the tuples that has same odor1 and odor2
    odor_tuples = [odor_tuple for odor_tuple in odor_tuples if odor_tuple[0] != odor_tuple[1]]
    # The order of odor1 and odor2 does not matter, so we only need to keep one of them
    odor_tuples = list({tuple(sorted(odor_tuple)) for odor_tuple in odor_tuples})
    print(odor_tuples)

    # dff has more index levels than just odor1 and odor2
    # Extract the 'odor1' and 'odor2' index levels from dff
    idx_df = dff.index.to_frame(index=False)
    
    # Create a mask by checking if each (odor1, odor2) in the index is in odor_tuples
    mask = idx_df[['odor1', 'odor2']].apply(
        lambda x: tuple((x['odor1'], x['odor2'])) in odor_tuples, axis=1
    )

    # Use the mask to filter the original DataFrame
    gvg = dff[mask.values]
    return gvg


def plot_stat_group_vs_group(dff, dff_shuffled, measure_names, conditions, odor1_group, odor2_group, title=None, test_type='mannwhitneyu', strip_size=2, do_plot_strip=True):
    dff = get_group_vs_group(dff, odor1_group, odor2_group)
    # Sort order of conditions
    dff = sort_conditions(dff, conditions)
    fig1, axs1, test_results_list1 = plot_all_measures(dff, measure_names=measure_names, name_to_label=None, test_type=test_type, do_plot_strip=do_plot_strip)
    fig1.suptitle(title, y=1.05, fontsize=22)

    dff_shuffled = get_group_vs_group(dff_shuffled, odor1_group, odor2_group)
    dff_shuffled = sort_conditions(dff_shuffled, conditions)
    fig2, ax2, _ = plot_measure(dff, measure_name='capacity', name_to_label=None, test_type=test_type, figsize=(3.5,6), strip_size=strip_size)
    fig2, ax2, test_results_list2 = plot_measure(dff_shuffled, measure_name='capacity', name_to_label=None, test_type=test_type, ax=ax2, box_color='tab:orange', ylevel_scale=1.2, mean_marker_color='brown', strip_size=strip_size)

    return fig1, axs1, fig2, ax2


def plot_stat_group_vs_group_single_measure(dff, dff_shuffled, measure_name, conditions, odor1_group, odor2_group, title=None, test_type='mannwhitneyu', do_plot_shuffled=True, plot_measure_params=PlotBoxplotParams()):
    dff = get_group_vs_group(dff, odor1_group, odor2_group)
    dff = sort_conditions(dff, conditions)
    dff_shuffled = get_group_vs_group(dff_shuffled, odor1_group, odor2_group)
    dff_shuffled = sort_conditions(dff_shuffled, conditions)

    do_capitalize_labels = plot_measure_params.do_capitalize_labels
    if do_plot_shuffled:
        plot_measure_params = replace(plot_measure_params, do_capitalize_labels=False)
    fig, test_results_raw = plot_measure(dff, measure_name=measure_name, name_to_label=None, test_type=test_type, params=plot_measure_params)
    if do_plot_shuffled:
        plot_measure_params = replace(plot_measure_params, do_capitalize_labels=do_capitalize_labels)

    ax = fig.get_axes()[0]
    test_results = {'raw': test_results_raw}
    if do_plot_shuffled:
        updated_params = replace(plot_measure_params,
                                 box_color='darkgrey',
                                 box_colors=None,
                                 ylevel_scale=1.4,
                                 mean_marker_color='tab:brown')
        fig, test_results_shuffled = plot_measure(dff_shuffled, measure_name=measure_name, name_to_label=None, test_type=test_type, ax=ax, params=updated_params)
        test_results['shuffled'] = test_results_shuffled

    #'tag:orange'
    #'brown'
    return fig, ax, test_results


def move_pvalue_indicator(ax, line_new_y, text_new_y=None):
    for collection in ax.collections:
        if collection.get_gid() == 'pvalue_line':
            # current_y = collection.get_segments()[0][0][1]  # Get current y-position
            # new_y = current_y - 0.5  # Calculate the new y-position
            xstart, xend = collection.get_segments()[0][0][0], collection.get_segments()[0][1][0]
            # Step 5: Update the y-position of the line (create new segments)
            new_segments = [
                [(xstart, line_new_y), (xend, line_new_y)]
            ]
            collection.set_segments(new_segments)

    if text_new_y is None:
        text_new_y = line_new_y*1.02
    for text_obj in ax.texts:
        if text_obj.get_gid() == 'pvalue_text':
            text_obj.set_position((text_obj.get_position()[0], text_new_y))


def plot_adult_and_juv(measure_name,
                       adult_df_per_fish,
                       adult_df_per_fish_shuffled,
                       adult_conditions,
                       ylim=None,
                       tick_interval=None,
                       do_plot_shuffled=False,
                       do_move_pvalue_indicator=False,
                       plot_measure_params=PlotBoxplotParams()):
    fig2, ax2, juvenile_test = plot_juv(measure_name, do_plot_shuffled=do_plot_shuffled, tick_interval=tick_interval, plot_measure_params=plot_measure_params)

    odor1_group = ['Trp', 'Ala', 'His', 'Ser', 'TDCA']
    odor2_group = ['Trp', 'Ala', 'His', 'Ser', 'TDCA']
    fig1, ax1, adult_test = plot_stat_group_vs_group_single_measure(adult_df_per_fish, adult_df_per_fish_shuffled, measure_name, adult_conditions, odor1_group, odor2_group, title='Adult', test_type='mannwhitneyu', do_plot_shuffled=do_plot_shuffled, plot_measure_params=plot_measure_params)

    if ylim is not None:
        ax1.set_ylim(ylim)
        ax2.set_ylim(ylim)
        if do_move_pvalue_indicator:
            move_pvalue_indicator(ax1, ylim[1])
            move_pvalue_indicator(ax2, ylim[1])

    if tick_interval is not None:
        # set yticks interval
        ax1.yaxis.set_major_locator(plt.MultipleLocator(tick_interval))
        ax2.yaxis.set_major_locator(plt.MultipleLocator(tick_interval))

    return fig1, ax1, fig2, ax2, adult_test, juvenile_test


def plot_juv(odor1_group, odor2_group, measure_name,
             df_pooled,
             df_pooled_shuffled,
             conditions,
             do_plot_shuffled=False,
             tick_interval=None,
             plot_measure_params=PlotBoxplotParams()):
    fig, ax, test_results = plot_stat_group_vs_group_single_measure(df_pooled, df_pooled_shuffled, measure_name, conditions, odor1_group, odor2_group, title='Juvenile', test_type='mannwhitneyu', do_plot_shuffled=do_plot_shuffled, plot_measure_params=plot_measure_params)
    if tick_interval is not None:
        # set yticks interval
        ax.yaxis.set_major_locator(plt.MultipleLocator(tick_interval))

    # Map xtick labels
    xtick_labels = ax.get_xticklabels()
    mapping = {'Naive': 'Naïve', 'Trained': 'Trained', 'naive': 'Naïve', 'trained': 'Trained'}
    xtick_labels = [mapping[label.get_text()] for label in xtick_labels]
    ax.set_xticklabels(xtick_labels)
    return fig, ax, test_results


def plot_cap_and_save(vsname, odor1_group, odor2_group,
                      df_pooled,
                      df_pooled_shuffled,
                      conditions,
                      dataset_name,
                      paper_fig_dir,
                      plot_measure_params,
                      tick_intervals,
                      capacity_ylim=None,
                      capacity_ylevel_scale=1.0,
                      do_plot_shuffled_measure=False,
                      do_save=False,
                      minor_tick_intervals=None,
                      xtick_labels=None,
                      do_plot_shuffled_on_capacity=True):
    test_results_dict = {}
    measure_name = 'capacity'
    if capacity_ylim is not None:
        plot_measure_params = replace(plot_measure_params, ylim=capacity_ylim)
    if capacity_ylevel_scale != 1.0:
        plot_measure_params = replace(plot_measure_params, ylevel_scale=capacity_ylevel_scale)
    fig, ax, test_results = plot_juv(odor1_group, odor2_group,
                                     measure_name,
                                     df_pooled,
                                     df_pooled_shuffled,
                                     conditions,
                                     do_plot_shuffled=do_plot_shuffled_on_capacity,
                                     tick_interval=tick_intervals[0],
                                     plot_measure_params=plot_measure_params,
                                     )
    if minor_tick_intervals is not None:
        minor_tick_interval = minor_tick_intervals[0]
        ax.yaxis.set_minor_locator(plt.MultipleLocator(minor_tick_interval))

    plot_measure_params = replace(plot_measure_params, ylim=None)
    test_results_dict[measure_name] = test_results

    if do_save:
        if xtick_labels is not None:
            ax.set_xticklabels(xtick_labels)
        figname = f'{dataset_name}_{vsname}_{measure_name}'
        save_figure_for_paper(fig, figname, paper_fig_dir)
        # save_stats_json(test_results, figname, paper_fig_dir, tuple_key_to_str=True)

    if 'axes_alignment' in df_pooled.columns:
        measure_names = ['radius', 'dimension', 'axes_alignment', 'center_alignment', 'center_axes_alignment']
    else:
        measure_names = ['radius', 'dimension', 'axis_alignment', 'center_alignment', 'center_axis_alignment']
    for idx, measure_name in enumerate(measure_names):
        tick_interval = tick_intervals[idx+1]
        fig, ax, test_results = plot_juv(odor1_group, odor2_group,
                                         measure_name,
                                         df_pooled,
                                         df_pooled_shuffled,
                                         conditions,
                                         do_plot_shuffled=do_plot_shuffled_measure,
                                         tick_interval=tick_interval,
                                         plot_measure_params=plot_measure_params)

        test_results_dict[measure_name] = test_results
        if minor_tick_intervals is not None:
            minor_tick_interval = minor_tick_intervals[idx+1]
            if minor_tick_interval is not None:
                ax.yaxis.set_minor_locator(plt.MultipleLocator(minor_tick_interval))

        ylabel = ax.get_ylabel()
        if ylabel == 'Axes_alignment' or ylabel == 'axes_alignment':
            new_ylabel = 'Axes alignment'
            ax.set_ylabel(new_ylabel)
        elif ylabel == 'Center_alignment' or ylabel == 'center_alignment':
            new_ylabel = 'Center alignment'
            ax.set_ylabel(new_ylabel)
        elif ylabel == 'Center_axes_alignment' or ylabel == 'center_axes_alignment':
            new_ylabel = 'Center-axes alignment'
            ax.set_ylabel(new_ylabel)
        
        if do_save:
            if xtick_labels is not None:
                ax.set_xticklabels(xtick_labels)

            figname = f'{dataset_name}_{vsname}_{measure_name}'
            if do_plot_shuffled_measure:
                figname += '_with_shuffled'
            save_figure_for_paper(fig, figname, paper_fig_dir)
            # save_stats_json(test_results, figname, paper_fig_dir, tuple_key_to_str=True)
    
    return test_results_dict



def get_per_vs(vsdict, dff):
    vsdffs = {}
    for vsname, (odor1_group, odor2_group) in vsdict.items():
        dff_vs = get_group_vs_group(dff, odor1_group, odor2_group)
        vsdffs[vsname] = dff_vs
    vsdff = pd.concat(vsdffs.values(), keys=vsdffs.keys(), names=['vsname'])
    return vsdff


from os.path import join as pjoin
from catrace.stats import pool_training_conditions
def load_juvenile_capacities(jobname, analysis_dir):
    jobfile = pjoin(analysis_dir, f'df_result_{jobname}.pkl')
    df_result = pd.read_pickle(jobfile)
    df_per_fish_juv = df_result.xs(False, level='shuffle').groupby(['fish_id', 'condition', 'odor1', 'odor2'], sort=False).mean()
    df_per_fish_juv_shuffled = df_result.xs(True, level='shuffle').groupby(['fish_id', 'condition', 'odor1', 'odor2'], sort=False).mean()

    juv_conditions =  ['naive', 'arg-phe', 'phe-arg', 'phe-trp']
    cond_mapping = {'naive': 'naive', 'phe-arg': 'trained', 'arg-phe': 'trained', 'phe-trp': 'trained'}
    conditions_pooled = ['naive', 'trained']

    df_pooled = pool_training_conditions(df_per_fish_juv, cond_mapping, keep_subconditions=True)

    df_pooled_shuffled = pool_training_conditions(df_per_fish_juv_shuffled, cond_mapping, keep_subconditions=True)
    return df_per_fish_juv, df_per_fish_juv_shuffled,df_pooled, df_pooled_shuffled, juv_conditions, cond_mapping, conditions_pooled



def compute_vsdff_percent(vsdff):
    vsdff_percent = vsdff.copy()
    # for each vsname, compute the mean of the naive condition
    for vsname, group in vsdff_percent.groupby('vsname'):
        naive_mean = group.xs('naive', level='condition').mean()
        # Divide all values by the naive mean
        vsdff_percent.loc[group.index] = (group - naive_mean)/ naive_mean * 100
    return vsdff_percent